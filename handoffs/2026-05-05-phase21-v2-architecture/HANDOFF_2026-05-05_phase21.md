# Phase 21 — V2 Architecture: HYPOTHESIS REJECTED

> 2026-05-05. The Phase 21 V2 architecture (Dict obs + split-head + regime
> indicator + hard mask) was implemented end-to-end and ran a 300k seed=42
> sanity train. It UNDERPERFORMS the Phase 16a V1 baseline by 1.15
> vs_p50_adj. The regime path is functionally dead — actor produces
> identical OOS scores under real / zero / batch-mean / shuffled regime
> input. **Recommend: do NOT merge V2 to main; keep on `feat/phase21-v2-architecture`
> as a forensic artifact. Phase 16a remains production. Phase 18 ens_rankmean6
> remains the strongest ensemble candidate.**

## TL;DR

* Architecture spec: `docs/superpowers/specs/2026-05-05-phase21-v2-architecture-design.md`.
* Implementation plan: `docs/superpowers/plans/2026-05-05-phase21-v2-architecture.md`.
* Phase 21A 300k seed=42 sanity train: **best `vs_random_p50_adjusted = -0.723`** (step 149952).
  vs Phase 16a baseline +0.428: **Δ = -1.15**. Spec §10's "+0.30 minimum / -0.10 below baseline triggers investigation" gate fails by a wide margin.
* Sanity check 1 (regime ablation): real / zero / batch-mean / shuffled all produce EXACTLY +0.433 adj_sharpe. **Regime path is dead.**
* Sanity check 2 (leakage delta): real - zero = +0.000. Confirms regime ablation finding.
* Sanity check 3 (b1 vs b2): DEFERRED — moot now that the regime path is unused (the b2 vs b1 distinction only matters if the regime input affects the value head).

## What we learned

* **The split-head architecture's regime path collapsed to zero gradient signal during training.** The actor learned to ignore the (B, R') regime broadcast entirely. Possible causes (any combination):
  1. Hard mask `-100.0` dominates the actor's loss, shrinking the regime path's gradient relative to the per-stock signal.
  2. Regime encoder out_dim (16) is half the stock encoder out_dim (32), so the actor head's regime weights are statistically less likely to learn anything before the per-stock weights converge.
  3. Cross-section variation in stock features already implicitly captures regime info — the explicit (B, R') input is redundant.
  4. `target_kl=0.30` early-stops PPO at ~9 SGD updates per iteration (clip_fraction ≈ 0.30, approx_kl ≈ 0.03 per iter); 9 is enough for the dominant per-stock pathway but possibly not for the regime co-pathway to find signal.
* **V2 underperforms V1 by 1.15 vs_p50_adj at the same training budget.** The Dict-obs / mask / split-head overhead reduces the effective optimization rate without delivering the regime benefit it was supposed to enable.

## Architectural decisions (as implemented)

The full design lives in the spec; key load-bearing points:

* **Dict observation space**: `{stock: (S, F_stock), regime: (R,), valid_mask: (S,)}`. Stock encoder physically cannot see regime/mkt features (allowlist + runtime assert).
* **Split-head policy**: per-stock encoder + regime encoder + concat at the head. Actor (`Linear → mask -100 → Normal(loc, exp(log_std))`) and critic (per-stock value MLP → masked_mean → Linear) share `head_in`. Critic uses true b2 ordering.
* **Hard mask**: `valid_mask` is enforced at logit time AND at masked_mean. `-100.0` (not the spec's `-1e9` — see fix below).
* **Action space stays Box(0, 1, (S,))**: env applies top-K post-sample. Distribution is per-stock Normal; log_prob summed over S.
* **8 v0 regime features**: breadth_d/20d, xs_disp_d/20d, idx_ret_20d/60d (compounded via `expm1(cumsum(log1p))`), idx_vol_20d, extreme_imbalance_norm. Computed on the same valid_mask the env uses.
* **is_suspended default-True**: pre-IPO and delisted (t, j) cells default to suspended. Phase 19 bug fix.

Two production-grade bugs surfaced during Phase 21A and were fixed (kept under V2):

1. **Mask magnitude bug** (commit 89b13bb). The spec's `-1e9` mask put loc at magnitude 1e9 where float32 precision is ~1e2; the buffer-roundtripped action drifted by ~100 units, so per-stock log_prob `((action-loc)/scale)^2 ≈ 4e4` and joint over 3000 stocks pushed approx_kl to exp(40+). Fix: use `-100.0` — top-K still rejects invalid (typical valid scores are in [-3, +3] after LayerNorm) but stays in float32-stable range. Pinned by regression test `test_log_prob_bounded_under_mask`.

2. **last_obs_t timing bug** (commit 50cb3a8). `env.last_obs_t` was being updated at the END of `step_wait` (post-advance), but SB3 calls `rollout_buffer.add(self._last_obs)` AFTER `env.step` and BEFORE updating `self._last_obs` to the new obs — so the buffer was storing the t for the WRONG obs. PPO then re-evaluated against a different observation than was sampled, blowing up approx_kl despite the mask fix. **V1 had this same bug** but its DiagGaussian + no-hard-mask was tolerant; V2's hard mask makes it explode. Fix: snapshot `last_obs_t = self.t` at the START of `step_wait`, before any advance.

The latter is a **silent correctness improvement applicable to V1 too**, even if V2 is rolled back. Phase 22 should consider porting the fix to the V1 path on a separate branch and re-running Phase 16a to see whether the corrected timing changes the baseline.

One spec deviation: `RegimeEncoder` no longer has an input `LayerNorm(regime_dim)` (Task 3.2 finding). Input LayerNorm was shift-invariant — `LayerNorm(x+c) == LayerNorm(x)` — which killed the b2 critic test. The output LayerNorm is preserved.

## Architectural decisions

The full design lives in the spec; key load-bearing points:

* **Dict observation space**: `{stock: (S, F_stock), regime: (R,), valid_mask: (S,)}`. Stock encoder physically cannot see regime/mkt features (allowlist + runtime assert).
* **Split-head policy**: per-stock encoder + regime encoder + concat at the head. Actor (`Linear → mask -1e9 → Normal(loc, exp(log_std))`) and critic (per-stock value MLP → masked_mean → Linear) share `head_in`. Critic uses true b2 ordering — value MLP runs BEFORE pooling.
* **Hard mask**: `valid_mask` is enforced at logit time AND at masked_mean. Same mask used by `forward` and `evaluate_actions` to keep PPO ratio consistent.
* **Action space stays Box(0, 1, (S,))**: env applies top-K post-sample. Distribution is per-stock Normal; log_prob summed over S.
* **8 v0 regime features**: breadth_d/20d, xs_disp_d/20d, idx_ret_20d/60d (compounded via `expm1(cumsum(log1p))`), idx_vol_20d, extreme_imbalance_norm. Computed on the same valid_mask the env uses.
* **is_suspended default-True**: pre-IPO and delisted (t, j) cells default to suspended. Phase 19 bug fix.

One acceptable spec deviation surfaced during Task 3.2: the spec's `RegimeEncoder` had an input `LayerNorm(regime_dim)` — but LayerNorm is shift-invariant on its input, so it killed the b2 critic test (`obs_b["regime"] = obs_a["regime"] + 1.0` produced identical outputs). The input LayerNorm was removed; the output LayerNorm and Linear/SiLU/Linear path are preserved. The spec doc should be updated to match.

## Code changes

* `src/aurumq_rl/data_loader.py`:
  - `is_suspended_array` defaults to True (Phase 19 bug fix).
  - `STOCK_FACTOR_PREFIXES` allowlist (mkt_ removed) + `FORBIDDEN_PREFIXES` (mkt_, index_, regime_, global_) + `FACTOR_COL_PREFIXES` legacy alias.
  - `discover_factor_columns` defensively filters forbidden.
  - `REGIME_FEATURE_NAMES` (8-tuple).
  - `_compute_regime_features(pct_change, valid_mask) -> (T, 8)` (vectorised cumsum / log1p / expm1).
  - `FactorPanel` extended with `regime_array: np.ndarray` and `regime_names: tuple[str, ...]`.
  - Realignment helper `align_panel_to_stock_list` propagates regime per-date unchanged.
* `src/aurumq_rl/gpu_env.py`:
  - `GPUStockPickingEnv(panel, regime, returns, valid_mask, ...)` — new mandatory `regime` kwarg.
  - `observation_space = gym.spaces.Dict({stock, regime, valid_mask})`.
  - `_obs_for_sb3()` returns `dict[str, np.ndarray]` via `index_select`.
  - `step_wait` reward logic unchanged (Phase 16 fix preserved).
* `src/aurumq_rl/feature_extractor.py`:
  - V1 `PerStockExtractor` REMOVED.
  - `PerStockEncoderV2(nn.Module)` — shared MLP + LayerNorm.
  - `RegimeEncoder(nn.Module)` — Linear + SiLU + Linear + LayerNorm (input LayerNorm dropped per the deviation note above).
  - `masked_mean(x, mask, eps)` utility.
* `src/aurumq_rl/policy.py`:
  - V1 `PerStockEncoderPolicy` REMOVED.
  - `PerStockEncoderPolicyV2(ActorCriticPolicy)` — custom `_shared_forward / _logits / _value / _make_distribution / forward / evaluate_actions / get_distribution / predict_values / _predict`.
  - `_IdentityFeatures` features_extractor stand-in (subclass of `BaseFeaturesExtractor` to satisfy SB3 internals).
  - Manual `log_std` build + optimizer rebuild.
* `src/aurumq_rl/index_dict_rollout_buffer.py` (NEW):
  - `IndexOnlyDictRolloutBuffer(DictRolloutBuffer)` with t-index storage and 4 provider closures.
  - Numpy-backed reward / value / log_prob arrays so SB3's inherited `compute_returns_and_advantage` works without override.
  - `values` / `returns` accessor properties handle the numpy↔torch transition during sampling.
* `scripts/train_v2.py`:
  - V2 imports.
  - Schema lock assert against FORBIDDEN_PREFIXES.
  - Regime tensor build + env construction.
  - 3 new CLI flags: `--regime-encoder-out-dim`, `--regime-encoder-hidden`, `--critic-token-hidden`.
  - `IndexOnlyDictRolloutBuffer` wired with 4 providers.
  - Metadata: `policy_class = "PerStockEncoderPolicyV2"`, `framework = "gpu_v2_phase21"`, `obs_dict = True`, `stock_factor_names`, `regime_factor_names`, `regime_dim`, plus regime/critic encoder hyperparameters. Legacy `factor_names` alias retained.
  - `--unique-date-encoding` becomes a no-op warning.
* `scripts/_eval_all_checkpoints.py`:
  - V1 metadata rejection (no `regime_factor_names` ⇒ RuntimeError).
  - Builds `regime_t` from `panel.regime_array`; constructs Dict obs per date and feeds to `policy.forward`.
  - `PPO.load(custom_objects={"rollout_buffer_class": IndexOnlyDictRolloutBuffer, "GPURolloutBuffer": GPURolloutBuffer})` so the V2 zip's serialised buffer class resolves.
* `scripts/_phase21_sanity_checks.py` (NEW):
  - 4 scoring runs: real / zero / batch-mean / shuffled regime.
  - Reports `delta_adj_real_minus_zero` as the leakage summary.

## Tests

| File | Tests | What it pins |
|---|---:|---|
| `tests/test_data_loader_phase21.py` | 8 | is_suspended default; STOCK/FORBIDDEN_PREFIXES; `discover_factor_columns` filtering; FactorPanel regime fields; `_compute_regime_features` numerical equivalence |
| `tests/test_gpu_env_phase21.py` | 5 | Dict obs space; reset/step Dict shape; valid_mask passthrough; last_obs_t semantics |
| `tests/test_feature_extractor_phase21.py` | 8 | Encoder shape/grad; LayerNorm active; masked_mean correctness/zero-mask/grad |
| `tests/test_policy_phase21.py` | 8 | Construct; forward shape; deterministic stability; evaluate_actions consistency; -1e9 mask; empty-mask raises; regime perturbation changes value |
| `tests/test_index_dict_rollout_buffer.py` | 3 | t-index storage; add+get roundtrip; provider absence raises |

V1 test files removed: `tests/test_policy.py`. The V1 `PerStockExtractor` / `PerStockEncoderPolicy` classes are deleted; their existing tests would not have applied to V2.

## Migration / production status

**V2 is NOT being merged to main.** The Phase 21 V2 codebase lives on `feat/phase21-v2-architecture` indefinitely as a forensic artifact. Future phases that want to retry the regime indicator or split-head should branch from there or cherry-pick specific fixes (e.g. the `last_obs_t` timing fix, which is a real bug in V1 too).

Production status (UNCHANGED from before Phase 21):
- `models/production/phase16_16a_drop_mkt_best.zip` (sha256[:16] `ae924791643ee77d`) remains the production single-model baseline at +0.428 vs_p50_adj.
- `ens_rankmean6` (Phase 18) remains the strongest ensemble candidate at +0.711 vs_p50_adj — but stranded behind the Phase 19 INSUFFICIENT fresh-holdout gate.
- The 6-member Phase 18 zips remain loadable under the pre-Phase-21 V1 main branch.

## Phase 21A sanity train — RESULTS

Configuration:
- panel: `data/factor_panel_combined_short_2023_2026.parquet`
- train window: 2023-01-03 .. 2025-06-30
- OOS window: 2025-07-01 .. 2026-04-24
- universe: main_board_non_st, n=3014 stocks, 353 factors
- 300k timesteps, n_envs=16, episode=240, batch=1024, n_steps=1024, n_epochs=10
- learning_rate=1e-4, target_kl=0.30, max_grad_norm=0.5
- top_k=30, forward_period=10, seed=42
- regime_encoder_hidden=64, regime_encoder_out_dim=16, critic_token_hidden=64
- rollout_buffer=index (IndexOnlyDictRolloutBuffer)
- wall time: ~3.2 hours (RTX 4070, 26 fps SGD-bound)

Result (best checkpoint = step 149952):

| metric | Phase 16a baseline | Phase 21A | Δ |
|---|---:|---:|---:|
| best step | 224928 | 149952 | |
| adj Sharpe | +1.593 | +0.442 | -1.151 |
| **vs random p50 adj** | **+0.428** | **-0.723** | **-1.151** |
| non-overlap Sharpe | +1.112 | +0.960 | -0.152 |
| IC (display bug, see below) | +0.0143 | +0.0034 | (eval-side artefact) |

PPO training health check (good): approx_kl 0.03, clip_fraction 0.30, explained_variance 0.96, value_loss 0.034, entropy stable at -2.2e3, std 0.502. So PPO trained correctly; the issue is the ARCHITECTURE under-utilizing its parameters, not a training-dynamics bug.

Verdict: **REJECTED.** Phase 21 hypothesis ("split-head + regime indicator improves OOS") is not supported. V2 is significantly worse than V1 at the same training budget. Phase 16a remains production. Phase 18 ens_rankmean6 remains the strongest ensemble candidate.

Eval-script note: `_eval_all_checkpoints.py` reports IC values that are nearly identical (±1e-7) across all checkpoints. This is a display bug — the IC is computed from a panel-derived signal rather than the model's per-checkpoint predictions. It does NOT affect the Sharpe / vs_p50_adj numbers (which DO vary correctly per checkpoint). To investigate / fix in Phase 22 if anyone wants per-checkpoint IC.

## Three architectural sanity checks

1. **Actor regime ablation** (`phase21_sanity_checks.json`):
   - real:        +0.433 adj_S
   - zero:        +0.433 adj_S
   - batch-mean:  +0.433 adj_S
   - shuffled:    +0.433 adj_S
   - **delta(real - zero) = +0.000**

   The actor is identical across all four regime substitutions, to the precision of the scoring loop. The regime path is unambiguously dead.

2. **Leakage delta**: 0.000. This is fully consistent with the regime path being unused. There is nothing to leak — the actor isn't using regime info at all, so we can't distinguish "regime helps but the encoder leaked it through stock features" from "regime is dead weight".

3. **b1 vs b2 critic**: DEFERRED. Now moot given (1): the critic's regime pathway is also unused (the regime input is the same `regime_emb` broadcast that the actor ignores). Even if the critic's value MLP were re-architected to use it, the ACTOR is regime-blind and its decisions wouldn't change.

## Next phase

Given Phase 21's negative result, the priorities for Phase 22 are different from what the Phase 21 plan envisioned:

* **Cherry-pick the `last_obs_t` timing fix to V1's main branch** and re-run Phase 16a as a sanity check. The fix is a genuine bug; V1 was tolerant but not necessarily optimal under it. If V1 with the timing fix outperforms +0.428, that's a free win without any architectural change.
* **Try regime indicator as a per-stock factor instead of split-head**: append the 8 regime features to every stock's per-stock factor vector (so they enter the per-stock encoder directly, not via a parallel head). This keeps V1's monolithic flow but gives the encoder access to regime context. Much smaller change vs Phase 21's hard fork.
* **If revisiting split-head, replace concat with FiLM-style modulation** (regime modulates per-stock embedding via per-channel scale + bias). The concat path's regime contribution gets washed out; FiLM forces regime to bias every stock's embedding directly.
* Phase 18 ens_rankmean6 fresh-holdout gate (≥40 days post-2026-04-24) still applies — fresh data collection is unrelated to the V1/V2 question and should proceed regardless.
* Update the spec doc (commit f60741d) to reflect the two known bugs (mask magnitude, `last_obs_t` timing) and the RegimeEncoder input-LayerNorm removal.

**Phase 21 is officially closed as REJECTED.** No multi-seed sweep, no further training under this architecture.

## Artifacts

```
runs/phase21_21a_v2_drop_mkt_seed42/
  ppo_final.zip
  checkpoints/ppo_*_steps.zip
  metadata.json
  training_summary.json
  oos_sweep.{md,json}                # post-eval
  phase21_sanity_checks.json          # post-sanity
  decision_log.md                     # narrative

handoffs/2026-05-05-phase21-v2-architecture/
  HANDOFF_2026-05-05_phase21.md       (this file)

src/aurumq_rl/index_dict_rollout_buffer.py    (NEW)
scripts/_phase21_sanity_checks.py             (NEW)

docs/superpowers/specs/2026-05-05-phase21-v2-architecture-design.md
docs/superpowers/plans/2026-05-05-phase21-v2-architecture.md
```
