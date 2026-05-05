# Phase 21 — V2 Architecture Hard Switch

> 2026-05-05. Hard fork from V1: Dict observation space + split-head policy
> + 8 v0 regime features + IndexOnlyDictRolloutBuffer +
> is_suspended_default_True fix. Phase 16-19 zips become forensic
> artifacts; the new V2 path is the only training path going forward.

## TL;DR

* Architecture per `docs/superpowers/specs/2026-05-05-phase21-v2-architecture-design.md`.
* Implementation plan at `docs/superpowers/plans/2026-05-05-phase21-v2-architecture.md`.
* Phase 21A 300k seed=42 sanity train: best `vs_random_p50_adjusted = ___`
  (vs Phase 16a baseline +0.428: Δ = ___).
* Sanity check 1 (regime ablation): see `phase21_sanity_checks.json`.
* Sanity check 2 (leakage delta): ___ .
* Sanity check 3 (b1 vs b2): DEFERRED to Phase 22 — see plan Task 5.4 Step 3.

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

## Migration

V1 zips at `models/production/phase16_*` `phase17_*` `phase18_*` `phase20_*` are NO LONGER LOADABLE under V2 codebase. **Do not delete them.** They remain on disk and OSS as forensic artifacts; several reports cite their sha256 digests.

V2 zips initially land at `runs/phase21_*/`. Promotion to `models/production/` requires a fresh-holdout pass (Phase 19's INSUFFICIENT verdict still applies — need ≥40 fresh post-2026-04-24 dates).

## Phase 21A sanity train — TO BE FILLED

Configuration:
- panel: `data/factor_panel_combined_short_2023_2026.parquet`
- train window: 2023-01-03 .. 2025-06-30
- OOS window: 2025-07-01 .. 2026-04-24
- universe: main_board_non_st, n=____, factors=___
- 300k timesteps, n_envs=16, episode=240, batch=1024, n_steps=1024, n_epochs=10
- learning_rate=1e-4, target_kl=0.30, max_grad_norm=0.5
- top_k=30, forward_period=10, seed=42
- regime_encoder_hidden=64, regime_encoder_out_dim=16, critic_token_hidden=64
- rollout_buffer=index (IndexOnlyDictRolloutBuffer)

Result:

| metric | Phase 16a baseline | Phase 21A | Δ |
|---|---:|---:|---:|
| best step | 224928 | ___ | |
| adj Sharpe | +1.593 | ___ | ___ |
| **vs random p50 adj** | **+0.428** | **___** | **___** |
| non-overlap Sharpe | +1.112 | ___ | ___ |
| IC | +0.0143 | ___ | ___ |

Verdict: ___

## Three architectural sanity checks

1. **Actor regime ablation**: see `phase21_sanity_checks.json`.
   - real: ___ adj_S
   - zero: ___ adj_S
   - batch-mean: ___ adj_S
   - shuffled: ___ adj_S
   - delta(real - zero): ___

2. **Leakage delta**: ___ . Interpretation: ___.

3. **b1 vs b2 critic**: DEFERRED to Phase 22.

## Next phase

* Phase 22 multi-seed sweep on V2 to rebuild the ensemble baseline (Phase 18 ens_rankmean6 is stranded under V1).
* Phase 22 fresh-holdout collection (≥40 days post-2026-04-24 needed for production promotion).
* Phase 22 Ubuntu-side regime enrichment: VIX-equivalent, fund flows, real index series (replace the equal-weight `idx_ret` proxy with a market-cap-weighted index).
* Phase 22 b1 vs true-b2 critic ablation (deferred from Phase 21).
* Phase 22 update spec doc to reflect the RegimeEncoder input-LayerNorm removal.

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
