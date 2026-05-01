# GPU-Vectorized RL Training Framework — Design

**Date:** 2026-05-01
**Author:** Claude (Opus 4.7) under user direction
**Status:** Approved for plan-writing
**Replaces:** scripts/train.py + src/aurumq_rl/{env.py, portfolio_weight_env.py} as primary training path
**Keeps as fallback:** existing flat-MLP / SubprocVecEnv path stays in tree for regression baseline

---

## 1. Goals

Replace the current flat-MLP + SubprocVecEnv training stack with a GPU-vectorized stack that

1. Uses **all ~343 factor columns** of the combined SHORT panel — no more arbitrary `n_factors=64` cap.
2. Targets **fps 1000–3000** (vs current 77) by keeping panel + rollout + SGD entirely on GPU. 5M training steps in roughly 1 h instead of 18 h.
3. Achieves **GPU 90 %+ stable utilisation** (vs current peak 100 % / mean 32 %, with 3.8 GB / 12 GB VRAM used).
4. Produces **per-factor-group importance** as a first-class training-framework output via Integrated Gradients (saliency) + permutation importance (IC drop). User can read off which prefixes (`alpha_`, `gtja_`, `mf_`, `mfp_`, `hk_`, `fund_`, `inst_`, `mg_`, `senti_`, `sh_`, `ind_`, `mkt_`, `hm_`, `cyq_`) drive the eventual IC / Sharpe.
5. Treats the codebase as a **scientific framework** — modular, tested, comparable to the existing baseline. Old code stays in tree for fallback and regression checks.

---

## 2. Hard constraints (non-negotiable)

- **Panel scope:** SHORT only (`factor_panel_combined_short_2023_2026.parquet`, 800 dates × 5643 stocks × 355 cols → after `main_board_non_st` filter ≈ 800 × 3014 × 343). LONG is deferred — pre-2023 regimes are too different.
- **Numerics:** panel + model weights + optimizer state are **fp32**. Forward / backward passes use **bf16 autocast** (PyTorch tensor cores). No fp16 (loss scaling complexity), no fp64 (4070's 1:64 fp64:fp32 throughput would lose 30× speed).
- **Hardware:** RTX 4070 (12 GB VRAM), i7-13700K (24 logical cores), 64 GB host RAM.
- **VRAM budget (estimated):** panel 3.3 GB + model+optim < 0.2 GB + bf16 activations 0.4 GB + bf16 grads 0.4 GB + headroom ~7 GB = comfortably under 12 GB. Headroom can fund larger `batch_size` (1024+) or more parallel envs.
- **Action / reward semantics** stay identical to the existing `StockPickingEnv` (top-K mean log-return − cost − turnover). The mathematical contract does not change; only the implementation moves to tensors on GPU.
- **No regression of existing pipeline:** old `train.py` / `StockPickingEnv` keep working (feature_group_weights, ablations, ONNX export, eval_backtest all still flow through the old path). Migration is opt-in via `train_v2.py`.

---

## 3. Architecture

```
┌─ Training loop (single process, single GPU) ──────────────────────────┐
│                                                                       │
│  GPUStockPickingEnv  (panel & returns as torch.Tensor on cuda)        │
│       ↓ obs (B, n_stocks, n_factors)                                  │
│       ↓                                                               │
│  PerStockEncoderPolicy  (SB3 ActorCriticPolicy subclass)              │
│       ├─ shared per-stock MLP: n_factors → 128 → 64 → 1 (~30K params) │
│       ├─ value head: mean-pool features → MLP → scalar                │
│       └─ stochastic policy: Bernoulli / DiagGaussian over scores      │
│       ↓ action (B, n_stocks)  (per-stock continuous logits)           │
│       ↓                                                               │
│  reward = top-K(action) → mean forward return − cost − turnover       │
│  done    = episode_length reached (e.g. 240 trading days = 1 yr)      │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
                          │
                          ↓
┌─ Factor importance (post-training, A + C combined) ───────────────────┐
│                                                                       │
│  A. Integrated Gradients on encoder                                   │
│     → per-factor saliency (single forward+backward pass batch)        │
│                                                                       │
│  C. Permutation importance                                            │
│     → drop / shuffle each group's columns across the OOS panel,       │
│       re-run backtest, record ΔIC                                     │
│                                                                       │
│  Output: runs/<id>/factor_importance.json                             │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
                          ↓
┌─ Web visualisation ───────────────────────────────────────────────────┐
│  - Existing training-curve panels keep working                        │
│  - New FactorImportancePanel: bar chart by group + per-factor heatmap │
└───────────────────────────────────────────────────────────────────────┘
```

---

## 4. New / modified modules

| Path | Status | Purpose |
|---|---|---|
| `src/aurumq_rl/gpu_env.py` | **NEW** | `GPUStockPickingEnv` — implements the `gymnasium.vector.VectorEnv` interface but keeps panel & returns as torch tensors on cuda. `step()` is a tensor-index + reward computation; no numpy, no IPC. |
| `src/aurumq_rl/feature_extractor.py` | **NEW** | `PerStockExtractor` — reshape obs `(B, n_stocks, n_factors)` and apply the shared per-stock MLP. Used as `features_extractor_class` in the SB3 policy. |
| `src/aurumq_rl/policy.py` | **NEW** | `PerStockEncoderPolicy(ActorCriticPolicy)` — wires the per-stock extractor, defines a stochastic action distribution over per-stock scores, plus a mean-pool value head. |
| `src/aurumq_rl/factor_importance.py` | **NEW** | Pure-functions library: `integrated_gradients(policy, panel_sample) → (n_factors,) saliency`, `permutation_importance(policy, panel, returns, prefixes) → dict[str, ic_drop]`. |
| `scripts/train_v2.py` | **NEW** | Training entry point using the new env + policy. Mirrors `train.py` flag surface where it overlaps; new flags only for genuinely new options (`--episode-length`, `--encoder-hidden`, `--bf16-autocast`). |
| `scripts/eval_factor_importance.py` | **NEW** | Post-training CLI: load run-dir model, run IG + permutation importance on the OOS panel, write `factor_importance.json`. |
| `web/components/FactorImportancePanel.tsx` | **NEW** | Bar chart of per-group importance + per-factor saliency heatmap. Reads `/api/runs/[id]?part=factor-importance`. |
| `web/app/runs/[...id]/page.tsx` | **MODIFY** | Read `factor_importance.json` server-side; render `<FactorImportancePanel>` when present. |
| `web/lib/runs.ts` + `runs-shared.ts` | **MODIFY** | Add `readFactorImportance()` + `FactorImportance` shared type. |
| `tests/test_gpu_env.py` | **NEW** | Numerical equivalence vs `StockPickingEnv`, GPU residency, no-numpy-allocation in step. |
| `tests/test_policy.py` | **NEW** | Permutation equivariance, parameter count < 100k, output shapes. |
| `tests/test_factor_importance.py` | **NEW** | Synthetic panel with one planted "true" factor; saliency + permutation must rank it first. |
| `src/aurumq_rl/env.py` | **NO CHANGE** | Stays as fallback / regression baseline. |
| `src/aurumq_rl/portfolio_weight_env.py` | **NO CHANGE** | Stays. |
| `scripts/train.py` | **NO CHANGE** | Stays as the legacy baseline trainer. |

---

## 5. `GPUStockPickingEnv` — detailed contract

```python
class GPUStockPickingEnv:
    """
    Implements gymnasium.vector.VectorEnv interface (NOT gym.Env).
    All n_envs share a single panel tensor on cuda; per-env state is just
    a time-index vector, kept on cuda. step() is one tensor op.
    """

    def __init__(
        self,
        panel: torch.Tensor,        # (n_dates, n_stocks, n_factors) fp32, cuda
        returns: torch.Tensor,      # (n_dates, n_stocks)            fp32, cuda
        valid_mask: torch.Tensor,   # (n_dates, n_stocks)            bool, cuda
                                    #   False = ST / suspended / new-IPO / limit-hit
        n_envs: int,
        episode_length: int = 240,  # trading days per rollout
        forward_period: int = 10,   # reward horizon (days)
        top_k: int = 30,
        cost_bps: float = 30.0,
        device: str = "cuda",
        seed: int | None = None,
    ): ...

    def reset(self, *, seed=None, options=None) -> tuple[obs, info]:
        # Sample n_envs different start indices uniformly from the
        # admissible window [0, n_dates - episode_length - forward_period]
        # Set self.t = those indices, self.steps_done = 0
        # Return obs = panel[self.t]  shape (n_envs, n_stocks, n_factors)

    def step(self, action: torch.Tensor) -> tuple[obs, rewards, terminated, truncated, info]:
        # action  shape (n_envs, n_stocks)  fp32
        # 1. mask invalid stocks: action = action.masked_fill(~valid_mask[self.t], -inf)
        # 2. top-K: top_idx = torch.topk(action, k=top_k, dim=-1).indices
        # 3. forward returns: r = returns[self.t + forward_period].gather(1, top_idx).mean(-1)
        # 4. cost: -cost_bps/1e4
        # 5. turnover: |action - prev_action|.sum() * coef  (held in self.prev_action)
        # 6. self.t += 1; self.steps_done += 1
        # 7. terminated = self.steps_done >= self.episode_length
        # 8. for terminated envs: resample t, reset prev_action — NO numpy roundtrip
        # Return obs = panel[self.t], rewards, terminated, truncated, info

    @property
    def observation_space(self) -> gym.spaces.Box: ...   # (n_stocks, n_factors)
    @property
    def action_space(self) -> gym.spaces.Box:      ...   # (n_stocks,)
    @property
    def num_envs(self) -> int: ...                       # = n_envs
```

**Why VectorEnv directly, not gym.Env + SubprocVecEnv:** SubprocVecEnv adds a fork-per-env + pickle obs over IPC. With obs shape (3014 × 343) = 1 M floats per step, IPC alone costs more than the env step itself. Implementing VectorEnv directly lets all envs share one panel tensor and step in a single batched tensor op.

**SB3 compatibility:** `OnPolicyAlgorithm.collect_rollouts` accepts any `VecEnv` whose `reset()` / `step_async()` / `step_wait()` follow the contract. We provide a thin shim if SB3 needs the legacy non-Vector interface.

---

## 6. `PerStockEncoderPolicy` — detailed contract

```python
class PerStockExtractor(BaseFeaturesExtractor):
    """
    obs:  (B, n_stocks, n_factors)
    out:  (B, n_stocks, encoder_out_dim)   per-stock features
          (B, encoder_out_dim)             pooled features for value head
    """
    def __init__(self, obs_space, n_factors, hidden=(128, 64), out_dim=32):
        # Shared MLP: n_factors → 128 → 64 → out_dim
        # Total params ~30K, applied identically per stock

class PerStockEncoderPolicy(ActorCriticPolicy):
    """
    Action distribution: per-stock independent DiagGaussian over scores
      (continuous, matches existing env's Box(n_stocks,) action space).
    Value head: mean-pool the per-stock features → MLP → scalar.
    """
    features_extractor_class = PerStockExtractor

    def forward(self, obs):
        per_stock_features, pooled = self.features_extractor(obs)
        # action net: linear from per_stock_features (B, S, D) → (B, S) score
        # value net:  MLP from pooled (B, D) → (B,) value
        ...
```

**Permutation equivariance:** the per-stock MLP applies identically to every stock; swapping stock indices produces the same swap in output scores. This is mathematically the correct prior for top-K stock selection — the policy is invariant to stock ordering and only cares about each stock's per-feature signal.

**bf16 autocast:** wrap the forward / backward in `with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):`. Optimizer state stays fp32 (no GradScaler needed for bf16; bf16 has fp32-equivalent dynamic range).

---

## 7. Factor-importance pipeline

### A. Integrated Gradients (per-factor saliency)

```python
def integrated_gradients(
    policy: PerStockEncoderPolicy,
    panel_batch: torch.Tensor,   # (B, n_stocks, n_factors)
    n_alpha_steps: int = 50,
    baseline: torch.Tensor | None = None,  # default: zeros
) -> torch.Tensor:
    """
    Returns (n_factors,) saliency: average |∂(policy_score)/∂(factor_value)|
    integrated along a path from baseline → actual factor values.
    """
```

Aggregated by prefix → per-group saliency mean, std, max.

### C. Permutation importance (per-group ΔIC)

```python
def permutation_importance(
    policy: PerStockEncoderPolicy,
    val_panel: torch.Tensor,       # (T, S, F) OOS panel
    val_returns: torch.Tensor,
    prefixes: list[str],           # ["alpha", "gtja", "mf", "mfp", ...]
    factor_names: list[str],       # mapping column index → name
    n_seeds: int = 5,              # average over multiple shuffle seeds
) -> dict[str, dict[str, float]]:
    """
    For each prefix, shuffle that group's factor columns randomly across
    (date, stock) cells; run OOS backtest; record IC, IR, Sharpe drop
    relative to the unperturbed baseline. Average over n_seeds shuffles.
    """
```

### Output: `runs/<id>/factor_importance.json`

```json
{
  "method": "integrated_gradients_v1+permutation_v1",
  "panel": "factor_panel_combined_short_2023_2026.parquet",
  "val_window": "2025-07-01..2026-04-24",
  "baseline_ic": 0.0123,
  "saliency_per_factor": {"alpha_001": 0.00231, "gtja_042": 0.00185, ...},
  "importance_per_group": {
    "alpha":  {"saliency_mean": 0.0021, "ic_drop_mean": 0.0035, "ic_drop_std": 0.0006, "n_factors": 105},
    "gtja":   {"saliency_mean": 0.0018, "ic_drop_mean": 0.0028, "ic_drop_std": 0.0005, "n_factors": 191},
    "mfp":    {"saliency_mean": 0.0034, "ic_drop_mean": 0.0078, "ic_drop_std": 0.0009, "n_factors": 12},
    ...
  }
}
```

`web/components/FactorImportancePanel.tsx` reads this file and renders:
1. Horizontal bar chart of `ic_drop_mean` per group, sorted desc.
2. Per-factor saliency heatmap (factors x stratification by family) for top-N families.

---

## 8. Testing strategy (TDD)

Every implementation task starts with the tests below.

### `tests/test_gpu_env.py`

- **Numerical equivalence:** with the same panel array and the same actions, `GPUStockPickingEnv` rewards match the existing `StockPickingEnv` rewards within `1e-4` (allowing for fp32 reduction order). Iterate over a 50-step rollout.
- **GPU residency:** after `__init__`, `panel.device.type == "cuda"`, `returns.device.type == "cuda"`.
- **No numpy allocation in step:** wrap step() with a memory-allocation tracer (`tracemalloc` on CPU side); assert no significant CPU allocations grow during a 100-step rollout.
- **VectorEnv contract:** `n_envs` independent rollouts terminate at `episode_length`; per-env start indices differ; reset replaces only the terminated envs.

### `tests/test_policy.py`

- **Parameter count:** `<= 100k` total trainable parameters for default `(128, 64)` encoder + value head.
- **Permutation equivariance:** for a random obs `(B, S, F)` and a random permutation `pi` of stock indices, `policy(obs[:, pi]) == policy(obs)[:, pi]` (within fp32 tolerance).
- **Output shapes:** action distribution param shape == `(B, S)`, value shape `(B,)`.

### `tests/test_factor_importance.py`

- **Synthetic identifiability:** build a synthetic panel with `n_stocks=200`, `n_factors=20`. Plant one "true" factor whose value strongly correlates with `forward_return`; the other 19 are random noise. Train a PerStockEncoderPolicy for 5k steps. Run `integrated_gradients` and `permutation_importance` — both must rank the planted factor in the top 1.
- **Permutation determinism:** running with the same seed twice produces identical ic_drop values.

### End-to-end smoke

`scripts/train_v2.py --total-timesteps 50000 --data-path data/factor_panel_combined_short_2023_2026.parquet --out-dir runs/smoke_v2` must produce a valid model + factor_importance.json + matching backtest, with **OOS Sharpe within ±20 % of the existing R3 50k run** as a no-regression check.

---

## 9. Work decomposition (parallel agents in worktrees)

| Agent | Worktree | Files | Blocked by |
|---|---|---|---|
| `agent-env` | `D:/dev/aurumq-rl-wt-env` | `src/aurumq_rl/gpu_env.py`, `tests/test_gpu_env.py` | none |
| `agent-policy` | `D:/dev/aurumq-rl-wt-policy` | `src/aurumq_rl/feature_extractor.py`, `src/aurumq_rl/policy.py`, `tests/test_policy.py` | none |
| `agent-importance` | `D:/dev/aurumq-rl-wt-importance` | `src/aurumq_rl/factor_importance.py`, `tests/test_factor_importance.py`, `scripts/eval_factor_importance.py` | none (uses synthetic stub of policy) |
| `agent-web` | `D:/dev/aurumq-rl-wt-web` | `web/components/FactorImportancePanel.tsx`, `web/lib/runs*.ts` adds | none (uses fixture JSON) |
| `agent-train` | main repo | `scripts/train_v2.py` integrating env+policy+importance | env + policy merged |

Each agent:
- Works in its own worktree to avoid file-write contention.
- Follows `superpowers:test-driven-development` (write the relevant tests above first, fail, then implement).
- Reports back with: list of commits, test pass/fail summary, any blocker.

After env + policy + importance + web all merge, `agent-train` integrates and runs the smoke. Smoke pass = ready to launch 5M overnight (next user decision).

---

## 10. Risks and known unknowns

1. **SB3 PPO + custom VectorEnv on a single process:** SB3's `OnPolicyAlgorithm.collect_rollouts` was written assuming `VecEnv` with `n_envs` independent processes. Need to verify our single-process batched env satisfies all expected contracts (`reset_infos`, `step_async`/`step_wait` split, episode-info dict format). If a corner case bites, fallback is to write a thin custom rollout loop that bypasses `collect_rollouts`.
2. **Action distribution choice:** existing env uses continuous Box; the per-stock policy can output continuous DiagGaussian (drop-in) or discrete Bernoulli (cleaner top-K interpretation). Defaulting to DiagGaussian to minimise surface change; revisit if KL spikes recur.
3. **Permutation equivariance vs critic:** mean-pool value head is permutation-invariant by construction. Confirmed mathematically correct; flagged here in case of subtle SB3 expectation that value scales with action magnitude.
4. **bf16 autocast and KL computation:** mid-training KL is computed in autocast region. bf16 has the same dynamic range as fp32 so no underflow expected, but worth a sanity check that approx_kl values look comparable to baseline.
5. **VRAM headroom:** estimate 5 GB total is conservative; if `batch_size` scales to 2048 or activations balloon under torch's checkpoint-free path, expect to need to dial batch back to 512–1024. Will discover during smoke.

---

## 11. Out of scope (for this design)

- LONG-panel training (deferred; would require fp16 panel or out-of-core streaming).
- Real-time / live-trading inference path (existing `inference.py` flow stays).
- Multi-GPU / distributed training.
- Replacing the existing portfolio_weight env.
- Drop-one-group ablation (Method B from the brainstorming Q2). Deferred — IG + permutation should already surface the top-3 candidates; explicit ablation can be a follow-up plan against those.
- Backwards migration of old `runs/*` to the new format. They remain readable by the legacy code path.

---

## 12. Acceptance criteria

The framework is "done" when:

1. All four parallel agents' tests pass and their PRs / merges are clean.
2. `train_v2.py` smoke run (50k steps, SHORT panel, all 343 factors) succeeds end-to-end and writes `factor_importance.json`.
3. fps measured during the smoke is **≥ 500** (>= 6× the current 77 baseline) and GPU mean utilisation **≥ 70 %**.
4. `FactorImportancePanel` renders for the smoke run in the web dashboard at http://localhost:3000.
5. No-regression: the 50k smoke's OOS Sharpe is within ±20 % of the existing R3 run's Sharpe (`+3.301`). (The 50k timescale is far below convergence so absolute metric quality is not the test — pipeline correctness is.)

After acceptance, a separate "go" from the user kicks off the 5M overnight run.

---

## 13. Glossary

- **Per-stock encoder / Deep Sets:** a network that applies the same MLP to each "element" of a set (here, each stock's factor row), then aggregates element-wise outputs. Result is permutation-equivariant: shuffling stock indices shuffles the output identically.
- **bf16 autocast:** PyTorch's automatic mixed-precision feature; matmul / linear / conv compute in bfloat16 (fp32-equivalent dynamic range, half memory) on tensor cores; everything else (LayerNorm, loss, gradient accumulation) stays fp32.
- **Integrated Gradients (IG):** axiomatic gradient-attribution method. Saliency = average absolute gradient along a path from a "neutral" baseline to the actual input.
- **Permutation importance:** model-agnostic feature-importance test: shuffle a feature's values and see how much the model's score drops. Group-level when applied to factor-prefix subsets.
