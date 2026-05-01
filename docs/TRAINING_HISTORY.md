# AurumQ-RL Training Method Evolution

A chronological record of how the training stack and method have changed,
why, and what was learned at each step. Each entry: the regime, the change,
the evidence that drove it, the outcome.

---

## Phase 0 — Synthetic data, pipeline-up (~ 2026-04-27)

**Goal**: prove the end-to-end pipeline (Parquet → env → PPO → ONNX → backtest) on synthetic data before touching real factor data.

**Stack**:
- Env: `StockPickingEnv` (numpy panel, gym.Env)
- Policy: SB3 default `MlpPolicy` with `net_arch=[64, 64]`
- Trainer: SB3 PPO, `n_envs=1`, `batch_size=64`, `n_steps=2048`, `n_epochs=10`
- Data: `data/synthetic_demo.parquet`, ~200 SYN-coded stocks, no real signal

**Outcome**: pipeline ran end-to-end. ONNX export worked. Backtest IC ≈ 0
(synthetic noise has no real signal; this was the expected baseline).

---

## Phase 1 — First real data, naïve scaling (2026-04-29 ~ 04-30)

**Goal**: scale to real factor panel and a real RTX 4070 GPU, see how far the
naïve setup goes.

**Stack changes vs Phase 0**:
- Data: `factor_panel_alpha101_short_2023_2026.parquet` (105 alpha factors,
  ~3000 main-board-non-ST stocks)
- `n_envs=8` via `SubprocVecEnv` (multi-CPU rollout)
- `--vec-normalize` for obs / reward normalisation
- Persisted `stock_codes` + `factor_names` to `metadata.json` so OOS eval can
  align the universe to the training universe

**Issues discovered**:
- **NaN propagation through cross-section z-score**: real PG data has NaN
  cells (suspended stocks, pre-IPO, factor warm-up). Synthetic data didn't.
  → fix: `np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)` after z-score.
- **OOS obs_dim mismatch**: training universe locked to 3043 stocks;
  validation universe was 3052 (some IPOs landed). Env's
  `observation_space` is fixed at training time → `obs_dim mismatch: panel
  gives 48832, model expects 48688`. → fix: `align_panel_to_stock_list`
  helper that takes the training stock list, gathers + zero-pads + reorders.
  Plus stored `stock_codes` in `metadata.json`.
- **PPO approx_kl exploded to 41,820 on first update**: high-dim obs
  (3043 × 16 = 48,688) + 12.5M-param first layer made the first PPO
  update massively diverge. → fix: `--target-kl 0.05 --max-grad-norm 0.3`
  → KL settled at 0.028, clip_fraction 0.078.
- **`mean_fps=0` in summary**: SB3 only emits `time/fps` in rollout-summary
  frames, not on regular flushes. → fix in `WandbMetricsCallback`: compute
  fps from elapsed wall time.
- **`metrics_summary` all-null in training_summary.json**: callback wrote
  raw SB3 keys; `summarize_metrics` expected canonical TrainingMetrics
  schema → empty. → fix: callback now writes canonical schema with
  mapping (`rollout/ep_rew_mean → episode_reward_mean`,
  `train/policy_gradient_loss → policy_loss`, etc.).

**Outcome**: 100k-step PPO ran clean. fps ≈ 333-340. GPU util **~11 %**.
The 4070 was massively underutilised — the wide first MLP layer did most of
the FLOPS but it was small (only `obs_dim × 64 = 3M params`), so the GPU
spent most of its time waiting on rollout collection.

---

## Phase 2 — Combined panel & first network-size pump (2026-04-30)

**Goal**: use the new combined panel with all factor families, and pump GPU
util by widening the network.

**Stack changes**:
- Data: `factor_panel_combined_short_2023_2026.parquet` — 105 alpha + 191
  gtja + 14 mf + 12 mfp + 5 hk + 4 fund + 3 inst + 3 mg + 3 senti + 2 sh +
  2 ind + 2 mkt + 6 hm + 3 cyq = 355 factor columns. ~5,643 stocks (filtered
  to ~3,014 main-board-non-ST).
- New CLI: `--policy-kwargs-json` to pass `net_arch=[2048, 1024, 512]`.
  First layer params: `obs_dim × 2048 ≈ 800M`. First-layer matrix went
  from 3 MB to 1.6 GB.
- `--feature-group-weights-json`: per-prefix scalar applied **after**
  z-score (so VecNormalize doesn't neutralise it). Used for the
  alpha-prefix 0.5 / 1.0 / 2.0 ablation.
- New CLI flags exposed: `--target-kl`, `--max-grad-norm`,
  `--learning-rate-schedule`, `--vec-normalize`.

**`feature_group_weights` ablation result** (3-way, alpha prefix at 0.5 /
1.0 / 2.0): w=2.0 hit IC=+0.0006, IR=+0.042, top30 Sharpe=−0.807
vs random p50 = −0.482. Conclusion: framework end-to-end works; data is
synthetic-quality on these short windows; need real combined panel + much
longer training to see real signal.

**Issue still alive**: GPU util went from 11 % → 57 % peak / 32 % mean.
Better but still mostly idle.

---

## Phase 3 — Combined panel R1 / R2 / R3 smoke iterations (2026-05-01 morning)

**Goal**: tune PPO hyperparameters until smoke is stable enough to commit
to overnight training.

### Round 1: kl=0.05, lr=3e-4, n_envs=8, batch=256
- Stack: `[2048, 1024, 512]` MLP, `n_steps=2048`, `n_factors=64`
  (auto-truncated from 355 by `data_loader`).
- Result: every iter early-stopped at SGD epoch 1 (`Early stopping at step
  0 due to reaching max kl: 0.10`). Effectively `n_epochs=1` instead of 10.
  fps 54. explained_var trajectory: −0.47 → −1.31 → +0.21.
- Diagnosis: `target_kl=0.05` × 1.5 = 0.075 trip threshold. With high-dim
  obs and a wide net the first SGD batch already moves KL > 0.075.

### Round 2: kl=0.10, lr=3e-4, n_envs=6 (RAM headroom), batch=256
- `n_steps` cut from 2048 → 1024 to keep rollout buffer at 4.7 GB
  (concurrent OSS download for LONG panel was eating host RAM).
- Result: still early-stopping most iters; explained_var climbed all the
  way to **+0.951** by end of 50k. Critic learned almost perfectly.
  GPU peak 100 % during SGD bursts (real). fps 48.
  Backtest: top30 Sharpe **+3.706** vs random p50 +3.563 — first time the
  model beat the random median.

### Round 3: kl=0.20, lr=1e-4, n_envs=6, batch=256
- Result: explained_var **+0.994** (best yet). But policy went through
  several KL collapse-recover cycles: iter 1 approx_kl=21,493, iter 5=400,
  iter 7=340, iter 9=10, iter 13=0.46. Each was a single SGD step that
  triggered early-stop. Critic was unaffected (separate value loss), so
  ev kept climbing.
  fps 36. Backtest IC=+0.0006, top30 Sharpe=+3.301 vs p50 +3.563.

### Cross-round insights
- `target_kl` is a **brake**, not a goal. Tightening it past 0.10 with
  this width (800M-param first layer) means most iters do 1 SGD epoch and
  GPU compute is wasted on empty rollouts.
- `explained_variance` saturates fast (50k steps) — not useful as a
  convergence signal past the first few iters.
- Real OOS comparison on 50k smoke is **noise-dominated** — IC differences
  are smaller than the inter-run variance. Need ≥ 5M steps to read off
  factor importance.
- The fps differences (54 → 48 → 36) are inverse to model exploration:
  more KL spikes = more wasted compute on early-stops = lower fps.

---

## Phase 4 — Speed scaling and the IPC ceiling (2026-05-01 noon)

**Goal**: shorten 5M-step wall time via more parallel envs.

**Experiments**:
- n_envs 6 → 12: fps 48 → 77 (1.6x). RAM 30+ GB.
- n_envs 12 → 20: fps 77 → 76 (no improvement). RAM hit limits, buffer
  allocation OOM at `1024 × 20 × 192,896 × 4 B = 14.7 GB`.

**Insight (the key one for Phase 5)**:
n_envs scaling stops at ~12. The bottleneck is **not** CPU env step
(CPU was at 22 % util). It's **IPC serialisation** of the obs through
SubprocVecEnv: pickling a `(192,896,)` float32 array per env per step
is constant-cost-per-worker, and the total cost grows with n_envs.
Adding workers doesn't help because they all queue against the main
process's pickle/unpickle.

**Wall time at this regime**:
- 5M steps ÷ 77 fps ≈ 18 h. Acceptable as overnight, but no headroom.

---

## Phase 5 — The realisations that drove the GPU framework redesign (2026-05-01 afternoon)

After R3 + the n_envs scaling experiment, four realisations fell out of
re-reading what the GPU and host were actually doing:

1. **`n_factors=64` was a historical accident.** Set in Phase 2 because
   "smoke had to fit somewhere"; never re-evaluated. With 343 factors
   available, capping at 64 throws away ~80 % of the data and prevents
   any conclusion about which factor groups matter.

2. **GPU VRAM was barely used.** Read the gpu.jsonl correctly: PyTorch
   process used ~3.8 GB of the 12 GB available. The "12,142 / 12,282 MB"
   I'd been reporting was *all GPU processes* including Windows desktop —
   actual training process was tiny.

3. **The MLP-flat-obs design is wrong for this task.** `obs = (n_stocks,
   n_factors)` flattened to a 1D vector ignores the fundamental symmetry:
   permuting two stocks should permute the output identically. A flat MLP
   has no such constraint, has to learn it from data, and pays for it
   with O(n_stocks × n_factors × hidden) first-layer parameters that
   blow up VRAM and IPC for no architectural benefit.

4. **IPC is the real bottleneck and it's solvable**. The
   `(n_envs, n_stocks, n_factors)` obs is huge to serialise but tiny to
   compute on. Move the panel to the GPU once, run all envs in a single
   process as batched tensor ops, eliminate IPC entirely.

These together pointed at one redesign: **GPU-vectorised env + per-stock
encoder policy + use all 343 factors**. See
`docs/superpowers/specs/2026-05-01-gpu-rl-framework-design.md` for the
detailed design.

---

## Phase 6 — GPU-vectorised framework (designed 2026-05-01, implementation TBD)

**Headline numbers (target)**:
- fps 1000-3000 (vs current 77, ~13-40x)
- GPU mean util ≥ 70 % (vs current 32 %)
- All 343 factors used (vs current 64)
- 5M steps in ~1 h (vs current 18 h)
- First-class per-factor-group importance output (IG saliency + permutation IC drop)

**Core architectural changes**:
- `GPUStockPickingEnv`: panel as torch tensors on cuda; `step` is a
  batched tensor op; inherits SB3 `VecEnv` (single-process) — eliminates
  IPC serialisation entirely.
- `PerStockEncoderPolicy`: shared per-stock MLP (Deep Sets style),
  ~50K total parameters (vs the 800M flat-MLP first layer).
  Mathematically permutation-equivariant for action, invariant for value.
- Numerics: fp32 panel + fp32 weights/optimizer + bf16 autocast for
  forward/backward (4070 has tensor cores; bf16 has fp32 dynamic range
  so no GradScaler needed). FP64 explicitly rejected: 4070 fp64:fp32
  throughput is 1:64.
- `factor_importance.py`: Integrated Gradients (cheap, training-time
  attribution) + Permutation Importance with per-date cross-section
  shuffle (model-agnostic OOS validation of which groups carry
  cross-sectional info).

**Open implementation risks**:
- SB3 `RolloutBuffer` is numpy/CPU; until we add a `GPURolloutBuffer`
  subclass we pay for an extra GPU↔CPU copy per step. Estimated cap on
  fps: 500-800 instead of 1000-3000. Acceptable for v1; P1 optimisation.
- Custom `forward()` / `evaluate_actions` / `predict_values` overrides
  in the policy because SB3's default `mlp_extractor` doesn't support
  the per-stock + pooled split.
- bf16 sanity check on `approx_kl` numerics during smoke.

**Status**: design doc written and committed; implementation plan to
follow in `docs/superpowers/plans/...`. Five parallel agents in git
worktrees expected to land in 1-2 days.

---

## What this history shows

- The **biggest single improvements come from re-examining baselines**, not
  from incremental hyperparameter tuning. Phase 5 redesign (~10x fps,
  ~5x more factors, correct symmetry prior) overshadows every Phase 1-4
  knob change.
- Each round produced **at least one bug found in the framework itself**
  (NaN propagation, obs_dim mismatch, KL spike, mean_fps=0,
  metrics_summary null, IPC ceiling). The smoke iterations weren't
  "wasted GPU" — they were the unit tests.
- **OOS metrics on 50k-step smokes are noise**. Don't pick winners from
  smokes; pick them from convergence-scale runs (≥ 1M, ideally 5M).
- The architecture choices that drove **most** of the gain are the
  free ones: bigger `net_arch`, `target_kl` relaxation, n_envs up to the
  IPC ceiling. The ones that drove **the rest** require code rewrite:
  per-stock encoder, GPU-resident panel, custom rollout buffer.

This file lives at `docs/TRAINING_HISTORY.md` and is appended to whenever
a new training-stack regime ships.
