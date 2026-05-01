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

### 5.1 Interface choice — SB3 VecEnv, NOT gymnasium.vector.VectorEnv

`GPUStockPickingEnv` inherits from `stable_baselines3.common.vec_env.VecEnv` (the SB3 base class, **not** `gymnasium.vector.VectorEnv` — they have different method signatures and SB3's PPO calls the SB3 one directly). The required SB3 contract:

```python
class GPUStockPickingEnv(VecEnv):
    # required by SB3 VecEnv ABC:
    def reset(self) -> np.ndarray | torch.Tensor: ...
    def step_async(self, actions): ...           # we ignore async; just stash
    def step_wait(self) -> tuple[obs, rewards, dones, infos]: ...
    def close(self) -> None: ...
    def seed(self, seed=None): ...
    def env_method(self, method_name, *args, **kwargs): ...  # raise NotImplementedError
    def get_attr(self, attr_name, indices=None): ...
    def set_attr(self, attr_name, value, indices=None): ...
    def env_is_wrapped(self, wrapper_class, indices=None): ...
```

`step_async / step_wait` are SB3's two-phase pattern; for a single-process GPU env they collapse: `step_async(actions)` stashes actions, `step_wait()` does the actual tensor op and returns. SB3's PPO `OnPolicyAlgorithm.collect_rollouts` only ever calls them in pairs.

### 5.2 Constructor

```python
class GPUStockPickingEnv(VecEnv):
    def __init__(
        self,
        panel: torch.Tensor,        # (n_dates, n_stocks, n_factors) fp32, cuda
        returns: torch.Tensor,      # (n_dates, n_stocks)            fp32, cuda
        valid_mask: torch.Tensor,   # (n_dates, n_stocks)            bool, cuda
        n_envs: int,
        episode_length: int = 240,  # trading days per rollout (1 yr)
        forward_period: int = 10,   # reward horizon (days)
        top_k: int = 30,
        cost_bps: float = 30.0,
        device: str = "cuda",
        seed: int | None = None,
    ): ...
```

### 5.3 `valid_mask` provenance (was missing in v1)

`valid_mask[t, s]` is `True` iff stock `s` is tradable on date `t`. Built **once at env-construction time** on CPU and shipped to GPU:

```python
valid_mask = (
    ~is_st                         # not ST / *ST
    & ~is_suspended                # vol > 0
    & (days_since_ipo >= 60)       # past new-stock protection
    & ~limit_hit                   # not at price limit (board-aware: main ±10%, ChiNext/STAR ±20%, BSE ±30%, ST ±5%)
)
```

`limit_hit` comes from `src/aurumq_rl/price_limits.py::compute_dynamic_limits()` applied to the panel's `pct_chg` column board-by-board. All four input arrays are produced by `data_loader.load_panel()` already; we just bitwise-AND them once and `.to(device)`.

### 5.4 `step_wait` semantics

```python
def step_wait(self) -> tuple[torch.Tensor, np.ndarray, np.ndarray, list[dict]]:
    action = self._pending_action          # shape (n_envs, n_stocks), fp32 on cuda
    # 1. mask invalid stocks (so they can never enter top-K):
    action = action.masked_fill(~self.valid_mask[self.t], float('-inf'))
    # 2. top-K: top_idx shape (n_envs, top_k)
    top_idx = torch.topk(action, k=self.top_k, dim=-1).indices
    # 3. forward returns gathered for the K picked stocks:
    fwd_t = self.t + self.forward_period
    fwd_rets = self.returns[fwd_t].gather(1, top_idx)         # (n_envs, top_k)
    rewards = fwd_rets.mean(dim=-1) - self.cost_bps / 1e4
    # 4. turnover penalty using prev top-K membership (Jaccard distance):
    rewards -= self.turnover_coef * self._jaccard(top_idx, self.prev_top_idx)
    self.prev_top_idx = top_idx
    # 5. advance time per env, vectorized:
    self.t += 1
    self.steps_done += 1
    dones = self.steps_done >= self.episode_length
    # 6. for done envs, build SB3-style episode info, then auto-reset
    #    (SB3 VecEnv contract: dones envs are silently reset; the obs
    #    returned for those envs is the FRESH obs after reset, not the
    #    terminal obs. info["episode"] = {"r": <total_reward>, "l":
    #    <length>} is what the Monitor wrapper / SB3 logger consumes
    #    for tb metrics; without it ep_rew_mean stays empty.):
    infos: list[dict] = [{} for _ in range(self.n_envs)]
    if dones.any():
        for i in dones.nonzero(as_tuple=True)[0].tolist():
            infos[i]["episode"] = {
                "r": float(self.episode_returns[i].item()),
                "l": int(self.steps_done[i].item()),
            }
        self._reset_done_envs(dones)        # resample t, zero steps_done & prev_top_idx
    obs = self.panel[self.t]               # (n_envs, n_stocks, n_factors)
    return obs, rewards.cpu().numpy(), dones.cpu().numpy(), infos
```

Notes on the SB3 VecEnv contract:
- `obs` is returned **as a numpy array**, NOT a cuda tensor. Although SB3 docs hint at `obs_as_tensor` accepting tensors, in stable-baselines3 2.8.0 (`stable_baselines3.common.utils.obs_as_tensor`) the function only handles `np.ndarray` and `dict[str, np.ndarray]` — passing a `torch.Tensor` raises `TypeError`. So `step_wait()` must materialise the cuda obs to numpy at the VecEnv boundary (`return self._current_obs().detach().cpu().numpy()`). The internal panel stays on cuda; only the SB3-facing return value is numpy. SB3 will then `.cpu().numpy() → torch.tensor → .to(device='cuda')` it again on the way to the policy. The round-trip cost is captured in §5.6 and mitigated by P1 (GPURolloutBuffer).
- `rewards` and `dones` similarly go to numpy because SB3's `RolloutBuffer.add()` expects numpy arrays — see §5.6.
- For done envs, the returned obs is the **post-reset** obs (auto-reset semantics), not the terminal obs. PPO doesn't need `terminal_observation` (it bootstraps with `next_value=0` on done), so we don't populate it. Off-policy algos would; out of scope here.
- `info["episode"]` is what the legacy gymnasium `Monitor` wrapper provides; SB3's `WandbMetricsCallback` and tb logger read it to populate `rollout/ep_rew_mean` / `rollout/ep_len_mean`. We construct it manually here because there is no `Monitor` wrapper in our single-process flow.
- `steps_done[i] = 0` and `prev_top_idx[i] = 0` are reset per-env via `_reset_done_envs(dones)`.

### 5.5 Why a single-process VecEnv (no SubprocVecEnv)

SubprocVecEnv adds fork-per-env + pickle obs over IPC. Obs shape `(3014, 343)` ≈ 1 M floats / step / env; IPC alone costs more than the env step. A single-process batched-tensor env eliminates that overhead entirely.

### 5.6 ⚠️ Critical SB3 RolloutBuffer overhead (Issue 2)

SB3's `RolloutBuffer.add(obs, action, reward, ...)` stores everything as **numpy arrays on CPU**. Even though our env returns cuda tensors, SB3 will:

1. Pull obs from GPU → CPU → numpy, store in buffer (`step_wait`).
2. Pull buffer slices back GPU at SGD time.

Per step: `(n_envs, n_stocks, n_factors) × fp32 = 12 × 3014 × 343 × 4 B ≈ 50 MB / step / direction`. At `n_steps=1024`: ~50 GB cross-PCIe per rollout. PCIe gen4 x16 ≈ 32 GB/s → ~1.5 s per rollout just in transfers. **Real impact: fps target may degrade from 1000-3000 down to 500-800**. Not catastrophic but well below ceiling.

**Mitigation tiers (in order of effort):**
1. **P0 (default for v1):** accept the overhead, measure actual fps in smoke. If ≥ 500 fps and GPU mean ≥ 70 %, ship.
2. **P1 optimisation (if smoke fps < 500):** subclass `RolloutBuffer` → `GPURolloutBuffer` that keeps obs on GPU. SB3's PPO accepts a `rollout_buffer_class` kwarg in `__init__` (PPO is on-policy; this is the right hook — `replay_buffer_class` is for off-policy algos like SAC and does not apply here). One pre-allocated tensor of shape `(n_steps, n_envs, *obs_shape)` on cuda. Eliminates both directions of transfer.
3. **P2 (if SB3 buffer hooks are too restrictive):** bypass SB3's `collect_rollouts` entirely with a custom rollout loop (~150 lines).

The framework targets **P0 first** for speed of delivery; the smoke measurement decides whether P1 is needed.

### 5.7 Public properties

```python
self.observation_space = gym.spaces.Box(low=-inf, high=inf, shape=(n_stocks, n_factors), dtype=np.float32)
self.action_space      = gym.spaces.Box(low=0.0, high=1.0, shape=(n_stocks,),            dtype=np.float32)
self.num_envs          = n_envs
```

---

## 6. `PerStockEncoderPolicy` — detailed contract

### 6.1 Per-stock extractor (returns a dict, not a flat tensor)

```python
class PerStockExtractor(BaseFeaturesExtractor):
    """
    obs:  (B, n_stocks, n_factors)
    out:  dict with keys:
        "per_stock":  (B, n_stocks, encoder_out_dim)  — for action head
        "pooled":     (B, encoder_out_dim)            — for value head
    Returns a dict (not a flat tensor) so the policy's forward() can route
    each stream to the right head; this is incompatible with SB3's default
    BaseFeaturesExtractor contract (which returns a single tensor) so we
    override the policy's forward / _predict / evaluate_actions to handle
    the dict directly. See §6.3.
    """
    def __init__(self, obs_space, hidden=(128, 64), out_dim=32):
        # Shared MLP: n_factors → 128 → 64 → out_dim, total ~30K params
        # Applied identically per stock via reshape(B*S, F) → reshape back
```

### 6.2 Why we override SB3's standard `forward()` pipeline (Issue 3)

SB3 `ActorCriticPolicy.forward()` flow is:

```
features = self.features_extractor(obs)              # expects: tensor (B, F_dim)
latent_pi, latent_vf = self.mlp_extractor(features)  # expects: tensor → (tensor, tensor)
action_logits = self.action_net(latent_pi)
value         = self.value_net(latent_vf)
```

Our network has **fundamentally different shapes per branch** — action head needs (B, n_stocks, D), value head needs (B, D). The default `mlp_extractor.split_out` doesn't fit. Resolution:

- **Disable** `mlp_extractor` (set `net_arch=dict(pi=[], vf=[])` and patch `_build_mlp_extractor` to a no-op identity).
- **Override** `forward(obs, deterministic=False)`: read `per_stock` + `pooled` from the extractor's dict, route them through custom `action_net` (Linear `out_dim → 1` applied per stock to produce scores) and custom `value_net` (MLP on pooled).
- **Override** `_predict(obs, deterministic)`, `evaluate_actions(obs, actions)`, `predict_values(obs)` — these all call into `forward` internally.

```python
class PerStockEncoderPolicy(ActorCriticPolicy):
    features_extractor_class = PerStockExtractor

    def _build_mlp_extractor(self):
        # No-op: we don't use SB3's mlp_extractor split. Pass through identity.
        self.mlp_extractor = _IdentityMlpExtractor(features_dim=self.features_dim)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        # Replace action_net / value_net with our custom heads:
        encoder_out_dim = self.features_extractor.out_dim
        self.action_net = nn.Linear(encoder_out_dim, 1)        # per-stock score
        self.value_net  = nn.Sequential(
            nn.Linear(encoder_out_dim, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.action_dist = DiagGaussianDistribution(self.action_space.shape[0])
        # log_std is a single learnable per-stock (or shared) parameter

    def forward(self, obs, deterministic=False):
        feats = self.features_extractor(obs)
        per_stock = feats["per_stock"]   # (B, S, D)
        pooled    = feats["pooled"]      # (B, D)
        scores = self.action_net(per_stock).squeeze(-1)  # (B, S)
        values = self.value_net(pooled).squeeze(-1)      # (B,)
        distribution = self.action_dist.proba_distribution(scores, self.log_std)
        actions = distribution.get_actions(deterministic=deterministic)
        log_probs = distribution.log_prob(actions)
        return actions, values, log_probs

    def evaluate_actions(self, obs, actions):
        # Mirror forward() but score given actions for PPO ratio computation
        ...
    def _predict(self, obs, deterministic=False):
        # Inference-only path
        ...
    def predict_values(self, obs):
        # Used by PPO when computing GAE
        ...
```

### 6.3 Action distribution

`DiagGaussianDistribution(n_stocks)` — per-stock independent Gaussian. Mean comes from the per-stock encoder; `log_std` is a single learnable parameter of shape `(n_stocks,)` initialised to `log(0.5)`. Action samples are clipped to `[0, 1]` by the env (existing `Box(0, 1)` action space). KL is computed on the unbounded distribution, which is standard PPO practice.

### 6.4 Permutation equivariance and invariance

- **Action head is equivariant**: `action_net(features[:, π])  ==  action_net(features)[:, π]` for any permutation π of the stock axis. Holds because the action_net is a single Linear applied per stock — no cross-stock interaction.
- **Value head is invariant**: `value_net(mean_pool(features[:, π]))  ==  value_net(mean_pool(features))`. Mean-pool is symmetric in stock axis.

Both properties are mathematically required for a top-K stock-picking task and are validated explicitly in tests (§8).

### 6.5 bf16 autocast

Wrap forward / backward in `torch.amp.autocast('cuda', dtype=torch.bfloat16)`. Optimizer state stays fp32 — no `GradScaler` needed for bf16 (bf16 has same dynamic range as fp32, no underflow risk). Verified by SB3 issue tracker: bf16 + Adam works without modification.

Risk note: PPO's `approx_kl = ((old_log_prob - new_log_prob) ** 2).mean() / 2` is computed inside the autocast block. bf16's 7-bit mantissa is theoretically lossier here, but the mean-of-squares operation is well-conditioned. No expected functional impact; will sanity-check during smoke that approx_kl values are comparable to baseline.

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

### C. Permutation importance (per-group ΔIC) — per-date cross-section shuffle

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
    Per-date cross-section shuffle: for each prefix, on each date t,
    randomly permute the stock dimension of every factor column belonging
    to that prefix. This preserves each factor's per-date marginal
    distribution and time-series structure (a date's factor mean / std
    don't change), while destroying the cross-section ranking that the
    policy uses to pick top-K. This shuffle isolates "does this group
    carry cross-sectional information that the policy actually uses."

    Implementation: for prefix p with column indices [i_1, .., i_k],
        for each t in [0, T):
            π_t = random permutation of [0, S)
            shuffled[t, :, i_1..i_k] = val_panel[t, π_t, i_1..i_k]
    Then run OOS backtest on `shuffled` and compare IC, IR, Sharpe to
    the unshuffled baseline.

    Why not other shuffle modes:
      - Per-stock time-series shuffle would destroy time but keep
        cross-section (wrong: top-K cares about cross-section).
      - Full (date, stock) shuffle destroys both — too aggressive,
        tells us less about why the model uses the group.

    Returns: dict[prefix, {ic_drop_mean, ic_drop_std, sharpe_drop_mean,
                            sharpe_drop_std, n_factors, n_seeds}]
    averaged over `n_seeds` independent shuffle seeds.
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

- **Parameter count:** ≤ 100k total trainable parameters for default `(128, 64)` encoder + value head.
- **Action equivariance:** for a random obs `(B, S, F)` and a random permutation π of the stock axis, the action mean output satisfies `action_mean(obs[:, π])  ==  action_mean(obs)[:, π]` (within fp32 tolerance, e.g. `1e-5`).
- **Value invariance:** for the same obs and π, `value(obs[:, π])  ==  value(obs)` (within tolerance). Mean-pool is symmetric.
- **Output shapes:** action distribution mean shape == `(B, S)`; `log_std` shape == `(S,)`; value shape == `(B,)`.
- **bf16 autocast smoke:** running `forward()` inside `torch.amp.autocast('cuda', dtype=torch.bfloat16)` produces values within 1e-3 of the fp32 path. Loss and gradients remain finite.

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

1. **SB3 RolloutBuffer is numpy/CPU (Issue 2 from cross-review):** every rollout step does cuda → cpu → numpy → cpu → cuda for obs / actions / rewards. Estimated cost ~1.5 s per rollout in PCIe transfers, may cap fps at 500-800 instead of the 1000-3000 ceiling. **Mitigation tiers**: (P0) accept and measure, (P1) implement `GPURolloutBuffer` subclass that pre-allocates `(n_steps, n_envs, *obs_shape)` on cuda and skips numpy entirely, (P2) bypass SB3's `collect_rollouts` with a custom rollout loop. Plan starts with P0; smoke fps measurement decides whether to escalate.
2. **SB3 ActorCriticPolicy default `forward()` pipeline assumes flat tensor (Issue 3 from cross-review):** our extractor returns a dict `{per_stock, pooled}` and the value/action heads need different shapes. We override `forward`, `_predict`, `evaluate_actions`, `predict_values`, and replace `mlp_extractor` with a no-op identity. Risk: any SB3 internal that calls these methods with assumptions about shapes will break — we'll discover this during integration testing.
3. **SB3 VecEnv contract surface:** must implement all of `reset / step_async / step_wait / close / seed / env_method / get_attr / set_attr / env_is_wrapped`. Most are trivial (`raise NotImplementedError` for `env_method`), but if `OnPolicyAlgorithm` calls one we don't support, smoke fails. Mitigation: read SB3 source for `collect_rollouts` once before implementing.
4. **Action distribution choice:** existing env uses continuous `Box(0, 1, n_stocks)`; we default to `DiagGaussianDistribution(n_stocks)`. Per-stock independent log_std is a learnable `(n_stocks,)` parameter. If KL spikes recur (as in R1-R3), revisit with shared scalar log_std or switch to `BernoulliDistribution` for cleaner top-K semantics.
5. **bf16 autocast and KL computation:** approx_kl is computed inside the autocast block. bf16 has fp32-equivalent dynamic range so no underflow expected. Sanity-check during smoke that approx_kl is within 5 % of fp32 baseline; if not, fall back to fp32 for the KL computation only.
6. **VRAM headroom:** estimate 5 GB total is conservative; if `batch_size` scales to 2048 or activations balloon under torch's checkpoint-free path, expect to need to dial batch back to 512–1024. Will discover during smoke.
7. **Permutation importance shuffle interpretation:** we shuffle per-date cross-section (Issue 4 from cross-review) — preserves per-date marginal but breaks cross-section ranking. This is the right shuffle for top-K but may under-report the importance of "absolute level" factors (e.g. `mkt_volatility` whose value matters in absolute terms, not just rank). Document this caveat in the importance JSON output and in the web panel tooltip.

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

1. All five parallel agents (env, policy, importance, web, train) pass their tests and merge cleanly.
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
