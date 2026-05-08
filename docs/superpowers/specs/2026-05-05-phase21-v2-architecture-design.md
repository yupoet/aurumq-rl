# Phase 21 — V2 Architecture: Dict obs + Split-Head Policy + Dynamic Universe

Status: design approved 2026-05-05.
Hard switch from V1: Phase 16-19 production zips are no longer loadable; new training starts from V2.

## 1. Goal and motivation

Phase 16-20 work has converged on three problems the V1 architecture could not cleanly solve:

1. **Regime overfit through per-stock encoder.** `mkt_*` factors are cross-section constants but were fed through `PerStockExtractor` together with per-stock features. The encoder learned regime-conditioned per-stock biases, producing brittle behaviour across regime shifts (Phase 16: `mkt_*` permutation `sharpe_drop = -0.69`; Phase 17: confirmed not retrain-transferable; Phase 20: long-data multi-regime training collapsed to a compromise policy worse than the short-data Phase 16a baseline).
2. **Universe drift hiding in the encoder.** The fixed-stock-list panel zero-pads pre-IPO stocks; `is_suspended_array` defaults to False so those zero rows look "tradeable" to the encoder's pooling step. Cross-section centering and `opportunity_max` get pulled toward zero by ~10% of fake rows when the train window starts before some stocks listed.
3. **No structural separation between stock-level signal and market-level context.** Even with `mkt_*` dropped, there is no clean place for regime indicators to enter the policy. We need a network where stock signals and regime context are **physically separate inputs** that meet only at the head layer.

Phase 21 V2 solves all three by:

* Splitting the observation into a `gymnasium.spaces.Dict` so `stock` / `regime` / `valid_mask` are physically distinct tensors.
* Splitting the network into a per-stock encoder, a separate regime encoder, and a head that concatenates both before the actor and per-stock value transform.
* Fixing `is_suspended` default-True and propagating `valid_mask` to both the actor (logit masking) and the critic (masked pooling).

The user has chosen a **hard switch** (option C in design discussion): no backwards-compat layer. Phase 16-19 zips become forensic artifacts; the new V2 path is the only training path going forward.

## 2. Architectural overview

```
obs : Dict {
    stock      : (S, F_stock)   float32   per-stock features ONLY (allowlist enforced)
    regime     : (R,)           float32   8 cross-section-constant date-level features
    valid_mask : (S,)           float32 0/1   1 = stock j is tradeable on this date
}
        │
        ├── obs["stock"]  ─── PerStockEncoderV2  ─── stock_emb   (B, S, D)
        │                     [LayerNorm + shared MLP per stock; D = encoder_out_dim, default 32]
        │
        ├── obs["regime"] ─── RegimeEncoder      ─── regime_emb  (B, R')
        │                     [LayerNorm + (R → 64) + SiLU + (64 → R') + LayerNorm; R' = 16 default]
        │
        └─ broadcast: regime_b = regime_emb[:, None, :].expand(-1, S, -1)   → (B, S, R')

        head_in = concat(stock_emb, regime_b, dim=-1)                       → (B, S, D + R')

        ┌── actor_head: Linear(D+R' → 1)
        │   logits = actor_head(head_in).squeeze(-1)                        → (B, S)
        │   logits.masked_fill_(~obs["valid_mask"].bool(), -1e9)
        │   distribution = Normal(loc=logits, scale=exp(log_std))           (HARD MASK on loc)
        │   action_space stays Box(0,1,(S,)); env applies top-K on sampled scores.
        │
        └── critic (TRUE b2):
            value_tokens = value_token_mlp(head_in)                         → (B, S, H)   H = critic_hidden, default 64
            pooled = masked_mean(value_tokens, obs["valid_mask"])           → (B, H)
            value  = value_head(pooled).squeeze(-1)                         → (B,)
```

### Key invariants

* **Schema lock**: per-stock encoder is physically incapable of seeing regime / market / global features. The `obs["stock"]` tensor only carries per-stock factor columns drawn from the allowlist (alpha_, mf_, mfp_, hk_, inst_, mg_, cyq_, senti_, sh_, fund_, ind_, gtja_). The forbidden prefix list (`mkt_`, `index_`, `regime_`, `global_`) is enforced by an assert at training startup.
* **Mask consistency**: the same `valid_mask` is applied at:
  - rollout time when sampling actions
  - PPO update time when computing `evaluate_actions(obs, actions)` log-probs
  - inference time in `predict_values` and deterministic `predict`
  This is critical for PPO's ratio `exp(log_prob_new − log_prob_old)`: if mask differs between sampling and evaluation the ratio gets polluted and policy updates become inconsistent.
* **Critic uses true b2, not b1**: the value pathway concatenates `regime_emb` to each stock's embedding, runs a per-stock value transform, and only THEN performs masked pool. The naive "pool stock embeddings → concat regime → MLP" is mathematically degenerate (regime is the same for all stocks, so it commutes with mean pooling).

## 3. Component-by-component design

### 3.1 `data_loader.py`

Three changes:

#### 3.1.1 `is_suspended_array` default-True fix

```python
# Before:
is_suspended_array = np.zeros((n_dates, n_stocks), dtype=np.bool_)
# default: every (t,j) starts as NOT suspended → pre-IPO rows wrongly look tradeable
for row in df.iter_rows(named=True):
    ...
    is_suspended_array[t, j] = (vol_v is None) or (vol_v == 0)

# After:
is_suspended_array = np.ones((n_dates, n_stocks), dtype=np.bool_)
# default: every (t,j) starts as suspended (untradeable); only update if a parquet row exists
for row in df.iter_rows(named=True):
    ...
    is_suspended_array[t, j] = (vol_v is None) or (vol_v == 0)
```

Effect: every (t, j) without a parquet row stays `is_suspended=True`. Pre-IPO and delisted dates are correctly marked. The env's `valid_mask = ~is_st & ~is_suspended & days_since_ipo>=60` is now correct without any encoder-side workaround.

#### 3.1.2 Regime feature computation

A new function `_compute_regime_features(close, pct_change, valid_mask, factor_array_meta) -> np.ndarray (n_dates, 8)`. Schema:

| index | name | formula |
|---|---|---|
| 0 | `regime_breadth_d` | `mean_i((pct_chg[t, i] > 0) & valid_mask[t, i])` over valid stocks |
| 1 | `regime_breadth_20d` | rolling mean of `regime_breadth_d` over 20 days |
| 2 | `regime_xs_disp_d` | `std_i(pct_chg[t, i])` over valid stocks |
| 3 | `regime_xs_disp_20d` | rolling mean of `regime_xs_disp_d` over 20 days |
| 4 | `regime_idx_ret_20d` | `prod_{k=0..19}(1 + idx_ret_d[t-k]) - 1`, `idx_ret_d = mean_i(pct_chg[t, valid])` |
| 5 | `regime_idx_ret_60d` | `prod_{k=0..59}(1 + idx_ret_d[t-k]) - 1` |
| 6 | `regime_idx_vol_20d` | `std(idx_ret_d, 20d window) * sqrt(252)` |
| 7 | `regime_extreme_imbalance_norm` | `(count(pct_chg ≥ 0.099) - count(pct_chg ≤ -0.099)) / valid_count` |

Implementation notes:

* `idx_ret_d` is the equal-weight mean of valid stocks' daily pct_chg, NOT a market-cap-weighted index (no shares_out in panel). Compounded with `prod(1+r)−1`, not summed/averaged.
* Rolling windows: for early dates (t < window_size), forward-fill from first available value rather than NaN. Validated by unit test.
* `pct_chg` is decimal (verified Phase 19); thresholds ±0.099 are correct for main-board non-ST universe (created by the existing universe filter).
* Naming uses `regime_extreme_imbalance_norm` (not `limit_imbalance_norm`) because the parquet has no native limit-up/down flag; we infer from `|pct_chg| ≥ 0.099`. This name is accurate to the implementation.
* Output is float32, shape `(n_dates, 8)`.

#### 3.1.3 `FactorPanel` extended fields

```python
class FactorPanel(NamedTuple):
    # existing
    factor_array: np.ndarray          # NOW: only per-stock factors. (n_dates, n_stocks, F_stock)
    return_array: np.ndarray
    pct_change_array: np.ndarray
    is_st_array: np.ndarray
    is_suspended_array: np.ndarray
    days_since_ipo_array: np.ndarray
    dates: list[datetime.date]
    stock_codes: list[str]
    factor_names: list[str]            # NOW: only per-stock factor names

    # NEW in Phase 21
    regime_array: np.ndarray          # (n_dates, R) where R = 8
    regime_names: list[str]           # length R, the 8 regime feature names
```

Migration: existing `factor_names` field semantics narrow to "per-stock factor names". Old code reading `factor_array.shape[2]` and assuming it includes regime/mkt cols must be reviewed.

#### 3.1.4 Schema lock at discovery

`discover_factor_columns` now returns ONLY columns whose prefix is in the allowlist:

```python
STOCK_FACTOR_PREFIXES = (
    "alpha_", "mf_", "mfp_", "hm_", "hk_", "inst_",
    "mg_", "cyq_", "senti_", "sh_", "fund_", "ind_", "gtja_",
)
FORBIDDEN_PREFIXES = ("mkt_", "index_", "regime_", "global_")
```

`mkt_*` is removed from the allowlist (it was in V1). The data_loader will silently skip `mkt_*` cols; if a user passes `--drop-factor-prefix mkt_` it now matches zero columns. The schema lock at training startup enforces this:

```python
for col in panel.factor_names:  # per-stock only after this change
    assert not col.startswith(FORBIDDEN_PREFIXES), (
        f"V2 schema lock violated: stock encoder cannot accept {col!r}"
    )
```

`mkt_*` columns are always available to the regime feature computation (they remain in the parquet) but are explicitly NOT used; we compute regime features from `pct_chg` only. This avoids the V1 `mkt_congestion` overfit pathology.

### 3.2 `gpu_env.py`

`GPUStockPickingEnv` is modified to:

* Take `regime_t: torch.Tensor (n_dates, R)` cuda alongside the existing `panel_t / returns_t / valid_mask`.
* Set `observation_space = gymnasium.spaces.Dict({"stock": Box(...), "regime": Box(...), "valid_mask": Box(0, 1, ...)})`.
* `_obs_for_sb3()` returns a `dict[str, np.ndarray]` matching the Dict space:

```python
def _obs_for_sb3(self):
    t = self.last_obs_t.detach().cpu().numpy()
    return {
        "stock": self.panel[t].detach().cpu().numpy(),
        "regime": self.regime[t].detach().cpu().numpy(),
        "valid_mask": self.valid_mask[t].detach().cpu().numpy().astype(np.float32),
    }
```

Per-env stride: `t` is per-env so the dict tensors are stacked across envs by SB3's VecEnv. Resulting shapes (after VecEnv stack):

* `obs["stock"]`: (n_envs, S, F_stock)
* `obs["regime"]`: (n_envs, R)
* `obs["valid_mask"]`: (n_envs, S)

The reward and step logic is unchanged: `returns[self.t]` (Phase 16 fix preserved), top-K selection from action.

### 3.3 `feature_extractor.py`

Replace `PerStockExtractor` (V1) with two new classes:

#### 3.3.1 `PerStockEncoderV2`

```python
class PerStockEncoderV2(nn.Module):
    def __init__(self, n_stocks: int, n_factors: int,
                 hidden: tuple[int, ...] = (128, 64), out_dim: int = 32):
        super().__init__()
        self.n_stocks = n_stocks
        self.n_factors = n_factors
        self.out_dim = out_dim
        layers = []
        prev = n_factors
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, out_dim)]
        self.mlp = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, stock_x: torch.Tensor) -> torch.Tensor:
        # stock_x: (B, S, F_stock)
        b, s, f = stock_x.shape
        flat = stock_x.reshape(b * s, f)
        return self.norm(self.mlp(flat).reshape(b, s, self.out_dim))
```

Differences vs V1:
* No regime / mkt input
* No cross-section centering inside the encoder (removed because masked centering is now done at the head layer if needed; V2's plain encoder is sufficient given the regime head)
* No dual pooling here (moved to the critic's value_token_mlp + masked_mean)
* No `unique_date` optimisation in v0; can be added back later if profiling shows it helps

#### 3.3.2 `RegimeEncoder`

```python
class RegimeEncoder(nn.Module):
    def __init__(self, regime_dim: int = 8, hidden: int = 64, out_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(regime_dim),
            nn.Linear(regime_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, regime_x: torch.Tensor) -> torch.Tensor:
        # regime_x: (B, R)
        return self.net(regime_x)  # (B, R')
```

R' = 16 for v0; configurable via train_v2 CLI flag `--regime-encoder-out-dim`.

#### 3.3.3 `masked_mean` utility

```python
def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """x: (B, S, H); mask: (B, S) where 1 = valid. Returns (B, H)."""
    m = mask.to(dtype=x.dtype).unsqueeze(-1)  # (B, S, 1)
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(eps)
```

### 3.4 `policy.py`

`PerStockEncoderPolicyV2` is a fresh `ActorCriticPolicy` subclass. It is the ONLY policy class we need; the V1 class is deleted.

```python
class PerStockEncoderPolicyV2(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule,
                 encoder_hidden=(128, 64), encoder_out_dim=32,
                 regime_encoder_hidden=64, regime_encoder_out_dim=16,
                 critic_token_hidden=64,
                 **kwargs):
        # We override the parent's mlp_extractor / action_net / value_net machinery.
        # Pass _init_setup_model=False to skip the parent's auto-build, then build ours.
        super().__init__(observation_space, action_space, lr_schedule,
                         _init_setup_model=False, **kwargs)
        # parent did partial init; we build the rest:
        n_stocks, F_stock = observation_space["stock"].shape
        R = observation_space["regime"].shape[0]
        self.stock_encoder = PerStockEncoderV2(n_stocks, F_stock,
                                               encoder_hidden, encoder_out_dim)
        self.regime_encoder = RegimeEncoder(R, regime_encoder_hidden,
                                            regime_encoder_out_dim)
        head_in_dim = encoder_out_dim + regime_encoder_out_dim
        self.actor_head = nn.Linear(head_in_dim, 1)
        self.value_token_mlp = nn.Sequential(
            nn.Linear(head_in_dim, critic_token_hidden),
            nn.ReLU(),
            nn.Linear(critic_token_hidden, critic_token_hidden),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(critic_token_hidden, 1)
        # Manually build optimizer pointing at all the heads we just added
        # (Phase 10's optimizer-orphan bug is preserved as a known pitfall here)
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1.0), **self.optimizer_kwargs
        )

    def _shared_forward(self, obs):
        stock_x = obs["stock"]                                 # (B, S, F_s)
        regime_x = obs["regime"]                               # (B, R)
        valid_mask = obs["valid_mask"].bool()                  # (B, S)
        b, s, _ = stock_x.shape
        stock_emb = self.stock_encoder(stock_x)                # (B, S, D)
        regime_emb = self.regime_encoder(regime_x)             # (B, R')
        regime_b = regime_emb.unsqueeze(1).expand(-1, s, -1)   # (B, S, R')
        head_in = torch.cat([stock_emb, regime_b], dim=-1)     # (B, S, D+R')
        return head_in, valid_mask

    def _logits(self, head_in, valid_mask):
        logits = self.actor_head(head_in).squeeze(-1)           # (B, S)
        logits = logits.masked_fill(~valid_mask, -1e9)
        return logits

    def _value(self, head_in, valid_mask):
        tokens = self.value_token_mlp(head_in)                 # (B, S, H)
        pooled = masked_mean(tokens, valid_mask.float())       # (B, H)
        return self.value_head(pooled).squeeze(-1)             # (B,)

    # Action space is preserved from V1: Box(0,1,(S,)) per-stock scores; the env
    # applies top-K post-sample. Distribution is therefore a per-stock Normal,
    # NOT a Categorical over S, matching the existing GPU env contract and
    # IndexOnlyRolloutBuffer expectations.
    #
    # `self.log_std` is created by ActorCriticPolicy's parent init when
    # action_space is Box; we reuse it. (If the parent skipped that because of
    # `_init_setup_model=False`, build it manually:
    #     self.log_std = nn.Parameter(torch.zeros(action_space.shape[0]))
    # )

    def _make_distribution(self, head_in, valid_mask):
        loc = self._logits(head_in, valid_mask)        # (B, S), invalid → -1e9
        scale = self.log_std.exp().expand_as(loc)      # broadcast scalar/per-stock std
        return torch.distributions.Normal(loc, scale)

    def forward(self, obs, deterministic=False):
        head_in, valid_mask = self._shared_forward(obs)
        if not valid_mask.any(dim=1).all():
            raise RuntimeError("empty valid_mask in observation; every sample needs ≥1 valid stock")
        dist = self._make_distribution(head_in, valid_mask)
        actions = dist.mean if deterministic else dist.rsample()
        log_prob = dist.log_prob(actions).sum(dim=-1)   # joint log-prob over S
        values = self._value(head_in, valid_mask)
        return actions, values, log_prob

    def evaluate_actions(self, obs, actions):
        head_in, valid_mask = self._shared_forward(obs)
        if not valid_mask.any(dim=1).all():
            raise RuntimeError("empty valid_mask in evaluate_actions")
        dist = self._make_distribution(head_in, valid_mask)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        values = self._value(head_in, valid_mask)
        return values, log_prob, entropy

    def get_distribution(self, obs):
        head_in, valid_mask = self._shared_forward(obs)
        return self._make_distribution(head_in, valid_mask)

    def predict_values(self, obs):
        head_in, valid_mask = self._shared_forward(obs)
        return self._value(head_in, valid_mask)
```

**Action space stays `Box(0, 1, (n_stocks,))`** — same as V1. The env's existing `step_wait` already does top-K post-sample. We do NOT switch to `Discrete(n_stocks)`: that would force the agent to sample one stock per env-step, breaking top-K rebalancing semantics, and would require a new env contract.

The hard mask still works correctly with `Normal`: invalid stocks get `loc = -1e9`, so even at `scale = 1.0` the sampled score is on the order of `-1e9 ± O(1)`. After top-K argsort the env will never pick them — equivalent to a `-inf` Categorical logit but compatible with the existing Box action plumbing.

`log_std` initialisation: parent `ActorCriticPolicy.__init__` creates `self.log_std = nn.Parameter(torch.zeros(...))` when `action_space` is `Box`. With `_init_setup_model=False` we MUST verify the parameter exists before our `__init__` returns; if not, create it explicitly with shape `(n_stocks,)` and zeros. Unit test pins this.

### 3.5 `index_rollout_buffer.py`

Replace `IndexOnlyRolloutBuffer` (subclass of GPURolloutBuffer / RolloutBuffer) with `IndexOnlyDictRolloutBuffer extends DictRolloutBuffer`. Use SB3's `DictRolloutBuffer` for the dict-storage plumbing; keep our index-lazy-gather optimisation.

```python
class IndexOnlyDictRolloutBuffer(DictRolloutBuffer):
    """Dict observation rollout buffer that stores t-indices instead of full
    obs tensors, materialising obs lazily via env-bound provider closures.
    
    DictRolloutBuffer stores per-key arrays of shape (buffer_size, n_envs, *obs_key.shape).
    For our (n_envs, S, F_stock) panel that's still (1024, 16, 3014, 343) ≈ 64 GiB host
    RAM. Instead we override the storage to keep just (1024, 16) longs per key, plus
    closures the encoder calls at sample time."""
    
    def __init__(self, *args, obs_provider=None, mask_provider=None,
                 regime_provider=None, t_provider=None, **kwargs):
        super().__init__(*args, **kwargs)
        # override storage: just t-indices
        self._t_storage = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)
        self.obs_provider = obs_provider          # callable(t) -> (S, F_stock) cuda tensor
        self.regime_provider = regime_provider    # callable(t) -> (R,) cuda tensor
        self.mask_provider = mask_provider        # callable(t) -> (S,) cuda tensor
        self.t_provider = t_provider              # callable() -> (n_envs,) current t
        # don't allocate the per-key obs storage that DictRolloutBuffer would
        for k in list(self.observations.keys()):
            self.observations[k] = None  # mark as not-stored
    
    def attach_providers(self, obs_provider, regime_provider, mask_provider, t_provider):
        self.obs_provider = obs_provider
        self.regime_provider = regime_provider
        self.mask_provider = mask_provider
        self.t_provider = t_provider
    
    def add(self, obs, action, reward, episode_start, value, log_prob):
        # obs is a dict; we don't store it. We DO store the current t, which can
        # be retrieved from env.last_obs_t at this point.
        self._t_storage[self.pos] = self.t_provider().detach().cpu().numpy()
        # call parent for action/reward/value/log_prob/episode_start handling
        # but bypass its dict obs storage by passing dummy dicts
        dummy = {k: None for k in self.observations.keys()}
        super().add(dummy, action, reward, episode_start, value, log_prob)
    
    def _get_samples(self, batch_inds, env_indices):
        # rebuild obs dict for the requested batch by calling providers
        # batch_inds, env_indices: arrays of len batch_size
        ts = self._t_storage[batch_inds, env_indices]  # (B,)
        stock_obs = torch.stack([self.obs_provider(int(t)) for t in ts], dim=0)
        regime_obs = torch.stack([self.regime_provider(int(t)) for t in ts], dim=0)
        mask_obs = torch.stack([self.mask_provider(int(t)) for t in ts], dim=0)
        observations = {
            "stock": stock_obs,
            "regime": regime_obs,
            "valid_mask": mask_obs,
        }
        # everything else stays the same
        return DictRolloutBufferSamples(
            observations=observations,
            actions=...,
            old_values=...,
            old_log_prob=...,
            advantages=...,
            returns=...,
        )
```

Memory: per-key Dict storage skipped, just (1024, 16) longs ≈ 130 KiB. Same VRAM saving as V1.

### 3.6 `train_v2.py`

Three new things plus existing logic:

1. **Schema lock assert** at startup, before model construction:

```python
FORBIDDEN_PREFIXES = ("mkt_", "index_", "regime_", "global_")
for col in panel.factor_names:
    assert not col.startswith(FORBIDDEN_PREFIXES), (
        f"Phase 21 schema lock: stock encoder cannot accept column {col!r}. "
        f"Either remove it from the allowlist or move it to regime_features."
    )
```

2. **Regime tensor on cuda** alongside panel_t:

```python
panel_t = torch.from_numpy(panel.factor_array).to("cuda")
regime_t = torch.from_numpy(panel.regime_array).to("cuda")  # (n_dates, R)
returns_t = ...
valid_mask_t = ~torch.from_numpy(panel.is_st_array).to("cuda") & \
               ~torch.from_numpy(panel.is_suspended_array).to("cuda") & \
               (torch.from_numpy(panel.days_since_ipo_array).to("cuda") >= 60)

env = GPUStockPickingEnv(panel_t, regime_t, returns_t, valid_mask_t, ...)
```

3. **New CLI flags**:

```
--regime-encoder-out-dim INT     default 16   (R')
--critic-token-hidden INT        default 64   (per-stock value MLP hidden dim)
```

4. **Metadata extension**:

```json
{
    ...
    "stock_factor_names": [...],   // per-stock cols, len F_stock
    "regime_factor_names": [...],  // length 8
    "regime_encoder_out_dim": 16,
    "critic_token_hidden": 64,
    "policy_class": "PerStockEncoderPolicyV2",
    ...
}
```

OOS eval scripts will read these to reconstruct the right input shapes.

### 3.7 Eval scripts

`scripts/_eval_all_checkpoints.py`, `scripts/_ensemble_eval.py`, `scripts/eval_factor_importance.py`, `scripts/eval_backtest.py` need updating:

* Read `stock_factor_names` / `regime_factor_names` from metadata, not just `factor_names`.
* Load panel using `factor_names=stock_factor_names`; ALSO call `_compute_regime_features` to get the regime array.
* Build Dict obs at inference: `{"stock": panel.factor_array[t], "regime": panel.regime_array[t], "valid_mask": valid[t]}`.
* `policy.predict / get_distribution` consumes the dict obs.

Existing legacy metadata (Phase 16-19) without `stock_factor_names` triggers a clear error: "this checkpoint was trained on V1 architecture; cannot evaluate with V2 codebase". No backwards-compat shim.

## 4. Data flow (one mini-batch)

```
1. Parquet on disk
   ↓
2. data_loader.load_panel:
   ├── filter universe / dates
   ├── discover stock factor cols (allowlist; FORBIDDEN_PREFIXES rejected)
   ├── _df_to_panel:
   │   ├── allocate factor_array (T, S, F_stock), default 0
   │   ├── allocate is_suspended_array (T, S), default TRUE
   │   ├── populate from df rows
   │   └── compute return_array via _safe_log_return
   ├── _compute_regime_features → regime_array (T, 8)
   └── return FactorPanel (factor_array, regime_array, is_st, is_suspended, ...)
   ↓
3. train_v2:
   ├── schema lock assert on panel.factor_names
   ├── panel_t / regime_t / valid_mask_t / returns_t to cuda
   ├── env = GPUStockPickingEnv with regime_t alongside panel_t
   ├── env.observation_space = Dict{"stock", "regime", "valid_mask"}
   └── PPO(policy=PerStockEncoderPolicyV2, env=env, rollout_buffer_class=IndexOnlyDictRolloutBuffer)
   ↓
4. Rollout collection:
   ├── env.reset → obs dict
   ├── for each step:
   │   ├── policy.forward(obs) → (actions, values, log_probs)
   │   ├── obs["valid_mask"] enforces -1e9 logits for invalid stocks
   │   ├── env.step(actions) → top-K reward
   │   └── buffer.add(obs_t_indices_only, actions, reward, ...)
   ↓
5. PPO update:
   ├── buffer.get(batch_size=1024) → DictRolloutBufferSamples with lazily gathered obs dict
   ├── policy.evaluate_actions(obs_dict, actions_old) → (values, log_probs, entropies)
   │   └── same masking logic as forward → ratio is consistent
   ├── compute clip-loss / value-loss / entropy-loss
   └── optimizer.step()
   ↓
6. Checkpoint:
   ├── ppo_*_steps.zip (full SB3 state including PerStockEncoderPolicyV2 weights)
   └── metadata.json with stock_factor_names / regime_factor_names / class name
```

## 5. Error handling

| location | check | action on failure |
|---|---|---|
| data_loader, after `_df_to_panel` | `is_suspended_array.dtype == bool_` and default-True initialised | unit test catches; production: not applicable (always set correctly) |
| data_loader, after `_compute_regime_features` | `regime_array.shape == (n_dates, 8)`, no NaN, no inf | raise ValueError with offending column name |
| train_v2, startup | `FORBIDDEN_PREFIXES` not present in `panel.factor_names` | assert raises early before any GPU allocation |
| policy.forward / evaluate_actions | `valid_mask.any(dim=1).all()` | raise RuntimeError("empty valid_mask: every sample must have ≥1 valid stock") |
| eval scripts | metadata has `stock_factor_names` and `regime_factor_names` | error message: "this checkpoint was trained on V1; not loadable with V2 codebase" |
| masked_mean | `mask.sum(dim=1) > 0` per row | clamp denominator to eps; combined with valid_mask assert above this should not trigger |

No silent fallbacks. Phase 21 fails loud.

## 6. Testing strategy

### 6.1 Unit tests (must pass before any sanity train)

Located in `tests/test_phase21_v2.py`:

| test | what it pins |
|---|---|
| `test_data_loader_is_suspended_default_true` | empty panel rows → `is_suspended[t,j] = True`, NOT False |
| `test_data_loader_regime_features_shape_and_finite` | regime_array is (n_dates, 8), all finite |
| `test_data_loader_regime_features_compounded_returns` | `idx_ret_20d` matches manual `prod(1+r)−1` on hand-computed series |
| `test_data_loader_regime_extreme_imbalance` | hand-built panel with 5 limit-up + 3 limit-down + 92 normal → `(5−3)/100 = 0.02` |
| `test_data_loader_schema_lock_rejects_mkt` | discover_factor_columns on a df with `mkt_a / alpha_a / regime_a` returns ONLY `alpha_a`; runtime assert in train_v2 catches `mkt_*` in stock factor list |
| `test_perstock_encoder_v2_shape` | input (B=4, S=8, F_stock=10) → output (B=4, S=8, D=32) |
| `test_regime_encoder_shape_and_layer_norm_active` | input (B=4, R=8) → output (B=4, R'=16); LayerNorm makes std≈1 across last dim |
| `test_masked_mean_correctness` | hand-built (x, mask) → expected mean over valid; full-zero mask returns 0 (eps clamp) |
| `test_masked_mean_grad_flows` | autograd through masked_mean works |
| `test_policy_v2_forward_evaluate_consistency` | given (obs, action), `forward` and `evaluate_actions` produce identical log_prob |
| `test_policy_v2_actor_logits_invalid_neg_inf` | invalid stocks get logit ≤ −1e8; `softmax` → ≤ 1e-30 |
| `test_policy_v2_critic_b2_uses_per_stock_value_mlp` | swapping order (pool then concat) produces a DIFFERENT value than (concat then per-stock value mlp then pool) — proves the implementation is b2 not b1 |
| `test_policy_v2_empty_mask_raises` | obs with all-zero valid_mask → RuntimeError |
| `test_dict_rollout_buffer_roundtrip` | add(obs_t_index) → get → reconstructed obs dict equals env's obs dict for that t |
| `test_train_v2_schema_lock_aborts_on_mkt_in_stock` | manually inject mkt_ col → assertion error with helpful message |

### 6.2 Sanity train (Phase 21A)

Single-seed validation that V2 doesn't regress vs V1 baselines:

```bash
.venv/Scripts/python.exe scripts/train_v2.py \
  --total-timesteps 300000 \
  --data-path data/factor_panel_combined_short_2023_2026.parquet \
  --start-date 2023-01-03 --end-date 2025-06-30 \
  --universe-filter main_board_non_st \
  --n-envs 16 --episode-length 240 \
  --batch-size 1024 --n-steps 1024 --n-epochs 10 \
  --learning-rate 1e-4 --target-kl 0.30 --max-grad-norm 0.5 \
  --rollout-buffer index --tf32 --matmul-precision high \
  --unique-date-encoding \
  --checkpoint-freq 25000 \
  --forward-period 10 --top-k 30 \
  --seed 42 \
  --regime-encoder-out-dim 16 \
  --critic-token-hidden 64 \
  --out-dir runs/phase21_21a_v2_drop_mkt_seed42
```

(`--drop-factor-prefix mkt_` is no longer needed because the schema lock + allowlist already excludes mkt_.)

Comparison target: Phase 16a `vs_random_p50_adjusted = +0.428`. Pass criterion: V2 ≥ +0.30 (small loss tolerated as architectural overhead; if V2 wins it's a bonus). Loss > +0.10 below baseline triggers an investigation before declaring V2 production-ready.

### 6.3 Three architectural sanity checks (after sanity train)

**Check 1 — actor regime ablation.**
After Phase 21A finishes, load the best checkpoint and rescore on OOS with the regime input replaced by:

* `regime_emb_zero` (zero vector): expected to drop adj_S meaningfully if regime is doing real work
* `regime_emb_batch_mean` (mean over OOS dates): expected to be slightly worse than zero (mean is the most meaningless single value)
* `regime_emb_shuffled` (shuffle date axis of regime_array before scoring): expected to be substantially worse than original

If all three give nearly the same adj_S as the original → regime is not being used by the actor.

**Check 2 — leakage check.**
With `regime_emb` REPLACED by zero, score on OOS, then group OOS dates into buckets by (a) breadth_d quantile (bull / bear days), (b) idx_vol_20d quantile (high / low vol days). Compute per-bucket actor logit moments. If logits still differ strongly across buckets WITHOUT the regime input, some regime info is leaking through `obs["stock"]`. Acceptable: stock encoder can naturally produce different statistics on different dates because per-stock factors themselves vary; a small bucket-difference is normal. RED FLAG: bucket-difference not meaningfully smaller than with regime input present.

**Check 3 — b1 vs true b2.**
Compile a b1 critic variant: `pooled = masked_mean(stock_emb, mask); pooled_with_regime = concat([pooled, regime_emb]); value = b1_value_mlp(pooled_with_regime)`. Train identical config seed=42 with b1 critic for 100k steps. Compare:

* value loss trajectory
* OOS adj_S
* If b2 has materially LOWER value loss AND HIGHER OOS Sharpe → the per-stock value transform is doing real work
* If b2 has materially LOWER value loss BUT NO OOS Sharpe difference → critic is overfitting; consider lowering value loss coefficient
* If both are equivalent → regime doesn't have per-stock value impact in this dataset; could justify falling back to b1 for simplicity

## 7. Implementation order (high-level)

1. **data_loader** (is_suspended fix + regime computation + schema lock at discovery + `regime_array` field) + unit tests
2. **gpu_env** (Dict observation space + regime_t plumbing + `_obs_for_sb3` returning dict) + unit tests
3. **feature_extractor** (PerStockEncoderV2, RegimeEncoder, masked_mean) + unit tests
4. **policy** (PerStockEncoderPolicyV2 with custom forward/evaluate_actions/get_distribution/predict_values + manual optimizer build) + unit tests
5. **rollout_buffer** (IndexOnlyDictRolloutBuffer extending DictRolloutBuffer) + unit tests
6. **train_v2** (schema lock assert + Dict obs + regime CLI flags + extended metadata) + smoke run on synthetic panel
7. **eval scripts** updated to read split metadata names and reconstruct Dict obs
8. **Phase 21A sanity train** (drop_mkt-equivalent regime, seed=42, 300k, full corrected eval)
9. **Three architectural sanity checks** (regime ablation, leakage, b1 vs b2)
10. **HANDOFF + commit + OSS**

## 8. Out of scope (deferred)

* **FiLM-style modulation** of stock_emb by regime_ctx (option C from design discussion). v0 is concat + nonlinearity. FiLM goes to a follow-up Phase 22 if v0 proves too weak.
* **Path B Ubuntu-side regime features** (VIX-equivalent, fund flows). v0 uses pct_chg-derived regime only. Ubuntu-side enrichment is Phase 22+.
* **Volume-z regime features** (`regime_rel_amount_z`). v1 ablation candidate, not v0.
* **Multi-seed Phase 21B/C/D** (extending the seed sweep on V2). Run only after Phase 21A passes the regression bar; Phase 21 spec only commits to ≥1 sanity train.
* **Cross-version ensembling** (V1 zips + V2 zips). Hard switch means V1 zips are not loadable; ensembling is V2-only after this. Re-running the Phase 18 6-seed sweep on V2 (to get the new ensemble baseline) is Phase 22+ work.
* **Dynamic universe step 2 — variable-size obs per day.** v0 keeps fixed S = n_stocks but uses valid_mask correctly throughout. True per-day variable obs (where un-listed stocks are not even allocated rows) is a much bigger refactor; deferred.

## 9. Migration checklist (V1 → V2 hard switch)

* `models/production/phase16_*` `phase17_*` `phase18_*` `phase20_*` zips remain on disk and OSS as forensic artifacts but are NO LONGER LOADABLE under V2 codebase. **Do not delete them**: they are the only record of pre-V2 weights and several reports cite their sha256 digests.
* `models/production/` directory layout is unchanged. V2 zips land under `models/phase21/` (new) initially; promotion to `models/production/` happens only after fresh-holdout gate clears.
* Phase 16-19 OOS sweep / ensemble / execution-sim JSON reports remain valid as historical baselines (already archived to OSS) but cannot be regenerated under V2.
* Phase 19 fresh-holdout INSUFFICIENT verdict applies equally to V2: don't promote V2 to production until ≥40 fresh post-2026-04-24 dates accumulate.
* Phase 18 ensemble candidate (rankmean6) is dropped from active candidate list; V2 will need its own multi-seed sweep to produce a new ensemble baseline.

## 10. Success criteria

The Phase 21 spec is "done" when:

* All unit tests in §6.1 pass.
* Phase 21A sanity train completes, OOS `vs_random_p50_adjusted ≥ +0.30` (within 0.13 of Phase 16a's +0.428; if it beats +0.428 that's a bonus).
* Three architectural sanity checks executed and reported (regardless of outcome — even a "regime is not being used" finding is valuable).
* Code is committed and pushed to main.
* HANDOFF written + uploaded to OSS.

The spec is NOT yet promoted to production after Phase 21 ends. Phase 22 picks up multi-seed validation, fresh holdout collection, and execution-constrained eval.
