# Architecture

> AurumQ-RL system architecture, data flow, and key design decisions.

## 1. System overview

AurumQ-RL is composed of four loosely-coupled stages: **data preparation**,
**environment construction**, **RL training**, and **production inference**.
Training and inference are *intentionally* split: training is GPU-heavy and
runs on dedicated machines; inference is CPU-only via ONNX Runtime so it can
be embedded in scheduled cron jobs, web services, or batch back-tests.

```
+----------------------------------------------------------------+
|                      Data preparation                          |
|                                                                |
|   (your factor pipeline)        scripts/generate_synthetic.py  |
|              \                                /                |
|               v                              v                 |
|             data/factor_panel.parquet   data/synthetic_demo.   |
|                                         parquet                |
+--------------------------+-------------------------------------+
                           |
                           v
+----------------------------------------------------------------+
|        FactorPanelLoader  (src/aurumq_rl/data_loader.py)       |
|  - prefix-based factor discovery                               |
|  - universe filter (5 modes)                                   |
|  - cross-section z-score                                       |
|  - 3D numpy panel (n_dates, n_stocks, n_factors)               |
+--------------------------+-------------------------------------+
                           |
                           v
+----------------------------------------------------------------+
|  Gym environments  (src/aurumq_rl/env.py, portfolio_weight_env)|
|  - StockPickingEnv  : top-k discrete selection                 |
|  - PortfolioWeightEnv: continuous weights, projected to simplex|
|  - A-share constraints (T+1, price limits, ST, suspension, IPO)|
+--------------------------+-------------------------------------+
                           |
                           v
+----------------------------------------------------------------+
|            Stable-Baselines3 trainer  (scripts/train.py)       |
|  - PPO / A2C / SAC                                             |
|  - SubprocVecEnv parallelism                                   |
|  - Callbacks: checkpoint, wandb, JSONL metrics                 |
|  - Output: policy.zip + checkpoints + tb_logs                  |
+--------------------------+-------------------------------------+
                           |
                           v  (one-time export)
+----------------------------------------------------------------+
|        ONNX export  (src/aurumq_rl/onnx_export.py)             |
|  - torch.onnx.export(policy, opset=17)                         |
|  - metadata.json (algorithm, obs_shape, factor_count, ...)     |
|  - Output: policy.onnx + metadata.json                         |
+--------------------------+-------------------------------------+
                           |
                           v
+----------------------------------------------------------------+
|       RlAgentInference  (src/aurumq_rl/inference.py)           |
|  - onnxruntime CPUExecutionProvider                            |
|  - session reuse, batched predict                              |
|  - ~150ms / 5000 stocks                                        |
+----------------------------------------------------------------+
```

## 2. Component breakdown

### 2.1 `data_loader.py`

The single source of truth for the input data contract.

| Component | Responsibility |
|---|---|
| `FACTOR_COL_PREFIXES` | The fixed list of recognised prefixes (`alpha_`, `mf_`, ...). |
| `discover_factor_columns()` | Sorted prefix matching with optional `n_factors` truncation. |
| `filter_universe()` | Five filter modes (`ALL_A`, `MAIN_BOARD_NON_ST`, `HS300`, `ZZ500`, `ZZ1000`). |
| `FactorPanelLoader.load_panel()` | Parquet → 3D numpy panel + auxiliary arrays. |
| `FactorPanelLoader.build_synthetic()` | Pure-Python synthetic panel for smoke tests. |

Key invariant: **the loader never raises because of missing factor groups**.
A user who exports only `alpha_*` columns gets an `n_factors` value equal to
however many alpha columns existed; the model just sees those positions.

### 2.2 Environments

Two environments are provided, sharing helper functions:

* **`StockPickingEnv`** — discrete top-k selection.
  Action ∈ `Box(0,1,(n_stocks,))` is interpreted as priority scores; the env
  applies the industry-cap constraint and equal-weights the chosen `top_k`.
* **`PortfolioWeightEnv`** — continuous weights.
  Action ∈ `Box(0,1,(n_stocks,))` is projected onto the simplex with per-stock
  and per-industry caps via bisection. Reward type is configurable
  (`return` / `sharpe` / `sortino` / `mean_variance`).

Both envs share `_apply_trading_mask`, which zeroes out untradeable stocks
(price-limit hit, ST, suspension, new-stock protection). Untradeable stocks
contribute 0 reward — they don't crash the env, they don't get credit.

### 2.3 Reward library (`reward_functions.py`)

Pure numpy implementations, **no env coupling**, so they can be unit-tested
and swapped freely:

* `simple_return_reward` — instantaneous portfolio return − cost − turnover.
* `sharpe_reward` — rolling Sharpe (annualised by sqrt(252)).
* `sortino_reward` — rolling Sortino (downside-deviation based).
* `mean_variance_reward` — Markowitz `μ − λ·σ²`.

### 2.4 ONNX export (`onnx_export.py`)

Loads an SB3 model checkpoint, calls `torch.onnx.export` on the policy
network with `opset_version=17`, and writes a JSON sidecar (`metadata.json`)
that describes the model: algorithm, training timesteps, observation shape,
universe filter, factor count, git SHA at training time. The metadata schema
is validated by `RlAgentMetadata` in `inference.py`.

### 2.5 Inference (`inference.py`)

A single `onnxruntime.InferenceSession` is created per agent and reused.
`predict()` accepts both single observations (1D) and batched (2D);
`batch_predict()` stacks a list of observations for higher throughput. The
inference engine has **zero PyTorch dependency** — it only requires
`onnxruntime`, `numpy`, and `pydantic` for metadata validation.

### 2.6 Metrics (`metrics.py`)

Training callbacks emit `TrainingMetrics` records (algorithm, timestep,
loss, fps, reward) to a JSONL file via `append_metrics`. `load_metrics`
reads them back, sorted by timestep, skipping malformed lines. This gives
us a structured training log that survives subprocess crashes and is
easy to plot post-hoc.

## 3. Data flow walk-through

A typical end-to-end run:

1. **Build factors** in your own pipeline (out of scope for this project).
   Output a Parquet with `ts_code`, `trade_date`, `close`, `pct_chg`, `vol`,
   plus any combination of factor columns matching the prefix convention.
2. **Smoke test** with `python scripts/train.py --smoke-test`. No PyTorch
   required. Verifies the data contract and the JSONL metrics path.
3. **Train** on a GPU machine: `python scripts/train.py --algorithm PPO ...`
   reads the Parquet via `FactorPanelLoader`, constructs `n_envs` parallel
   workers, runs `model.learn(total_timesteps=…)`, checkpoints at
   intervals, and exports `policy.onnx` at the end.
4. **Deploy** the directory `models/ppo_v1/` containing `policy.onnx` +
   `metadata.json` to the inference host. No GPU needed there.
5. **Predict daily**: load via `RlAgentInference("models/ppo_v1/")`, pass
   the latest cross-section observation, get back per-stock priority scores,
   apply your top-k or weight projection, and execute via your broker
   integration (out of scope for this project).

## 4. Why split training and inference?

**GPU resources are scarce and expensive.** Forcing every consumer of the
model to install PyTorch + CUDA defeats the purpose of a deployable
RL policy. By exporting to ONNX:

* Inference hosts only need `onnxruntime` (~50 MB) and `numpy`.
* Latency for 5000 stocks × ~100 factor dims is roughly **150 ms on a
  recent x86 CPU** — fast enough for daily/weekly cron runs.
* The same `policy.onnx` file is portable across Linux, macOS, Windows,
  and ARM (e.g. Apple Silicon, Raspberry Pi).

**Training on the inference host is a hard fail.** Loading PyTorch into a
14 GB ECS instance was the original motivation for the split — it OOM-kills
the process within seconds. The CLAUDE.md document for this project is
explicit: training must run on a GPU machine, not the inference host.

## 5. Key design decisions

### 5.1 Prefix-based factor discovery

**Decision**: factors are recognised by *column prefix*, not by an explicit
list passed at training time.

**Rationale**:
* Users add or remove factor groups by changing their export pipeline; the
  RL code does not need to know.
* `n_factors` truncates from the discovered set, so you can train a 32-dim
  model and a 64-dim model from the same Parquet.
* The model sees missing prefixes as zero — no exception.

**Trade-off**: if a user accidentally names a non-factor column `alpha_xxx`
it gets picked up. Mitigation: keep the export script disciplined.

### 5.2 No factor computation in this project

**Decision**: AurumQ-RL is RL only. Factor formulas are explicitly out of
scope.

**Rationale**:
* Factor computation is research-heavy and changes frequently. Coupling it
  to the RL repo would slow both down.
* Different users have different data sources and different alpha factor
  preferences.
* The contract — a Parquet with the documented columns — is small enough
  to be implemented in pandas, polars, DuckDB, or Spark.

### 5.3 Cross-section z-score at load time, not in env

**Decision**: `FactorPanelLoader._df_to_panel` z-scores once during loading;
`StockPickingEnv` consumes the result without further normalisation.

**Rationale**:
* Z-scoring is `O(T·F·N)` and runs once per training run; doing it inside
  `env.step` would multiply that cost by `total_timesteps`.
* Keeps the env focused on a single responsibility (sampling rewards),
  which makes profiling and unit-testing simpler.

### 5.4 Untradeable stocks get zeroed reward, not env error

**Decision**: when a stock hits a price limit, is suspended, ST-flagged, or
within the new-stock protection window, its reward contribution is set to
zero rather than triggering an exception.

**Rationale**:
* In production the policy still has to *pick* something — it cannot
  refuse to trade for the day.
* Zero-reward-and-no-cost mirrors the real outcome of putting an order on
  a frozen stock: nothing happens.
* The model learns to avoid these stocks by observing their flat reward
  history, without us hard-coding a punishment.

### 5.5 ONNX over TorchScript

**Decision**: export to ONNX, not TorchScript or PyTorch's `state_dict`.

**Rationale**:
* Decouples runtime from training framework: SB3 may switch backends, the
  policy network architecture may change, but ONNX remains a stable IR.
* Cross-language: any language with an ONNX runtime (C++, Rust, JS, Java)
  can serve the model without re-implementing the network.
* `metadata.json` carries the obs shape, so the inference layer can
  validate inputs without parsing the model graph.

## 6. Performance targets

| Hot path | Target | Measured baseline (RTX 4070 / i7-13700K) |
|---|---|---|
| `env.step` | < 2 ms | ~1.0 ms with 5000 stocks |
| `load_panel(all_a, 5y)` | < 30 s | ~12 s for 4 M rows |
| `inference.predict(5000)` | < 200 ms | ~150 ms |

A 20% regression on any of these hot paths is treated as a bug, not a
trade-off.

## 7. Out of scope

* Factor computation (use your own pipeline).
* Order routing / broker integration (build your own thin layer on top of
  `RlAgentInference.predict`).
* Live tick-data streaming (this is a daily-frequency framework).
* Risk management beyond the env-level industry cap (we provide signals;
  position-sizing is your problem).
