# Inference

> Running a trained AurumQ-RL agent in production. CPU-only, no PyTorch.

## 1. Runtime requirements

The inference path is intentionally minimal:

```bash
pip install aurumq-rl       # core install only — ~50 MB
```

Required dependencies (already pulled in by core install):
* `onnxruntime` (CPU)
* `numpy`
* `polars` (for loading the Parquet input)
* `pydantic` (metadata validation)

You do **not** need: PyTorch, gymnasium, stable-baselines3, CUDA, or the
training extras.

## 2. Latency benchmarks

Measured on a typical 8-core x86 server:

| Universe size | Factor dim | Single `predict()` | Batched `batch_predict(64)` |
|---|---|---|---|
| 300 stocks | 64 | ~3 ms | ~25 ms |
| 1000 stocks | 64 | ~12 ms | ~80 ms |
| 5000 stocks | 64 | ~150 ms | ~600 ms |
| 5000 stocks | 108 | ~250 ms | ~950 ms |

Throughput scales sub-linearly with batch size — for daily inference
(one cross-section per day), the single-call path is more than fast enough.
Batching matters for back-testing, where you replay 100s of days quickly.

## 3. Loading a model

The minimal example:

```python
from aurumq_rl import RlAgentInference

agent = RlAgentInference("models/ppo_v1/")
print(agent.metadata.algorithm)        # "PPO"
print(agent.metadata.obs_shape)        # (5000 * 64,)
print(agent.metadata.training_timesteps)
```

The constructor:
1. Loads `policy.onnx` into a single ORT `InferenceSession` with
   `CPUExecutionProvider`.
2. Validates `metadata.json` against the `RlAgentMetadata` Pydantic schema.
3. Caches the input/output tensor names so subsequent calls are
   zero-overhead.

The `InferenceSession` is reused for the lifetime of the `RlAgentInference`
instance — do not reconstruct it inside a hot loop.

## 4. Single observation prediction

```python
import numpy as np

# Build observation: factor cross-section flattened, shape == metadata.obs_shape
obs = build_observation_for_today()   # your pipeline; returns np.ndarray
action = agent.predict(obs)

# action is a per-stock priority score in [0, 1]
top_k_indices = np.argsort(action)[::-1][:30]
```

`predict()` accepts:
* 1D arrays matching `metadata.obs_shape` exactly.
* 2D arrays with a leading batch dimension (returns the first batch entry).
* Auto-converts dtype to `float32`.

It raises `ValueError` if the shape doesn't match — *fail fast* is
preferable to silent garbage.

## 5. Batched inference

For replaying many days at once (back-testing, evaluation):

```python
observations = [obs_day_1, obs_day_2, ..., obs_day_N]   # list of np.ndarray
actions = agent.batch_predict(observations)             # list of np.ndarray
```

Internally, the observations are stacked into a single `(N, *obs_shape)`
tensor and fed through one ORT call. This amortises the per-call kernel
launch overhead and is typically 4-8× faster than looping `predict()`.

Empty list returns empty list — no exception.

## 6. metadata.json schema

```json
{
  "algorithm": "PPO",
  "training_timesteps": 1000000,
  "final_reward": 0.0143,
  "obs_shape": [320000],
  "action_shape": [5000],
  "git_sha": "a1b2c3d",
  "onnx_opset": 17,
  "framework": "stable-baselines3",
  "universe": "main_board_non_st",
  "env_type": "stock_picking",
  "reward_type": "return",
  "top_k": 30,
  "factor_count": 64,
  "exported_at": null
}
```

The `RlAgentMetadata` Pydantic model permits extra fields (`extra: allow`),
so future training runs can add new metadata keys without breaking
backwards compatibility.

## 7. Daily prediction workflow

A typical cron-style production setup:

```bash
# crontab: run every weekday at 17:00
0 17 * * 1-5 /usr/bin/python /opt/aurumq-rl/scripts/infer.py \
    --model /models/ppo_v1/ \
    --data /data/factor_panel.parquet \
    --date $(date +\%Y-\%m-\%d) \
    --top-k 30 \
    --out /reports/picks_$(date +\%Y\%m\%d).json
```

Or programmatically:

```python
import datetime
import polars as pl
from aurumq_rl import FactorPanelLoader, RlAgentInference

today = datetime.date.today()
loader = FactorPanelLoader(parquet_path="data/factor_panel.parquet")
panel = loader.load_panel(
    start_date=today - datetime.timedelta(days=30),  # need history for z-scoring
    end_date=today,
    n_factors=64,
)
obs_today = panel.factor_array[-1].reshape(-1)

agent = RlAgentInference("models/ppo_v1/")
action = agent.predict(obs_today)

top_k_idx = action.argsort()[::-1][:30]
top_k_codes = [panel.stock_codes[i] for i in top_k_idx]
print("Today's picks:", top_k_codes)
```

## 8. Performance tips

* **Reuse the agent instance.** Each `RlAgentInference(...)` constructor
  takes 50-200 ms to load the model — never put it inside a request handler.
* **Batch when you can.** If you're back-testing 1000 days, call
  `batch_predict` once, not `predict` 1000 times.
* **Pin numpy and onnxruntime versions.** ORT bumps occasionally affect
  numeric outputs; pin major versions in `requirements.txt` for production.
* **CPU thread count.** Set `OMP_NUM_THREADS` and `MKL_NUM_THREADS` to the
  physical core count of your inference host. ORT uses these underneath.

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `FileNotFoundError: policy.onnx` | Wrong directory passed | Pass the directory containing both `policy.onnx` and `metadata.json`. |
| `ValueError: shape mismatch` | Observation reshape forgot to flatten | Pass `(n_stocks * n_factors,)` flat array, not 2D. |
| Predictions look identical day-to-day | Z-scoring window too long | Z-scoring is per-cross-section in the loader; check that your daily slice is being z-scored together with recent history, not alone. |
| Wildly different scores from training | Different opset version on inference host | Confirm `onnxruntime>=1.17`. |
| `RuntimeError: numpy ABI mismatch` | numpy major version skew | Re-install onnxruntime against your numpy. |
