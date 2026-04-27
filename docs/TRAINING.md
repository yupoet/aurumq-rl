# Training

> How to train an AurumQ-RL agent end-to-end: hardware, smoke testing,
> hyperparameters, parallelism, monitoring, and common failures.

## 1. Hardware recommendations

**Do not run training on a low-resource machine.** PyTorch alone occupies
~3 GB RSS; SB3's PPO with `n_envs=6` and a 5000-stock observation easily
peaks above 10 GB. Use a GPU machine.

### GPU comparison

| GPU | VRAM | Recommended `n_envs` | 1 M PPO steps (~5000 stocks) | Cost (typical) |
|---|---|---|---|---|
| RTX 4070 (12 GB) | 12 GB | 4-6 | ~5-6 hours | ~¥4500 one-time |
| RTX 4080 (16 GB) | 16 GB | 6-8 | ~3-4 hours | ~¥7500 one-time |
| RTX 4090 (24 GB) | 24 GB | 8-12 | ~2-3 hours | ~¥13000 one-time |
| A10 (cloud) | 24 GB | 8-12 | ~2-3 hours | ~¥4-6/hr |
| V100 (cloud) | 16 GB | 6-8 | ~3-4 hours | ~¥3-5/hr |
| A100 (cloud) | 40 GB | 12-16 | ~1-1.5 hours | ~¥15-25/hr |

For experimentation, a rented A10 or V100 hour is cheaper than buying a
consumer GPU and faster than waiting on a 4070. For a stable production
training cadence (weekly retrain), an owned RTX 4070+ amortises within a
few months.

CPU and RAM matter less, but aim for:
* >= 8 cores (so `n_envs=6` parallelism doesn't oversubscribe).
* >= 32 GB system RAM (the panel can be > 10 GB in memory for multi-year
  full-A-share training).
* SSD storage (Parquet IO benefits significantly).

## 2. Smoke test

The smoke test runs entirely on CPU, requires only the core install, and
verifies the IO path + JSONL metrics + ONNX export shape without invoking
PyTorch:

```bash
pip install -e .
python scripts/train.py --smoke-test --out-dir /tmp/aurumq_rl_smoke
cat /tmp/aurumq_rl_smoke/smoke_summary.json
```

Expected output:
* `smoke_summary.json` with `status: "ok"` and a populated `metrics_summary`.
* `training_metrics.jsonl` with 10 records.

If the smoke test fails on your machine, do **not** proceed to GPU
training — fix the environment first.

## 3. Real training

### 3.1 Install training extras

```bash
pip install -e ".[train]"   # adds torch, sb3, gymnasium, onnx, wandb
```

### 3.2 Prepare data

Either use the synthetic demo (for sanity checks):

```bash
python scripts/generate_synthetic.py --out data/synthetic_demo.parquet
```

Or export real factor panels from your data warehouse:

```bash
python scripts/export_factor_panel.py \
    --pg-url postgresql://user:pass@host/db \
    --start 2023-01-01 --end 2025-06-30 \
    --out data/factor_panel.parquet
```

(See `scripts/export_factor_panel.py --help` for all options. The export
script supports SQL templates and prefix-aware column selection.)

### 3.3 Train

A typical PPO run on a 12 GB GPU:

```bash
python scripts/train.py \
    --algorithm PPO \
    --total-timesteps 1000000 \
    --data-path data/factor_panel.parquet \
    --start-date 2023-01-01 \
    --end-date 2025-06-30 \
    --universe-filter main_board_non_st \
    --n-factors 64 \
    --top-k 30 \
    --forward-period 10 \
    --cost-bps 30 \
    --learning-rate 3e-4 \
    --n-envs 6 \
    --out-dir models/ppo_v1
```

Outputs in `models/ppo_v1/`:

| File | What it is |
|---|---|
| `policy.onnx` | The exported policy (CPU-ready). |
| `metadata.json` | algorithm, obs_shape, factor_count, training timesteps, git SHA. |
| `ppo_final.zip` | The raw SB3 checkpoint (keep for resume / fine-tune). |
| `checkpoints/` | Periodic SB3 checkpoints. |
| `training_metrics.jsonl` | Per-update training metrics. |
| `tb_logs/` | TensorBoard event files. |
| `training_summary.json` | Top-level training run metadata. |

### 3.4 Validate the export

```python
from aurumq_rl import RlAgentInference
agent = RlAgentInference("models/ppo_v1/")
print(agent.metadata)   # algorithm, obs_shape, etc.
```

If this print succeeds, the ONNX file and metadata are well-formed and the
model is ready for inference.

## 4. Hyperparameter tuning

### Learning rate (`--learning-rate`)

* Default `3e-4` works well for PPO with default network sizes.
* If `episode_reward_mean` plateaus early, try `1e-4` (slower, more stable).
* If reward oscillates wildly without converging, try `1e-4` or add
  gradient clipping (modify `train.py` to pass `max_grad_norm=0.5`).

### `top_k`

* Default 30 is a good starting point for ~3500-stock universes (~0.85%
  per name). It keeps single-name idiosyncratic risk bounded and matches
  what most fund managers actually run.
* Smaller `top_k` (5-15) gives the model more room to express conviction
  but increases volatility — use it for high-conviction strategies.
* Larger `top_k` (50-100) approaches a smart-beta strategy with lower
  alpha but lower turnover.

### `forward_period`

* Default 10 trading days corresponds to a roughly two-week holding period.
* For higher-turnover strategies, try 5.
* For trend-following, try 20-40 (but watch out for label leakage in
  back-test evaluation).

### `cost_bps`

* Default 30 bps is conservative for A-shares (commission + stamp duty +
  transfer fee). Real total cost for retail accounts is often 25-35 bps
  one-side.
* Setting cost too low encourages spurious turnover. Setting it too high
  pushes the model to extreme inactivity.

## 5. Parallelism (`--n-envs` + SubprocVecEnv)

`SubprocVecEnv` runs each env in a separate Python process, sharing only
the model weights via gradient updates. The right number of envs balances:

| Factor | Effect |
|---|---|
| GPU compute | More envs = more rollouts per update = better gradient signal but linearly more VRAM. |
| CPU cores | Each env consumes one Python process; oversubscription stalls. |
| Sample diversity | More envs = more independent trajectories = more diverse learning signal. |

Rule of thumb: `n_envs = min(num_cpu_cores - 2, vram_gb / 2)`. For RTX 4070
12 GB on an 8-core CPU, `--n-envs 6` is a sensible starting point.

The default `--vec-env-method` is `spawn`, which is slower to start but
safer (avoids inherited mutable state). Use `fork` only if you've verified
none of your code relies on global state.

## 6. Wandb integration

### 6.1 Offline mode (default when `--wandb` is set)

```bash
python scripts/train.py --wandb --out-dir models/ppo_v1 ...
```

This logs metrics to a local `wandb/` directory without uploading anything.
You can sync later with `wandb sync wandb/run-XYZ` or simply inspect the
files locally.

### 6.2 Online mode

```bash
wandb login   # one-time, paste your API key
python scripts/train.py --wandb --wandb-online --wandb-project my-quant ...
```

The training run will appear in your wandb dashboard with live charts of
reward, loss, fps, and entropy. Artifacts (checkpoints + final ONNX) are
uploaded automatically.

### 6.3 What gets logged

* All `TrainingMetrics` fields (timestep, reward, loss, fps, entropy, ...).
* Algorithm config (hyperparameters from CLI args).
* Periodic checkpoint zips as wandb artifacts.
* Final ONNX model as a wandb artifact.

## 7. Reading TensorBoard logs

```bash
tensorboard --logdir models/ppo_v1/tb_logs --port 6006
```

Useful charts:

| Tag | What to watch |
|---|---|
| `rollout/ep_rew_mean` | Average reward per episode. Should trend up. |
| `train/policy_gradient_loss` | Should decrease and stabilise. Spikes mean instability. |
| `train/value_loss` | Should decrease initially; small spikes are fine. |
| `train/entropy_loss` | Negative entropy. Becomes less negative as the policy sharpens. |
| `train/explained_variance` | Should approach 1. Below 0.3 means the value function is failing. |
| `train/clip_fraction` | PPO clip fraction. Above 0.3 suggests LR is too high. |

## 8. Common failures

### 8.1 "CUDA out of memory"

Reduce `--n-envs` first (each env adds VRAM). If that's not enough, reduce
the universe (use `--universe-filter hs300`) or shorten the date range.

### 8.2 Reward stuck at zero

Check `training_metrics.jsonl` for the first 1000 steps. If `episode_reward_mean`
is exactly 0, your trading mask is probably zeroing every stock. Verify:
* `is_st` data is sane (not all True).
* `vol` is non-zero for most stocks.
* `pct_chg` is in decimal form (not percentage points — i.e. 0.05 not 5.0).

### 8.3 NaN losses

Almost always a malformed factor column with infinite or NaN values that
slipped past z-scoring. Check the input Parquet:

```python
import polars as pl
df = pl.read_parquet("data/factor_panel.parquet")
for c in df.columns:
    nan_count = df[c].is_nan().sum() if df[c].dtype.is_float() else 0
    inf_count = df[c].is_infinite().sum() if df[c].dtype.is_float() else 0
    if nan_count or inf_count:
        print(c, "NaN:", nan_count, "Inf:", inf_count)
```

### 8.4 ONNX export fails

* Ensure you ran with `[train]` extras (PyTorch + onnx + onnxscript).
* The default opset is 17. If you target an older `onnxruntime`, override
  in `onnx_export.py`.
* `dynamic_axes` is configured for the batch dimension only; if your model
  produces variable-length output, edit `_export_policy_onnx`.

### 8.5 Slow data loading

`load_panel(all_a, 5y)` should run in < 30 s. If it takes minutes:
* Confirm the Parquet is on local SSD, not a network mount.
* Check that the Parquet uses a reasonable compression (zstd > snappy >
  uncompressed for our column types).
* If you partition the Parquet by year, glob it via the path and let
  `polars.scan_parquet` push down predicates.
