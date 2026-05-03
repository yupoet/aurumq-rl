# Phase 18 — 6h 无人值守迭代训练优化指令

> 基于 `/data/AurumQ/handoffs/2026-05-03-phase17-7h/`。本轮目标不是继续追逐单次 permutation importance 的删因子结论，而是把 Phase 16/17 已验证的 `drop_mkt` regime 做成更稳健的多 seed/多 checkpoint 选择，并补齐 seed=2 失败诊断。

## 0. 绝对约束

1. **总预算 6 小时**。到 5h40m 必须停止新训练，进入报告和 handoff。
2. **主评价口径只用 corrected eval**：
   - `forward_period=10`
   - `top_k=30`
   - OOS: `2025-07-01` → `2026-04-24`
   - primary metric: `vs_random_p50_adjusted`
   - secondary gates: `adjusted_sharpe`, `IC`, `non_overlap`
   - adjusted Sharpe = `mean / std * sqrt(252 / forward_period)`；不要用 legacy `sqrt(252)` 口径排序。
3. **不要再基于 importance 直接删因子组**。Phase 17A 已证明 cyq/inst 的 robust negative importance 不会转移到 retrain。
4. **只允许继续 drop `mkt_`**。默认保留 `gtja_ / alpha_ / mfp_ / cyq_ / inst_ / fund_ / mf_` 等所有非市场因子。
5. **不要覆盖 Phase 16a production baseline**：
   - baseline: `phase16_16a_drop_mkt_best.zip`
   - step: `224928`
   - adjusted Sharpe: `+1.593`
   - vs random p50 adjusted: `+0.428`
   - non-overlap: `+1.112`
   - IC: `+0.0143`
6. **单次 OOS 优胜不能直接视为生产 Sharpe**。Phase15/16/17 的 OOS Sharpe 是相对 ranking 指标，仍需后续真实持仓回测、交易成本、涨跌停/ST/停牌约束和最终 holdout。

## 1. 已知输入

Phase 17 结论：

| model | config | best step | adj S | vs p50 adj | non-overlap | IC | decision |
|---|---|---:|---:|---:|---:|---:|---|
| Phase16a / 17E | drop `mkt_`, seed=42 | 224928 | +1.593 | +0.428 | +1.112 | +0.0143 | baseline |
| 17D | drop `mkt_`, seed=3 | 24992 | +1.514 | +0.348 | +1.690 | +0.0087 | ensemble candidate |
| 17B | drop `mkt_`, seed=1 | 174944 | +1.446 | +0.280 | +0.759 | +0.0049 | ensemble candidate |
| 17C | drop `mkt_`, seed=2 | 224928 | +1.105 | -0.060 | +1.754 | -0.0025 | failed seed |
| 17A | drop `mkt_+cyq_+inst_`, seed=42 | final | +0.861 | -0.304 | +0.876 | +0.0038 | do not deploy |

Model artifacts are in:

```text
/data/AurumQ/handoffs/2026-05-03-phase16-corrected-eval/models/
/data/AurumQ/handoffs/2026-05-03-phase17-7h/models/
```

If running on the Windows GPU box used in Phase 17, map these to the equivalent `D:\Dev\aurumq-rl\...` paths. Keep path handling explicit in the decision log.

## 2. Preflight hard gate

Before running any new training:

1. Create run workspace:

```bash
mkdir -p runs/phase18_6h reports/phase18_6h models/phase18
```

2. Log:
   - current branch / HEAD
   - working tree status
   - Python path
   - CUDA device
   - data file checksum or file size + mtime
   - exact Phase16/17 model paths being used

3. Verify evaluation script emits corrected fields:
   - `top_k_sharpe_adjusted`
   - `random_p50_sharpe_adjusted`
   - `vs_random_p50_adjusted`
   - `top_k_sharpe_non_overlap`

If local `scripts/_eval_all_checkpoints.py` only emits legacy `top_k_sharpe`, stop and patch/evaluate with the Phase16/17 corrected logic before doing any training. Do **not** rank by legacy Sharpe.

## 3. Phase 18 mandatory queue

### Stage A — cheap ensemble evaluation, max 75 min

Build/evaluate score ensembles from existing good models before spending GPU time.

Candidate base models:

1. `phase16_16a_drop_mkt_best.zip` or Phase17 `phase17_17e_drop_mkt_seed42_450k_best.zip` (bit-equivalent)
2. `phase17_17d_drop_mkt_seed3_best.zip`
3. `phase17_17b_drop_mkt_seed1_best.zip`

Evaluate these aggregation variants:

| ensemble | members | aggregation |
|---|---|---|
| `ens3_zmean` | 16a + 17D + 17B | per-date z-score each model, then mean |
| `ens3_zmedian` | 16a + 17D + 17B | per-date z-score each model, then median |
| `ens3_rankmean` | 16a + 17D + 17B | per-date percentile/rank score, then mean |
| `ens2_16a_17d_zmean` | 16a + 17D | per-date z-score mean |
| `ens4_with_seed2_sensitivity` | 16a + 17D + 17B + 17C | rankmean only, sensitivity check; do not deploy if seed2 degrades |

Implementation notes:

- Load the validation panel once.
- Load one PPO model at a time; collect score matrix `(n_dates, n_stocks)`; then free the model and clear CUDA cache.
- Align every model to its own training `stock_codes`, then to a shared validation stock axis.
- For each date/model, normalize only finite stock scores. If daily score std is near zero, fall back to rank aggregation for that model/date.
- Do **not** average raw PPO scores across models without normalization.
- Evaluate using the same `panel.return_array` alignment as corrected `_eval_all_checkpoints.py`. Do not use factor_importance's return indexing for final selection.

Stage A success gate:

- `candidate`: `vs_random_p50_adjusted >= +0.478` (Phase16a + 0.05), `IC > 0`, and `non_overlap >= +0.90`
- `strong_candidate`: `vs_random_p50_adjusted >= +0.528` (Phase16a + 0.10), `IC >= +0.010`, and monthly results are not concentrated in one month

If no ensemble clears `candidate`, continue to Stage B anyway; the seed robustness data is still valuable.

### Stage B — additional drop_mkt seeds, target 3 runs

Train more seeds under exactly the Phase16a/17B/17D regime:

```bash
$PY scripts/train_v2.py \
  --data-path data/factor_panel_combined_short_2023_2026.parquet \
  --start-date 2023-01-03 \
  --end-date 2025-06-30 \
  --universe-filter main_board_non_st \
  --n-envs 16 \
  --episode-length 240 \
  --batch-size 1024 \
  --n-steps 1024 \
  --n-epochs 10 \
  --learning-rate 1e-4 \
  --target-kl 0.30 \
  --max-grad-norm 0.5 \
  --rollout-buffer index \
  --tf32 \
  --matmul-precision high \
  --unique-date-encoding \
  --checkpoint-freq 25000 \
  --forward-period 10 \
  --top-k 30 \
  --total-timesteps 300000 \
  --seed <SEED> \
  --out-dir runs/phase18_18<letter>_drop_mkt_seed<SEED> \
  --drop-factor-prefix mkt_
```

Default seed order:

| run | seed | priority |
|---|---:|---|
| 18A | 4 | mandatory |
| 18B | 5 | mandatory if remaining >= 3h45m after Stage A |
| 18C | 6 | mandatory if remaining >= 2h35m after 18B eval |
| 18D | 7 | optional only if previous stages finished and remaining >= 85 min |

After each seed:

```bash
$PY scripts/_eval_all_checkpoints.py \
  --run-dir runs/phase18_18<letter>_drop_mkt_seed<SEED> \
  --data-path data/factor_panel_combined_short_2023_2026.parquet \
  --val-start 2025-07-01 \
  --val-end 2026-04-24 \
  --top-k 30 \
  --device cuda
```

Then immediately:

1. Copy the best checkpoint to `models/phase18/phase18_18<letter>_drop_mkt_seed<SEED>_best.zip`.
2. Add it to the seed table.
3. Re-run ensemble variants including only eligible seeds:
   - eligible seed: `vs_random_p50_adjusted > +0.20`, `IC > 0`, and `non_overlap > +0.70`
   - exclude seed=2 from deployable ensembles unless an explicit sensitivity report proves it helps

Adaptive stop rules:

- If two consecutive new seeds have `vs_random_p50_adjusted <= 0` or `IC <= 0`, stop training new seeds and switch to Stage C.
- If one new seed beats Phase16a by `>= +0.10 vs_p50_adj`, run extra validation on that seed but do not overwrite production.
- If remaining time is below 90 min, do not launch another 300k run.

### Stage C — seed=2 failure diagnostics, 45-60 min

Write `reports/phase18_6h/seed2_failure_diagnostics.md`.

Compare Phase17C seed=2 against Phase17D seed=3 and Phase16a seed=42:

1. Training trajectory:
   - `training_metrics.jsonl` reward/value loss/entropy/approx_kl if available
   - checkpoint adj Sharpe trajectory
   - does seed=2 recover late or stay flat?
2. OOS shape:
   - monthly adjusted Sharpe / mean top-K return
   - per-month IC
   - drawdown/cumret path from corrected series
3. Pick overlap:
   - daily top30 overlap with 16a and 17D
   - industries/sectors over-selected by seed=2
4. Score behavior:
   - per-date score std/quantiles
   - count of near-constant or saturated days
5. Conclusion must classify seed=2 as one of:
   - optimization failure
   - bad but stable local optimum
   - OOS date/industry concentration problem
   - evaluation/data alignment issue
   - inconclusive

If `scripts/_industry_oos_analysis.py` is available on the run host, use it for 17C/17D/16a. Otherwise compute the diagnostics from saved predictions.

### Stage D — optional checkpoint diversity ensemble

Only run if Stage A/B/C are done and remaining time >= 45 min.

If the original training run directories still exist, evaluate ensembles that include strong secondary checkpoints:

- 17D: `24992`, `49984`, `final`
- 17B: `174944`, `274912`
- 17E/16a: `199936`, `224928`, `349888`, `374880`

Purpose: test whether checkpoint diversity improves ensemble stability. Do not use the same OOS window to overfit a large checkpoint cocktail; report only small, interpretable variants.

## 4. Decision rules

Rank by `vs_random_p50_adjusted`, not legacy Sharpe.

Promotion language:

| result | action |
|---|---|
| no ensemble/seed beats Phase16a by +0.05 | keep Phase16a, report seed CI only |
| ensemble beats +0.05 but not +0.10 | mark as Phase18 candidate, not production |
| ensemble beats +0.10 and passes IC/non-overlap/monthly checks | recommend implementation of runtime ensemble in serving path, but still require final holdout/live-style portfolio validation |
| single new seed beats +0.10 | do not promote alone; include in ensemble and request more seed confirmation |
| seed=2-like failures continue | recommend production ensemble exclude failing seeds and report seed-risk distribution |

Never recommend dropping cyq/inst/fund solely from Phase17/18 importance. Only retrain evidence can justify a factor removal.

## 5. Required outputs

At the end, produce:

```text
runs/phase18_6h/decision_log.md
reports/phase18_6h/ensemble_eval.md
reports/phase18_6h/ensemble_eval.json
reports/phase18_6h/seed2_failure_diagnostics.md
reports/phase18_6h/final_ranking.md
models/phase18/*_best.zip
handoffs/2026-05-03-phase18-6h/HANDOFF_2026-05-03_phase18.md
```

Final handoff must include:

1. Exact run list with seed/config/steps.
2. Corrected eval table for every single model and ensemble.
3. Seed distribution stats:
   - mean/median/std/min/max of `vs_random_p50_adjusted`
   - win rate vs random p50
   - IC positive rate
4. Best deployable ensemble config:
   - member model paths
   - aggregation method
   - score normalization
   - metrics
5. Explicit recommendation:
   - keep Phase16a
   - mark Phase18 ensemble candidate
   - or no promotion
6. A list of anything skipped due to time.

## 6. Timebox schedule

| elapsed | expected state |
|---:|---|
| 0:00-0:10 | preflight, corrected eval verification |
| 0:10-1:15 | Stage A existing-model ensemble eval |
| 1:15-2:25 | train/eval seed 4 |
| 2:25-3:35 | train/eval seed 5 |
| 3:35-4:45 | train/eval seed 6, unless stop rule fires |
| 4:45-5:35 | seed=2 diagnostics + final ensemble refresh |
| 5:35-6:00 | final report + handoff |

If Stage A implementation overruns by more than 45 min, skip seed 6 first, not the final handoff.

## 7. Final answer format for the next agent

Use this final summary shape:

```text
Phase18 completed in <elapsed>.

Best result:
- <model_or_ensemble>: adj_S=<...>, vs_p50_adj=<...>, non_overlap=<...>, IC=<...>
- Delta vs Phase16a: <...>

Recommendation:
- <KEEP Phase16a | Phase18 ensemble candidate | no promotion>

Key evidence:
- seed robustness: <mean/median/win rate>
- ensemble result: <best aggregation>
- seed=2 diagnosis: <classification>

Artifacts:
- <paths>
```

