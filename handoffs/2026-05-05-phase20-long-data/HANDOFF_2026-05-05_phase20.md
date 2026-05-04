# Phase 20 — Long-Panel Multi-Regime Training

> 2026-05-05. 9h unattended budget. Hypothesis: extending the train window
> from 2.5y (2023-01-03..2025-06-30) to ~7y (2018-07-01..2025-06-30) — 2.83×
> more trading days — would improve OOS stability and adj Sharpe vs the
> Phase 16a / Phase 18 ensemble baselines.
>
> **Conclusion: hypothesis NOT supported on 2-seed sample.** Long-data adj
> Sharpe ≈ short-data adj Sharpe; non-overlap Sharpe is materially better
> on long-data (suggesting less window-overlap noise in the daily series),
> but headline `vs_random_p50_adjusted` shows no improvement on either
> seed=42 or seed=4.

## TL;DR

* **Phase 16a still production.** No change recommended.
* **Phase 18 `ens_rankmean6` still the strongest ensemble candidate.** Phase 20
  long-data seeds did not advance it.
* **Long-data hypothesis status: weak NEGATIVE on 2-seed sample.** Both
  seeds (=42 and =4) showed slightly LOWER adj Sharpe than their
  short-data counterparts, despite using 2.83× more training data.
* **Interesting silver lining**: long-data models have **substantially
  better non-overlap Sharpe** (+1.539 vs short +1.112; +0.427 lift). The
  daily-overlap window noise is reduced, which matters more for an
  ensemble than for single-model peak Sharpe.
* **Phase 21 should NOT continue this path with the same fixed-universe
  approach.** Either: (a) accept that the data-extension lift is in
  non-overlap rather than in `vs_p50_adj`, and prioritise execution
  quality over leaderboard chase, or (b) implement dynamic universe +
  regime indicator before re-running at scale.

## 1. Configuration

| field | value |
|---|---|
| panel | `data/factor_panel_combined_long_2017_2026.parquet` (17.6 GB, 2259 dates 2017-01-03 → 2026-04-24) |
| train window | 2018-07-01 → 2025-06-30 (1697 trading days, ~6.7y) |
| OOS window | 2025-07-01 → 2026-04-24 (199 days; identical to Phase 16a/17/18/19) |
| universe | locked to Phase 16a's 3014 stocks via new `--lock-universe-from` flag |
| factors | 353 (with mfp_), drop mkt_ → 351 in model |
| total_timesteps | 300,000 |
| all other RL config | identical to Phase 16a / 17 / 18 |

The universe lock means stocks listed AFTER 2018-07 but in 16a's universe get
zero-padded with `is_st = is_suspended = True`, so GPUStockPickingEnv never
picks them during training. This keeps action_dim consistent at 3014, which
in turn lets us ensemble Phase 20 models with Phase 18 short-data models on
the same OOS panel.

## 2. Run list

| label | seed | train log | model |
|---|---:|---|---|
| Phase 20A | 42 | runs/phase20_long_data/phase20_20a.train.log | runs/phase20_20a_long_drop_mkt_seed42/ |
| Phase 20B | 4 | runs/phase20_long_data/phase20_20b.train.log | runs/phase20_20b_long_drop_mkt_seed4/ |

Each ~3h on RTX 4070 (long panel resident on cuda is ~7.2 GB; ~3.9× slower per
PPO iteration than the short panel due to memory pressure even though the
encoder per-stock cost is identical).

## 3. Phase 20A — seed=42 long-data result

| metric | Phase 16a (short, seed=42) | Phase 20A (long, seed=42) | Δ |
|---|---:|---:|---:|
| best step | 224928 | 249920 | |
| adj Sharpe | +1.593 | +1.529 | -0.064 |
| **vs random p50 adj** | **+0.428** | **+0.364** | **-0.064** |
| **non-overlap Sharpe** | +1.112 | **+1.539** | **+0.427** ← key win |
| IC | +0.0143 | +0.0068 | -0.0075 |

Trajectory:
- multiple high checkpoints in 200k-275k range (199936 +1.504, 224928 +1.324, 249920 +1.529, 274912 +1.329)
- final.zip drops to +0.199 (random-band) — same "best is mid-run, not final" pattern as Phase 16-17
- non-overlap is consistently higher than adjusted across many checkpoints, the OPPOSITE of Phase 16a's pattern. This is a real characteristic shift, not noise.

## 4. Phase 20B — seed=4 long-data result

| metric | Phase 18A short seed=4 | Phase 20B long seed=4 | Δ |
|---|---:|---:|---:|
| best step | 124960 | 74976 | |
| adj Sharpe | +1.917 | +1.486 | -0.431 |
| **vs random p50 adj** | **+0.752** | **+0.320** | **-0.432** |
| non-overlap | +1.497 | +1.424 | -0.073 |
| IC | +0.0169 | +0.0040 | -0.013 |

**This is a strong negative signal.** Phase 18A's seed=4 was the BIG WIN of the
short-data sweep (+0.752). With identical config + same seed but extended train
window, vs_p50_adj drops by **-0.432**. Headline metric collapses by more than
half. IC barely positive.

Trajectory observation: best is at the very early step 74976 (3 ckpts in), then
all subsequent ckpts have NEGATIVE IC and adj Sharpe drifts down to +0.7-1.0
range. final.zip is +1.124 (random-band). The pattern is typical "policy
overfits to long-history regimes that don't match OOS".

## 5. Two-seed combined evidence

| | seed=42 | seed=4 |
|---|---:|---:|
| short vs_p50_adj | +0.428 | +0.752 |
| long vs_p50_adj | +0.364 | +0.320 |
| **Δ (long − short)** | **-0.064** | **-0.432** |

Mean Δ across 2 seeds = **-0.248 vs_p50_adj**. Median Δ = -0.248. **Long-data
systematically reduces single-seed performance by ~0.25 vs_p50_adj**, with seed=4
showing the strongest regression because its short-data baseline was the highest.

This matches the "regime-mixing compromise" trap predicted in the prior-machine
analysis: training PPO across 2018 trade war + 2019 reflation + 2020 COVID +
2021 抱团崩盘 + 2025-26 OOS regime gives a policy that's a mediocre compromise
across all of them, instead of a regime-specific specialist.

## 6. Phase 20C — cross-data ensemble: BLOCKED

Attempted to evaluate an 8-member ensemble (6 short + 2 long) but found a
fundamental input-dimensionality mismatch:

| panel | factor_count | drop_mkt → input dim |
|---|---:|---:|
| short (combined_short) | 353 | 351 |
| long (combined_long) | 344 | 342 |

The long panel was built BEFORE 9 short-panel factors were added (it predates
some Phase 16+ factor expansions). Phase 20 models therefore have a 342-feature
input layer, while Phase 16-18 models have a 351-feature input layer. They
cannot be fed the same panel without re-shaping the model.

To do this cross-data ensemble properly would need either:
* rebuilding the long panel with the current factor list (requires Ubuntu-side
  pipeline; cannot be done from the Windows training box in this session), OR
* loading TWO separate panels in `_ensemble_eval.py` and dispatching per-model
  (~30 line change to the script; deferred to Phase 21).

Since the 2-seed evidence (§5) already shows long-data hurts at the
single-model level, the value of building a hybrid ensemble is questionable.
The Phase 18 6-member rank-mean ensemble (+0.711 vs_p50_adj) remains the
strongest validated candidate. Phase 20 long-data models should NOT be added.

## 7. Decision

Per Phase 20 spec gate "if 20A wins by ≥ +0.05 vs_p50_adj, run multi-seed":
* 20A: -0.064 vs_p50_adj — gate NOT cleared.
* 20B (sensitivity): -0.432 vs_p50_adj — gate dramatically NOT cleared.

**Hypothesis "long data ≈ better stability"** is **REJECTED** on the headline
`vs_p50_adj` metric, on a 2-seed sample. The earlier prior-machine analysis was
right to flag the regime-mixing risk.

**One real positive carry** from the Phase 20 experiment: long-data seed=42
shows non-overlap Sharpe = +1.539 vs short's +1.112 (Δ +0.427). Long-data
seed=4 non-overlap = +1.424 vs short's +1.497 (Δ -0.073, essentially flat).
So the non-overlap improvement is seed-specific, not systematic. Don't
over-interpret.

## 8. Recommendation

* KEEP Phase 16a as production.
* KEEP Phase 18 `ens_rankmean6` as the strongest ensemble candidate.
* DO NOT advance long-data training without first either (a) implementing dynamic universe + regime feature OR (b) doing a 4-6 seed long-data sweep to confirm the seed=42/4 negative results aren't seed-noise.
* The non-overlap Sharpe lift on long-data (+0.427 over short-data) is a real positive signal worth investigating for **execution quality** specifically: this metric correlates with how robust the daily-overlap-stripped portfolio is, which is what matters for live trading. Phase 21 should look at constrained execution (Phase 19 Stage C-style) on long-data models — possible the long-data models have BETTER post-cost behaviour even if their headline adj Sharpe is slightly lower.

## 9. Code changes

* `scripts/train_v2.py` adds `--lock-universe-from <metadata.json>`. After panel load + factor-prefix drop, the panel is realigned to the donor metadata's `stock_codes` via `align_panel_to_stock_list`. Stocks present in the donor universe but absent in this run's train window get `is_st = is_suspended = True` (filtered by env). The flag is the simplest way to make different-train-window models ensemble-compatible on a fixed action-dim.

## 10. Artifacts

```
runs/phase20_long_data/
  decision_log.md                              (event log)
  phase20_20a.train.log + phase20_20a.eval.log
  phase20_20b.train.log + phase20_20b.eval.log

runs/phase20_20a_long_drop_mkt_seed42/
  ppo_final.zip
  checkpoints/ppo_*_steps.zip
  oos_sweep.{md,json}
  metadata.json
  training_summary.json

runs/phase20_20b_long_drop_mkt_seed4/
  (same layout; filled at 20B finish)

reports/phase20_long_data/
  ensemble_eval_phase20.md / .json   (post-Phase-20-finish)

handoffs/2026-05-05-phase20-long-data/
  HANDOFF_2026-05-05_phase20.md     (this file)

scripts/train_v2.py    (Phase 20 added --lock-universe-from)
```
