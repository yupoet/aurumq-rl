# Phase 18 — Ensemble evaluation summary

OOS window: 2025-07-01 → 2026-04-24 (199 dates, fp=10).
Phase 16a baseline: adj_S=+1.593, vs_p50_adj=+0.428, non_overlap=+1.112, IC=+0.0143.

## Variants tested

This file collates results from these stages of the orchestrator:

* **Stage A** (`reports/phase18_6h/ensemble_eval_stage_a.md`): 4 candidate models (16a / 17b / 17d / 17c), 5 ensemble variants + 4 singletons.
* **Stage B refresh after seed=4** (`ensemble_eval_stage_b_after_seed4.md`): orchestrator retried after subprocess argument-parsing bug fix, see decision_log.
* **Stage B refresh after seed=5** (`ensemble_eval_stage_b_after_seed5.md`): 5 eligible members, 3 ensemble variants.
* **Stage B refresh after seed=6** (`ensemble_eval_stage_b_after_seed6.md`): 6 eligible members, 3 ensemble variants.
* **Stage B refresh after seed=7** (`ensemble_eval_stage_b_after_seed7.md`): seed=7 ineligible (vs_p50_adj +0.004 < 0.20 gate); same 6 eligible members; identical to seed=6 ensembles.
* **Final** (`ensemble_eval_final.md`): 6 eligible members + checkpoint-diversity baseline `ens_final_top3_by_vsp50_zmean` (16a+17b+17d, kept for forensic comparison).

The headline numbers are concentrated in `final_ranking.md`; this file documents the methodology and the per-stage refresh comparison.

## Aggregation methods

Per-date z-score-mean (`zmean`):
```
For each (model m, date t):
    finite_mask = isfinite(scores[m,t,:])
    z[m,t,:] = (scores[m,t,:] - mean[m,t]) / std[m,t]   # nan-respecting
combined[t,:] = mean over m of z[m,t,:]
```

Per-date z-score-median (`zmedian`): same as zmean but median instead of mean. More robust to outliers; less responsive to a single seed-4-class strong member.

Per-date percentile-rank-mean (`rankmean`):
```
rank[m,t,:] = (argsort(argsort(scores[m,t,:])) + 0.5) / n_finite
combined[t,:] = mean over m of rank[m,t,:]
```

## Methodology choices

* Single panel load with canonical `factor_names` + `stock_codes` from 16a's metadata — verified (via warnings) that all members were trained on the same factor schema and stock universe.
* All members loaded onto cuda, scored once, cached as `.npy` in `reports/phase18_6h/_scores_cache_<label>.npy` so the orchestrator could iterate variants without rescoring.
* Models freed and `torch.cuda.empty_cache()` between loads to keep VRAM under 6 GiB.
* Each variant evaluated with the same `run_backtest_with_series(forward_period=10)` path used by `_eval_all_checkpoints.py`, so the ranking is on identical metric definitions.

## Eligibility gate (instructions §3.B)

A new seed enters the deployable ensemble only if all hold:

* `vs_random_p50_adjusted` > +0.20
* `IC` > 0
* `non-overlap Sharpe` > +0.70

Phase 18 outcome by gate:

| seed | vs_p50_adj | IC | non-overlap | eligible? |
|---:|---:|---:|---:|---|
| 4 | +0.752 | +0.0169 | +1.497 | ✓ |
| 5 | +0.596 | +0.0175 | +1.368 | ✓ |
| 6 | +0.469 | +0.0111 | +2.298 | ✓ |
| 42 (16a) | +0.428 | +0.0143 | +1.112 | ✓ |
| 3 (17D) | +0.348 | +0.0087 | +1.690 | ✓ |
| 1 (17B) | +0.280 | +0.0049 | +0.759 | ✓ |
| 7 | +0.004 | -0.0002 | +1.250 | ✗ (vs_p50_adj < 0.20, IC ≤ 0) |
| 2 (17C) | -0.060 | -0.0025 | +1.754 | ✗ (vs_p50_adj < 0.20, IC ≤ 0) |

6 of 8 seeds eligible. The deployable ensemble uses those 6.

## Final ranking and best variant

See `final_ranking.md`. Headline: **`ens_final_rankmean`** (6 members) with vs_p50_adj = +0.711, IC = +0.0278, non-overlap = +1.938.

## Per-aggregation pattern across stages

```
                       members:   3    5    6    6
Stage A    after-Sd4   after-Sd5  after-Sd6  final
zmean      +0.422      +0.603     +0.561     +0.561
zmedian    +0.250      n/a*       +0.623     +0.623
rankmean   +0.408      n/a*       +0.711     +0.711

* Stage B after-Sd4 ensemble call failed due to a path-parsing bug (Windows
absolute paths split on ":"); the after-Sd5 refresh ran on the fixed script
and is the first multi-seed ensemble result we have.
```

Two observations:

1. **More members helps**: rankmean grew from +0.408 (3 members) to +0.711 (6 members). This is a strong signal that the underlying drop_mkt regime is a real signal — adding more independently-trained policies adds independent information, not noise.
2. **Aggregation choice matters**: with 3 weak-to-medium members (Stage A), zmean was the best (+0.422). With 6 members (mixing in two Phase 18 BIG WINs), rankmean wins (+0.711). The intuition: with one strong outlier (seed=4 at +0.752) plus 5 medium members, rank-mean gives the strong member appropriate weight without letting its raw Sharpe dominate the average. zmean tends to be dragged down by the weaker members, while zmedian throws away the strong member's signal.

The runtime ensemble path should use **rankmean over the 6 eligible members**.
