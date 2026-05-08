# Phase 26 — RL-side reply to upstream tech_panel_v1 handoff

> 2026-05-07. Reply to `oss://ledashi-oss/aurumq-rl/handoffs/2026-05-07-tech-panel-v1/HANDOFF_TECH_PANEL_V1.md`.
> RL side: ledashi · Data side: paris

## TL;DR

Phase 26A and Phase 26B-baseline both **regressed materially** vs the
Phase 23A production baseline on the same OOS window:

| run | factors | best T-1 lift (top5) | median T-1 lift | verdict |
|---|---:|---:|---:|---|
| Phase 23A (production) | 353 (343 useful) | **2.38×** | ~1.50× | KEEP |
| Phase 26A (343 + 30 tech) | 373 | 1.36× | 1.25× | regression −43% |
| Phase 26B-baseline (343 only) | 343 | 1.47× | 0.68× | regression worse |

Phase 26B-baseline drops the 30 new tech/cmf/zt cols, runs Phase 23A's exact
343-col config on the new combined panel — and is *worse* than 26A. So the
regression is **not from the tech factors**. It's from the panel data itself,
specifically the cyq_panel canonical-source change.

## Root cause: cyq_perf backfill ≠ real-source distribution

The HANDOFF noted:
> Before this work `cyq_perf` table only had data from 2025-10-20+;
> backfilled to 2023-01-03 (3.65M rows).

The backfilled period and the real-data period have very different
distributions:

| metric | backfill (<2025-10-20) | real (≥2025-10-20) | Δ |
|---|---:|---:|---|
| null rate (3 cyq_ cols) | 0.61% | 26.53% | +25.9pp |
| `cyq_cost_distance` std | 0.1976 | 0.0662 | **3.0× wider** |
| `cyq_cost_distance` median | +0.0019 | -0.0228 | -0.025 |
| `cyq_concentration_70` std | 0.0925 | 0.0705 | +31% |
| `cyq_winning_ratio` median | 42.73 | 37.65 | -5.08 |

The training window (2023-01..2024-12) is **100% backfilled**. The eval
window (2025-07..2026-04) is **~37% backfilled + ~63% real**.

After cross-section z-score, the `cyq_cost_distance` column at OOS gets
3× tighter spread than what the model trained on. The encoder's learned
weights are mis-calibrated → the cyq_ feature ranking flips → lift collapses.

This explains why:

1. Phase 23A baseline does fine — its `cyq_*` was 88% NaN→0 (legacy cyq
   from quotes_enriched only had ~12% coverage in this date range). The
   model effectively ignored the column. The other 350 factors carry the
   load. No distribution shift to suffer.
2. Phase 26B-baseline (343 cols, no tech) regresses — same 343 columns,
   but now `cyq_*` is 95% populated with a distribution shift across the
   train→OOS boundary.
3. Phase 26A (with tech_) regresses *less* than 26B-baseline because the
   30 tech_ columns absorb some of the per-stock attention that would
   otherwise route to the now-misleading cyq_ signal. Tech factors aren't
   helping; they're *diluting* the harmful cyq_ signal.

## RL-side conclusion

We cannot fix this on the RL side. Cross-section z-score would equalize
the backfill/real difference IF training and eval were both fully one-or-
the-other, but training is 100% backfill and eval mixes both regimes.

The fix has to land upstream:

1. **Recompute the backfill with a methodology that matches the real
   `cyq_perf` table's source data and null-rate.** The current backfill
   appears to use a different upstream input (which is why it has 0.6%
   nulls vs the real table's 26.5%). Either:
   - Backfill from the same source and propagate the same nulls, OR
   - Forward-fill / interpolate the real data instead of recomputing.

2. **OR drop cyq_panel from the include list and revert to 23A's sparse
   cyq_* from quotes_enriched.** Net 26A factor count = 370 (or 340 if
   also dropping tech_/cmf_/zt_ to reset to 23A baseline + nothing).

3. **OR produce a parquet covering only post-2025-10-20** real cyq_perf
   data, and we change train+eval windows to that range. But 6 months of
   data is too short for our PPO run.

Option 1 is cleanest. Until it lands, **production stays on Phase 23A
unchanged**.

## What we tried

- **Phase 26A (300k, seed 42, 373 cols)** — same training config as 23A,
  panel = 343 23A-base + 30 new tech_/cmf_/zt_, with canonical cyq_panel
  overriding legacy quotes_enriched cyq.

- **Phase 26B-baseline (300k, seed 42, 343 cols)** — same as 26A but with
  the 30 new factors dropped (using `--include-columns-file` with 23A's
  exact 343-col allowlist). Tests "is the regression from the tech
  factors or from the panel data change?" Answer: from the panel data.

- **IG + permutation importance on Phase 26A** —
  `runs/phase26a_episode_seed42/factor_importance.json`. Net IC drop of
  the 30 new factors: -0.0029 (slightly harmful). hm_ at -0.0042 is the
  most harmful in the new panel — also a known stable group in 23A —
  consistent with the cross-attention-displacement story above.

## What we did NOT try (ranked by upside if the upstream fix lands)

1. **26C: 23A's 343 cols on a fixed-cyq panel.** Once upstream re-emits
   cyq_panel without the distribution discontinuity, this should match
   or beat 23A.

2. **26D: 343 cols + 30 tech with bigger encoder (256, 128).** Tests
   capacity-vs-dilution if 26C confirms the panel is sound.

3. **Permutation-importance-based subset.** Drop hm_ + cyq_ + tech_ +
   hk_ + zt_ (groups with negative IC drop in 26A perm importance). Net
   ~330 cols. Phase 25-style "trust IG" but applied at group level.

## Files

- `docs/phase26/PHASE26A_RESULT.md` — Phase 26A detailed analysis
- `runs/phase26a_episode_seed42/` — 26A train + eval + IG outputs
- `runs/phase26b_baseline_seed42/` — 26B-baseline train + eval outputs
- `data/tech_panel_v1/` — original 17-file OSS bundle from upstream
- `data/factor_panel_phase26a_2023_2026.parquet` — combined panel (8.1 GB)

## Reproduce diagnosis

```python
import polars as pl
from datetime import date
df = pl.read_parquet("data/factor_panel_phase26a_2023_2026.parquet")
cutoff = date(2025, 10, 20)
backfill = df.filter(pl.col("trade_date") < cutoff)
real     = df.filter(pl.col("trade_date") >= cutoff)
for col in ["cyq_winning_ratio","cyq_concentration_70","cyq_cost_distance"]:
    nb = backfill[col].null_count() / len(backfill) * 100
    nr = real[col].null_count() / len(real) * 100
    print(f"{col}: backfill null={nb:.2f}%  real null={nr:.2f}%")
```

Expected output (2026-05-07 build):

```
cyq_winning_ratio: backfill null=0.61%  real null=26.53%
cyq_concentration_70: backfill null=0.61%  real null=26.53%
cyq_cost_distance: backfill null=0.61%  real null=26.53%
```
