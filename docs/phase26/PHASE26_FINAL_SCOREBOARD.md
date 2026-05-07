# Phase 26 — final scoreboard (v1.2 cyq + tech_panel evaluation)

> 2026-05-07. Reply to upstream:
> - `oss://ledashi-oss/aurumq-rl/handoffs/2026-05-07-tech-panel-v1/` (v1 tech_panel)
> - `oss://ledashi-oss/aurumq-rl/handoffs/2026-05-07-tech-panel-v1.2-cyq-fix/` (v1.2 cyq fix)
>
> Reply target: paris.

## TL;DR

**v1.2 cyq fix works.** ✅ Production candidate: **Phase 26C2** = 23A's exact 353-col config on v1.2-cyq panel = **2.61× T-1 lift** (vs 23A baseline 2.38×, +9.7%, AND converges 4× earlier at step 50k vs step 200k).

**tech_panel doesn't help.** ❌ Adding the 30 new `tech_*` / `cmf_*` / `zt_*` columns on top of 23A's clean config (Phase 26D) collapses lift to 1.13×, even though the underlying cyq panel is now correct. Confirms Phase 26A's IG/permutation finding that these factors have negative IC-drop at the current encoder capacity (128→64→32 per-stock MLP).

## Full scoreboard (eval window 2025-07-01 .. 2026-04-24)

| run | factors | cyq panel | train end | best T-1 lift (top5) | best T-1 hit | best ckpt | verdict |
|---|---:|---|---|---:|---:|---|---|
| **23A baseline** | 353 | legacy (88% NaN) | 2025-06-30 | 2.38× | 2.11% | step200k | original prod |
| Phase 26A | 373 | v1.0 (broken) | 2024-12-31 | 1.36× | 1.21% | step100k+150k+final | regression −43% |
| Phase 26B-baseline | 343 (no tech) | v1.0 (broken) | 2024-12-31 | 1.47× | 1.31% | step50k | regression worse |
| Phase 26C | 343 (no tech) | v1.2 (fixed) | 2024-12-31 | 1.47× | 1.31% | step50k | confirmed train-window matters |
| **Phase 26C2** ⭐ | **353 (23A exact)** | **v1.2 (fixed)** | **2025-06-30** | **2.61×** | **2.31%** | **step50k** | **PRODUCTION** |
| Phase 26D | 383 (353+tech) | v1.2 (fixed) | 2025-06-30 | 1.13× | 1.01% | step100k (top3) | tech rejected |

## Key insights

### 1. v1.2 cyq fix is correct (26C2 vs 23A)

Same 353-col config, same train window, only difference is cyq columns:
- 23A used legacy `quotes_enriched.cyq_*` (88% NaN → effectively ignored by model)
- 26C2 uses v1.2 `cyq_panel.*` (0.86% NaN train, 0.00% NaN OOS)

26C2 not only matches 23A but **slightly exceeds it** (2.61× vs 2.38×) AND **converges 4× faster** (best at step 50k vs step 200k). The cleaner cyq signal helps optimization, not just inference.

### 2. Train-window matters (26B vs 26C2)

My initial Phase 26 attempts used `--end-date 2024-12-31`, which left a 6-month gap between training end and OOS start. 23A trained through 2025-06-30 (immediate continuation into OOS).

Recovering the correct train window alone (Phase 26C2) reproduced 23A baseline parity. Phase 26B / 26C with the wrong train window stuck at 1.47× best regardless of cyq version.

### 3. tech_panel hurts at current encoder capacity (26D vs 26C2)

Same panel, same train window, same factor list except for 30 added tech cols:
- 26C2 (353 cols): **2.61×** lift
- 26D (353 + 30 tech = 383 cols): **1.13×** lift  →  −57%

Phase 26A's IG/permutation analysis already diagnosed this: tech_/cmf_/zt_ have **negative cross-validation IC drop** (IC improves when these factors are zeroed). At the 128→64→32 per-stock encoder capacity, adding 30 new columns dilutes attention away from the strong alpha_/gtja_/mfp_ signals.

Possible future paths if tech is wanted:
- Bigger encoder (e.g. 256→128→64); needs separate baseline rerun
- Permutation-pruned subset of tech_ (only `tech_boll_percent` + `cmf_120d_pct_amt` had IG > 1e-9)
- Drop tech entirely (current recommendation)

## What we changed in the RL repo

**Branch: `feat/phase26-tech-panel-v1`** (commits 069ee0c, b7d9ee2, d5c9a29, d841b74).

```
src/aurumq_rl/data_loader.py       — added tech_/cmf_/zt_ prefixes; inf-protection in z-score
scripts/export_factor_panel.py     — same prefix update + missing mfp_
scripts/train_v2.py                — new --include-columns-file flag for explicit factor pinning
scripts/build_combined_panel_v26.py — joins base + tech_panel + cyq_panel
tests/test_data_loader.py          — updated prefix lockdown test
docs/phase26/PHASE26A_RESULT.md            — Phase 26A regression analysis
docs/phase26/PHASE26_HANDOFF_REPLY.md      — initial cyq-backfill diagnosis (paris fixed in v1.2)
docs/phase26/PHASE26_DATA_QUALITY_AUDIT.md — 94 problematic factor cols, full csv attached
docs/phase26/PHASE26_FINAL_SCOREBOARD.md   — this file
```

## Recommendations

### Immediate (production)

1. **Switch production from 23A to 26C2.** Same 353 cols, same architecture, +9.7% T-1 lift, 4× faster convergence. Only data change is v1.2 cyq override.

2. **Skip tech_panel for now.** Re-evaluate after upstream cleans the inf'd alpha/gtja columns AND we've tested a bigger encoder.

### Short-term (data quality)

Per `PHASE26_DATA_QUALITY_AUDIT.md`:

1. Drop 3 100%-null columns (alpha_029, alpha_031, gtja_143) from `alpha_panel`.
2. Fix gtja_017 / gtja_005 / gtja_114 / alpha_045 / +9 inf-overflow columns.
3. Emit `_log` or `_pct` variants for the 50+ unnormalized HUGE_TAIL columns.

### Future experiments (after upstream fix)

- Phase 26E: 26C2 setup but with fixed alpha/gtja (no inf, no overflow).
- Phase 26F: 26E + tech_panel with bigger encoder (256→128→64).
- Phase 26G: permutation-pruned subset (drop hm_, hk_, sh_, mg_, plus tech subset).

## Files

```
runs/phase26a_episode_seed42/         (regression study, v1.0 cyq)
runs/phase26b_baseline_seed42/        (343-col, v1.0 cyq, wrong train window)
runs/phase26c_baseline_seed42/        (343-col, v1.2 cyq, wrong train window)
runs/phase26c2_23a_exact_seed42/      (353-col 23A exact, v1.2 cyq, correct train window) ⭐ PRODUCTION
runs/phase26d_full_seed42/            (383-col 23A + tech, v1.2 cyq)
data/factor_panel_phase26a_2023_2026.parquet   8.1 GB  (v1.0 cyq + tech)
data/factor_panel_phase26c_2023_2026.parquet   8.1 GB  (v1.2 cyq + tech, 343-col list)
data/factor_panel_phase26c2_2023_2026.parquet  8.1 GB  (23A panel + v1.2 cyq override) ⭐
data/factor_panel_phase26d_2023_2026.parquet   8.4 GB  (23A panel + v1.2 cyq + tech_panel)
docs/phase26/data_quality_audit.csv            28 KiB  (385-col audit table)
```
