# Phase 26 — factor data quality audit

> 2026-05-07. Audit of `data/factor_panel_phase26d_2023_2026.parquet`
> (4.26M rows × 394 cols, train 2023..2025-06 + OOS 2025-07..2026-04).
> Reply target: paris (data team).

## TL;DR

Audited 385 factor columns. **94 have material data-quality issues**:

| flag | count | severity | description |
|---|---:|:---:|---|
| 100% null | **3** | 🚫 critical | entire column is null across the panel |
| INF_HEAVY (>100 inf cells) | **13** | 🚫 critical | upstream factor calc overflows / div-by-zero |
| INF_FEW (≤100 inf cells) | 7 | ⚠️ moderate | same, less prevalent |
| HUGE_TAIL (abs_p99 > 100, no inf) | 59 | ⚠️ moderate | unnormalized scale (mf_*: yuan; gtja_*: rate*price*volume) |
| DEAD_NULL (>99% null) | 1 | 🚫 critical | `senti_ths_hot_pct` data starts 2024-01-15 (HANDOFF v1 noted) |
| DEAD_VAR (std < 1e-6) | 10 | 🚫 critical | constant after fp32 cast (mfp_main_net_*, hm_net_*) |

Net 35 columns are flat-out broken (100% null + INF_HEAVY + DEAD_NULL + DEAD_VAR).

Detail CSV: `docs/phase26/data_quality_audit.csv`.

## RL-side mitigation already applied

`src/aurumq_rl/data_loader.py::_cross_section_zscore` patched to replace
`±inf` with `nan` before `nanmean` / `nanstd`. Without this, a single inf
in a (date, factor) cross-section poisons the entire row through:

```
nanmean([1, 2, inf, 4]) = inf
nanstd([1, 2, inf, 4]) = nan
z = (x - inf) / (nan + 1e-8) = nan
nan_to_num(...) = 0   ← the WHOLE 3000-stock cross-section is zeroed
```

Now the offending stock alone is treated as missing (→ 0 after fillna),
the other 2999 stocks get correct z-scores. Phase 26D and later trainings
benefit; Phase 23A / 26A / 26B / 26C / 26C2 were trained without this
protection but the upstream-side fix below should make it moot.

## 🚫 Critical — 3 columns 100% null

Entire panel (4.26M rows) is null:

```
alpha_029
alpha_031
gtja_143
```

**Action:** drop from upstream `alpha_panel` / `gtja_panel`. They
contribute zero signal but consume 3 input slots in the Per-Stock Encoder
(at 343-col baseline that's ~0.9% of input dimensionality wasted).

## 🚫 Critical — 13 columns with massive inf rates

Upstream factor computation has unprotected division / log / power that
overflows. Most likely sources: division by 0-volume bars, log of 0/negative
prices, accumulating products that exceed fp64 range.

| column | inf cells | inf% | std (finite) | abs_p99 (finite) | max_abs (finite) |
|---|---:|---:|---:|---:|---:|
| `gtja_005` | 108,787 | 2.55% | 5.6e+05 | 1.00 | 6.16e+08 |
| `alpha_045` | 48,514 | 1.14% | 0.33 | 0.86 | 78.8 |
| `gtja_114` | 9,331 | 0.22% | 1.0e+02 | 53.6 | 3.15e+04 |
| **`gtja_017`** | **7,647** | 0.18% | **inf** | **9.3e+37** | **1.4e+308** |
| `alpha_083` | 3,738 | 0.09% | 1.0e+02 | 53.6 | 3.15e+04 |
| `gtja_164` | 1,777 | 0.04% | 1.1e+10 | 1.26e+03 | 1.08e+13 |
| `gtja_190` | 1,026 | 0.02% | 1.18 | 4.29 | 20.1 |
| `alpha_022` | 231 | 0.005% | 0.38 | 1.28 | 1.96 |
| `gtja_104` | 231 | 0.005% | 0.38 | 1.28 | 1.96 |
| `gtja_191` | 160 | 0.004% | 21.8 | 8.68 | 1.4e+04 |
| `alpha_026` | 156 | 0.004% | 0.40 | 1.00 | 1.00 |
| `gtja_062` | 133 | 0.003% | 0.52 | 0.99 | 1.00 |
| `alpha_044` | 127 | 0.003% | 0.52 | 0.99 | 1.27 |

`gtja_017` is the worst — even the finite p99 is **9.3 × 10³⁷**, max is
**1.4 × 10³⁰⁸** (near `np.finfo(np.float64).max`). Any `float32` cast
becomes `inf`; the column is unusable. The Phase 26 RL z-score patch will
treat its inf cells as missing, but the **finite tail of 10³⁷ scale** still
dominates the cross-section after it converts the inf to nan.

**Action 1:** in upstream factor calc, wrap `gtja_017` and the other 12
in `np.clip(x, -1e6, 1e6)` or replace `x / y` with `np.where(y > 1e-9, x/y, np.nan)`.
**Action 2:** double-check that `gtja_017`'s formula isn't fundamentally
ill-posed — abs_p99=10³⁷ is too far for clip to fix without distortion.

## ⚠️ 59 columns with huge unnormalized tails (abs_p99 > 100)

Two sub-groups:

### Group A: `mf_net_*` raw yuan-scale (8 cols)

```
mf_net_60d:  std = 8.4e+08    abs_p99 = 3.2e+09    max_abs = 7.6e+10
mf_net_20d:  std = 3.6e+08    abs_p99 = 1.3e+09    max_abs = 3.3e+10
mf_net_10d:  std = 2.2e+08    abs_p99 = 7.7e+08    max_abs = 2.5e+10
mf_net_5d:   std = 1.4e+08    abs_p99 = 4.9e+08    max_abs = 2.2e+10
mf_net_3d:   std = 1.0e+08    abs_p99 = 3.6e+08    max_abs = 1.9e+10
mf_net_1d:   std = 5.8e+07    abs_p99 = 1.9e+08    max_abs = 1.2e+10
mf_net_60d:  std = 8.4e+08    ...
mf_net_accel_5_20: std = 2.2e+07
```

These are **net inflow in yuan** — natural scale 10⁸..10¹⁰. After
cross-section z-score they're fine, BUT a few stocks per day with bulk
trades push the single-day std to multi-billion territory and squeeze
the rest of the cross-section into [-0.1, +0.1] z. That's bad signal-to-noise.

**Action:** upstream emit a `_log` or `_pct` variant — `sign(x) *
log(1 + |x|/median_amount)` or `x / 20d_avg_amount`. Keeps direction,
fixes scale.

### Group B: `gtja_*` price-volume products (~50 cols)

```
gtja_132:  std = 5.8e+08    abs_p99 = 2.5e+09    max_abs = 3.5e+10
gtja_095:  std = 2.7e+08    abs_p99 = 1.2e+09    max_abs = 2.3e+10
gtja_094:  std = 2.7e+08    abs_p99 = 1.0e+09    max_abs = 2.6e+10
gtja_134:  std = 9.7e+08    abs_p99 = 4.4e+07    max_abs = 1.0e+12   ← 1万亿!
gtja_178:  std = 7.9e+08    abs_p99 = 9.5e+06    max_abs = 1.5e+12
... (50 more)
```

Same root cause: GTJA Alpha191 formulas often involve `volume * price`
or `volume * volatility * window_sum` which natively scale to 10⁸..10¹².
Same fix: emit `log1p(abs) * sign` variants, or normalize by trailing
20-day window mean.

`gtja_134` and `gtja_178` having **max_abs > 10¹²** suggests these are
single bulk-block-trade outliers — winsorize at p99.5 upstream is enough.

## 🚫 10 dead-variance columns (std < 1e-6)

```
mfp_main_net_5d:   std = 8.2e-11
mfp_main_net_20d:  std = 2.0e-10
mfp_main_net_60d:  std = 4.5e-10
mfp_main_net_180d: std = 9.0e-10
mfp_md_net_60d:    std = 2.3e-10
mfp_sm_net_60d:    std = 5.4e-10

hm_net_5d:         std = 2.9e-07
hm_net_20d:        std = 5.9e-07
hm_net_60d:        std = 9.9e-07
sh_major_net_60d:  std = 3.0e-07
```

After cross-section z-score these become numerically unstable (1e-10 / 1e-8
≈ z=0.01..1 with random sign from float-noise). Effectively encoded as
random noise that the model can't generalize from.

The 6 `mfp_*` columns are already in HANDOFF v1's "deprecated_columns"
list. The 4 `hm_*` and `sh_major_net_60d` cases are NEW (not in deprecated):
likely the underlying table has `0` rows for >90% of stocks and only a
sparse minority has non-zero data. The cross-section ends up with std
near float epsilon.

**Action:** upstream verify whether `hm_net_*d` / `sh_major_net_60d`
should be (a) percentage-of-amount ratios (currently raw count?) or
(b) excluded from `alpha_panel` entirely until the underlying table
fills out.

## ⚠️ 1 mostly-null column already known: `senti_ths_hot_pct`

99.12% null. HANDOFF v1 already explained: `ths_hot` table starts
2024-01-15. Phase 26 pinned it out via include list, no action needed.

## Cross-list summary for upstream PR

```diff
# alpha_panel — drop these 3, factor library produces all-null:
- alpha_029
- alpha_031
- gtja_143

# alpha_panel — these have inf/overflow; clip or rewrite formula:
* alpha_022, alpha_026, alpha_044, alpha_045, alpha_083
* gtja_005, gtja_017 (worst — abs_p99 = 10^37), gtja_062, gtja_104,
  gtja_114, gtja_164, gtja_190, gtja_191

# alpha_panel — emit normalized variants for these scale-explosion cols:
* mf_net_{1d,3d,5d,10d,20d,60d,accel_5_20}  (yuan scale)
* gtja_{011,043,060,070,081,084,094,095,097,100,111,132,134,155,164,
        171,178,180,181,134,...} (~50 cols, price*volume products)

# alpha_panel — verify or drop:
* hm_net_{5d,20d,60d}      (std ~ 1e-7, possibly miscategorized as raw count)
* sh_major_net_60d         (std ~ 3e-7, same suspicion)

# Optional: validate after fix
* Re-export single trade_date (e.g. 2026-04-24), compare std distribution
  per column to existing data — should look log-normal-ish, not 10^9 spikes.
```

## Cyq backfill regime difference (already documented)

`cyq_cost_distance.std`: train 0.22 vs OOS 0.07 = **3.15× ratio**.
HANDOFF v1.2 acknowledges this is real market regime, not a bug. RL
side accepts; possible mitigations on RL side:

- Winsorize: `cost_distance.clip(-0.3, +0.3)` before z-score
- Drop fillna(0), let encoder learn the 0.86% NaN mask
- Train window restriction to 2024 only (single-regime, but halves data)

Will defer to whether 26D shows the v1.2 panel + 23A's full 353-col config
+ tech_panel actually outperforms 23A baseline. If 26D underperforms,
revisit.

## Reproduce

```python
.venv/Scripts/python.exe -c "
import polars as pl, numpy as np
df = pl.read_parquet('data/factor_panel_phase26d_2023_2026.parquet')

# 100% null
for c in df.columns:
    if df[c].null_count() == len(df):
        print(f'{c}: 100% null')

# Inf detection
for c in df.columns:
    s = df[c].drop_nulls().to_numpy()
    if np.issubdtype(s.dtype, np.floating):
        n_inf = int(np.isinf(s).sum())
        if n_inf > 0:
            print(f'{c}: {n_inf} inf cells')
"
```

Expected (2026-05-07 panel):
- 3 cols 100% null
- 20 cols with inf (13 heavy + 7 light)
- 10 cols std < 1e-6
