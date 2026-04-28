# Factors

> Factor naming convention, normalisation policy, and how the loader
> consumes factors. **This project does not compute factors** — it only
> consumes them.

## 1. The 12 prefixes

`data_loader.py` recognises factors by **column prefix**. The canonical list
is fixed (changing it is a contract break) and lives in
`FACTOR_COL_PREFIXES`:

| Prefix | Domain | Suggested dim | Typical computation approach |
|---|---|---|---|
| `alpha_*` | alpha101-style quant-volume | 16-64 | OHLCV-derived rank/regression formulas (Kakushadze 2016). |
| `mf_*` | Main-force capital flow | 4-12 | Rolling-window net inflow of large/super-large orders. |
| `hm_*` | Hot-money seats | 3-6 | Daily seat-level netflow for famous speculator desks. |
| `hk_*` | Northbound holdings | 2-4 | Day-over-day shareholding deltas from cross-border flow. |
| `inst_*` | Institutional flow | 2-3 | Net buy from institutional seats on the limit-up/down list. |
| `mg_*` | Margin trading | 2-3 | Margin-buy ratio, financing balance changes. |
| `cyq_*` | Chip distribution | 2-3 | Cost-distribution profile, support-resistance proxies. |
| `senti_*` | Limit-up sentiment | 2-3 | Counts of consecutive limit-ups, hot-board strength. |
| `sh_*` | Shareholder dynamics | 2-3 | Number-of-shareholders deltas, large-holder transactions. |
| `fund_*` | Fundamentals | 3-6 | PE / PB / ROE / revenue growth, point-in-time aware. |
| `ind_*` | Industry relative | 2-3 | Stock return − industry-index return, rolling rank. |
| `mkt_*` | Market regime | 2-3 | Index-level features: trend, volatility, breadth. |
| `gtja_*` | 国泰君安 Alpha191 短周期量价因子 | 191 | 日频 OHLCV + vwap + amount (+ benchmark for ~10 factors) |

**Suggested dimensions** are starting points, not hard requirements. You can
ship 200 alpha columns or zero `mf_*` columns; the loader handles both.

### A note on out-of-scope semantics

We deliberately do not specify the formula behind each factor. Implementations
vary: some teams use 60-day rolling windows, others use 252-day; some lag
fundamentals by 45 days, others by 90. The RL framework treats all factors
as opaque cross-section signals — your job is to compute them correctly
and consistently.

## 2. Z-score normalisation

The loader applies **cross-sectional z-score** at load time, per (date,
factor):

```text
z[t, j, f] = (x[t, j, f] - mean(x[t, :, f])) / (std(x[t, :, f]) + 1e-8)
```

Why per-date and per-factor:

* **Factor-wise**: the magnitude of `alpha_010` (a price-derived ratio) and
  `fund_pe` (an earnings ratio) differ by orders of magnitude. Z-scoring
  per factor makes them comparable.
* **Per-date**: market volatility is non-stationary. Normalising within a
  cross-section avoids the model learning to "predict the regime" from raw
  scale alone.

The `_MIN_STD` epsilon guards against division by zero on dates where every
stock has the same factor value (rare, but it happens when a data pipeline
backfills with a constant).

### What the loader does NOT do

* No time-series normalisation. If you want rolling z-score by stock, do
  that in your factor pipeline before exporting.
* No winsorisation. If your raw factors have extreme outliers (PE > 1000),
  clip them before exporting; the loader will faithfully pass them through.
* No NaN imputation beyond the pipeline below.

## 3. NaN handling policy

The loader's per-row ingestion treats `None` / missing values as **0 after
z-scoring** (default of `np.zeros`). This means:

* During z-scoring, `np.nanmean` / `np.nanstd` are used so a single NaN in
  a cross-section does not poison the whole row.
* In the final 3D array, any cell that was originally NaN ends up at the
  cross-section mean (`(0 - mean) / std = -mean/std`, but since we use the
  pre-fill value of 0 in the array, you effectively get a "neutral" signal).

If a factor has many NaNs (e.g. fundamental data not yet released for a new
listing), the model will see those positions as near-neutral and learn to
weight them down. You should still aim for low NaN density (<5%) to keep
the signal strong.

## 4. `n_factors` selection

`StockPickingConfig.n_factors` controls how many factor columns enter the
observation. The loader:

1. Discovers all matching columns across all prefixes via
   `discover_factor_columns(df)`.
2. Sorts them alphabetically.
3. Truncates to the first `n_factors`.

This means the **alphabetical ordering of column names matters**. Our
convention — `alpha_000`, `alpha_001`, ..., `mf_000`, `mf_001`, ... — keeps
factors of the same prefix grouped, with prefixes themselves ordered
alphabetically (`alpha_*` before `fund_*` before `hk_*`, etc.).

If you want a particular subset of factors (e.g. only fundamentals), drop
the unwanted columns at *export time* rather than relying on the truncation.
Explicit is better than implicit.

## 5. Universe filter modes

The loader's `filter_universe()` supports five modes via the
`UniverseFilter` enum:

| Mode | What it keeps | Notes |
|---|---|---|
| `ALL_A` | Everything | No filtering — useful for back-testing the factor-only universe. |
| `MAIN_BOARD_NON_ST` | SH/SZ main board, non-ST | **Default**. Excludes ChiNext (300/301), STAR (688), BSE (8x/4x), and ST/*ST/退. |
| `HS300` | Approximated as main-board (heuristic) | Without explicit index-membership data, falls back to main-board. Plug your own index data via a separate filter step in production. |
| `ZZ500` | Approximated as main-board | Same caveat as HS300. |
| `ZZ1000` | Approximated as main-board | Same caveat. |

The default `MAIN_BOARD_NON_ST` corresponds to ~3500 stocks at the time of
writing. ChiNext / STAR / BSE are excluded by default because their price
limits and listing-day rules differ from the main board, and the env-level
constraints are tuned for the main board.

### ST detection

ST detection is purely name-based: the regex `r"\*?ST|退"` is applied to the
`name` column. If the `name` column is absent, ST filtering is silently
skipped. We trust the data pipeline to keep names current; we do **not**
maintain an internal list of ST tickers.

## 6. Data contract recap

Every Parquet must contain these required columns:

| Column | Type | Semantics |
|---|---|---|
| `ts_code` | str | Industry-standard format `XXXXXX.SH/SZ/BJ`. |
| `trade_date` | date | A single trading day. |
| `close` | float | Adjusted close price. |
| `pct_chg` | float | Daily percent change as a **decimal** (`+10% = 0.10`). |
| `vol` | float | Volume. `0` is treated as suspended. |

Optional but strongly recommended:

| Column | Type | Used by |
|---|---|---|
| `is_st` | bool | Trading mask. |
| `days_since_ipo` | int | New-stock protection (60-day default). |
| `industry_code` | int | Industry-cap constraint in env. |
| `name` | str | ST detection in `filter_universe`. |

## 7. Factor computation is your responsibility

The README and this document repeat this point intentionally: AurumQ-RL is
deliberately small. We do not bundle alpha101 implementations, rolling
window helpers, or point-in-time fundamental joins. There are excellent
libraries for that (e.g. open-source alpha101 ports) — pick one, run it
in your pipeline, and write the output to a Parquet with the column
conventions documented above. Then the RL stack just works.

If you want a reference shape: `scripts/generate_synthetic.py` produces a
fully-conforming Parquet with **synthetic factor distributions** — useful
for smoke tests, CI, and for debugging your own loader by comparing schemas.
