# Data

> Parquet schema, conventions, and how to inspect or generate panels.

## Required + optional columns

| Column | Type | Required | Notes |
|---|---|---|---|
| `ts_code` | str | required | Industry-standard format `XXXXXX.SH/SZ/BJ` (synthetic data uses `SYN_NNNNN`). |
| `trade_date` | date | required | One row per (ts_code, trade_date). |
| `close` | float32 | required | Adjusted close price. |
| `pct_chg` | float32 | required | Daily change as a **decimal** (`+10% = 0.10`, not `10.0`). |
| `vol` | float32 | required | Trading volume. `0` is treated as suspended. |
| `is_st` | bool | optional | True if the stock is ST/*ST/退. |
| `days_since_ipo` | int32 | optional | Trading days since listing. |
| `industry_code` | int16 | optional | SW-1 industry code for the industry-cap constraint. |
| `name` | str | optional | Used by ST-detection regex when `is_st` is missing. |
| `is_hs300` | bool | optional | True if stock is a HS300 constituent on `trade_date`. Per-row, history-aware. Consumed by `--universe-filter hs300`. |
| `is_zz500` | bool | optional | True if stock is a CSI500 constituent on `trade_date`. Consumed by `--universe-filter zz500`. |
| `alpha_*`, `mf_*`, `hm_*`, `hk_*`, `inst_*`, `mg_*`, `cyq_*`, `senti_*`, `sh_*`, `fund_*`, `ind_*`, `mkt_*`, `gtja_*` | float32 | at-least-one | Factor columns; recognised by prefix. See `docs/FACTORS.md`. |

## Example schema (synthetic demo)

```text
ts_code:         String
trade_date:      Date
close:           Float32
pct_chg:         Float32
vol:             Float32
is_st:           Boolean
days_since_ipo:  Int32
industry_code:   Int16
name:            String
alpha_000..007:  Float32  (8 alpha factors)
mf_000..003:     Float32  (4 main-force factors)
hm_000..002:     Float32  (3 hot-money factors)
hk_000..001:     Float32  (2 northbound factors)
fund_000..002:   Float32  (3 fundamental factors)
ind_000..001:    Float32  (2 industry factors)
```

## Inspecting a Parquet

Use polars to peek at the schema and a few rows:

```python
import polars as pl

# Schema only — fast, doesn't load data
schema = pl.scan_parquet("data/factor_panel.parquet").collect_schema()
for col, dtype in schema.items():
    print(f"{col}: {dtype}")

# Quick sanity check
df = pl.read_parquet("data/factor_panel.parquet")
print("rows:", len(df))
print("dates:", df["trade_date"].n_unique())
print("stocks:", df["ts_code"].n_unique())
```

## Generating data

* `scripts/generate_synthetic.py` — produces a fully-conforming Parquet
  with synthetic factor distributions. Useful for smoke tests, CI, and
  for debugging your own loader. Output is < 10 MB by default.
* `scripts/export_factor_panel.py` — extracts a real factor panel from a
  PostgreSQL data warehouse using SQL templates. Configure with
  `--pg-url`, `--start`, `--end`, `--out`.

## File `data/synthetic_demo.parquet`

The demo file in this directory is synthetic. Stock codes look like
`SYN_00001` and do **not** correspond to any real ticker. It exists to
make `scripts/train.py --smoke-test`, the test suite, and the
`examples/quickstart.py` script runnable without external data.
