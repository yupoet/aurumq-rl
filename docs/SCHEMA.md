# Database Schema for `export_factor_panel.py`

> Reference schema for the data warehouse that feeds the AurumQ-RL training
> pipeline. The export script (`scripts/export_factor_panel.py`) reads from
> a PostgreSQL database and writes a Parquet file matching the AurumQ-RL
> input contract (see `src/aurumq_rl/data_loader.py`).

This document is **descriptive, not prescriptive**: AurumQ-RL only cares
about the shape of the resulting Parquet. How you compose that result
inside your warehouse is your choice. The view name `factor_panel_view` is
a convenient canonical interface but not required.

## Output contract (Parquet schema)

The Parquet file produced by the export script must satisfy the contract
declared in `aurumq_rl.data_loader`:

### Required columns

| Column        | SQL type        | Description                                      |
|---------------|-----------------|--------------------------------------------------|
| `ts_code`     | `TEXT`          | Stock code, e.g. `600519.SH`, `000001.SZ`.        |
| `trade_date`  | `DATE`          | Trading day.                                     |
| `close`       | `DOUBLE PRECISION` | Day close (adjusted or unadjusted, your choice — the RL env normalizes). |
| `pct_chg`     | `DOUBLE PRECISION` | Daily % change as a **decimal** (`+10% = 0.10`). |
| `vol`         | `DOUBLE PRECISION` | Volume in shares. `vol = 0` means suspended.     |

### Factor columns (at least one prefix required)

Columns matching any of the recognized prefixes are picked up automatically
by `aurumq_rl.data_loader.discover_factor_columns`. Prefixes are defined in
`FACTOR_COL_PREFIXES`:

| Prefix       | Domain (suggested)                                |
|--------------|---------------------------------------------------|
| `alpha_*`    | Quant / volume factors (alpha101 family).          |
| `mf_*`       | Main-force capital flow.                          |
| `hm_*`       | Hot-money seats.                                  |
| `hk_*`       | Northbound capital.                               |
| `inst_*`     | Institutional desk activity (limit-list).         |
| `mg_*`       | Margin trading.                                   |
| `cyq_*`      | Chip distribution.                                |
| `senti_*`    | Limit-up sentiment.                               |
| `sh_*`       | Shareholders / large-holder flows.                |
| `fund_*`     | Fundamentals (PE/PB/ROE/...).                     |
| `ind_*`      | Industry relative strength.                       |
| `mkt_*`      | Market regime / breadth.                          |
| `gtja_*`     | Guotai Junan Alpha191 short-period price-volume factors. |

You decide which prefixes to include. AurumQ-RL never errors on missing
factor groups — the model just sees those positions as zero.

### Optional columns (used if present)

| Column            | SQL type   | Description                                  |
|-------------------|------------|----------------------------------------------|
| `is_st`           | `BOOLEAN`  | True if ST/*ST/退市. Used by universe filter. |
| `days_since_ipo`  | `INTEGER`  | Trading days since IPO. Drives new-stock protection. |
| `industry_code`   | `INTEGER`  | SW level-1 industry code. Used for risk caps. |
| `name`            | `TEXT`     | Stock name (lets the loader detect ST/退 from text). |
| `is_hs300`        | `BOOLEAN`  | True if stock is a HS300 constituent on `trade_date`. Per-row, supports historical changes. Consumed by `--universe-filter hs300`. |
| `is_zz500`        | `BOOLEAN`  | True if stock is a CSI500 constituent on `trade_date`. Consumed by `--universe-filter zz500`. |

## Recommended source tables

This is illustrative — names will differ in your warehouse. Wire the JOINs
in your SQL template however your data is laid out.

| Logical name              | Granularity            | Typical columns                                                |
|---------------------------|------------------------|----------------------------------------------------------------|
| `daily_quotes`            | `(ts_code, trade_date)`| `open, high, low, close, vol, amount, pct_chg, adj_factor`      |
| `stock_basic`             | `ts_code`              | `name, list_date, industry, exchange, is_st`                    |
| `moneyflow`               | `(ts_code, trade_date)`| Buy/sell volumes by lot size; powers `mf_*`.                    |
| `hm_detail`               | `(ts_code, trade_date)`| Hot-money seat detail; powers `hm_*`.                           |
| `hk_hold`                 | `(ts_code, trade_date)`| Northbound holdings; powers `hk_*`.                             |
| `top_inst`                | `(ts_code, trade_date)`| Institutional limit-list; powers `inst_*`.                      |
| `margin_detail`           | `(ts_code, trade_date)`| Margin balances; powers `mg_*`.                                 |
| `cyq_perf`                | `(ts_code, trade_date)`| Chip distribution metrics; powers `cyq_*`.                      |
| `limit_up_sentiment`      | `trade_date` (panel)   | Daily limit-up board metrics; powers `senti_*`.                 |
| `shareholders`            | `(ts_code, end_date)`  | Holder count, top-10 changes; powers `sh_*`.                    |
| `fundamentals`            | `(ts_code, ann_date)`  | PE/PB/PS/PCF/ROE/ROA; powers `fund_*`.                          |
| `industry_index_daily`    | `(industry_code, date)`| Industry index level/return; powers `ind_*`.                    |
| `market_index_daily`      | `(index_code, date)`   | CSI/SSE indices; powers `mkt_*`.                                |

> Names like `daily_quotes` are placeholders. Replace with whatever schema
> you maintain in your own warehouse. AurumQ-RL has no opinion on this.

## Reference `factor_panel_view` (example)

Below is an *example* `CREATE VIEW` that composes a wide panel from
representative source tables. Adapt the JOIN conditions, column names, and
factor expressions to your warehouse.

```sql
-- example only: rewrite to match your warehouse
CREATE OR REPLACE VIEW factor_panel_view AS
SELECT
    -- required columns
    q.trade_date,
    q.ts_code,
    q.close,
    q.pct_chg,
    q.vol,

    -- optional columns
    sb.name,
    sb.is_st,
    sb.days_since_ipo,
    sb.industry_code,

    -- factor columns (subset shown — extend as needed)
    a.alpha_001,
    a.alpha_002,
    a.alpha_003,
    -- ... up to alpha_101

    mf.mf_super_large_net,
    mf.mf_large_net,
    mf.mf_medium_net,
    mf.mf_small_net,

    hm.hm_seat_count,
    hm.hm_net_amount,

    hk.hk_holding_pct,
    hk.hk_holding_pct_chg_5d,

    inst.inst_net_buy,
    inst.inst_appearance_count,

    mg.mg_balance,
    mg.mg_balance_pct_chg_5d,

    cyq.cyq_winner_rate,
    cyq.cyq_cost_5d,

    senti.senti_limit_up_count,
    senti.senti_limit_up_streak,

    sh.sh_holder_count_qoq,
    sh.sh_top10_chg,

    f.fund_pe_ttm,
    f.fund_pb,
    f.fund_roe,
    f.fund_revenue_yoy,

    i.ind_relative_strength,
    i.ind_momentum_20d,

    m.mkt_regime,
    m.mkt_breadth_above_ma60

FROM public.daily_quotes q
LEFT JOIN public.stock_basic           sb   USING (ts_code)
LEFT JOIN public.alpha_factors         a    ON a.ts_code = q.ts_code AND a.trade_date = q.trade_date
LEFT JOIN public.moneyflow_features    mf   ON mf.ts_code = q.ts_code AND mf.trade_date = q.trade_date
LEFT JOIN public.hot_money_features    hm   ON hm.ts_code = q.ts_code AND hm.trade_date = q.trade_date
LEFT JOIN public.northbound_features   hk   ON hk.ts_code = q.ts_code AND hk.trade_date = q.trade_date
LEFT JOIN public.institutional_features inst ON inst.ts_code = q.ts_code AND inst.trade_date = q.trade_date
LEFT JOIN public.margin_features       mg   ON mg.ts_code = q.ts_code AND mg.trade_date = q.trade_date
LEFT JOIN public.chip_features         cyq  ON cyq.ts_code = q.ts_code AND cyq.trade_date = q.trade_date
LEFT JOIN public.sentiment_features    senti ON senti.trade_date = q.trade_date
LEFT JOIN public.shareholder_features  sh   ON sh.ts_code = q.ts_code AND sh.trade_date = q.trade_date
LEFT JOIN public.fundamental_features  f    ON f.ts_code = q.ts_code AND f.trade_date = q.trade_date
LEFT JOIN public.industry_features     i    ON i.industry_code = sb.industry_code AND i.trade_date = q.trade_date
LEFT JOIN public.market_features       m    ON m.trade_date = q.trade_date
;
```

Indexes that materially help the export:

```sql
-- example only
CREATE INDEX IF NOT EXISTS idx_daily_quotes_date_code
    ON public.daily_quotes (trade_date, ts_code);
CREATE INDEX IF NOT EXISTS idx_alpha_factors_date_code
    ON public.alpha_factors (trade_date, ts_code);
-- ... repeat for each (date, code) feature table
```

## Universe filtering

The export script can apply a universe filter via `--universe-filter`. It
wraps your query with an outer `WHERE`, so the filter is independent of how
the inner query is composed.

| Mode                | Predicate (illustrative)                                                                 |
|---------------------|------------------------------------------------------------------------------------------|
| `all`               | No filter.                                                                               |
| `main_board`        | SH/SZ main board only (`60[0135]xxxx.SH` or `00[0123]xxxx.SZ`).                          |
| `main_board_non_st` | Main board minus rows where `is_st = TRUE` (default).                                    |
| `exclude_bse`       | Drops `.BJ` (Beijing Stock Exchange) only.                                               |

Equivalent SQL fragments you can inline if you prefer doing it in the view:

```sql
-- exclude BSE entirely
ts_code NOT LIKE '%.BJ'

-- main board only
(ts_code ~ '^60[0135][0-9]{3}\\.SH$' OR ts_code ~ '^00[0123][0-9]{3}\\.SZ$')

-- exclude STAR market and ChiNext (688 / 300 / 301)
ts_code !~ '^(688|30[01])[0-9]{3}\\.S[HZ]$'

-- exclude ST / *ST / 退
COALESCE(is_st, FALSE) = FALSE
```

## Operational notes

* **Server-side cursor**: the export script uses a named cursor with
  `itersize = --chunk-size` (default 100 000) to keep memory usage flat
  even for multi-year all-A exports.
* **Read-only session**: the connection is set to read-only — the script
  cannot accidentally mutate the warehouse.
* **Date placeholders**: SQL templates must use psycopg2 named-parameter
  style `:start_date` and `:end_date`. The script binds them automatically.
* **NULL handling**: missing factor values are perfectly fine. The RL
  pipeline z-scores cross-sectionally and treats NaN appropriately
  (`tests/test_factors_*.py` exercises this behavior).
* **Time zone**: A-share trading dates are naturally `Asia/Shanghai`-local
  (`DATE` is sufficient, no timestamp gymnastics required).
* **Sort order**: ordering by `(trade_date, ts_code)` in your SELECT is a
  best practice — keeps Parquet row groups well-clustered for downstream
  partition pruning.

## Compatibility checklist

Before running training, verify your Parquet:

- [ ] `ts_code`, `trade_date`, `close`, `pct_chg`, `vol` all present.
- [ ] At least one factor prefix from `FACTOR_COL_PREFIXES` is populated.
- [ ] `pct_chg` values are decimals (range roughly `[-0.20, +0.20]`), **not** percentages (`-20..+20`).
- [ ] `vol = 0` correctly marks suspended stocks.
- [ ] Universe filter has been applied either inside the SQL or via `--universe-filter`.

```python
# quick validation snippet
import polars as pl
df = pl.scan_parquet("data/factor_panel.parquet").collect()
print(df.shape)
print([c for c in df.columns if any(c.startswith(p) for p in (
    "alpha_", "mf_", "hm_", "hk_", "inst_", "mg_",
    "cyq_", "senti_", "sh_", "fund_", "ind_", "mkt_", "gtja_",
))])
```

## See also

* `scripts/export_factor_panel.py` — the export CLI.
* `docs/example_query.sql` — a complete, runnable SQL template.
* `src/aurumq_rl/data_loader.py` — the Parquet → numpy panel loader and
  the canonical source of `FACTOR_COL_PREFIXES` / `REQUIRED_COLUMNS`.
