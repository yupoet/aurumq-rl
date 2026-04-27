-- example_query.sql
-- =============================================================================
-- Reference SQL template for `scripts/export_factor_panel.py`.
--
-- This produces a wide-format result that satisfies the AurumQ-RL Parquet
-- contract (see docs/SCHEMA.md and src/aurumq_rl/data_loader.py).
--
-- Placeholders:
--   :start_date   inclusive lower bound on trade_date (YYYY-MM-DD)
--   :end_date     inclusive upper bound on trade_date (YYYY-MM-DD)
--
-- The export script binds these via psycopg2 named parameters. Do NOT
-- replace them with literal dates — the script needs them for templating.
--
-- All table names are generic (`public.*`). Adapt them to whatever schema
-- your warehouse uses. The JOIN structure here assumes one row per
-- (ts_code, trade_date) per feature table; sentiment/market features are
-- panel-wide (one row per trade_date) and broadcast via a date-only JOIN.
--
-- Run the export with:
--   python scripts/export_factor_panel.py \
--       --pg-url postgresql://user:pass@host/db \
--       --start-date 2023-01-01 --end-date 2025-06-30 \
--       --sql-file docs/example_query.sql \
--       --out data/factor_panel.parquet
-- =============================================================================

SELECT
    -------------------------------------------------------------------------
    -- Required columns (REQUIRED by aurumq_rl.data_loader.REQUIRED_COLUMNS)
    -------------------------------------------------------------------------
    q.trade_date,
    q.ts_code,
    q.close,
    q.pct_chg,           -- expected as decimal (+10% = 0.10)
    q.vol,               -- 0 ⇒ suspended

    -------------------------------------------------------------------------
    -- Optional metadata columns (used if present, defaulted otherwise)
    -------------------------------------------------------------------------
    sb.name,
    sb.is_st,
    sb.days_since_ipo,
    sb.industry_code,

    -------------------------------------------------------------------------
    -- Factor columns. Recognized by prefix:
    --   alpha_ / mf_ / hm_ / hk_ / inst_ / mg_ /
    --   cyq_   / senti_ / sh_ / fund_ / ind_ / mkt_
    -- Add or remove as your warehouse provides.
    -------------------------------------------------------------------------

    -- alpha101 family (subset shown)
    a.alpha_001,
    a.alpha_002,
    a.alpha_003,
    a.alpha_004,
    a.alpha_005,

    -- main-force capital flow (mf_*)
    mf.mf_super_large_net,
    mf.mf_large_net,
    mf.mf_medium_net,
    mf.mf_small_net,

    -- hot-money seats (hm_*)
    hm.hm_seat_count,
    hm.hm_net_amount,

    -- northbound capital (hk_*)
    hk.hk_holding_pct,
    hk.hk_holding_pct_chg_5d,

    -- institutional desk activity (inst_*)
    inst.inst_net_buy,
    inst.inst_appearance_count,

    -- margin trading (mg_*)
    mg.mg_balance,
    mg.mg_balance_pct_chg_5d,

    -- chip distribution (cyq_*)
    cyq.cyq_winner_rate,
    cyq.cyq_cost_5d,

    -- limit-up sentiment (senti_*) — panel-wide, broadcast by date
    senti.senti_limit_up_count,
    senti.senti_limit_up_streak,

    -- shareholders (sh_*)
    sh.sh_holder_count_qoq,
    sh.sh_top10_chg,

    -- fundamentals (fund_*)
    f.fund_pe_ttm,
    f.fund_pb,
    f.fund_roe,
    f.fund_revenue_yoy,

    -- industry relative strength (ind_*) — joined via industry_code
    i.ind_relative_strength,
    i.ind_momentum_20d,

    -- market regime (mkt_*) — panel-wide
    m.mkt_regime,
    m.mkt_breadth_above_ma60

FROM public.daily_quotes q

-- Static metadata (name / IPO date / industry / ST flag)
LEFT JOIN public.stock_basic sb
    ON sb.ts_code = q.ts_code

-- Pre-computed alpha101 factors keyed by (ts_code, trade_date)
LEFT JOIN public.alpha_factors a
    ON a.ts_code    = q.ts_code
   AND a.trade_date = q.trade_date

-- Main-force flow features (per stock per day)
LEFT JOIN public.moneyflow_features mf
    ON mf.ts_code    = q.ts_code
   AND mf.trade_date = q.trade_date

-- Hot-money seat features (per stock per day)
LEFT JOIN public.hot_money_features hm
    ON hm.ts_code    = q.ts_code
   AND hm.trade_date = q.trade_date

-- Northbound capital features (per stock per day)
LEFT JOIN public.northbound_features hk
    ON hk.ts_code    = q.ts_code
   AND hk.trade_date = q.trade_date

-- Institutional desk features (per stock per day; sparse — most rows NULL)
LEFT JOIN public.institutional_features inst
    ON inst.ts_code    = q.ts_code
   AND inst.trade_date = q.trade_date

-- Margin trading features (per stock per day)
LEFT JOIN public.margin_features mg
    ON mg.ts_code    = q.ts_code
   AND mg.trade_date = q.trade_date

-- Chip distribution features (per stock per day)
LEFT JOIN public.chip_features cyq
    ON cyq.ts_code    = q.ts_code
   AND cyq.trade_date = q.trade_date

-- Limit-up sentiment (panel-wide → broadcast by date)
LEFT JOIN public.sentiment_features senti
    ON senti.trade_date = q.trade_date

-- Shareholder data (per stock per day; usually carried-forward)
LEFT JOIN public.shareholder_features sh
    ON sh.ts_code    = q.ts_code
   AND sh.trade_date = q.trade_date

-- Fundamentals (per stock per day; usually carried-forward from latest report)
LEFT JOIN public.fundamental_features f
    ON f.ts_code    = q.ts_code
   AND f.trade_date = q.trade_date

-- Industry features keyed by (industry_code, trade_date)
LEFT JOIN public.industry_features i
    ON i.industry_code = sb.industry_code
   AND i.trade_date    = q.trade_date

-- Market regime (panel-wide → broadcast by date)
LEFT JOIN public.market_features m
    ON m.trade_date = q.trade_date

WHERE q.trade_date >= :start_date
  AND q.trade_date <= :end_date

ORDER BY q.trade_date, q.ts_code
;
