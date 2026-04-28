-- docs/example_query_with_index.sql
-- Reference factor_panel query with HS300 / ZZ500 membership flags.
-- Adapt schema names (hs300_constituents / zz500_constituents) to your warehouse.
--
-- Membership tables are expected to have at least (trade_date, ts_code) so the
-- join produces a per-row boolean. If your warehouse only carries the current
-- snapshot, replace the LEFT JOIN with the snapshot table and accept that the
-- flag will not reflect historical changes.

WITH base AS (
    SELECT
        d.trade_date,
        d.ts_code,
        d.close,
        d.pct_chg,
        d.vol,
        sb.name,
        sb.is_st,
        sb.days_since_ipo,
        sb.industry_code
    FROM daily_quotes d
    LEFT JOIN stock_basic sb USING (ts_code)
    WHERE d.trade_date >= :start_date
      AND d.trade_date <= :end_date
)
SELECT
    b.*,
    -- index membership (per trade_date)
    (h.ts_code IS NOT NULL) AS is_hs300,
    (z.ts_code IS NOT NULL) AS is_zz500,
    -- factor groups (extend with more LEFT JOINs as needed)
    a.alpha_001, a.alpha_002, a.alpha_003, a.alpha_004, a.alpha_005,
    a.alpha_006, a.alpha_007, a.alpha_008,
    mf.mf_001, mf.mf_002, mf.mf_003, mf.mf_004,
    hk.hk_001, hk.hk_002,
    fund.fund_001, fund.fund_002, fund.fund_003,
    ind.ind_001, ind.ind_002
FROM base b
LEFT JOIN hs300_constituents h
    ON h.trade_date = b.trade_date AND h.ts_code = b.ts_code
LEFT JOIN zz500_constituents z
    ON z.trade_date = b.trade_date AND z.ts_code = b.ts_code
LEFT JOIN alpha_factors a
    ON a.trade_date = b.trade_date AND a.ts_code = b.ts_code
LEFT JOIN moneyflow_features mf
    ON mf.trade_date = b.trade_date AND mf.ts_code = b.ts_code
LEFT JOIN northbound_features hk
    ON hk.trade_date = b.trade_date AND hk.ts_code = b.ts_code
LEFT JOIN fundamental_features fund
    ON fund.trade_date = b.trade_date AND fund.ts_code = b.ts_code
LEFT JOIN industry_features ind
    ON ind.trade_date = b.trade_date AND ind.ts_code = b.ts_code
ORDER BY b.trade_date, b.ts_code;
