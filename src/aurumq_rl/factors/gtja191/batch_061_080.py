"""GTJA-191 factor library — batch 061 through 080.

Translated from Daic115/alpha191 (formula reference, no code vendored).
"""

from __future__ import annotations

import polars as pl

from aurumq_rl.factors.registry import FactorEntry, register_gtja191

from ._ops import (
    abs_,
    corr,
    decay_linear,
    delay,
    delta,
    mean,
    rank,
    sma,
    std_,
    sum_,
    ts_max,
    ts_min,
    ts_rank,
)

# ---------------------------------------------------------------------------
# gtja_061 — MAX of two rank-decay arms negated
# ---------------------------------------------------------------------------


def gtja_061(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #061 — Max of rank(decay-VWAP-delta), rank(decay-rank-corr).

    Guotai Junan Formula
    --------------------
        MAX(RANK(DECAYLINEAR(DELTA(VWAP, 1), 12)),
            RANK(DECAYLINEAR(RANK(CORR(LOW, MEAN(V, 80), 8)), 17))) * -1

    Daic115 omits the `* -1`. We follow Daic115 (no negation).

    Required panel columns: ``vwap``, ``low``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volume_price``
    """
    arm1_inner = decay_linear(delta(pl.col("vwap"), 1), 12)
    cor = corr(pl.col("low"), mean(pl.col("volume"), 80), 8)
    staged = panel.with_columns(
        arm1_inner.alias("__g061_a1"),
        cor.alias("__g061_c"),
    )
    staged = staged.with_columns(
        rank(pl.col("__g061_a1")).alias("__g061_r1"),
        rank(pl.col("__g061_c")).alias("__g061_rc"),
    )
    staged = staged.with_columns(decay_linear(pl.col("__g061_rc"), 17).alias("__g061_a2_inner"))
    staged = staged.with_columns(rank(pl.col("__g061_a2_inner")).alias("__g061_r2"))
    return staged.select(
        pl.max_horizontal(pl.col("__g061_r1"), pl.col("__g061_r2"))
        .alias("gtja_061")
        .cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# gtja_062 — -CORR(HIGH, RANK(turn proxy), 5)
# ---------------------------------------------------------------------------


def gtja_062(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #062 — -CORR(HIGH, RANK(TURN), 5) — turnover proxied by amount/cap.

    Guotai Junan Formula
    --------------------
        -CORR(HIGH, RANK(TURN), 5)

    Daic115 references ``data["turn"]`` (turnover_rate) which we don't
    have in the synthetic panel. Use ``amount / cap`` as turnover proxy.
    Reference parquet does NOT include gtja_062; reference test is
    skipped.

    Required panel columns: ``high``, ``amount``, ``cap``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volume_price``
    """
    turn_proxy = pl.col("amount") / pl.col("cap")
    staged = panel.with_columns(rank(turn_proxy).alias("__g062_rt"))
    return staged.select(
        (-1.0 * corr(pl.col("high"), pl.col("__g062_rt"), 5)).alias("gtja_062").cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# gtja_063 — RSI-style 6d
# ---------------------------------------------------------------------------


def gtja_063(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #063 — 6-day RSI: SMA(MAX(C-C-1, 0), 6, 1) / SMA(|C-C-1|, 6, 1) × 100.

    Required panel columns: ``vwap``, ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``momentum``
    """
    v = pl.col("vwap")
    diff = v - delay(v, 1)
    up = pl.max_horizontal(diff, pl.lit(0.0))
    return panel.select(
        (sma(up, 6, 1) / sma(abs_(diff), 6, 1) * 100.0).alias("gtja_063").cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# gtja_064 — MAX of two rank-decay corr arms (no negation)
# ---------------------------------------------------------------------------


def gtja_064(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #064 — Max of two rank(decay-corr) arms.

    Guotai Junan Formula
    --------------------
        MAX(
          RANK(DECAYLINEAR(CORR(RANK(VWAP), RANK(VOLUME), 4), 4)),
          RANK(DECAYLINEAR(MAX(CORR(RANK(CLOSE), RANK(MEAN(V, 60)), 4), 13), 14))
        ) * -1

    Daic115 omits the `* -1`; we follow.

    Required panel columns: ``vwap``, ``volume``, ``close``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volume_price``
    """
    staged = panel.with_columns(
        rank(pl.col("vwap")).alias("__g064_rw"),
        rank(pl.col("volume")).alias("__g064_rv"),
        rank(pl.col("close")).alias("__g064_rc"),
        rank(mean(pl.col("volume"), 60)).alias("__g064_rmv"),
    )
    staged = staged.with_columns(
        corr(pl.col("__g064_rw"), pl.col("__g064_rv"), 4).alias("__g064_c1"),
        corr(pl.col("__g064_rc"), pl.col("__g064_rmv"), 4).alias("__g064_c2"),
    )
    staged = staged.with_columns(
        decay_linear(pl.col("__g064_c1"), 4).alias("__g064_a1_inner"),
        ts_max(pl.col("__g064_c2"), 13).alias("__g064_a2_max"),
    )
    staged = staged.with_columns(decay_linear(pl.col("__g064_a2_max"), 14).alias("__g064_a2_inner"))
    staged = staged.with_columns(
        rank(pl.col("__g064_a1_inner")).alias("__g064_r1"),
        rank(pl.col("__g064_a2_inner")).alias("__g064_r2"),
    )
    return staged.select(
        pl.max_horizontal(pl.col("__g064_r1"), pl.col("__g064_r2"))
        .alias("gtja_064")
        .cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# gtja_065 — MEAN(C, 6) / C
# ---------------------------------------------------------------------------


def gtja_065(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #065 — MEAN(close, 6) / close.

    Required panel columns: ``close``, ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``mean_reversion``
    """
    c = pl.col("close")
    return panel.select((mean(c, 6) / c).alias("gtja_065").cast(pl.Float64)).to_series()


# ---------------------------------------------------------------------------
# gtja_066 — (C - MEAN(C, 6)) / MEAN(C, 6) × 100
# ---------------------------------------------------------------------------


def gtja_066(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #066 — (close - MA6) / MA6 × 100.

    Required panel columns: ``close``, ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``mean_reversion``
    """
    c = pl.col("close")
    m6 = mean(c, 6)
    return panel.select(((c - m6) / m6 * 100.0).alias("gtja_066").cast(pl.Float64)).to_series()


# ---------------------------------------------------------------------------
# gtja_067 — RSI-style 24d
# ---------------------------------------------------------------------------


def gtja_067(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #067 — 24-day RSI.

    Guotai Junan Formula
    --------------------
        SMA(MAX(C-C-1, 0), 24, 1) / SMA(|C-C-1|, 24, 1) × 100

    Required panel columns: ``close``, ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``momentum``
    """
    c = pl.col("close")
    diff = c - delay(c, 1)
    up = pl.max_horizontal(diff, pl.lit(0.0))
    return panel.select(
        (sma(up, 24, 1) / sma(abs_(diff), 24, 1) * 100.0).alias("gtja_067").cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# gtja_068 — SMA of mid-acceleration × range / volume
# ---------------------------------------------------------------------------


def gtja_068(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #068 — EWMA(15,2) of mid-price acceleration × (H-L)/V.

    Guotai Junan Formula
    --------------------
        SMA(((H+L)/2 - (DELAY(H,1)+DELAY(L,1))/2) * (H-L)/V, 15, 2)

    Required panel columns: ``high``, ``low``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``volume_price``
    """
    mid = (pl.col("high") + pl.col("low")) / 2.0
    mid_lag = (delay(pl.col("high"), 1) + delay(pl.col("low"), 1)) / 2.0
    inner = (mid - mid_lag) * (pl.col("high") - pl.col("low")) / pl.col("volume")
    return panel.select(sma(inner, 15, 2).alias("gtja_068").cast(pl.Float64)).to_series()


# ---------------------------------------------------------------------------
# gtja_069 — DTM/DBM 20-day asymmetric momentum
# ---------------------------------------------------------------------------


def gtja_069(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #069 — DTM/DBM 20-day asymmetric momentum ratio.

    Guotai Junan Formula
    --------------------
        DTM = (O <= DELAY(O,1)) ? 0 : MAX((H-O), (O-DELAY(O,1)))
        DBM = (O >= DELAY(O,1)) ? 0 : MAX((O-L), (O-DELAY(O,1)))
        S_DTM = SUM(DTM, 20); S_DBM = SUM(DBM, 20)
        S_DTM > S_DBM ? (S_DTM - S_DBM)/S_DTM
                       : S_DTM == S_DBM ? 0 : (S_DTM - S_DBM)/S_DBM

    Required panel columns: ``open``, ``high``, ``low``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``momentum``
    """
    o = pl.col("open")
    o_lag = delay(o, 1)
    dtm_inner = pl.max_horizontal(pl.col("high") - o, o - o_lag)
    dbm_inner = pl.max_horizontal(o - pl.col("low"), o - o_lag)
    dtm = pl.when(o <= o_lag).then(0.0).otherwise(dtm_inner)
    dbm = pl.when(o >= o_lag).then(0.0).otherwise(dbm_inner)
    s_dtm = sum_(dtm, 20)
    s_dbm = sum_(dbm, 20)
    expr = (
        pl.when(s_dtm > s_dbm)
        .then((s_dtm - s_dbm) / s_dtm)
        .otherwise(pl.when(s_dtm == s_dbm).then(0.0).otherwise((s_dtm - s_dbm) / s_dbm))
    )
    return panel.select(expr.alias("gtja_069").cast(pl.Float64)).to_series()


# ---------------------------------------------------------------------------
# gtja_070 — STD(amount, 6)
# ---------------------------------------------------------------------------


def gtja_070(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #070 — 6-day std of amount.

    Required panel columns: ``amount``, ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``volatility``
    """
    return panel.select(std_(pl.col("amount"), 6).alias("gtja_070").cast(pl.Float64)).to_series()


# ---------------------------------------------------------------------------
# gtja_071 — (C - MA24) / MA24 × 100
# ---------------------------------------------------------------------------


def gtja_071(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #071 — (close - MA24) / MA24 × 100.

    Required panel columns: ``close``, ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``mean_reversion``
    """
    c = pl.col("close")
    m = mean(c, 24)
    return panel.select(((c - m) / m * 100.0).alias("gtja_071").cast(pl.Float64)).to_series()


# ---------------------------------------------------------------------------
# gtja_072 — Smoothed Williams %R-style
# ---------------------------------------------------------------------------


def gtja_072(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #072 — SMA((TSMAX(H,6)-C)/(TSMAX(H,6)-TSMIN(L,6)) × 100, 15, 1).

    Required panel columns: ``high``, ``low``, ``close``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``mean_reversion``
    """
    hmax = ts_max(pl.col("high"), 6)
    lmin = ts_min(pl.col("low"), 6)
    raw = (hmax - pl.col("close")) / (hmax - lmin) * 100.0
    return panel.select(sma(raw, 15, 1).alias("gtja_072").cast(pl.Float64)).to_series()


# ---------------------------------------------------------------------------
# gtja_073 — TS-rank decay nested - rank decay corr
# ---------------------------------------------------------------------------


def gtja_073(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #073 — -TS_RANK(decay-decay-corr) - RANK(decay-corr-MA30).

    Guotai Junan Formula
    --------------------
        -1 * TS_RANK(DECAYLINEAR(DECAYLINEAR(CORR(C, V, 10), 16), 4), 5) -
        RANK(DECAYLINEAR(CORR(VWAP, MEAN(V, 30), 4), 3))

    Required panel columns: ``close``, ``volume``, ``vwap``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volume_price``
    """
    c1 = corr(pl.col("close"), pl.col("volume"), 10)
    arm1_inner = decay_linear(decay_linear(c1, 16), 4)
    arm1 = -1.0 * ts_rank(arm1_inner, 5)
    c2 = corr(pl.col("vwap"), mean(pl.col("volume"), 30), 4)
    arm2_inner = decay_linear(c2, 3)
    staged = panel.with_columns(arm2_inner.alias("__g073_a2_inner"))
    staged = staged.with_columns(rank(pl.col("__g073_a2_inner")).alias("__g073_r2"))
    return staged.select(
        (arm1 - pl.col("__g073_r2")).alias("gtja_073").cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# gtja_074 — Two rank-corr arms summed
# ---------------------------------------------------------------------------


def gtja_074(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #074 — Rank(corr(sum-weighted, sum-MA-V, 7)) + rank(corr(rank-VWAP, rank-V, 6)).

    Guotai Junan Formula
    --------------------
        RANK(CORR(SUM(L*0.35 + VWAP*0.65, 20), SUM(MEAN(V, 40), 20), 7)) +
        RANK(CORR(RANK(VWAP), RANK(VOLUME), 6))

    Required panel columns: ``low``, ``vwap``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``volume_price``
    """
    weighted = pl.col("low") * 0.35 + pl.col("vwap") * 0.65
    arm1_corr = corr(sum_(weighted, 20), sum_(mean(pl.col("volume"), 40), 20), 7)
    staged = panel.with_columns(
        arm1_corr.alias("__g074_c1"),
        rank(pl.col("vwap")).alias("__g074_rw"),
        rank(pl.col("volume")).alias("__g074_rv"),
    )
    staged = staged.with_columns(
        corr(pl.col("__g074_rw"), pl.col("__g074_rv"), 6).alias("__g074_c2")
    )
    staged = staged.with_columns(
        rank(pl.col("__g074_c1")).alias("__g074_r1"),
        rank(pl.col("__g074_c2")).alias("__g074_r2"),
    )
    return staged.select(
        (pl.col("__g074_r1") + pl.col("__g074_r2")).alias("gtja_074").cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# gtja_075 — Conditional count ratio (benchmark proxy = cross-section mean ret)
# ---------------------------------------------------------------------------


def gtja_075(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #075 — Conditional up-day count ratio vs benchmark down-days.

    Guotai Junan Formula
    --------------------
        COUNT(C > O & BENCH_C < BENCH_O, 50) / COUNT(BENCH_C < BENCH_O, 50)

    Daic115 substitutes a CS-mean return for the benchmark. We do the
    same: ``bench_ret = mean(returns)`` per trade_date, then `bench<0`
    is our "benchmark down" indicator.

    Required panel columns: ``close``, ``open``, ``returns``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``momentum``
    """
    cs_ret = pl.col("returns").mean().over("trade_date")
    bench_dn = cs_ret < 0.0
    cond_a = (pl.col("close") > pl.col("open")) & bench_dn
    cond_b = (pl.col("close") != pl.col("open")) & bench_dn
    a = sum_(cond_a.cast(pl.Float64), 50)
    b = sum_(cond_b.cast(pl.Float64), 50)
    return panel.select((a / b).alias("gtja_075").cast(pl.Float64)).to_series()


# ---------------------------------------------------------------------------
# gtja_076 — Volatility / Mean of |return|/V over 20d
# ---------------------------------------------------------------------------


def gtja_076(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #076 — STD(|ret|/V, 20) / MEAN(|ret|/V, 20).

    Required panel columns: ``close``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volatility``
    """
    c = pl.col("close")
    rel_ret_per_v = abs_(c / delay(c, 1) - 1.0) / pl.col("volume")
    return panel.select(
        (std_(rel_ret_per_v, 20) / mean(rel_ret_per_v, 20)).alias("gtja_076").cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# gtja_077 — MIN of two rank-decay arms
# ---------------------------------------------------------------------------


def gtja_077(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #077 — MIN of two rank(DECAYLINEAR(...)) arms.

    Guotai Junan Formula
    --------------------
        MIN(
          RANK(DECAYLINEAR(((H+L)/2 + H) - (VWAP + H), 20)),
          RANK(DECAYLINEAR(CORR((H+L)/2, MEAN(V, 40), 3), 6))
        )

    Required panel columns: ``high``, ``low``, ``vwap``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volume_price``
    """
    mid = (pl.col("high") + pl.col("low")) / 2.0
    inner1 = (mid + pl.col("high")) - (pl.col("vwap") + pl.col("high"))
    arm1_inner = decay_linear(inner1, 20)
    cor = corr(mid, mean(pl.col("volume"), 40), 3)
    arm2_inner = decay_linear(cor, 6)
    staged = panel.with_columns(
        arm1_inner.alias("__g077_a1"),
        arm2_inner.alias("__g077_a2"),
    )
    staged = staged.with_columns(
        rank(pl.col("__g077_a1")).alias("__g077_r1"),
        rank(pl.col("__g077_a2")).alias("__g077_r2"),
    )
    return staged.select(
        pl.min_horizontal(pl.col("__g077_r1"), pl.col("__g077_r2"))
        .alias("gtja_077")
        .cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# gtja_078 — CCI-style typical price
# ---------------------------------------------------------------------------


def gtja_078(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #078 — CCI-style typical price oscillator.

    Guotai Junan Formula
    --------------------
        ((H+L+C)/3 - MA((H+L+C)/3, 12)) /
        (0.015 * MEAN(|C - MEAN((H+L+C)/3, 12)|, 12))

    Required panel columns: ``high``, ``low``, ``close``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``mean_reversion``
    """
    typ = (pl.col("high") + pl.col("low") + pl.col("close")) / 3.0
    typ_ma = mean(typ, 12)
    expr = (typ - typ_ma) / (0.015 * mean(abs_(pl.col("close") - typ_ma), 12))
    return panel.select(expr.alias("gtja_078").cast(pl.Float64)).to_series()


# ---------------------------------------------------------------------------
# gtja_079 — RSI-style 12d
# ---------------------------------------------------------------------------


def gtja_079(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #079 — 12-day RSI.

    Guotai Junan Formula
    --------------------
        SMA(MAX(C-C-1, 0), 12, 1) / SMA(|C-C-1|, 12, 1) × 100

    Required panel columns: ``close``, ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``momentum``
    """
    c = pl.col("close")
    diff = c - delay(c, 1)
    up = pl.max_horizontal(diff, pl.lit(0.0))
    return panel.select(
        (sma(up, 12, 1) / sma(abs_(diff), 12, 1) * 100.0).alias("gtja_079").cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# gtja_080 — 5d % change of volume × 100
# ---------------------------------------------------------------------------


def gtja_080(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #080 — (V - DELAY(V, 5)) / DELAY(V, 5) × 100.

    Required panel columns: ``volume``, ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``volume_price``
    """
    v = pl.col("volume")
    v_lag = delay(v, 5)
    return panel.select(
        ((v - v_lag) / v_lag * 100.0).alias("gtja_080").cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_DOC_BASE = "docs/factor_library/gtja191"
_REF_BASE = "Guotai Junan 2017, '191 Alphas', via Daic115/alpha191 (formula only)"

_ENTRIES: list[FactorEntry] = [
    FactorEntry(
        id="gtja_061",
        impl=gtja_061,
        direction="reverse",
        category="volume_price",
        description="MAX of rank(decay-VWAP-delta), rank(decay-rank(corr(L, MA(V,80), 8)))",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_061.md",
    ),
    FactorEntry(
        id="gtja_062",
        impl=gtja_062,
        direction="reverse",
        category="volume_price",
        description="-CORR(HIGH, RANK(turn proxy=amount/cap), 5)",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_062.md",
        quality_flag=1,
    ),
    FactorEntry(
        id="gtja_063",
        impl=gtja_063,
        direction="normal",
        category="momentum",
        description="6-day RSI (vwap-anchored)",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_063.md",
    ),
    FactorEntry(
        id="gtja_064",
        impl=gtja_064,
        direction="reverse",
        category="volume_price",
        description="MAX of two rank(decay-corr) arms (vwap-vol, close-MA-vol)",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_064.md",
    ),
    FactorEntry(
        id="gtja_065",
        impl=gtja_065,
        direction="reverse",
        category="mean_reversion",
        description="MEAN(close, 6) / close",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_065.md",
    ),
    FactorEntry(
        id="gtja_066",
        impl=gtja_066,
        direction="reverse",
        category="mean_reversion",
        description="(close - MA6) / MA6 × 100",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_066.md",
    ),
    FactorEntry(
        id="gtja_067",
        impl=gtja_067,
        direction="normal",
        category="momentum",
        description="24-day RSI",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_067.md",
    ),
    FactorEntry(
        id="gtja_068",
        impl=gtja_068,
        direction="normal",
        category="volume_price",
        description="EWMA(15,2) of mid-price acceleration × (H-L)/V",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_068.md",
    ),
    FactorEntry(
        id="gtja_069",
        impl=gtja_069,
        direction="normal",
        category="momentum",
        description="DTM/DBM 20-day asymmetric momentum ratio",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_069.md",
        quality_flag=1,
    ),
    FactorEntry(
        id="gtja_070",
        impl=gtja_070,
        direction="normal",
        category="volatility",
        description="6-day std of amount",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_070.md",
    ),
    FactorEntry(
        id="gtja_071",
        impl=gtja_071,
        direction="reverse",
        category="mean_reversion",
        description="(close - MA24) / MA24 × 100",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_071.md",
    ),
    FactorEntry(
        id="gtja_072",
        impl=gtja_072,
        direction="reverse",
        category="mean_reversion",
        description="SMA((TSMAX(H,6)-C)/(TSMAX(H,6)-TSMIN(L,6)) × 100, 15, 1)",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_072.md",
    ),
    FactorEntry(
        id="gtja_073",
        impl=gtja_073,
        direction="reverse",
        category="volume_price",
        description="-TS_RANK(decay-decay-corr(C,V)) - RANK(decay-corr(VWAP, MA30(V)))",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_073.md",
        quality_flag=1,
    ),
    FactorEntry(
        id="gtja_074",
        impl=gtja_074,
        direction="normal",
        category="volume_price",
        description="Rank-corr-sum-weighted-prices + Rank-corr-rank(VWAP, V)",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_074.md",
    ),
    FactorEntry(
        id="gtja_075",
        impl=gtja_075,
        direction="normal",
        category="momentum",
        description="Up-day count ratio vs CS-mean-return-as-benchmark down-days (50d)",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_075.md",
    ),
    FactorEntry(
        id="gtja_076",
        impl=gtja_076,
        direction="reverse",
        category="volatility",
        description="STD(|ret|/V, 20) / MEAN(|ret|/V, 20)",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_076.md",
    ),
    FactorEntry(
        id="gtja_077",
        impl=gtja_077,
        direction="reverse",
        category="volume_price",
        description="MIN of two rank(decay) arms (synthetic-mid-vs-VWAP, mid-MA40V-corr)",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_077.md",
    ),
    FactorEntry(
        id="gtja_078",
        impl=gtja_078,
        direction="normal",
        category="mean_reversion",
        description="CCI-style typical-price oscillator (12d)",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_078.md",
    ),
    FactorEntry(
        id="gtja_079",
        impl=gtja_079,
        direction="normal",
        category="momentum",
        description="12-day RSI",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_079.md",
    ),
    FactorEntry(
        id="gtja_080",
        impl=gtja_080,
        direction="normal",
        category="volume_price",
        description="(V - DELAY(V, 5)) / DELAY(V, 5) × 100",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_080.md",
    ),
]

for _e in _ENTRIES:
    register_gtja191(_e)
