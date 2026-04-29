"""GTJA-191 batch 101..120 (20 factors).

Each factor is implemented as a polars callable that takes the enriched
panel ``pl.DataFrame`` and returns a ``pl.Series`` aligned to the panel
rows. Self-registers into :data:`GTJA191_REGISTRY` at import time.

The Guotai Junan Alpha-191 formulas are sourced from the public report
(国泰君安证券 191 短周期价量因子, 2017). Numerical reference: Daic115/alpha191
(no LICENSE — we use it as a *formula reference only* and never vendor
its code; only the resulting reference parquet is committed).

Special factor in this batch
----------------------------

* ``gtja_116`` — Daic115's reference uses qlib's ``rolling_slope``.
  We implement it natively using :func:`regbeta` against
  :func:`sequence` (the GTJA paper formula is
  ``REGBETA(CLOSE, SEQUENCE, 20)``). Quality flag = 0.
"""
from __future__ import annotations

import polars as pl

from aurumq_rl.factors.registry import FactorEntry, register_gtja191

from ._ops import (
    TS_PART,
    corr,
    decay_linear,
    delay,
    delta,
    mean,
    rank,
    regbeta,
    sma,
    std_,
    sum_,
    ts_min,
    ts_rank,
)

# ---------------------------------------------------------------------------
# helper — boolean comparison cast to Float64 with NaN propagation, then *-1
# ---------------------------------------------------------------------------


def _bool_lt_signed(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    """``(a < b) * -1`` — Daic115 returns -1/0 with null where either is null."""
    return (
        pl.when(a.is_null() | b.is_null())
        .then(None)
        .otherwise((a < b).cast(pl.Float64) * -1.0)
        .cast(pl.Float64)
    )


# ---------------------------------------------------------------------------
# gtja_101
# ---------------------------------------------------------------------------


def gtja_101(panel: pl.DataFrame) -> pl.Series:
    """GTJA #101 — VWAP-volume corr ranked < volume-mean corr ranked.

    Guotai Junan Formula
    --------------------
        ((RANK(CORR(CLOSE, SUM(MEAN(VOLUME, 30), 37), 15)) <
          RANK(CORR(RANK(((HIGH * 0.1) + (VWAP * 0.9))),
                    RANK(VOLUME), 11))) * -1)

    Polars Implementation Notes
    ---------------------------
    Two-stage ``with_columns`` to materialise the per-stock CORRs before
    cross-section ranking, since polars cannot mix CS+TS partitions in a
    single expression.

    Direction: ``reverse`` (binary -1/0 — multiply by -1 in spec).
    Category: ``correlation``.
    """
    df = panel.with_columns(
        [
            mean(pl.col("volume"), 30).alias("__mv30"),
            rank(pl.col("high") * 0.1 + pl.col("vwap") * 0.9).alias("__rp"),
            rank(pl.col("volume")).alias("__rv"),
        ]
    )
    df = df.with_columns(
        [
            sum_(pl.col("__mv30"), 37).alias("__smv30_37"),
        ]
    )
    df = df.with_columns(
        [
            corr(pl.col("close"), pl.col("__smv30_37"), 15).alias("__c1"),
            corr(pl.col("__rp"), pl.col("__rv"), 11).alias("__c2"),
        ]
    )
    df = df.with_columns(
        [
            rank(pl.col("__c1")).alias("__r1"),
            rank(pl.col("__c2")).alias("__r2"),
        ]
    )
    return df.select(_bool_lt_signed(pl.col("__r1"), pl.col("__r2")).alias("gtja_101")).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_101",
        impl=gtja_101,
        direction="reverse",
        category="correlation",
        description="VWAP-volume corr ranked < volume-mean corr ranked",
        references=("Guotai Junan 191 short-period factor report, 2017",),
    )
)


# ---------------------------------------------------------------------------
# gtja_102
# ---------------------------------------------------------------------------


def gtja_102(panel: pl.DataFrame) -> pl.Series:
    """GTJA #102 — Volume RSI: SMA(MAX(dV,0))/SMA(|dV|).

    Guotai Junan Formula
    --------------------
        SMA(MAX(VOLUME-DELAY(VOLUME,1),0),6,1)/
        SMA(ABS(VOLUME-DELAY(VOLUME,1)),6,1)*100
    """
    dv = pl.col("volume") - delay(pl.col("volume"), 1)
    # Preserve null at first row so EWMA does not see a synthetic 0.
    pos = pl.when(dv.is_null()).then(None).when(dv > 0).then(dv).otherwise(0.0)
    num = sma(pos, 6, 1)
    den = sma(dv.abs(), 6, 1)
    expr = (num / den * 100.0).alias("gtja_102")
    return panel.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_102",
        impl=gtja_102,
        direction="normal",
        category="volume",
        description="Volume RSI — SMA(max(dV,0))/SMA(|dV|)*100",
    )
)


# ---------------------------------------------------------------------------
# gtja_103
# ---------------------------------------------------------------------------


def gtja_103(panel: pl.DataFrame) -> pl.Series:
    """GTJA #103 — (20-LOWDAY(LOW,20))/20*100 — recency of recent low.

    Implemented via :func:`ts_min` + per-stock back-search using a
    closed-form: distance-to-min as ``20 - argmin``. We use ``regbeta``-
    style trick: Daic115 has a slow ``LOWDAY`` here. We fall back to the
    operator's slow path :func:`_ops.lowday`.
    """
    from ._ops import lowday

    expr = ((20.0 - lowday(pl.col("low"), 20)) / 20.0 * 100.0).alias("gtja_103")
    return panel.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_103",
        impl=gtja_103,
        direction="normal",
        category="momentum",
        description="(20-LOWDAY(LOW,20))/20*100 — recency-of-low oscillator",
    )
)


# ---------------------------------------------------------------------------
# gtja_104
# ---------------------------------------------------------------------------


def gtja_104(panel: pl.DataFrame) -> pl.Series:
    """GTJA #104 — -1 * DELTA(CORR(HIGH,VOL,5),5) * RANK(STD(CLOSE,20)).

    Guotai Junan Formula
    --------------------
        -1 * (DELTA(CORR(HIGH, VOLUME, 5), 5) * RANK(STD(CLOSE, 20)))
    """
    df = panel.with_columns(
        [
            corr(pl.col("high"), pl.col("volume"), 5).alias("__c"),
            std_(pl.col("close"), 20).alias("__s"),
        ]
    )
    df = df.with_columns(
        [
            delta(pl.col("__c"), 5).alias("__dc"),
            rank(pl.col("__s")).alias("__rs"),
        ]
    )
    return df.select((-1.0 * pl.col("__dc") * pl.col("__rs")).alias("gtja_104")).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_104",
        impl=gtja_104,
        direction="reverse",
        category="correlation",
        description="-1 * delta(corr(high,vol,5),5) * rank(std(close,20))",
    )
)


# ---------------------------------------------------------------------------
# gtja_105
# ---------------------------------------------------------------------------


def gtja_105(panel: pl.DataFrame) -> pl.Series:
    """GTJA #105 — -1 * CORR(RANK(OPEN), RANK(VOLUME), 10)."""
    df = panel.with_columns(
        [rank(pl.col("open")).alias("__ro"), rank(pl.col("volume")).alias("__rv")]
    )
    return df.select((-1.0 * corr(pl.col("__ro"), pl.col("__rv"), 10)).alias("gtja_105")).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_105",
        impl=gtja_105,
        direction="reverse",
        category="correlation",
        description="-1 * corr(rank(open), rank(volume), 10)",
    )
)


# ---------------------------------------------------------------------------
# gtja_106
# ---------------------------------------------------------------------------


def gtja_106(panel: pl.DataFrame) -> pl.Series:
    """GTJA #106 — CLOSE - DELAY(CLOSE, 20). Pure 20-day momentum."""
    expr = (pl.col("close") - delay(pl.col("close"), 20)).alias("gtja_106")
    return panel.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_106",
        impl=gtja_106,
        direction="normal",
        category="momentum",
        description="20-day close momentum",
    )
)


# ---------------------------------------------------------------------------
# gtja_107
# ---------------------------------------------------------------------------


def gtja_107(panel: pl.DataFrame) -> pl.Series:
    """GTJA #107 — Triple-rank gap product across O-H/O-C/O-L.

    Guotai Junan Formula
    --------------------
        ((-1 * RANK(OPEN - DELAY(HIGH, 1))) *
          RANK(OPEN - DELAY(CLOSE, 1))) *
          RANK(OPEN - DELAY(LOW, 1))
    """
    df = panel.with_columns(
        [
            (pl.col("open") - delay(pl.col("high"), 1)).alias("__a"),
            (pl.col("open") - delay(pl.col("close"), 1)).alias("__b"),
            (pl.col("open") - delay(pl.col("low"), 1)).alias("__c"),
        ]
    )
    return df.select(
        (
            -1.0
            * rank(pl.col("__a"))
            * rank(pl.col("__b"))
            * rank(pl.col("__c"))
        ).alias("gtja_107")
    ).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_107",
        impl=gtja_107,
        direction="reverse",
        category="momentum",
        description="-rank(o-prev_h)*rank(o-prev_c)*rank(o-prev_l)",
    )
)


# ---------------------------------------------------------------------------
# gtja_108
# ---------------------------------------------------------------------------


def gtja_108(panel: pl.DataFrame) -> pl.Series:
    """GTJA #108 — RANK(HIGH - MIN(HIGH,2)) ^ RANK(CORR(VWAP, MA(V,120),6))) * -1.

    The XOR-looking caret in the paper is exponentiation in Daic115's
    pandas reference (``a ** b``); we follow that.
    """
    df = panel.with_columns(
        [
            (pl.col("high") - ts_min(pl.col("high"), 2)).alias("__d"),
            corr(pl.col("vwap"), mean(pl.col("volume"), 120), 6).alias("__c"),
        ]
    )
    df = df.with_columns(
        [
            rank(pl.col("__d")).alias("__rd"),
            rank(pl.col("__c")).alias("__rc"),
        ]
    )
    expr = (pl.col("__rd").pow(pl.col("__rc")) * -1.0).alias("gtja_108")
    return df.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_108",
        impl=gtja_108,
        direction="reverse",
        category="correlation",
        description="(rank(high-min(high,2)) ^ rank(corr(vwap, ma(v,120), 6))) * -1",
    )
)


# ---------------------------------------------------------------------------
# gtja_109
# ---------------------------------------------------------------------------


def gtja_109(panel: pl.DataFrame) -> pl.Series:
    """GTJA #109 — SMA(H-L,10,2) / SMA(SMA(H-L,10,2),10,2).

    Range-relative trend oscillator.
    """
    hl = pl.col("high") - pl.col("low")
    inner = sma(hl, 10, 2)
    expr = (inner / sma(inner, 10, 2)).alias("gtja_109")
    return panel.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_109",
        impl=gtja_109,
        direction="normal",
        category="volatility",
        description="SMA(H-L,10,2) / SMA(SMA(H-L,10,2),10,2)",
    )
)


# ---------------------------------------------------------------------------
# gtja_110
# ---------------------------------------------------------------------------


def gtja_110(panel: pl.DataFrame) -> pl.Series:
    """GTJA #110 — SUM(MAX(0,H-prev_C),20) / SUM(MAX(0,prev_C-L),20) * 100.

    Buying-pressure / selling-pressure ratio over 20 days.
    """
    pc = delay(pl.col("close"), 1)
    up = pl.when(pl.col("high") - pc > 0).then(pl.col("high") - pc).otherwise(0.0)
    dn = pl.when(pc - pl.col("low") > 0).then(pc - pl.col("low")).otherwise(0.0)
    expr = (sum_(up, 20) / sum_(dn, 20) * 100.0).alias("gtja_110")
    return panel.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_110",
        impl=gtja_110,
        direction="normal",
        category="momentum",
        description="20d up-move-vs-down-move pressure ratio",
    )
)


# ---------------------------------------------------------------------------
# gtja_111
# ---------------------------------------------------------------------------


def gtja_111(panel: pl.DataFrame) -> pl.Series:
    """GTJA #111 — VOL * intra-day position SMA differential."""
    rng = pl.col("high") - pl.col("low")
    pos = ((pl.col("close") - pl.col("low")) - (pl.col("high") - pl.col("close"))) / rng
    weighted = pl.col("volume") * pos
    expr = (sma(weighted, 11, 2) - sma(weighted, 4, 2)).alias("gtja_111")
    return panel.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_111",
        impl=gtja_111,
        direction="normal",
        category="volume",
        description="Volume * intraday position SMA-differential (11-4)",
    )
)


# ---------------------------------------------------------------------------
# gtja_112
# ---------------------------------------------------------------------------


def gtja_112(panel: pl.DataFrame) -> pl.Series:
    """GTJA #112 — Up-vs-down 12-day cumulative move ratio (CMO-style)."""
    dc = pl.col("close") - delay(pl.col("close"), 1)
    up = pl.when(dc > 0).then(dc).otherwise(0.0)
    dn = pl.when(dc < 0).then(-dc).otherwise(0.0)
    sup = sum_(up, 12)
    sdn = sum_(dn, 12)
    expr = ((sup - sdn) / (sup + sdn) * 100.0).alias("gtja_112")
    return panel.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_112",
        impl=gtja_112,
        direction="normal",
        category="momentum",
        description="Chande momentum oscillator over 12 days",
    )
)


# ---------------------------------------------------------------------------
# gtja_113
# ---------------------------------------------------------------------------


def gtja_113(panel: pl.DataFrame) -> pl.Series:
    """GTJA #113 — -1 * RANK(SUM(DELAY(CLOSE,5),20)/20) * CORR(C,V,2) * RANK(CORR(SUM(C,5), SUM(C,20),2)).

    NB: Daic115 broadcasts the inner ``corr_period=20`` rather than the
    paper's ``2`` for the second corr; we follow Daic115 (parity goal).
    """
    df = panel.with_columns(
        [
            (sum_(delay(pl.col("close"), 5), 20) / 20.0).alias("__a"),
            corr(pl.col("close"), pl.col("volume"), 20).alias("__c1"),
            corr(sum_(pl.col("close"), 5), sum_(pl.col("close"), 20), 20).alias("__c2"),
        ]
    )
    df = df.with_columns(
        [
            rank(pl.col("__a")).alias("__ra"),
            rank(pl.col("__c2")).alias("__rc2"),
        ]
    )
    return df.select(
        (-1.0 * pl.col("__ra") * pl.col("__c1") * pl.col("__rc2")).alias("gtja_113")
    ).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_113",
        impl=gtja_113,
        direction="reverse",
        category="correlation",
        description="-rank(sum(delay(c,5),20)/20)*corr(c,v,20)*rank(corr(sum_c5,sum_c20,20))",
    )
)


# ---------------------------------------------------------------------------
# gtja_114
# ---------------------------------------------------------------------------


def gtja_114(panel: pl.DataFrame) -> pl.Series:
    """GTJA #114 — Range-over-MA scaled by VWAP-Close gap."""
    part = (pl.col("high") - pl.col("low")) / mean(pl.col("close"), 5)
    df = panel.with_columns(
        [
            part.alias("__p"),
            delay(part, 2).alias("__pd"),
        ]
    )
    df = df.with_columns(
        [
            rank(pl.col("__pd")).alias("__rpd"),
            rank(rank(pl.col("volume"))).alias("__rrv"),
        ]
    )
    den = pl.col("__p") / (pl.col("vwap") - pl.col("close") + 1e-7)
    expr = ((pl.col("__rpd") * pl.col("__rrv")) / den).alias("gtja_114")
    return df.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_114",
        impl=gtja_114,
        direction="normal",
        category="volatility",
        description="rank(delay(rng/ma5,2))*rank(rank(v)) / ((rng/ma5)/(vwap-c))",
    )
)


# ---------------------------------------------------------------------------
# gtja_115
# ---------------------------------------------------------------------------


def gtja_115(panel: pl.DataFrame) -> pl.Series:
    """GTJA #115 — Pow of two corrs: (HIGH*0.9+CLOSE*0.1)~MA(V,30) ^ HL2 mid-rank ~ vol-rank."""
    df = panel.with_columns(
        [
            corr(pl.col("high") * 0.9 + pl.col("close") * 0.1, mean(pl.col("volume"), 30), 10).alias("__c1"),
        ]
    )
    df = df.with_columns(
        [
            ts_rank((pl.col("high") + pl.col("low")) / 2.0, 4).alias("__t1"),
            ts_rank(pl.col("volume"), 10).alias("__t2"),
        ]
    )
    df = df.with_columns(
        [
            corr(pl.col("__t1"), pl.col("__t2"), 7).alias("__c2"),
        ]
    )
    df = df.with_columns(
        [
            rank(pl.col("__c1")).alias("__r1"),
            rank(pl.col("__c2")).alias("__r2"),
        ]
    )
    expr = pl.col("__r1").pow(pl.col("__r2")).alias("gtja_115")
    return df.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_115",
        impl=gtja_115,
        direction="normal",
        category="correlation",
        description="rank(corr(0.9H+0.1C, ma(v,30),10)) ^ rank(corr(tsr_HL2_4, tsr_v_10, 7))",
    )
)


# ---------------------------------------------------------------------------
# gtja_116 — Daic115 uses qlib rolling_slope. Implement via regbeta+sequence.
# ---------------------------------------------------------------------------


def gtja_116(panel: pl.DataFrame) -> pl.Series:
    """GTJA #116 — REGBETA(CLOSE, SEQUENCE(20), 20).

    Rolling slope of CLOSE on a monotonic 1..N time index. Daic115 uses
    qlib's ``rolling_slope`` (slope on per-stock row index). We replicate
    that semantics by building a per-stock row counter (``cum_count``)
    and feeding it into our native :func:`regbeta`. Since the x-axis is
    monotonic, ``regbeta`` (cov/var) produces the same OLS slope as
    qlib's ``rolling_slope``.

    Direction: ``normal``. Quality flag: ``0``.
    """
    # per-stock row index 1..N (matches GTJA SEQUENCE semantics on each stock)
    row_idx = pl.int_range(1, pl.len() + 1).over(TS_PART).cast(pl.Float64)
    df = panel.with_columns(row_idx.alias("__row_idx"))
    expr = regbeta(pl.col("close"), pl.col("__row_idx"), 20).alias("gtja_116")
    return df.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_116",
        impl=gtja_116,
        direction="normal",
        category="momentum",
        description="20-day rolling OLS slope of CLOSE on time-index",
        quality_flag=0,
    )
)


# ---------------------------------------------------------------------------
# gtja_117
# ---------------------------------------------------------------------------


def gtja_117(panel: pl.DataFrame) -> pl.Series:
    """GTJA #117 — TSRANK(VOL,32) * (1-TSRANK(C+H-L,16)) * (1-TSRANK(RET,32))."""
    ret = pl.col("close") / delay(pl.col("close"), 1) - 1.0
    chl = pl.col("close") + pl.col("high") - pl.col("low")
    expr = (
        ts_rank(pl.col("volume"), 32)
        * (1.0 - ts_rank(chl, 16))
        * (1.0 - ts_rank(ret, 32))
    ).alias("gtja_117")
    return panel.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_117",
        impl=gtja_117,
        direction="normal",
        category="momentum",
        description="tsrank(v,32) * (1-tsrank(c+h-l,16)) * (1-tsrank(ret,32))",
    )
)


# ---------------------------------------------------------------------------
# gtja_118
# ---------------------------------------------------------------------------


def gtja_118(panel: pl.DataFrame) -> pl.Series:
    """GTJA #118 — SUM(H-O,20) / SUM(O-L,20) * 100.

    Open-relative range-skew over 20 days.
    """
    expr = (
        sum_(pl.col("high") - pl.col("open"), 20)
        / sum_(pl.col("open") - pl.col("low"), 20)
        * 100.0
    ).alias("gtja_118")
    return panel.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_118",
        impl=gtja_118,
        direction="normal",
        category="volatility",
        description="20-day open-relative range skew (h-o vs o-l)",
    )
)


# ---------------------------------------------------------------------------
# gtja_119
# ---------------------------------------------------------------------------


def gtja_119(panel: pl.DataFrame) -> pl.Series:
    """GTJA #119 — Decay-linear of corrs and tsranks, V→Liquidity.

    Guotai Junan Formula
    --------------------
        RANK(DECAYLINEAR(CORR(VWAP, SUM(MEAN(VOLUME,5),26), 5), 7)) -
        RANK(DECAYLINEAR(TSRANK(MIN(CORR(RANK(OPEN), RANK(MEAN(VOLUME,15)),21), 9), 7), 8))
    """
    df = panel.with_columns(
        [
            mean(pl.col("volume"), 5).alias("__mv5"),
            mean(pl.col("volume"), 15).alias("__mv15"),
        ]
    )
    df = df.with_columns(
        [
            sum_(pl.col("__mv5"), 26).alias("__smv5_26"),
            rank(pl.col("open")).alias("__ro"),
            rank(pl.col("__mv15")).alias("__rmv"),
        ]
    )
    df = df.with_columns(
        [
            corr(pl.col("vwap"), pl.col("__smv5_26"), 5).alias("__c1"),
        ]
    )
    df = df.with_columns(
        [
            corr(pl.col("__ro"), pl.col("__rmv"), 21).alias("__c2"),
        ]
    )
    df = df.with_columns(
        [
            ts_min(pl.col("__c2"), 9).alias("__m"),
        ]
    )
    df = df.with_columns(
        [
            ts_rank(pl.col("__m"), 7).alias("__tr"),
        ]
    )
    df = df.with_columns(
        [
            decay_linear(pl.col("__c1"), 7).alias("__dl1"),
            decay_linear(pl.col("__tr"), 8).alias("__dl2"),
        ]
    )
    df = df.with_columns(
        [
            rank(pl.col("__dl1")).alias("__r1"),
            rank(pl.col("__dl2")).alias("__r2"),
        ]
    )
    expr = (pl.col("__r1") - pl.col("__r2")).alias("gtja_119")
    return df.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_119",
        impl=gtja_119,
        direction="normal",
        category="correlation",
        description="decay-linear of corr(vwap, sum_mean_v) minus decay-linear tsrank-min-corr(rank_o, rank_mv)",
    )
)


# ---------------------------------------------------------------------------
# gtja_120
# ---------------------------------------------------------------------------


def gtja_120(panel: pl.DataFrame) -> pl.Series:
    """GTJA #120 — RANK(VWAP-CLOSE) / RANK(VWAP+CLOSE)."""
    df = panel.with_columns(
        [
            rank(pl.col("vwap") - pl.col("close")).alias("__a"),
            rank(pl.col("vwap") + pl.col("close")).alias("__b"),
        ]
    )
    return df.select((pl.col("__a") / pl.col("__b")).alias("gtja_120")).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_120",
        impl=gtja_120,
        direction="normal",
        category="volume",
        description="rank(vwap-close) / rank(vwap+close)",
    )
)
