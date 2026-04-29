"""GTJA-191 batch 121..140 (20 factors).

Special factors in this batch
-----------------------------

* ``gtja_121`` — Errata factor per ``wpwp/Alpha-101-GTJA-191`` README
  (returns 0 in that repo). We implement best-effort using the paper
  formula directly. Quality flag = 1 (errata-conservative).

* ``gtja_131`` — Errata factor per same source. Implement best-effort.
  Quality flag = 1.

The XOR / power ambiguity (``^``) in the GTJA paper is resolved as
exponentiation (``**``) following Daic115's pandas reference.
"""

from __future__ import annotations

import polars as pl

from aurumq_rl.factors.registry import FactorEntry, register_gtja191

from ._ops import (
    corr,
    decay_linear,
    delay,
    delta,
    highday,
    log_,
    lowday,
    mean,
    rank,
    sma,
    sum_,
    ts_max,
    ts_min,
    ts_rank,
)

# ---------------------------------------------------------------------------
# gtja_121 — errata
# ---------------------------------------------------------------------------


def gtja_121(panel: pl.DataFrame) -> pl.Series:
    """GTJA #121 — Errata factor (best-effort implementation).

    Guotai Junan Formula
    --------------------
        (RANK((VWAP - MIN(VWAP, 12))) ^ TSRANK(CORR(TSRANK(VWAP, 20),
            TSRANK(MEAN(VOLUME, 60), 2), 18), 3)) * -1

    Listed in ``wpwp/Alpha-101-GTJA-191`` errata as ``return 0``. The
    Daic115 reference implements the paper formula literally. We follow
    Daic115 here (best-effort) and tag with quality_flag=1 so downstream
    code can mask it out if desired.

    Direction: ``reverse``. Quality flag: ``1``.
    """
    df = panel.with_columns(
        [
            (pl.col("vwap") - ts_min(pl.col("vwap"), 12)).alias("__d"),
            ts_rank(pl.col("vwap"), 20).alias("__t1"),
            ts_rank(mean(pl.col("volume"), 60), 2).alias("__t2"),
        ]
    )
    df = df.with_columns([corr(pl.col("__t1"), pl.col("__t2"), 18).alias("__c")])
    df = df.with_columns(
        [
            rank(pl.col("__d")).alias("__r"),
            ts_rank(pl.col("__c"), 3).alias("__tr"),
        ]
    )
    expr = (pl.col("__r").pow(pl.col("__tr")) * -1.0).alias("gtja_121")
    return df.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_121",
        impl=gtja_121,
        direction="reverse",
        category="correlation",
        description="(rank(vwap-min(vwap,12)) ^ tsrank(corr(tsr_vwap_20, tsr_mv60_2, 18), 3)) * -1",
        quality_flag=1,
    )
)


# ---------------------------------------------------------------------------
# gtja_122
# ---------------------------------------------------------------------------


def gtja_122(panel: pl.DataFrame) -> pl.Series:
    """GTJA #122 — Triple-SMA log-close TSI-style oscillator.

    Guotai Junan Formula
    --------------------
        (SMA^3(LOG(CLOSE),13,2) - DELAY(SMA^3(LOG(CLOSE),13,2), 1)) /
         DELAY(SMA^3(LOG(CLOSE),13,2), 1)
    """
    log_c = log_(pl.col("close"))
    triple = sma(sma(sma(log_c, 13, 2), 13, 2), 13, 2)
    df = panel.with_columns(triple.alias("__t"))
    expr = ((pl.col("__t") - delay(pl.col("__t"), 1)) / delay(pl.col("__t"), 1)).alias("gtja_122")
    return df.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_122",
        impl=gtja_122,
        direction="normal",
        category="momentum",
        description="Triple-SMA log-close 1-day delta ratio",
    )
)


# ---------------------------------------------------------------------------
# gtja_123
# ---------------------------------------------------------------------------


def gtja_123(panel: pl.DataFrame) -> pl.Series:
    """GTJA #123 — Binary cross of two corr-ranks (Daic115 style with NaN map).

    Guotai Junan Formula
    --------------------
        (RANK(CORR(SUM((H+L)/2, 20), SUM(MEAN(V,60), 20), 9))
         < RANK(CORR(LOW, VOLUME, 6))) * -1
    """
    df = panel.with_columns(
        [
            corr(
                sum_((pl.col("high") + pl.col("low")) / 2.0, 20),
                sum_(mean(pl.col("volume"), 60), 20),
                9,
            ).alias("__c1"),
            corr(pl.col("low"), pl.col("volume"), 6).alias("__c2"),
        ]
    )
    df = df.with_columns(
        [
            rank(pl.col("__c1")).alias("__r1"),
            rank(pl.col("__c2")).alias("__r2"),
        ]
    )
    expr = (
        pl.when(pl.col("__r1").is_null() | pl.col("__r2").is_null())
        .then(None)
        .otherwise((pl.col("__r1") < pl.col("__r2")).cast(pl.Float64) * -1.0)
        .alias("gtja_123")
    )
    return df.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_123",
        impl=gtja_123,
        direction="reverse",
        category="correlation",
        description="(rank(corr_HL2_sum_mv60) < rank(corr_low_vol)) * -1",
    )
)


# ---------------------------------------------------------------------------
# gtja_124
# ---------------------------------------------------------------------------


def gtja_124(panel: pl.DataFrame) -> pl.Series:
    """GTJA #124 — (CLOSE - VWAP) / DECAYLINEAR(RANK(TSMAX(CLOSE,30)),2)."""
    df = panel.with_columns(ts_max(pl.col("close"), 30).alias("__tc"))
    df = df.with_columns(rank(pl.col("__tc")).alias("__r"))
    df = df.with_columns(decay_linear(pl.col("__r"), 2).alias("__dl"))
    expr = ((pl.col("close") - pl.col("vwap")) / pl.col("__dl")).alias("gtja_124")
    return df.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_124",
        impl=gtja_124,
        direction="normal",
        category="volume",
        description="(close-vwap) / decay_linear(rank(ts_max(close,30)), 2)",
    )
)


# ---------------------------------------------------------------------------
# gtja_125
# ---------------------------------------------------------------------------


def gtja_125(panel: pl.DataFrame) -> pl.Series:
    """GTJA #125 — Decay-linear ratios on corr(VWAP, MA(V,80)) and DELTA(0.5C+0.5VWAP)."""
    df = panel.with_columns(
        [
            corr(pl.col("vwap"), mean(pl.col("volume"), 80), 17).alias("__c"),
            delta((pl.col("close") + pl.col("vwap")) / 2.0, 3).alias("__d"),
        ]
    )
    df = df.with_columns(
        [
            decay_linear(pl.col("__c"), 20).alias("__dlc"),
            decay_linear(pl.col("__d"), 16).alias("__dld"),
        ]
    )
    df = df.with_columns(
        [
            rank(pl.col("__dlc")).alias("__r1"),
            rank(pl.col("__dld")).alias("__r2"),
        ]
    )
    return df.select((pl.col("__r1") / pl.col("__r2")).alias("gtja_125")).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_125",
        impl=gtja_125,
        direction="normal",
        category="correlation",
        description="rank(decay_linear(corr_vwap_mv80,20)) / rank(decay_linear(delta(0.5C+0.5VWAP,3),16))",
    )
)


# ---------------------------------------------------------------------------
# gtja_126
# ---------------------------------------------------------------------------


def gtja_126(panel: pl.DataFrame) -> pl.Series:
    """GTJA #126 — Typical price (C+H+L)/3."""
    expr = ((pl.col("close") + pl.col("high") + pl.col("low")) / 3.0).alias("gtja_126")
    return panel.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_126",
        impl=gtja_126,
        direction="normal",
        category="price",
        description="(close + high + low) / 3 — typical price",
    )
)


# ---------------------------------------------------------------------------
# gtja_127
# ---------------------------------------------------------------------------


def gtja_127(panel: pl.DataFrame) -> pl.Series:
    """GTJA #127 — RMS of pct-distance from 12-day rolling max."""
    tmax = ts_max(pl.col("close"), 12)
    inner = (100.0 * (pl.col("close") - tmax) / tmax).pow(2)
    expr = mean(inner, 12).pow(0.5).alias("gtja_127")
    return panel.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_127",
        impl=gtja_127,
        direction="normal",
        category="volatility",
        description="sqrt(mean((100*(c-max(c,12))/max(c,12))^2, 12))",
    )
)


# ---------------------------------------------------------------------------
# gtja_128
# ---------------------------------------------------------------------------


def gtja_128(panel: pl.DataFrame) -> pl.Series:
    """GTJA #128 — Money-flow index over 14 days using typical price.

    Guotai Junan Formula
    --------------------
        100 - 100 / (1 + SUMIF(TP*V, 14, TP > prev_TP) /
                          SUMIF(TP*V, 14, TP < prev_TP))
    """
    tp = (pl.col("high") + pl.col("low") + pl.col("close")) / 3.0
    df = panel.with_columns(tp.alias("__tp"))
    df = df.with_columns(delay(pl.col("__tp"), 1).alias("__ptp"))
    # Preserve null at first row in both branches so SUM does not include a fake 0.
    null_mask = pl.col("__ptp").is_null()
    cond = pl.col("__tp") > pl.col("__ptp")
    tp_v = pl.col("__tp") * pl.col("volume")
    pos = pl.when(null_mask).then(None).when(cond).then(tp_v).otherwise(0.0)
    neg = pl.when(null_mask).then(None).when(~cond).then(tp_v).otherwise(0.0)
    df = df.with_columns(
        [
            sum_(pos, 14).alias("__sp"),
            sum_(neg, 14).alias("__sn"),
        ]
    )
    expr = (100.0 - 100.0 / (1.0 + pl.col("__sp") / (pl.col("__sn") + 1e-7))).alias("gtja_128")
    return df.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_128",
        impl=gtja_128,
        direction="normal",
        category="volume",
        description="14-day money flow index (MFI) on typical price",
    )
)


# ---------------------------------------------------------------------------
# gtja_129
# ---------------------------------------------------------------------------


def gtja_129(panel: pl.DataFrame) -> pl.Series:
    """GTJA #129 — SUM(IFELSE(dC<0, |dC|, 0), 12). Down-move 12-day cumsum."""
    dc = pl.col("close") - delay(pl.col("close"), 1)
    expr = sum_(pl.when(dc < 0).then(dc.abs()).otherwise(0.0), 12).alias("gtja_129")
    return panel.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_129",
        impl=gtja_129,
        direction="reverse",
        category="momentum",
        description="12-day downside move cumsum",
    )
)


# ---------------------------------------------------------------------------
# gtja_130
# ---------------------------------------------------------------------------


def gtja_130(panel: pl.DataFrame) -> pl.Series:
    """GTJA #130 — Decay-linear corr ratio: HL2~MA(V,40) over rank(VWAP)~rank(VOL)."""
    df = panel.with_columns(
        [
            corr((pl.col("high") + pl.col("low")) / 2.0, mean(pl.col("volume"), 40), 9).alias(
                "__c1"
            ),
            rank(pl.col("vwap")).alias("__rv"),
            rank(pl.col("volume")).alias("__rvo"),
        ]
    )
    df = df.with_columns(
        [
            corr(pl.col("__rv"), pl.col("__rvo"), 7).alias("__c2"),
        ]
    )
    df = df.with_columns(
        [
            decay_linear(pl.col("__c1"), 10).alias("__dl1"),
            decay_linear(pl.col("__c2"), 3).alias("__dl2"),
        ]
    )
    df = df.with_columns(
        [
            rank(pl.col("__dl1")).alias("__r1"),
            rank(pl.col("__dl2")).alias("__r2"),
        ]
    )
    return df.select((pl.col("__r1") / pl.col("__r2")).alias("gtja_130")).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_130",
        impl=gtja_130,
        direction="normal",
        category="correlation",
        description="rank(decay_linear(corr(HL2,mv40,9),10)) / rank(decay_linear(corr(rk_vwap,rk_v,7),3))",
    )
)


# ---------------------------------------------------------------------------
# gtja_131 — errata
# ---------------------------------------------------------------------------


def gtja_131(panel: pl.DataFrame) -> pl.Series:
    """GTJA #131 — Errata factor (best-effort implementation).

    Guotai Junan Formula
    --------------------
        RANK(DELTA(VWAP, 1)) ^ TSRANK(CORR(CLOSE, MEAN(VOLUME, 50), 18), 18)

    Listed in ``wpwp/Alpha-101-GTJA-191`` errata as ``return 0``. The
    Daic115 reference implements the paper literally. We follow Daic115
    (best-effort) and tag with quality_flag=1.

    Direction: ``normal``. Quality flag: ``1``.
    """
    df = panel.with_columns(
        [
            delta(pl.col("vwap"), 1).alias("__d"),
            corr(pl.col("close"), mean(pl.col("volume"), 50), 18).alias("__c"),
        ]
    )
    df = df.with_columns(
        [
            rank(pl.col("__d")).alias("__r"),
            ts_rank(pl.col("__c"), 18).alias("__tr"),
        ]
    )
    expr = pl.col("__r").pow(pl.col("__tr")).alias("gtja_131")
    return df.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_131",
        impl=gtja_131,
        direction="normal",
        category="correlation",
        description="rank(delta(vwap,1)) ^ tsrank(corr(close, mv50, 18), 18)",
        quality_flag=1,
    )
)


# ---------------------------------------------------------------------------
# gtja_132
# ---------------------------------------------------------------------------


def gtja_132(panel: pl.DataFrame) -> pl.Series:
    """GTJA #132 — MEAN(AMOUNT, 20). Average daily turnover."""
    expr = mean(pl.col("amount"), 20).alias("gtja_132")
    return panel.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_132",
        impl=gtja_132,
        direction="normal",
        category="volume",
        description="20-day mean amount (turnover proxy)",
    )
)


# ---------------------------------------------------------------------------
# gtja_133
# ---------------------------------------------------------------------------


def gtja_133(panel: pl.DataFrame) -> pl.Series:
    """GTJA #133 — Recency-of-high vs recency-of-low oscillator."""
    expr = (
        ((20.0 - highday(pl.col("high"), 20)) / 20.0 * 100.0)
        - ((20.0 - lowday(pl.col("low"), 20)) / 20.0 * 100.0)
    ).alias("gtja_133")
    return panel.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_133",
        impl=gtja_133,
        direction="normal",
        category="momentum",
        description="(20-highday(high,20))/20*100 - (20-lowday(low,20))/20*100",
    )
)


# ---------------------------------------------------------------------------
# gtja_134
# ---------------------------------------------------------------------------


def gtja_134(panel: pl.DataFrame) -> pl.Series:
    """GTJA #134 — (C-prev_C12)/prev_C12 * V — vol-weighted 12d return."""
    pc = delay(pl.col("close"), 12)
    expr = ((pl.col("close") - pc) / pc * pl.col("volume")).alias("gtja_134")
    return panel.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_134",
        impl=gtja_134,
        direction="normal",
        category="volume",
        description="(c-prev_c12)/prev_c12 * volume",
    )
)


# ---------------------------------------------------------------------------
# gtja_135
# ---------------------------------------------------------------------------


def gtja_135(panel: pl.DataFrame) -> pl.Series:
    """GTJA #135 — SMA(DELAY(CLOSE/DELAY(CLOSE,20),1), 20, 1).

    The leading nulls (from DELAY(C,20) and DELAY(...,1)) must propagate
    through the SMA — :func:`_ops.sma` honours ``ignore_nulls`` so that
    nulls do not pollute the EWMA recursion.
    """
    inner = pl.col("close") / delay(pl.col("close"), 20)
    expr = sma(delay(inner, 1), 20, 1).alias("gtja_135")
    return panel.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_135",
        impl=gtja_135,
        direction="normal",
        category="momentum",
        description="SMA(delay(close/delay(close,20),1), 20, 1)",
    )
)


# ---------------------------------------------------------------------------
# gtja_136
# ---------------------------------------------------------------------------


def gtja_136(panel: pl.DataFrame) -> pl.Series:
    """GTJA #136 — -RANK(DELTA(RET,3)) * CORR(OPEN, VOLUME, 10)."""
    ret = pl.col("close") / delay(pl.col("close"), 1) - 1.0
    df = panel.with_columns(
        [
            delta(ret, 3).alias("__dr"),
            corr(pl.col("open"), pl.col("volume"), 10).alias("__c"),
        ]
    )
    df = df.with_columns(rank(pl.col("__dr")).alias("__r"))
    return df.select((-1.0 * pl.col("__r") * pl.col("__c")).alias("gtja_136")).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_136",
        impl=gtja_136,
        direction="reverse",
        category="momentum",
        description="-rank(delta(ret,3)) * corr(open, volume, 10)",
    )
)


# ---------------------------------------------------------------------------
# gtja_137
# ---------------------------------------------------------------------------


def gtja_137(panel: pl.DataFrame) -> pl.Series:
    """GTJA #137 — Complex true-range-normalised price change.

    Daic115 reference implementation followed verbatim (with conditional
    decomposed using IFELSE chain).
    """
    pc = delay(pl.col("close"), 1)
    pl_ = delay(pl.col("low"), 1)
    po = delay(pl.col("open"), 1)
    abshc = (pl.col("high") - pc).abs()
    abslc = (pl.col("low") - pc).abs()
    absco = (pc - po).abs()
    abshl = (pl.col("high") - pl_).abs()

    num = 16.0 * (pl.col("close") - pc + (pl.col("close") - pl.col("open")) / 2.0 + pc - po)
    case1 = abshc + abslc / 2.0 + absco / 4.0
    case2 = abslc + abshc / 2.0 + absco / 4.0
    case3 = abshl + absco / 4.0

    cond1 = (abshc > abslc) & (abshc > abshl)
    cond2 = (abslc > abshl) & (abslc > abshc)

    den = pl.when(cond1).then(case1).when(cond2).then(case2).otherwise(case3) + 1e-7
    max_h = pl.when(abshc > abslc).then(abshc).otherwise(abslc)
    expr = (num / den * max_h).alias("gtja_137")
    return panel.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_137",
        impl=gtja_137,
        direction="normal",
        category="volatility",
        description="Wilders'-style TR-normalised close move",
    )
)


# ---------------------------------------------------------------------------
# gtja_138
# ---------------------------------------------------------------------------


def gtja_138(panel: pl.DataFrame) -> pl.Series:
    """GTJA #138 — Decay-linear delta of (0.7L+0.3VWAP) minus tsrank-cascade.

    Guotai Junan Formula
    --------------------
        (RANK(DECAYLINEAR(DELTA(0.7L + 0.3VWAP, 3), 20)) -
         TSRANK(DECAYLINEAR(TSRANK(CORR(TSRANK(LOW, 8),
                                        TSRANK(MEAN(VOLUME,60), 17), 5), 19), 16), 7)) * -1
    """
    df = panel.with_columns(
        [
            delta(pl.col("low") * 0.7 + pl.col("vwap") * 0.3, 3).alias("__d"),
            ts_rank(pl.col("low"), 8).alias("__tl"),
            ts_rank(mean(pl.col("volume"), 60), 17).alias("__tv"),
        ]
    )
    df = df.with_columns(corr(pl.col("__tl"), pl.col("__tv"), 5).alias("__c"))
    df = df.with_columns(ts_rank(pl.col("__c"), 19).alias("__tc"))
    df = df.with_columns(decay_linear(pl.col("__tc"), 16).alias("__dlt"))
    df = df.with_columns(decay_linear(pl.col("__d"), 20).alias("__dld"))
    df = df.with_columns(
        [
            rank(pl.col("__dld")).alias("__r"),
            ts_rank(pl.col("__dlt"), 7).alias("__tr"),
        ]
    )
    expr = ((pl.col("__r") - pl.col("__tr")) * -1.0).alias("gtja_138")
    return df.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_138",
        impl=gtja_138,
        direction="reverse",
        category="correlation",
        description="(rank(dl(delta(0.7L+0.3VWAP,3),20)) - tsr(dl(tsr_corr_ll_mv60),16),7)) * -1",
    )
)


# ---------------------------------------------------------------------------
# gtja_139
# ---------------------------------------------------------------------------


def gtja_139(panel: pl.DataFrame) -> pl.Series:
    """GTJA #139 — -1 * CORR(OPEN, VOLUME, 10)."""
    expr = (-1.0 * corr(pl.col("open"), pl.col("volume"), 10)).alias("gtja_139")
    return panel.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_139",
        impl=gtja_139,
        direction="reverse",
        category="correlation",
        description="-corr(open, volume, 10)",
    )
)


# ---------------------------------------------------------------------------
# gtja_140
# ---------------------------------------------------------------------------


def gtja_140(panel: pl.DataFrame) -> pl.Series:
    """GTJA #140 — MIN of two decay-linear ranks."""
    df = panel.with_columns(
        [
            (
                rank(pl.col("open"))
                + rank(pl.col("low"))
                - rank(pl.col("high"))
                - rank(pl.col("close"))
            ).alias("__a"),
            ts_rank(pl.col("close"), 8).alias("__tc"),
            ts_rank(mean(pl.col("volume"), 60), 20).alias("__tv"),
        ]
    )
    df = df.with_columns(corr(pl.col("__tc"), pl.col("__tv"), 8).alias("__c"))
    df = df.with_columns(decay_linear(pl.col("__c"), 7).alias("__dlc"))
    df = df.with_columns(decay_linear(pl.col("__a"), 8).alias("__dla"))
    df = df.with_columns(
        [
            rank(pl.col("__dla")).alias("__r1"),
            ts_rank(pl.col("__dlc"), 3).alias("__r2"),
        ]
    )
    expr = pl.min_horizontal([pl.col("__r1"), pl.col("__r2")]).alias("gtja_140")
    return df.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_140",
        impl=gtja_140,
        direction="normal",
        category="correlation",
        description="min(rank(dl(rank_OL_HC,8)), tsr(dl(corr(tsr_c8,tsr_mv60_20,8),7),3))",
    )
)
