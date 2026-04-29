"""GTJA-191 batch 161..180 (20 factors).

Special factors in this batch
-----------------------------

* ``gtja_165`` — Errata factor per ``wpwp/Alpha-101-GTJA-191`` README.
  We follow Daic115's pandas reference verbatim (which expands SUMAC to
  rolling sum of mean-deviations). Quality flag = 1.

* ``gtja_166`` — Errata factor (same source). Daic115 implements an
  approximation:
  ``5 * SUM(part-1-mean(part-1,20),20) / (SUM(mean(part,20)^2,20))^1.5``.
  We follow that. Quality flag = 1.
"""

from __future__ import annotations

import polars as pl

from aurumq_rl.factors.registry import FactorEntry, register_gtja191

from ._ops import (
    corr,
    delay,
    delta,
    highday,
    log_,
    mean,
    rank,
    sign_,
    sma,
    std_,
    sum_,
    ts_max,
    ts_min,
    ts_rank,
)

# ---------------------------------------------------------------------------
# gtja_161
# ---------------------------------------------------------------------------


def gtja_161(panel: pl.DataFrame) -> pl.Series:
    """GTJA #161 — MEAN(MAX of 3 ATR components, 12). 12d average true range."""
    pc = delay(pl.col("close"), 1)
    a = pl.col("high") - pl.col("low")
    b = (pc - pl.col("high")).abs()
    c = (pc - pl.col("low")).abs()
    tr = pl.max_horizontal([pl.max_horizontal([a, b]), c])
    expr = mean(tr, 12).alias("gtja_161")
    return panel.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_161",
        impl=gtja_161,
        direction="normal",
        category="volatility",
        description="12-day mean true range",
    )
)


# ---------------------------------------------------------------------------
# gtja_162
# ---------------------------------------------------------------------------


def gtja_162(panel: pl.DataFrame) -> pl.Series:
    """GTJA #162 — RSI-style normalised: (RSI - min(RSI,12)) / (max(RSI,12) - min(RSI,12))."""
    dc = pl.col("close") - delay(pl.col("close"), 1)
    p2 = sma(pl.when(dc > 0).then(dc).otherwise(0.0), 12, 1)
    p3 = sma(dc.abs(), 12, 1)
    rsi = p2 / p3 * 100.0
    df = panel.with_columns(rsi.alias("__rsi"))
    df = df.with_columns(
        [
            ts_min(pl.col("__rsi"), 12).alias("__rmin"),
            ts_max(pl.col("__rsi"), 12).alias("__rmax"),
        ]
    )
    expr = ((pl.col("__rsi") - pl.col("__rmin")) / (pl.col("__rmax") - pl.col("__rmin"))).alias(
        "gtja_162"
    )
    return df.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_162",
        impl=gtja_162,
        direction="normal",
        category="momentum",
        description="Stochastic-RSI: (RSI-min(RSI,12)) / (max(RSI,12)-min(RSI,12))",
    )
)


# ---------------------------------------------------------------------------
# gtja_163
# ---------------------------------------------------------------------------


def gtja_163(panel: pl.DataFrame) -> pl.Series:
    """GTJA #163 — RANK(-RET * MEAN(V,20) * VWAP * (HIGH - CLOSE))."""
    ret = pl.col("close") / delay(pl.col("close"), 1) - 1.0
    inner = (
        -1.0
        * ret
        * mean(pl.col("volume"), 20)
        * pl.col("vwap")
        * (pl.col("high") - pl.col("close"))
    )
    df = panel.with_columns(inner.alias("__i"))
    return df.select(rank(pl.col("__i")).alias("gtja_163")).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_163",
        impl=gtja_163,
        direction="reverse",
        category="momentum",
        description="rank(-ret * mv20 * vwap * (high-close))",
    )
)


# ---------------------------------------------------------------------------
# gtja_164
# ---------------------------------------------------------------------------


def gtja_164(panel: pl.DataFrame) -> pl.Series:
    """GTJA #164 — Daic115 reference (reciprocal-diff stochastic, SMA13 smoothed).

    Note: Daic115's parens in the original are slightly off — we follow
    their parsing literally for parity.
    """
    pc = delay(pl.col("close"), 1)
    diff = pl.col("close") - pc
    cond = pl.col("close") > pc
    rec = pl.when(cond).then(1.0 / diff).otherwise(1.0)
    inner = rec - ts_min(rec, 12) / (pl.col("high") - pl.col("low")) * 100.0
    expr = sma(inner, 13, 2).alias("gtja_164")
    return panel.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_164",
        impl=gtja_164,
        direction="normal",
        category="momentum",
        description="SMA((rec - min(rec,12) / (h-l) * 100), 13, 2) — Daic115 parsing",
    )
)


# ---------------------------------------------------------------------------
# gtja_165 — errata
# ---------------------------------------------------------------------------


def gtja_165(panel: pl.DataFrame) -> pl.Series:
    """GTJA #165 — Errata (Daic115 expands SUMAC).

    Guotai Junan Formula (paper)
    ----------------------------
        MAX(SUMAC(CLOSE-MEAN(CLOSE,48))) - MIN(SUMAC(CLOSE-MEAN(CLOSE,48))) / STD(CLOSE,48)

    Daic115 expands SUMAC as ``SUM(diff, 48)`` and computes:
        TS_MAX(SUM(diff,48),48) - TS_MIN(SUM(diff,48),48) / STD(CLOSE,48)

    Listed in errata as ``return 0`` in some references. We follow
    Daic115's expansion. Quality flag = 1.

    Direction: ``normal``. Quality flag: ``1``.
    """
    diff = pl.col("close") - mean(pl.col("close"), 48)
    s48 = sum_(diff, 48)
    expr = (ts_max(s48, 48) - ts_min(s48, 48) / std_(pl.col("close"), 48)).alias("gtja_165")
    return panel.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_165",
        impl=gtja_165,
        direction="normal",
        category="volatility",
        description="MAX(SUMAC(close-ma48)) - MIN(SUMAC(close-ma48)) / STD(close,48) — Daic115 SUMAC expansion",
        quality_flag=1,
    )
)


# ---------------------------------------------------------------------------
# gtja_166 — errata
# ---------------------------------------------------------------------------


def gtja_166(panel: pl.DataFrame) -> pl.Series:
    """GTJA #166 — Errata (Daic115 simplification of skewness-of-returns).

    Guotai Junan Formula (paper, with errata)
    ------------------------------------------
        -20 * 19^1.5 * SUM(part1 - mean(part1,20), 20) /
        ((20-1)*(20-2)*(SUM((part^2,20))^1.5))
        where part = CLOSE / DELAY(CLOSE,1)

    Daic115 simplifies to:
        5 * SUM(part-1 - mean(part-1,20),20) / (SUM(mean(part,20)^2,20))^1.5

    Listed in errata. We follow Daic115's simplification. Quality flag = 1.

    Direction: ``normal``. Quality flag: ``1``.
    """
    part = pl.col("close") / delay(pl.col("close"), 1)
    p1 = part - 1.0
    df = panel.with_columns(
        [
            (p1 - mean(p1, 20)).alias("__centered"),
            mean(part, 20).alias("__mp"),
        ]
    )
    expr = (5.0 * sum_(pl.col("__centered"), 20) / sum_(pl.col("__mp").pow(2), 20).pow(1.5)).alias(
        "gtja_166"
    )
    return df.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_166",
        impl=gtja_166,
        direction="normal",
        category="momentum",
        description="5*sum(centered_ret,20)/(sum(ma_ret_squared,20))^1.5 — Daic115 errata simplification",
        quality_flag=1,
    )
)


# ---------------------------------------------------------------------------
# gtja_167
# ---------------------------------------------------------------------------


def gtja_167(panel: pl.DataFrame) -> pl.Series:
    """GTJA #167 — SUM(MAX(C-prev_C,0), 12). 12-day cumulative up-move."""
    dc = pl.col("close") - delay(pl.col("close"), 1)
    expr = sum_(pl.when(dc > 0).then(dc).otherwise(0.0), 12).alias("gtja_167")
    return panel.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_167",
        impl=gtja_167,
        direction="normal",
        category="momentum",
        description="12-day cumulative up-move",
    )
)


# ---------------------------------------------------------------------------
# gtja_168
# ---------------------------------------------------------------------------


def gtja_168(panel: pl.DataFrame) -> pl.Series:
    """GTJA #168 — -V / MEAN(V,20). Inverse-volume-relative."""
    expr = (-1.0 * pl.col("volume") / mean(pl.col("volume"), 20)).alias("gtja_168")
    return panel.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_168",
        impl=gtja_168,
        direction="reverse",
        category="volume",
        description="-volume / mean(volume,20)",
    )
)


# ---------------------------------------------------------------------------
# gtja_169
# ---------------------------------------------------------------------------


def gtja_169(panel: pl.DataFrame) -> pl.Series:
    """GTJA #169 — DEA-style on SMA-of-dC."""
    inner = sma(pl.col("close") - delay(pl.col("close"), 1), 9, 1)
    df = panel.with_columns(delay(inner, 1).alias("__d"))
    expr = sma(mean(pl.col("__d"), 12) - mean(pl.col("__d"), 26), 10, 1).alias("gtja_169")
    return df.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_169",
        impl=gtja_169,
        direction="normal",
        category="momentum",
        description="SMA(mean(delay(SMA(dC,9,1),1),12) - mean(delay(SMA(dC,9,1),1),26), 10, 1)",
    )
)


# ---------------------------------------------------------------------------
# gtja_170
# ---------------------------------------------------------------------------


def gtja_170(panel: pl.DataFrame) -> pl.Series:
    """GTJA #170 — Weighted multi-rank composite."""
    df = panel.with_columns(rank(1.0 / pl.col("close")).alias("__r1"))
    df = df.with_columns(rank(pl.col("high") - pl.col("close")).alias("__r2"))
    a = pl.col("__r1") * pl.col("volume") / mean(pl.col("volume"), 20)
    b = (pl.col("high") * pl.col("__r2")) / (sum_(pl.col("high"), 5) / 5.0)
    df = df.with_columns((pl.col("vwap") - delay(pl.col("vwap"), 5)).alias("__dvwap5"))
    df = df.with_columns(rank(pl.col("__dvwap5")).alias("__r3"))
    expr = (a * b - pl.col("__r3")).alias("gtja_170")
    return df.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_170",
        impl=gtja_170,
        direction="normal",
        category="volume",
        description="rank(1/c)*v/mv20 * (h*rank(h-c))/(sum(h,5)/5) - rank(vwap-d_vwap_5)",
    )
)


# ---------------------------------------------------------------------------
# gtja_171
# ---------------------------------------------------------------------------


def gtja_171(panel: pl.DataFrame) -> pl.Series:
    """GTJA #171 — -1 * (L-C) * O^5 / ((C-H)*C^5)."""
    expr = (
        -1.0
        * (pl.col("low") - pl.col("close"))
        * pl.col("open").pow(5)
        / ((pl.col("close") - pl.col("high") + 1e-7) * pl.col("close").pow(5))
    ).alias("gtja_171")
    return panel.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_171",
        impl=gtja_171,
        direction="reverse",
        category="momentum",
        description="-(low-close)*open^5 / ((close-high)*close^5)",
    )
)


# ---------------------------------------------------------------------------
# gtja_172
# ---------------------------------------------------------------------------


def gtja_172(panel: pl.DataFrame) -> pl.Series:
    """GTJA #172 — DI-difference oscillator (DX) averaged over 6 days."""
    pc = delay(pl.col("close"), 1)
    a = pl.col("high") - pl.col("low")
    b = (pl.col("high") - pc).abs()
    c = (pl.col("low") - pc).abs()
    tr = pl.max_horizontal([pl.max_horizontal([a, b]), c])
    hd = pl.col("high") - delay(pl.col("high"), 1)
    ld = delay(pl.col("low"), 1) - pl.col("low")
    pos_ld = pl.when((ld > 0) & (ld > hd)).then(ld).otherwise(0.0)
    pos_hd = pl.when((hd > 0) & (hd > ld)).then(hd).otherwise(0.0)
    sum_tr = sum_(tr, 14)
    p1 = sum_(pos_ld, 14) * 100.0 / sum_tr
    p2 = sum_(pos_hd, 14) * 100.0 / sum_tr
    expr = mean((p1 - p2).abs() / (p1 + p2) * 100.0, 6).alias("gtja_172")
    return panel.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_172",
        impl=gtja_172,
        direction="normal",
        category="momentum",
        description="6-day mean of DI-difference oscillator (DX)",
    )
)


# ---------------------------------------------------------------------------
# gtja_173
# ---------------------------------------------------------------------------


def gtja_173(panel: pl.DataFrame) -> pl.Series:
    """GTJA #173 — 3*SMA(C,13,2) - 2*SMA^2(C,13,2) + SMA^3(LOG(C),13,2)."""
    ma = sma(pl.col("close"), 13, 2)
    expr = (3.0 * ma - 2.0 * sma(ma, 13, 2) + sma(sma(log_(pl.col("close")), 13, 2), 13, 2)).alias(
        "gtja_173"
    )
    return panel.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_173",
        impl=gtja_173,
        direction="normal",
        category="momentum",
        description="3*sma(c,13,2) - 2*sma^2(c,13,2) + sma^3(log(c),13,2)",
    )
)


# ---------------------------------------------------------------------------
# gtja_174
# ---------------------------------------------------------------------------


def gtja_174(panel: pl.DataFrame) -> pl.Series:
    """GTJA #174 — SMA(C>prev_C ? STD(C,20) : 0, 20, 1). Up-day vol EWMA."""
    pc = delay(pl.col("close"), 1)
    s = std_(pl.col("close"), 20)
    masked = (
        pl.when(pc.is_null() | s.is_null())
        .then(None)
        .when(pl.col("close") > pc)
        .then(s)
        .otherwise(0.0)
    )
    expr = sma(masked, 20, 1).alias("gtja_174")
    return panel.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_174",
        impl=gtja_174,
        direction="normal",
        category="volatility",
        description="EWMA of up-day std(close,20)",
    )
)


# ---------------------------------------------------------------------------
# gtja_175
# ---------------------------------------------------------------------------


def gtja_175(panel: pl.DataFrame) -> pl.Series:
    """GTJA #175 — 6-day mean true range."""
    pc = delay(pl.col("close"), 1)
    a = pl.col("high") - pl.col("low")
    b = (pc - pl.col("high")).abs()
    c = (pc - pl.col("low")).abs()
    tr = pl.max_horizontal([pl.max_horizontal([a, b]), c])
    expr = mean(tr, 6).alias("gtja_175")
    return panel.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_175",
        impl=gtja_175,
        direction="normal",
        category="volatility",
        description="6-day mean true range",
    )
)


# ---------------------------------------------------------------------------
# gtja_176
# ---------------------------------------------------------------------------


def gtja_176(panel: pl.DataFrame) -> pl.Series:
    """GTJA #176 — CORR(RANK(stoch_K), RANK(VOLUME), 6)."""
    stoch = (pl.col("close") - ts_min(pl.col("low"), 12)) / (
        ts_max(pl.col("high"), 12) - ts_min(pl.col("low"), 12)
    )
    df = panel.with_columns(stoch.alias("__stoch"))
    df = df.with_columns(
        [
            rank(pl.col("__stoch")).alias("__rs"),
            rank(pl.col("volume")).alias("__rv"),
        ]
    )
    expr = corr(pl.col("__rs"), pl.col("__rv"), 6).alias("gtja_176")
    return df.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_176",
        impl=gtja_176,
        direction="normal",
        category="correlation",
        description="corr(rank(stoch_K_12), rank(volume), 6)",
    )
)


# ---------------------------------------------------------------------------
# gtja_177
# ---------------------------------------------------------------------------


def gtja_177(panel: pl.DataFrame) -> pl.Series:
    """GTJA #177 — (20 - HIGHDAY(HIGH,20)) / 20 * 100. Recency-of-high."""
    expr = ((20.0 - highday(pl.col("high"), 20)) / 20.0 * 100.0).alias("gtja_177")
    return panel.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_177",
        impl=gtja_177,
        direction="normal",
        category="momentum",
        description="(20-highday(high,20))/20*100 — recency-of-high oscillator",
    )
)


# ---------------------------------------------------------------------------
# gtja_178
# ---------------------------------------------------------------------------


def gtja_178(panel: pl.DataFrame) -> pl.Series:
    """GTJA #178 — (C-prev_C)/prev_C * V. Vol-weighted daily return."""
    pc = delay(pl.col("close"), 1)
    expr = ((pl.col("close") - pc) / pc * pl.col("volume")).alias("gtja_178")
    return panel.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_178",
        impl=gtja_178,
        direction="normal",
        category="volume",
        description="(c-prev_c)/prev_c * volume",
    )
)


# ---------------------------------------------------------------------------
# gtja_179
# ---------------------------------------------------------------------------


def gtja_179(panel: pl.DataFrame) -> pl.Series:
    """GTJA #179 — RANK(CORR(VWAP,V,4)) * RANK(CORR(RANK(LOW), RANK(MEAN(V,50)), 12))."""
    df = panel.with_columns(
        [
            corr(pl.col("vwap"), pl.col("volume"), 4).alias("__c1"),
            mean(pl.col("volume"), 50).alias("__mv50"),
            rank(pl.col("low")).alias("__rl"),
        ]
    )
    df = df.with_columns(rank(pl.col("__mv50")).alias("__rmv"))
    df = df.with_columns(corr(pl.col("__rl"), pl.col("__rmv"), 12).alias("__c2"))
    df = df.with_columns(
        [
            rank(pl.col("__c1")).alias("__r1"),
            rank(pl.col("__c2")).alias("__r2"),
        ]
    )
    return df.select((pl.col("__r1") * pl.col("__r2")).alias("gtja_179")).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_179",
        impl=gtja_179,
        direction="normal",
        category="correlation",
        description="rank(corr(vwap,v,4)) * rank(corr(rank(low), rank(mv50), 12))",
    )
)


# ---------------------------------------------------------------------------
# gtja_180
# ---------------------------------------------------------------------------


def gtja_180(panel: pl.DataFrame) -> pl.Series:
    """GTJA #180 — Conditional momentum vs negative volume.

    Guotai Junan Formula
    --------------------
        MEAN(VOLUME,20) < VOLUME ?
            -TSRANK(|DELTA(CLOSE,7)|,60) * SIGN(DELTA(CLOSE,7))
            : -VOLUME
    """
    df = panel.with_columns(delta(pl.col("close"), 7).alias("__d7"))
    cond = mean(pl.col("volume"), 20) < pl.col("volume")
    branch_true = -1.0 * ts_rank(pl.col("__d7").abs(), 60) * sign_(pl.col("__d7"))
    branch_false = -1.0 * pl.col("volume")
    expr = pl.when(cond).then(branch_true).otherwise(branch_false).alias("gtja_180")
    return df.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_180",
        impl=gtja_180,
        direction="reverse",
        category="momentum",
        description="Conditional: -tsrank(|d_c7|,60)*sign(d_c7) when V > MV20 else -V",
    )
)
