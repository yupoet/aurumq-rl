"""GTJA-191 batch 141..160 (20 factors).

Special factors in this batch
-----------------------------

* ``gtja_143`` — STUB. Daic115 marks it ``unfinished=True`` (recursive
  SELF reference, paper formula has ambiguity:
  ``CLOSE > DELAY(CLOSE,1) ? (CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)*SELF : SELF``).
  We return all-null. Quality flag = 2.

* ``gtja_147`` — Daic115 reference uses qlib's ``rolling_slope`` on
  ``MEAN(CLOSE, 12)``. We implement using our :func:`regbeta` against
  per-stock row index. Quality flag = 0.

* ``gtja_149`` — Benchmark factor. The Daic115 reference uses
  cross-section mean of CLOSE returns as proxy for the CSI300 benchmark
  (a known degraded data source per the alpha191 handoff doc §4.1). We
  follow the same proxy here so unit tests on the synthetic panel match
  the reference. Production wiring to true CSI300 OHLC is a Phase D
  task. Quality flag = 0 (formula correct, only data source degraded
  for testing).

* ``gtja_151`` — Errata factor per ``wpwp/Alpha-101-GTJA-191`` README.
  Implement best-effort (``SMA(CLOSE-DELAY(CLOSE,20),20,1)``).
  Quality flag = 1.
"""

from __future__ import annotations

import polars as pl

from aurumq_rl.factors.registry import FactorEntry, register_gtja191

from ._ops import (
    TS_PART,
    corr,
    count_,
    decay_linear,
    delay,
    delta,
    log_,
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
# gtja_141
# ---------------------------------------------------------------------------


def gtja_141(panel: pl.DataFrame) -> pl.Series:
    """GTJA #141 — RANK(CORR(RANK(HIGH), RANK(MEAN(VOLUME,15)), 9)) * -1."""
    df = panel.with_columns(
        [
            mean(pl.col("volume"), 15).alias("__mv15"),
            rank(pl.col("high")).alias("__rh"),
        ]
    )
    df = df.with_columns(rank(pl.col("__mv15")).alias("__rmv"))
    df = df.with_columns(corr(pl.col("__rh"), pl.col("__rmv"), 9).alias("__c"))
    df = df.with_columns(rank(pl.col("__c")).alias("__r"))
    return df.select((pl.col("__r") * -1.0).alias("gtja_141")).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_141",
        impl=gtja_141,
        direction="reverse",
        category="correlation",
        description="-rank(corr(rank(high), rank(mean(v,15)), 9))",
    )
)


# ---------------------------------------------------------------------------
# gtja_142
# ---------------------------------------------------------------------------


def gtja_142(panel: pl.DataFrame) -> pl.Series:
    """GTJA #142 — Triple-rank acceleration product."""
    df = panel.with_columns(
        [
            ts_rank(pl.col("close"), 10).alias("__tc"),
            delta(delta(pl.col("close"), 1), 1).alias("__d2"),
            ts_rank(pl.col("volume") / mean(pl.col("volume"), 20), 5).alias("__tv"),
        ]
    )
    df = df.with_columns(
        [
            rank(pl.col("__tc")).alias("__r1"),
            rank(pl.col("__d2")).alias("__r2"),
            rank(pl.col("__tv")).alias("__r3"),
        ]
    )
    return df.select(
        (-1.0 * pl.col("__r1") * pl.col("__r2") * pl.col("__r3")).alias("gtja_142")
    ).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_142",
        impl=gtja_142,
        direction="reverse",
        category="momentum",
        description="-rank(tsr_c10) * rank(d2_close) * rank(tsr_vrel_5)",
    )
)


# ---------------------------------------------------------------------------
# gtja_143 — STUB
# ---------------------------------------------------------------------------


def gtja_143(panel: pl.DataFrame) -> pl.Series:
    """GTJA #143 — STUB (Daic115 marks unfinished, paper has SELF recursion).

    Guotai Junan Formula
    --------------------
        CLOSE > DELAY(CLOSE,1) ? (CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)*SELF : SELF

    The recursive ``SELF`` reference makes the formula ambiguous. The
    Daic115 reference returns ``None`` (function body commented out).
    We follow that — output is all-null Float64. Tests against this
    factor verify dtype + length only.

    Direction: ``normal``. Quality flag: ``2`` (stub).
    """
    return pl.Series("gtja_143", [None] * panel.height, dtype=pl.Float64)


register_gtja191(
    FactorEntry(
        id="gtja_143",
        impl=gtja_143,
        direction="normal",
        category="momentum",
        description="STUB — recursive SELF reference, formula ambiguous",
        quality_flag=2,
    )
)


# ---------------------------------------------------------------------------
# gtja_144
# ---------------------------------------------------------------------------


def gtja_144(panel: pl.DataFrame) -> pl.Series:
    """GTJA #144 — Down-day average abs-return / log-amount.

    Daic115's reference uses ``log(amount)`` rather than raw amount in
    the denominator (deviation from the paper). We follow Daic115 for
    parity.
    """
    pc = delay(pl.col("close"), 1)
    abs_ret = (pl.col("close") / pc - 1.0).abs() / log_(pl.col("amount"))
    cond = pl.col("close") < pc
    masked = pl.when(cond).then(abs_ret).otherwise(0.0)
    expr = (sum_(masked, 20) / count_(cond, 20)).alias("gtja_144")
    return panel.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_144",
        impl=gtja_144,
        direction="normal",
        category="volatility",
        description="20d down-day mean(|ret|/log(amount))",
    )
)


# ---------------------------------------------------------------------------
# gtja_145
# ---------------------------------------------------------------------------


def gtja_145(panel: pl.DataFrame) -> pl.Series:
    """GTJA #145 — (MEAN(V,9) - MEAN(V,26)) / MEAN(V,12) * 100."""
    expr = (
        (mean(pl.col("volume"), 9) - mean(pl.col("volume"), 26))
        / mean(pl.col("volume"), 12)
        * 100.0
    ).alias("gtja_145")
    return panel.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_145",
        impl=gtja_145,
        direction="normal",
        category="volume",
        description="(mean(v,9) - mean(v,26)) / mean(v,12) * 100",
    )
)


# ---------------------------------------------------------------------------
# gtja_146
# ---------------------------------------------------------------------------


def gtja_146(panel: pl.DataFrame) -> pl.Series:
    """GTJA #146 — Daic115 variant: mean(part_ma,20) * part_ma / SMA(part-part_ma)^2."""
    pc = delay(pl.col("close"), 1)
    part = (pl.col("close") - pc) / pc
    df = panel.with_columns(part.alias("__p"))
    df = df.with_columns(sma(pl.col("__p"), 61, 2).alias("__sp"))
    df = df.with_columns((pl.col("__p") - pl.col("__sp")).alias("__pm"))
    df = df.with_columns(
        [
            mean(pl.col("__pm"), 20).alias("__m"),
            sma((pl.col("__p") - pl.col("__pm")).pow(2), 61, 2).alias("__s"),
        ]
    )
    expr = (pl.col("__m") * pl.col("__pm") / pl.col("__s")).alias("gtja_146")
    return df.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_146",
        impl=gtja_146,
        direction="normal",
        category="momentum",
        description="mean(ret-sma(ret,61,2),20) * (ret-sma) / SMA((ret-(ret-sma))^2,61,2)",
    )
)


# ---------------------------------------------------------------------------
# gtja_147 — qlib rolling_slope -> regbeta on MEAN(CLOSE,12)
# ---------------------------------------------------------------------------


def gtja_147(panel: pl.DataFrame) -> pl.Series:
    """GTJA #147 — REGBETA(MEAN(CLOSE,12), SEQUENCE(12)).

    Daic115's reference uses qlib's ``rolling_slope`` on MEAN(CLOSE,12).
    We use our native :func:`regbeta` against a per-stock row index — a
    monotonic x-axis, which produces the same OLS slope.

    Direction: ``normal``. Quality flag: ``0``.
    """
    row_idx = pl.int_range(1, pl.len() + 1).over(TS_PART).cast(pl.Float64)
    df = panel.with_columns(
        [
            mean(pl.col("close"), 12).alias("__mc"),
            row_idx.alias("__row_idx"),
        ]
    )
    expr = regbeta(pl.col("__mc"), pl.col("__row_idx"), 12).alias("gtja_147")
    return df.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_147",
        impl=gtja_147,
        direction="normal",
        category="momentum",
        description="12-day rolling OLS slope of MEAN(close,12) on time-index",
        quality_flag=0,
    )
)


# ---------------------------------------------------------------------------
# gtja_148
# ---------------------------------------------------------------------------


def gtja_148(panel: pl.DataFrame) -> pl.Series:
    """GTJA #148 — (RANK(CORR(OPEN, SUM(MEAN(V,60),9), 6)) < RANK(OPEN-TSMIN(OPEN,14))) * -1."""
    df = panel.with_columns(
        [
            corr(pl.col("open"), sum_(mean(pl.col("volume"), 60), 9), 6).alias("__c"),
            (pl.col("open") - ts_min(pl.col("open"), 14)).alias("__d"),
        ]
    )
    df = df.with_columns([rank(pl.col("__c")).alias("__r1"), rank(pl.col("__d")).alias("__r2")])
    expr = (
        pl.when(pl.col("__r1").is_null() | pl.col("__r2").is_null())
        .then(None)
        .otherwise((pl.col("__r1") < pl.col("__r2")).cast(pl.Float64) * -1.0)
        .alias("gtja_148")
    )
    return df.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_148",
        impl=gtja_148,
        direction="reverse",
        category="correlation",
        description="(rank(corr(o, sum(mv60,9), 6)) < rank(o-tsmin(o,14))) * -1",
    )
)


# ---------------------------------------------------------------------------
# gtja_149 — BENCHMARK
# ---------------------------------------------------------------------------


def gtja_149(panel: pl.DataFrame) -> pl.Series:
    """GTJA #149 — Downside-beta vs benchmark over 252 days.

    Guotai Junan Formula
    --------------------
        REGBETA(
            FILTER(CLOSE/DELAY(CLOSE,1)-1, BMK < DELAY(BMK,1)),
            FILTER(BMK/DELAY(BMK,1)-1, BMK < DELAY(BMK,1)),
            252)

    Benchmark sourcing
    ------------------
    The Daic115 reference uses cross-section mean of CLOSE/DELAY(CLOSE,1)
    returns as proxy for CSI300 — a known degraded data source per the
    alpha191 handoff doc §4.1. We follow the same proxy here so unit
    tests on the synthetic panel match the reference parquet.

    Production wiring to the real CSI300 OHLC (available in our
    ``index_daily`` table from 2015-10 onwards) is a Phase D task. The
    formula itself is correct; only the data source is degraded.

    Daic115 also drops the ``FILTER`` step (commented out) — we match
    that and feed the full series into REGBETA, again for parity.

    Direction: ``normal``. Quality flag: ``0``.
    """
    # cross-section-mean proxy: per-day mean of close-return across stocks
    ret = pl.col("close") / delay(pl.col("close"), 1) - 1.0
    df = panel.with_columns(ret.alias("__ret"))
    bench = pl.col("__ret").mean().over("trade_date")
    df = df.with_columns(bench.alias("__bench"))
    expr = regbeta(pl.col("__ret"), pl.col("__bench"), 252).alias("gtja_149")
    return df.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_149",
        impl=gtja_149,
        direction="normal",
        category="benchmark",
        description="252d beta vs cross-section-mean-return benchmark proxy (CSI300 wiring is Phase D)",
        quality_flag=0,
    )
)


# ---------------------------------------------------------------------------
# gtja_150
# ---------------------------------------------------------------------------


def gtja_150(panel: pl.DataFrame) -> pl.Series:
    """GTJA #150 — (C+H+L)/3 * LOG(VOLUME). Daic115 uses log(volume)."""
    expr = (
        (pl.col("close") + pl.col("high") + pl.col("low")) / 3.0 * log_(pl.col("volume"))
    ).alias("gtja_150")
    return panel.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_150",
        impl=gtja_150,
        direction="normal",
        category="volume",
        description="typical price * log(volume) (Daic115 variant)",
    )
)


# ---------------------------------------------------------------------------
# gtja_151 — errata
# ---------------------------------------------------------------------------


def gtja_151(panel: pl.DataFrame) -> pl.Series:
    """GTJA #151 — Errata factor (best-effort).

    Guotai Junan Formula
    --------------------
        SMA(CLOSE - DELAY(CLOSE, 20), 20, 1)

    Listed in ``wpwp/Alpha-101-GTJA-191`` errata. The formula itself is
    well-defined; we implement it as Daic115 does. Quality flag = 1
    pending downstream verification.

    Direction: ``normal``. Quality flag: ``1``.
    """
    expr = sma(pl.col("close") - delay(pl.col("close"), 20), 20, 1).alias("gtja_151")
    return panel.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_151",
        impl=gtja_151,
        direction="normal",
        category="momentum",
        description="SMA(close - delay(close,20), 20, 1)",
        quality_flag=1,
    )
)


# ---------------------------------------------------------------------------
# gtja_152
# ---------------------------------------------------------------------------


def gtja_152(panel: pl.DataFrame) -> pl.Series:
    """GTJA #152 — DEA-style triple-EWMA differential."""
    inner = sma(delay(pl.col("close") / delay(pl.col("close"), 9), 1), 9, 1)
    part = delay(inner, 1)
    expr = sma(mean(part, 12) - mean(part, 26), 9, 1).alias("gtja_152")
    return panel.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_152",
        impl=gtja_152,
        direction="normal",
        category="momentum",
        description="sma(mean(part,12)-mean(part,26),9,1) where part is delayed sma(c/c9,9,1)",
    )
)


# ---------------------------------------------------------------------------
# gtja_153
# ---------------------------------------------------------------------------


def gtja_153(panel: pl.DataFrame) -> pl.Series:
    """GTJA #153 — (MA3+MA6+MA12+MA24)/4 — multi-MA average."""
    expr = (
        (
            mean(pl.col("close"), 3)
            + mean(pl.col("close"), 6)
            + mean(pl.col("close"), 12)
            + mean(pl.col("close"), 24)
        )
        / 4.0
    ).alias("gtja_153")
    return panel.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_153",
        impl=gtja_153,
        direction="normal",
        category="momentum",
        description="Average of 4 different moving averages (BBI)",
    )
)


# ---------------------------------------------------------------------------
# gtja_154
# ---------------------------------------------------------------------------


def gtja_154(panel: pl.DataFrame) -> pl.Series:
    """GTJA #154 — Sign indicator: -1/0 of (vwap-min(vwap,16)) < CORR(vwap, MA(V,180), 18)."""
    df = panel.with_columns(
        [
            (pl.col("vwap") - ts_min(pl.col("vwap"), 16)).alias("__a"),
            corr(pl.col("vwap"), mean(pl.col("volume"), 180), 18).alias("__c"),
        ]
    )
    expr = (
        pl.when(pl.col("__a").is_null() | pl.col("__c").is_null())
        .then(None)
        .otherwise(pl.when(pl.col("__a") < pl.col("__c")).then(1.0).otherwise(-1.0))
        .alias("gtja_154")
    )
    return df.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_154",
        impl=gtja_154,
        direction="normal",
        category="correlation",
        description="sign((vwap-min(vwap,16)) < corr(vwap, mv180, 18))",
    )
)


# ---------------------------------------------------------------------------
# gtja_155
# ---------------------------------------------------------------------------


def gtja_155(panel: pl.DataFrame) -> pl.Series:
    """GTJA #155 — MACD-style on volume."""
    macd = sma(pl.col("volume"), 13, 2) - sma(pl.col("volume"), 27, 2)
    signal = sma(macd, 10, 2)
    expr = (macd - signal).alias("gtja_155")
    return panel.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_155",
        impl=gtja_155,
        direction="normal",
        category="volume",
        description="Volume MACD (13,27,10) histogram",
    )
)


# ---------------------------------------------------------------------------
# gtja_156
# ---------------------------------------------------------------------------


def gtja_156(panel: pl.DataFrame) -> pl.Series:
    """GTJA #156 — MAX of two decay-linear ranks * -1."""
    a = pl.col("vwap") - delay(pl.col("vwap"), 5)
    b_inner = pl.col("open") * 0.15 + pl.col("low") * 0.85
    b = -delta(b_inner, 2) / b_inner
    df = panel.with_columns(
        [
            decay_linear(a, 3).alias("__dla"),
            decay_linear(b, 3).alias("__dlb"),
        ]
    )
    df = df.with_columns(
        [
            rank(pl.col("__dla")).alias("__r1"),
            rank(pl.col("__dlb")).alias("__r2"),
        ]
    )
    expr = (pl.max_horizontal([pl.col("__r1"), pl.col("__r2")]) * -1.0).alias("gtja_156")
    return df.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_156",
        impl=gtja_156,
        direction="reverse",
        category="momentum",
        description="-max(rank(dl(d_vwap_5,3)), rank(dl(-d_inner_2/inner,3)))",
    )
)


# ---------------------------------------------------------------------------
# gtja_157
# ---------------------------------------------------------------------------


def gtja_157(panel: pl.DataFrame) -> pl.Series:
    """GTJA #157 — TS_MIN of triple-rank-log + tsrank of delay(-ret,6)."""
    df = panel.with_columns(
        [
            delta(pl.col("close") - 1.0, 5).alias("__d"),
            (pl.col("close") / delay(pl.col("close"), 1) - 1.0).alias("__ret"),
        ]
    )
    df = df.with_columns(
        [
            rank(-1.0 * rank(pl.col("__d"))).alias("__rd"),
        ]
    )
    df = df.with_columns(rank(pl.col("__rd")).alias("__rrd"))
    df = df.with_columns(ts_min(pl.col("__rrd"), 2).alias("__tm"))
    df = df.with_columns(sum_(pl.col("__tm"), 1).alias("__sum"))
    df = df.with_columns(rank(rank(log_(pl.col("__sum")))).alias("__lhs"))
    df = df.with_columns(
        [
            ts_min(pl.col("__lhs"), 5).alias("__lhs_min"),
            ts_rank(delay(-1.0 * pl.col("__ret"), 6), 5).alias("__rhs"),
        ]
    )
    expr = (pl.col("__lhs_min") + pl.col("__rhs")).alias("gtja_157")
    return df.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_157",
        impl=gtja_157,
        direction="normal",
        category="momentum",
        description="ts_min(rank(rank(log(sum(ts_min(rank(rank(-rank(d(c-1,5)))),2),1)))),5) + tsr(delay(-ret,6),5)",
    )
)


# ---------------------------------------------------------------------------
# gtja_158
# ---------------------------------------------------------------------------


def gtja_158(panel: pl.DataFrame) -> pl.Series:
    """GTJA #158 — ((H - SMA(C,15,2)) - (L - SMA(C,15,2))) / C — high-low spread / close."""
    s = sma(pl.col("close"), 15, 2)
    expr = (((pl.col("high") - s) - (pl.col("low") - s)) / pl.col("close")).alias("gtja_158")
    return panel.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_158",
        impl=gtja_158,
        direction="normal",
        category="volatility",
        description="(H - L) / C — high-low spread normalised by close (SMA terms cancel)",
    )
)


# ---------------------------------------------------------------------------
# gtja_159
# ---------------------------------------------------------------------------


def gtja_159(panel: pl.DataFrame) -> pl.Series:
    """GTJA #159 — Three-window cumulative range-position oscillator."""
    pc = delay(pl.col("close"), 1)
    p2 = pl.min_horizontal([pl.col("low"), pc])
    p3 = pl.max_horizontal([pl.col("high"), pc])
    p1 = p3 - p2
    df = panel.with_columns(
        [
            p1.alias("__p1"),
            p2.alias("__p2"),
            p3.alias("__p3"),
        ]
    )
    a = (pl.col("close") - sum_(pl.col("__p2"), 6)) / sum_(pl.col("__p1"), 6) * 288.0
    b = (
        (pl.col("close") - sum_(pl.col("__p2"), 12))
        / sum_(pl.col("__p3") - pl.col("__p2"), 12)
        * 144.0
    )
    c = (pl.col("close") - sum_(pl.col("__p2"), 24)) / sum_(pl.col("__p1"), 24) * 144.0
    expr = ((a + b + c) * 100.0 / 504.0).alias("gtja_159")
    return df.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_159",
        impl=gtja_159,
        direction="normal",
        category="momentum",
        description="3-window cumulative range-position oscillator (KDJ-like)",
    )
)


# ---------------------------------------------------------------------------
# gtja_160
# ---------------------------------------------------------------------------


def gtja_160(panel: pl.DataFrame) -> pl.Series:
    """GTJA #160 — SMA(C<=prev_C ? STD(C,20) : 0, 20, 1). Down-day vol EWMA."""
    pc = delay(pl.col("close"), 1)
    s = std_(pl.col("close"), 20)
    # Preserve null on the first row so SMA's EWMA does not see a fake 0.
    masked = (
        pl.when(pc.is_null() | s.is_null())
        .then(None)
        .when(pl.col("close") <= pc)
        .then(s)
        .otherwise(0.0)
    )
    expr = sma(masked, 20, 1).alias("gtja_160")
    return panel.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_160",
        impl=gtja_160,
        direction="normal",
        category="volatility",
        description="EWMA of down-day std(close,20)",
    )
)
