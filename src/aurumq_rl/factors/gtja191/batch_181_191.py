"""GTJA-191 batch 181..191 (11 factors).

Special factors in this batch
-----------------------------

* ``gtja_181`` — BENCHMARK + ERRATA. Daic115 reference uses cross-section
  mean of CLOSE returns as proxy for CSI300 (degraded data per
  alpha191.md handoff §4.1). We follow that for parity. Errata (paper
  formula has missing window argument). Quality flag = 1.

* ``gtja_182`` — BENCHMARK. Daic115 uses cross-section mean of OHLC as
  benchmark proxy. We follow that for parity. Quality flag = 0.

* ``gtja_183`` — Errata. SUMAC expansion (matches gtja_165 pattern).
  Quality flag = 1.

* ``gtja_191`` — Errata factor per ``wpwp/Alpha-101-GTJA-191`` README.
  Implement best-effort. Quality flag = 1.
"""

from __future__ import annotations

import polars as pl

from aurumq_rl.factors.registry import FactorEntry, register_gtja191

from ._ops import (
    corr,
    count_,
    delay,
    log_,
    mean,
    rank,
    sma,
    std_,
    sum_,
    sumif,
    ts_max,
    ts_min,
)

# ---------------------------------------------------------------------------
# gtja_181 — BENCHMARK + ERRATA
# ---------------------------------------------------------------------------


def gtja_181(panel: pl.DataFrame) -> pl.Series:
    """GTJA #181 — Skewness-adjusted return-vs-benchmark composite (errata).

    Guotai Junan Formula
    --------------------
        SUM(((CLOSE/DELAY(CLOSE,1)-1) - MEAN(C/Cprev-1, 20)) -
            (BMK - MEAN(BMK,20))^2, 20) /
        SUM((BMK - MEAN(BMK,20))^3)

    Benchmark sourcing
    ------------------
    Same proxy approach as :func:`gtja_149` — Daic115's reference uses
    cross-section mean of CLOSE returns. Production wiring to CSI300
    OHLC is Phase D.

    Direction: ``normal``. Quality flag: ``1`` (errata + degraded data).
    """
    ret = pl.col("close") / delay(pl.col("close"), 1) - 1.0
    df = panel.with_columns(ret.alias("__ret"))
    bench = pl.col("__ret").mean().over("trade_date")
    df = df.with_columns(bench.alias("__bench"))
    df = df.with_columns(
        [
            (pl.col("__ret") - mean(pl.col("__ret"), 20)).alias("__centered"),
            (pl.col("__bench") - mean(pl.col("__bench"), 20)).alias("__bcent"),
        ]
    )
    num = sum_(pl.col("__centered") - pl.col("__bcent").pow(2), 20)
    den = sum_(pl.col("__bcent").pow(3), 20)
    expr = (num / den).alias("gtja_181")
    return df.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_181",
        impl=gtja_181,
        direction="normal",
        category="benchmark",
        description="Skew-adjusted return-vs-benchmark — CSI300 proxied via CS mean (Phase D)",
        quality_flag=1,
    )
)


# ---------------------------------------------------------------------------
# gtja_182 — BENCHMARK
# ---------------------------------------------------------------------------


def gtja_182(panel: pl.DataFrame) -> pl.Series:
    """GTJA #182 — Co-movement count: (C>O & BMK_C>BMK_O) | (C<O & BMK_C<BMK_O).

    Guotai Junan Formula
    --------------------
        COUNT(
            (CLOSE > OPEN & BMK_C > BMK_O) | (CLOSE < OPEN & BMK_C < BMK_O),
            20
        ) / 20

    Benchmark sourcing — cross-section mean of OHLC (matches Daic115).
    Phase D: switch to true CSI300.

    Direction: ``normal``. Quality flag: ``0``.
    """
    df = panel.with_columns(
        [
            pl.col("close").mean().over("trade_date").alias("__bc"),
            pl.col("open").mean().over("trade_date").alias("__bo"),
        ]
    )
    bench_up = pl.col("__bc") > pl.col("__bo")
    stock_up = pl.col("close") > pl.col("open")
    same_dir = stock_up == bench_up  # both up or both down
    expr = (sum_(same_dir.cast(pl.Float64), 20) / 20.0).alias("gtja_182")
    return df.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_182",
        impl=gtja_182,
        direction="normal",
        category="benchmark",
        description="20d frac of days stock & benchmark co-move — CS-mean OHLC proxy (Phase D)",
        quality_flag=0,
    )
)


# ---------------------------------------------------------------------------
# gtja_183 — errata
# ---------------------------------------------------------------------------


def gtja_183(panel: pl.DataFrame) -> pl.Series:
    """GTJA #183 — Errata (Daic115 SUMAC expansion).

    Guotai Junan Formula
    --------------------
        MAX(SUMAC(C-MEAN(C,24))) - MIN(SUMAC(C-MEAN(C,24))) / STD(C,24)

    Daic115 expands SUMAC = SUM(diff, 24). Quality flag = 1.

    Direction: ``normal``. Quality flag: ``1``.
    """
    diff = pl.col("close") - mean(pl.col("close"), 24)
    s = sum_(diff, 24)
    expr = (ts_max(s, 24) - ts_min(s, 24) / std_(pl.col("close"), 24)).alias("gtja_183")
    return panel.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_183",
        impl=gtja_183,
        direction="normal",
        category="volatility",
        description="MAX(SUMAC(c-ma24,24)) - MIN(SUMAC(c-ma24,24)) / STD(c,24)",
        quality_flag=1,
    )
)


# ---------------------------------------------------------------------------
# gtja_184
# ---------------------------------------------------------------------------


def gtja_184(panel: pl.DataFrame) -> pl.Series:
    """GTJA #184 — RANK(CORR(DELAY(O-C,1), C, 200)) + RANK(O-C)."""
    df = panel.with_columns(
        [
            delay(pl.col("open") - pl.col("close"), 1).alias("__d"),
            (pl.col("open") - pl.col("close")).alias("__oc"),
        ]
    )
    df = df.with_columns(corr(pl.col("__d"), pl.col("close"), 200).alias("__c"))
    df = df.with_columns(
        [
            rank(pl.col("__c")).alias("__r1"),
            rank(pl.col("__oc")).alias("__r2"),
        ]
    )
    return df.select((pl.col("__r1") + pl.col("__r2")).alias("gtja_184")).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_184",
        impl=gtja_184,
        direction="normal",
        category="correlation",
        description="rank(corr(delay(o-c,1), close, 200)) + rank(o-c)",
    )
)


# ---------------------------------------------------------------------------
# gtja_185
# ---------------------------------------------------------------------------


def gtja_185(panel: pl.DataFrame) -> pl.Series:
    """GTJA #185 — RANK(-1 * (1 - O/C)^2). Squared open-close gap, ranked."""
    inner = -1.0 * (1.0 - pl.col("open") / pl.col("close")).pow(2)
    df = panel.with_columns(inner.alias("__i"))
    return df.select(rank(pl.col("__i")).alias("gtja_185")).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_185",
        impl=gtja_185,
        direction="reverse",
        category="momentum",
        description="rank(-(1 - open/close)^2)",
    )
)


# ---------------------------------------------------------------------------
# gtja_186
# ---------------------------------------------------------------------------


def gtja_186(panel: pl.DataFrame) -> pl.Series:
    """GTJA #186 — Smoothed DI-difference oscillator."""
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
    p3 = (p1 - p2).abs() / (p1 + p2) * 100.0
    expr = ((mean(p3, 6) + delay(mean(p3, 6), 6)) / 2.0).alias("gtja_186")
    return panel.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_186",
        impl=gtja_186,
        direction="normal",
        category="momentum",
        description="(mean(DX,6) + delay(mean(DX,6),6)) / 2 — smoothed ADX-style",
    )
)


# ---------------------------------------------------------------------------
# gtja_187
# ---------------------------------------------------------------------------


def gtja_187(panel: pl.DataFrame) -> pl.Series:
    """GTJA #187 — SUM(O<=prev_O ? 0 : MAX(H-O, O-prev_O), 20). Open-gap up cumsum."""
    po = delay(pl.col("open"), 1)
    body = pl.max_horizontal([pl.col("high") - pl.col("open"), pl.col("open") - po])
    masked = pl.when(pl.col("open") <= po).then(0.0).otherwise(body)
    expr = sum_(masked, 20).alias("gtja_187")
    return panel.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_187",
        impl=gtja_187,
        direction="normal",
        category="momentum",
        description="20-day cumulative open-up-gap or daily upper-body-range",
    )
)


# ---------------------------------------------------------------------------
# gtja_188
# ---------------------------------------------------------------------------


def gtja_188(panel: pl.DataFrame) -> pl.Series:
    """GTJA #188 — ((H-L) - SMA(H-L,11,2)) / SMA(H-L,11,2) * 100. Range deviation."""
    rng = pl.col("high") - pl.col("low")
    s = sma(rng, 11, 2)
    expr = ((rng - s) / s * 100.0).alias("gtja_188")
    return panel.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_188",
        impl=gtja_188,
        direction="normal",
        category="volatility",
        description="((H-L) - SMA(H-L,11,2)) / SMA(H-L,11,2) * 100",
    )
)


# ---------------------------------------------------------------------------
# gtja_189
# ---------------------------------------------------------------------------


def gtja_189(panel: pl.DataFrame) -> pl.Series:
    """GTJA #189 — MEAN(|C - MEAN(C,6)|, 6). Mean abs deviation from MA6."""
    expr = mean((pl.col("close") - mean(pl.col("close"), 6)).abs(), 6).alias("gtja_189")
    return panel.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_189",
        impl=gtja_189,
        direction="normal",
        category="volatility",
        description="6-day mean abs deviation from MA6",
    )
)


# ---------------------------------------------------------------------------
# gtja_190
# ---------------------------------------------------------------------------


def gtja_190(panel: pl.DataFrame) -> pl.Series:
    """GTJA #190 — Log-asymmetric return classifier.

    Guotai Junan Formula
    --------------------
        LOG((COUNT(p1>p2,20)-1) * SUMIF((p1-p2)^2,20,p1<p2) /
            ((COUNT(p1<p2,20)) * SUMIF((p1-p2)^2,20,p1>p2)))
        where p1 = C/prev_C - 1, p2 = (C/C-19)^(1/20) - 1
    """
    pc = delay(pl.col("close"), 1)
    pc19 = delay(pl.col("close"), 19)
    p1 = pl.col("close") / pc - 1.0
    p2 = (pl.col("close") / pc19).pow(1.0 / 20.0) - 1.0
    df = panel.with_columns([p1.alias("__p1"), p2.alias("__p2")])
    df = df.with_columns(((pl.col("__p1") - pl.col("__p2")).pow(2)).alias("__sq"))
    cond_gt = pl.col("__p1") > pl.col("__p2")
    cond_lt = pl.col("__p1") < pl.col("__p2")
    sumif_lt = sumif(pl.col("__sq"), 20, cond_lt)
    sumif_gt = sumif(pl.col("__sq"), 20, cond_gt)
    cnt_gt = count_(cond_gt, 20)
    cnt_lt = count_(cond_lt, 20)
    expr = log_((cnt_gt - 1.0) * sumif_lt / (cnt_lt * sumif_gt + 1e-12)).alias("gtja_190")
    return df.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_190",
        impl=gtja_190,
        direction="normal",
        category="momentum",
        description="log((cnt_p1>p2-1)*SUMIF((p1-p2)^2,20,p1<p2)/(cnt_p1<p2 * SUMIF(.,20,p1>p2)))",
    )
)


# ---------------------------------------------------------------------------
# gtja_191 — errata
# ---------------------------------------------------------------------------


def gtja_191(panel: pl.DataFrame) -> pl.Series:
    """GTJA #191 — Errata (best-effort).

    Guotai Junan Formula
    --------------------
        CORR(MEAN(VOLUME,20), LOW, 5) + (HIGH+LOW)/2 - CLOSE

    Listed in errata. The formula is well-defined; we implement it as
    Daic115 does. Quality flag = 1.

    Direction: ``normal``. Quality flag: ``1``.
    """
    expr = (
        corr(mean(pl.col("volume"), 20), pl.col("low"), 5)
        + (pl.col("high") + pl.col("low")) / 2.0
        - pl.col("close")
    ).alias("gtja_191")
    return panel.select(expr).to_series()


register_gtja191(
    FactorEntry(
        id="gtja_191",
        impl=gtja_191,
        direction="normal",
        category="volume",
        description="corr(mv20, low, 5) + (h+l)/2 - close",
        quality_flag=1,
    )
)
