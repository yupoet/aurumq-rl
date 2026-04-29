"""GTJA-191 factor library — batch 041 through 060.

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
    ifelse,
    mean,
    rank,
    sign_,
    sma,
    std_,
    sum_,
    sumif,
    ts_max,
    ts_min,
    ts_rank,
)

# ---------------------------------------------------------------------------
# gtja_041 — RANK(MAX(DELTA(VWAP,3), 5)) * -1
# ---------------------------------------------------------------------------


def gtja_041(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #041 — Negated rank of 5d-max of 3d VWAP delta.

    Guotai Junan Formula
    --------------------
        RANK(MAX(DELTA(VWAP, 3), 5)) * -1

    Required panel columns: ``vwap``, ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``momentum``
    """
    inner = ts_max(delta(pl.col("vwap"), 3), 5)
    staged = panel.with_columns(inner.alias("__g041_x"))
    return staged.select(
        (rank(pl.col("__g041_x")) * -1.0)
        .alias("gtja_041")
        .cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# gtja_042 — -RANK(STD(H,10)) * CORR(H, V, 10)
# ---------------------------------------------------------------------------


def gtja_042(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #042 — Negated rank-of-std times rolling H-V correlation.

    Guotai Junan Formula
    --------------------
        -1 * RANK(STD(HIGH, 10)) * CORR(HIGH, VOLUME, 10)

    Required panel columns: ``high``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volume_price``
    """
    staged = panel.with_columns(
        std_(pl.col("high"), 10).alias("__g042_s"),
        corr(pl.col("high"), pl.col("volume"), 10).alias("__g042_c"),
    )
    return staged.select(
        (-1.0 * rank(pl.col("__g042_s")) * pl.col("__g042_c"))
        .alias("gtja_042")
        .cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# gtja_043 — Signed-volume sum over 6d
# ---------------------------------------------------------------------------


def gtja_043(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #043 — 6d signed-volume sum (vwap-anchored direction).

    Guotai Junan Formula
    --------------------
        SUM((C > DELAY(C,1) ? V : (C < DELAY(C,1) ? -V : 0)), 6)

    Required panel columns: ``vwap``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``volume_price``
    """
    vwap = pl.col("vwap")
    cond = vwap > delay(vwap, 1)
    signed_vol = ifelse(cond, pl.col("volume"), -pl.col("volume"))
    return panel.select(
        sum_(signed_vol, 6).alias("gtja_043").cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# gtja_044 — TS-rank decay corr + TS-rank decay delta
# ---------------------------------------------------------------------------


def gtja_044(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #044 — Sum of two TSRANK(DECAYLINEAR(...)) arms.

    Guotai Junan Formula
    --------------------
        TSRANK(DECAYLINEAR(CORR(LOW, MEAN(VOLUME, 10), 7), 6), 4) +
        TSRANK(DECAYLINEAR(DELTA(VWAP, 3), 10), 15)

    Required panel columns: ``low``, ``volume``, ``vwap``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``volume_price``
    """
    cor_inner = corr(pl.col("low"), mean(pl.col("volume"), 10), 7)
    arm1 = ts_rank(decay_linear(cor_inner, 6), 4)
    arm2 = ts_rank(decay_linear(delta(pl.col("vwap"), 3), 10), 15)
    return panel.select(
        (arm1 + arm2).alias("gtja_044").cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# gtja_045 — Rank delta × rank corr (long-window mean-volume)
# ---------------------------------------------------------------------------


def gtja_045(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #045 — Rank(C0.6+O0.4 delta) × Rank(corr(VWAP, MEAN(V,150), 15)).

    Guotai Junan Formula
    --------------------
        RANK(DELTA(C*0.6 + O*0.4, 1)) * RANK(CORR(VWAP, MEAN(VOLUME, 150), 15))

    Daic115 multiplies by raw corr (not rank); we follow.

    Required panel columns: ``close``, ``open``, ``vwap``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volume_price``
    """
    weighted = pl.col("close") * 0.6 + pl.col("open") * 0.4
    inner_d = delta(weighted, 1)
    corr_v = corr(pl.col("vwap"), mean(pl.col("volume"), 150), 15)
    staged = panel.with_columns(inner_d.alias("__g045_d"))
    staged = staged.with_columns(rank(pl.col("__g045_d")).alias("__g045_r"))
    return staged.select(
        (pl.col("__g045_r") * corr_v).alias("gtja_045").cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# gtja_046 — Multi-window MA average / close
# ---------------------------------------------------------------------------


def gtja_046(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #046 — Multi-window MA average / close.

    Guotai Junan Formula
    --------------------
        (MEAN(C,3) + MEAN(C,6) + MEAN(C,12) + MEAN(C,24)) / (4 * C)

    Required panel columns: ``close``, ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``mean_reversion``
    """
    c = pl.col("close")
    expr = (
        mean(c, 3) + mean(c, 6) + mean(c, 12) + mean(c, 24)
    ) / (4.0 * c)
    return panel.select(
        expr.alias("gtja_046").cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# gtja_047 — RSV-style smoothed
# ---------------------------------------------------------------------------


def gtja_047(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #047 — Smoothed RSV: SMA((TSMAX(H,6)-C)/(TSMAX(H,6)-TSMIN(L,6))*100, 9, 1).

    Required panel columns: ``high``, ``low``, ``close``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``mean_reversion``
    """
    hmax = ts_max(pl.col("high"), 6)
    lmin = ts_min(pl.col("low"), 6)
    raw = (hmax - pl.col("close")) / (hmax - lmin) * 100.0
    return panel.select(
        sma(raw, 9, 1).alias("gtja_047").cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# gtja_048 — Sign sum × volume ratio rank
# ---------------------------------------------------------------------------


def gtja_048(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #048 — Rank(sign-sum) × SUM(V,5)/SUM(V,20).

    Guotai Junan Formula
    --------------------
        -1 * RANK(SIGN(C - DELAY(C,1)) + SIGN(DELAY(C,1) - DELAY(C,2)) +
                  SIGN(DELAY(C,2) - DELAY(C,3))) * SUM(V,5) / SUM(V,20)

    Required panel columns: ``close``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``momentum``
    """
    c = pl.col("close")
    s1 = sign_(delta(c, 1))
    s2 = sign_(delta(delay(c, 1), 1))
    s3 = sign_(delta(delay(c, 2), 1))
    sgn_sum = s1 + s2 + s3
    staged = panel.with_columns(sgn_sum.alias("__g048_s"))
    staged = staged.with_columns(rank(pl.col("__g048_s")).alias("__g048_r"))
    expr = pl.col("__g048_r") * sum_(pl.col("volume"), 5) / sum_(pl.col("volume"), 20)
    return staged.select(
        expr.alias("gtja_048").cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# gtja_049 — Asymmetric high/low range share
# ---------------------------------------------------------------------------


def gtja_049(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #049 — Down-day range share over 12d.

    Guotai Junan Formula
    --------------------
        cond = (H + L) >= (DELAY(H, 1) + DELAY(L, 1))
        part = MAX(|H - DELAY(H, 1)|, |L - DELAY(L, 1)|)
        SUM((!cond ? part : 0), 12) /
        (SUM((!cond ? part : 0), 12) + SUM((cond ? part : 0), 12))

    Required panel columns: ``high``, ``low``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volatility``
    """
    h = pl.col("high")
    lw = pl.col("low")
    cond = (h + lw) >= (delay(h, 1) + delay(lw, 1))
    part = pl.max_horizontal(abs_(h - delay(h, 1)), abs_(lw - delay(lw, 1)))
    s_dn = sumif(part, 12, ~cond)
    s_up = sumif(part, 12, cond)
    return panel.select(
        (s_dn / (s_dn + s_up)).alias("gtja_049").cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# gtja_050 — Asymmetric range diff ratio
# ---------------------------------------------------------------------------


def gtja_050(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #050 — Down-share minus Up-share asymmetric range ratio.

    Guotai Junan Formula
    --------------------
        cond1 = (H + L) <= (DELAY(H,1) + DELAY(L,1))
        cond2 = (H + L) >= (DELAY(H,1) + DELAY(L,1))
        part = MAX(|H - DELAY(H,1)|, |L - DELAY(L,1)|)
        SUM(!cond1?part:0, 12) / (SUM(!cond1?part:0, 12) + SUM(!cond2?part:0, 12)) -
        SUM(!cond2?part:0, 12) / (SUM(!cond2?part:0, 12) + SUM(!cond1?part:0, 12))

    Required panel columns: ``high``, ``low``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volatility``
    """
    h = pl.col("high")
    lw = pl.col("low")
    cond1 = (h + lw) <= (delay(h, 1) + delay(lw, 1))
    cond2 = (h + lw) >= (delay(h, 1) + delay(lw, 1))
    part = pl.max_horizontal(abs_(h - delay(h, 1)), abs_(lw - delay(lw, 1)))
    s_a = sumif(part, 12, ~cond1)
    s_b = sumif(part, 12, ~cond2)
    expr = (s_a - s_b) / (s_a + s_b + 1e-7)
    return panel.select(
        expr.alias("gtja_050").cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# gtja_051 — Down-share asymmetric range ratio
# ---------------------------------------------------------------------------


def gtja_051(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #051 — Down-share asymmetric range ratio over 12d.

    Guotai Junan Formula
    --------------------
        cond1 = (H + L) <= (DELAY(H,1) + DELAY(L,1))
        cond2 = (H + L) >= (DELAY(H,1) + DELAY(L,1))
        part = MAX(|H - DELAY(H,1)|, |L - DELAY(L,1)|)
        SUM(!cond1?part:0, 12) / (SUM(!cond1?part:0, 12) + SUM(!cond2?part:0, 12))

    Required panel columns: ``high``, ``low``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volatility``
    """
    h = pl.col("high")
    lw = pl.col("low")
    cond1 = (h + lw) <= (delay(h, 1) + delay(lw, 1))
    cond2 = (h + lw) >= (delay(h, 1) + delay(lw, 1))
    part = pl.max_horizontal(abs_(h - delay(h, 1)), abs_(lw - delay(lw, 1)))
    s_a = sumif(part, 12, ~cond1)
    s_b = sumif(part, 12, ~cond2)
    expr = s_a / (s_a + s_b + 1e-7)
    return panel.select(
        expr.alias("gtja_051").cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# gtja_052 — Up-pressure / down-pressure ratio × 100 over 26d
# ---------------------------------------------------------------------------


def gtja_052(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #052 — 26d upward / downward typical-price pressure × 100.

    Guotai Junan Formula
    --------------------
        SUM(MAX(0, H - DELAY((H+L+C)/3, 1)), 26) /
        SUM(MAX(0, DELAY((H+L+C)/3, 1) - L), 26) * 100

    Required panel columns: ``high``, ``low``, ``close``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``momentum``
    """
    typ = (pl.col("high") + pl.col("low") + pl.col("close")) / 3.0
    typ_lag = delay(typ, 1)
    up = pl.max_horizontal(pl.col("high") - typ_lag, pl.lit(0.0))
    dn = pl.max_horizontal(typ_lag - pl.col("low"), pl.lit(0.0))
    expr = sum_(up, 26) / sum_(dn, 26) * 100.0
    return panel.select(
        expr.alias("gtja_052").cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# gtja_053 — COUNT(C > DELAY(C,1), 12) / 12 * 100
# ---------------------------------------------------------------------------


def gtja_053(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #053 — % of up-days over 12d × 100.

    Guotai Junan Formula
    --------------------
        COUNT(CLOSE > DELAY(CLOSE, 1), 12) / 12 * 100

    Required panel columns: ``close``, ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``momentum``
    """
    c = pl.col("close")
    cond = (c > delay(c, 1)).cast(pl.Float64)
    return panel.select(
        (sum_(cond, 12) / 12.0 * 100.0).alias("gtja_053").cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# gtja_054 — -RANK(STD(|C-O|+(C-O)) + CORR(C, O, 10))
# ---------------------------------------------------------------------------


def gtja_054(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #054 — Negated rank of std-of-asymmetric-spread + close-open corr.

    Guotai Junan Formula
    --------------------
        -1 * RANK(STD(|C-O| + (C-O), 10) + CORR(C, O, 10))

    Required panel columns: ``close``, ``open``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volatility``
    """
    diff = pl.col("close") - pl.col("open")
    inner = std_(abs_(diff) + diff, 10) + corr(pl.col("close"), pl.col("open"), 10)
    staged = panel.with_columns(inner.alias("__g054_x"))
    return staged.select(
        (-1.0 * rank(pl.col("__g054_x")))
        .alias("gtja_054")
        .cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# gtja_055 — TR-normalised acceleration sum over 20d
# ---------------------------------------------------------------------------


def gtja_055(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #055 — 20d sum of TR-normalised acceleration × max(|H-C-1|, |L-C-1|).

    Guotai Junan Formula
    --------------------
        SUM(16 * (C - DELAY(C,1) + (C-O)/2 + DELAY(C,1) - DELAY(O,1)) /
            (asymmetric TR normaliser per spec) *
            MAX(|H - DELAY(C,1)|, |L - DELAY(C,1)|), 20)

    Required panel columns: ``open``, ``high``, ``low``, ``close``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``momentum``
    """
    c = pl.col("close")
    o = pl.col("open")
    h = pl.col("high")
    lw = pl.col("low")
    p1 = abs_(h - delay(c, 1))
    p2 = abs_(lw - delay(c, 1))
    p3 = abs_(h - delay(lw, 1))
    p4 = abs_(delay(c, 1) - delay(o, 1))
    var1 = p1 + p2 / 2.0 + p4 / 4.0
    var2 = p2 + p1 / 2.0 + p4 / 4.0
    var3 = p3 + p4 / 4.0
    cond_a = (p1 > p2) & (p1 > p3)
    cond_b = (p2 > p3) & (p2 > p1)
    denom = pl.when(cond_a).then(var1).otherwise(
        pl.when(cond_b).then(var2).otherwise(var3)
    )
    accel = c - delay(c, 1) + (c - o) / 2.0 + delay(c, 1) - delay(o, 1)
    inner = 16.0 * accel / denom * pl.max_horizontal(p1, p2)
    return panel.select(
        sum_(inner, 20).alias("gtja_055").cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# gtja_056 — Rank inequality: (open - tsmin(open, 12)) vs corr-based composite
# ---------------------------------------------------------------------------


def gtja_056(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #056 — Cross-sectional rank inequality: open-min vs corr^5.

    Guotai Junan Formula
    --------------------
        RANK(OPEN - TSMIN(OPEN, 12)) <
        RANK(RANK(CORR(SUM((H+L)/2, 19), SUM(MEAN(V, 40), 19), 13))^5)

    Returns a 0/1 indicator (cast to float).

    Required panel columns: ``open``, ``high``, ``low``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volume_price``
    """
    o = pl.col("open")
    arm1_inner = o - ts_min(o, 12)
    sum_mid = sum_((pl.col("high") + pl.col("low")) / 2.0, 19)
    sum_v = sum_(mean(pl.col("volume"), 40), 19)
    cor = corr(sum_mid, sum_v, 13)
    staged = panel.with_columns(
        arm1_inner.alias("__g056_a1"),
        cor.alias("__g056_c"),
    )
    staged = staged.with_columns(
        rank(pl.col("__g056_a1")).alias("__g056_r1"),
        rank(pl.col("__g056_c")).alias("__g056_rc"),
    )
    staged = staged.with_columns(
        rank(pl.col("__g056_rc") ** 5.0).alias("__g056_r2")
    )
    return staged.select(
        (pl.col("__g056_r1") < pl.col("__g056_r2"))
        .cast(pl.Float64)
        .alias("gtja_056")
    ).to_series()


# ---------------------------------------------------------------------------
# gtja_057 — Smoothed RSV9 (3,1)
# ---------------------------------------------------------------------------


def gtja_057(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #057 — 3-period EWMA of 9-period stochastic %K.

    Guotai Junan Formula
    --------------------
        SMA((C - TSMIN(L, 9)) / (TSMAX(H, 9) - TSMIN(L, 9)) * 100, 3, 1)

    Required panel columns: ``close``, ``low``, ``high``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``momentum``
    """
    raw = (pl.col("close") - ts_min(pl.col("low"), 9)) / (
        ts_max(pl.col("high"), 9) - ts_min(pl.col("low"), 9)
    ) * 100.0
    return panel.select(
        sma(raw, 3, 1).alias("gtja_057").cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# gtja_058 — COUNT(C>DELAY(C,1), 20) / 20 * 100
# ---------------------------------------------------------------------------


def gtja_058(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #058 — % of up-days over 20d × 100 (vwap-anchored).

    Guotai Junan Formula
    --------------------
        COUNT(CLOSE > DELAY(CLOSE, 1), 20) / 20 * 100

    Required panel columns: ``vwap``, ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``momentum``
    """
    v = pl.col("vwap")
    cond = (v > delay(v, 1)).cast(pl.Float64)
    return panel.select(
        (sum_(cond, 20) / 20.0 * 100.0).alias("gtja_058").cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# gtja_059 — 20d sum of close-vs-extreme conditional flow
# ---------------------------------------------------------------------------


def gtja_059(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #059 — 20d sum of close-vs-extreme conditional flow (vwap).

    Guotai Junan Formula
    --------------------
        SUM((C = DELAY(C,1) ? 0 : C - (C > DELAY(C,1) ?
             MIN(L, DELAY(C,1)) : MAX(H, DELAY(C,1)))), 20)

    Required panel columns: ``vwap``, ``low``, ``high``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volume_price``
    """
    v = pl.col("vwap")
    v_lag = delay(v, 1)
    pivot = pl.when(v > v_lag).then(
        pl.min_horizontal(pl.col("low"), v_lag)
    ).otherwise(
        pl.max_horizontal(pl.col("high"), v_lag)
    )
    inner = pl.when(v != v_lag).then(v - pivot).otherwise(0.0)
    return panel.select(
        sum_(inner, 20).alias("gtja_059").cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# gtja_060 — 20d sum of normalised mid-range × volume
# ---------------------------------------------------------------------------


def gtja_060(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #060 — 20d sum of normalised mid-range × volume.

    Guotai Junan Formula
    --------------------
        SUM(((C - L) - (H - C)) / (H - L) * VOLUME, 20)

    Required panel columns: ``close``, ``low``, ``high``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``volume_price``
    """
    base = (
        (pl.col("close") - pl.col("low")) - (pl.col("high") - pl.col("close"))
    ) / (pl.col("high") - pl.col("low"))
    return panel.select(
        sum_(base * pl.col("volume"), 20).alias("gtja_060").cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_DOC_BASE = "docs/factor_library/gtja191"
_REF_BASE = "Guotai Junan 2017, '191 Alphas', via Daic115/alpha191 (formula only)"

_ENTRIES: list[FactorEntry] = [
    FactorEntry(
        id="gtja_041",
        impl=gtja_041,
        direction="reverse",
        category="momentum",
        description="-Rank(MAX(DELTA(VWAP, 3), 5))",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_041.md",
    ),
    FactorEntry(
        id="gtja_042",
        impl=gtja_042,
        direction="reverse",
        category="volume_price",
        description="-RANK(STD(H, 10)) × CORR(H, V, 10)",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_042.md",
    ),
    FactorEntry(
        id="gtja_043",
        impl=gtja_043,
        direction="normal",
        category="volume_price",
        description="6d signed-volume sum (vwap-anchored direction)",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_043.md",
    ),
    FactorEntry(
        id="gtja_044",
        impl=gtja_044,
        direction="normal",
        category="volume_price",
        description="TS-rank decay corr(low, MA(V,10), 7) + TS-rank decay delta(VWAP, 3)",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_044.md",
    ),
    FactorEntry(
        id="gtja_045",
        impl=gtja_045,
        direction="reverse",
        category="volume_price",
        description="Rank(C0.6+O0.4 delta) × CORR(VWAP, MEAN(V, 150), 15)",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_045.md",
    ),
    FactorEntry(
        id="gtja_046",
        impl=gtja_046,
        direction="reverse",
        category="mean_reversion",
        description="(MA3 + MA6 + MA12 + MA24) / (4 × close)",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_046.md",
    ),
    FactorEntry(
        id="gtja_047",
        impl=gtja_047,
        direction="reverse",
        category="mean_reversion",
        description="Smoothed inverse-RSV: SMA((TSMAX(H,6)-C)/(TSMAX(H,6)-TSMIN(L,6))×100, 9, 1)",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_047.md",
    ),
    FactorEntry(
        id="gtja_048",
        impl=gtja_048,
        direction="reverse",
        category="momentum",
        description="-Rank(3-day sign sum) × SUM(V,5)/SUM(V,20)",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_048.md",
    ),
    FactorEntry(
        id="gtja_049",
        impl=gtja_049,
        direction="reverse",
        category="volatility",
        description="Down-day range share over 12d",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_049.md",
    ),
    FactorEntry(
        id="gtja_050",
        impl=gtja_050,
        direction="reverse",
        category="volatility",
        description="Down-share - Up-share asymmetric range ratio over 12d",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_050.md",
    ),
    FactorEntry(
        id="gtja_051",
        impl=gtja_051,
        direction="reverse",
        category="volatility",
        description="Down-share asymmetric range ratio over 12d",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_051.md",
    ),
    FactorEntry(
        id="gtja_052",
        impl=gtja_052,
        direction="normal",
        category="momentum",
        description="26d typical-price up/down pressure ratio × 100",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_052.md",
    ),
    FactorEntry(
        id="gtja_053",
        impl=gtja_053,
        direction="normal",
        category="momentum",
        description="% of up-days over 12d × 100",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_053.md",
    ),
    FactorEntry(
        id="gtja_054",
        impl=gtja_054,
        direction="reverse",
        category="volatility",
        description="-Rank(STD(|C-O|+(C-O), 10) + CORR(C, O, 10))",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_054.md",
    ),
    FactorEntry(
        id="gtja_055",
        impl=gtja_055,
        direction="normal",
        category="momentum",
        description="20d sum of TR-normalised acceleration × max(|H-C-1|,|L-C-1|)",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_055.md",
    ),
    FactorEntry(
        id="gtja_056",
        impl=gtja_056,
        direction="reverse",
        category="volume_price",
        description="Rank-inequality: open-min vs rank-corr^5 (returns 0/1)",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_056.md",
    ),
    FactorEntry(
        id="gtja_057",
        impl=gtja_057,
        direction="normal",
        category="momentum",
        description="3-period EWMA of 9-period stochastic %K",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_057.md",
    ),
    FactorEntry(
        id="gtja_058",
        impl=gtja_058,
        direction="normal",
        category="momentum",
        description="% of up-days over 20d × 100 (vwap-anchored)",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_058.md",
    ),
    FactorEntry(
        id="gtja_059",
        impl=gtja_059,
        direction="reverse",
        category="volume_price",
        description="20d sum of close-vs-extreme conditional flow (vwap-anchored)",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_059.md",
    ),
    FactorEntry(
        id="gtja_060",
        impl=gtja_060,
        direction="normal",
        category="volume_price",
        description="20d sum of normalised mid-range × volume",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_060.md",
    ),
]

for _e in _ENTRIES:
    register_gtja191(_e)
