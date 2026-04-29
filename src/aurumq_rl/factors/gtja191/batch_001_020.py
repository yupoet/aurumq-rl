"""GTJA-191 factor library — batch 001 through 020.

Each function takes a sorted (``stock_code``, ``trade_date``) panel
:class:`pl.DataFrame` and returns a :class:`pl.Series` aligned to its rows.

Formulas are translated from the Daic115/alpha191 reference (no LICENSE,
formula-only reference — code is NOT vendored). The Guotai Junan 2017
paper (refs/gtja191/wpwp__Alpha-101-GTJA-191/) is the original source.

When a polars expression mixes a TS partition (``stock_code``) and a CS
partition (``trade_date``), we materialise the inner result with
``with_columns(...)`` before applying the outer partition — polars
cannot reliably nest different ``over(...)`` partitions inside one
expression.
"""

from __future__ import annotations

import polars as pl

from aurumq_rl.factors.registry import FactorEntry, register_gtja191

from ._ops import (
    abs_,
    corr,
    delay,
    delta,
    ifelse,
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
# gtja_001 — Volume change rank vs intraday return correlation
# ---------------------------------------------------------------------------


def gtja_001(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #001 — Volume change rank vs intraday return correlation.

    Guotai Junan Formula
    --------------------
        (-1 * CORR(RANK(DELTA(LOG(VOLUME), 1)), RANK(((CLOSE - OPEN) / OPEN)), 6))

    Reference: Daic115/alpha191 alpha191_001 (formula only)

    Required panel columns: ``volume``, ``close``, ``open``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volume_price``
    """
    rv = delta(log_(pl.col("volume")), 1)
    rret = (pl.col("close") - pl.col("open")) / pl.col("open")
    # Stage TS-partitioned columns before applying CS-partitioned rank
    staged = panel.with_columns(
        rv.alias("__g001_dlv"),
        rret.alias("__g001_ret"),
    )
    staged = staged.with_columns(
        rank(pl.col("__g001_dlv")).alias("__g001_rv"),
        rank(pl.col("__g001_ret")).alias("__g001_rret"),
    )
    return staged.select(
        (-1.0 * corr(pl.col("__g001_rv"), pl.col("__g001_rret"), 6))
        .alias("gtja_001")
        .cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# gtja_002 — Williams %R proxy negation
# ---------------------------------------------------------------------------


def gtja_002(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #002 — One-period delta of normalised mid-range.

    Guotai Junan Formula
    --------------------
        (-1 * DELTA((((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW)), 1))

    Required panel columns: ``close``, ``low``, ``high``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``mean_reversion``
    """
    base = ((pl.col("close") - pl.col("low")) - (pl.col("high") - pl.col("close"))) / (
        pl.col("high") - pl.col("low")
    )
    return panel.select((-1.0 * delta(base, 1)).alias("gtja_002").cast(pl.Float64)).to_series()


# ---------------------------------------------------------------------------
# gtja_003 — Conditional close-vs-extreme flow over 6d
# ---------------------------------------------------------------------------


def gtja_003(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #003 — 6-day sum of close-vs-extreme conditional flow.

    Guotai Junan Formula
    --------------------
        SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?
             MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6)

    Required panel columns: ``close``, ``low``, ``high``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``volume_price``
    """
    delay1 = delay(pl.col("close"), 1)
    cond_up = pl.col("close") > delay1
    cond_dn = pl.col("close") < delay1
    pivot = ifelse(
        cond_up,
        pl.min_horizontal(pl.col("low"), delay1),
        pl.max_horizontal(pl.col("high"), delay1),
    )
    inner = pl.when(cond_up | cond_dn).then(pl.col("close") - pivot).otherwise(0.0)
    return panel.select(sum_(inner, 6).alias("gtja_003").cast(pl.Float64)).to_series()


# ---------------------------------------------------------------------------
# gtja_004 — Trend regime + volume gate ternary
# ---------------------------------------------------------------------------


def gtja_004(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #004 — Trend regime conditional with volume gate.

    Guotai Junan Formula
    --------------------
        if((MEAN(CLOSE,8)+STD(CLOSE,8))<MEAN(CLOSE,2)) -1
        elif(MEAN(CLOSE,2)<(MEAN(CLOSE,8)-STD(CLOSE,8))) 1
        elif(VOLUME/MEAN(VOLUME,20) >= 1) 1
        else -1

    Required panel columns: ``close``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``mean_reversion``
    """
    m8 = mean(pl.col("close"), 8)
    s8 = std_(pl.col("close"), 8)
    m2 = mean(pl.col("close"), 2)
    vol_ratio = pl.col("volume") / mean(pl.col("volume"), 20)
    expr = (
        pl.when((m8 + s8) < m2)
        .then(-1.0)
        .otherwise(
            pl.when(m2 < (m8 - s8))
            .then(1.0)
            .otherwise(pl.when(vol_ratio >= 1.0).then(1.0).otherwise(-1.0))
        )
    )
    return panel.select(expr.alias("gtja_004").cast(pl.Float64)).to_series()


# ---------------------------------------------------------------------------
# gtja_005 — TS rolling-rank corr (skipped from reference parquet)
# ---------------------------------------------------------------------------


def gtja_005(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #005 — Negated 3d rolling-max of 5d ts-rank corr.

    Guotai Junan Formula
    --------------------
        (-1 * TSMAX(CORR(TSRANK(VOLUME, 5), TSRANK(HIGH, 5), 5), 3))

    Reference parquet does not include gtja_005 because Daic115 used
    pandas ``rolling.rank`` whose semantics changed across versions.
    Our implementation uses our own ``ts_rank`` (rank of last value in
    the window, polars rolling_rank); reference test will be skipped.

    Required panel columns: ``volume``, ``high``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volume_price``
    """
    staged = panel.with_columns(
        ts_rank(pl.col("volume"), 5).alias("__g005_tv"),
        ts_rank(pl.col("high"), 5).alias("__g005_th"),
    )
    staged = staged.with_columns(
        corr(pl.col("__g005_tv"), pl.col("__g005_th"), 5).alias("__g005_c")
    )
    return staged.select(
        (-1.0 * ts_max(pl.col("__g005_c"), 3)).alias("gtja_005").cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# gtja_006 — Sign of weighted price-change rank, negated
# ---------------------------------------------------------------------------


def gtja_006(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #006 — Negated rank of sign of 4d weighted O/H delta.

    Guotai Junan Formula
    --------------------
        (RANK(SIGN(DELTA((OPEN * 0.85 + HIGH * 0.15), 4))) * -1)

    Required panel columns: ``open``, ``high``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``momentum``
    """
    val = pl.col("open") * 0.85 + pl.col("high") * 0.15
    sgn = sign_(delta(val, 4))
    staged = panel.with_columns(sgn.alias("__g006_sgn"))
    return staged.select(
        (-1.0 * rank(pl.col("__g006_sgn"))).alias("gtja_006").cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# gtja_007 — VWAP-close range × volume-delta rank
# ---------------------------------------------------------------------------


def gtja_007(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #007 — VWAP-close 3d max+min ranks × volume-delta rank.

    Guotai Junan Formula
    --------------------
        (RANK(MAX(VWAP - CLOSE, 3)) + RANK(MIN(VWAP - CLOSE, 3))) *
        RANK(DELTA(VOLUME, 3))

    Required panel columns: ``vwap``, ``close``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``volume_price``
    """
    diff = pl.col("vwap") - pl.col("close")
    staged = panel.with_columns(
        ts_max(diff, 3).alias("__g007_max"),
        ts_min(diff, 3).alias("__g007_min"),
        delta(pl.col("volume"), 3).alias("__g007_dv"),
    )
    return staged.select(
        ((rank(pl.col("__g007_max")) + rank(pl.col("__g007_min"))) * rank(pl.col("__g007_dv")))
        .alias("gtja_007")
        .cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# gtja_008 — Negated rank of 4d delta of weighted price
# ---------------------------------------------------------------------------


def gtja_008(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #008 — Negated rank of 4d delta of HL10+VWAP80 weighted price.

    Guotai Junan Formula
    --------------------
        RANK(DELTA(((HIGH+LOW)/2)*0.2 + VWAP*0.8, 4) * -1)

    Daic115 wrote this as ``-1*(H+L)*0.1 + VWAP*0.8`` (different op
    precedence) which differs from the spec. We follow Daic115 for
    parity with the reference parquet.

    Required panel columns: ``high``, ``low``, ``vwap``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``momentum``
    """
    val = -1.0 * (pl.col("high") + pl.col("low")) * 0.1 + pl.col("vwap") * 0.8
    staged = panel.with_columns(delta(val, 4).alias("__g008_d"))
    return staged.select(rank(pl.col("__g008_d")).alias("gtja_008").cast(pl.Float64)).to_series()


# ---------------------------------------------------------------------------
# gtja_009 — EWMA of mid-price acceleration / volume
# ---------------------------------------------------------------------------


def gtja_009(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #009 — EWMA(7,2) of mid-price acceleration weighted by HL/Volume.

    Guotai Junan Formula
    --------------------
        SMA(((H+L)/2 - (DELAY(H,1)+DELAY(L,1))/2) * (H-L)/VOLUME, 7, 2)

    Required panel columns: ``high``, ``low``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``volume_price``
    """
    mid = (pl.col("high") + pl.col("low")) / 2.0
    mid_prev = (delay(pl.col("high"), 1) + delay(pl.col("low"), 1)) / 2.0
    range_per_vol = (pl.col("high") - pl.col("low")) / pl.col("volume")
    inner = (mid - mid_prev) * range_per_vol
    return panel.select(sma(inner, 7, 2).alias("gtja_009").cast(pl.Float64)).to_series()


# ---------------------------------------------------------------------------
# gtja_010 — Rank of 5d max of conditional return^2
# ---------------------------------------------------------------------------


def gtja_010(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #010 — Rank of conditional 20d-return-std-or-close squared.

    Guotai Junan Formula
    --------------------
        (RANK(MAX(((RET<0)?STD(RET,20):CLOSE)^2),5))

    Daic115 implementation uses ``np.maximum(alpha, 5)`` which is an
    elementwise scalar floor (probably an upstream bug — should be
    ``ts_max(alpha, 5)``). We match Daic115 for reference parity.

    Required panel columns: ``close``, ``returns``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volatility``
    """
    ret = pl.col("returns")
    cond_branch = ifelse(ret < 0.0, std_(ret, 20), pl.col("close"))
    inner = cond_branch * cond_branch
    floored = pl.max_horizontal(inner, pl.lit(5.0))
    staged = panel.with_columns(floored.alias("__g010_inner"))
    return staged.select(
        rank(pl.col("__g010_inner")).alias("gtja_010").cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# gtja_011 — 6d sum of normalised mid-range × volume
# ---------------------------------------------------------------------------


def gtja_011(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #011 — 6-day sum of normalised mid-range times volume.

    Guotai Junan Formula
    --------------------
        SUM((2*CLOSE - LOW - HIGH) / (HIGH - LOW) * VOLUME, 6)

    Required panel columns: ``close``, ``low``, ``high``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``volume_price``
    """
    base = (2.0 * pl.col("close") - pl.col("low") - pl.col("high")) / (
        pl.col("high") - pl.col("low")
    )
    return panel.select(
        sum_(base * pl.col("volume"), 6).alias("gtja_011").cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# gtja_012 — Rank(open - mean(vwap,10)) × -rank(|close-vwap|)
# ---------------------------------------------------------------------------


def gtja_012(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #012 — Rank(O - MA(VWAP,10)) * -1 * Rank(|C - VWAP|).

    Guotai Junan Formula
    --------------------
        RANK(OPEN - SUM(VWAP,10)/10) * (-1 * RANK(ABS(CLOSE - VWAP)))

    Required panel columns: ``open``, ``vwap``, ``close``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``mean_reversion``
    """
    staged = panel.with_columns(
        (pl.col("open") - mean(pl.col("vwap"), 10)).alias("__g012_o"),
        abs_(pl.col("close") - pl.col("vwap")).alias("__g012_d"),
    )
    return staged.select(
        (rank(pl.col("__g012_o")) * (-1.0 * rank(pl.col("__g012_d"))))
        .alias("gtja_012")
        .cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# gtja_013 — sqrt(H*L) - VWAP
# ---------------------------------------------------------------------------


def gtja_013(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #013 — Geometric mean of (H, L) minus VWAP.

    Guotai Junan Formula
    --------------------
        (HIGH * LOW)^0.5 - VWAP

    Required panel columns: ``high``, ``low``, ``vwap``.

    Direction: ``normal``
    Category: ``mean_reversion``
    """
    expr = (pl.col("high") * pl.col("low")) ** 0.5 - pl.col("vwap")
    return panel.select(expr.alias("gtja_013").cast(pl.Float64)).to_series()


# ---------------------------------------------------------------------------
# gtja_014 — close - delay(close, 5)
# ---------------------------------------------------------------------------


def gtja_014(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #014 — 5-day price change.

    Guotai Junan Formula
    --------------------
        CLOSE - DELAY(CLOSE, 5)

    Required panel columns: ``close``, ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``momentum``
    """
    return panel.select(delta(pl.col("close"), 5).alias("gtja_014").cast(pl.Float64)).to_series()


# ---------------------------------------------------------------------------
# gtja_015 — open / delay(close, 1) - 1 (overnight gap)
# ---------------------------------------------------------------------------


def gtja_015(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #015 — Overnight gap return (open / prior close - 1).

    Guotai Junan Formula
    --------------------
        OPEN / DELAY(CLOSE, 1) - 1

    Required panel columns: ``open``, ``close``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``momentum``
    """
    expr = pl.col("open") / delay(pl.col("close"), 1) - 1.0
    return panel.select(expr.alias("gtja_015").cast(pl.Float64)).to_series()


# ---------------------------------------------------------------------------
# gtja_016 — -1 * 5d rolling max of rank(corr(rank(volume), rank(vwap), 5))
# ---------------------------------------------------------------------------


def gtja_016(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #016 — Negated 5d max of rank(volume,vwap)-corr rank.

    Guotai Junan Formula
    --------------------
        -1 * TSMAX(RANK(CORR(RANK(VOLUME), RANK(VWAP), 5)), 5)

    Required panel columns: ``volume``, ``vwap``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volume_price``
    """
    staged = panel.with_columns(
        rank(pl.col("volume")).alias("__g016_rv"),
        rank(pl.col("vwap")).alias("__g016_rw"),
    )
    staged = staged.with_columns(
        corr(pl.col("__g016_rv"), pl.col("__g016_rw"), 5).alias("__g016_c")
    )
    staged = staged.with_columns(rank(pl.col("__g016_c")).alias("__g016_r"))
    return staged.select(
        (-1.0 * ts_max(pl.col("__g016_r"), 5)).alias("gtja_016").cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# gtja_017 — RANK(VWAP - max(VWAP,15)) ^ DELTA(close,5)
# ---------------------------------------------------------------------------


def gtja_017(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #017 — Rank(VWAP - 15d-max-VWAP) raised to 5d close delta.

    Guotai Junan Formula
    --------------------
        RANK(VWAP - MAX(VWAP, 15)) ^ DELTA(CLOSE, 5)

    Required panel columns: ``vwap``, ``close``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``momentum``
    """
    inner = pl.col("vwap") - ts_max(pl.col("vwap"), 15)
    staged = panel.with_columns(inner.alias("__g017_inner"))
    staged = staged.with_columns(rank(pl.col("__g017_inner")).alias("__g017_r"))
    return staged.select(
        (pl.col("__g017_r") ** delta(pl.col("close"), 5)).alias("gtja_017").cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# gtja_018 — close / delay(close,5)
# ---------------------------------------------------------------------------


def gtja_018(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #018 — 5d price ratio (close / delay(close, 5)).

    Guotai Junan Formula
    --------------------
        CLOSE / DELAY(CLOSE, 5)

    Required panel columns: ``close``, ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``momentum``
    """
    expr = pl.col("close") / delay(pl.col("close"), 5)
    return panel.select(expr.alias("gtja_018").cast(pl.Float64)).to_series()


# ---------------------------------------------------------------------------
# gtja_019 — Asymmetric price change ratio
# ---------------------------------------------------------------------------


def gtja_019(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #019 — Asymmetric 6-day price change ratio (vwap-anchored).

    Guotai Junan Formula
    --------------------
        if (VWAP < DELAY(VWAP, 6)) (VWAP - DELAY(VWAP, 6)) / DELAY(VWAP, 6)
        elif VWAP == DELAY(VWAP, 6) 0
        else (VWAP - DELAY(VWAP, 6)) / VWAP

    Daic115 uses VWAP by default (use_vwap=True). We follow that.

    Required panel columns: ``vwap``, ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``mean_reversion``
    """
    vwap = pl.col("vwap")
    vwap_lag = delay(vwap, 6)
    diff = vwap - vwap_lag
    branch_down = diff / vwap_lag
    branch_up = diff / vwap
    expr = pl.when(vwap < vwap_lag).then(branch_down).otherwise(branch_up)
    return panel.select(expr.alias("gtja_019").cast(pl.Float64)).to_series()


# ---------------------------------------------------------------------------
# gtja_020 — 6d % change × 100
# ---------------------------------------------------------------------------


def gtja_020(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #020 — 6-day % change times 100 (vwap-anchored).

    Guotai Junan Formula
    --------------------
        (VWAP - DELAY(VWAP, 6)) / DELAY(VWAP, 6) * 100

    Required panel columns: ``vwap``, ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``momentum``
    """
    vwap = pl.col("vwap")
    vwap_lag = delay(vwap, 6)
    expr = (vwap - vwap_lag) / vwap_lag * 100.0
    return panel.select(expr.alias("gtja_020").cast(pl.Float64)).to_series()


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_DOC_BASE = "docs/factor_library/gtja191"
_REF_BASE = "Guotai Junan 2017, '191 Alphas', via Daic115/alpha191 (formula only)"

_ENTRIES: list[FactorEntry] = [
    FactorEntry(
        id="gtja_001",
        impl=gtja_001,
        direction="reverse",
        category="volume_price",
        description="Volume-change rank vs intraday return rank correlation, 6d, negated",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_001.md",
    ),
    FactorEntry(
        id="gtja_002",
        impl=gtja_002,
        direction="reverse",
        category="mean_reversion",
        description="One-day delta of normalised intraday mid-range, negated",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_002.md",
    ),
    FactorEntry(
        id="gtja_003",
        impl=gtja_003,
        direction="normal",
        category="volume_price",
        description="6d sum of close-vs-extreme conditional flow",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_003.md",
    ),
    FactorEntry(
        id="gtja_004",
        impl=gtja_004,
        direction="normal",
        category="mean_reversion",
        description="Trend regime + volume gate ternary signal",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_004.md",
    ),
    FactorEntry(
        id="gtja_005",
        impl=gtja_005,
        direction="reverse",
        category="volume_price",
        description="Negated 3d max of 5d ts-rank corr (volume vs high)",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_005.md",
        quality_flag=1,
    ),
    FactorEntry(
        id="gtja_006",
        impl=gtja_006,
        direction="reverse",
        category="momentum",
        description="Rank of sign of 4d weighted (open*0.85 + high*0.15) delta, negated",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_006.md",
    ),
    FactorEntry(
        id="gtja_007",
        impl=gtja_007,
        direction="normal",
        category="volume_price",
        description="(Rank max + Rank min) of (vwap-close,3) × Rank(volume delta,3)",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_007.md",
    ),
    FactorEntry(
        id="gtja_008",
        impl=gtja_008,
        direction="reverse",
        category="momentum",
        description="Negated rank of 4d delta of HL10+VWAP80 weighted price (Daic115 parity)",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_008.md",
    ),
    FactorEntry(
        id="gtja_009",
        impl=gtja_009,
        direction="normal",
        category="volume_price",
        description="EWMA(7,2) of mid-price acceleration weighted by (H-L)/Volume",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_009.md",
    ),
    FactorEntry(
        id="gtja_010",
        impl=gtja_010,
        direction="reverse",
        category="volatility",
        description="Rank of MAX5(((ret<0?std20:close))^2) — Daic115 floor-by-5 parity",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_010.md",
    ),
    FactorEntry(
        id="gtja_011",
        impl=gtja_011,
        direction="normal",
        category="volume_price",
        description="6d sum of normalised mid-range × volume",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_011.md",
    ),
    FactorEntry(
        id="gtja_012",
        impl=gtja_012,
        direction="reverse",
        category="mean_reversion",
        description="Rank(O-MA10VWAP) × -Rank(|C-VWAP|)",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_012.md",
    ),
    FactorEntry(
        id="gtja_013",
        impl=gtja_013,
        direction="normal",
        category="mean_reversion",
        description="Geometric mean of (H, L) minus VWAP",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_013.md",
    ),
    FactorEntry(
        id="gtja_014",
        impl=gtja_014,
        direction="normal",
        category="momentum",
        description="5d price change (close - delay(close,5))",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_014.md",
    ),
    FactorEntry(
        id="gtja_015",
        impl=gtja_015,
        direction="normal",
        category="momentum",
        description="Overnight gap return: open / prior close - 1",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_015.md",
    ),
    FactorEntry(
        id="gtja_016",
        impl=gtja_016,
        direction="reverse",
        category="volume_price",
        description="-1 × TSMAX(rank(corr(rank(vol), rank(vwap), 5)), 5)",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_016.md",
    ),
    FactorEntry(
        id="gtja_017",
        impl=gtja_017,
        direction="reverse",
        category="momentum",
        description="Rank(vwap - max15(vwap)) ^ delta(close, 5)",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_017.md",
    ),
    FactorEntry(
        id="gtja_018",
        impl=gtja_018,
        direction="normal",
        category="momentum",
        description="5d close ratio: close / delay(close, 5)",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_018.md",
    ),
    FactorEntry(
        id="gtja_019",
        impl=gtja_019,
        direction="normal",
        category="mean_reversion",
        description="Asymmetric 6d vwap change ratio",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_019.md",
    ),
    FactorEntry(
        id="gtja_020",
        impl=gtja_020,
        direction="normal",
        category="momentum",
        description="6d vwap % change × 100",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_020.md",
    ),
]

for _e in _ENTRIES:
    register_gtja191(_e)
