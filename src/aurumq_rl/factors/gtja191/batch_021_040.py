"""GTJA-191 factor library — batch 021 through 040.

Translated from Daic115/alpha191 (formula reference, no code vendored).
"""
from __future__ import annotations

import polars as pl

from aurumq_rl.factors.registry import FactorEntry, register_gtja191

from ._ops import (
    corr,
    decay_linear,
    delay,
    delta,
    ifelse,
    mean,
    rank,
    regbeta,
    sma,
    std_,
    sum_,
    ts_max,
    ts_min,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stock_row_index(panel: pl.DataFrame) -> pl.Expr:
    """Per-stock cumulative row counter (0-based), expressed as Float64.

    ``regbeta(y, _stock_row_index(...), n)`` returns the rolling slope of
    ``y`` against ``[k, k+1, …, k+n-1]``, which (by translation
    invariance of covariance) equals the slope against the natural
    SEQUENCE(1..n) used in the Guotai Junan paper.
    """
    return pl.int_range(pl.len(), dtype=pl.Int64).over("stock_code").cast(pl.Float64)


# ---------------------------------------------------------------------------
# gtja_021 — REGBETA(MEAN(CLOSE,6), SEQUENCE(6))
# ---------------------------------------------------------------------------


def gtja_021(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #021 — Rolling 6d slope of MEAN(close, 6) vs sequence.

    Guotai Junan Formula
    --------------------
        REGBETA(MEAN(CLOSE, 6), SEQUENCE(6))

    Daic115 uses ``qlib.data.ops.rolling_slope`` (cython). We compute
    the same number via ``regbeta(y, x, 6)`` where ``x`` is a per-stock
    row counter — the rolling slope is invariant to additive shifts of
    ``x``. Reference parquet does NOT include gtja_021 because the
    Daic115 builder skipped qlib-dependent alphas; reference test is
    skipped.

    Required panel columns: ``vwap``, ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``momentum``
    """
    val_mean = mean(pl.col("vwap"), 6)
    staged = panel.with_columns(
        val_mean.alias("__g021_y"),
        _stock_row_index(panel).alias("__g021_x"),
    )
    return staged.select(
        regbeta(pl.col("__g021_y"), pl.col("__g021_x"), 6)
        .alias("gtja_021")
        .cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# gtja_022 — SMA of (close mean-detrend - lag-3) over 12d
# ---------------------------------------------------------------------------


def gtja_022(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #022 — EWMA(12,1) of (close mean-detrend - lag3).

    Guotai Junan Formula
    --------------------
        SMA((C - MEAN(C,6))/MEAN(C,6) -
             DELAY((C - MEAN(C,6))/MEAN(C,6), 3), 12, 1)

    Required panel columns: ``vwap``, ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``mean_reversion``
    """
    vwap = pl.col("vwap")
    val_mean = mean(vwap, 6)
    detrend = (vwap - val_mean) / val_mean
    diff = detrend - delay(detrend, 3)
    return panel.select(
        sma(diff, 12, 1).alias("gtja_022").cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# gtja_023 — Up vs total smoothed STD share × 100
# ---------------------------------------------------------------------------


def gtja_023(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #023 — Up-day std share over 20d, smoothed by SMA(20,1).

    Guotai Junan Formula
    --------------------
        SMA(cond ? STD(C,20) : 0, 20, 1) /
        (SMA(cond ? STD(C,20) : 0, 20, 1) + SMA(!cond ? STD(C,20) : 0, 20, 1)) * 100
        where cond = C > DELAY(C, 1)

    Required panel columns: ``vwap``, ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``volatility``
    """
    vwap = pl.col("vwap")
    cond = vwap > delay(vwap, 1)
    s = std_(vwap, 20)
    up = ifelse(cond, s, 0.0)
    dn = ifelse(~cond, s, 0.0)
    sma_up = sma(up, 20, 1)
    sma_dn = sma(dn, 20, 1)
    expr = sma_up / (sma_up + sma_dn) * 100.0
    return panel.select(
        expr.alias("gtja_023").cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# gtja_024 — SMA(C - DELAY(C,5), 5, 1)
# ---------------------------------------------------------------------------


def gtja_024(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #024 — EWMA(5,1) of 5d price change.

    Guotai Junan Formula
    --------------------
        SMA(CLOSE - DELAY(CLOSE, 5), 5, 1)

    Required panel columns: ``vwap``, ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``momentum``
    """
    vwap = pl.col("vwap")
    diff = vwap - delay(vwap, 5)
    return panel.select(
        sma(diff, 5, 1).alias("gtja_024").cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# gtja_025 — multi-component momentum-volume composite
# ---------------------------------------------------------------------------


def gtja_025(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #025 — Composite of close-delta rank, volume EWMA rank, return-sum rank.

    Guotai Junan Formula
    --------------------
        (-1 * RANK(DELTA(CLOSE, 7) * (1 - RANK(DECAYLINEAR(VOLUME / MEAN(VOLUME, 20), 9))))) *
        (1 + RANK(SUM(RET, 250)))

    Daic115 uses ``ewm(alpha=1/9)`` for the decay step (not the linear-
    weighted DECAYLINEAR), and `period=150` instead of 250 by default.
    We follow Daic115 (period=150, ewma instead of decay_linear).

    Required panel columns: ``vwap``, ``volume``, ``returns``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``momentum``
    """
    vwap = pl.col("vwap")
    ret_sum = sum_(pl.col("returns"), 150)
    vol_ratio = pl.col("volume") / mean(pl.col("volume"), 20)
    vol_ewma = sma(vol_ratio, 9, 1)  # ewm(alpha=1/9)
    staged = panel.with_columns(
        delta(vwap, 7).alias("__g025_d7"),
        vol_ewma.alias("__g025_ve"),
        ret_sum.alias("__g025_rs"),
    )
    staged = staged.with_columns(
        rank(pl.col("__g025_d7")).alias("__g025_rd"),
        rank(pl.col("__g025_ve")).alias("__g025_rv"),
        rank(pl.col("__g025_rs")).alias("__g025_rr"),
    )
    expr = (
        -1.0
        * pl.col("__g025_rd")
        * pl.col("__g025_rv")
        * (1.0 + pl.col("__g025_rr"))
    )
    return staged.select(
        expr.alias("gtja_025").cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# gtja_026 — Mean(close,12) - close + corr(vwap, delay(close,5), 200)
# ---------------------------------------------------------------------------


def gtja_026(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #026 — Long-window VWAP/close-lag correlation + mean reversion.

    Guotai Junan Formula
    --------------------
        (MEAN(CLOSE, 7) - CLOSE) + CORR(VWAP, DELAY(CLOSE, 5), 230)

    Daic115 default: ma_period=12, corr_period=200.

    Required panel columns: ``close``, ``vwap``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``mean_reversion``
    """
    expr = (
        mean(pl.col("close"), 12)
        - pl.col("close")
        + corr(pl.col("vwap"), delay(pl.col("close"), 5), 200)
    )
    return panel.select(
        expr.alias("gtja_026").cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# gtja_027 — WMA of two-window momentum (SMA(.,12,1) actually per Daic115)
# ---------------------------------------------------------------------------


def gtja_027(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #027 — EWMA(12,1) of 3d+6d % momentum sum.

    Guotai Junan Formula
    --------------------
        WMA(((C/DELAY(C,3) - 1)*100 + (C/DELAY(C,6) - 1)*100), 12)

    Daic115 substitutes WMA with SMA(.,12,1) — we follow that.

    Required panel columns: ``vwap``, ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``momentum``
    """
    vwap = pl.col("vwap")
    short = (vwap / delay(vwap, 3) - 1.0) * 100.0
    long_ = (vwap / delay(vwap, 6) - 1.0) * 100.0
    return panel.select(
        sma(short + long_, 12, 1).alias("gtja_027").cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# gtja_028 — KDJ-style stochastic oscillator
# ---------------------------------------------------------------------------


def gtja_028(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #028 — KDJ-style 9d stochastic with two SMA layers.

    Guotai Junan Formula
    --------------------
        3 * SMA((C - TSMIN(L,9)) / (TSMAX(H,9) - TSMIN(L,9)) * 100, 3, 1) -
        2 * SMA(SMA((C - TSMIN(L,9)) / (TSMAX(H,9) - TSMIN(L,9)) * 100, 3, 1), 3, 1)

    Required panel columns: ``close``, ``low``, ``high``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``momentum``
    """
    low_min = ts_min(pl.col("low"), 9)
    high_max = ts_max(pl.col("high"), 9)
    raw = (pl.col("close") - low_min) / (high_max - low_min) * 100.0
    sma1 = sma(raw, 3, 1)
    sma2 = sma(sma1, 3, 1)
    return panel.select(
        (3.0 * sma1 - 2.0 * sma2).alias("gtja_028").cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# gtja_029 — 6d % change × log(volume)
# ---------------------------------------------------------------------------


def gtja_029(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #029 — 6d % change × log(volume).

    Guotai Junan Formula
    --------------------
        (CLOSE - DELAY(CLOSE, 6)) / DELAY(CLOSE, 6) * VOLUME

    Daic115 uses ``log(volume)`` rather than raw volume — we follow.

    Required panel columns: ``vwap``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``volume_price``
    """
    vwap = pl.col("vwap")
    expr = (vwap - delay(vwap, 6)) / delay(vwap, 6) * pl.col("volume").log()
    return panel.select(
        expr.alias("gtja_029").cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# gtja_030 — STUB (Daic115 marks unfinished)
# ---------------------------------------------------------------------------


def gtja_030(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #030 — STUB (Fama-French residual^2 WMA).

    Guotai Junan Formula
    --------------------
        WMA((REGRESI(CLOSE/DELAY(CLOSE)-1, MKT, SMB, HML, 60))^2, 20)

    Daic115 marks this as ``unfinished=True`` and returns ``None``.
    A faithful implementation requires Fama-French factor exposures
    which the synthetic panel does not provide. We return an all-NaN
    series and tag with quality_flag=2 (stub). Reference parquet does
    not include gtja_030; reference test is skipped.

    Direction: ``normal``
    Category: ``volatility``
    """
    return panel.select(
        (pl.col("close") * 0.0 + float("nan")).alias("gtja_030").cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# gtja_031 — (close - mean(close,12)) / mean(close,12) * 100
# ---------------------------------------------------------------------------


def gtja_031(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #031 — 12d mean-distance ratio × 100 (vwap-anchored).

    Guotai Junan Formula
    --------------------
        (CLOSE - MEAN(CLOSE, 12)) / MEAN(CLOSE, 12) * 100

    Required panel columns: ``vwap``, ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``mean_reversion``
    """
    vwap = pl.col("vwap")
    m12 = mean(vwap, 12)
    return panel.select(
        ((vwap - m12) / m12 * 100.0).alias("gtja_031").cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# gtja_032 — -1 * SUM(RANK(CORR(RANK(H), RANK(VOL), 3)), 3)
# ---------------------------------------------------------------------------


def gtja_032(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #032 — Negated 3d sum of rank(corr(rank-H, rank-Vol, 3)).

    Guotai Junan Formula
    --------------------
        -1 * SUM(RANK(CORR(RANK(HIGH), RANK(VOLUME), 3)), 3)

    Required panel columns: ``high``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volume_price``
    """
    staged = panel.with_columns(
        rank(pl.col("high")).alias("__g032_rh"),
        rank(pl.col("volume")).alias("__g032_rv"),
    )
    staged = staged.with_columns(
        corr(pl.col("__g032_rh"), pl.col("__g032_rv"), 3).alias("__g032_c")
    )
    staged = staged.with_columns(rank(pl.col("__g032_c")).alias("__g032_rc"))
    return staged.select(
        (-1.0 * sum_(pl.col("__g032_rc"), 3))
        .alias("gtja_032")
        .cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# gtja_033 — Low-min × ret-sum × turnover (turn proxy)
# ---------------------------------------------------------------------------


def gtja_033(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #033 — Long/medium return spread × low-min change × turnover rank.

    Guotai Junan Formula
    --------------------
        (((-1 * TSMIN(LOW, 5)) + DELAY(TSMIN(LOW, 5), 5)) *
         RANK((SUM(RET, 240) - SUM(RET, 20)) / 220)) * TSRANK(VOLUME, 5)

    Daic115 references ``data["turn"]`` (turnover_rate) which we don't
    have. Use ``amount / cap`` as proxy. Reference parquet does NOT
    include gtja_033; reference test is skipped.

    Required panel columns: ``low``, ``returns``, ``amount``, ``cap``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``volume_price``
    """
    low_min = ts_min(pl.col("low"), 5)
    ret = pl.col("returns")
    ret_sum = sum_(ret, 240) - sum_(ret, 20)
    turn_proxy = pl.col("amount") / pl.col("cap")
    staged = panel.with_columns(
        ret_sum.alias("__g033_rs"),
        turn_proxy.alias("__g033_tp"),
    )
    staged = staged.with_columns(
        rank(pl.col("__g033_rs")).alias("__g033_rr"),
        rank(pl.col("__g033_tp")).alias("__g033_rt"),
    )
    expr = (
        ((-1.0 * low_min) + delay(low_min, 5))
        * pl.col("__g033_rr")
        * delay(pl.col("__g033_rt"), 5)
    )
    return staged.select(
        expr.alias("gtja_033").cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# gtja_034 — MEAN(CLOSE, 12) / CLOSE
# ---------------------------------------------------------------------------


def gtja_034(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #034 — 12d MA over current price ratio.

    Guotai Junan Formula
    --------------------
        MEAN(CLOSE, 12) / CLOSE

    Required panel columns: ``vwap``, ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``mean_reversion``
    """
    vwap = pl.col("vwap")
    return panel.select(
        (mean(vwap, 12) / vwap).alias("gtja_034").cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# gtja_035 — MIN(rank decay open delta, rank decay vol-corr) negated
# ---------------------------------------------------------------------------


def gtja_035(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #035 — Min of two decay-linear/EWMA-rank arms, negated.

    Guotai Junan Formula
    --------------------
        MIN(
          RANK(DECAYLINEAR(DELTA(OPEN, 1), 15)),
          RANK(DECAYLINEAR(CORR(VOLUME, OPEN*0.65 + CLOSE*0.35, 17), 7))
        ) * -1

    Daic115 substitutes DECAYLINEAR with EWMA(alpha=1/n). We follow.

    Required panel columns: ``open``, ``close``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volume_price``
    """
    open_d = delta(pl.col("open"), 1)
    part1_inner = sma(open_d, 15, 1)
    weighted = pl.col("open") * 0.65 + pl.col("close") * 0.35
    cor_inner = corr(pl.col("volume"), weighted, 17)
    part2_inner = sma(cor_inner, 7, 1)
    staged = panel.with_columns(
        part1_inner.alias("__g035_p1"),
        part2_inner.alias("__g035_p2"),
    )
    staged = staged.with_columns(
        rank(pl.col("__g035_p1")).alias("__g035_r1"),
        rank(pl.col("__g035_p2")).alias("__g035_r2"),
    )
    expr = pl.min_horizontal(pl.col("__g035_r1"), pl.col("__g035_r2")) * -1.0
    return staged.select(
        expr.alias("gtja_035").cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# gtja_036 — Rank of 2d sum of corr(rank-vol, rank-vwap, 6)
# ---------------------------------------------------------------------------


def gtja_036(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #036 — Rank of 2d sum of rank-volume/rank-vwap 6d corr.

    Guotai Junan Formula
    --------------------
        RANK(SUM(CORR(RANK(VOLUME), RANK(VWAP), 6), 2))

    Required panel columns: ``volume``, ``vwap``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``volume_price``
    """
    staged = panel.with_columns(
        rank(pl.col("volume")).alias("__g036_rv"),
        rank(pl.col("vwap")).alias("__g036_rw"),
    )
    staged = staged.with_columns(
        corr(pl.col("__g036_rv"), pl.col("__g036_rw"), 6).alias("__g036_c")
    )
    staged = staged.with_columns(sum_(pl.col("__g036_c"), 2).alias("__g036_s"))
    return staged.select(
        rank(pl.col("__g036_s")).alias("gtja_036").cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# gtja_037 — -1 * Rank((Sum(O,5)*Sum(R,5)) - delay(Sum(O,5)*Sum(R,5), 10))
# ---------------------------------------------------------------------------


def gtja_037(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #037 — Negated rank of 10d acceleration of (Sum(open,5) × Sum(ret,5)).

    Guotai Junan Formula
    --------------------
        -1 * RANK((SUM(OPEN, 5) * SUM(RET, 5)) - DELAY(SUM(OPEN, 5) * SUM(RET, 5), 10))

    Daic115 uses ``ret = vwap/delay(vwap)-1`` not `returns` column —
    these differ by adj_factor + vwap vs close. We use the panel
    ``returns`` column for consistency.

    Required panel columns: ``open``, ``returns``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``momentum``
    """
    vwap = pl.col("vwap")
    ret = vwap / delay(vwap, 1) - 1.0
    product = sum_(pl.col("open"), 5) * sum_(ret, 5)
    accel = product - delay(product, 10)
    staged = panel.with_columns(accel.alias("__g037_a"))
    return staged.select(
        (-1.0 * rank(pl.col("__g037_a")))
        .alias("gtja_037")
        .cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# gtja_038 — (MEAN(H,20)<H ? -DELTA(H,2) : 0)
# ---------------------------------------------------------------------------


def gtja_038(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #038 — Conditional negated 2d high delta when high>20d-MA.

    Guotai Junan Formula
    --------------------
        (MEAN(HIGH, 20) < HIGH) ? (-1 * DELTA(HIGH, 2)) : 0

    Required panel columns: ``high``, ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``mean_reversion``
    """
    cond = mean(pl.col("high"), 20) < pl.col("high")
    expr = pl.when(cond).then(-1.0 * delta(pl.col("high"), 2)).otherwise(0.0)
    return panel.select(
        expr.alias("gtja_038").cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# gtja_039 — Negated diff of two rank-decay arms
# ---------------------------------------------------------------------------


def gtja_039(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #039 — Difference of two rank(decay-linear) arms, negated.

    Guotai Junan Formula
    --------------------
        (RANK(DECAYLINEAR(DELTA(CLOSE, 2), 8)) -
         RANK(DECAYLINEAR(CORR(VWAP*0.3 + OPEN*0.7,
                              SUM(MEAN(VOLUME, 180), 37), 14), 12))) * -1

    Required panel columns: ``close``, ``vwap``, ``open``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``momentum``
    """
    p1_inner = decay_linear(delta(pl.col("close"), 2), 8)
    weighted = pl.col("vwap") * 0.3 + pl.col("open") * 0.7
    vol_long = sum_(mean(pl.col("volume"), 180), 37)
    cor = corr(weighted, vol_long, 14)
    p2_inner = decay_linear(cor, 12)
    staged = panel.with_columns(
        p1_inner.alias("__g039_p1"),
        p2_inner.alias("__g039_p2"),
    )
    staged = staged.with_columns(
        rank(pl.col("__g039_p1")).alias("__g039_r1"),
        rank(pl.col("__g039_p2")).alias("__g039_r2"),
    )
    expr = (pl.col("__g039_r1") - pl.col("__g039_r2")) * -1.0
    return staged.select(
        expr.alias("gtja_039").cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# gtja_040 — Volume-where-up vs volume-where-down ratio × 100 over 26d
# ---------------------------------------------------------------------------


def gtja_040(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #040 — 26d up/down volume ratio × 100.

    Guotai Junan Formula
    --------------------
        SUM((C > DELAY(C, 1) ? VOLUME : 0), 26) /
        SUM((C <= DELAY(C, 1) ? VOLUME : 0), 26) * 100

    Required panel columns: ``vwap``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``volume_price``
    """
    vwap = pl.col("vwap")
    cond = vwap > delay(vwap, 1)
    up = ifelse(cond, pl.col("volume"), 0.0)
    dn = ifelse(~cond, pl.col("volume"), 0.0)
    expr = sum_(up, 26) / sum_(dn, 26) * 100.0
    return panel.select(
        expr.alias("gtja_040").cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_DOC_BASE = "docs/factor_library/gtja191"
_REF_BASE = "Guotai Junan 2017, '191 Alphas', via Daic115/alpha191 (formula only)"

_ENTRIES: list[FactorEntry] = [
    FactorEntry(
        id="gtja_021",
        impl=gtja_021,
        direction="reverse",
        category="momentum",
        description="Rolling 6d slope of MEAN(close,6) vs sequence (regbeta vs row-index)",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_021.md",
        quality_flag=1,
    ),
    FactorEntry(
        id="gtja_022",
        impl=gtja_022,
        direction="reverse",
        category="mean_reversion",
        description="EWMA(12,1) of (close mean-detrend - 3d-lag) over 6d window",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_022.md",
    ),
    FactorEntry(
        id="gtja_023",
        impl=gtja_023,
        direction="normal",
        category="volatility",
        description="Up-day STD share over 20d (×100), smoothed via SMA(20,1)",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_023.md",
    ),
    FactorEntry(
        id="gtja_024",
        impl=gtja_024,
        direction="normal",
        category="momentum",
        description="EWMA(5,1) of 5d price change",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_024.md",
    ),
    FactorEntry(
        id="gtja_025",
        impl=gtja_025,
        direction="reverse",
        category="momentum",
        description="-Rank(close7d-delta) × Rank(EWMA volume ratio) × (1+Rank(150d ret sum))",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_025.md",
    ),
    FactorEntry(
        id="gtja_026",
        impl=gtja_026,
        direction="normal",
        category="mean_reversion",
        description="(MEAN(close,12) - close) + 200d corr(vwap, delay(close,5))",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_026.md",
    ),
    FactorEntry(
        id="gtja_027",
        impl=gtja_027,
        direction="normal",
        category="momentum",
        description="EWMA(12,1) of 3d+6d % momentum sum (Daic115 SMA-substituted-WMA)",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_027.md",
    ),
    FactorEntry(
        id="gtja_028",
        impl=gtja_028,
        direction="normal",
        category="momentum",
        description="KDJ-style 9d stochastic with two SMA(3,1) layers",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_028.md",
    ),
    FactorEntry(
        id="gtja_029",
        impl=gtja_029,
        direction="normal",
        category="volume_price",
        description="6d % change × log(volume) (Daic115 log-vol substitution)",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_029.md",
    ),
    FactorEntry(
        id="gtja_030",
        impl=gtja_030,
        direction="normal",
        category="volatility",
        description="STUB — Fama-French residual^2 WMA (Daic115 unfinished)",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_030.md",
        quality_flag=2,
    ),
    FactorEntry(
        id="gtja_031",
        impl=gtja_031,
        direction="reverse",
        category="mean_reversion",
        description="12d MA-distance ratio × 100",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_031.md",
    ),
    FactorEntry(
        id="gtja_032",
        impl=gtja_032,
        direction="reverse",
        category="volume_price",
        description="-SUM(rank(corr(rank-H, rank-Vol, 3)), 3)",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_032.md",
    ),
    FactorEntry(
        id="gtja_033",
        impl=gtja_033,
        direction="normal",
        category="volume_price",
        description="Low-min change × ret-spread rank × turnover-proxy rank (amount/cap)",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_033.md",
        quality_flag=1,
    ),
    FactorEntry(
        id="gtja_034",
        impl=gtja_034,
        direction="reverse",
        category="mean_reversion",
        description="MEAN(close, 12) / close",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_034.md",
    ),
    FactorEntry(
        id="gtja_035",
        impl=gtja_035,
        direction="reverse",
        category="volume_price",
        description="Min of two rank(decay) arms (open delta + vol-weighted-price corr) negated",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_035.md",
    ),
    FactorEntry(
        id="gtja_036",
        impl=gtja_036,
        direction="normal",
        category="volume_price",
        description="Rank of 2d sum of corr(rank-volume, rank-vwap, 6)",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_036.md",
    ),
    FactorEntry(
        id="gtja_037",
        impl=gtja_037,
        direction="reverse",
        category="momentum",
        description="-Rank of 10d acceleration of (sum(open,5) × sum(ret,5))",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_037.md",
    ),
    FactorEntry(
        id="gtja_038",
        impl=gtja_038,
        direction="reverse",
        category="mean_reversion",
        description="Conditional negated 2d high-delta when high>MEAN(high,20)",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_038.md",
    ),
    FactorEntry(
        id="gtja_039",
        impl=gtja_039,
        direction="reverse",
        category="momentum",
        description="-(rank-decay close-delta - rank-decay long-vol corr)",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_039.md",
    ),
    FactorEntry(
        id="gtja_040",
        impl=gtja_040,
        direction="normal",
        category="volume_price",
        description="26d up-volume / down-volume ratio × 100",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_040.md",
    ),
]

for _e in _ENTRIES:
    register_gtja191(_e)
