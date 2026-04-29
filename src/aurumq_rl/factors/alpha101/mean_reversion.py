"""Alpha101 — mean_reversion category factors.

Eleven factors translated from the legacy AQML string-expression library to
self-contained polars implementations:

* WorldQuant 101 originals: alpha004, alpha032, alpha033, alpha037, alpha041,
  alpha042, alpha053, alpha057, alpha101.
* AurumQ custom mean-reversion factors: alpha_custom_zscore_5d,
  alpha_custom_argmin_recent.

All operators that are not yet promoted to ``_ops.py`` are defined here as
private ``_local_*`` callables to avoid coordinating with sub-agents that own
the shared operator module.

Direction is ``reverse`` for every factor in this module — large positive
values indicate over-extension and predict mean-reverting moves.
"""

from __future__ import annotations

import polars as pl

from aurumq_rl.factors.registry import FactorEntry, register_alpha101

from ._ops import (
    cs_rank,
    cs_scale,
    delay,
    delta,
    ts_argmax_last,
    ts_argmin_last,
    ts_corr_safe,
    ts_decay_linear,
    ts_rank_int,
    ts_sum,
    ts_zscore,
)

# ---------------------------------------------------------------------------
# Factor implementations
# ---------------------------------------------------------------------------


def alpha004(panel: pl.DataFrame) -> pl.Series:
    """Alpha #004 — short-window time-series rank of cross-section rank of low.

    WorldQuant Formula
    ------------------
        -1 * Ts_Rank(rank(low), 9)

    Legacy AQML Expression
    ----------------------
        -1 * Ts_Rank(Rank(low), 9)

    Polars Implementation Notes
    ---------------------------
    1. ``Rank(low)`` is a per-day cross-section pct rank.
    2. ``Ts_Rank(..., 9)`` is the 9-day rolling pct rank (last value within
       window) of that ranked column, per stock.
    3. Final ``-1 *`` flips the sign so high values indicate "low has been
       persistently low" — a contrarian / mean-revert signal.

    Required panel columns: ``low``, ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``mean_reversion``
    """
    staged = panel.with_columns(cs_rank(pl.col("low")).alias("__a004_lr"))
    return staged.select((-1.0 * ts_rank_int(pl.col("__a004_lr"), 9)).alias("alpha004")).to_series()


def alpha032(panel: pl.DataFrame) -> pl.Series:
    """Alpha #032 — short MA divergence + long-horizon vwap × delayed-close corr.

    WorldQuant Formula
    ------------------
        scale((sum(close, 7) / 7 - close)) +
            20 * scale(correlation(vwap, delay(close, 5), 230))

    Legacy AQML Expression
    ----------------------
        Scale(Ts_Sum(close, 7) / 7 - close)
            + 20 * Scale(Ts_Corr(vwap, Delay(close, 5), 230))

    Polars Implementation Notes
    ---------------------------
    1. Two cross-section scaled terms summed; both ``scale`` calls normalise
       ``sum(|x|) == 1`` per trade_date, matching STHSF.
    2. The 230-day correlation is heavy — most synthetic panel rows will be
       NaN. That's fine for parity (reference is also all-NaN here).

    Required panel columns: ``close``, ``vwap``, ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``mean_reversion``
    """
    close = pl.col("close")
    vwap = pl.col("vwap")
    short_ma_dev = (ts_sum(close, 7) / 7.0) - close
    corr = ts_corr_safe(vwap, delay(close, 5), 230)
    staged = panel.with_columns(
        short_ma_dev.alias("__a032_ma"),
        corr.alias("__a032_corr"),
    )
    return staged.select(
        (cs_scale(pl.col("__a032_ma")) + 20.0 * cs_scale(pl.col("__a032_corr"))).alias("alpha032")
    ).to_series()


def alpha033(panel: pl.DataFrame) -> pl.Series:
    """Alpha #033 — cross-section rank of (-1 + open/close).

    WorldQuant Formula
    ------------------
        rank((-1 * ((1 - (open / close))^1)))

    Legacy AQML Expression
    ----------------------
        Rank(-1 * Power(1 - (open / close), 1))

    Polars Implementation Notes
    ---------------------------
    Algebraic simplification used by STHSF: ``-1 * (1 - open/close)``
    collapses to ``open/close - 1``. Cross-section pct rank only.

    Required panel columns: ``open``, ``close``, ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``mean_reversion``
    """
    inner = (pl.col("open") / pl.col("close")) - 1.0
    return panel.select(cs_rank(inner).alias("alpha033")).to_series()


def alpha037(panel: pl.DataFrame) -> pl.Series:
    """Alpha #037 — long-horizon (open-close) vs close correlation + open-close rank.

    WorldQuant Formula
    ------------------
        rank(correlation(delay((open - close), 1), close, 200)) +
            rank((open - close))

    Legacy AQML Expression
    ----------------------
        Rank(Ts_Corr(Delay(open - close, 1), close, 200))
            + Rank(open - close)

    Polars Implementation Notes
    ---------------------------
    The 200-day window will yield NaN on the synthetic panel (60 days). The
    second term still produces values immediately. Both ranks are CS pct.

    Required panel columns: ``open``, ``close``, ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``mean_reversion``
    """
    oc = pl.col("open") - pl.col("close")
    corr = ts_corr_safe(delay(oc, 1), pl.col("close"), 200)
    staged = panel.with_columns(corr.alias("__a037_corr"), oc.alias("__a037_oc"))
    return staged.select(
        (cs_rank(pl.col("__a037_corr")) + cs_rank(pl.col("__a037_oc"))).alias("alpha037")
    ).to_series()


def alpha041(panel: pl.DataFrame) -> pl.Series:
    """Alpha #041 — geometric mean of (high, low) minus vwap.

    WorldQuant Formula
    ------------------
        ((high * low)^0.5) - vwap

    Legacy AQML Expression
    ----------------------
        Power(high * low, 0.5) - vwap

    Polars Implementation Notes
    ---------------------------
    Pure scalar arithmetic — no rolling or cross-section ops. Negative values
    indicate vwap above the geometric mid-price.

    Required panel columns: ``high``, ``low``, ``vwap``

    Direction: ``reverse``
    Category: ``mean_reversion``
    """
    return panel.select(
        ((pl.col("high") * pl.col("low")).sqrt() - pl.col("vwap")).alias("alpha041")
    ).to_series()


def alpha042(panel: pl.DataFrame) -> pl.Series:
    """Alpha #042 — relative rank of vwap-close versus vwap+close.

    WorldQuant Formula
    ------------------
        rank((vwap - close)) / rank((vwap + close))

    Legacy AQML Expression
    ----------------------
        Rank(vwap - close) / Rank(vwap + close)

    Polars Implementation Notes
    ---------------------------
    Two CS ranks; divide. ``rank(vwap + close)`` should never be zero on real
    data but the synthetic panel may produce ties — Polars handles that with
    average-rank semantics.

    Required panel columns: ``vwap``, ``close``, ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``mean_reversion``
    """
    vmc = pl.col("vwap") - pl.col("close")
    vpc = pl.col("vwap") + pl.col("close")
    return panel.select((cs_rank(vmc) / cs_rank(vpc)).alias("alpha042")).to_series()


def alpha053(panel: pl.DataFrame) -> pl.Series:
    """Alpha #053 — 9-day delta of normalised position-within-day.

    WorldQuant Formula
    ------------------
        -1 * delta(((close - low) - (high - close)) / (close - low), 9)

    Legacy AQML Expression
    ----------------------
        -1 * Delta(((close - low) - (high - close)) / (close - low), 9)

    Polars Implementation Notes
    ---------------------------
    1. STHSF guards against ``(close - low) == 0`` by replacing with ``1e-4``.
       We mirror that to avoid division-by-zero NaN.
    2. The inner ratio sits in [-1, 1] (close at low → -1, close at high → 1).

    Required panel columns: ``close``, ``low``, ``high``, ``stock_code``,
    ``trade_date``

    Direction: ``reverse``
    Category: ``mean_reversion``
    """
    close = pl.col("close")
    low = pl.col("low")
    high = pl.col("high")
    cl = close - low
    safe_cl = pl.when(cl == 0.0).then(1e-4).otherwise(cl)
    inner = (cl - (high - close)) / safe_cl
    return panel.select((-1.0 * delta(inner, 9)).alias("alpha053")).to_series()


def alpha057(panel: pl.DataFrame) -> pl.Series:
    """Alpha #057 — premium-to-vwap divided by decayed rank-of-30d-argmax.

    WorldQuant Formula
    ------------------
        0 - (1 * ((close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2)))

    Legacy AQML Expression
    ----------------------
        -1 * ((close - vwap) / Ts_DecayLinear(Rank(Ts_ArgMax(close, 30)), 2))

    Polars Implementation Notes
    ---------------------------
    1. ``ts_argmax(close, 30)``: position of 30-day max — encodes recency of
       the recent peak.
    2. CS rank of that position, then 2-day decay-linear smoothing.
    3. Divide ``close - vwap`` (intraday premium) by the smoothed rank.
    4. Sign flipped so that "premium plus stale peak" is bearish.

    NOTE: Not present in STHSF reference parquet — parity test skipped.

    Required panel columns: ``close``, ``vwap``, ``stock_code``,
    ``trade_date``

    Direction: ``reverse``
    Category: ``mean_reversion``
    """
    close = pl.col("close")
    vwap = pl.col("vwap")
    arg = ts_argmax_last(close, 30)
    staged = panel.with_columns(arg.alias("__a057_arg"))
    staged = staged.with_columns(cs_rank(pl.col("__a057_arg")).alias("__a057_rk"))
    staged = staged.with_columns(ts_decay_linear(pl.col("__a057_rk"), 2).alias("__a057_dl"))
    return staged.select(
        (-1.0 * (close - vwap) / pl.col("__a057_dl")).alias("alpha057")
    ).to_series()


def alpha101(panel: pl.DataFrame) -> pl.Series:
    """Alpha #101 — intraday body over range.

    WorldQuant Formula
    ------------------
        ((close - open) / ((high - low) + 0.001))

    Legacy AQML Expression
    ----------------------
        (close - open) / ((high - low) + 0.001)

    Polars Implementation Notes
    ---------------------------
    The simplest WorldQuant alpha. Despite its name (#101) it is purely
    intraday and ships in STHSF's reference. Acts as a candle-strength
    contrarian — large positive bodies relative to the day's range are
    expected to fade.

    Required panel columns: ``close``, ``open``, ``high``, ``low``

    Direction: ``reverse``
    Category: ``mean_reversion``
    """
    return panel.select(
        ((pl.col("close") - pl.col("open")) / ((pl.col("high") - pl.col("low")) + 0.001)).alias(
            "alpha101"
        )
    ).to_series()


def alpha_custom_zscore_5d(panel: pl.DataFrame) -> pl.Series:
    """AurumQ custom — 5-day rolling z-score of close (sign-flipped).

    Legacy AQML Expression
    ----------------------
        -1 * Ts_Zscore(close, 5)

    Polars Implementation Notes
    ---------------------------
    A simple short-horizon mean-reversion factor: positive values indicate
    close is well below its 5-day mean (after sign flip), suggesting a
    contrarian buy.

    Required panel columns: ``close``, ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``mean_reversion``
    """
    return panel.select(
        (-1.0 * ts_zscore(pl.col("close"), 5)).alias("alpha_custom_zscore_5d")
    ).to_series()


def alpha_custom_argmin_recent(panel: pl.DataFrame) -> pl.Series:
    """AurumQ custom — cross-section rank of 20-day argmin of close.

    Legacy AQML Expression
    ----------------------
        Rank(Ts_ArgMin(close, 20))

    Polars Implementation Notes
    ---------------------------
    ``ts_argmin(close, 20)`` returns the index of the recent 20-day low.
    A larger value (= more recent low) ranks higher; the resulting CS rank
    flags stocks where the recent bottom is fresh — a setup for mean-revert
    bounce trades.

    Required panel columns: ``close``, ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``mean_reversion``
    """
    arg = ts_argmin_last(pl.col("close"), 20)
    staged = panel.with_columns(arg.alias("__a_argmin_recent"))
    return staged.select(
        cs_rank(pl.col("__a_argmin_recent")).alias("alpha_custom_argmin_recent")
    ).to_series()


# ---------------------------------------------------------------------------
# Registry self-population — runs at module import time
# ---------------------------------------------------------------------------

_ENTRIES: tuple[FactorEntry, ...] = (
    FactorEntry(
        id="alpha004",
        impl=alpha004,
        direction="reverse",
        category="mean_reversion",
        description="Negative 9-day Ts_Rank of cross-section rank of low",
        legacy_aqml_expr="-1 * Ts_Rank(Rank(low), 9)",
        references=("Kakushadze 2015, '101 Formulaic Alphas', arXiv:1601.00991, eq. 4",),
    ),
    FactorEntry(
        id="alpha032",
        impl=alpha032,
        direction="reverse",
        category="mean_reversion",
        description=(
            "Scaled 7-day MA divergence + 20x scaled 230-day correlation between "
            "vwap and delayed close"
        ),
        legacy_aqml_expr=(
            "Scale(Ts_Sum(close, 7) / 7 - close) + 20 * Scale(Ts_Corr(vwap, Delay(close, 5), 230))"
        ),
        references=("Kakushadze 2015, '101 Formulaic Alphas', arXiv:1601.00991, eq. 32",),
    ),
    FactorEntry(
        id="alpha033",
        impl=alpha033,
        direction="reverse",
        category="mean_reversion",
        description="Cross-section rank of (open/close - 1)",
        legacy_aqml_expr="Rank(-1 * Power(1 - (open / close), 1))",
        references=("Kakushadze 2015, '101 Formulaic Alphas', arXiv:1601.00991, eq. 33",),
    ),
    FactorEntry(
        id="alpha037",
        impl=alpha037,
        direction="reverse",
        category="mean_reversion",
        description=(
            "Rank of 200-day correlation between delayed (open-close) and close, "
            "plus rank of (open-close)"
        ),
        legacy_aqml_expr=("Rank(Ts_Corr(Delay(open - close, 1), close, 200)) + Rank(open - close)"),
        references=("Kakushadze 2015, '101 Formulaic Alphas', arXiv:1601.00991, eq. 37",),
    ),
    FactorEntry(
        id="alpha041",
        impl=alpha041,
        direction="reverse",
        category="mean_reversion",
        description="Geometric mean of high and low minus vwap",
        legacy_aqml_expr="Power(high * low, 0.5) - vwap",
        references=("Kakushadze 2015, '101 Formulaic Alphas', arXiv:1601.00991, eq. 41",),
    ),
    FactorEntry(
        id="alpha042",
        impl=alpha042,
        direction="reverse",
        category="mean_reversion",
        description="Rank(vwap - close) / Rank(vwap + close)",
        legacy_aqml_expr="Rank(vwap - close) / Rank(vwap + close)",
        references=("Kakushadze 2015, '101 Formulaic Alphas', arXiv:1601.00991, eq. 42",),
    ),
    FactorEntry(
        id="alpha053",
        impl=alpha053,
        direction="reverse",
        category="mean_reversion",
        description="Negative 9-day delta of (close-low minus high-close)/(close-low)",
        legacy_aqml_expr=("-1 * Delta(((close - low) - (high - close)) / (close - low), 9)"),
        references=("Kakushadze 2015, '101 Formulaic Alphas', arXiv:1601.00991, eq. 53",),
    ),
    FactorEntry(
        id="alpha057",
        impl=alpha057,
        direction="reverse",
        category="mean_reversion",
        description=(
            "Negative (close - vwap) divided by 2-day decay-linear of CS rank of "
            "30-day argmax of close"
        ),
        legacy_aqml_expr=("-1 * ((close - vwap) / Ts_DecayLinear(Rank(Ts_ArgMax(close, 30)), 2))"),
        references=("Kakushadze 2015, '101 Formulaic Alphas', arXiv:1601.00991, eq. 57",),
    ),
    FactorEntry(
        id="alpha101",
        impl=alpha101,
        direction="reverse",
        category="mean_reversion",
        description="Intraday body over range — (close - open) / (high - low + 0.001)",
        legacy_aqml_expr="(close - open) / ((high - low) + 0.001)",
        references=("Kakushadze 2015, '101 Formulaic Alphas', arXiv:1601.00991, eq. 101",),
    ),
    FactorEntry(
        id="alpha_custom_zscore_5d",
        impl=alpha_custom_zscore_5d,
        direction="reverse",
        category="mean_reversion",
        description="Negative 5-day rolling z-score of close",
        legacy_aqml_expr="-1 * Ts_Zscore(close, 5)",
    ),
    FactorEntry(
        id="alpha_custom_argmin_recent",
        impl=alpha_custom_argmin_recent,
        direction="reverse",
        category="mean_reversion",
        description="Cross-section rank of 20-day Ts_ArgMin of close",
        legacy_aqml_expr="Rank(Ts_ArgMin(close, 20))",
    ),
)

for _entry in _ENTRIES:
    register_alpha101(_entry)
