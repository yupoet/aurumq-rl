"""Alpha101 — breakout category factors.

Three WorldQuant 101 alphas migrated from
:data:`aurumq.rules.alpha101_library.ALPHA101_FACTORS` with
``category='breakout'``: ``alpha023``, ``alpha054`` and ``alpha095``.
All three carry ``direction='reverse'``.

Each factor takes a sorted (``stock_code``, ``trade_date``) panel and
returns a :class:`pl.Series`. Helpers come from
:mod:`aurumq_rl.factors.alpha101._ops`.
"""
from __future__ import annotations

import polars as pl

from aurumq_rl.factors.registry import FactorEntry, register_alpha101

from ._ops import (
    cs_rank,
    delta,
    if_then_else,
    ts_corr,
    ts_min,
    ts_rank,
    ts_sum,
)

# ---------------------------------------------------------------------------
# alpha023 — High-breakout-conditional negative 2d high change
# ---------------------------------------------------------------------------


def alpha023(panel: pl.DataFrame) -> pl.Series:
    """Alpha #023 — High-breakout-conditional negative 2d high change.

    WorldQuant Formula (Kakushadze 2015, eq. 23)
    --------------------------------------------
        (((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0)

    Legacy AQML Expression
    ----------------------
        If(Ts_Sum(high, 20) / 20 < high, -1 * Delta(high, 2), 0)

    Polars Implementation Notes
    ---------------------------
    1. The condition flags any day whose ``high`` exceeds the 20-day
       average — a breakout candidate.
    2. On such days the alpha equals the **negative** 2-day change of
       ``high``: rising momentum -> negative signal, falling -> positive.
       On non-breakout days the alpha is zero (no opinion).

    Required panel columns: ``high``, ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``breakout``
    """
    avg20_high = ts_sum(pl.col("high"), 20) / 20.0
    delta_high2 = delta(pl.col("high"), 2)
    expr = if_then_else(avg20_high < pl.col("high"), -1.0 * delta_high2, pl.lit(0.0))
    return panel.select(expr.alias("alpha023").cast(pl.Float64)).to_series()


# ---------------------------------------------------------------------------
# alpha054 — Open^5 / Close^5 weighted intraday tail signal
# ---------------------------------------------------------------------------


def alpha054(panel: pl.DataFrame) -> pl.Series:
    """Alpha #054 — Open^5 / Close^5 weighted intraday tail signal.

    WorldQuant Formula (Kakushadze 2015, eq. 54)
    --------------------------------------------
        ((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))

    Legacy AQML Expression
    ----------------------
        -1 * ((low - close) * Power(open, 5)) / ((low - high) * Power(close, 5))

    Polars Implementation Notes
    ---------------------------
    1. Element-wise only — no rolling, no rank.
    2. The denominator ``low - high`` is non-positive on every regular
       trading day (low <= high). It is zero when high == low (a fully
       static day), which yields ``inf``/``nan`` — STHSF reference shows
       the same behaviour, so we don't guard against it.

    Required panel columns: ``low``, ``high``, ``open``, ``close``.

    Direction: ``reverse``
    Category: ``breakout``
    """
    open5 = pl.col("open").pow(5)
    close5 = pl.col("close").pow(5)
    numer = -1.0 * (pl.col("low") - pl.col("close")) * open5
    denom = (pl.col("low") - pl.col("high")) * close5
    expr = numer / denom
    return panel.select(expr.alias("alpha054").cast(pl.Float64)).to_series()


# ---------------------------------------------------------------------------
# alpha095 — Open-trough rank vs medium-term correlation rank-power
# ---------------------------------------------------------------------------


def alpha095(panel: pl.DataFrame) -> pl.Series:
    """Alpha #095 — Open-trough rank vs medium-term correlation rank-power.

    WorldQuant Formula (Kakushadze 2015, eq. 95)
    --------------------------------------------
        (rank((open - ts_min(open, 12.4105))) <
         Ts_Rank((rank(correlation(sum(((high + low) / 2), 19.1351),
                                    sum(adv40, 19.1351), 12.8742))^5), 11.7584))

    Legacy AQML Expression (windows rounded to integers)
    ---------------------------------------------------
        If(Rank(open - Ts_Min(open, 12)) <
           Ts_Rank(Power(Rank(Ts_Corr(Ts_Sum((high + low) / 2, 19),
                                      Ts_Sum(adv40, 19), 13)), 5), 12), 1, 0)

    Polars Implementation Notes
    ---------------------------
    1. Left side: cross-section rank of ``open - ts_min(open, 12)``
       (how high open is above its 12d trough).
    2. Right side: rolling rank of (rank-of-corr ** 5) — a heavy-tailed
       indicator of how unusual the medium-term correlation between
       ``hl2`` sums and ``adv40`` sums has been over the last 12 days.
    3. Two cross-section ranks need staging; the final boolean cast
       returns the WorldQuant 0/1 flag.

    Required panel columns: ``open``, ``high``, ``low``, ``adv40``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``breakout``
    """
    # --- left side: rank(open - ts_min(open, 12)) -------------------------
    left_input = pl.col("open") - ts_min(pl.col("open"), 12)

    # --- right side: ts_rank(rank(corr(...))^5, 12) ----------------------
    hl2 = (pl.col("high") + pl.col("low")) / 2.0
    sum_hl2_19 = ts_sum(hl2, 19)
    sum_adv40_19 = ts_sum(pl.col("adv40"), 19)
    corr = ts_corr(sum_hl2_19, sum_adv40_19, 13)

    # Stage the corr column so we can cross-rank it, then re-stage the
    # rank-pow-5 column for the second time-series rank.
    staged = panel.with_columns(
        left_input.alias("__a095_left"),
        corr.alias("__a095_corr"),
    )
    staged = staged.with_columns(
        cs_rank(pl.col("__a095_left")).alias("__a095_left_rank"),
        cs_rank(pl.col("__a095_corr")).pow(5).alias("__a095_corr_rank5"),
    )
    right = ts_rank(pl.col("__a095_corr_rank5"), 12)
    staged = staged.with_columns(right.alias("__a095_right"))
    expr = if_then_else(
        pl.col("__a095_left_rank") < pl.col("__a095_right"),
        pl.lit(1.0),
        pl.lit(0.0),
    )
    return staged.select(expr.alias("alpha095").cast(pl.Float64)).to_series()


# ---------------------------------------------------------------------------
# Registry self-population — runs at module import time
# ---------------------------------------------------------------------------


_ENTRIES = [
    FactorEntry(
        id="alpha023",
        impl=alpha023,
        direction="reverse",
        category="breakout",
        description="High-breakout-conditional negative 2d high change",
        legacy_aqml_expr="If(Ts_Sum(high, 20) / 20 < high, -1 * Delta(high, 2), 0)",
        references=("Kakushadze 2015, '101 Formulaic Alphas', arXiv:1601.00991, eq. 23",),
        formula_doc_path="docs/factor_library/alpha101/alpha_023.md",
    ),
    FactorEntry(
        id="alpha054",
        impl=alpha054,
        direction="reverse",
        category="breakout",
        description="Open^5 / Close^5 weighted intraday tail signal",
        legacy_aqml_expr=(
            "-1 * ((low - close) * Power(open, 5)) / ((low - high) * Power(close, 5))"
        ),
        references=("Kakushadze 2015, '101 Formulaic Alphas', arXiv:1601.00991, eq. 54",),
        formula_doc_path="docs/factor_library/alpha101/alpha_054.md",
    ),
    FactorEntry(
        id="alpha095",
        impl=alpha095,
        direction="reverse",
        category="breakout",
        description="Open-trough rank vs medium-term correlation rank-power",
        legacy_aqml_expr=(
            "If(Rank(open - Ts_Min(open, 12)) < Ts_Rank(Power(Rank(Ts_Corr("
            "Ts_Sum((high + low) / 2, 19), Ts_Sum(adv40, 19), 13)), 5), 12), 1, 0)"
        ),
        references=("Kakushadze 2015, '101 Formulaic Alphas', arXiv:1601.00991, eq. 95",),
        formula_doc_path="docs/factor_library/alpha101/alpha_095.md",
    ),
]

for _e in _ENTRIES:
    register_alpha101(_e)
