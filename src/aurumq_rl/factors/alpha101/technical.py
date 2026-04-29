"""Alpha101 — technical category factors.

The single technical-category alpha (``alpha092``) lives here. It is the
``Min(Ts_Rank(...), Ts_Rank(...))`` composite from
:data:`aurumq.rules.alpha101_library.ALPHA101_FACTORS`. Direction is
``reverse``.

The factor mixes time-series rank, decay-linear weighted moving average,
rolling correlation, and cross-section rank, so several intermediates
must be materialised between TS- and CS-partitioned operators.
"""
from __future__ import annotations

import polars as pl

from aurumq_rl.factors.registry import FactorEntry, register_alpha101

from ._ops import (
    cs_rank,
    if_then_else,
    ts_corr,
    ts_decay_linear,
    ts_rank,
)

# ---------------------------------------------------------------------------
# alpha092 — Pattern-flag rank vs low-adv30 rank correlation
# ---------------------------------------------------------------------------


def alpha092(panel: pl.DataFrame) -> pl.Series:
    """Alpha #092 — Pattern-flag rank vs low-adv30 rank correlation.

    WorldQuant Formula (Kakushadze 2015, eq. 92)
    --------------------------------------------
        Min(Ts_Rank(decay_linear(((((high + low) / 2) + close) < (low + open)),
                                 14.7221), 18.7484),
            Ts_Rank(decay_linear(correlation(rank(low), rank(adv30), 7.58553),
                                 6.94279), 6.80584))

    Legacy AQML Expression (windows rounded to integers)
    ---------------------------------------------------
        Min(Ts_Rank(Ts_DecayLinear(If((high + low) / 2 + close < low + open, 1.0, 0.0),
                                   15), 19),
            Ts_Rank(Ts_DecayLinear(Ts_Corr(Rank(low), Rank(adv30), 8), 7), 7))

    Polars Implementation Notes
    ---------------------------
    1. **Left branch** — a binary pattern flag (``hl2 + close < low + open``)
       smoothed by a 15-day decay-linear MA, then ranked over 19 days.
       The condition triggers when the candle has a high upper-wick close
       (rough proxy for distribution).
    2. **Right branch** — rolling 8-day correlation between cross-section
       rank of ``low`` and rank of ``adv30``, smoothed by a 7-day
       decay-linear MA, then 7-day rolling rank.
    3. The output is the **element-wise min** of the two ranks. We use
       ``pl.min_horizontal`` for clarity.
    4. Two ``cs_rank`` partitions appear (rank of ``low``, rank of
       ``adv30``) — both must be materialised before being fed into the
       8-window ``ts_corr``.

    Required panel columns: ``high``, ``low``, ``open``, ``close``,
    ``adv30``, ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``technical``
    """
    # --- left branch: ts_rank(decay(flag, 15), 19) ------------------------
    flag = if_then_else(
        (pl.col("high") + pl.col("low")) / 2.0 + pl.col("close")
        < pl.col("low") + pl.col("open"),
        pl.lit(1.0),
        pl.lit(0.0),
    )
    left_inner = ts_decay_linear(flag, 15)
    left = ts_rank(left_inner, 19)

    # --- right branch: ts_rank(decay(corr(rank(low), rank(adv30), 8), 7), 7) ---
    # Stage the rank columns first because cs_rank uses CS partition while
    # ts_corr uses TS partition.
    staged = panel.with_columns(
        cs_rank(pl.col("low")).alias("__a092_rk_low"),
        cs_rank(pl.col("adv30")).alias("__a092_rk_adv30"),
    )
    corr8 = ts_corr(pl.col("__a092_rk_low"), pl.col("__a092_rk_adv30"), 8)
    right_inner = ts_decay_linear(corr8, 7)
    right = ts_rank(right_inner, 7)

    # Materialise both branches and take element-wise min.
    staged = staged.with_columns(
        left.alias("__a092_left"),
        right.alias("__a092_right"),
    )
    expr = pl.min_horizontal(pl.col("__a092_left"), pl.col("__a092_right"))
    return staged.select(expr.alias("alpha092").cast(pl.Float64)).to_series()


# ---------------------------------------------------------------------------
# Registry self-population — runs at module import time
# ---------------------------------------------------------------------------


_ENTRIES = [
    FactorEntry(
        id="alpha092",
        impl=alpha092,
        direction="reverse",
        category="technical",
        description="Pattern-flag rank vs low-adv30 rank correlation",
        legacy_aqml_expr=(
            "Min(Ts_Rank(Ts_DecayLinear(If((high + low) / 2 + close < low + open, "
            "1.0, 0.0), 15), 19), "
            "Ts_Rank(Ts_DecayLinear(Ts_Corr(Rank(low), Rank(adv30), 8), 7), 7))"
        ),
        references=("Kakushadze 2015, '101 Formulaic Alphas', arXiv:1601.00991, eq. 92",),
        formula_doc_path="docs/factor_library/alpha101/alpha_092.md",
    ),
]

for _e in _ENTRIES:
    register_alpha101(_e)
