"""Alpha101 — cap-weighted category factors (2 factors).

Implements the alphas previously phantom in ``SKIPPED_ALPHAS`` because
they reference market-cap data:

* alpha024 — long-term close-mean acceleration switch (uses ``cap``
  implicitly: STHSF/yli188 omit the cap term entirely; the WorldQuant
  formula does NOT actually reference cap. SKIPPED_ALPHAS conflates
  it with alpha056, which IS the genuine cap-weighted alpha. We
  implement alpha024 per the literal paper formula here.)
* alpha056 — return-ratio rank scaled by rank(returns * cap).

Stocks with ``cap IS NULL`` produce NaN output for alpha056 (and a
vanilla output for alpha024 since it doesn't actually use cap).
"""

from __future__ import annotations

import polars as pl

from aurumq_rl.factors.registry import FactorEntry, register_alpha101

from ._ops import (
    CS_PART,
    TS_PART,
    cs_rank,
    delay,
    delta,
    ts_mean,
    ts_min,
    ts_sum,
)


def alpha024(panel: pl.DataFrame) -> pl.Series:
    """Alpha #024 — Long-horizon mean-acceleration switch.

    WorldQuant Formula
    ------------------
        ((delta(sum(close, 100) / 100, 100) / delay(close, 100)) <= 0.05)
        ? -1 * (close - ts_min(close, 100))
        : -1 * delta(close, 3)

    Polars Implementation Notes
    ---------------------------
    The 100-day windows make this a long-horizon factor; on the synthetic
    60-day panel the output is mostly NaN until day ~100 (which never
    arrives on synthetic). Steady-state test is consequently best-effort.

    Required panel columns: ``close``, ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``cap_weighted``
    """
    cond_num = delta(ts_mean(pl.col("close"), 100), 100)
    cond_den = delay(pl.col("close"), 100)
    cond = (cond_num / cond_den) <= 0.05
    branch_true = -1.0 * (pl.col("close") - ts_min(pl.col("close"), 100))
    branch_false = -1.0 * delta(pl.col("close"), 3)
    return panel.select(
        pl.when(cond)
        .then(branch_true)
        .otherwise(branch_false)
        .alias("alpha024")
    ).to_series()


def alpha056(panel: pl.DataFrame) -> pl.Series:
    """Alpha #056 — Inverse rank of returns ratio scaled by cap-weighted return rank.

    WorldQuant Formula
    ------------------
        0 - 1 * (rank(sum(returns, 10) / sum(sum(returns, 2), 3)) *
                 rank(returns * cap))

    Polars Implementation Notes
    ---------------------------
    The denominator ``sum(sum(returns,2),3)`` is the 3-day rolling sum of
    the 2-day rolling sum, i.e. cumulative returns over an effective
    4-day window. ``returns * cap`` is the dollar return; CS rank
    captures cross-section preference for high-dollar-return stocks.

    Stocks with ``cap IS NULL`` produce a NaN factor row (because
    ``returns * NULL == NULL``, which propagates through CS rank).

    Required panel columns: ``returns``, ``cap``, ``stock_code``,
    ``trade_date``

    Direction: ``reverse``
    Category: ``cap_weighted``
    """
    inner_ratio = ts_sum(pl.col("returns"), 10) / ts_sum(
        ts_sum(pl.col("returns"), 2), 3
    )
    staged = panel.with_columns(
        inner_ratio.alias("__a056_ratio"),
        (pl.col("returns") * pl.col("cap")).alias("__a056_dollar"),
    )
    return staged.select(
        (
            -1.0
            * cs_rank(pl.col("__a056_ratio"))
            * cs_rank(pl.col("__a056_dollar"))
        ).alias("alpha056")
    ).to_series()


# ---------------------------------------------------------------------------
# Registry self-population
# ---------------------------------------------------------------------------


_ENTRIES: tuple[FactorEntry, ...] = (
    FactorEntry(
        id="alpha024",
        impl=alpha024,
        direction="reverse",
        category="cap_weighted",
        description=(
            "If delta(sma(close,100),100)/delay(close,100) <= 0.05 then "
            "-(close-ts_min(close,100)) else -delta(close,3)"
        ),
        references=("Kakushadze 2015, eq. 24",),
    ),
    FactorEntry(
        id="alpha056",
        impl=alpha056,
        direction="reverse",
        category="cap_weighted",
        description=(
            "-rank(sum(returns,10)/sum(sum(returns,2),3)) * rank(returns * cap)"
        ),
        references=("Kakushadze 2015, eq. 56",),
    ),
)


for _entry in _ENTRIES:
    register_alpha101(_entry)


# Tie module-level imports for static analysers.
_ = TS_PART
_ = CS_PART
