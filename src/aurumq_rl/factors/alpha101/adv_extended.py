"""Alpha101 — adv-extended category factors (4 factors).

Implements the alphas previously phantom in ``SKIPPED_ALPHAS`` whose
formulas reference adv-N day windows (021, 061, 064, 075). The panel
already has all adv5/10/20/30/40/50/60/120/150/180 columns; we only
needed to clear the phantom status.
"""

from __future__ import annotations

import polars as pl

from aurumq_rl.factors.registry import FactorEntry, register_alpha101

from ._ops import (
    CS_PART,
    TS_PART,
    cs_rank,
    ts_corr,
    ts_mean,
    ts_min,
    ts_std,
    ts_sum,
)


def alpha021(panel: pl.DataFrame) -> pl.Series:
    """Alpha #021 — Volatility-vs-momentum regime switch with volume confirmation.

    WorldQuant Formula
    ------------------
        ((sum(close, 8) / 8 + stddev(close, 8)) < (sum(close, 2) / 2))
        ? -1 :
        ((sum(close, 2) / 2) < (sum(close, 8) / 8 - stddev(close, 8)))
        ? 1 :
        ((1 < (volume / adv20)) || ((volume / adv20) == 1)) ? 1 : -1

    Polars Implementation Notes
    ---------------------------
    The STHSF reference simplifies the cascade: cond1 OR cond2 ⇒ -1
    (any volatility-narrowing), else +1. We implement the literal paper
    cascade for fidelity.

    Required panel columns: ``close``, ``volume``, ``adv20``,
    ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``adv_extended``
    """
    avg8 = ts_mean(pl.col("close"), 8)
    avg2 = ts_mean(pl.col("close"), 2)
    std8 = ts_std(pl.col("close"), 8)
    cond_a = (avg8 + std8) < avg2
    cond_b = avg2 < (avg8 - std8)
    cond_c = (pl.col("volume") / pl.col("adv20")) >= 1.0
    return panel.select(
        pl.when(cond_a)
        .then(-1.0)
        .when(cond_b)
        .then(1.0)
        .when(cond_c)
        .then(1.0)
        .otherwise(-1.0)
        .alias("alpha021")
    ).to_series()


def alpha061(panel: pl.DataFrame) -> pl.Series:
    """Alpha #061 — Vwap-from-min rank inequality with vwap-adv180 corr rank.

    WorldQuant Formula
    ------------------
        rank(vwap - ts_min(vwap, 16.1219)) <
        rank(correlation(vwap, adv180, 17.9282))

    Required panel columns: ``vwap``, ``adv180``, ``stock_code``,
    ``trade_date``

    Direction: ``reverse``
    Category: ``adv_extended``
    """
    staged = panel.with_columns(
        (pl.col("vwap") - ts_min(pl.col("vwap"), 16)).alias("__a061_diff"),
        ts_corr(pl.col("vwap"), pl.col("adv180"), 18).alias("__a061_corr"),
    )
    return staged.select(
        (cs_rank(pl.col("__a061_diff")) < cs_rank(pl.col("__a061_corr")))
        .cast(pl.Float64)
        .alias("alpha061")
    ).to_series()


def alpha064(panel: pl.DataFrame) -> pl.Series:
    """Alpha #064 — Open-low blend sum-vs-adv120-sum corr inequality with delta-blend rank.

    WorldQuant Formula
    ------------------
        (rank(correlation(sum(open*0.178404 + low*(1-0.178404), 12.7054),
                          sum(adv120, 12.7054), 16.6208)) <
         rank(delta((((high+low)/2)*0.178404 + vwap*(1-0.178404), 3.69741))) * -1

    Required panel columns: ``open``, ``low``, ``adv120``, ``high``, ``vwap``,
    ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``adv_extended``
    """
    blend1 = pl.col("open") * 0.178404 + pl.col("low") * (1.0 - 0.178404)
    midpoint = (pl.col("high") + pl.col("low")) / 2.0
    blend2 = midpoint * 0.178404 + pl.col("vwap") * (1.0 - 0.178404)
    staged = panel.with_columns(
        ts_sum(blend1, 13).alias("__a064_s1"),
        ts_sum(pl.col("adv120"), 13).alias("__a064_s2"),
        (blend2 - blend2.shift(4).over(TS_PART)).alias("__a064_db"),
    )
    staged2 = staged.with_columns(
        ts_corr(pl.col("__a064_s1"), pl.col("__a064_s2"), 17).alias("__a064_corr")
    )
    return staged2.select(
        (
            (cs_rank(pl.col("__a064_corr")) < cs_rank(pl.col("__a064_db"))).cast(pl.Float64) * -1.0
        ).alias("alpha064")
    ).to_series()


def alpha075(panel: pl.DataFrame) -> pl.Series:
    """Alpha #075 — vwap-volume corr rank inequality with low-adv50 rank corr.

    WorldQuant Formula
    ------------------
        rank(correlation(vwap, volume, 4.24304)) <
        rank(correlation(rank(low), rank(adv50), 12.4413))

    Required panel columns: ``vwap``, ``volume``, ``low``, ``adv50``,
    ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``adv_extended``
    """
    staged = panel.with_columns(
        ts_corr(pl.col("vwap"), pl.col("volume"), 4).alias("__a075_c1"),
        cs_rank(pl.col("low")).alias("__a075_rl"),
        cs_rank(pl.col("adv50")).alias("__a075_ra"),
    )
    staged2 = staged.with_columns(
        ts_corr(pl.col("__a075_rl"), pl.col("__a075_ra"), 12).alias("__a075_c2")
    )
    return staged2.select(
        (cs_rank(pl.col("__a075_c1")) < cs_rank(pl.col("__a075_c2")))
        .cast(pl.Float64)
        .alias("alpha075")
    ).to_series()


# ---------------------------------------------------------------------------
# Registry self-population
# ---------------------------------------------------------------------------


_ENTRIES: tuple[FactorEntry, ...] = (
    FactorEntry(
        id="alpha021",
        impl=alpha021,
        direction="reverse",
        category="adv_extended",
        description=(
            "Volatility-vs-momentum regime switch using sma(close,8)+/-std(close,8) "
            "with volume/adv20 confirmation"
        ),
        references=("Kakushadze 2015, eq. 21",),
    ),
    FactorEntry(
        id="alpha061",
        impl=alpha061,
        direction="reverse",
        category="adv_extended",
        description=("rank(vwap-ts_min(vwap,16)) < rank(corr(vwap, adv180, 18))"),
        references=("Kakushadze 2015, eq. 61",),
    ),
    FactorEntry(
        id="alpha064",
        impl=alpha064,
        direction="reverse",
        category="adv_extended",
        description=(
            "(rank(corr(sum(open-low blend,13), sum(adv120,13), 17)) "
            "< rank(delta(midpoint-vwap blend, 4))) * -1"
        ),
        references=("Kakushadze 2015, eq. 64",),
    ),
    FactorEntry(
        id="alpha075",
        impl=alpha075,
        direction="reverse",
        category="adv_extended",
        description=("rank(corr(vwap,volume,4)) < rank(corr(rank(low), rank(adv50), 12))"),
        references=("Kakushadze 2015, eq. 75",),
    ),
)


for _entry in _ENTRIES:
    register_alpha101(_entry)


# Tie module-level imports for static analysers.
_ = TS_PART
_ = CS_PART
