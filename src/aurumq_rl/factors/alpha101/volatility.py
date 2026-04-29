"""Alpha101 — volatility category factors."""

from __future__ import annotations

import polars as pl

from aurumq_rl.factors.registry import FactorEntry, register_alpha101

from ._ops import (
    cs_rank,
    delta,
    signed_power,
    ts_argmax,
    ts_corr_safe,
    ts_kurt,
    ts_skew,
    ts_std,
)


def alpha001(panel: pl.DataFrame) -> pl.Series:
    """Alpha #001 — Rank of squared-clip ts_argmax within past 5 days.

    WorldQuant Formula (Kakushadze 2015, eq. 1)
    -------------------------------------------
        rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5

    Legacy AQML Expression (deprecated 2026-04-29, kept as cross-check)
    -------------------------------------------------------------------
        Rank(Ts_ArgMax(SignedPower(If(returns < 0, Ts_Std(returns, 20), close), 2), 5)) - 0.5

    Polars Implementation Notes
    ---------------------------
    1. Conditional input: when returns < 0 use 20-day std of returns
       (volatility regime), otherwise use raw close
    2. Square with sign preservation amplifies extreme moves
    3. ts_argmax: position (0..4) of max within last 5 rows
    4. Cross-section rank centered at 0.5 (subtract 0.5 -> [-0.5, +0.5])

    Required panel columns: ``returns``, ``close``, ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``volatility``

    References
    ----------
    - Kakushadze 2015, "101 Formulaic Alphas", arXiv:1601.00991, eq. 1
    - STHSF/alpha101 (MIT) for pandas reference impl
    """
    # Materialise stage-by-stage. Polars can't reliably mix nested ``over``
    # partitions (TS partition for ts_argmax + CS partition for cs_rank) in a
    # single expression — we must compute the per-stock ts_argmax first, then
    # rank cross-sectionally on the materialised column.
    cond_input = (
        pl.when(pl.col("returns") < 0)
        .then(ts_std(pl.col("returns"), 20))
        .otherwise(pl.col("close"))
    )
    sq = signed_power(cond_input, 2.0)
    arg = ts_argmax(sq, 5)
    staged = panel.with_columns(arg.alias("__a001_arg"))
    return staged.select((cs_rank(pl.col("__a001_arg")) - 0.5).alias("alpha001")).to_series()


# ---------------------------------------------------------------------------
# Registry self-population — runs at module import time
# ---------------------------------------------------------------------------

register_alpha101(
    FactorEntry(
        id="alpha001",
        impl=alpha001,
        direction="reverse",
        category="volatility",
        description="Rank of squared-clip ts_argmax within past 5 days",
        legacy_aqml_expr=(
            "Rank(Ts_ArgMax(SignedPower(If(returns < 0, Ts_Std(returns, 20), close), 2), 5)) - 0.5"
        ),
        references=(
            "Kakushadze 2015, '101 Formulaic Alphas', arXiv:1601.00991, eq. 1",
            "STHSF/alpha101 (MIT) for pandas reference impl",
        ),
        formula_doc_path="docs/factor_library/alpha101/alpha_001.md",
    )
)


# ---------------------------------------------------------------------------
# alpha018 / alpha034 / alpha040 / alpha_custom_skew_reversal /
# alpha_custom_kurt_filter
#
# Operators sourced from ``_ops.py`` after the Phase B Wave 2 CLEAN step
# folded the per-module ``_local_*`` shims into the shared module.
# ---------------------------------------------------------------------------


def alpha018(panel: pl.DataFrame) -> pl.Series:
    """Alpha #018 — body volatility plus body plus close-open correlation, ranked.

    WorldQuant Formula
    ------------------
        -1 * rank((stddev(abs((close - open)), 5) +
                   (close - open)) +
                  correlation(close, open, 10))

    Legacy AQML Expression
    ----------------------
        -1 * Rank(Ts_Std(Abs(close - open), 5) +
                  (close - open) +
                  Ts_Corr(close, open, 10))

    Polars Implementation Notes
    ---------------------------
    1. ``Ts_Std(Abs(close - open), 5)``: 5-day std of body magnitude.
    2. Plus today's body ``close - open``.
    3. Plus 10-day rolling correlation between close and open (NaN/inf
       replaced with null then handled by CS rank's nulls treatment).
    4. CS rank, sign-flipped.

    Required panel columns: ``close``, ``open``, ``stock_code``,
    ``trade_date``

    Direction: ``reverse``
    Category: ``volatility``
    """
    body = pl.col("close") - pl.col("open")
    body_std = ts_std(body.abs(), 5)
    corr = ts_corr_safe(pl.col("close"), pl.col("open"), 10)
    inner = body_std + body + corr
    staged = panel.with_columns(inner.alias("__a018_inner"))
    return staged.select((-1.0 * cs_rank(pl.col("__a018_inner"))).alias("alpha018")).to_series()


def alpha034(panel: pl.DataFrame) -> pl.Series:
    """Alpha #034 — short/long return-vol ratio plus 1-day close delta.

    WorldQuant Formula
    ------------------
        rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) +
              (1 - rank(delta(close, 1)))))

    Legacy AQML Expression
    ----------------------
        Rank((1 - Rank(Ts_Std(returns, 2) / Ts_Std(returns, 5))) +
             (1 - Rank(Delta(close, 1))))

    Polars Implementation Notes
    ---------------------------
    STHSF rewrites the inner expression as ``2 - rank(ratio) - rank(delta)``
    and replaces inf/NaN in the volatility ratio with 1 to avoid
    constant-window pollution. We mirror both behaviours.

    Required panel columns: ``returns``, ``close``, ``stock_code``,
    ``trade_date``

    Direction: ``reverse``
    Category: ``volatility``
    """
    ratio = ts_std(pl.col("returns"), 2) / ts_std(pl.col("returns"), 5)
    delta_close = delta(pl.col("close"), 1)
    staged = panel.with_columns(
        ratio.alias("__a034_ratio"),
        delta_close.alias("__a034_d"),
    )
    # STHSF: replace inf and NaN in ratio with 1 — match by clipping
    # non-finite to a value that ranks neutrally.
    staged = staged.with_columns(
        pl.when(pl.col("__a034_ratio").is_finite())
        .then(pl.col("__a034_ratio"))
        .otherwise(1.0)
        .alias("__a034_ratio")
    )
    inner = 2.0 - cs_rank(pl.col("__a034_ratio")) - cs_rank(pl.col("__a034_d"))
    staged = staged.with_columns(inner.alias("__a034_inner"))
    return staged.select(cs_rank(pl.col("__a034_inner")).alias("alpha034")).to_series()


def alpha040(panel: pl.DataFrame) -> pl.Series:
    """Alpha #040 — high-vol amplitude weighted by high-volume correlation.

    WorldQuant Formula
    ------------------
        -1 * rank(stddev(high, 10)) * correlation(high, volume, 10)

    Legacy AQML Expression
    ----------------------
        -1 * Rank(Ts_Std(high, 10)) * Ts_Corr(high, volume, 10)

    Polars Implementation Notes
    ---------------------------
    1. CS rank of 10-day std of high prices (volatility regime).
    2. 10-day rolling correlation between high and volume (price-volume
       confirmation).
    3. Multiply with sign flip: large vol + positive corr ⇒ persistent
       breakout that mean-reverts.

    Required panel columns: ``high``, ``volume``, ``stock_code``,
    ``trade_date``

    Direction: ``reverse``
    Category: ``volatility``
    """
    std_h = ts_std(pl.col("high"), 10)
    corr_hv = ts_corr_safe(pl.col("high"), pl.col("volume"), 10)
    staged = panel.with_columns(
        std_h.alias("__a040_std"),
        corr_hv.alias("__a040_corr"),
    )
    return staged.select(
        (-1.0 * cs_rank(pl.col("__a040_std")) * pl.col("__a040_corr")).alias("alpha040")
    ).to_series()


def alpha_custom_skew_reversal(panel: pl.DataFrame) -> pl.Series:
    """AurumQ custom — negative CS rank of 20-day rolling skew of returns.

    Legacy AQML Expression
    ----------------------
        -1 * Rank(Ts_Skew(returns, 20))

    Polars Implementation Notes
    ---------------------------
    Captures distributional asymmetry: stocks with strongly positive return
    skew (rare large up moves) rank higher in raw form, and the sign flip
    bets on reversal.

    Required panel columns: ``returns``, ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``volatility``
    """
    skew = ts_skew(pl.col("returns"), 20)
    staged = panel.with_columns(skew.alias("__a_skew_rev"))
    return staged.select(
        (-1.0 * cs_rank(pl.col("__a_skew_rev"))).alias("alpha_custom_skew_reversal")
    ).to_series()


def alpha_custom_kurt_filter(panel: pl.DataFrame) -> pl.Series:
    """AurumQ custom — negative CS rank of 20-day rolling kurtosis of returns.

    Legacy AQML Expression
    ----------------------
        -1 * Rank(Ts_Kurt(returns, 20))

    Polars Implementation Notes
    ---------------------------
    Excess kurtosis flags fat-tailed return regimes. The sign flip filters
    out names whose recent distribution is most leptokurtic — a contrarian
    risk-off tilt.

    Required panel columns: ``returns``, ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``volatility``
    """
    kurt = ts_kurt(pl.col("returns"), 20)
    staged = panel.with_columns(kurt.alias("__a_kurt_filter"))
    return staged.select(
        (-1.0 * cs_rank(pl.col("__a_kurt_filter"))).alias("alpha_custom_kurt_filter")
    ).to_series()


_ENTRIES_EXTRA: tuple[FactorEntry, ...] = (
    FactorEntry(
        id="alpha018",
        impl=alpha018,
        direction="reverse",
        category="volatility",
        description=(
            "Negative CS rank of: 5-day std(|close-open|) + (close-open) + "
            "10-day correlation(close, open)"
        ),
        legacy_aqml_expr=(
            "-1 * Rank(Ts_Std(Abs(close - open), 5) + (close - open) + Ts_Corr(close, open, 10))"
        ),
        references=("Kakushadze 2015, '101 Formulaic Alphas', arXiv:1601.00991, eq. 18",),
    ),
    FactorEntry(
        id="alpha034",
        impl=alpha034,
        direction="reverse",
        category="volatility",
        description=(
            "Rank((1 - rank(stddev(returns,2)/stddev(returns,5))) + (1 - rank(delta(close,1))))"
        ),
        legacy_aqml_expr=(
            "Rank((1 - Rank(Ts_Std(returns, 2) / Ts_Std(returns, 5))) "
            "+ (1 - Rank(Delta(close, 1))))"
        ),
        references=("Kakushadze 2015, '101 Formulaic Alphas', arXiv:1601.00991, eq. 34",),
    ),
    FactorEntry(
        id="alpha040",
        impl=alpha040,
        direction="reverse",
        category="volatility",
        description=("Negative rank(stddev(high,10)) * correlation(high, volume, 10)"),
        legacy_aqml_expr=("-1 * Rank(Ts_Std(high, 10)) * Ts_Corr(high, volume, 10)"),
        references=("Kakushadze 2015, '101 Formulaic Alphas', arXiv:1601.00991, eq. 40",),
    ),
    FactorEntry(
        id="alpha_custom_skew_reversal",
        impl=alpha_custom_skew_reversal,
        direction="reverse",
        category="volatility",
        description="Negative CS rank of 20-day rolling skew of returns",
        legacy_aqml_expr="-1 * Rank(Ts_Skew(returns, 20))",
    ),
    FactorEntry(
        id="alpha_custom_kurt_filter",
        impl=alpha_custom_kurt_filter,
        direction="reverse",
        category="volatility",
        description="Negative CS rank of 20-day rolling kurtosis of returns",
        legacy_aqml_expr="-1 * Rank(Ts_Kurt(returns, 20))",
    ),
)

for _entry in _ENTRIES_EXTRA:
    register_alpha101(_entry)
