"""Alpha101 — volatility category factors."""

from __future__ import annotations

import math

import polars as pl

from aurumq_rl.factors.registry import FactorEntry, register_alpha101

from ._ops import (
    TS_PART,
    cs_rank,
    delay,
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


# ---------------------------------------------------------------------------
# Part 2 — Yang-Zhang / Garman-Klass volatility estimators (new custom
# factors, no parity constraint). Existing custom volatility factors above
# only look at close-to-close returns (ts_std); these consume the OHLC
# already present in the panel for a drift-independent range estimator.
# ---------------------------------------------------------------------------

YZ_GK_DEFAULT_WINDOW: int = 20
"""Rolling window (trading days) used by the default-registered YZ/GK factors."""


def yang_zhang_volatility(window: int = YZ_GK_DEFAULT_WINDOW) -> pl.Expr:
    """Yang-Zhang drift-independent volatility estimator (per-stock rolling).

    Combines the overnight (close-to-open), open-to-close, and
    Rogers-Satchell components:

        V_o  = rolling_var(ln(open / prev_close), window)
        V_c  = rolling_var(ln(close / open), window)
        V_rs = rolling_mean(RS, window)   where
               RS = ln(high/close)*ln(high/open) + ln(low/close)*ln(low/open)
        k    = 0.34 / (1.34 + (window + 1) / (window - 1))
        YZ   = sqrt(V_o + k*V_c + (1 - k)*V_rs)

    Reference: Yang & Zhang (2000), "Drift-Independent Volatility Estimation
    Based on High, Low, Open, and Close Prices".

    Limit-locked day handling
    --------------------------
    Days where ``high == low`` (一字板 / limit-locked — no intraday range
    traded) make the Rogers-Satchell range term degenerate (it collapses
    toward an artificial value that is not reflective of true volatility,
    biasing the window average). Those days' RS contribution is set to
    null and skipped by the rolling mean (``min_samples=1`` — a handful of
    limit-locked days inside the window do not null out the whole
    estimate; the ``V_o``/``V_c`` terms still require a full window of
    non-degenerate history via ``ts_std``'s ``min_samples=window``, so the
    usual "first ``window - 1`` rows are null" warm-up convention holds).

    Required columns: ``open``, ``high``, ``low``, ``close``, ``stock_code``.
    Output is clipped to ``>= 0`` before the square root as a defensive
    guard against a (theoretically rare) negative combined-variance
    estimate from sampling noise.
    """
    if window < 2:
        raise ValueError(f"window={window} must be >= 2")

    prev_close = delay(pl.col("close"), 1)
    log_overnight = (pl.col("open") / prev_close).log()
    log_open_close = (pl.col("close") / pl.col("open")).log()
    limit_locked = pl.col("high") == pl.col("low")
    rogers_satchell = (
        pl.when(limit_locked)
        .then(None)
        .otherwise(
            (pl.col("high") / pl.col("close")).log() * (pl.col("high") / pl.col("open")).log()
            + (pl.col("low") / pl.col("close")).log() * (pl.col("low") / pl.col("open")).log()
        )
    )

    v_o = ts_std(log_overnight, window).pow(2.0)
    v_c = ts_std(log_open_close, window).pow(2.0)
    v_rs = rogers_satchell.rolling_mean(window_size=window, min_samples=1).over(TS_PART)

    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    yz_var = v_o + k * v_c + (1.0 - k) * v_rs
    return yz_var.clip(lower_bound=0.0).sqrt()


def garman_klass_volatility(window: int = YZ_GK_DEFAULT_WINDOW) -> pl.Expr:
    """Garman-Klass range-based volatility estimator (per-stock rolling).

        GK = 0.5*ln(high/low)^2 - (2*ln(2) - 1)*ln(close/open)^2
        estimate = sqrt(rolling_mean(GK, window))

    Reference: Garman & Klass (1980), "On the Estimation of Security Price
    Volatilities from Historical Data".

    Limit-locked day handling
    --------------------------
    Same rationale as :func:`yang_zhang_volatility`: ``high == low`` days
    are excluded from the ``GK`` rolling mean (null, skipped) rather than
    contributing an artificial low-range observation. Because this
    estimator has no separate overnight/open-close term to gate warm-up
    (unlike Yang-Zhang), an explicit "calendar readiness" flag — a count
    of the last ``window`` rows via ``close.is_not_null()`` — enforces the
    usual "first ``window - 1`` rows are null" convention independently of
    how many of those rows happened to be limit-locked.

    Required columns: ``open``, ``high``, ``low``, ``close``, ``stock_code``.
    Output is clipped to ``>= 0`` before the square root as a defensive
    guard against sampling noise.
    """
    if window < 2:
        raise ValueError(f"window={window} must be >= 2")

    ln2_term = 2.0 * math.log(2.0) - 1.0
    limit_locked = pl.col("high") == pl.col("low")
    gk_raw = (
        pl.when(limit_locked)
        .then(None)
        .otherwise(
            0.5 * (pl.col("high") / pl.col("low")).log().pow(2.0)
            - ln2_term * (pl.col("close") / pl.col("open")).log().pow(2.0)
        )
    )
    raw_mean = gk_raw.rolling_mean(window_size=window, min_samples=1).over(TS_PART)
    calendar_ready = (
        pl.col("close")
        .is_not_null()
        .cast(pl.Float64)
        .rolling_sum(window_size=window, min_samples=window)
        .over(TS_PART)
    )
    gk_var = pl.when(calendar_ready.is_not_null()).then(raw_mean).otherwise(None)
    return gk_var.clip(lower_bound=0.0).sqrt()


def alpha_custom_yang_zhang_vol(panel: pl.DataFrame) -> pl.Series:
    """AurumQ custom — Yang-Zhang drift-independent volatility (20-day).

    See :func:`yang_zhang_volatility` for the formula and the
    limit-locked-day exclusion rule.

    Required panel columns: ``open``, ``high``, ``low``, ``close``,
    ``stock_code``, ``trade_date``

    Direction: ``normal``
    Category: ``volatility``
    """
    staged = panel.with_columns(
        yang_zhang_volatility(YZ_GK_DEFAULT_WINDOW).alias("alpha_custom_yang_zhang_vol")
    )
    return staged.select("alpha_custom_yang_zhang_vol").to_series()


def alpha_custom_garman_klass_vol(panel: pl.DataFrame) -> pl.Series:
    """AurumQ custom — Garman-Klass range-based volatility (20-day).

    See :func:`garman_klass_volatility` for the formula and the
    limit-locked-day exclusion rule.

    Required panel columns: ``open``, ``high``, ``low``, ``close``,
    ``stock_code``, ``trade_date``

    Direction: ``normal``
    Category: ``volatility``
    """
    staged = panel.with_columns(
        garman_klass_volatility(YZ_GK_DEFAULT_WINDOW).alias("alpha_custom_garman_klass_vol")
    )
    return staged.select("alpha_custom_garman_klass_vol").to_series()


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
    FactorEntry(
        id="alpha_custom_yang_zhang_vol",
        impl=alpha_custom_yang_zhang_vol,
        direction="normal",
        category="volatility",
        description=(
            "Yang-Zhang drift-independent volatility estimator (20-day; "
            "overnight + open-close + Rogers-Satchell, limit-locked days excluded)"
        ),
        legacy_aqml_expr=None,
        references=(
            "Yang & Zhang (2000), 'Drift-Independent Volatility Estimation "
            "Based on High, Low, Open, and Close Prices', Journal of Business",
        ),
    ),
    FactorEntry(
        id="alpha_custom_garman_klass_vol",
        impl=alpha_custom_garman_klass_vol,
        direction="normal",
        category="volatility",
        description=(
            "Garman-Klass range-based volatility estimator (20-day; limit-locked days excluded)"
        ),
        legacy_aqml_expr=None,
        references=(
            "Garman & Klass (1980), 'On the Estimation of Security Price "
            "Volatilities from Historical Data', Journal of Business",
        ),
    ),
)

for _entry in _ENTRIES_EXTRA:
    register_alpha101(_entry)
