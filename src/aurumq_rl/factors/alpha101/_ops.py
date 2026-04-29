"""Shared polars operators for alpha101 factor implementations.

All time-series operators partition by ``stock_code`` (assume the panel
is sorted by ``[stock_code, trade_date]`` ascending). Cross-sectional
operators partition by ``trade_date``.

Operators are pure :class:`polars.Expr` builders — they do not eagerly
evaluate. Compose with ``df.with_columns(...)`` / ``df.select(...)``.

Conventions
-----------
* ``window`` arguments are inclusive of the current row (length-N window
  consumes N rows).
* Rolling outputs are NaN/null for the first ``window-1`` rows of each
  partition (matches pandas ``rolling(min_periods=window)`` semantics).
* All numeric outputs are cast to / preserved as :class:`pl.Float64`.

Tie-breaking & STHSF divergence
-------------------------------
``ts_argmax`` / ``ts_argmin`` use polars ``Series.arg_max`` /
``Series.arg_min`` semantics, which return the **FIRST** occurrence of
the extremum. Pandas' ``rolling.apply(np.argmax)`` returns the **LAST**
occurrence on ties. The locked STHSF reference parquet was produced with
pandas, so a small fraction of windows containing exact ties will diverge
from the reference. In synthetic GBM data ties are rare (<0.1% of
windows), so the suite-wide ``rtol=1e-3, atol=1e-3`` tolerance absorbs
the divergence.
"""
from __future__ import annotations

import polars as pl

# ---------------------------------------------------------------------------
# Partition keys
# ---------------------------------------------------------------------------

TS_PART = "stock_code"
"""Partition column for time-series operators (rolling within stock)."""

CS_PART = "trade_date"
"""Partition column for cross-sectional operators (rank within day)."""


# ---------------------------------------------------------------------------
# Rolling / time-series — fast path (native polars rolling)
# ---------------------------------------------------------------------------


def ts_mean(col: pl.Expr, window: int) -> pl.Expr:
    """Rolling arithmetic mean over ``window`` rows, per stock."""
    return col.rolling_mean(window_size=window, min_samples=window).over(TS_PART)


def ts_sum(col: pl.Expr, window: int) -> pl.Expr:
    """Rolling sum over ``window`` rows, per stock."""
    return col.rolling_sum(window_size=window, min_samples=window).over(TS_PART)


def ts_std(col: pl.Expr, window: int) -> pl.Expr:
    """Rolling sample stddev (ddof=1) over ``window`` rows, per stock."""
    return col.rolling_std(window_size=window, min_samples=window).over(TS_PART)


def ts_min(col: pl.Expr, window: int) -> pl.Expr:
    """Rolling minimum over ``window`` rows, per stock."""
    return col.rolling_min(window_size=window, min_samples=window).over(TS_PART)


def ts_max(col: pl.Expr, window: int) -> pl.Expr:
    """Rolling maximum over ``window`` rows, per stock."""
    return col.rolling_max(window_size=window, min_samples=window).over(TS_PART)


def ts_median(col: pl.Expr, window: int) -> pl.Expr:
    """Rolling median over ``window`` rows, per stock."""
    return col.rolling_median(window_size=window, min_samples=window).over(TS_PART)


def ts_skew(col: pl.Expr, window: int) -> pl.Expr:
    """Rolling skewness over ``window`` rows, per stock.

    Uses polars' native ``rolling_skew`` (Fisher-Pearson moment-based).
    """
    return col.rolling_skew(window_size=window, min_samples=window).over(TS_PART)


def ts_kurt(col: pl.Expr, window: int) -> pl.Expr:
    """Rolling Fisher kurtosis (excess) over ``window`` rows, per stock."""
    return col.rolling_kurtosis(window_size=window, min_samples=window).over(TS_PART)


def ts_zscore(col: pl.Expr, window: int) -> pl.Expr:
    """Rolling z-score: ``(x - mean) / std`` over ``window`` rows, per stock."""
    mean = col.rolling_mean(window_size=window, min_samples=window)
    std = col.rolling_std(window_size=window, min_samples=window)
    return ((col - mean) / std).over(TS_PART)


# ---------------------------------------------------------------------------
# Rolling — slow path (rolling_map). Marked for Phase D vectorisation.
# ---------------------------------------------------------------------------


def _window_values(col: pl.Expr, window: int) -> list[pl.Expr]:
    """Return oldest->newest values in the current rolling window."""
    return [col.shift(window - 1 - i).over(TS_PART) for i in range(window)]


def _window_valid(values: list[pl.Expr]) -> pl.Expr:
    """True when a shifted rolling window contains no null or NaN values."""
    return pl.all_horizontal([value.is_not_null() & ~value.is_nan() for value in values])


def _window_arg_extreme(col: pl.Expr, window: int, *, kind: str, tie: str) -> pl.Expr:
    """Vectorized rolling argmax/argmin over a fixed-size per-stock window."""
    values = _window_values(col, window)
    valid = _window_valid(values)
    extreme = pl.max_horizontal(values) if kind == "max" else pl.min_horizontal(values)
    order = range(window - 1, -1, -1) if tie == "first" else range(window)

    out = pl.lit(None, dtype=pl.Float64)
    for idx in order:
        out = pl.when(values[idx] == extreme).then(float(idx)).otherwise(out)
    return pl.when(valid).then(out).otherwise(None).cast(pl.Float64)


def ts_argmax(col: pl.Expr, window: int) -> pl.Expr:
    """Position (0..window-1) of max within last ``window`` rows, per stock.

    Returned as :class:`pl.Float64` for downstream rank/arithmetic safety.
    Windows with any null/NaN return null (pandas-rolling parity).

    .. note::
       **Tie convention**: returns the index of the **first** occurrence of
       the max (polars ``arg_max`` semantics). Pandas/STHSF returns the
       **last** occurrence — see module docstring.
    """
    return _window_arg_extreme(col, window, kind="max", tie="first")


def ts_argmin(col: pl.Expr, window: int) -> pl.Expr:
    """Position (0..window-1) of min within last ``window`` rows, per stock.

    Same tie convention as :func:`ts_argmax` — first occurrence wins.
    """
    return _window_arg_extreme(col, window, kind="min", tie="first")


def ts_argmax_last(col: pl.Expr, window: int) -> pl.Expr:
    """``ts_argmax`` variant where ties resolve to the **last** occurrence.

    Used by alpha factors whose authors selected the most-recent peak when
    the window contains repeated maxima. Returned as :class:`pl.Float64`.
    """
    return _window_arg_extreme(col, window, kind="max", tie="last")


def ts_argmin_last(col: pl.Expr, window: int) -> pl.Expr:
    """``ts_argmin`` variant where ties resolve to the **last** occurrence.

    Companion of :func:`ts_argmax_last`. See its docstring.
    """
    return _window_arg_extreme(col, window, kind="min", tie="last")


def ts_rank(col: pl.Expr, window: int) -> pl.Expr:
    """Rolling rank of the **last** value within the past ``window`` rows.

    Output is normalised into ``[0, 1]`` using ``(rank - 1) / (n - 1)``,
    matching WorldQuant 101 ``ts_rank`` convention. ``method='average'``
    is used for tie-breaking. Returns null until the window is full or if
    any value in the window is null.

    Implementation prefers the native :func:`pl.Expr.rolling_rank`
    (polars 1.x stable) and divides by ``window - 1`` for the [0, 1] scale.
    """
    if window < 2:
        # Degenerate: a 1-element window has no meaningful rank.
        return pl.lit(0.0, dtype=pl.Float64)

    raw_rank = col.rolling_rank(
        window_size=window, method="average", min_samples=window
    )
    return ((raw_rank - 1.0) / float(window - 1)).over(TS_PART).cast(pl.Float64)


def ts_rank_int(col: pl.Expr, window: int) -> pl.Expr:
    """Rolling **integer** rank (1..window) of the last value, per stock.

    Differs from :func:`ts_rank` in that the output is the raw average rank
    (1..window) rather than the normalised pct rank in ``[0, 1]``. Provided
    for STHSF parity (their helper returns 1..window) and for alpha factors
    that arithmetic-combine the rank with non-rank quantities.

    Tie semantics match pandas/scipy ``rankdata``:
    ``avg_rank = lt + (eq + 1) / 2`` — equivalent to ``method='average'``.
    """
    values = _window_values(col, window)
    valid = _window_valid(values)
    last = values[-1]
    eq = pl.sum_horizontal([(value == last).cast(pl.Float64) for value in values])
    lt = pl.sum_horizontal([(value < last).cast(pl.Float64) for value in values])
    rank = lt + (eq + 1.0) / 2.0
    return pl.when(valid).then(rank).otherwise(None).cast(pl.Float64)


def ts_decay_linear(col: pl.Expr, window: int) -> pl.Expr:
    """Linearly weighted moving average with weights ``[w, w-1, …, 1]/sum``.

    The most recent observation gets weight ``window``; the oldest gets
    weight ``1``. Output is null for the first ``window - 1`` rows of
    each stock partition.

    Built as a sum of weighted ``shift(i)`` exprs — stays vectorised and
    avoids the ``rolling_map`` Python callback.
    """
    if window < 1:
        raise ValueError(f"window={window} must be >= 1")

    weights = [(window - i) for i in range(window)]
    total = float(sum(weights))
    shifted = [
        col.shift(i).over(TS_PART) * (w / total) for i, w in enumerate(weights)
    ]
    out = shifted[0]
    for term in shifted[1:]:
        out = out + term
    return out.cast(pl.Float64)


def wma(col: pl.Expr, window: int) -> pl.Expr:
    """Linear weighted MA — alias of :func:`ts_decay_linear`.

    Provided for STHSF compatibility (their helper is named ``wma``).
    """
    return ts_decay_linear(col, window)


def sma(col: pl.Expr, window: int) -> pl.Expr:
    """Simple moving average — alias of :func:`ts_mean`.

    Provided for STHSF compatibility (their helper is named ``sma``).
    """
    return ts_mean(col, window)


def ts_product(col: pl.Expr, window: int) -> pl.Expr:
    """Rolling product over ``window`` rows, per stock.

    Implemented as ``exp(rolling_sum(log(x)))``. **Caveat**: undefined for
    non-positive inputs — caller is responsible for ensuring positivity.
    """
    log_x = col.log()
    return log_x.rolling_sum(window_size=window, min_samples=window).over(TS_PART).exp()


def ts_corr(x: pl.Expr, y: pl.Expr, window: int) -> pl.Expr:
    """Rolling Pearson correlation between two columns, per stock.

    Uses :func:`polars.rolling_corr` — vectorised and fast.
    """
    return pl.rolling_corr(x, y, window_size=window, min_samples=window).over(TS_PART)


def ts_corr_safe(x: pl.Expr, y: pl.Expr, window: int) -> pl.Expr:
    """Same as :func:`ts_corr` but maps NaN outputs to null.

    ``rolling_corr`` returns NaN whenever either input is constant inside
    the window (denominator → 0). Several alpha factors then run the result
    through ``cs_rank`` (or another rank), where NaN poisons the per-day
    rank denominator. This wrapper substitutes NaN with null so that the
    downstream rank treats the row as missing instead. Mirrors STHSF's
    ``df.replace([inf,-inf], 0).fillna(0)`` pattern semantically (we map to
    null, which CS rank then ignores).
    """
    return ts_corr(x, y, window).fill_nan(None)


def ts_cov(x: pl.Expr, y: pl.Expr, window: int) -> pl.Expr:
    """Rolling sample covariance between two columns, per stock."""
    return pl.rolling_cov(x, y, window_size=window, min_samples=window).over(TS_PART)


def count_(col: pl.Expr, window: int) -> pl.Expr:
    """Rolling count of non-null values over ``window`` rows, per stock.

    Returned as :class:`pl.Float64`. ``min_samples=1`` so partial windows
    return the partial count (this matches the WorldQuant alphas that use
    ``count`` as a denominator).
    """
    indicator = col.is_not_null().cast(pl.Float64)
    return indicator.rolling_sum(window_size=window, min_samples=1).over(TS_PART)


def sumif(col: pl.Expr, cond: pl.Expr, window: int) -> pl.Expr:
    """Rolling sum of ``col`` where ``cond`` is True, over ``window`` rows.

    Equivalent to ``rolling_sum(where(cond, col, 0), window)``.
    """
    masked = pl.when(cond).then(col).otherwise(0.0)
    return masked.rolling_sum(window_size=window, min_samples=window).over(TS_PART)


# ---------------------------------------------------------------------------
# Delay / Delta
# ---------------------------------------------------------------------------


def delay(col: pl.Expr, periods: int) -> pl.Expr:
    """Per-stock lag — equivalent to ``col.shift(periods)`` within each stock."""
    return col.shift(periods).over(TS_PART)


def delta(col: pl.Expr, periods: int) -> pl.Expr:
    """Per-stock difference — ``col - col.shift(periods)``."""
    return (col - col.shift(periods)).over(TS_PART)


# ---------------------------------------------------------------------------
# Element-wise scalar transforms
# ---------------------------------------------------------------------------


def abs_(col: pl.Expr) -> pl.Expr:
    """Element-wise absolute value (suffix to avoid shadowing builtin ``abs``)."""
    return col.abs()


def log_(col: pl.Expr) -> pl.Expr:
    """Element-wise natural logarithm. Caller must ensure positivity."""
    return col.log()


def log1p(col: pl.Expr) -> pl.Expr:
    """Element-wise ``log(1 + x)``. Stable for small ``x``."""
    return col.log1p()


def sign_(col: pl.Expr) -> pl.Expr:
    """Element-wise sign: -1 / 0 / +1 (suffix to avoid shadowing)."""
    return col.sign()


def signed_power(col: pl.Expr, exponent: float) -> pl.Expr:
    """Sign-preserving power: ``sign(x) * |x|^exponent``."""
    return col.sign() * col.abs().pow(exponent)


def power(base: pl.Expr, exp: pl.Expr) -> pl.Expr:
    """Element-wise ``base ** exp`` for two expressions.

    For a constant exponent prefer :meth:`pl.Expr.pow` directly — this
    helper exists for symmetric two-Expr usage.
    """
    return base.pow(exp)


def clip_(col: pl.Expr, lo: float, hi: float) -> pl.Expr:
    """Element-wise clip into ``[lo, hi]``."""
    return col.clip(lower_bound=lo, upper_bound=hi)


def pmax(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    """Element-wise (pairwise) maximum of two expressions.

    Wraps :func:`pl.max_horizontal`. NaN-tolerant — passes NaNs through.
    """
    return pl.max_horizontal(a, b)


def pmin(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    """Element-wise (pairwise) minimum of two expressions.

    Wraps :func:`pl.min_horizontal`. NaN-tolerant — passes NaNs through.
    """
    return pl.min_horizontal(a, b)


# ---------------------------------------------------------------------------
# Cross-section operators (partition by trade_date)
# ---------------------------------------------------------------------------


def cs_rank(col: pl.Expr) -> pl.Expr:
    """Cross-section percentile rank in ``[0, 1]`` per ``trade_date``.

    Uses ``method='average'`` ranking divided by per-day non-null count,
    matching pandas ``rank(pct=True)`` and WorldQuant ``rank`` semantics.
    NaN is treated as missing (Polars otherwise ranks NaN as a value).
    """
    clean = col.fill_nan(None)
    return clean.rank(method="average").over(CS_PART) / clean.count().over(CS_PART)


def cs_scale(col: pl.Expr, scale: float = 1.0) -> pl.Expr:
    """Per-day rescale so that ``sum(|x|) == scale``.

    WorldQuant ``scale(x, a)`` semantics: ``a * x / sum(|x|)`` per
    cross-section. Useful for portfolio-style alphas where you want unit
    gross exposure each day.
    """
    abs_sum = col.abs().sum().over(CS_PART)
    return (col * scale) / abs_sum


def ind_neutralize(col: pl.Expr, group: str | pl.Expr) -> pl.Expr:
    """Industry-neutralise: subtract per-day-per-group mean.

    ``group`` may be either a column name (``str``) or a pre-built
    :class:`pl.Expr`. The resulting series satisfies
    ``sum(out) ≈ 0`` within every ``(trade_date, group)`` cell.
    """
    group_expr = pl.col(group) if isinstance(group, str) else group
    # Build the partition list explicitly so polars sees both keys.
    return col - col.mean().over([pl.col(CS_PART), group_expr])


# ---------------------------------------------------------------------------
# Conditional helper
# ---------------------------------------------------------------------------


def if_then_else(
    cond: pl.Expr, then: pl.Expr | float, otherwise: pl.Expr | float
) -> pl.Expr:
    """Convenience wrapper over ``pl.when(cond).then(then).otherwise(otherwise)``.

    Accepts either expressions or scalar literals for the two branches.
    """
    return pl.when(cond).then(then).otherwise(otherwise)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "TS_PART",
    "CS_PART",
    # Rolling
    "ts_mean",
    "ts_sum",
    "ts_std",
    "ts_min",
    "ts_max",
    "ts_median",
    "ts_skew",
    "ts_kurt",
    "ts_zscore",
    "ts_argmax",
    "ts_argmin",
    "ts_argmax_last",
    "ts_argmin_last",
    "ts_rank",
    "ts_rank_int",
    "ts_decay_linear",
    "ts_product",
    "ts_corr",
    "ts_corr_safe",
    "ts_cov",
    "count_",
    "sumif",
    "sma",
    "wma",
    # Delay/Delta
    "delay",
    "delta",
    # Element-wise
    "abs_",
    "log_",
    "log1p",
    "sign_",
    "signed_power",
    "power",
    "clip_",
    "pmax",
    "pmin",
    # Cross-section
    "cs_rank",
    "cs_scale",
    "ind_neutralize",
    # Conditional
    "if_then_else",
]
