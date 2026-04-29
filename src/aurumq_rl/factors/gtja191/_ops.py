"""Shared polars operators for the GTJA-191 (国泰君安 191) factor library.

These operators are the GTJA-paper analogues of the alpha101 helpers in
``aurumq_rl.factors.alpha101._ops``. They share the same partition
conventions:

* Time-series operators partition by ``stock_code`` (assume the panel is
  sorted by ``[stock_code, trade_date]`` ascending).
* Cross-sectional operators partition by ``trade_date``.

All operators are pure :class:`polars.Expr` builders — they do not
eagerly evaluate. Compose with ``df.with_columns(...)``.

Semantics — important divergences from the WorldQuant/alpha101 conventions
--------------------------------------------------------------------------

* **GTJA SMA** (``sma(x, n, m)``) is **not** the simple moving average.
  In the GTJA paper, ``SMA(X, N, M)`` is recursively defined as
  ``today = (M * X_today + (N - M) * SMA_prev) / N``. This is exactly an
  exponentially weighted mean with smoothing factor ``alpha = M / N``,
  so we implement it via :meth:`pl.Expr.ewm_mean` with that alpha. (The
  Daic115 reference uses ``pandas.ewm(alpha=m/n)`` which is the same
  formulation.) The argument order is ``sma(col, n, m)`` and we require
  ``n > m`` (matching Daic115).

* **WMA** (``wma(x, n)``) uses **increasing** linear weights
  ``[1, 2, …, n] / sum``. The most recent observation gets weight ``n``,
  the oldest gets weight ``1``. Same direction as alpha101
  ``ts_decay_linear`` but with non-normalised weights (this normaliser
  matches Daic115's ``WMA``).

* **DECAYLINEAR** (``decay_linear(x, n)``) uses weights
  ``[2*i / (n*(n+1)) for i in 1..n]``. Sum is 1, direction matches
  ``wma`` (newest gets the highest weight). Algebraically identical to
  ``wma(x, n)`` because ``sum(1..n) == n*(n+1)/2`` — the two are
  preserved as separate names for fidelity to the GTJA paper.

* **HIGHDAY / LOWDAY** return the **distance in days** from today back
  to the high/low of the past N rows (so the latest row would yield 1 if
  the high/low is today). Range is ``[1, N]``. This matches Daic115
  ``HIGHDAY/LOWDAY`` (`n - argmax/argmin`).

* **REGBETA(y, x, n)** is the rolling slope of regressing y on x:
  ``cov(x, y) / var(x)`` over the last n rows. Argument order in our
  Python signature is ``(y, x, n)`` to match the natural "regress y on
  x" reading; Daic115's ``REGBETA(X, Y, N)`` is the same calculation
  but with positional names swapped — ours and theirs agree numerically.

* **REGRESI(y, x, n)** returns the residual at the **last point** of the
  rolling window: ``y[-1] - (alpha + beta * x[-1])`` where
  ``(alpha, beta)`` are the OLS estimates over the last n rows. Computed
  via :meth:`pl.Expr.rolling_map` + numpy ``lstsq`` — slow path.

Performance
-----------

Most operators are built from native polars rolling primitives and stay
vectorised: ``sma`` (via ``ewm_mean``), ``wma`` / ``decay_linear`` (via
weighted-shift composition), the ``ts_*`` rollers, ``lowday`` /
``highday`` (via shifted-window argmin/argmax), and ``regbeta`` (via
rolling cov / var). ``regresi`` remains the only slow-path rolling map.
"""

from __future__ import annotations

import polars as pl

# ---------------------------------------------------------------------------
# Partition keys (must match aurumq_rl.factors.alpha101._ops)
# ---------------------------------------------------------------------------

TS_PART = "stock_code"
"""Partition column for time-series operators (rolling within stock)."""

CS_PART = "trade_date"
"""Partition column for cross-sectional operators (rank within day)."""


# ---------------------------------------------------------------------------
# Cross-section operators
# ---------------------------------------------------------------------------


def rank(col: pl.Expr) -> pl.Expr:
    """GTJA ``RANK`` — cross-section percentile rank in ``[0, 1]`` per day.

    Uses ``method='average'`` ranking divided by the per-day count of
    non-null values, matching pandas ``rank(axis=1, pct=True)``.
    NaN is treated as missing.
    """
    clean = col.fill_nan(None)
    return clean.rank(method="average").over(CS_PART) / clean.count().over(CS_PART)


# ---------------------------------------------------------------------------
# Rolling time-series — fast (native polars) primitives
# ---------------------------------------------------------------------------


def ts_rank(col: pl.Expr, window: int) -> pl.Expr:
    """Rolling rank (pct) of the last value over the past ``window`` rows.

    Matches pandas ``rolling(N).rank(pct=True)``: returns the rank of the
    last observation in ``[1/n, 1]``. Null until the window is full or if
    any value in the window is null.
    """
    if window < 2:
        return pl.lit(0.0, dtype=pl.Float64)
    raw = col.rolling_rank(window_size=window, method="average", min_samples=window)
    return (raw / float(window)).over(TS_PART).cast(pl.Float64)


def mean(col: pl.Expr, window: int) -> pl.Expr:
    """Rolling arithmetic mean over ``window`` rows, per stock."""
    return col.rolling_mean(window_size=window, min_samples=window).over(TS_PART)


def sum_(col: pl.Expr, window: int) -> pl.Expr:
    """Rolling sum over ``window`` rows, per stock.

    Trailing underscore avoids shadowing Python's :func:`sum` builtin.
    """
    return col.rolling_sum(window_size=window, min_samples=window).over(TS_PART)


def std_(col: pl.Expr, window: int) -> pl.Expr:
    """Rolling sample stddev (ddof=1) over ``window`` rows, per stock.

    Trailing underscore avoids shadowing tools that expect ``std`` to be
    a class attribute (e.g. polars expression parsers).
    """
    return col.rolling_std(window_size=window, min_samples=window).over(TS_PART)


def ts_min(col: pl.Expr, window: int) -> pl.Expr:
    """Rolling minimum over ``window`` rows, per stock."""
    return col.rolling_min(window_size=window, min_samples=window).over(TS_PART)


def ts_max(col: pl.Expr, window: int) -> pl.Expr:
    """Rolling maximum over ``window`` rows, per stock."""
    return col.rolling_max(window_size=window, min_samples=window).over(TS_PART)


def delta(col: pl.Expr, periods: int) -> pl.Expr:
    """Per-stock difference: ``x - x.shift(periods)``."""
    return (col - col.shift(periods)).over(TS_PART)


def delay(col: pl.Expr, periods: int) -> pl.Expr:
    """Per-stock lag: ``x.shift(periods)``."""
    return col.shift(periods).over(TS_PART)


def corr(x: pl.Expr, y: pl.Expr, window: int) -> pl.Expr:
    """Rolling Pearson correlation between two columns, per stock."""
    return pl.rolling_corr(x, y, window_size=window, min_samples=window).over(TS_PART)


def covariance(x: pl.Expr, y: pl.Expr, window: int) -> pl.Expr:
    """Rolling sample covariance between two columns, per stock."""
    return pl.rolling_cov(x, y, window_size=window, min_samples=window).over(TS_PART)


# ---------------------------------------------------------------------------
# Weighted moving averages
# ---------------------------------------------------------------------------


def sma(col: pl.Expr, n: int, m: int = 1) -> pl.Expr:
    """GTJA paper ``SMA(X, N, M)`` — recursive EWMA with smoothing ``M/N``.

    Defined by the recursion ``today = (M * X_today + (N - M) * SMA_prev) / N``,
    which is exactly :meth:`polars.Expr.ewm_mean` with ``alpha = M / N``.
    Daic115 implements this as ``pandas.DataFrame.ewm(alpha=M/N).mean()``,
    so our output matches theirs to within float-rounding.

    Parameters
    ----------
    col:
        Input series.
    n, m:
        Smoothing parameters; we require ``n > m > 0`` (Daic115's assertion).
    """
    if not (n > m > 0):
        raise ValueError(f"sma requires n > m > 0, got n={n}, m={m}")
    alpha = float(m) / float(n)
    return col.ewm_mean(alpha=alpha, ignore_nulls=True).over(TS_PART).cast(pl.Float64)


def wma(col: pl.Expr, window: int) -> pl.Expr:
    """GTJA ``WMA`` — linear weighted MA with weights ``[1, 2, …, n] / sum``.

    Most recent observation gets the highest weight. Implemented as a
    sum of weighted ``shift(i)`` exprs (vectorised, no Python callback).
    Output is null for the first ``window - 1`` rows of each stock.
    """
    if window < 1:
        raise ValueError(f"window={window} must be >= 1")

    # weight at lag i (0 = today) is (window - i)
    weights = [float(window - i) for i in range(window)]
    total = float(sum(weights))
    shifted = [col.shift(i).over(TS_PART) * (w / total) for i, w in enumerate(weights)]
    out = shifted[0]
    for term in shifted[1:]:
        out = out + term
    return out.cast(pl.Float64)


def decay_linear(col: pl.Expr, window: int) -> pl.Expr:
    """GTJA ``DECAYLINEAR`` — weights ``[2*i / (n*(n+1)) for i in 1..n]``.

    Mathematically identical to :func:`wma` (since the sum of 1..n is
    n*(n+1)/2, both produce the same normalised increasing-weight MA).
    Kept as a separate symbol for fidelity to the GTJA paper.
    """
    return wma(col, window)


# ---------------------------------------------------------------------------
# Conditional / count / sumif
# ---------------------------------------------------------------------------


def ifelse(cond: pl.Expr, then: pl.Expr | float, otherwise: pl.Expr | float) -> pl.Expr:
    """GTJA ``IFELSE(cond, A, B)`` — ``pl.when(cond).then(A).otherwise(B)``."""
    return pl.when(cond).then(then).otherwise(otherwise)


def count_(cond: pl.Expr, window: int) -> pl.Expr:
    """Rolling count of ``True`` values of a boolean condition.

    Trailing underscore avoids shadowing Python's :func:`count` (none in
    builtins, but consistent with ``sum_`` / ``std_``).
    """
    indicator = cond.cast(pl.Float64)
    return indicator.rolling_sum(window_size=window, min_samples=window).over(TS_PART)


def sumif(col: pl.Expr, window: int, cond: pl.Expr) -> pl.Expr:
    """Rolling sum of ``col`` where ``cond`` is ``True``, over ``window`` rows.

    Equivalent to ``rolling_sum(where(cond, col, 0), window)``.
    """
    masked = pl.when(cond).then(col).otherwise(0.0)
    return masked.rolling_sum(window_size=window, min_samples=window).over(TS_PART)


# ---------------------------------------------------------------------------
# Argmin / argmax position
# ---------------------------------------------------------------------------


def _window_values(col: pl.Expr, window: int) -> list[pl.Expr]:
    """Return oldest->newest values in the current rolling window."""
    return [col.shift(window - 1 - i).over(TS_PART) for i in range(window)]


def _window_valid(values: list[pl.Expr]) -> pl.Expr:
    """True when a shifted rolling window contains no null or NaN values."""
    return pl.all_horizontal([value.is_not_null() & ~value.is_nan() for value in values])


def _window_arg_extreme(col: pl.Expr, window: int, *, kind: str) -> pl.Expr:
    """Vectorized rolling argmax/argmin with first-tie semantics."""
    if window < 1:
        raise ValueError(f"window={window} must be >= 1")

    values = _window_values(col, window)
    valid = _window_valid(values)
    extreme = pl.max_horizontal(values) if kind == "max" else pl.min_horizontal(values)

    out = pl.lit(None, dtype=pl.Float64)
    # Iterate newest->oldest so later replacements leave the oldest index
    # for ties, matching numpy argmin/argmax and Daic115 LOWDAY/HIGHDAY.
    for idx in range(window - 1, -1, -1):
        out = pl.when(values[idx] == extreme).then(float(idx)).otherwise(out)
    return pl.when(valid).then(out).otherwise(None).cast(pl.Float64)


def lowday(col: pl.Expr, window: int) -> pl.Expr:
    """GTJA ``LOWDAY`` — distance from today to the min of the past N rows.

    Returns ``n - argmin(window)`` so that "today is the new low" yields
    ``1`` and "the low was n-1 rows ago" yields ``n``. Matches Daic115's
    ``LOWDAY``. Output is :class:`pl.Float64` (NaN for partial windows).
    """
    return (float(window) - _window_arg_extreme(col, window, kind="min")).cast(pl.Float64)


def highday(col: pl.Expr, window: int) -> pl.Expr:
    """GTJA ``HIGHDAY`` — distance from today to the max of the past N rows.

    Range ``[1, n]``. See :func:`lowday` for the convention.
    """
    return (float(window) - _window_arg_extreme(col, window, kind="max")).cast(pl.Float64)


# ---------------------------------------------------------------------------
# Rolling regression — slow path (rolling_map + numpy)
# ---------------------------------------------------------------------------


def regbeta(y: pl.Expr, x: pl.Expr, window: int) -> pl.Expr:
    """Rolling OLS slope of ``y`` on ``x`` over the last ``window`` rows.

    ``slope = cov(x, y) / var(x)``. Built from native rolling cov/var so
    it stays vectorised. Returns null until the window is full.

    Notes
    -----
    Argument order ``(y, x, n)`` reads as "regress y on x". Daic115's
    ``REGBETA(X, Y, N)`` computes ``cov(X, Y) / var(X)`` — the same number
    with the X/Y names swapped at the call site. Numerical agreement
    holds.
    """
    cov_xy = pl.rolling_cov(x, y, window_size=window, min_samples=window).over(TS_PART)
    var_x = x.rolling_var(window_size=window, min_samples=window).over(TS_PART)
    # Avoid div-by-zero when x is constant within the window.
    safe_var = pl.when(var_x == 0).then(None).otherwise(var_x)
    return (cov_xy / safe_var).cast(pl.Float64)


def regresi(y: pl.Expr, x: pl.Expr, window: int) -> pl.Expr:
    """Rolling OLS residual of ``y`` on ``x`` at the **last** window point.

    For each window of ``window`` rows ending at the current row, fit
    ``y = a + b*x`` and return ``y[-1] - (a + b*x[-1])``.

    Implemented via the closed-form OLS identity (vectorised, no Python
    callback). Algebra:

    .. code-block:: text

        b      = cov(x, y) / var(x)
        a      = mean(y) - b * mean(x)
        resid  = y[-1] - (a + b * x[-1])
               = (y[-1] - mean(y)) - b * (x[-1] - mean(x))

    so the residual is ``y_dev_today - beta * x_dev_today`` where the
    deviations are from the rolling mean.
    """
    cov_xy = pl.rolling_cov(x, y, window_size=window, min_samples=window).over(TS_PART)
    var_x = x.rolling_var(window_size=window, min_samples=window).over(TS_PART)
    safe_var = pl.when(var_x == 0).then(None).otherwise(var_x)
    beta = cov_xy / safe_var

    mean_x = x.rolling_mean(window_size=window, min_samples=window).over(TS_PART)
    mean_y = y.rolling_mean(window_size=window, min_samples=window).over(TS_PART)

    return ((y - mean_y) - beta * (x - mean_x)).cast(pl.Float64)


# ---------------------------------------------------------------------------
# Element-wise scalar transforms
# ---------------------------------------------------------------------------


def sign_(col: pl.Expr) -> pl.Expr:
    """Element-wise sign: ``-1 / 0 / +1``. Trailing underscore avoids shadowing."""
    return col.sign()


def abs_(col: pl.Expr) -> pl.Expr:
    """Element-wise absolute value."""
    return col.abs()


def log_(col: pl.Expr) -> pl.Expr:
    """Element-wise natural log. Caller must ensure positivity."""
    return col.log()


# ---------------------------------------------------------------------------
# Deterministic input — sequence helper
# ---------------------------------------------------------------------------


def sequence(n: int) -> pl.Expr:
    """Return the literal expression representing the constant series ``[1, 2, …, n]``.

    Used as a deterministic ``x`` argument to :func:`regbeta` / :func:`regresi`
    when the GTJA formula calls ``REGBETA(SEQUENCE(N), …)``. The returned
    expression is a polars ``Series literal`` so it broadcasts correctly
    when joined with another expression of the same length.
    """
    if n < 1:
        raise ValueError(f"sequence requires n >= 1, got {n}")
    return pl.lit(pl.Series("__seq__", list(range(1, n + 1)), dtype=pl.Float64))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "TS_PART",
    "CS_PART",
    # Cross-section
    "rank",
    # Rolling — vectorised
    "ts_rank",
    "mean",
    "sum_",
    "std_",
    "ts_min",
    "ts_max",
    "delta",
    "delay",
    "corr",
    "covariance",
    # Weighted MAs
    "sma",
    "wma",
    "decay_linear",
    # Conditional / count / sumif
    "ifelse",
    "count_",
    "sumif",
    # Slow path (rolling_map)
    "lowday",
    "highday",
    "regbeta",
    "regresi",
    # Element-wise
    "sign_",
    "abs_",
    "log_",
    # Helpers
    "sequence",
]
