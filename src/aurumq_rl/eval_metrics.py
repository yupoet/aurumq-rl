"""Robust, selection-bias-aware evaluation metrics for backtests.

Pure ``numpy`` + Python standard library — deliberately **no scipy
dependency** so this module stays safe to import from :mod:`aurumq_rl.backtest`
(used inside the GPU training loop's validation callback, which only
installs the ``train`` extra — no ``scipy``) as well as from CLI eval
scripts. The two special functions that would normally come from
``scipy.stats.norm`` (the standard normal CDF / inverse CDF used by the
Deflated Sharpe Ratio and the HAC confidence interval) are reimplemented
below with well-known closed-form / rational approximations, see
:func:`_normal_cdf` and :func:`_normal_ppf`.

Terminology note (issue #6): "Deflated Sharpe Ratio" here means the
**Bailey & Lopez de Prado (2014)** multiple-testing / non-normality
deflation of a backtested Sharpe ratio. This is UNRELATED to the
*Differential* Sharpe Ratio reward used as a training reward shaping term
(issue #2) — the two are conceptually different objects that happen to
share the word "Sharpe"; do not conflate them.

References
----------
* Bailey, D. H., & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio:
  Correcting for Selection Bias, Backtest Overfitting, and Non-Normality."
  The Journal of Portfolio Management, 40(5), 94-107.
* Bailey, D. H., Borwein, J. M., Lopez de Prado, M., & Zhu, Q. J. (2017).
  "The Probability of Backtest Overfitting." Journal of Computational
  Finance, 20(4), 39-69.
* Newey, W. K., & West, K. D. (1987). "A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix."
  Econometrica, 55(3), 703-708.
* Acklam, P. J. (2000). "An algorithm for computing the inverse normal
  cumulative distribution function" (rational approximation used by
  :func:`_normal_ppf`; accurate to ~1.15e-9 relative error before the
  Halley refinement step applied here, which pushes it to ~machine
  precision).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Standard normal CDF / inverse CDF (no scipy dependency; see module docstring)
# ---------------------------------------------------------------------------

_EULER_GAMMA = 0.5772156649015329  # Euler-Mascheroni constant


def _normal_cdf(x: float) -> float:
    """Standard normal CDF ``Phi(x)`` via the exact ``math.erf`` identity."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _normal_ppf(p: float) -> float:
    """Inverse standard normal CDF ``Phi^-1(p)``.

    Rational approximation by Acklam (2000), refined with one step of
    Halley's method against the exact :func:`_normal_cdf` (erf-based) to
    reach ~machine precision. Scalar-only (all call sites in this module
    need scalars); avoids a hard ``scipy`` dependency, see module docstring.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")

    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )
    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        x = ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) / (
            ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
        )
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )

    # One Halley refinement step (exact erf-based cdf/pdf) for extra precision.
    e = _normal_cdf(x) - p
    u = e * math.sqrt(2.0 * math.pi) * math.exp(x * x / 2.0)
    x = x - u / (1.0 + x * u / 2.0)
    return x


# ---------------------------------------------------------------------------
# 1. Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014)
# ---------------------------------------------------------------------------


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    returns_skew: float,
    returns_kurtosis: float,
    n_obs: int,
    trial_sharpe_variance: float | None = None,
) -> float:
    """Probability the true Sharpe ratio is positive, deflated for multiple
    testing and non-normal returns (Bailey & Lopez de Prado, 2014).

    The Deflated Sharpe Ratio (DSR) is the Probabilistic Sharpe Ratio (PSR)
    of the observed Sharpe evaluated against a benchmark ``SR*`` equal to
    the *expected maximum* Sharpe ratio one would observe by chance alone
    after running ``n_trials`` independent (skill-less) strategies:

        SR* = sqrt(V) * [(1 - gamma) * Phi^-1(1 - 1/N) + gamma * Phi^-1(1 - 1/(N*e))]

    where ``V`` is the variance of the Sharpe ratio *across* the N trials
    (``trial_sharpe_variance``), ``gamma`` is the Euler-Mascheroni constant,
    and ``Phi^-1`` is the inverse standard normal CDF (this is the
    Bailey-Lopez de Prado closed-form approximation to ``E[max_n SR_n]``
    for i.i.d. trials, itself due to extreme-value theory). For
    ``n_trials <= 1`` there is no multiple-testing effect and ``SR* = 0``.

    The PSR itself corrects the Sharpe ratio's standard error for skew and
    (non-excess) kurtosis of the return distribution (a Sharpe ratio
    estimated from skewed / fat-tailed returns is noisier than under the
    i.i.d.-Gaussian assumption):

        se(SR_hat) = sqrt((1 - skew*SR_hat + (kurtosis-1)/4*SR_hat^2) / (n_obs - 1))
        DSR = Phi((SR_hat - SR*) / se(SR_hat))

    Parameters
    ----------
    observed_sharpe:
        The (per-period, i.e. same period length as ``n_obs`` counts)
        Sharpe ratio of the strategy actually selected/reported.
    n_trials:
        Number of independent strategy variants effectively tried before
        settling on this one (e.g. hyperparameter/checkpoint sweep size).
        ``n_trials=1`` means "no selection bias to correct for".
    returns_skew, returns_kurtosis:
        Sample skewness and (non-excess, i.e. normal=3) kurtosis of the
        strategy's per-period return series underlying ``observed_sharpe``.
    n_obs:
        Number of return observations underlying ``observed_sharpe``.
    trial_sharpe_variance:
        Variance of the Sharpe ratio estimates *across* the ``n_trials``
        strategies (the paper's ``V[{SR_n}]``). When ``None`` (not
        available — e.g. only the trial count is known, not each trial's
        Sharpe), we fall back to the textbook single-trial estimation
        variance evaluated under the null of zero skill (``SR=0``):
        ``1 / (n_obs - 1)``. This is the standard approximation used when
        individual trial results are not retained; passing the empirical
        variance of the actual trial Sharpes is preferred when available.

    Returns
    -------
    float in [0, 1]: probability the true (population) Sharpe ratio is
    positive after deflation. Values well above 0.5 indicate the observed
    edge survives multiple-testing/non-normality scrutiny; values near or
    below 0.5 indicate it does not.
    """
    if n_obs < 2:
        raise ValueError(f"n_obs must be >= 2, got {n_obs}")
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")

    if trial_sharpe_variance is None:
        trial_sharpe_variance = 1.0 / (n_obs - 1)
    if trial_sharpe_variance < 0:
        raise ValueError("trial_sharpe_variance must be >= 0")

    if n_trials <= 1:
        sr_benchmark = 0.0
    else:
        trial_sr_std = math.sqrt(trial_sharpe_variance)
        sr_benchmark = trial_sr_std * (
            (1.0 - _EULER_GAMMA) * _normal_ppf(1.0 - 1.0 / n_trials)
            + _EULER_GAMMA * _normal_ppf(1.0 - 1.0 / (n_trials * math.e))
        )

    variance_term = (
        1.0 - returns_skew * observed_sharpe + (returns_kurtosis - 1.0) / 4.0 * observed_sharpe**2
    )
    variance_term = max(variance_term, 1e-12)  # guard against pathological skew/kurtosis inputs
    se = math.sqrt(variance_term / (n_obs - 1))

    z = (observed_sharpe - sr_benchmark) / se
    return _normal_cdf(z)


# ---------------------------------------------------------------------------
# 2. Probability of Backtest Overfitting via CSCV (Bailey et al., 2017)
# ---------------------------------------------------------------------------


def _split_sharpe_stat(block: np.ndarray) -> np.ndarray:
    """Per-strategy Sharpe-like ranking statistic (mean / std) for a block
    of rows. Falls back to the mean alone when std is degenerate (single
    row, or a constant column) so a short/degenerate split still ranks
    strategies by their central tendency instead of raising/NaN-ing."""
    mean = block.mean(axis=0)
    if block.shape[0] > 1:
        std = block.std(axis=0, ddof=1)
    else:
        std = np.zeros_like(mean)
    std = np.where(std < 1e-12, 1.0, std)
    return mean / std


def probability_of_backtest_overfitting(perf_matrix: np.ndarray, n_splits: int = 16) -> float:
    """Probability of Backtest Overfitting via Combinatorially Symmetric
    Cross-Validation (CSCV), Bailey, Borwein, Lopez de Prado & Zhu (2017).

    ``perf_matrix`` is a ``(T, N)`` array of per-period performance
    (e.g. per-date returns) for ``N`` candidate strategies observed over
    ``T`` periods. The algorithm:

    1. Split the ``T`` rows (in original time order) into ``n_splits``
       contiguous blocks ``S_1, ..., S_S``.
    2. For every way of choosing ``S/2`` blocks as the "in-sample" (IS) set
       (the complementary ``S/2`` blocks form the "out-of-sample" (OOS)
       set) — ``C(S, S/2)`` combinations in total:

       a. Rank the ``N`` strategies by a Sharpe-like statistic computed on
          the IS rows; let ``n*`` be the IS-best strategy.
       b. Find ``n*``'s relative rank ``omega_c = rank_OOS(n*) / (N + 1)``
          among the ``N`` strategies' OOS performance (rank 1 = worst).
       c. Compute the logit ``lambda_c = ln(omega_c / (1 - omega_c))``.
          ``lambda_c <= 0`` means the IS-best strategy performed at or
          below the OOS median — i.e. its in-sample edge did not persist.

    3. ``PBO = fraction of combinations with lambda_c <= 0``.

    A strategy selection process with no real skill (pure noise) is
    expected to reproduce the median OOS rank about half the time, so
    ``PBO`` for a pure-noise strategy pool is expected near 0.5. A strategy
    pool containing one strategy with genuine persistent edge (the
    IS-selection procedure reliably finds it, and it reliably ranks well
    OOS too) drives ``PBO`` toward 0.

    Parameters
    ----------
    perf_matrix: (T, N) array-like of per-period performance.
    n_splits: number of contiguous blocks ``S`` (must be even, >= 2, and
        <= T). The original paper's illustrative examples use ``S=16``.

    Returns
    -------
    float in [0, 1]: the estimated probability of backtest overfitting.
    """
    perf = np.asarray(perf_matrix, dtype=np.float64)
    if perf.ndim != 2:
        raise ValueError(f"perf_matrix must be 2D (T, N), got shape {perf.shape}")
    t_obs, n_strategies = perf.shape
    if n_strategies < 2:
        raise ValueError("need at least 2 strategies to rank")
    if n_splits < 2 or n_splits % 2 != 0:
        raise ValueError(f"n_splits must be an even integer >= 2, got {n_splits}")
    if t_obs < n_splits:
        raise ValueError(f"n_splits={n_splits} exceeds T={t_obs} observations")

    groups = np.array_split(np.arange(t_obs), n_splits)
    half = n_splits // 2
    all_groups = set(range(n_splits))

    below_median = 0
    n_combos = 0
    for combo in itertools.combinations(range(n_splits), half):
        is_rows = np.concatenate([groups[g] for g in combo])
        oos_rows = np.concatenate([groups[g] for g in sorted(all_groups - set(combo))])

        is_stat = _split_sharpe_stat(perf[is_rows])
        oos_stat = _split_sharpe_stat(perf[oos_rows])

        best_is = int(np.argmax(is_stat))
        # Rank of the IS-best strategy within the OOS statistics (1 = worst, N = best).
        rank = int(np.sum(oos_stat <= oos_stat[best_is]))
        omega = rank / (n_strategies + 1)
        omega = min(max(omega, 1e-9), 1.0 - 1e-9)
        logit = math.log(omega / (1.0 - omega))

        if logit <= 0:
            below_median += 1
        n_combos += 1

    return below_median / n_combos


# ---------------------------------------------------------------------------
# 3. Spearman rank IC (pure numpy; alongside the existing Pearson IC)
# ---------------------------------------------------------------------------


def _rank_average_ties(x: np.ndarray) -> np.ndarray:
    """1-based ranks with ties resolved to the average rank of the tied
    group (the standard convention for Spearman's rho), pure numpy."""
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    n = len(x)
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def spearman_ic(scores: np.ndarray, forward_returns: np.ndarray) -> float:
    """Spearman rank correlation between ``scores`` and ``forward_returns``
    for a single cross-section (1D arrays of equal length).

    Unlike Pearson IC, Spearman IC is invariant to any monotone (not
    necessarily linear) transform of either input, so it is not penalised
    by a monotone-nonlinear relationship between model score and forward
    return the way Pearson IC is. NaN-aware: non-finite entries in either
    array are dropped pairwise before ranking. Returns 0.0 when fewer than
    2 valid pairs remain or either side is constant (degenerate rank
    correlation).
    """
    s = np.asarray(scores, dtype=np.float64)
    r = np.asarray(forward_returns, dtype=np.float64)
    if s.shape != r.shape:
        raise ValueError(f"shape mismatch: scores {s.shape} vs forward_returns {r.shape}")
    mask = np.isfinite(s) & np.isfinite(r)
    if mask.sum() < 2:
        return 0.0
    rs = _rank_average_ties(s[mask])
    rr = _rank_average_ties(r[mask])
    if np.std(rs) < 1e-12 or np.std(rr) < 1e-12:
        return 0.0
    c = np.corrcoef(rs, rr)[0, 1]
    return float(c) if np.isfinite(c) else 0.0


def spearman_ic_per_date(predictions: np.ndarray, returns: np.ndarray) -> list[float]:
    """Per-date Spearman IC for a (n_dates, n_stocks) panel.

    Mirrors ``backtest._per_date_ics`` (the Pearson counterpart): degenerate
    dates (fewer than 2 valid pairs, or a constant side) are SKIPPED, so the
    result is suitable for a scalar aggregate (``np.mean``), not for
    plotting against a dates axis (which would need one entry per date).
    """
    if predictions.shape != returns.shape:
        raise ValueError(
            f"shape mismatch: predictions {predictions.shape} vs returns {returns.shape}"
        )
    out: list[float] = []
    for t in range(predictions.shape[0]):
        p, r = predictions[t], returns[t]
        mask = np.isfinite(p) & np.isfinite(r)
        if mask.sum() < 2:
            continue
        if np.std(p[mask]) < 1e-12 or np.std(r[mask]) < 1e-12:
            continue
        out.append(spearman_ic(p[mask], r[mask]))
    return out


# ---------------------------------------------------------------------------
# 4. Newey-West / HAC standard error (Newey & West, 1987)
# ---------------------------------------------------------------------------


def hac_standard_error(series: np.ndarray, lag: int) -> float:
    """Newey-West (1987) HAC standard error of the sample MEAN of ``series``.

    The backtest's top-K return series is built from OVERLAPPING
    ``forward_period``-day forward returns re-sampled daily, so consecutive
    observations are mechanically autocorrelated up to ``lag = forward_period
    - 1``. The naive standard error of the mean (``std(ddof=1) / sqrt(n)``)
    assumes i.i.d. observations and therefore UNDERSTATES the true
    uncertainty; the HAC/Newey-West estimator corrects for this.

    Long-run variance (Bartlett kernel, ``n - 1`` denominator throughout so
    ``lag=0`` reduces EXACTLY to the naive ``std(ddof=1)``-based variance):

        u_t = x_t - mean(x)
        gamma_0 = sum(u_t^2) / (n - 1)
        gamma_j = sum(u_t * u_{t-j}) / (n - 1),  j = 1..lag
        w_j = 1 - j / (lag + 1)                   (Bartlett kernel weight)
        S = gamma_0 + 2 * sum_j w_j * gamma_j
        HAC SE of the mean = sqrt(S / n)

    Parameters
    ----------
    series: 1D array of per-period values (e.g. a top-K return series).
        Non-finite entries are dropped before computing.
    lag: maximum autocovariance lag to include (Bartlett truncation). Use
        ``forward_period - 1`` for an overlapping ``forward_period``-day
        return series. ``lag=0`` reduces to the naive (i.i.d.) SEM.

    Returns
    -------
    float: the HAC standard error of the sample mean. 0.0 if fewer than 2
    finite observations remain.
    """
    x = np.asarray(series, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return 0.0
    if lag < 0:
        raise ValueError(f"lag must be >= 0, got {lag}")
    if lag >= n:
        raise ValueError(f"lag ({lag}) must be < number of observations ({n})")

    u = x - x.mean()
    gamma0 = float(np.dot(u, u) / (n - 1))
    s = gamma0
    for j in range(1, lag + 1):
        w = 1.0 - j / (lag + 1)
        gamma_j = float(np.dot(u[j:], u[:-j]) / (n - 1))
        s += 2.0 * w * gamma_j
    s = max(s, 0.0)  # long-run variance estimate can dip slightly negative for small n
    return math.sqrt(s / n)


def hac_mean_ci(series: np.ndarray, lag: int, confidence: float = 0.95) -> dict[str, float]:
    """HAC-adjusted t-stat and confidence interval for the mean of ``series``.

    Uses the asymptotic normal critical value (appropriate for HAC/Newey-West,
    itself an asymptotic estimator) rather than a small-sample t-distribution
    quantile. See :func:`hac_standard_error` for the SE formula and the
    autocorrelation-from-overlap rationale.

    Returns a dict with keys ``mean``, ``se``, ``t_stat``, ``ci_low``,
    ``ci_high``, ``lag``, ``confidence``, ``n``.
    """
    x = np.asarray(series, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if n == 0:
        return {
            "mean": 0.0,
            "se": 0.0,
            "t_stat": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "lag": float(lag),
            "confidence": confidence,
            "n": 0.0,
        }

    mean = float(x.mean())
    se = hac_standard_error(x, lag)
    t_stat = mean / se if se > 1e-15 else 0.0
    z = _normal_ppf(0.5 + confidence / 2.0)
    half_width = z * se
    return {
        "mean": mean,
        "se": se,
        "t_stat": t_stat,
        "ci_low": mean - half_width,
        "ci_high": mean + half_width,
        "lag": float(lag),
        "confidence": confidence,
        "n": float(n),
    }


# ---------------------------------------------------------------------------
# 6. Selection / confirmation date split
# ---------------------------------------------------------------------------


def split_selection_confirmation(dates: Sequence, confirm_frac: float = 0.3) -> tuple[list, list]:
    """Partition an ordered OOS date range into a selection window and an
    untouched confirmation window (the tail), so a "best checkpoint" (or
    best strategy variant) can be CHOSEN on the selection window while its
    metric on the confirmation window is REPORTED, not used to choose.

    This is the same multiple-testing hazard the Deflated Sharpe Ratio
    corrects for analytically (Bailey & Lopez de Prado, 2014): picking the
    best-looking result out of many candidates on the same evaluation
    window overstates its true out-of-sample edge. A held-out confirmation
    window measured strictly AFTER selection is the model-free way to
    detect that bias.

    Parameters
    ----------
    dates: ordered (ascending) sequence of dates.
    confirm_frac: fraction of ``dates`` (rounded to the nearest integer
        count, floored to leave >= 1 date in the selection window)
        assigned to the confirmation window, taken from the TAIL of
        ``dates``. Must be in ``(0, 1)``.

    Returns
    -------
    (selection_dates, confirmation_dates): two lists that partition
    ``dates`` with no overlap; ``confirmation_dates`` is strictly the tail
    (all dates in ``confirmation_dates`` are later than all dates in
    ``selection_dates``, since ``dates`` is assumed ascending); the split
    is deterministic given ``(dates, confirm_frac)``.
    """
    n = len(dates)
    if n < 2:
        raise ValueError(f"need at least 2 dates to split, got {n}")
    if not 0.0 < confirm_frac < 1.0:
        raise ValueError(f"confirm_frac must be in (0, 1), got {confirm_frac}")

    n_confirm = round(n * confirm_frac)
    n_confirm = max(1, min(n_confirm, n - 1))
    split_idx = n - n_confirm
    return list(dates[:split_idx]), list(dates[split_idx:])


__all__ = [
    "deflated_sharpe_ratio",
    "probability_of_backtest_overfitting",
    "spearman_ic",
    "spearman_ic_per_date",
    "hac_standard_error",
    "hac_mean_ci",
    "split_selection_confirmation",
]
