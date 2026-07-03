"""Backtest utilities for evaluating a trained policy on a held-out window.

Pure-numpy module with no torch / SB3 dependency, so it can be used both
inside the training loop (validation callback) and from a CLI that loads
an ONNX policy via onnxruntime.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from aurumq_rl.eval_metrics import spearman_ic_per_date


@dataclass
class BacktestResult:
    """Outcome of a single backtest run.

    Phase 16: when ``forward_period > 1``, the per-day returns are
    overlapping ``forward_period``-day windows. Annualising by
    ``sqrt(252)`` then over-states Sharpe by ``sqrt(forward_period)``.
    The result therefore carries three Sharpe values:

    * ``top_k_sharpe_legacy``: ``sqrt(252)`` annualisation. Inflated.
    * ``top_k_sharpe_adjusted``: ``sqrt(252 / forward_period)``.
      The honest annualised Sharpe of an N-day forward-return stream
      sampled daily. **Phase 16's primary metric.**
    * ``top_k_sharpe_non_overlap``: same as adjusted but on a
      non-overlapping subsample (every ``forward_period``-th row).
      Lower-variance independent estimator; sanity check.

    For backward compatibility ``top_k_sharpe`` is set to the adjusted
    Sharpe (Phase 16's primary), so callers that index ``top_k_sharpe``
    automatically pick up the corrected scale. ``random_baseline``
    contains both ``*_sharpe`` (legacy) and ``*_sharpe_adjusted`` /
    ``*_sharpe_non_overlap`` keys so the comparison can be done at
    matching scales.

    Issue #6 (additive): ``ic_spearman`` is the Spearman rank-IC companion
    to ``ic`` (Pearson), always computed alongside it — Spearman is
    invariant to monotone-nonlinear score/return relationships that
    penalise Pearson. ``top_k_sharpe_cost_adjusted`` /
    ``top_k_cumret_cost_adjusted`` are OPT-IN (only computed when
    ``run_backtest(..., cost_bps=...)`` is passed a positive value; they
    default to 0.0 and do not affect any existing field). ``hac`` is an
    OPT-IN dict (empty by default) that CLI callers (e.g.
    ``scripts/eval_backtest.py``) may populate with
    :func:`aurumq_rl.eval_metrics.hac_mean_ci` on the top-K return series —
    left to the caller because the honest HAC lag depends on which return
    series (padded per-date vs skip-degenerate) the caller has in hand.
    """

    ic: float
    ic_ir: float
    top_k_sharpe: float
    top_k_cumret: float
    random_baseline: dict[str, float] = field(default_factory=dict)
    n_dates: int = 0
    n_stocks: int = 0
    top_k: int = 0
    forward_period: int = 1
    top_k_sharpe_legacy: float = 0.0
    top_k_sharpe_adjusted: float = 0.0
    top_k_sharpe_non_overlap: float = 0.0
    ic_spearman: float = 0.0
    cost_bps: float = 0.0
    top_k_sharpe_cost_adjusted: float = 0.0
    top_k_cumret_cost_adjusted: float = 0.0
    hac: dict[str, float] = field(default_factory=dict)

    def to_json(self, path: Path | str) -> None:
        Path(path).write_text(
            json.dumps(asdict(self), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def from_json(cls, path: Path | str) -> BacktestResult:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)


def _apply_tradeable_mask(predictions: np.ndarray, tradeable_mask: np.ndarray | None) -> np.ndarray:
    """NaN-out predictions for untradeable (date, stock) cells (M5).

    Semantics: cells where ``tradeable_mask`` is False are excluded from
    BOTH top-k selection and IC computation (every helper already drops
    non-finite predictions), matching the training-side valid_mask built by
    ``data_loader.build_tradeable_mask``.
    """
    if tradeable_mask is None:
        return predictions
    mask = np.asarray(tradeable_mask, dtype=bool)
    if mask.shape != predictions.shape:
        raise ValueError(
            f"tradeable_mask shape {mask.shape} != predictions shape {predictions.shape}"
        )
    return np.where(mask, predictions, np.nan)


def _per_date_ics(predictions: np.ndarray, returns: np.ndarray) -> list[float]:
    if predictions.shape != returns.shape:
        raise ValueError(
            f"shape mismatch: predictions {predictions.shape} vs returns {returns.shape}"
        )

    ics: list[float] = []
    for t in range(predictions.shape[0]):
        p, r = predictions[t], returns[t]
        mask = np.isfinite(p) & np.isfinite(r)
        if mask.sum() < 2:
            continue
        if np.std(p[mask]) < 1e-12 or np.std(r[mask]) < 1e-12:
            continue
        c = np.corrcoef(p[mask], r[mask])[0, 1]
        if np.isfinite(c):
            ics.append(float(c))
    return ics


def _per_date_ics_aligned(predictions: np.ndarray, returns: np.ndarray) -> list[float]:
    """Per-date IC aligned to predictions.shape[0]. 0.0 for degenerate days.

    Unlike ``_per_date_ics`` (which is the canonical helper for scalar
    statistics and SKIPS degenerate days), this returns one entry per row of
    ``predictions`` so the result can be plotted directly against a dates
    axis. Use this for per-date series, never for scalar IC / IC-IR.
    """
    if predictions.shape != returns.shape:
        raise ValueError(
            f"shape mismatch: predictions {predictions.shape} vs returns {returns.shape}"
        )
    out: list[float] = []
    for t in range(predictions.shape[0]):
        p, r = predictions[t], returns[t]
        mask = np.isfinite(p) & np.isfinite(r)
        if mask.sum() < 2 or np.std(p[mask]) < 1e-12 or np.std(r[mask]) < 1e-12:
            out.append(0.0)
            continue
        c = np.corrcoef(p[mask], r[mask])[0, 1]
        out.append(float(c) if np.isfinite(c) else 0.0)
    return out


def compute_ic(predictions: np.ndarray, returns: np.ndarray) -> float:
    """Mean per-date Pearson IC between predictions and forward returns.

    Both arrays have shape (n_dates, n_stocks). NaNs are handled per-date.
    """
    ics = _per_date_ics(predictions, returns)
    return float(np.mean(ics)) if ics else 0.0


def compute_ic_ir(predictions: np.ndarray, returns: np.ndarray) -> float:
    """IC / std(IC) over time. Returns 0.0 when std is degenerate."""
    ics = _per_date_ics(predictions, returns)
    if len(ics) < 2:
        return 0.0
    arr = np.asarray(ics)
    std = float(arr.std(ddof=1))
    if std < 1e-12:
        return 0.0
    return float(arr.mean() / std)


def compute_ic_spearman(predictions: np.ndarray, returns: np.ndarray) -> float:
    """Mean per-date Spearman rank IC between predictions and forward returns.

    Additive companion to :func:`compute_ic` (Pearson) — issue #6. Spearman
    IC is invariant to monotone-nonlinear score/return relationships that
    would otherwise depress the Pearson correlation; see
    :func:`aurumq_rl.eval_metrics.spearman_ic`. Does not affect
    :func:`compute_ic`'s value.
    """
    ics = spearman_ic_per_date(predictions, returns)
    return float(np.mean(ics)) if ics else 0.0


def _top_k_indices(pred_row: np.ndarray, ret_row: np.ndarray, top_k: int) -> np.ndarray | None:
    """Full-universe column indices of the top-K picks for one date, sorted
    by descending prediction. ``None`` when fewer than ``top_k`` cells are
    finite (degenerate day, matches ``_top_k_returns_series`` semantics)."""
    mask = np.isfinite(pred_row) & np.isfinite(ret_row)
    if mask.sum() < top_k:
        return None
    valid_idx = np.nonzero(mask)[0]
    order = np.argsort(-pred_row[valid_idx])[:top_k]
    return valid_idx[order]


def _jaccard_turnover(prev: set[int] | None, curr: set[int]) -> float:
    """Jaccard turnover (distance) between two top-K stock-index sets:
    ``1 - |intersection| / |union|``. 0.0 = identical portfolio, 1.0 =
    fully disjoint. ``prev=None`` (no prior portfolio, e.g. the first
    tradeable day) costs nothing."""
    if prev is None:
        return 0.0
    union = prev | curr
    if not union:
        return 0.0
    inter = prev & curr
    return 1.0 - len(inter) / len(union)


def top_k_returns_series_cost_adjusted(
    predictions: np.ndarray,
    returns: np.ndarray,
    top_k: int,
    cost_bps: float = 0.0,
) -> list[float]:
    """Per-date top-K equal-weight return, net of an OPT-IN turnover cost.

    Issue #6: each day's gross top-K return is reduced by
    ``turnover_t * cost_bps / 1e4``, where ``turnover_t`` is the Jaccard
    turnover (:func:`_jaccard_turnover`) between today's and yesterday's
    top-K stock-index sets. This mirrors the ~30bps/step transaction cost
    already subtracted from the training reward, so the eval metric can
    target the same net quantity training optimizes.

    With ``cost_bps == 0`` (the default) this reproduces
    ``_top_k_returns_series`` EXACTLY, byte-for-byte: the turnover term
    always multiplies to ``0.0`` regardless of the (always-finite,
    always-in-``[0, 1]``) turnover value, and the gross-return computation
    is identical (same mask, same ``argsort``, same ``.mean()``). Degenerate
    days (fewer than ``top_k`` finite pairs) are SKIPPED, exactly like
    ``_top_k_returns_series``, and do not reset the turnover chain — the
    last tracked portfolio carries forward across a skipped day.
    """
    if predictions.shape != returns.shape:
        raise ValueError("shape mismatch")
    out: list[float] = []
    prev_set: set[int] | None = None
    for t in range(predictions.shape[0]):
        idx = _top_k_indices(predictions[t], returns[t], top_k)
        if idx is None:
            continue
        curr_set = set(idx.tolist())
        gross = float(returns[t][idx].mean())
        turnover = _jaccard_turnover(prev_set, curr_set)
        cost = turnover * (cost_bps / 1e4)
        out.append(gross - cost)
        prev_set = curr_set
    return out


def _top_k_returns_series(predictions: np.ndarray, returns: np.ndarray, top_k: int) -> list[float]:
    """Per-date top-K equal-weight portfolio return; degenerate days skipped."""
    if predictions.shape != returns.shape:
        raise ValueError("shape mismatch")
    out: list[float] = []
    for t in range(predictions.shape[0]):
        p, r = predictions[t], returns[t]
        mask = np.isfinite(p) & np.isfinite(r)
        if mask.sum() < top_k:
            continue
        idx = np.argsort(-p[mask])[:top_k]
        out.append(float(r[mask][idx].mean()))
    return out


def compute_top_k_sharpe(predictions: np.ndarray, returns: np.ndarray, top_k: int) -> float:
    """Legacy ``sqrt(252)`` annualised Sharpe (Phase ≤15 metric).

    Kept as the historical name so existing callers / tests behave the same.
    For Phase 16 use :func:`compute_top_k_sharpes`.
    """
    series = _top_k_returns_series(predictions, returns, top_k)
    if len(series) < 2:
        return 0.0
    arr = np.asarray(series)
    std = arr.std(ddof=1)
    if std < 1e-12:
        return 0.0
    return float(arr.mean() / std * np.sqrt(252))


def compute_top_k_sharpes(
    predictions: np.ndarray,
    returns: np.ndarray,
    top_k: int,
    forward_period: int = 1,
) -> dict[str, float]:
    """Three Sharpe estimates of the top-K portfolio.

    Returns a dict with keys ``legacy``, ``adjusted``, ``non_overlap``.
    See :class:`BacktestResult` for semantics.
    """
    series = _top_k_returns_series(predictions, returns, top_k)
    if len(series) < 2:
        return {"legacy": 0.0, "adjusted": 0.0, "non_overlap": 0.0}
    arr = np.asarray(series)
    std = arr.std(ddof=1)
    if std < 1e-12:
        return {"legacy": 0.0, "adjusted": 0.0, "non_overlap": 0.0}
    legacy = float(arr.mean() / std * np.sqrt(252))
    adjusted = float(arr.mean() / std * np.sqrt(252 / max(forward_period, 1)))
    if forward_period > 1 and len(arr) >= 2 * forward_period:
        sub = arr[::forward_period]
        sub_std = sub.std(ddof=1)
        if sub_std > 1e-12:
            non_overlap = float(sub.mean() / sub_std * np.sqrt(252 / forward_period))
        else:
            non_overlap = 0.0
    else:
        non_overlap = adjusted
    return {"legacy": legacy, "adjusted": adjusted, "non_overlap": non_overlap}


def _sharpe_and_cumret_from_series(series: list[float], forward_period: int) -> tuple[float, float]:
    """Adjusted (``sqrt(252/forward_period)``) Sharpe + total cumulative
    return of an arbitrary per-date return series. Shared by the gross and
    cost-adjusted paths so both use identical arithmetic."""
    if len(series) < 2:
        return 0.0, 0.0
    arr = np.asarray(series)
    cumret = float(np.prod(1.0 + arr) - 1.0)
    std = arr.std(ddof=1)
    if std < 1e-12:
        return 0.0, cumret
    sharpe = float(arr.mean() / std * np.sqrt(252 / max(forward_period, 1)))
    return sharpe, cumret


def compute_top_k_cumret(predictions: np.ndarray, returns: np.ndarray, top_k: int) -> float:
    """Total cumulative return of the top-K portfolio."""
    if predictions.shape != returns.shape:
        raise ValueError("shape mismatch")

    cum = 1.0
    for t in range(predictions.shape[0]):
        p, r = predictions[t], returns[t]
        mask = np.isfinite(p) & np.isfinite(r)
        if mask.sum() < top_k:
            continue
        idx = np.argsort(-p[mask])[:top_k]
        cum *= 1.0 + float(r[mask][idx].mean())
    return cum - 1.0


def random_baseline(
    returns: np.ndarray,
    top_k: int,
    n_simulations: int = 100,
    seed: int = 0,
    forward_period: int = 1,
    tradeable_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Sharpe distribution of random top-K portfolios over the same dates.

    Phase 16 reports legacy / adjusted / non-overlap percentiles. The
    legacy fields are kept because existing dashboards consume them; for
    the production "vs random" comparison use the ``*_adjusted`` keys.

    ``tradeable_mask`` (M5): random portfolios sample from the SAME
    tradeable set as the policy, otherwise the "vs random" comparison is
    skewed (random could buy limit-up/suspended stocks the policy cannot).
    """
    rng = np.random.default_rng(seed)
    legacy: list[float] = []
    adjusted: list[float] = []
    non_overlap: list[float] = []
    for _ in range(n_simulations):
        preds = _apply_tradeable_mask(rng.normal(size=returns.shape), tradeable_mask)
        d = compute_top_k_sharpes(
            preds,
            returns,
            top_k=top_k,
            forward_period=forward_period,
        )
        legacy.append(d["legacy"])
        adjusted.append(d["adjusted"])
        non_overlap.append(d["non_overlap"])

    def _pct(arr_list: list[float]) -> dict[str, float]:
        a = np.asarray(arr_list)
        return {
            "mean": float(a.mean()),
            "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "p05": float(np.percentile(a, 5)),
            "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)),
        }

    leg = _pct(legacy)
    adj = _pct(adjusted)
    nov = _pct(non_overlap)
    # Backward-compatible flat keys (legacy scale) plus explicit adjusted/non-overlap.
    return {
        "mean_sharpe": leg["mean"],
        "std_sharpe": leg["std"],
        "p05_sharpe": leg["p05"],
        "p50_sharpe": leg["p50"],
        "p95_sharpe": leg["p95"],
        "mean_sharpe_adjusted": adj["mean"],
        "std_sharpe_adjusted": adj["std"],
        "p05_sharpe_adjusted": adj["p05"],
        "p50_sharpe_adjusted": adj["p50"],
        "p95_sharpe_adjusted": adj["p95"],
        "mean_sharpe_non_overlap": nov["mean"],
        "p05_sharpe_non_overlap": nov["p05"],
        "p50_sharpe_non_overlap": nov["p50"],
        "p95_sharpe_non_overlap": nov["p95"],
    }


def _truncate_trailing_forward_rows(
    predictions: np.ndarray,
    returns: np.ndarray,
    forward_period: int,
    tradeable_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Drop the trailing ``forward_period`` rows when ``forward_period > 1``.

    :class:`FactorPanelLoader` leaves the last ``forward_period`` rows of
    ``return_array`` as literal ``0.0`` (there is no future close to compute
    the forward log-return). Those rows are FINITE, so the per-date
    ``mask.sum() < top_k`` "degenerate day" guard does NOT skip them — they
    would otherwise enter every return series as spurious all-zero
    observations, understating the HAC SE and biasing the mean toward zero.

    This is the single source of truth for that truncation, shared by
    :func:`run_backtest` (scalar path, Phase 16) and
    :func:`run_backtest_with_series` (the issue #6 series path) so the two
    can never drift. Returns the (possibly) truncated
    ``(predictions, returns, tradeable_mask)`` triple; ``tradeable_mask``
    stays ``None`` if it was ``None``.
    """
    if forward_period > 1 and predictions.shape[0] > forward_period:
        keep = predictions.shape[0] - forward_period
        predictions = predictions[:keep]
        returns = returns[:keep]
        if tradeable_mask is not None:
            tradeable_mask = tradeable_mask[:keep]
    return predictions, returns, tradeable_mask


def run_backtest(
    predictions: np.ndarray,
    returns: np.ndarray,
    top_k: int = 30,
    n_random_simulations: int = 100,
    random_seed: int = 0,
    forward_period: int = 1,
    tradeable_mask: np.ndarray | None = None,
    cost_bps: float = 0.0,
) -> BacktestResult:
    """One-shot evaluation: IC + IR + top-K Sharpe trio + random baseline.

    Phase 16: when ``forward_period > 1`` we additionally truncate the
    last ``forward_period`` rows of both ``predictions`` and ``returns``
    so the trailing all-zero rows produced by
    :class:`FactorPanelLoader` (which has no future close to compute the
    forward log-return) do not drag down the Sharpe mean.

    ``tradeable_mask`` (M5): optional (n_dates, n_stocks) bool. False cells
    (suspended / ST / at price limit / IPO window) are excluded from top-k
    selection AND from IC (predictions set to NaN), matching the training
    valid_mask convention. The random baseline samples from the same set.

    Issue #6 (additive): ``ic_spearman`` is always computed alongside
    ``ic`` (Pearson) — cheap, does not change ``ic``. ``cost_bps`` is
    OPT-IN and defaults to 0.0: the cost-adjusted fields
    (``top_k_sharpe_cost_adjusted`` / ``top_k_cumret_cost_adjusted``) are
    only computed (non-zero) when ``cost_bps > 0``; at the default they
    stay at their dataclass default and every other field is byte-for-byte
    identical to the pre-#6 output.
    """
    predictions = _apply_tradeable_mask(predictions, tradeable_mask)
    predictions, returns, tradeable_mask = _truncate_trailing_forward_rows(
        predictions, returns, forward_period, tradeable_mask
    )
    sharpes = compute_top_k_sharpes(
        predictions,
        returns,
        top_k=top_k,
        forward_period=forward_period,
    )
    cost_sharpe = 0.0
    cost_cumret = 0.0
    if cost_bps > 0:
        cost_series = top_k_returns_series_cost_adjusted(
            predictions, returns, top_k=top_k, cost_bps=cost_bps
        )
        cost_sharpe, cost_cumret = _sharpe_and_cumret_from_series(cost_series, forward_period)
    return BacktestResult(
        ic=compute_ic(predictions, returns),
        ic_ir=compute_ic_ir(predictions, returns),
        top_k_sharpe=sharpes["adjusted"],  # primary metric for Phase 16
        top_k_cumret=compute_top_k_cumret(predictions, returns, top_k),
        random_baseline=random_baseline(
            returns,
            top_k=top_k,
            n_simulations=n_random_simulations,
            seed=random_seed,
            forward_period=forward_period,
            tradeable_mask=tradeable_mask,
        ),
        n_dates=predictions.shape[0],
        n_stocks=predictions.shape[1],
        top_k=top_k,
        forward_period=forward_period,
        top_k_sharpe_legacy=sharpes["legacy"],
        top_k_sharpe_adjusted=sharpes["adjusted"],
        top_k_sharpe_non_overlap=sharpes["non_overlap"],
        ic_spearman=compute_ic_spearman(predictions, returns),
        cost_bps=cost_bps,
        top_k_sharpe_cost_adjusted=cost_sharpe,
        top_k_cumret_cost_adjusted=cost_cumret,
    )


@dataclass
class BacktestSeries:
    """Per-date series produced alongside the BacktestResult.

    Issue #6 (additive): ``top_k_returns_cost_adjusted`` is OPT-IN — empty
    unless ``run_backtest_with_series(..., cost_bps=...)`` is passed a
    positive value; see :func:`top_k_returns_series_cost_adjusted`.
    ``top_k_returns_skip_degenerate`` is always populated: the same gross
    top-K return series as ``top_k_returns`` but with degenerate days
    SKIPPED rather than padded with ``0.0`` AND with the trailing
    ``forward_period`` FactorPanelLoader zero-return rows truncated (exactly
    as ``run_backtest``'s scalar path does) — the correct input for
    autocorrelation-sensitive statistics (e.g. HAC SE), since neither a
    padded 0.0 nor a trailing loader-0.0 row is a real observation and
    either would distort the estimated autocovariance structure. Because of
    the skip + truncation this series is generally SHORTER than
    ``dates`` / ``top_k_returns`` and is not date-aligned — do not plot it
    against the date axis; use ``top_k_returns`` for charts.
    """

    dates: list[str]
    ic: list[float]
    top_k_returns: list[float]
    equity_curve: list[float]
    random_baseline_sharpes: list[float] = field(default_factory=list)
    top_k_returns_cost_adjusted: list[float] = field(default_factory=list)
    top_k_returns_skip_degenerate: list[float] = field(default_factory=list)

    def to_json(self, path: Path | str) -> None:
        Path(path).write_text(
            json.dumps(asdict(self), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def from_json(cls, path: Path | str) -> BacktestSeries:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)


def _per_date_top_k_returns(
    predictions: np.ndarray, returns: np.ndarray, top_k: int
) -> list[float]:
    out: list[float] = []
    for t in range(predictions.shape[0]):
        p, r = predictions[t], returns[t]
        mask = np.isfinite(p) & np.isfinite(r)
        if mask.sum() < top_k:
            out.append(0.0)
            continue
        idx = np.argsort(-p[mask])[:top_k]
        out.append(float(r[mask][idx].mean()))
    return out


def _random_sharpes(
    returns: np.ndarray,
    top_k: int,
    n_simulations: int,
    seed: int,
    tradeable_mask: np.ndarray | None = None,
) -> list[float]:
    rng = np.random.default_rng(seed)
    out: list[float] = []
    for _ in range(n_simulations):
        # M5: random portfolios sample from the same tradeable set.
        preds = _apply_tradeable_mask(rng.normal(size=returns.shape), tradeable_mask)
        out.append(compute_top_k_sharpe(preds, returns, top_k=top_k))
    return out


def run_backtest_with_series(
    predictions: np.ndarray,
    returns: np.ndarray,
    dates: list,
    top_k: int = 30,
    n_random_simulations: int = 100,
    random_seed: int = 0,
    forward_period: int = 1,
    tradeable_mask: np.ndarray | None = None,
    cost_bps: float = 0.0,
) -> tuple[BacktestResult, BacktestSeries]:
    """One-shot evaluation that also returns per-date / per-simulation series.

    The scalar BacktestResult uses identical semantics to ``run_backtest`` —
    degenerate days are SKIPPED, not padded — so ``backtest.json`` is stable
    across both code paths. The per-date series, by contrast, must align to
    every entry in ``dates``; degenerate days are filled with 0.0 in the
    series so chart positions line up with the date axis.

    ``tradeable_mask`` (M5): see :func:`run_backtest` — applied to top-k
    selection, IC and the random baseline alike.

    ``cost_bps`` (issue #6, additive/opt-in): defaults to 0.0, in which case
    ``series.top_k_returns_cost_adjusted`` stays empty and every other
    field is unchanged. When positive,
    ``series.top_k_returns_cost_adjusted`` holds the turnover-cost-net
    series (skipped/degenerate days are simply absent, unlike
    ``top_k_returns`` which pads them with 0.0 — see
    :func:`top_k_returns_series_cost_adjusted`).
    """
    if predictions.shape != returns.shape:
        raise ValueError("shape mismatch")
    if len(dates) != predictions.shape[0]:
        raise ValueError(f"dates length {len(dates)} != n_dates {predictions.shape[0]}")

    # Canonical scalars — same semantics as run_backtest() so backtest.json is stable.
    result = run_backtest(
        predictions=predictions,
        returns=returns,
        top_k=top_k,
        n_random_simulations=n_random_simulations,
        random_seed=random_seed,
        forward_period=forward_period,
        tradeable_mask=tradeable_mask,
        cost_bps=cost_bps,
    )

    # Per-date series for charts (aligned to dates; degenerate days -> 0.0).
    predictions = _apply_tradeable_mask(predictions, tradeable_mask)
    ic_per_date = _per_date_ics_aligned(predictions, returns)
    top_k_rets = _per_date_top_k_returns(predictions, returns, top_k)

    equity = []
    cum = 1.0
    for ret in top_k_rets:
        cum *= 1.0 + ret
        equity.append(cum)

    random_sharpes = _random_sharpes(
        returns,
        top_k=top_k,
        n_simulations=n_random_simulations,
        seed=random_seed,
        tradeable_mask=tradeable_mask,
    )

    # Skip-degenerate / cost-adjusted series feed scalar, autocorrelation-
    # sensitive statistics (Sharpe, HAC SE), so they must use the SAME
    # trailing-row truncation as run_backtest()'s scalar path — otherwise
    # FactorPanelLoader's literal-0.0 trailing rows (finite, hence NOT
    # skipped by the degenerate-day guard) leak in as spurious all-zero
    # observations that understate the HAC SE and bias the mean to zero.
    # Reuse the shared helper so the two paths can't drift.
    trunc_preds, trunc_rets, _ = _truncate_trailing_forward_rows(
        predictions, returns, forward_period, tradeable_mask
    )
    cost_adjusted_series: list[float] = []
    if cost_bps > 0:
        cost_adjusted_series = top_k_returns_series_cost_adjusted(
            trunc_preds, trunc_rets, top_k=top_k, cost_bps=cost_bps
        )
    skip_degenerate_series = _top_k_returns_series(trunc_preds, trunc_rets, top_k)

    series = BacktestSeries(
        dates=[str(d) for d in dates],
        ic=ic_per_date,
        top_k_returns=top_k_rets,
        equity_curve=equity,
        random_baseline_sharpes=random_sharpes,
        top_k_returns_cost_adjusted=cost_adjusted_series,
        top_k_returns_skip_degenerate=skip_degenerate_series,
    )

    return result, series


__all__ = [
    "BacktestResult",
    "BacktestSeries",
    "compute_ic",
    "compute_ic_ir",
    "compute_ic_spearman",
    "compute_top_k_sharpe",
    "compute_top_k_sharpes",
    "compute_top_k_cumret",
    "top_k_returns_series_cost_adjusted",
    "random_baseline",
    "run_backtest",
    "run_backtest_with_series",
]
