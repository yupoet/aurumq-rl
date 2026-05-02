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

    def to_json(self, path: Path | str) -> None:
        Path(path).write_text(
            json.dumps(asdict(self), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def from_json(cls, path: Path | str) -> BacktestResult:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)


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


def _top_k_returns_series(
    predictions: np.ndarray, returns: np.ndarray, top_k: int
) -> list[float]:
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
) -> dict[str, float]:
    """Sharpe distribution of random top-K portfolios over the same dates.

    Phase 16 reports legacy / adjusted / non-overlap percentiles. The
    legacy fields are kept because existing dashboards consume them; for
    the production "vs random" comparison use the ``*_adjusted`` keys.
    """
    rng = np.random.default_rng(seed)
    legacy: list[float] = []
    adjusted: list[float] = []
    non_overlap: list[float] = []
    for _ in range(n_simulations):
        preds = rng.normal(size=returns.shape)
        d = compute_top_k_sharpes(
            preds, returns, top_k=top_k, forward_period=forward_period,
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


def run_backtest(
    predictions: np.ndarray,
    returns: np.ndarray,
    top_k: int = 30,
    n_random_simulations: int = 100,
    random_seed: int = 0,
    forward_period: int = 1,
) -> BacktestResult:
    """One-shot evaluation: IC + IR + top-K Sharpe trio + random baseline.

    Phase 16: when ``forward_period > 1`` we additionally truncate the
    last ``forward_period`` rows of both ``predictions`` and ``returns``
    so the trailing all-zero rows produced by
    :class:`FactorPanelLoader` (which has no future close to compute the
    forward log-return) do not drag down the Sharpe mean.
    """
    if forward_period > 1 and predictions.shape[0] > forward_period:
        predictions = predictions[: predictions.shape[0] - forward_period]
        returns = returns[: returns.shape[0] - forward_period]
    sharpes = compute_top_k_sharpes(
        predictions, returns, top_k=top_k, forward_period=forward_period,
    )
    return BacktestResult(
        ic=compute_ic(predictions, returns),
        ic_ir=compute_ic_ir(predictions, returns),
        top_k_sharpe=sharpes["adjusted"],  # primary metric for Phase 16
        top_k_cumret=compute_top_k_cumret(predictions, returns, top_k),
        random_baseline=random_baseline(
            returns, top_k=top_k, n_simulations=n_random_simulations,
            seed=random_seed, forward_period=forward_period,
        ),
        n_dates=predictions.shape[0],
        n_stocks=predictions.shape[1],
        top_k=top_k,
        forward_period=forward_period,
        top_k_sharpe_legacy=sharpes["legacy"],
        top_k_sharpe_adjusted=sharpes["adjusted"],
        top_k_sharpe_non_overlap=sharpes["non_overlap"],
    )


@dataclass
class BacktestSeries:
    """Per-date series produced alongside the BacktestResult."""

    dates: list[str]
    ic: list[float]
    top_k_returns: list[float]
    equity_curve: list[float]
    random_baseline_sharpes: list[float] = field(default_factory=list)

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


def _random_sharpes(returns: np.ndarray, top_k: int, n_simulations: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    out: list[float] = []
    for _ in range(n_simulations):
        preds = rng.normal(size=returns.shape)
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
) -> tuple[BacktestResult, BacktestSeries]:
    """One-shot evaluation that also returns per-date / per-simulation series.

    The scalar BacktestResult uses identical semantics to ``run_backtest`` —
    degenerate days are SKIPPED, not padded — so ``backtest.json`` is stable
    across both code paths. The per-date series, by contrast, must align to
    every entry in ``dates``; degenerate days are filled with 0.0 in the
    series so chart positions line up with the date axis.
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
    )

    # Per-date series for charts (aligned to dates; degenerate days -> 0.0).
    ic_per_date = _per_date_ics_aligned(predictions, returns)
    top_k_rets = _per_date_top_k_returns(predictions, returns, top_k)

    equity = []
    cum = 1.0
    for ret in top_k_rets:
        cum *= 1.0 + ret
        equity.append(cum)

    random_sharpes = _random_sharpes(
        returns, top_k=top_k, n_simulations=n_random_simulations, seed=random_seed
    )

    series = BacktestSeries(
        dates=[str(d) for d in dates],
        ic=ic_per_date,
        top_k_returns=top_k_rets,
        equity_curve=equity,
        random_baseline_sharpes=random_sharpes,
    )

    return result, series


__all__ = [
    "BacktestResult",
    "BacktestSeries",
    "compute_ic",
    "compute_ic_ir",
    "compute_top_k_sharpe",
    "compute_top_k_sharpes",
    "compute_top_k_cumret",
    "random_baseline",
    "run_backtest",
    "run_backtest_with_series",
]
