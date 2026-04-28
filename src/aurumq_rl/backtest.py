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
    """Outcome of a single backtest run."""

    ic: float
    ic_ir: float
    top_k_sharpe: float
    top_k_cumret: float
    random_baseline: dict[str, float] = field(default_factory=dict)
    n_dates: int = 0
    n_stocks: int = 0
    top_k: int = 0

    def to_json(self, path: Path | str) -> None:
        Path(path).write_text(
            json.dumps(asdict(self), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def from_json(cls, path: Path | str) -> "BacktestResult":
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


def compute_top_k_sharpe(
    predictions: np.ndarray, returns: np.ndarray, top_k: int
) -> float:
    """Annualized Sharpe of an equal-weight top-K portfolio.

    Each date: pick the top_k stocks by prediction, equal-weight, take the
    realized return as the cross-sectional mean of those K stocks. Compute
    annualized Sharpe assuming 252 trading days.
    """
    if predictions.shape != returns.shape:
        raise ValueError("shape mismatch")

    portfolio_returns: list[float] = []
    for t in range(predictions.shape[0]):
        p, r = predictions[t], returns[t]
        mask = np.isfinite(p) & np.isfinite(r)
        if mask.sum() < top_k:
            continue
        idx = np.argsort(-p[mask])[:top_k]
        portfolio_returns.append(float(r[mask][idx].mean()))

    if len(portfolio_returns) < 2:
        return 0.0
    arr = np.asarray(portfolio_returns)
    std = arr.std(ddof=1)
    if std < 1e-12:
        return 0.0
    return float(arr.mean() / std * np.sqrt(252))


def compute_top_k_cumret(
    predictions: np.ndarray, returns: np.ndarray, top_k: int
) -> float:
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
) -> dict[str, float]:
    """Sharpe distribution of random top-K portfolios over the same dates."""
    rng = np.random.default_rng(seed)
    sharpes: list[float] = []
    for _ in range(n_simulations):
        preds = rng.normal(size=returns.shape)
        sharpes.append(compute_top_k_sharpe(preds, returns, top_k=top_k))

    arr = np.asarray(sharpes)
    return {
        "mean_sharpe": float(arr.mean()),
        "std_sharpe": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "p05_sharpe": float(np.percentile(arr, 5)),
        "p50_sharpe": float(np.percentile(arr, 50)),
        "p95_sharpe": float(np.percentile(arr, 95)),
    }


def run_backtest(
    predictions: np.ndarray,
    returns: np.ndarray,
    top_k: int = 30,
    n_random_simulations: int = 100,
    random_seed: int = 0,
) -> BacktestResult:
    """One-shot evaluation: IC + IR + top-K Sharpe + random baseline."""
    return BacktestResult(
        ic=compute_ic(predictions, returns),
        ic_ir=compute_ic_ir(predictions, returns),
        top_k_sharpe=compute_top_k_sharpe(predictions, returns, top_k),
        top_k_cumret=compute_top_k_cumret(predictions, returns, top_k),
        random_baseline=random_baseline(
            returns, top_k=top_k, n_simulations=n_random_simulations, seed=random_seed
        ),
        n_dates=predictions.shape[0],
        n_stocks=predictions.shape[1],
        top_k=top_k,
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
    def from_json(cls, path: Path | str) -> "BacktestSeries":
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
    returns: np.ndarray, top_k: int, n_simulations: int, seed: int
) -> list[float]:
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
) -> tuple["BacktestResult", "BacktestSeries"]:
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
        raise ValueError(
            f"dates length {len(dates)} != n_dates {predictions.shape[0]}"
        )

    # Canonical scalars — same semantics as run_backtest() so backtest.json is stable.
    result = run_backtest(
        predictions=predictions,
        returns=returns,
        top_k=top_k,
        n_random_simulations=n_random_simulations,
        random_seed=random_seed,
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
    "compute_top_k_cumret",
    "random_baseline",
    "run_backtest",
    "run_backtest_with_series",
]
