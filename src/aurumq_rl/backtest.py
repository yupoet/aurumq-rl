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
    """One-shot evaluation that also returns per-date / per-simulation series."""
    if predictions.shape != returns.shape:
        raise ValueError("shape mismatch")
    if len(dates) != predictions.shape[0]:
        raise ValueError(
            f"dates length {len(dates)} != n_dates {predictions.shape[0]}"
        )

    ic_per_date = _per_date_ics(predictions, returns)
    if len(ic_per_date) < predictions.shape[0]:
        # Pad to align with dates; degenerate days fill with 0.0
        ic_per_date = ic_per_date + [0.0] * (predictions.shape[0] - len(ic_per_date))
    top_k_rets = _per_date_top_k_returns(predictions, returns, top_k)

    equity = []
    cum = 1.0
    for ret in top_k_rets:
        cum *= 1.0 + ret
        equity.append(cum)

    random_sharpes = _random_sharpes(
        returns, top_k=top_k, n_simulations=n_random_simulations, seed=random_seed
    )

    arr = np.asarray(top_k_rets)
    sharpe = (
        float(arr.mean() / arr.std(ddof=1) * np.sqrt(252))
        if arr.size > 1 and arr.std(ddof=1) > 1e-12
        else 0.0
    )
    cumret = float(equity[-1] - 1.0) if equity else 0.0

    arr_ic = np.asarray(ic_per_date)
    ic_mean = float(arr_ic.mean()) if arr_ic.size else 0.0
    ic_ir = (
        float(arr_ic.mean() / arr_ic.std(ddof=1))
        if arr_ic.size > 1 and arr_ic.std(ddof=1) > 1e-12
        else 0.0
    )

    arr_rs = np.asarray(random_sharpes)
    baseline = {
        "mean_sharpe": float(arr_rs.mean()) if arr_rs.size else 0.0,
        "std_sharpe": float(arr_rs.std(ddof=1)) if arr_rs.size > 1 else 0.0,
        "p05_sharpe": float(np.percentile(arr_rs, 5)) if arr_rs.size else 0.0,
        "p50_sharpe": float(np.percentile(arr_rs, 50)) if arr_rs.size else 0.0,
        "p95_sharpe": float(np.percentile(arr_rs, 95)) if arr_rs.size else 0.0,
    }

    result = BacktestResult(
        ic=ic_mean,
        ic_ir=ic_ir,
        top_k_sharpe=sharpe,
        top_k_cumret=cumret,
        random_baseline=baseline,
        n_dates=predictions.shape[0],
        n_stocks=predictions.shape[1],
        top_k=top_k,
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
