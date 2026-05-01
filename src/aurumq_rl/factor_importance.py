"""Factor-importance attribution: Integrated Gradients + permutation.

Both functions are pure and torch-only; they take a callable
``score_fn(panel) -> (B, n_stocks)`` so they don't depend on SB3
internals — the caller (eval_factor_importance.py) wraps a trained
policy into the right closure.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable

import numpy as np
import torch


def integrated_gradients(
    score_fn: Callable[[torch.Tensor], torch.Tensor],
    panel_batch: torch.Tensor,                  # (B, n_stocks, n_factors)
    n_alpha_steps: int = 50,
    baseline: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-factor saliency = average |IG| across batch and stocks.

    Returns a 1D tensor of length n_factors.
    """
    if baseline is None:
        baseline = torch.zeros_like(panel_batch)
    delta = (panel_batch - baseline).detach()
    saliency_sum = torch.zeros(panel_batch.shape[-1], device=panel_batch.device)
    for k in range(n_alpha_steps):
        alpha = (k + 0.5) / n_alpha_steps
        x = (baseline + alpha * (panel_batch - baseline)).detach().requires_grad_(True)
        scores = score_fn(x)               # (B, n_stocks)
        scores.sum().backward()
        # |grad| × Δ, averaged over batch and stocks
        attribution = (x.grad * delta).abs().mean(dim=(0, 1))
        saliency_sum += attribution
    return (saliency_sum / n_alpha_steps).detach()


def per_date_cross_section_shuffle(
    panel: torch.Tensor,                        # (T, S, F)
    cols: list[int],
    seed: int,
) -> torch.Tensor:
    """Return a copy of `panel` where the columns in `cols` are
    independently permuted across the stock axis on each date.
    Preserves time-series + per-date marginal; breaks cross-section
    ranking. See spec §7.
    """
    out = panel.clone()
    g = torch.Generator(device=panel.device)
    g.manual_seed(seed)
    T, S, _ = panel.shape
    for t in range(T):
        perm = torch.randperm(S, generator=g, device=panel.device)
        out[t, :, cols] = panel[t, perm][:, cols]
    return out


def permutation_importance(
    score_fn: Callable[[torch.Tensor], torch.Tensor],
    val_panel: torch.Tensor,                    # (T, S, F) cuda fp32
    val_returns: torch.Tensor,                  # (T, S)    cuda fp32
    factor_names: list[str],
    forward_period: int = 10,
    top_k: int = 30,
    n_seeds: int = 5,
    base_seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Per-prefix ΔIC + ΔSharpe via per-date cross-section shuffle.

    Returns a dict keyed by prefix (e.g. "alpha", "gtja", "mfp", ...)
    with statistics over n_seeds shuffles.
    """
    cols_by_prefix: dict[str, list[int]] = defaultdict(list)
    for i, name in enumerate(factor_names):
        prefix = name.split("_", 1)[0] if "_" in name else name
        cols_by_prefix[prefix].append(i)

    baseline_metrics = _eval_top_k_metrics(
        score_fn, val_panel, val_returns, forward_period, top_k,
    )
    out: dict[str, dict[str, float]] = {}
    for prefix, cols in cols_by_prefix.items():
        ic_drops, sharpe_drops = [], []
        for seed in range(base_seed, base_seed + n_seeds):
            shuffled = per_date_cross_section_shuffle(val_panel, cols, seed)
            m = _eval_top_k_metrics(
                score_fn, shuffled, val_returns, forward_period, top_k,
            )
            ic_drops.append(baseline_metrics["ic"] - m["ic"])
            sharpe_drops.append(baseline_metrics["sharpe"] - m["sharpe"])
        out[prefix] = {
            "n_factors": len(cols),
            "n_seeds": n_seeds,
            "ic_drop_mean": float(np.mean(ic_drops)),
            "ic_drop_std": float(np.std(ic_drops, ddof=1) if len(ic_drops) > 1 else 0.0),
            "sharpe_drop_mean": float(np.mean(sharpe_drops)),
            "sharpe_drop_std": float(np.std(sharpe_drops, ddof=1) if len(sharpe_drops) > 1 else 0.0),
        }
    return out


def _eval_top_k_metrics(
    score_fn: Callable[[torch.Tensor], torch.Tensor],
    panel: torch.Tensor,
    returns: torch.Tensor,
    forward_period: int,
    top_k: int,
) -> dict[str, float]:
    """Score every date independently, build top-K portfolio, return IC + annualised Sharpe."""
    T = panel.shape[0]
    valid_T = T - forward_period
    if valid_T <= 0:
        return {"ic": 0.0, "sharpe": 0.0}
    portfolio_returns = []
    ics = []
    with torch.no_grad():
        for t in range(valid_T):
            obs = panel[t : t + 1]                         # (1, S, F)
            scores = score_fn(obs)[0]                       # (S,)
            top_idx = torch.topk(scores, top_k).indices
            r = returns[t + forward_period][top_idx].mean()
            portfolio_returns.append(r.item())
            # IC: corr(scores, future returns)
            f = returns[t + forward_period]
            mask = torch.isfinite(scores) & torch.isfinite(f)
            if mask.sum() < 2 or scores[mask].std().item() < 1e-12:
                continue
            c = torch.corrcoef(torch.stack([scores[mask], f[mask]]))[0, 1].item()
            if np.isfinite(c):
                ics.append(c)
    if len(portfolio_returns) < 2:
        return {"ic": 0.0, "sharpe": 0.0}
    arr = np.asarray(portfolio_returns)
    s = arr.std(ddof=1)
    sharpe = float(arr.mean() / s * np.sqrt(252)) if s > 1e-12 else 0.0
    ic = float(np.mean(ics)) if ics else 0.0
    return {"ic": ic, "sharpe": sharpe}


__all__ = ["integrated_gradients", "permutation_importance", "per_date_cross_section_shuffle"]
