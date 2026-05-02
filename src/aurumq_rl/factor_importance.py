"""Factor-importance attribution: Integrated Gradients + permutation.

Both functions are pure and torch-only; they take a callable
``score_fn(panel) -> (B, n_stocks)`` so they don't depend on SB3
internals — the caller (eval_factor_importance.py) wraps a trained
policy into the right closure.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

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
    """Score every date independently, build top-K portfolio, return IC + Sharpe metrics.

    Phase 16 corrections vs Phase 15:

    1. ``returns[t]`` is used (NOT ``returns[t + forward_period]``).
       :class:`FactorPanelLoader` already encodes ``return_array[t] =
       log(close[t+fp]/close[t])``, so the t-th row IS the forward return
       realised by selecting at t. The earlier ``[t + fp]`` indexing
       double-shifted the eval into ``t+fp .. t+2fp`` returns.
    2. Three Sharpe values are reported:

       * ``sharpe_legacy``: ``sqrt(252)`` annualisation. Inflated when
         ``forward_period > 1`` because the per-day returns are overlapping
         ``forward_period``-day windows. Kept for backward comparison only.
       * ``sharpe_adjusted``: ``sqrt(252 / forward_period)`` annualisation —
         the correct value for a stream of N-day forward returns sampled
         daily. This is Phase 16's primary metric.
       * ``sharpe_non_overlap``: take every ``forward_period``-th row only
         (0, fp, 2fp, …) and annualise with ``sqrt(252 / forward_period)``.
         Lower-variance estimator that is independent across rows. Used as
         a sanity check; if it diverges sharply from ``sharpe_adjusted`` the
         daily series has structure beyond the rolling-window overlap.

    Backwards-compat: dict still contains ``sharpe`` set to
    ``sharpe_adjusted`` so callers that already index ``["sharpe"]`` get
    the corrected scale by default.
    """
    T = panel.shape[0]
    valid_T = T - forward_period
    if valid_T <= 0:
        return {
            "ic": 0.0,
            "sharpe": 0.0,
            "sharpe_legacy": 0.0,
            "sharpe_adjusted": 0.0,
            "sharpe_non_overlap": 0.0,
        }
    portfolio_returns = []
    ics = []
    with torch.no_grad():
        for t in range(valid_T):
            obs = panel[t : t + 1]                         # (1, S, F)
            scores = score_fn(obs)[0]                       # (S,)
            top_idx = torch.topk(scores, top_k).indices
            r = returns[t][top_idx].mean()
            portfolio_returns.append(r.item())
            # IC: corr(scores, future returns at the SAME t — the t-th row
            # is already the forward-window return realised by t→t+fp).
            f = returns[t]
            mask = torch.isfinite(scores) & torch.isfinite(f)
            if mask.sum() < 2 or scores[mask].std().item() < 1e-12:
                continue
            c = torch.corrcoef(torch.stack([scores[mask], f[mask]]))[0, 1].item()
            if np.isfinite(c):
                ics.append(c)
    if len(portfolio_returns) < 2:
        return {
            "ic": 0.0,
            "sharpe": 0.0,
            "sharpe_legacy": 0.0,
            "sharpe_adjusted": 0.0,
            "sharpe_non_overlap": 0.0,
        }
    arr = np.asarray(portfolio_returns)
    s = arr.std(ddof=1)
    if s > 1e-12:
        sharpe_legacy = float(arr.mean() / s * np.sqrt(252))
        sharpe_adjusted = float(arr.mean() / s * np.sqrt(252 / forward_period))
    else:
        sharpe_legacy = 0.0
        sharpe_adjusted = 0.0
    # non-overlap subsample: every forward_period-th observation
    sub = arr[::forward_period] if forward_period > 1 else arr
    if len(sub) > 1 and sub.std(ddof=1) > 1e-12:
        sharpe_non_overlap = float(sub.mean() / sub.std(ddof=1) * np.sqrt(252 / forward_period))
    else:
        sharpe_non_overlap = 0.0
    ic = float(np.mean(ics)) if ics else 0.0
    return {
        "ic": ic,
        # Phase 16: primary metric is the adjusted Sharpe.
        "sharpe": sharpe_adjusted,
        "sharpe_legacy": sharpe_legacy,
        "sharpe_adjusted": sharpe_adjusted,
        "sharpe_non_overlap": sharpe_non_overlap,
    }


__all__ = ["integrated_gradients", "permutation_importance", "per_date_cross_section_shuffle"]
