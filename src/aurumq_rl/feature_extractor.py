"""Phase 21 V2 feature extractors.

Two independent modules:

* :class:`PerStockEncoderV2` — applies a shared MLP to each stock row
  individually and LayerNorms the per-stock embedding. Strictly per-stock
  input; the schema lock at training startup forbids mkt_/index_/regime_/
  global_ columns from reaching it.
* :class:`RegimeEncoder` — small MLP from the (R,) date-level regime
  feature vector to (R',). Output is broadcast to every stock at the head
  layer in :class:`PerStockEncoderPolicyV2`.

The earlier V1 :class:`PerStockExtractor` (cross-section centering + dual
pooling) is removed. Cross-section centering is no longer needed because
the regime path supplies the date-level signal explicitly; dual pooling
moves into the critic via :func:`masked_mean` over the value-token MLP.
"""
from __future__ import annotations

import torch
from torch import nn


class PerStockEncoderV2(nn.Module):
    """Shared per-stock MLP followed by LayerNorm. Input is per-stock ONLY.

    Parameters
    ----------
    n_factors:
        Number of per-stock factor channels (F_stock).
    hidden:
        Hidden layer widths. Default (128, 64).
    out_dim:
        Output embedding width (D). Default 32.
    """

    def __init__(
        self,
        n_factors: int,
        hidden: tuple[int, ...] = (128, 64),
        out_dim: int = 32,
    ) -> None:
        super().__init__()
        self.n_factors = n_factors
        self.out_dim = out_dim
        layers: list[nn.Module] = []
        prev = n_factors
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.mlp = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, stock_x: torch.Tensor) -> torch.Tensor:
        # stock_x: (B, S, F_stock)
        b, s, f = stock_x.shape
        flat = stock_x.reshape(b * s, f)
        return self.norm(self.mlp(flat).reshape(b, s, self.out_dim))


class RegimeEncoder(nn.Module):
    """LayerNorm + (R → hidden) + SiLU + (hidden → R') + LayerNorm.

    R defaults to 8 (the v0 regime feature count). R' defaults to 16.
    """

    def __init__(
        self,
        regime_dim: int = 8,
        hidden: int = 64,
        out_dim: int = 16,
    ) -> None:
        super().__init__()
        self.regime_dim = regime_dim
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.LayerNorm(regime_dim),
            nn.Linear(regime_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, regime_x: torch.Tensor) -> torch.Tensor:
        # regime_x: (B, R) → (B, R')
        return self.net(regime_x)


def masked_mean(
    x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Cross-stock masked mean.

    Parameters
    ----------
    x:
        (B, S, H) per-stock token tensor.
    mask:
        (B, S) where 1 marks a valid stock. Floating-point dtype acceptable;
        will be cast to ``x.dtype``.
    eps:
        Denominator floor. A row with zero valid stocks returns zeros.
    """
    m = mask.to(dtype=x.dtype).unsqueeze(-1)  # (B, S, 1)
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(eps)
