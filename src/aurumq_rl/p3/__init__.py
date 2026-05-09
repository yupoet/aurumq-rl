"""P3 — Hybrid PPO Residual training (4070 side).

Architecture: PPO learns Δscore on top of P2 LightGBM ensemble baseline.
    score_raw = logit(p_baseline) + λ · Δ
    p_final   = isotonic_h1(score_raw)

See handoffs/2026-05-09-p3-4070-training/ALGORITHM_SPEC.md (v2) for the
9-correction audit by paris.
"""

from .data import P3Bundle, load_bundle
from .env import ResidualPPOEnv
from .policy import ResidualPerStockPolicy

__all__ = [
    "P3Bundle",
    "load_bundle",
    "ResidualPPOEnv",
    "ResidualPerStockPolicy",
]
