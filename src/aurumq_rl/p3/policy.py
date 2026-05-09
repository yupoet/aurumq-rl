"""P3 — Residual actor head + zero-init self-test.

Subclasses PerStockEncoderPolicy. Replaces action_net with a zero-init
Linear so Δ_init ≈ 0 (修正 #6 from ALGORITHM_SPEC v2).

Action range: ±0.2 in logit space (修正 #5). SB3's Box action_space
clips automatically. Sampled actions = mean (zero) + Normal(0, std=0.3
at init), most samples will be clipped to [-0.2, 0.2].

NOTE: This still relies on ResidualPPOEnv to interpret action as
logit-space Δ and convert via `score = logit(p_baseline) + λ·Δ` before
selecting top_k.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn

from aurumq_rl.policy import PerStockEncoderPolicy


logger = logging.getLogger(__name__)


# ALGORITHM_SPEC v2 §5
ACTION_RANGE = 0.2                # ± clip in logit space
LOG_STD_INIT = math.log(0.3)      # ln(0.3) ≈ -1.20397
LOG_STD_FINAL = math.log(0.1)     # ln(0.1) ≈ -2.30259
LOG_STD_FREEZE_STEPS = 100_000
LOG_STD_ANNEAL_STEPS = 200_000


class ResidualPerStockPolicy(PerStockEncoderPolicy):
    """PerStockEncoderPolicy with zero-init action head for residual learning."""

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        # Re-init action_net to zero (修正 #6)
        with torch.no_grad():
            nn.init.zeros_(self.action_net.weight)
            if self.action_net.bias is not None:
                nn.init.zeros_(self.action_net.bias)
        # log_std init to ln(0.3) (修正 #2)
        with torch.no_grad():
            if hasattr(self, "log_std"):
                self.log_std.data.fill_(LOG_STD_INIT)
        logger.info(
            "ResidualPerStockPolicy built: action_range=±%.2f, log_std_init=%.4f (std=%.3f)",
            ACTION_RANGE, LOG_STD_INIT, math.exp(LOG_STD_INIT),
        )

    def self_test_zero_init(self, sample_obs) -> None:
        """Verify Δ output is ~0 at init. Call before training."""
        with torch.no_grad():
            actions, _, _ = self(sample_obs, deterministic=True)
            max_abs = actions.abs().max().item()
        assert max_abs < 1e-3, (
            f"Δ should be ~0 at init (zero-weight action_net), got max|Δ|={max_abs:.6f}"
        )
        logger.info("Δ zero-init self-test PASSED: max|Δ|=%.2e", max_abs)


__all__ = [
    "ResidualPerStockPolicy",
    "ACTION_RANGE",
    "LOG_STD_INIT",
    "LOG_STD_FINAL",
    "LOG_STD_FREEZE_STEPS",
    "LOG_STD_ANNEAL_STEPS",
]
