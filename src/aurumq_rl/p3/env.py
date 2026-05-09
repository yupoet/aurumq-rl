"""P3 — ResidualPPOEnv: PPO env with logit-space residual.

Step semantics:
    obs[t]  = concat(feature_panel[t,:,:], p_baseline[t,:], rank_pct_baseline[t,:])
    action  = Δscore[j] for each j in universe, ∈ [-0.2, 0.2] (clip post tanh*0.2)
    reward  = mean(realized_excess for top_k stocks selected by score)
              where score = logit(p_baseline) + λ·Δ
    done    = t == end_idx - 1

Universe handling: out-of-universe stocks get -inf score so they are never
selected. NaN realized returns are masked to 0 to keep reward finite.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


logger = logging.getLogger(__name__)


def _logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


class ResidualPPOEnv(gym.Env):
    """One PPO env over a contiguous window of trading days."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        bundle,                                # P3Bundle
        start_date: date,
        end_date: date,
        top_k: int = 50,
        lambda_logit: float = 1.0,
        action_range: float = 0.2,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.bundle = bundle
        self.top_k = top_k
        self.lambda_logit = lambda_logit
        self.action_range = action_range
        self._rng = np.random.default_rng(seed)

        all_dates = bundle.trade_dates
        # Find slice indices [start, end]
        start_idx = next(i for i, d in enumerate(all_dates) if d >= start_date)
        end_idx = next(
            (i for i, d in enumerate(all_dates) if d > end_date),
            len(all_dates),
        )
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.t = start_idx
        logger.debug("ResidualPPOEnv window [%s, %s] = idx [%d, %d)",
                     start_date, end_date, start_idx, end_idx)

        self.n_stocks = bundle.n_stocks
        # obs = features (F) + p_baseline (1) + rank_pct (1) per stock
        F = bundle.n_features
        obs_dim_per_stock = F + 2
        self.obs_dim_per_stock = obs_dim_per_stock

        # SB3 default expects flat obs; PerStockEncoderPolicy will reshape.
        # We follow gpu_env convention: action shape (n_stocks,), obs shape (n_stocks, obs_dim_per_stock).
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.n_stocks, obs_dim_per_stock),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-self.action_range, high=self.action_range,
            shape=(self.n_stocks,),
            dtype=np.float32,
        )

    def _build_obs(self, t: int) -> np.ndarray:
        if t >= self.bundle.n_dates:
            return np.zeros((self.n_stocks, self.obs_dim_per_stock), dtype=np.float32)
        feats = self.bundle.feature_panel[t]                     # (S, F)
        p = self.bundle.p_baseline[t][:, None]                   # (S, 1)
        r = self.bundle.rank_pct_baseline[t][:, None]            # (S, 1)
        # Replace NaN with 0 (out-of-universe / missing)
        obs = np.concatenate([feats, p, r], axis=1).astype(np.float32, copy=False)
        np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        return obs

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.t = self.start_idx
        return self._build_obs(self.t), {}

    def step(self, action: np.ndarray):
        delta = np.clip(action, -self.action_range, self.action_range).astype(np.float32, copy=False)

        # 1. Build score on logit space, mask out-of-universe.
        p = self.bundle.p_baseline[self.t]                       # (S,)
        in_uni = self.bundle.in_universe[self.t]                 # (S,)
        score = _logit(np.where(np.isnan(p), 0.5, p))            # (S,) in real
        score = score + self.lambda_logit * delta                # apply residual
        score = np.where(in_uni, score, -np.inf)                 # block oou

        # 2. Top-k selection (k = min(top_k, n_in_universe))
        n_uni = int(in_uni.sum())
        k = min(self.top_k, n_uni) if n_uni > 0 else 0
        if k == 0:
            reward = 0.0
        else:
            topk_idx = np.argpartition(-score, k - 1)[:k]
            r = self.bundle.realized_pct_t_plus_1[self.t]        # (S,)
            m = self.bundle.market_pct_t_plus_1[self.t]
            r_vals = r[topk_idx]
            r_vals = np.where(np.isnan(r_vals), 0.0, r_vals)     # mask suspended
            m = 0.0 if (m is None or np.isnan(m)) else float(m)
            reward = float((r_vals - m).mean())

        # 3. Saturation diagnostic
        delta_abs = np.abs(delta)
        info = {
            "delta_abs_mean": float(delta_abs.mean()),
            "delta_abs_p95": float(np.percentile(delta_abs, 95)),
            "saturation_fraction": float((delta_abs >= 0.9 * self.action_range).mean()),
            "n_in_universe": n_uni,
        }

        self.t += 1
        terminated = self.t >= self.end_idx
        truncated = False
        next_obs = self._build_obs(self.t) if not terminated else self._build_obs(self.t - 1)
        return next_obs, reward, terminated, truncated, info


__all__ = ["ResidualPPOEnv"]
