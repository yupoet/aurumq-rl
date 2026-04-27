#!/usr/bin/env python3
"""End-to-end quickstart: load synthetic panel, run a few env steps, print stats.

This script runs in ~30 seconds on CPU. It exercises:
  1. FactorPanelLoader.build_synthetic — pure-Python panel
  2. discover_factor_columns — prefix-based factor discovery
  3. StockPickingEnv (if gymnasium is installed) OR a manual reward loop fallback
  4. Basic descriptive statistics

Usage
-----
    python examples/quickstart.py

No GPU, no PyTorch, no real data needed.
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

import numpy as np

# Ensure src/ is importable when running from a fresh checkout
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from aurumq_rl import FactorPanelLoader  # noqa: E402
from aurumq_rl.data_loader import FACTOR_COL_PREFIXES  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_DATES: int = 60
N_STOCKS: int = 50
N_FACTORS: int = 12
TOP_K: int = 10
FORWARD_PERIOD: int = 5
N_STEPS: int = 20


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def step_1_build_panel() -> None:
    print("=" * 60)
    print("Step 1: build a synthetic factor panel")
    print("=" * 60)

    panel = FactorPanelLoader.build_synthetic(
        n_dates=N_DATES,
        n_stocks=N_STOCKS,
        n_factors=N_FACTORS,
        forward_period=FORWARD_PERIOD,
        seed=42,
    )
    print(f"  factor_array shape : {panel.factor_array.shape}")
    print(f"  return_array shape : {panel.return_array.shape}")
    print(f"  dates              : {panel.dates[0]} .. {panel.dates[-1]}")
    print(f"  stock codes (sample): {panel.stock_codes[:3]}...")
    print(f"  factor names (sample): {panel.factor_names[:3]}...")
    return panel


def step_2_show_factor_prefixes() -> None:
    print()
    print("=" * 60)
    print("Step 2: factor column prefix conventions")
    print("=" * 60)
    for p in FACTOR_COL_PREFIXES:
        print(f"  {p:<10}")


def step_3_run_env_or_fallback(panel) -> dict[str, float]:
    print()
    print("=" * 60)
    print("Step 3: run a few simulated steps")
    print("=" * 60)

    rng = np.random.default_rng(0)
    rewards: list[float] = []

    try:
        import gymnasium  # noqa: F401  -- presence check
    except ImportError:
        gymnasium_available = False
    else:
        gymnasium_available = True

    if gymnasium_available:
        from aurumq_rl.env import StockPickingConfig, StockPickingEnv

        config = StockPickingConfig(
            start_date=datetime.date(2022, 1, 1),
            end_date=datetime.date(2022, 12, 31),
            n_factors=N_FACTORS,
            top_k=TOP_K,
            forward_period=FORWARD_PERIOD,
            cost_bps=20.0,
            turnover_penalty=0.0,
            respect_dynamic_price_limits=False,
        )
        env = StockPickingEnv(
            config=config,
            factor_panel=panel.factor_array,
            return_panel=panel.return_array,
            pct_change_panel=panel.pct_change_array,
            is_st_panel=panel.is_st_array,
            is_suspended_panel=panel.is_suspended_array,
            days_since_ipo_panel=panel.days_since_ipo_array,
        )
        obs, _info = env.reset(seed=0)
        print(f"  using StockPickingEnv (obs dim = {obs.shape[0]})")
        for step in range(N_STEPS):
            action = rng.uniform(0.0, 1.0, size=N_STOCKS).astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(float(reward))
            if terminated or truncated:
                break
    else:
        # Fallback: replicate the env's reward formula manually
        print("  gymnasium not installed — using simplified reward loop fallback")
        prev_holdings = np.zeros(N_STOCKS, dtype=np.float32)
        for t in range(N_STEPS):
            scores = rng.uniform(0.0, 1.0, size=N_STOCKS)
            top_idx = np.argsort(scores)[-TOP_K:]
            holdings = np.zeros(N_STOCKS, dtype=np.float32)
            holdings[top_idx] = 1.0 / TOP_K
            forward = panel.return_array[t]
            ret = float(np.dot(holdings, forward))
            turnover = float(np.sum(np.abs(holdings - prev_holdings)))
            reward = ret - 20.0 / 10_000.0 if turnover > 0 else ret
            rewards.append(reward)
            prev_holdings = holdings

    print(f"  ran {len(rewards)} steps")
    print(f"  reward mean : {np.mean(rewards):+.5f}")
    print(f"  reward std  : {np.std(rewards):.5f}")
    print(f"  reward min  : {np.min(rewards):+.5f}")
    print(f"  reward max  : {np.max(rewards):+.5f}")

    return {
        "mean": float(np.mean(rewards)),
        "std": float(np.std(rewards)),
        "min": float(np.min(rewards)),
        "max": float(np.max(rewards)),
        "n_steps": len(rewards),
    }


def step_4_summary(panel, stats: dict[str, float]) -> None:
    print()
    print("=" * 60)
    print("Step 4: summary")
    print("=" * 60)
    print(f"  panel : {N_DATES} dates × {N_STOCKS} stocks × {N_FACTORS} factors")
    print(
        f"  steps : {stats['n_steps']} | mean reward {stats['mean']:+.5f} "
        f"(σ={stats['std']:.5f})"
    )
    print(f"  factor std: {float(panel.factor_array.std()):.4f}")
    print()
    print("Next steps:")
    print("  - try `scripts/train.py --smoke-test` for a CPU-only end-to-end run")
    print("  - generate richer demo data with `scripts/generate_synthetic.py`")
    print("  - read docs/ARCHITECTURE.md, docs/TRAINING.md, docs/INFERENCE.md")


def main() -> int:
    panel = step_1_build_panel()
    step_2_show_factor_prefixes()
    stats = step_3_run_env_or_fallback(panel)
    step_4_summary(panel, stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
