"""AurumQ-RL — Reinforcement learning stock selection for China A-share market.

Public API
----------
* :class:`StockPickingEnv` — Gymnasium environment for daily stock selection.
* :class:`PortfolioWeightEnv` — Continuous-weight portfolio optimization environment.
* :class:`FactorPanelLoader` — Loads factor panels from Parquet for training/inference.
* :class:`RlAgentInference` — ONNX-based CPU inference engine.
* :class:`StockBoard` / :func:`identify_board` — A-share board identification.

Scope
-----
This package is **RL only**. Factor computation is OUT OF SCOPE — the project
assumes input Parquet already contains pre-computed factor columns matching
the prefix convention defined in :mod:`aurumq_rl.data_loader`.

To populate the input Parquet from a PostgreSQL data warehouse, use
``scripts/export_factor_panel.py``. To compute factors from raw OHLCV +
trading data, use your own pipeline (or the parent AurumQ project).

Optional dependencies
---------------------
* **core** (default): inference + data loading, no PyTorch
* **train**: adds PyTorch / SB3 / Gymnasium for training
* **factors**: adds PG / pandas for ``scripts/export_factor_panel.py``

Importing this package never pulls in PyTorch. Each submodule degrades
gracefully when its optional deps are missing.

Example
-------
>>> from aurumq_rl import RlAgentInference
>>> agent = RlAgentInference("models/ppo_v1/")
>>> action = agent.predict(obs)
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = [
    "__version__",
    # Inference (always available)
    "RlAgentInference",
    "RlAgentMetadata",
    # Data
    "FactorPanelLoader",
    "FactorPanel",
    # Constants
    "StockBoard",
    "identify_board",
    "is_at_limit_up",
    "is_at_limit_down",
]

# Always-available imports (no PyTorch required)
from aurumq_rl.inference import RlAgentInference, RlAgentMetadata
from aurumq_rl.data_loader import FactorPanel, FactorPanelLoader
from aurumq_rl.price_limits import (
    StockBoard,
    identify_board,
    is_at_limit_up,
    is_at_limit_down,
)


def get_version() -> str:
    """Return the package version."""
    return __version__
