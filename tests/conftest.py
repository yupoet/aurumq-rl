"""Shared pytest fixtures for AurumQ-RL test suite."""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

# Make scripts/ importable as a top-level package for tests
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Absolute path to the project root."""
    return _PROJECT_ROOT


@pytest.fixture(scope="session")
def tiny_panel_df() -> pl.DataFrame:
    """A very small synthetic panel used by fast unit tests.

    20 stocks × 30 dates with 3 factor prefix groups (alpha/mf/fund).
    """
    from generate_synthetic import build_synthetic_dataframe

    return build_synthetic_dataframe(n_stocks=20, n_dates=30, seed=7)


@pytest.fixture(scope="session")
def tiny_panel_parquet(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write tiny_panel_df to a Parquet so loaders can ingest it."""
    from generate_synthetic import write_synthetic_parquet

    out_dir = tmp_path_factory.mktemp("tiny_panel")
    out_path = out_dir / "tiny.parquet"
    write_synthetic_parquet(out_path=out_path, n_stocks=20, n_dates=30, seed=7)
    return out_path


@pytest.fixture(scope="session")
def demo_parquet(project_root: Path) -> Path:
    """Path to the committed synthetic demo. Skips if missing."""
    p = project_root / "data" / "synthetic_demo.parquet"
    if not p.exists():
        pytest.skip(
            "data/synthetic_demo.parquet not found — run "
            "`python scripts/generate_synthetic.py` first."
        )
    return p


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic RNG used by reward/inference tests."""
    return np.random.default_rng(0)


@pytest.fixture
def date_range_short() -> tuple[datetime.date, datetime.date]:
    """A short date range covering most of the tiny panel."""
    return datetime.date(2022, 1, 3), datetime.date(2022, 2, 28)
