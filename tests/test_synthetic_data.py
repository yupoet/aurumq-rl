"""Tests for the synthetic data generator + the committed demo Parquet."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

from aurumq_rl.data_loader import FACTOR_COL_PREFIXES, REQUIRED_COLUMNS, OPTIONAL_COLUMNS

# Importing from the scripts/ directory works because conftest adds it to sys.path
from generate_synthetic import (
    DEFAULT_OUT,
    FACTOR_GROUPS,
    build_synthetic_dataframe,
    write_synthetic_parquet,
)


# ---------------------------------------------------------------------------
# Direct API
# ---------------------------------------------------------------------------


def test_generator_factor_groups_at_least_six() -> None:
    """Spec requires >= 6 distinct factor prefix groups."""
    assert len(FACTOR_GROUPS) >= 6


def test_generator_only_uses_recognised_prefixes() -> None:
    for prefix in FACTOR_GROUPS:
        assert prefix in FACTOR_COL_PREFIXES, (
            f"Prefix {prefix!r} not in canonical FACTOR_COL_PREFIXES"
        )


def test_build_dataframe_has_all_required_columns() -> None:
    df = build_synthetic_dataframe(n_stocks=10, n_dates=5, seed=1)
    for col in REQUIRED_COLUMNS:
        assert col in df.columns


def test_build_dataframe_has_optional_columns() -> None:
    df = build_synthetic_dataframe(n_stocks=10, n_dates=5, seed=1)
    for col in OPTIONAL_COLUMNS:
        assert col in df.columns


def test_build_dataframe_synthetic_codes() -> None:
    df = build_synthetic_dataframe(n_stocks=5, n_dates=3, seed=1)
    codes = df["ts_code"].unique().sort().to_list()
    for c in codes:
        assert c.startswith("SYN_")
        # Demo codes must NOT look like real Tushare codes
        assert not c.endswith((".SH", ".SZ", ".BJ"))


def test_build_dataframe_pct_chg_decimal_form() -> None:
    df = build_synthetic_dataframe(n_stocks=20, n_dates=20, seed=2)
    pct = df["pct_chg"].to_numpy()
    assert pct.min() > -0.11
    assert pct.max() < 0.11


def test_build_dataframe_invalid_args() -> None:
    with pytest.raises(ValueError):
        build_synthetic_dataframe(n_stocks=0, n_dates=10)
    with pytest.raises(ValueError):
        build_synthetic_dataframe(n_stocks=10, n_dates=0)


def test_build_dataframe_factor_groups_present() -> None:
    df = build_synthetic_dataframe(n_stocks=10, n_dates=5, seed=1)
    seen_prefixes = {
        prefix for prefix in FACTOR_GROUPS
        if any(c.startswith(prefix) for c in df.columns)
    }
    assert seen_prefixes == set(FACTOR_GROUPS)


# ---------------------------------------------------------------------------
# write_synthetic_parquet
# ---------------------------------------------------------------------------


def test_write_synthetic_parquet_smaller(tmp_path: Path) -> None:
    out = tmp_path / "tiny.parquet"
    written = write_synthetic_parquet(out, n_stocks=20, n_dates=30, seed=3)
    assert written == out
    assert out.exists()
    # The tiny file should be under 1 MB
    size_mb = out.stat().st_size / (1024 * 1024)
    assert size_mb < 1.0


def test_write_synthetic_parquet_loadable(tmp_path: Path) -> None:
    out = tmp_path / "rt.parquet"
    write_synthetic_parquet(out, n_stocks=15, n_dates=20, seed=4)
    df = pl.read_parquet(str(out))
    assert len(df) == 15 * 20
    assert df["trade_date"].dtype == pl.Date


# ---------------------------------------------------------------------------
# CLI invocation
# ---------------------------------------------------------------------------


def test_cli_creates_parquet(tmp_path: Path, project_root: Path) -> None:
    out = tmp_path / "cli_out.parquet"
    cmd = [
        sys.executable,
        str(project_root / "scripts" / "generate_synthetic.py"),
        "--n-stocks", "10",
        "--n-dates", "10",
        "--seed", "9",
        "--out", str(out),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
    assert res.returncode == 0, f"CLI failed: {res.stderr!r}"
    assert out.exists()


# ---------------------------------------------------------------------------
# Committed demo file (data/synthetic_demo.parquet)
# ---------------------------------------------------------------------------


def test_demo_parquet_under_size_budget(demo_parquet: Path) -> None:
    size_mb = demo_parquet.stat().st_size / (1024 * 1024)
    assert size_mb <= 10.0, f"demo Parquet exceeds 10 MB budget (got {size_mb:.2f} MB)"


def test_demo_parquet_schema_complete(demo_parquet: Path) -> None:
    df = pl.read_parquet(str(demo_parquet))
    for col in REQUIRED_COLUMNS:
        assert col in df.columns, f"missing required column {col!r}"
    for col in OPTIONAL_COLUMNS:
        assert col in df.columns, f"missing optional column {col!r}"


def test_demo_parquet_factor_groups(demo_parquet: Path) -> None:
    df = pl.read_parquet(str(demo_parquet))
    prefixes_seen = {
        prefix for prefix in FACTOR_COL_PREFIXES
        if any(c.startswith(prefix) for c in df.columns)
    }
    assert len(prefixes_seen) >= 6, (
        f"Demo parquet must include at least 6 prefix groups, got {prefixes_seen}"
    )
