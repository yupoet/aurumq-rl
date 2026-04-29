"""Shared pytest fixtures for the alpha101 + gtja191 factor library tests.

Two fixtures live here:

* :func:`synthetic_panel` — a session-scoped polars DataFrame that mirrors
  the schema produced by AurumQ's panel loader. The same data feeds both
  the runtime unit tests and the locked numerical reference data; see
  :mod:`tests.factors._synthetic` for the single source of truth.

* :func:`alpha101_reference` — a session-scoped polars DataFrame loaded
  from ``tests/factors/alpha101/data/alpha101_reference.parquet``,
  produced offline by ``scripts/reference_data/build_alpha101_reference.py``
  using the (MIT-licensed) STHSF/alpha101 reference implementation.

If the reference parquet has not been built yet, :func:`alpha101_reference`
**skips** rather than fails — that way new contributors can run the test
suite without first building the reference, and CI explicitly requires
the parquet to be committed.
"""

from __future__ import annotations

import pathlib

import polars as pl
import pytest

from tests.factors._synthetic import build_synthetic_panel

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = pathlib.Path(__file__).resolve().parent
ALPHA101_REFERENCE_PATH: pathlib.Path = _HERE / "alpha101" / "data" / "alpha101_reference.parquet"
GTJA191_REFERENCE_PATH: pathlib.Path = _HERE / "gtja191" / "data" / "gtja191_reference.parquet"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def synthetic_panel() -> pl.DataFrame:
    """Canonical 10-stock × 60-day synthetic factor panel.

    Session-scoped to avoid regenerating on every test. Seeded with 42 for
    determinism. See :mod:`tests.factors._synthetic` for the schema.
    """
    return build_synthetic_panel(seed=42, n_stocks=10, n_days=60)


@pytest.fixture(scope="session")
def alpha101_reference() -> pl.DataFrame:
    """Locked alpha101 reference values from STHSF/alpha101 (MIT).

    The parquet is produced offline by
    ``scripts/reference_data/build_alpha101_reference.py``. If absent,
    the test using this fixture is skipped — never failed — so that a
    fresh checkout without the artifact still has a green baseline.
    """
    if not ALPHA101_REFERENCE_PATH.exists():
        pytest.skip(
            "alpha101 reference parquet not built yet; run "
            "`python scripts/reference_data/build_alpha101_reference.py`"
        )
    return pl.read_parquet(ALPHA101_REFERENCE_PATH)


@pytest.fixture(scope="session")
def gtja191_reference() -> pl.DataFrame:
    """Locked gtja191 reference values from Daic115/alpha191 (formula only).

    The parquet is produced offline by
    ``scripts/reference_data/build_gtja191_reference.py``. If absent,
    the test using this fixture is skipped — never failed — so that a
    fresh checkout without the artifact still has a green baseline.

    Daic115/alpha191 has **no LICENSE** — we treat it strictly as a
    numerical formula reference and never vendor its code into the
    runtime. Only the resulting parquet (numerical values, not
    copyrightable) is committed.
    """
    if not GTJA191_REFERENCE_PATH.exists():
        pytest.skip(
            "gtja191 reference parquet not built yet; run "
            "`python scripts/reference_data/build_gtja191_reference.py`"
        )
    return pl.read_parquet(GTJA191_REFERENCE_PATH)
