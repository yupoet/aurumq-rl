"""Production-length smoke checks for GTJA long-lookback factors.

The standard synthetic fixture is only 60 days, so several factors were allowed
to return all-null under ``_LONG_LOOKBACK`` test exceptions. This test uses a
longer deterministic panel to ensure those factors actually produce finite
values once enough history exists.
"""

from __future__ import annotations

import polars as pl
import pytest

from aurumq_rl.factors.gtja191.batch_101_120 import gtja_101, gtja_119
from aurumq_rl.factors.gtja191.batch_121_140 import gtja_124
from aurumq_rl.factors.gtja191.batch_141_160 import gtja_141
from aurumq_rl.factors.gtja191.batch_161_180 import gtja_170, gtja_176, gtja_179
from tests.factors._synthetic import build_synthetic_panel

_FACTORS = [
    ("gtja_101", gtja_101),
    ("gtja_119", gtja_119),
    ("gtja_124", gtja_124),
    ("gtja_141", gtja_141),
    ("gtja_170", gtja_170),
    ("gtja_176", gtja_176),
    ("gtja_179", gtja_179),
]


@pytest.fixture(scope="module")
def long_panel() -> pl.DataFrame:
    """260 business days is enough for every factor in this smoke set."""
    return build_synthetic_panel(seed=42, n_stocks=10, n_days=260)


@pytest.mark.parametrize("name,fn", _FACTORS)
def test_long_lookback_factor_has_finite_values(name, fn, long_panel: pl.DataFrame) -> None:
    out = fn(long_panel)
    assert out.dtype == pl.Float64
    assert len(out) == long_panel.height
    assert int(out.is_finite().sum() or 0) > 0, f"{name} produced no finite values"
