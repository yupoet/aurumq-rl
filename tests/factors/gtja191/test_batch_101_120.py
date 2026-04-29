"""Tests for gtja191.batch_101_120 (factors 101..120, 20 total)."""

from __future__ import annotations

import polars as pl
import pytest

from aurumq_rl.factors.gtja191.batch_101_120 import (
    gtja_101,
    gtja_102,
    gtja_103,
    gtja_104,
    gtja_105,
    gtja_106,
    gtja_107,
    gtja_108,
    gtja_109,
    gtja_110,
    gtja_111,
    gtja_112,
    gtja_113,
    gtja_114,
    gtja_115,
    gtja_116,
    gtja_117,
    gtja_118,
    gtja_119,
    gtja_120,
)
from tests.factors.gtja191._parity import parity_check

# Factors that compute 0 non-null on the 60-day synthetic panel because
# they require lookbacks > 60 days (e.g. ts_rank(60), corr(., 200), etc.).
# For these the steady-state test only checks "no exception, dtype OK,
# length OK". On real production panels (4000+ days) they fill in.
_LONG_LOOKBACK = {"gtja_101", "gtja_108", "gtja_115", "gtja_119"}


_FACTORS = [
    ("gtja_101", gtja_101),
    ("gtja_102", gtja_102),
    ("gtja_103", gtja_103),
    ("gtja_104", gtja_104),
    ("gtja_105", gtja_105),
    ("gtja_106", gtja_106),
    ("gtja_107", gtja_107),
    ("gtja_108", gtja_108),
    ("gtja_109", gtja_109),
    ("gtja_110", gtja_110),
    ("gtja_111", gtja_111),
    ("gtja_112", gtja_112),
    ("gtja_113", gtja_113),
    ("gtja_114", gtja_114),
    ("gtja_115", gtja_115),
    ("gtja_116", gtja_116),
    ("gtja_117", gtja_117),
    ("gtja_118", gtja_118),
    ("gtja_119", gtja_119),
    ("gtja_120", gtja_120),
]


# Status overrides — only listed if status != "match".
_STATUS = {
    "gtja_119": "drift",
}


@pytest.mark.parametrize("name,fn", _FACTORS)
def test_dtype(name, fn, synthetic_panel):
    assert fn(synthetic_panel).dtype == pl.Float64


@pytest.mark.parametrize("name,fn", _FACTORS)
def test_length(name, fn, synthetic_panel):
    assert len(fn(synthetic_panel)) == synthetic_panel.height


@pytest.mark.parametrize("name,fn", _FACTORS)
def test_steady_state(name, fn, synthetic_panel):
    s = fn(synthetic_panel)
    if name in _LONG_LOOKBACK:
        # Synthetic panel only has 60 days; these factors need >60.
        # Verify dtype/length pass and accept 0 non-null. Production
        # panels fill these in.
        return
    assert s.is_not_null().sum() > 0


@pytest.mark.parametrize("name,fn", _FACTORS)
def test_parity(name, fn, synthetic_panel, gtja191_reference):
    parity_check(
        synthetic_panel,
        gtja191_reference,
        name,
        fn,
        status=_STATUS.get(name, "match"),
    )
