"""Tests for gtja191.batch_141_160 (factors 141..160, 20 total).

Special factors: gtja_143 (stub, all-null), gtja_147 (qlib-dep variant),
gtja_149 (benchmark — CS-mean proxy), gtja_151 (errata).
"""
from __future__ import annotations

import polars as pl
import pytest

from aurumq_rl.factors.gtja191.batch_141_160 import (
    gtja_141,
    gtja_142,
    gtja_143,
    gtja_144,
    gtja_145,
    gtja_146,
    gtja_147,
    gtja_148,
    gtja_149,
    gtja_150,
    gtja_151,
    gtja_152,
    gtja_153,
    gtja_154,
    gtja_155,
    gtja_156,
    gtja_157,
    gtja_158,
    gtja_159,
    gtja_160,
)
from tests.factors.gtja191._parity import parity_check

# gtja_143 is a stub (Daic115 unfinished, recursive SELF reference).
_STUBS = {"gtja_143"}
_LONG_LOOKBACK = {"gtja_141", "gtja_148", "gtja_149", "gtja_154"}


_FACTORS = [
    ("gtja_141", gtja_141),
    ("gtja_142", gtja_142),
    ("gtja_143", gtja_143),
    ("gtja_144", gtja_144),
    ("gtja_145", gtja_145),
    ("gtja_146", gtja_146),
    ("gtja_147", gtja_147),
    ("gtja_148", gtja_148),
    ("gtja_149", gtja_149),
    ("gtja_150", gtja_150),
    ("gtja_151", gtja_151),
    ("gtja_152", gtja_152),
    ("gtja_153", gtja_153),
    ("gtja_154", gtja_154),
    ("gtja_155", gtja_155),
    ("gtja_156", gtja_156),
    ("gtja_157", gtja_157),
    ("gtja_158", gtja_158),
    ("gtja_159", gtja_159),
    ("gtja_160", gtja_160),
]


_STATUS = {
    "gtja_141": "drift",
    "gtja_143": "stub",
    "gtja_147": "missing",  # Daic115 ref uses qlib rolling_slope; we use regbeta
    "gtja_149": "missing",  # benchmark proxy: CS-mean differs from Daic115's mean(axis=1)
    "gtja_156": "drift",
    "gtja_162": "drift",  # not in this batch but listed for record
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
    if name in _STUBS:
        # Stub returns all-null; that's the contract.
        assert s.null_count() == synthetic_panel.height
        return
    if name in _LONG_LOOKBACK:
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
