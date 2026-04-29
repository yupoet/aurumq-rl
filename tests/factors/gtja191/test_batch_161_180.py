"""Tests for gtja191.batch_161_180 (factors 161..180, 20 total).

Special factors: gtja_165 (errata), gtja_166 (errata).
"""

from __future__ import annotations

import polars as pl
import pytest

from aurumq_rl.factors.gtja191.batch_161_180 import (
    gtja_161,
    gtja_162,
    gtja_163,
    gtja_164,
    gtja_165,
    gtja_166,
    gtja_167,
    gtja_168,
    gtja_169,
    gtja_170,
    gtja_171,
    gtja_172,
    gtja_173,
    gtja_174,
    gtja_175,
    gtja_176,
    gtja_177,
    gtja_178,
    gtja_179,
    gtja_180,
)
from tests.factors.gtja191._parity import parity_check

_LONG_LOOKBACK = {"gtja_165", "gtja_170", "gtja_176", "gtja_179"}


_FACTORS = [
    ("gtja_161", gtja_161),
    ("gtja_162", gtja_162),
    ("gtja_163", gtja_163),
    ("gtja_164", gtja_164),
    ("gtja_165", gtja_165),
    ("gtja_166", gtja_166),
    ("gtja_167", gtja_167),
    ("gtja_168", gtja_168),
    ("gtja_169", gtja_169),
    ("gtja_170", gtja_170),
    ("gtja_171", gtja_171),
    ("gtja_172", gtja_172),
    ("gtja_173", gtja_173),
    ("gtja_174", gtja_174),
    ("gtja_175", gtja_175),
    ("gtja_176", gtja_176),
    ("gtja_177", gtja_177),
    ("gtja_178", gtja_178),
    ("gtja_179", gtja_179),
    ("gtja_180", gtja_180),
]


_STATUS = {
    "gtja_162": "drift",
    "gtja_164": "drift",
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
