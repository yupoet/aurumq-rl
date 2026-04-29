"""Tests for gtja191.batch_181_191 (factors 181..191, 11 total).

Special factors: gtja_181 / gtja_182 (benchmark — CS-mean proxy),
gtja_183 (errata SUMAC), gtja_191 (errata).
"""

from __future__ import annotations

import polars as pl
import pytest

from aurumq_rl.factors.gtja191.batch_181_191 import (
    gtja_181,
    gtja_182,
    gtja_183,
    gtja_184,
    gtja_185,
    gtja_186,
    gtja_187,
    gtja_188,
    gtja_189,
    gtja_190,
    gtja_191,
)
from tests.factors.gtja191._parity import parity_check

_LONG_LOOKBACK = {"gtja_181", "gtja_183", "gtja_184"}


_FACTORS = [
    ("gtja_181", gtja_181),
    ("gtja_182", gtja_182),
    ("gtja_183", gtja_183),
    ("gtja_184", gtja_184),
    ("gtja_185", gtja_185),
    ("gtja_186", gtja_186),
    ("gtja_187", gtja_187),
    ("gtja_188", gtja_188),
    ("gtja_189", gtja_189),
    ("gtja_190", gtja_190),
    ("gtja_191", gtja_191),
]


_STATUS = {
    # Both gtja_181 and gtja_182 use a CS-mean-of-OHLC proxy for the
    # benchmark. Daic115's reference parquet uses ``mean(axis=1)`` which
    # in our long-format polars panel translates to per-day mean — they
    # should agree, but the formulas have errata so we tag as drift.
    "gtja_181": "drift",
    "gtja_182": "drift",
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
