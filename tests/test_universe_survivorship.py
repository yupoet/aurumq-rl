"""Tests for the C3 survivorship-mitigation fix.

Per-date ST filtering: the ``*_non_st`` universe variants must NOT drop a
stock's entire history because its *current* name contains ST/退. Rows stay in
the panel; ST-ness is carried per (date, stock) in ``is_st_array`` and
enforced by the downstream eligibility masks (env trading mask, train_v2
valid mask).

Static-membership disclosure: MAIN_BOARD / NPF* / GROWTH_BOARDS are date-less
snapshots locked on 2026-05-14. Applying them to a panel that starts
meaningfully earlier must emit a survivorship-bias UserWarning (mitigation by
disclosure — real point-in-time membership needs upstream data).
"""

from __future__ import annotations

import datetime
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from aurumq_rl.data_loader import (
    NEW_STOCK_PROTECT_DAYS,
    STATIC_UNIVERSE_LOCK_DATE,
    FactorPanelLoader,
    UniverseFilter,
    _load_static_universe,
    filter_universe,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _weekdays(start: datetime.date, n: int) -> list[datetime.date]:
    out: list[datetime.date] = []
    cur = start
    while len(out) < n:
        if cur.weekday() < 5:
            out.append(cur)
        cur += datetime.timedelta(days=1)
    return out


def _row(
    d: datetime.date,
    code: str,
    close: float,
    name: str | None = None,
    is_st: bool | None = None,
    alpha: float = 1.0,
) -> dict:
    rec = {
        "trade_date": d,
        "ts_code": code,
        "close": close,
        "pct_chg": 0.0,
        "vol": 1000.0,
        "alpha_x": alpha,
        "adj_factor": 1.0,  # keep suite pristine — no adj_factor warning
    }
    if name is not None:
        rec["name"] = name
    if is_st is not None:
        rec["is_st"] = is_st
    return rec


def _load(path: Path, dates: list[datetime.date]):
    loader = FactorPanelLoader(parquet_path=path)
    return loader.load_panel(
        start_date=dates[0],
        end_date=dates[-1],
        forward_period=1,
        universe_filter=UniverseFilter.MAIN_BOARD_NON_ST,
    )


# ---------------------------------------------------------------------------
# 1) Per-date ST eligibility: healthy early, ST later
# ---------------------------------------------------------------------------

N_DAYS = 10
ST_FROM = 5  # first day index on which 600001.SH is ST


@pytest.fixture
def becomes_st_panel(tmp_path: Path):
    """600001.SH is healthy on days 0-4 and ST from day 5; 600002.SH healthy.

    Both the ``is_st`` column and the per-row name flip on day 5 (the name is
    what a point-in-time exporter would store). Close prices trend so forward
    returns are non-zero.
    """
    dates = _weekdays(datetime.date(2022, 1, 3), N_DAYS)
    recs = []
    for i, d in enumerate(dates):
        st = i >= ST_FROM
        recs.append(
            _row(
                d,
                "600001.SH",
                close=10.0 + i,
                name="*ST得润" if st else "得润电子",
                is_st=st,
                alpha=1.0,
            )
        )
        recs.append(_row(d, "600002.SH", close=20.0 + i, name="健康股", is_st=False, alpha=2.0))
    path = tmp_path / "panel_becomes_st.parquet"
    pl.DataFrame(recs).write_parquet(path)
    return _load(path, dates)


def test_becoming_st_stock_keeps_full_history(becomes_st_panel) -> None:
    panel = becomes_st_panel
    assert "600001.SH" in panel.stock_codes, (
        "a stock that is currently ST-named must keep its pre-ST history"
    )
    j = panel.stock_codes.index("600001.SH")
    # Pre-ST days carry real data, not missing-cell defaults.
    assert not panel.is_suspended_array[:ST_FROM, j].any()
    assert np.isfinite(panel.return_array[0, j])


def test_is_st_array_is_per_date(becomes_st_panel) -> None:
    panel = becomes_st_panel
    j = panel.stock_codes.index("600001.SH")
    assert not panel.is_st_array[:ST_FROM, j].any(), "healthy early dates must not be ST"
    assert panel.is_st_array[ST_FROM:, j].all(), "ST dates must be flagged"


def test_env_trading_mask_enforces_st_per_date(becomes_st_panel) -> None:
    from aurumq_rl.env import _apply_trading_mask

    panel = becomes_st_panel
    j = panel.stock_codes.index("600001.SH")

    def _mask(t: int) -> np.ndarray:
        return _apply_trading_mask(
            returns=panel.return_array[t].astype(np.float64),
            pct_changes=panel.pct_change_array[t].astype(np.float64),
            is_st=panel.is_st_array[t],
            is_suspended=panel.is_suspended_array[t],
            days_since_ipo=panel.days_since_ipo_array[t].astype(np.float64),
            stock_codes=None,
            respect_dynamic_price_limits=False,
        )

    # Early (non-ST) date: tradeable, forward return passes through.
    early = _mask(1)
    assert early[j] != 0.0
    assert early[j] == pytest.approx(float(panel.return_array[1, j]))
    # ST date: excluded even though the raw forward return is non-zero.
    t_st = ST_FROM + 1
    assert float(panel.return_array[t_st, j]) != 0.0
    assert _mask(t_st)[j] == 0.0


# ---------------------------------------------------------------------------
# 2) Currently-ST-named stock: pre-ST history stays in the panel
# ---------------------------------------------------------------------------


def test_current_name_snapshot_does_not_erase_history(tmp_path: Path) -> None:
    """Exporters that stamp the CURRENT name on all rows must not lose history.

    The name says *ST everywhere, but the per-date ``is_st`` column is
    authoritative: early dates stay eligible.
    """
    dates = _weekdays(datetime.date(2022, 1, 3), N_DAYS)
    recs = []
    for i, d in enumerate(dates):
        recs.append(_row(d, "600001.SH", close=10.0 + i, name="*ST得润", is_st=i >= ST_FROM))
        recs.append(_row(d, "600002.SH", close=20.0 + i, name="健康股", is_st=False))
    path = tmp_path / "panel_current_name.parquet"
    pl.DataFrame(recs).write_parquet(path)
    panel = _load(path, dates)

    assert "600001.SH" in panel.stock_codes
    j = panel.stock_codes.index("600001.SH")
    assert not panel.is_suspended_array[:ST_FROM, j].any()
    assert not panel.is_st_array[:ST_FROM, j].any()
    assert panel.is_st_array[ST_FROM:, j].all()


def test_name_fallback_populates_is_st_when_column_missing(tmp_path: Path) -> None:
    """No ``is_st`` column: derive per-ROW ST flags from the name column."""
    dates = _weekdays(datetime.date(2022, 1, 3), N_DAYS)
    recs = []
    for i, d in enumerate(dates):
        recs.append(
            _row(d, "600001.SH", close=10.0 + i, name="*ST得润" if i >= ST_FROM else "得润电子")
        )
        recs.append(_row(d, "600002.SH", close=20.0 + i, name="健康股"))
    path = tmp_path / "panel_name_only.parquet"
    pl.DataFrame(recs).write_parquet(path)
    panel = _load(path, dates)

    assert "600001.SH" in panel.stock_codes
    j = panel.stock_codes.index("600001.SH")
    k = panel.stock_codes.index("600002.SH")
    assert not panel.is_st_array[:ST_FROM, j].any()
    assert panel.is_st_array[ST_FROM:, j].all()
    assert not panel.is_st_array[:, k].any()
    # Days-since-IPO default for present rows must still pass the gate.
    assert panel.days_since_ipo_array[0, j] >= NEW_STOCK_PROTECT_DAYS


# ---------------------------------------------------------------------------
# 3) Static-membership survivorship disclosure warning
# ---------------------------------------------------------------------------


@pytest.fixture
def static_membership_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point AURUMQ_UNIVERSE_DIR at a synthetic MAIN_BOARD membership parquet.

    ``_load_static_universe`` is lru_cached, so the cache is cleared both
    before (stale empty-set entries) and after (do not leak the synthetic
    membership into other tests).
    """
    d = tmp_path / "universes"
    d.mkdir()
    pl.DataFrame({"stock_code": ["600001.SH", "600002.SH"]}).write_parquet(
        d / "MAIN_BOARD_membership.parquet"
    )
    monkeypatch.setenv("AURUMQ_UNIVERSE_DIR", str(d))
    _load_static_universe.cache_clear()
    yield d
    _load_static_universe.cache_clear()


def _universe_df(start: datetime.date) -> pl.DataFrame:
    dates = _weekdays(start, 3)
    return pl.DataFrame(
        {
            "ts_code": ["600001.SH"] * len(dates),
            "trade_date": dates,
        }
    )


def test_disclosure_warning_fires_for_early_panel(static_membership_dir) -> None:
    df = _universe_df(datetime.date(2023, 1, 3))
    with pytest.warns(UserWarning, match="survivorship"):
        filter_universe(df, mode=UniverseFilter.MAIN_BOARD)


def test_no_disclosure_warning_for_post_lock_panel(static_membership_dir) -> None:
    df = _universe_df(STATIC_UNIVERSE_LOCK_DATE + datetime.timedelta(days=7))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        filter_universe(df, mode=UniverseFilter.MAIN_BOARD)
    assert not [w for w in caught if "survivorship" in str(w.message)]


def test_no_disclosure_warning_on_regex_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No membership parquet staged → regex fallback is code-based, no snapshot."""
    empty = tmp_path / "no_universes"
    empty.mkdir()
    monkeypatch.setenv("AURUMQ_UNIVERSE_DIR", str(empty))
    _load_static_universe.cache_clear()
    try:
        df = _universe_df(datetime.date(2023, 1, 3))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            filter_universe(df, mode=UniverseFilter.MAIN_BOARD)
        assert not [w for w in caught if "survivorship" in str(w.message)]
    finally:
        _load_static_universe.cache_clear()


def test_disclosure_warning_names_the_lock_date(static_membership_dir) -> None:
    df = _universe_df(datetime.date(2023, 1, 3))
    with pytest.warns(UserWarning, match=str(STATIC_UNIVERSE_LOCK_DATE)):
        filter_universe(df, mode=UniverseFilter.MAIN_BOARD)
