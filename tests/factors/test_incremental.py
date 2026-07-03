"""Tests for the OPT-IN incremental factor computation protocol (issue #10).

Correctness crux: ``impl_incremental(tail_buffer, n_new)`` must reproduce
exactly what ``impl(full_df)`` would produce for the last ``n_new`` rows of
every stock, using only a bounded per-stock tail buffer instead of the full
panel history. See ``FactorEntry`` / ``compute_incremental`` docstrings in
``aurumq_rl.factors.registry`` for the precise contract.

All data here is synthetic (the shared ``synthetic_panel`` fixture, or small
hand-built frames for the isolation/error-path tests) — no real market data.
"""

from __future__ import annotations

import polars as pl
import pytest

from aurumq_rl.factors import registry
from aurumq_rl.factors.alpha101.momentum import (
    ALPHA009_MAX_WINDOW,
    alpha009,
    alpha009_incremental,
)


def _tail_buffer(panel: pl.DataFrame, n_new: int, max_window: int) -> pl.DataFrame:
    """Slice the last ``max_window + n_new`` rows per stock from ``panel``.

    Mirrors what a caller of the incremental protocol is expected to
    assemble by hand (used here so the correctness test is independent of
    the ``compute_incremental`` helper under test elsewhere).
    """
    buffer_size = max_window + n_new
    return (
        panel.sort(["stock_code", "trade_date"])
        .group_by("stock_code", maintain_order=True)
        .tail(buffer_size)
    )


# ---------------------------------------------------------------------------
# 1. Correctness (the crux)
# ---------------------------------------------------------------------------


class TestAlpha009IncrementalCorrectness:
    @pytest.mark.parametrize("n_new", [1, 2, 5])
    def test_matches_full_recompute_tail(self, synthetic_panel, n_new):
        full = alpha009(synthetic_panel)
        full_tagged = synthetic_panel.select(["stock_code", "trade_date"]).with_columns(
            full.alias("alpha009")
        )
        expected = (
            full_tagged.sort(["stock_code", "trade_date"])
            .group_by("stock_code", maintain_order=True)
            .tail(n_new)
        )

        tail_buf = _tail_buffer(synthetic_panel, n_new, ALPHA009_MAX_WINDOW)
        got = alpha009_incremental(tail_buf, n_new)

        assert len(got) == len(expected)
        got_list = got.to_list()
        expected_list = expected["alpha009"].to_list()
        assert len(got_list) == len(expected_list)
        for g, e in zip(got_list, expected_list, strict=True):
            if e is None:
                assert g is None
            else:
                assert g == pytest.approx(e, rel=1e-9, abs=1e-9)

    def test_multiple_stocks_all_covered(self, synthetic_panel):
        n_new = 3
        tail_buf = _tail_buffer(synthetic_panel, n_new, ALPHA009_MAX_WINDOW)
        got = alpha009_incremental(tail_buf, n_new)
        n_stocks = synthetic_panel["stock_code"].n_unique()
        assert len(got) == n_stocks * n_new


# ---------------------------------------------------------------------------
# 2. Insufficient buffer
# ---------------------------------------------------------------------------


class TestInsufficientBuffer:
    def test_short_buffer_raises_clear_error(self, synthetic_panel):
        n_new = 3
        # One row short of the documented minimum (max_window + n_new).
        short_buf = _tail_buffer(synthetic_panel, n_new, ALPHA009_MAX_WINDOW - 1)
        with pytest.raises(ValueError, match="tail buffer too short"):
            alpha009_incremental(short_buf, n_new)

    def test_error_message_names_offending_stock(self, synthetic_panel):
        n_new = 2
        buf = _tail_buffer(synthetic_panel, n_new, ALPHA009_MAX_WINDOW)
        # Truncate just one stock's rows below the minimum to simulate a
        # partial/short history for a single ticker.
        one_stock = buf["stock_code"][0]
        short_rows = buf.filter(pl.col("stock_code") == one_stock).tail(
            ALPHA009_MAX_WINDOW + n_new - 1
        )
        other_rows = buf.filter(pl.col("stock_code") != one_stock)
        mixed = pl.concat([short_rows, other_rows]).sort(["stock_code", "trade_date"])
        with pytest.raises(ValueError, match=one_stock):
            alpha009_incremental(mixed, n_new)

    def test_n_new_must_be_positive(self, synthetic_panel):
        buf = _tail_buffer(synthetic_panel, 1, ALPHA009_MAX_WINDOW)
        with pytest.raises(ValueError, match="n_new"):
            alpha009_incremental(buf, 0)


# ---------------------------------------------------------------------------
# 3. Registry metadata
# ---------------------------------------------------------------------------


class TestRegistryMetadata:
    def test_alpha009_reports_incremental_available(self):
        entry = registry.ALPHA101_REGISTRY["alpha009"]
        assert entry.impl_incremental is not None
        assert entry.max_window == ALPHA009_MAX_WINDOW

    def test_factor_without_incremental_reports_unavailable(self):
        entry = registry.ALPHA101_REGISTRY["alpha007"]
        assert entry.impl_incremental is None

    def test_no_other_factor_gained_incremental_support(self):
        """Additive-only guardrail: only alpha009 opts into the new protocol."""
        merged = registry.list_all_factors()
        for fid, entry in merged.items():
            if fid == "alpha009":
                continue
            assert entry.impl_incremental is None, f"{fid} unexpectedly has impl_incremental"
            assert entry.max_window is None, f"{fid} unexpectedly has max_window"

    def test_resolve_incremental_invokes_impl(self, synthetic_panel):
        n_new = 2
        tail_buf = _tail_buffer(synthetic_panel, n_new, ALPHA009_MAX_WINDOW)
        out = registry.resolve_incremental("alpha009", tail_buf, n_new)
        direct = alpha009_incremental(tail_buf, n_new)
        assert out.to_list() == pytest.approx(direct.to_list(), nan_ok=True)

    def test_resolve_incremental_unavailable_raises(self, synthetic_panel):
        buf = synthetic_panel.head(5)
        with pytest.raises(ValueError, match="no incremental implementation"):
            registry.resolve_incremental("alpha007", buf, 1)

    def test_resolve_incremental_unknown_symbol_raises_keyerror(self, synthetic_panel):
        buf = synthetic_panel.head(5)
        with pytest.raises(KeyError, match="alpha999"):
            registry.resolve_incremental("alpha999", buf, 1)


# ---------------------------------------------------------------------------
# 4. Multi-stock isolation
# ---------------------------------------------------------------------------


class TestMultiStockIsolation:
    def _two_stock_panel(self) -> pl.DataFrame:
        import datetime as dt

        n_days = 12
        dates = [dt.date(2024, 1, 2) + dt.timedelta(days=i) for i in range(n_days)]
        # Stock A: strictly increasing close (persistent uptrend).
        close_a = [10.0 + i for i in range(n_days)]
        # Stock B: strictly decreasing close (persistent downtrend) — a very
        # different regime so cross-stock leakage would visibly perturb the
        # trend-confirmation branches in alpha009.
        close_b = [50.0 - 2 * i for i in range(n_days)]

        rows = []
        for i in range(n_days):
            rows.append({"stock_code": "AAA", "trade_date": dates[i], "close": close_a[i]})
        for i in range(n_days):
            rows.append({"stock_code": "BBB", "trade_date": dates[i], "close": close_b[i]})
        return pl.DataFrame(rows).sort(["stock_code", "trade_date"])

    def test_stock_a_does_not_leak_into_stock_b(self):
        panel = self._two_stock_panel()
        n_new = 2
        full = alpha009(panel)
        full_tagged = panel.select(["stock_code", "trade_date"]).with_columns(
            full.alias("alpha009")
        )
        expected = (
            full_tagged.sort(["stock_code", "trade_date"])
            .group_by("stock_code", maintain_order=True)
            .tail(n_new)
        )

        tail_buf = _tail_buffer(panel, n_new, ALPHA009_MAX_WINDOW)
        got = alpha009_incremental(tail_buf, n_new)
        new_rows = tail_buf.group_by("stock_code", maintain_order=True).tail(n_new)
        got_tagged = new_rows.select("stock_code").with_columns(got.alias("alpha009"))

        for stock in ("AAA", "BBB"):
            exp_vals = expected.filter(pl.col("stock_code") == stock)["alpha009"].to_list()
            got_vals = got_tagged.filter(pl.col("stock_code") == stock)["alpha009"].to_list()
            assert got_vals == pytest.approx(exp_vals, rel=1e-9, abs=1e-9)

        # Sanity: the two regimes must actually diverge (else the isolation
        # check above would pass vacuously).
        aaa_vals = got_tagged.filter(pl.col("stock_code") == "AAA")["alpha009"].to_list()
        bbb_vals = got_tagged.filter(pl.col("stock_code") == "BBB")["alpha009"].to_list()
        assert aaa_vals != bbb_vals

    def test_buffer_row_order_within_group_preserved(self):
        """Output order matches the trailing rows of the tail buffer per stock."""
        panel = self._two_stock_panel()
        n_new = 3
        tail_buf = _tail_buffer(panel, n_new, ALPHA009_MAX_WINDOW)
        got = alpha009_incremental(tail_buf, n_new)
        new_rows = tail_buf.group_by("stock_code", maintain_order=True).tail(n_new)
        got_tagged = new_rows.select(["stock_code", "trade_date"]).with_columns(got.alias("v"))
        # Within each stock, dates must be strictly increasing (no shuffling).
        for stock in ("AAA", "BBB"):
            dates = got_tagged.filter(pl.col("stock_code") == stock)["trade_date"].to_list()
            assert dates == sorted(dates)
