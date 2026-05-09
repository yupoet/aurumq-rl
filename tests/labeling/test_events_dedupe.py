"""Test event dedupe and label derivation logic."""

from datetime import date, timedelta

from aurumq_rl.labeling.events import Event, dedupe_events, derive_labels, HORIZON_WINDOW


def _trade_dates(n: int) -> list[date]:
    base = date(2024, 1, 1)
    return [base + timedelta(days=i) for i in range(n)]


def test_dedupe_non_overlapping_kept():
    e1 = Event(ts_code="A", event_start_idx=10, event_peak_idx=15, event_quality=1.0, event_method="A")
    e2 = Event(ts_code="A", event_start_idx=20, event_peak_idx=25, event_quality=2.0, event_method="A")
    out = dedupe_events([e1, e2])
    assert len(out) == 2


def test_dedupe_overlap_keeps_higher_quality():
    e1 = Event(ts_code="A", event_start_idx=10, event_peak_idx=20, event_quality=1.0, event_method="A")
    # e2 starts at 15, which is <= e1.peak (20), overlap, e2 lower quality → keep e1
    e2 = Event(ts_code="A", event_start_idx=15, event_peak_idx=25, event_quality=0.5, event_method="A")
    out = dedupe_events([e1, e2])
    assert len(out) == 1
    assert out[0].event_start_idx == 10
    assert out[0].event_quality == 1.0


def test_dedupe_overlap_keeps_higher_quality_swap():
    e1 = Event(ts_code="A", event_start_idx=10, event_peak_idx=20, event_quality=1.0, event_method="A")
    # e2 at 15, overlap, but e2 has higher quality → e2 wins
    e2 = Event(ts_code="A", event_start_idx=15, event_peak_idx=25, event_quality=2.0, event_method="A")
    out = dedupe_events([e1, e2])
    assert len(out) == 1
    assert out[0].event_quality == 2.0


def test_dedupe_per_stock_isolation():
    e1 = Event(ts_code="A", event_start_idx=10, event_peak_idx=20, event_quality=1.0, event_method="A")
    e2 = Event(ts_code="B", event_start_idx=15, event_peak_idx=25, event_quality=1.0, event_method="A")
    # different stocks → both keep
    out = dedupe_events([e1, e2])
    assert len(out) == 2


def test_derive_labels_t1():
    """y_t1[t] = 1 iff event_start = t+1."""
    dates = _trade_dates(50)
    ts_codes = ["A", "B"]
    # event A starts at idx 10 (decision day = 9)
    e1 = Event(ts_code="A", event_start_idx=10, event_peak_idx=15, event_quality=2.0, event_method="A")
    df = derive_labels([e1], dates, ts_codes, "t1")
    # Decision day 9 should have y=1 for A
    rec = df.filter((df["trade_date"] == dates[9]) & (df["ts_code"] == "A"))
    assert rec.height == 1
    assert rec["y"].item() == 1
    # Decision day 8 should have y=0
    rec0 = df.filter((df["trade_date"] == dates[8]) & (df["ts_code"] == "A"))
    assert rec0["y"].item() == 0
    # Day 10 should be 0 (event already started)
    rec1 = df.filter((df["trade_date"] == dates[10]) & (df["ts_code"] == "A"))
    assert rec1["y"].item() == 0


def test_derive_labels_t3_window():
    """y_t3[t] = 1 iff event_start in {t+1, t+2, t+3}."""
    dates = _trade_dates(50)
    ts_codes = ["A"]
    e1 = Event(ts_code="A", event_start_idx=10, event_peak_idx=15, event_quality=2.0, event_method="A")
    df = derive_labels([e1], dates, ts_codes, "t3")
    # Day 7 (t=7, t+3=10) → y=1
    assert df.filter(df["trade_date"] == dates[7])["y"].item() == 1
    # Day 8 (t+2=10) → y=1
    assert df.filter(df["trade_date"] == dates[8])["y"].item() == 1
    # Day 9 (t+1=10) → y=1
    assert df.filter(df["trade_date"] == dates[9])["y"].item() == 1
    # Day 6 (t+3=9, no event) → y=0
    assert df.filter(df["trade_date"] == dates[6])["y"].item() == 0


def test_derive_labels_quality_threshold():
    dates = _trade_dates(50)
    ts_codes = ["A"]
    e_low = Event(ts_code="A", event_start_idx=10, event_peak_idx=15, event_quality=0.3, event_method="A")
    e_high = Event(ts_code="A", event_start_idx=20, event_peak_idx=25, event_quality=2.0, event_method="A")
    df = derive_labels([e_low, e_high], dates, ts_codes, "t1", quality_threshold=1.0)
    # Day 9 → e_low at 10, quality 0.3 < 1.0, dropped → y=0
    assert df.filter(df["trade_date"] == dates[9])["y"].item() == 0
    # Day 19 → e_high at 20, quality 2.0 → y=1
    assert df.filter(df["trade_date"] == dates[19])["y"].item() == 1


def test_derive_labels_lookahead_truncation():
    """Cells where t+window > n_dates - 1 must not appear in output."""
    dates = _trade_dates(30)
    ts_codes = ["A"]
    df = derive_labels([], dates, ts_codes, "e20")
    # max_t = 30 - 20 - 1 = 9, so output covers t=0..9 (inclusive) = 10 days
    assert df["trade_date"].n_unique() == 10
    assert df["trade_date"].max() == dates[9]
