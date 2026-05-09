"""Event primitive + dedupe + label derivation.

See SPEC §1. Each labeling method generates events. Events are deduplicated
per stock (non-overlapping). Then labels at horizons {t1, t3, e20} are derived.

Decision day t = event_start - 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, Literal

import polars as pl


__all__ = ["Event", "dedupe_events", "derive_labels", "events_to_dataframe"]


HORIZONS = ("t1", "t3", "e20")
HORIZON_WINDOW: dict[str, int] = {"t1": 1, "t3": 3, "e20": 20}


@dataclass(frozen=True)
class Event:
    """One detected main-wave event for a single stock.

    By convention: decision day t = event_start_idx - 1 (T-1).
    event_quality is method-specific continuous score (used for thresholding).
    """

    ts_code: str
    event_start_idx: int       # index into the per-stock trading day array
    event_peak_idx: int        # index of peak (>= event_start_idx)
    event_quality: float       # method-specific continuous score
    event_method: str          # 'A' | 'B' | 'C' | 'D'

    @property
    def duration(self) -> int:
        return self.event_peak_idx - self.event_start_idx


def dedupe_events(events: Iterable[Event]) -> list[Event]:
    """Per-stock non-overlapping events.

    Sort by event_start_idx, sweep. If an event overlaps the prior accepted event
    (event_start_idx <= prev.event_peak_idx), keep the one with higher event_quality;
    drop the other. After resolution, advance.
    """
    by_stock: dict[str, list[Event]] = {}
    for e in events:
        by_stock.setdefault(e.ts_code, []).append(e)

    out: list[Event] = []
    for ts_code, evs in by_stock.items():
        evs.sort(key=lambda e: (e.event_start_idx, -e.event_quality))
        accepted: list[Event] = []
        for e in evs:
            if not accepted:
                accepted.append(e)
                continue
            prev = accepted[-1]
            # New event overlaps with prev if its start <= prev's peak
            if e.event_start_idx <= prev.event_peak_idx:
                # Keep higher-quality one
                if e.event_quality > prev.event_quality:
                    accepted[-1] = e
                # else drop new
            else:
                accepted.append(e)
        out.extend(accepted)
    return out


def events_to_dataframe(
    events: list[Event],
    trade_dates: list[date],
) -> pl.DataFrame:
    """Convert events to long-form parquet-friendly DataFrame."""
    rows = []
    for e in events:
        rows.append({
            "ts_code": e.ts_code,
            "event_start_date": trade_dates[e.event_start_idx],
            "event_peak_date": trade_dates[e.event_peak_idx],
            "event_start_idx": e.event_start_idx,
            "event_peak_idx": e.event_peak_idx,
            "event_quality": float(e.event_quality),
            "event_method": e.event_method,
            "duration": int(e.duration),
        })
    if not rows:
        return pl.DataFrame(schema={
            "ts_code": pl.Utf8,
            "event_start_date": pl.Date,
            "event_peak_date": pl.Date,
            "event_start_idx": pl.Int32,
            "event_peak_idx": pl.Int32,
            "event_quality": pl.Float64,
            "event_method": pl.Utf8,
            "duration": pl.Int32,
        })
    return pl.DataFrame(rows)


def derive_labels(
    events: list[Event],
    trade_dates: list[date],
    ts_codes: list[str],
    horizon: Literal["t1", "t3", "e20"],
    quality_threshold: float = 0.0,
) -> pl.DataFrame:
    """Convert events to per-(t, j) binary labels at one horizon.

    y[t, j] = 1 iff exists event e with
        ts_code == j
        event_quality >= quality_threshold
        event_start_idx in {t+1, t+2, ..., t+window}
    where window = HORIZON_WINDOW[horizon].

    Returns long-form DataFrame (trade_date, ts_code, y).
    Cells where t+window > len(trade_dates) - 1 are excluded (insufficient lookahead).
    Vectorized via numpy.
    """
    import numpy as np

    if horizon not in HORIZON_WINDOW:
        raise ValueError(f"Unknown horizon {horizon}; expected one of {HORIZONS}")
    window = HORIZON_WINDOW[horizon]
    n_dates = len(trade_dates)
    n_stocks = len(ts_codes)
    code_to_j = {c: i for i, c in enumerate(ts_codes)}

    # event_start[t, j] = 1 iff a qualified event starts at (t, j)
    event_start = np.zeros((n_dates, n_stocks), dtype=np.int8)
    for e in events:
        if e.event_quality < quality_threshold:
            continue
        j = code_to_j.get(e.ts_code)
        if j is None or e.event_start_idx >= n_dates:
            continue
        event_start[e.event_start_idx, j] = 1

    # y[t, j] = max over k=1..window of event_start[t+k, j]
    # Equivalent: shift event_start backward by k and OR.
    max_t = n_dates - window - 1
    if max_t < 0:
        return pl.DataFrame(
            schema={"trade_date": pl.Date, "ts_code": pl.Utf8, "y": pl.Int8},
        )
    y = np.zeros((max_t + 1, n_stocks), dtype=np.int8)
    for k in range(1, window + 1):
        # event_start at row t+k → contributes to label at row t
        y[: max_t + 1] |= event_start[k : max_t + 1 + k]

    # Long-form output via polars (date objects → cast to pl.Date explicitly)
    date_col = pl.Series(
        "trade_date",
        [trade_dates[t] for t in range(max_t + 1) for _ in range(n_stocks)],
        dtype=pl.Date,
    )
    code_col = pl.Series(
        "ts_code",
        np.tile(np.array(ts_codes), max_t + 1),
        dtype=pl.Utf8,
    )
    y_col = pl.Series("y", y.reshape(-1), dtype=pl.Int8)
    return pl.DataFrame([date_col, code_col, y_col])
