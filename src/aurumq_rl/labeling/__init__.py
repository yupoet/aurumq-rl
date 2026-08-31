"""Main-wave (主升浪) label scanning — pure-math RL-side port.

Originated from AurumQ data-side label ablation (commit 3209eda → bf67910).
P0 winner = Method A (`v2_excess_adaptive`) at horizon t3.

See `handoffs/2026-05-09-wave-label-ablation/SPEC.md` and `RESULTS.md` for
the full design and ablation study (4 methods × 3 horizons × LightGBM
learnability + null tests, all PASSED).

This package is RL-side only — no DB / FastAPI / Alembic dependencies.
Inputs come from a pre-loaded `MarketPanel` (numpy arrays).

Public API
----------
- MarketPanel:                    container for (T, S) arrays
- Event:                          one detected main-wave event
- detect_events_v2:               Method A — user-original v2 (P0 winner)
- detect_events_trend_scanning:   Method B — Lopez de Prado 2020
- detect_events_triple_barrier:   Method C — Lopez de Prado 2018
- detect_events_directional_change: Method D — Glattfelder/Tsang 2011
- dedupe_events:                  per-stock non-overlap resolution
- derive_labels:                  events → (t, j) binary labels at horizon
- search_threshold:               target-pos-rate threshold search
- scan_main_wave_p0:              top-level convenience: A_t3 with τ=1.2327
- cusum_filter:                   symmetric CUSUM event sampler (opt-in seeding, issue #8 Part 1)
- label_concurrency:              per-bar overlap count for outcome windows (issue #8 Part 2)
- average_uniqueness:             per-label sample_weight from concurrency (issue #8 Part 2)
- adaptive_threshold:             v2_excess_adaptive threshold formula, opt-in √horizon scaling
                                   (issue #8 Part 4; P0-locked default unaffected)

Example
-------
>>> from aurumq_rl.labeling import scan_main_wave_p0, MarketPanel
>>> panel = MarketPanel(trade_dates=..., ts_codes=..., adj_close=..., ...)
>>> events, label_df = scan_main_wave_p0(panel)
>>> # events: deduped Method A events, quality ≥ τ_A=1.2327
>>> # label_df: (trade_date, ts_code, y_t3) for each decision cell
"""

from .directional_change import detect_events_directional_change
from .events import Event, dedupe_events, derive_labels, events_to_dataframe
from .panels import MarketPanel
from .sampling import average_uniqueness, cusum_filter, label_concurrency
from .thresholds import ThresholdResult, search_threshold
from .trend_scanning import detect_events_trend_scanning
from .triple_barrier import detect_events_triple_barrier
from .v2_excess_adaptive import adaptive_threshold, detect_events_v2

# ---------------------------------------------------------------------------
# P0 locked configuration (data-side ablation winner)
# ---------------------------------------------------------------------------

P0_LABEL_NAME: str = "v2_excess_adaptive"
P0_HORIZON: str = "t3"
P0_THRESHOLD: float = 1.2327  # event_quality threshold from train_eff
P0_PANEL_VERSION: str = "phase26f_v3_344"
P0_FEATURE_SCHEMA_HASH: str = "5e71e158e331"


def scan_main_wave_p0(
    panel: MarketPanel,
    quality_threshold: float = P0_THRESHOLD,
):
    """Run P0 (Method A_t3) end-to-end on a MarketPanel.

    Returns
    -------
    events : list[Event]
        Deduped Method A events with event_quality >= quality_threshold.
    label_df : polars.DataFrame
        Long-form (trade_date, ts_code, y) at horizon t3.
    """
    raw_events = detect_events_v2(panel)
    events = dedupe_events(raw_events)
    label_df = derive_labels(
        events=events,
        trade_dates=panel.trade_dates,
        ts_codes=panel.ts_codes,
        horizon=P0_HORIZON,
        quality_threshold=quality_threshold,
    )
    return events, label_df


__all__ = [
    "MarketPanel",
    "Event",
    "dedupe_events",
    "derive_labels",
    "events_to_dataframe",
    "search_threshold",
    "ThresholdResult",
    "detect_events_v2",
    "detect_events_trend_scanning",
    "detect_events_triple_barrier",
    "detect_events_directional_change",
    "scan_main_wave_p0",
    "cusum_filter",
    "label_concurrency",
    "average_uniqueness",
    "adaptive_threshold",
    "P0_LABEL_NAME",
    "P0_HORIZON",
    "P0_THRESHOLD",
    "P0_PANEL_VERSION",
    "P0_FEATURE_SCHEMA_HASH",
]
