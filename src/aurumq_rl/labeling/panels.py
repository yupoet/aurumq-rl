"""Market panel dataclass for label scanning — RL-side, no DB.

This is the AurumQ-RL counterpart of `aurumq.labeling.panels.MarketPanel`
on the data side. The data-side version loads from PG; here we just hold
pre-loaded numpy arrays so labeling math stays portable.

Caller responsibility: produce the (T, S) arrays from whatever bundle source
(parquet shards, OSS-fetched panels, in-memory arrays from training data).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np


__all__ = ["MarketPanel"]


@dataclass
class MarketPanel:
    """Pre-loaded (T, S) market panel for event detection.

    Layout
    ------
    trade_dates:   list[date], length T
    ts_codes:      list[str],  length S, stable order
    adj_close:     (T, S) float64, NaN for missing or out-of-universe cells
    pct_change:    (T, S) float64 — daily simple returns (raw)
    amount:        (T, S) float64 — raw amount in 元 (used for liquidity gate)
    universe:      (T, S) bool — True iff main-board, non-ST, non-suspended,
                                 listed≥60d, data complete that day
    benchmark_close: (T,) float64 — main-board equal-weighted close index
    """

    trade_dates: list[date]
    ts_codes: list[str]
    adj_close: np.ndarray
    pct_change: np.ndarray
    amount: np.ndarray
    universe: np.ndarray
    benchmark_close: np.ndarray

    @property
    def n_dates(self) -> int:
        return len(self.trade_dates)

    @property
    def n_stocks(self) -> int:
        return len(self.ts_codes)

    def __post_init__(self) -> None:
        T = self.n_dates
        S = self.n_stocks
        for name, arr, expected in [
            ("adj_close", self.adj_close, (T, S)),
            ("pct_change", self.pct_change, (T, S)),
            ("amount", self.amount, (T, S)),
            ("universe", self.universe, (T, S)),
            ("benchmark_close", self.benchmark_close, (T,)),
        ]:
            if arr.shape != expected:
                raise ValueError(
                    f"MarketPanel.{name} shape mismatch: got {arr.shape}, expected {expected}"
                )
