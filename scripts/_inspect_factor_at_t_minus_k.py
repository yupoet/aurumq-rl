#!/usr/bin/env python3
"""Phase 23 — focused factor analysis at T-1 / T-2 / T-3 of episodes.

Companion to ``_inspect_main_wave_episodes.py``. This script computes the
cross-section z-score of EVERY factor in a user-supplied prefix list at
each of T-k for k in [1, K_max], showing how the signal builds up
heading into the episode start.

Used to validate hypotheses about which factors are pre-T-1 leading
indicators (e.g., user's hypothesis: cumulative main-force POSITION
matters more than recent FLOW).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np
import polars as pl

from aurumq_rl.main_wave_episodes import (
    EpisodeConfig, find_main_wave_episodes,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-path", type=Path, required=True)
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--out", type=Path, default=Path("runs/_episode_inspect/factor_t_minus_k.md"))
    p.add_argument("--prefixes", nargs="+",
                   default=["mf_", "mfp_", "hk_", "mg_", "inst_", "hm_",
                            "senti_", "cyq_", "ind_", "fund_"],
                   help="Factor prefixes to analyse")
    p.add_argument("--k-max", type=int, default=5,
                   help="Look back up to T-k_max days before t_start")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    start = dt.date.fromisoformat(args.start_date)
    end = dt.date.fromisoformat(args.end_date)
    print(f"[factor_tmk] reading {args.data_path}, window {start}..{end}")
    df = pl.read_parquet(args.data_path).filter(
        (pl.col("trade_date") >= start) & (pl.col("trade_date") <= end)
    )

    # Pivot helpers
    def _pivot_arr(field: str, codes: list[str], dates: list, dtype=np.float32) -> np.ndarray:
        piv = (df.select(["trade_date", "ts_code", field])
                 .pivot(values=field, index="trade_date", on="ts_code")
                 .sort("trade_date"))
        existing = {c for c in piv.columns if c != "trade_date"}
        n_dates = len(dates)
        arrs = []
        for code in codes:
            if code in existing:
                series = piv.get_column(code)
                if series.dtype == pl.Boolean:
                    col = series.fill_null(False).to_numpy().astype(np.bool_)
                else:
                    col = series.fill_null(0.0).to_numpy()
            else:
                col = np.zeros(n_dates, dtype=(np.bool_ if dtype == np.bool_ else np.float32))
            if dtype == np.bool_:
                arrs.append(col.astype(np.bool_, copy=False))
            else:
                arrs.append(col.astype(np.float32, copy=False))
        # Re-align to dates
        piv_dates = piv.get_column("trade_date").to_list()
        d2r = {d: i for i, d in enumerate(piv_dates)}
        idx = [d2r[d] for d in dates if d in d2r]
        return np.stack(arrs, axis=1)[idx]

    # Build close + universe
    close_piv = (df.select(["trade_date", "ts_code", "close"])
                   .pivot(values="close", index="trade_date", on="ts_code")
                   .sort("trade_date"))
    stock_codes = [c for c in close_piv.columns if c != "trade_date"]
    dates = close_piv.get_column("trade_date").to_list()
    n_dates, n_stocks = len(dates), len(stock_codes)
    print(f"[factor_tmk] panel: dates={n_dates} stocks={n_stocks}")

    close = _pivot_arr("close", stock_codes, dates)
    vol = _pivot_arr("vol", stock_codes, dates)
    is_st = _pivot_arr("is_st", stock_codes, dates, dtype=np.bool_)
    is_susp = (vol == 0)
    if "days_since_ipo" in df.columns:
        days = _pivot_arr("days_since_ipo", stock_codes, dates)
    else:
        days = np.full((n_dates, n_stocks), 365.0, dtype=np.float32)
    valid_basic = (~is_st) & (~is_susp) & (days >= 60)

    # Episode scan
    print("[factor_tmk] scanning episodes...")
    eps = find_main_wave_episodes(close, vol, valid_basic, EpisodeConfig())
    print(f"[factor_tmk] {len(eps):,} episodes")

    # Coords for each T-k offset, k=1..k_max
    coords_by_k: dict[int, list[tuple[int, int]]] = {}
    for k in range(1, args.k_max + 1):
        coords_by_k[k] = [
            (e.t_start - k, e.stock_idx) for e in eps if e.t_start - k >= 0
        ]
    print("[factor_tmk] coords ready: " + ", ".join(
        f"T-{k}={len(v)}" for k, v in coords_by_k.items()
    ))

    # Find factors matching prefixes
    factor_cols = [c for c in df.columns if any(c.startswith(p) for p in args.prefixes)]
    print(f"[factor_tmk] {len(factor_cols)} factor cols to analyse")

    # Per-factor analysis: for each factor, compute cross-section z then
    # sample at each T-k coord set
    rows: list[dict] = []
    for i, col in enumerate(factor_cols):
        if i % 20 == 0:
            print(f"[factor_tmk] {i}/{len(factor_cols)}: {col}")
        try:
            arr = _pivot_arr(col, stock_codes, dates)
        except Exception:
            continue
        # cross-section z per date
        with np.errstate(invalid="ignore"):
            mu = np.nanmean(arr, axis=1, keepdims=True)
            sigma = np.nanstd(arr, axis=1, keepdims=True)
            sigma_safe = np.where(sigma > 1e-9, sigma, 1.0)
            z = (arr - mu) / sigma_safe
        z = np.where(np.isfinite(z), z, 0.0)

        record: dict = {"factor": col}
        for k in range(1, args.k_max + 1):
            zs = np.array([z[t, j] for t, j in coords_by_k[k]
                           if 0 <= t < n_dates and 0 <= j < n_stocks])
            zs = zs[np.isfinite(zs)]
            if len(zs) == 0:
                record[f"mean_z_T-{k}"] = float("nan")
                record[f"median_z_T-{k}"] = float("nan")
            else:
                record[f"mean_z_T-{k}"] = float(zs.mean())
                record[f"median_z_T-{k}"] = float(np.median(zs))
        rows.append(record)

    # Sort by |T-1 mean_z|
    rows.sort(key=lambda r: abs(r.get("mean_z_T-1", 0.0)), reverse=True)

    # Output markdown
    lines = ["# Phase 23 — factor signal at T-k of main-wave episodes",
             "",
             f"- Window: {start} .. {end}",
             f"- Episodes: {len(eps):,}",
             f"- Factor prefixes analysed: {args.prefixes}",
             f"- N factors: {len(rows)}",
             "",
             "Cross-section z-score at T-k of episodes (universe baseline = 0).",
             "Mean over all valid episode coords; positive = factor higher than peers.",
             ""]
    cols_header = ["factor"] + [
        f"mean@T-{k}" for k in range(1, args.k_max + 1)
    ] + [f"median@T-{k}" for k in range(1, args.k_max + 1)]
    lines.append("| " + " | ".join(cols_header) + " |")
    lines.append("| " + " | ".join("---" for _ in cols_header) + " |")
    for r in rows:
        cells = [r["factor"]]
        for k in range(1, args.k_max + 1):
            v = r.get(f"mean_z_T-{k}", float("nan"))
            cells.append(f"{v:+.3f}" if np.isfinite(v) else "nan")
        for k in range(1, args.k_max + 1):
            v = r.get(f"median_z_T-{k}", float("nan"))
            cells.append(f"{v:+.3f}" if np.isfinite(v) else "nan")
        lines.append("| " + " | ".join(cells) + " |")

    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[factor_tmk] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
