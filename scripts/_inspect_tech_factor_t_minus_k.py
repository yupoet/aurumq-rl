#!/usr/bin/env python3
"""Phase 25 helper — compute T-k z-score for the 38 technical factors.

These factors are computed at panel-load time, not present in the
parquet, so the standard inspect script can't see them. This helper
constructs them from raw close/vol/pct_chg, applies cross-section
z-score per date, and samples at T-k episode coords. Output appended
to the existing factor_t_minus_k_ALL355.md as additional rows.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np
import polars as pl

from aurumq_rl.main_wave_episodes import EpisodeConfig, find_main_wave_episodes
from aurumq_rl.technical_factors import compute_technical_factors


DATA_PATH = Path("data/factor_panel_combined_short_2023_2026.parquet")
OUT_PATH = Path("runs/_episode_inspect/tech_factor_t_minus_k.json")


def main() -> int:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("[tech-tk] reading parquet...")
    df = pl.read_parquet(DATA_PATH).filter(
        (pl.col("trade_date") >= dt.date(2023, 1, 3))
        & (pl.col("trade_date") <= dt.date(2026, 4, 24))
    )

    def _pivot(field: str, dtype=np.float32) -> tuple[pl.DataFrame, np.ndarray]:
        piv = (df.select(["trade_date", "ts_code", field])
                 .pivot(values=field, index="trade_date", on="ts_code")
                 .sort("trade_date"))
        codes = [c for c in piv.columns if c != "trade_date"]
        arrs = []
        for code in codes:
            ser = piv.get_column(code)
            if ser.dtype == pl.Boolean:
                col = ser.fill_null(False).to_numpy().astype(np.bool_)
            else:
                col = ser.fill_null(0.0).to_numpy()
            arrs.append(col.astype(dtype, copy=False))
        return piv, np.stack(arrs, axis=1)

    close_piv, close = _pivot("close")
    _, vol = _pivot("vol")
    _, pct = _pivot("pct_chg")
    is_st_piv, is_st = _pivot("is_st", dtype=np.bool_)
    is_susp = (vol == 0)
    if "days_since_ipo" in df.columns:
        _, days = _pivot("days_since_ipo")
    else:
        days = np.full_like(close, 365.0)
    dates = close_piv.get_column("trade_date").to_list()
    stock_codes = [c for c in close_piv.columns if c != "trade_date"]
    valid_basic = (~is_st) & (~is_susp) & (days >= 60)

    # Pull mf_net_1d for cmf
    if "mf_net_1d" in df.columns:
        _, mf1 = _pivot("mf_net_1d")
    else:
        mf1 = None

    print(f"[tech-tk] panel: dates={len(dates)} stocks={len(stock_codes)}")
    print("[tech-tk] computing tech factors...")
    tech = compute_technical_factors(close, vol, pct, mf_net_1d=mf1)
    tech_names = list(tech.keys())
    print(f"[tech-tk] computed {len(tech_names)} tech factors")

    print("[tech-tk] scanning episodes...")
    eps = find_main_wave_episodes(close, vol, valid_basic, EpisodeConfig())
    print(f"[tech-tk] {len(eps)} episodes")

    coords_by_k = {
        k: [(e.t_start - k, e.stock_idx) for e in eps if e.t_start - k >= 0]
        for k in range(1, 6)
    }
    print(f"[tech-tk] T-1 coords: {len(coords_by_k[1])}")

    rows = {}
    for name in tech_names:
        arr = tech[name]
        # Cross-section z-score per date
        with np.errstate(invalid="ignore"):
            mu = np.nanmean(arr, axis=1, keepdims=True)
            sigma = np.nanstd(arr, axis=1, keepdims=True)
            sigma_safe = np.where(sigma > 1e-9, sigma, 1.0)
            z = (arr - mu) / sigma_safe
        z = np.where(np.isfinite(z), z, 0.0)
        record = {}
        for k in range(1, 6):
            zs = np.array([z[t, j] for t, j in coords_by_k[k]
                           if 0 <= t < z.shape[0] and 0 <= j < z.shape[1]])
            zs = zs[np.isfinite(zs)]
            if len(zs) == 0:
                record[f"mean@T-{k}"] = float("nan")
            else:
                record[f"mean@T-{k}"] = float(zs.mean())
        rows[name] = record

    # Sort by |T-1 z| and print top
    ranked = sorted(rows.items(), key=lambda kv: -abs(kv[1]["mean@T-1"]))
    print(f"\n[tech-tk] Top 20 by |T-1 z|:")
    for name, r in ranked[:20]:
        print(f"  {name:<28}  T-1={r['mean@T-1']:+.3f}  T-3={r['mean@T-3']:+.3f}  T-5={r['mean@T-5']:+.3f}")
    print(f"\n[tech-tk] Bottom 10:")
    for name, r in ranked[-10:]:
        print(f"  {name:<28}  T-1={r['mean@T-1']:+.3f}")

    OUT_PATH.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[tech-tk] wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
