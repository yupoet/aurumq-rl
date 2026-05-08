#!/usr/bin/env python3
"""Phase 23 — descriptive analytics on main-wave episodes.

Scans a parquet (training or OOS window), runs the episode detector, and
reports:

1. Episode catalog: count, peak_return / duration / max_dd distributions
2. Per-month: episodes started per month
3. Per-industry: episode density and avg peak_return by industry name
4. Factor profile at T-1 of episodes vs universe baseline (z-score deltas
   for mf_*, mfp_*, hk_*, inst_*, senti_* prefixes; top-N by absolute delta)
5. Sample episodes: 10 best by peak_return, with stock/date/duration

Output: ``runs/_episode_inspect/<window>.md`` plus optional JSON.

Embeds the user's "找规律" step inside Phase 23 (A). Run BEFORE training to
verify the scanner finds enough episodes to make a target signal.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np
import polars as pl

from aurumq_rl.main_wave_episodes import (
    EpisodeConfig,
    find_main_wave_episodes,
    episodes_summary,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-path", type=Path, required=True)
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--out-dir", type=Path, default=Path("runs/_episode_inspect"))
    p.add_argument("--label", default="window",
                   help="Label appended to output filenames")
    p.add_argument("--top-n-factors", type=int, default=20)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    start = dt.date.fromisoformat(args.start_date)
    end = dt.date.fromisoformat(args.end_date)
    print(f"[inspect_episodes] reading {args.data_path}, window {start}..{end}")
    df_full = pl.read_parquet(args.data_path)
    df = df_full.filter(
        (pl.col("trade_date") >= start) & (pl.col("trade_date") <= end)
    )
    if len(df) == 0:
        print(f"[err] no rows in window")
        return 2

    # Pivot close + vol
    def _pivot(field: str) -> tuple[pl.DataFrame, list[str]]:
        piv = (df.select(["trade_date", "ts_code", field])
                 .pivot(values=field, index="trade_date", on="ts_code")
                 .sort("trade_date"))
        codes = [c for c in piv.columns if c != "trade_date"]
        return piv, codes

    close_piv, close_codes = _pivot("close")
    # Use stock_codes present in the close pivot as the universe
    stock_codes = close_codes
    n_stocks = len(stock_codes)
    dates = close_piv.get_column("trade_date").to_list()
    n_dates = len(dates)
    print(f"[inspect_episodes] panel: dates={n_dates} stocks={n_stocks}")

    def _to_arr(piv: pl.DataFrame, dtype=np.float32) -> np.ndarray:
        cols = [c for c in piv.columns if c != "trade_date"]
        col_set = set(cols)
        arrs: list[np.ndarray] = []
        for code in stock_codes:
            if code in col_set:
                series = piv.get_column(code)
                # bool series can't fill_null with float; cast first
                if series.dtype == pl.Boolean:
                    col = series.fill_null(False).to_numpy().astype(np.bool_)
                else:
                    col = series.fill_null(0.0).to_numpy()
            else:
                if dtype == np.bool_:
                    col = np.zeros(n_dates, dtype=np.bool_)
                else:
                    col = np.zeros(n_dates, dtype=np.float32)
            if dtype == np.bool_:
                arrs.append(col.astype(np.bool_, copy=False))
            else:
                arrs.append(col.astype(np.float32, copy=False))
        return np.stack(arrs, axis=1)

    close_arr = _to_arr(close_piv)
    vol_piv, _ = _pivot("vol")
    vol_arr = _to_arr(vol_piv)

    # is_st pivot (default False if missing)
    is_st_piv, _ = _pivot("is_st")
    is_st_arr = _to_arr(is_st_piv, dtype=np.bool_)

    # is_suspended — vol == 0 proxy
    is_suspended_arr = (vol_arr == 0)

    # days_since_ipo pivot — fill with high value where missing
    if "days_since_ipo" in df.columns:
        days_piv, _ = _pivot("days_since_ipo")
        days_arr = _to_arr(days_piv)
    else:
        days_arr = np.full((n_dates, n_stocks), 365.0, dtype=np.float32)

    valid_basic = (~is_st_arr) & (~is_suspended_arr) & (days_arr >= 60)
    print(f"[inspect_episodes] valid_basic cells: {int(valid_basic.sum()):,} / "
          f"{n_dates * n_stocks:,} ({100.0 * valid_basic.sum() / (n_dates * n_stocks):.1f}%)")

    # ---- Run episode detector ----
    cfg = EpisodeConfig()
    print(f"[inspect_episodes] scanning episodes (peak ≥ {cfg.min_peak_return}, "
          f"duration ∈ [{cfg.min_duration}, {cfg.max_duration}], "
          f"min_avg_daily ≥ {cfg.min_avg_daily_return}, "
          f"amount_ma_min={cfg.amount_ma_min:.0e})...")
    episodes = find_main_wave_episodes(close_arr, vol_arr, valid_basic, cfg)
    summary = episodes_summary(episodes)
    print(f"[inspect_episodes] found {len(episodes):,} episodes")
    print(f"  avg peak_return = {summary['avg_peak_return']:+.4f}")
    print(f"  median peak_return = {summary['median_peak_return']:+.4f}")
    print(f"  p90 peak_return = {summary['p90_peak_return']:+.4f}")
    print(f"  max peak_return = {summary['max_peak_return']:+.4f}")
    print(f"  avg duration = {summary['avg_duration']:.2f}")
    print(f"  avg max_dd_during = {summary['avg_max_dd']:.4f}")

    if not episodes:
        print(f"[inspect_episodes] no episodes — adjust thresholds or check panel data")
        return 1

    # ---- 2. Per-month ----
    month_count: dict[str, int] = defaultdict(int)
    month_peak: dict[str, list[float]] = defaultdict(list)
    for e in episodes:
        ym = dates[e.t_start].strftime("%Y-%m")
        month_count[ym] += 1
        month_peak[ym].append(e.peak_return)

    # ---- 3. Per-industry ----
    industry_lookup: dict[str, str] = {}
    if "industry_code" in df_full.columns:
        ind_df = (df_full.select(["ts_code", "industry_code"]).unique())
        for row in ind_df.iter_rows(named=True):
            industry_lookup[row["ts_code"]] = (
                str(row["industry_code"]) if row["industry_code"] is not None else "?"
            )

    industry_counts: dict[str, int] = defaultdict(int)
    industry_peaks: dict[str, list[float]] = defaultdict(list)
    industry_stock_counts: dict[str, set] = defaultdict(set)
    for e in episodes:
        code = stock_codes[e.stock_idx]
        ind = industry_lookup.get(code, "?")
        industry_counts[ind] += 1
        industry_peaks[ind].append(e.peak_return)
        industry_stock_counts[ind].add(code)

    # ---- 4. Factor profile at T-1 ----
    # T-1 = t_start - 1. For each factor column we have, compare its
    # value at (t_start - 1, stock_idx) across episodes vs at random
    # (universe-wide same-day baseline).
    factor_prefixes = ("mf_", "mfp_", "hk_", "inst_", "senti_", "cyq_",
                       "fund_", "ind_", "hm_", "sh_", "mg_", "alpha_", "gtja_")
    factor_cols = [c for c in df.columns if c.startswith(factor_prefixes)]
    print(f"[inspect_episodes] {len(factor_cols)} factor cols")

    # Build a (date, stock) → row_idx map for fast lookup
    factor_z_deltas: list[tuple[str, float, float, float]] = []
    # We compute z-score across the universe per date, then look at the
    # average z at T-1 of episodes vs 0 (universe baseline by definition).
    # To make this efficient we pivot each factor once.

    # Pre-compute T-1 (t_start - 1) coordinates of episodes
    t_minus_1_coords = [(e.t_start - 1, e.stock_idx) for e in episodes if e.t_start >= 1]
    n_episodes_with_tm1 = len(t_minus_1_coords)
    print(f"[inspect_episodes] computing factor z-score deltas at T-1 "
          f"({n_episodes_with_tm1} episode points)...")

    # Limit to a manageable subset of factors for compute (top by data presence).
    # Sample 60 factors total across prefixes if the full set is too many.
    if len(factor_cols) > 60:
        # take the first 5 from each prefix to keep balanced coverage
        by_prefix: dict[str, list[str]] = defaultdict(list)
        for c in factor_cols:
            for p in factor_prefixes:
                if c.startswith(p):
                    by_prefix[p].append(c)
                    break
        sample = []
        for p, cols in by_prefix.items():
            sample.extend(cols[:5])
        factor_cols = sample
        print(f"[inspect_episodes] sampled to {len(factor_cols)} factor cols for analysis")

    for col in factor_cols:
        try:
            piv, _ = _pivot(col)
            arr = _to_arr(piv)   # (n_dates, n_stocks)
        except Exception:
            continue
        # cross-section z-score per date
        # avoid division by zero where all-stocks-equal
        mu = arr.mean(axis=1, keepdims=True)
        sigma = arr.std(axis=1, keepdims=True)
        sigma_safe = np.where(sigma > 1e-9, sigma, 1.0)
        z = (arr - mu) / sigma_safe   # (T, S)
        # Sample z at T-1 coords
        zs = np.array([z[t, j] for t, j in t_minus_1_coords if 0 <= t < n_dates and 0 <= j < n_stocks])
        zs = zs[np.isfinite(zs)]
        if len(zs) == 0:
            continue
        mean_z_at_tm1 = float(zs.mean())
        median_z_at_tm1 = float(np.median(zs))
        # Effect size (sign + magnitude)
        effect = mean_z_at_tm1
        factor_z_deltas.append((col, effect, median_z_at_tm1, len(zs)))

    factor_z_deltas.sort(key=lambda x: abs(x[1]), reverse=True)
    top_factors = factor_z_deltas[:args.top_n_factors]

    # ---- Output ----
    out_md_path = args.out_dir / f"episodes_{args.label}.md"
    out_json_path = args.out_dir / f"episodes_{args.label}.json"

    lines: list[str] = []
    def _e(s: str) -> None:
        lines.append(s)

    _e(f"# Phase 23 episode inspection — {args.label}")
    _e("")
    _e(f"- Window: {start} .. {end}")
    _e(f"- Panel: dates={n_dates}, stocks={n_stocks}")
    _e(f"- Config: peak_return ≥ {cfg.min_peak_return}, duration ∈ "
       f"[{cfg.min_duration}, {cfg.max_duration}], avg_daily ≥ "
       f"{cfg.min_avg_daily_return}, dd_allowance = "
       f"{cfg.base_dd_allowance} + {cfg.dd_per_peak} × peak, "
       f"amount_ma_min = {cfg.amount_ma_min:.0e}")
    _e("")
    _e("## 1. Catalog")
    for k, v in summary.items():
        _e(f"- {k}: {v}")
    _e("")
    _e("## 2. Per-month")
    _e("| month | n_episodes | avg peak_return | median peak_return |")
    _e("|---|---:|---:|---:|")
    for ym in sorted(month_count):
        n = month_count[ym]
        peaks = month_peak[ym]
        _e(f"| {ym} | {n} | {float(np.mean(peaks)):+.4f} | "
           f"{float(np.median(peaks)):+.4f} |")
    _e("")
    _e("## 3. Per-industry (top 20 by episode count)")
    _e("| industry | n_episodes | n_stocks | avg peak_return | median peak_return |")
    _e("|---|---:|---:|---:|---:|")
    for ind, n in sorted(industry_counts.items(), key=lambda kv: -kv[1])[:20]:
        peaks = industry_peaks[ind]
        _e(f"| {ind} | {n} | {len(industry_stock_counts[ind])} | "
           f"{float(np.mean(peaks)):+.4f} | {float(np.median(peaks)):+.4f} |")
    _e("")
    _e("## 4. Factor profile at T-1 (top {} by |mean_z|)".format(args.top_n_factors))
    _e("Mean cross-section z-score at T-1 of episodes. "
       "Universe baseline = 0; positive = factor higher than peers on T-1.")
    _e("")
    _e("| factor | mean_z@T-1 | median_z@T-1 | n_samples |")
    _e("|---|---:|---:|---:|")
    for col, mean_z, median_z, n in top_factors:
        _e(f"| {col} | {mean_z:+.3f} | {median_z:+.3f} | {n} |")
    _e("")
    _e("## 5. Sample episodes")
    _e("### Top 10 by peak_return")
    eps_sorted = sorted(episodes, key=lambda e: -e.peak_return)[:10]
    _e("| t_start_date | stock_code | industry | peak_return | duration | max_dd |")
    _e("|---|---|---|---:|---:|---:|")
    for e in eps_sorted:
        code = stock_codes[e.stock_idx]
        ind = industry_lookup.get(code, "?")
        _e(f"| {dates[e.t_start]} | {code} | {ind} | "
           f"{e.peak_return:+.4f} | {e.duration} | {e.max_dd_during:.4f} |")
    _e("")
    _e("### Median 5 (around the centre of distribution)")
    eps_sorted_pr = sorted(episodes, key=lambda e: e.peak_return)
    n = len(eps_sorted_pr)
    median_slice = eps_sorted_pr[max(0, n // 2 - 2): n // 2 + 3]
    _e("| t_start_date | stock_code | industry | peak_return | duration | max_dd |")
    _e("|---|---|---|---:|---:|---:|")
    for e in median_slice:
        code = stock_codes[e.stock_idx]
        ind = industry_lookup.get(code, "?")
        _e(f"| {dates[e.t_start]} | {code} | {ind} | "
           f"{e.peak_return:+.4f} | {e.duration} | {e.max_dd_during:.4f} |")

    out_md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[inspect_episodes] wrote {out_md_path}")

    # JSON: episodes list + factor deltas + summary
    out_json = {
        "config": cfg.__dict__,
        "data_path": str(args.data_path),
        "start": start.isoformat(), "end": end.isoformat(),
        "n_dates": n_dates, "n_stocks": n_stocks,
        "summary": summary,
        "month_count": dict(month_count),
        "industry_top20": [
            {"industry": ind, "n_episodes": n,
             "n_stocks": len(industry_stock_counts[ind]),
             "avg_peak_return": float(np.mean(industry_peaks[ind])),
             "median_peak_return": float(np.median(industry_peaks[ind]))}
            for ind, n in sorted(industry_counts.items(), key=lambda kv: -kv[1])[:20]
        ],
        "factor_top_deltas": [
            {"factor": c, "mean_z_at_tm1": z, "median_z_at_tm1": mz, "n_samples": n}
            for c, z, mz, n in top_factors
        ],
        "n_episodes": len(episodes),
    }
    out_json_path.write_text(
        json.dumps(out_json, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"[inspect_episodes] wrote {out_json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
