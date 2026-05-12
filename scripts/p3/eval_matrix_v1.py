"""Time-slice evaluation matrix v1.

For each (path, time-slice) combination, compute:
  - mean_y (proximity-weighted excess, primary metric)
  - top50 mean fwd_1d / fwd_3d / fwd_5d excess return
  - T1_hit / T3_hit / T5_hit (fraction of top-50 with positive excess at horizon)
  - max_drawdown of the daily top-50 mean P&L
  - turnover (1 - jaccard between consecutive days' top-50)

Time slices (limited by realized_returns range = 2023-01-03 ~ 2025-12-31):
  - VAL: 2025-01-01 ~ 2025-06-04   (early-stopping window)
  - H1:  2025-07-01 ~ 2025-09-30   (isotonic calibration window)
  - H2:  2025-10-01 ~ 2025-12-31   (clean holdout)

Paths evaluated (using predictions.parquet):
  - Path 1 short / long
  - Path 4 short / long (= Path D)
  - Path 2 short / long
  - Path 5 long (regime stacking)
  - Hybrid (Path 1 long + Path 4 short 50/50)
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import polars as pl


BUNDLE = Path("data/p3_4070")
OUT_ROOT = Path("runs/eval_matrix_v1")


SLICES = {
    "VAL":  (dt.date(2025, 1, 1),  dt.date(2025, 6, 4)),
    "H1":   (dt.date(2025, 7, 1),  dt.date(2025, 9, 30)),
    "H2":   (dt.date(2025, 10, 1), dt.date(2025, 12, 31)),
}


PATHS = [
    ("Path 1 short",      "runs/sl_path1/predictions.parquet",                "score_calibrated"),
    ("Path 1 long",       "runs/sl_path1_long/predictions.parquet",           "score_calibrated"),
    ("Path 4 short (prod)", "runs/sl_path4/predictions.parquet",              "score_calibrated"),
    ("Path 4 long",       "runs/sl_path_d/predictions.parquet",               "score_calibrated"),
    ("Path 2 short",      "runs/sl_path2/predictions.parquet",                "score_calibrated"),
    ("Path 2 long",       "runs/sl_path2_long/predictions.parquet",           "score_calibrated"),
    ("Path 5 long",       "runs/sl_regime_stack_long/predictions.parquet",    "score_calibrated"),
    ("Hybrid p1L+p4S",    "runs/sl_hybrid_p1long_p4short/predictions.parquet","score"),
]


def build_forward_returns(realized: pl.DataFrame, market: pl.DataFrame) -> pl.DataFrame:
    """Build per-(date, stock) forward-K excess returns for K=1,3,5.

    realized has pct_chg_t_plus_1 (close_t -> close_t+1). To get fwd_K starting
    at date T, sum log(1 + pct_chg_t_plus_1) over rows for days T, T+1, ..., T+K-1.
    """
    r = realized.select(
        ["trade_date", "ts_code", "pct_chg_t_plus_1"]
    ).sort(["ts_code", "trade_date"])

    # Per-stock rolling forward log returns
    for K in (1, 3, 5):
        r = r.with_columns(
            pl.col("pct_chg_t_plus_1").log1p()
              .rolling_sum(window_size=K)
              .shift(-(K - 1))
              .exp().sub(1.0)
              .over("ts_code")
              .alias(f"fwd_{K}d")
        )

    # Market forward returns (universe eq-weight)
    m = market.select(["trade_date", "eq_weight_pct_chg_t_plus_1"]).sort("trade_date")
    for K in (1, 3, 5):
        m = m.with_columns(
            pl.col("eq_weight_pct_chg_t_plus_1").log1p()
              .rolling_sum(window_size=K)
              .shift(-(K - 1))
              .exp().sub(1.0)
              .alias(f"mkt_fwd_{K}d")
        )
    m = m.select(["trade_date", "mkt_fwd_1d", "mkt_fwd_3d", "mkt_fwd_5d"])

    df = r.join(m, on="trade_date", how="left")
    # Excess returns
    for K in (1, 3, 5):
        df = df.with_columns((pl.col(f"fwd_{K}d") - pl.col(f"mkt_fwd_{K}d")).alias(f"excess_{K}d"))
    return df.select(
        ["trade_date", "ts_code",
         "excess_1d", "excess_3d", "excess_5d",
         "fwd_1d", "mkt_fwd_1d"]
    )


def evaluate_slice(
    preds: pl.DataFrame,
    score_col: str,
    fwd: pl.DataFrame,
    target_y: pl.DataFrame,
    start: dt.date,
    end: dt.date,
    top_k: int = 50,
) -> dict:
    """Compute metrics for one (path, slice) pair."""
    joined = (
        preds.select(["trade_date", "ts_code", pl.col(score_col).alias("score")])
        .join(fwd, on=["trade_date", "ts_code"], how="inner")
        .join(target_y.select(["trade_date", "ts_code", "y"]), on=["trade_date", "ts_code"], how="inner")
        .filter((pl.col("trade_date") >= start) & (pl.col("trade_date") <= end))
    )
    if len(joined) == 0:
        return {"n_days": 0}

    # Per-day top-K
    top = (
        joined.sort(["trade_date", "score"], descending=[False, True])
        .group_by("trade_date", maintain_order=True)
        .head(top_k)
    )

    # Daily aggregates
    daily = top.group_by("trade_date").agg(
        pl.col("y").mean().alias("daily_mean_y"),
        pl.col("excess_1d").mean().alias("daily_top_excess_1d"),
        pl.col("excess_3d").mean().alias("daily_top_excess_3d"),
        pl.col("excess_5d").mean().alias("daily_top_excess_5d"),
        # T1_hit / T3_hit / T5_hit: fraction of selected with positive excess
        (pl.col("excess_1d") > 0).cast(pl.Float64).mean().alias("daily_T1_hit"),
        (pl.col("excess_3d") > 0).cast(pl.Float64).mean().alias("daily_T3_hit"),
        (pl.col("excess_5d") > 0).cast(pl.Float64).mean().alias("daily_T5_hit"),
        pl.col("ts_code").alias("picks"),
    ).sort("trade_date")
    n_days = len(daily)

    # Turnover: 1 - |today ∩ yesterday| / K, averaged
    picks_lists = daily["picks"].to_list()
    overlaps = []
    for i in range(1, len(picks_lists)):
        a, b = set(picks_lists[i - 1]), set(picks_lists[i])
        overlaps.append(len(a & b) / top_k)
    turnover = 1.0 - float(np.mean(overlaps)) if overlaps else float("nan")

    # Max drawdown — use REALIZED top-50 1d excess as daily P&L
    # (NOT daily_mean_y which is clipped to >=0 by proximity-weighted target).
    daily_realized = daily["daily_top_excess_1d"].to_numpy()
    cum = np.cumprod(1.0 + daily_realized)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    max_drawdown = float(dd.min()) if len(dd) else float("nan")
    # Cumulative realized excess return over slice (compounded)
    cum_excess_total = float(cum[-1] - 1.0) if len(cum) else float("nan")

    return {
        "n_days": n_days,
        "mean_y":              float(daily["daily_mean_y"].mean()),
        "top50_excess_1d":     float(daily["daily_top_excess_1d"].mean()),
        "top50_excess_3d":     float(daily["daily_top_excess_3d"].mean()),
        "top50_excess_5d":     float(daily["daily_top_excess_5d"].mean()),
        "T1_hit":              float(daily["daily_T1_hit"].mean()),
        "T3_hit":              float(daily["daily_T3_hit"].mean()),
        "T5_hit":              float(daily["daily_T5_hit"].mean()),
        "turnover":            turnover,
        "max_drawdown":        max_drawdown,
        "cum_excess_total":    cum_excess_total,
    }


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"[load] realized + market + target_y from {BUNDLE}")
    realized = pl.read_parquet(BUNDLE / "realized_returns.parquet")
    market = pl.read_parquet(BUNDLE / "market_returns.parquet")
    target_y = pl.read_parquet(BUNDLE / "target_y.parquet")

    print(f"[build] forward 1d/3d/5d excess returns")
    fwd = build_forward_returns(realized, market)
    print(f"  fwd rows: {len(fwd):,}")
    print(f"  date range: {fwd['trade_date'].min()} ~ {fwd['trade_date'].max()}")

    results: dict[str, dict[str, dict]] = {}
    for name, path, score_col in PATHS:
        p = Path(path)
        if not p.exists():
            print(f"[skip] {name} — missing {p}")
            continue
        preds = pl.read_parquet(p)
        print(f"[eval] {name}")
        results[name] = {}
        for sname, (start, end) in SLICES.items():
            m = evaluate_slice(preds, score_col, fwd, target_y, start, end)
            results[name][sname] = m
            print(f"  {sname}: n={m['n_days']}  mean_y={m.get('mean_y',0):+.5f}  "
                  f"T1={m.get('T1_hit',0):.3f}  T3={m.get('T3_hit',0):.3f}  "
                  f"top50_3d={m.get('top50_excess_3d',0):+.5f}  "
                  f"DD={m.get('max_drawdown',0):.3f}  turnover={m.get('turnover',0):.3f}")

    # JSON for downstream consumption
    (OUT_ROOT / "results.json").write_text(json.dumps(results, indent=2, default=str))

    # Markdown report
    md = ["# Time-Slice Evaluation Matrix v1", f"Generated: {dt.datetime.now().isoformat()}", "",
          "Time slices (per `path1_eval.py` H1/H2 definitions):"]
    for s, (a, b) in SLICES.items():
        md.append(f"- **{s}**: {a} ~ {b}")
    md.append("")
    md.append("## Headline matrix — mean_y (proximity-weighted excess, ↑ better)\n")
    md.append("| Path | VAL | H1 | H2 |")
    md.append("|---|---:|---:|---:|")
    for name in results:
        row = [name]
        for s in ("VAL", "H1", "H2"):
            v = results[name].get(s, {}).get("mean_y")
            row.append(f"{v:+.5f}" if v is not None else "—")
        md.append("| " + " | ".join(row) + " |")
    md.append("")

    for metric, label, fmt in [
        ("top50_excess_1d",  "Top-50 fwd-1d realized excess (per-day mean)",  "+.5f"),
        ("top50_excess_3d",  "Top-50 fwd-3d realized excess",                 "+.5f"),
        ("top50_excess_5d",  "Top-50 fwd-5d realized excess",                 "+.5f"),
        ("T1_hit",           "T1 hit (fwd-1d excess > 0)",                    ".3f"),
        ("T3_hit",           "T3 hit (fwd-3d excess > 0)",                    ".3f"),
        ("T5_hit",           "T5 hit (fwd-5d excess > 0)",                    ".3f"),
        ("cum_excess_total", "Cumulative top-50 P&L over slice (compounded)", "+.4f"),
        ("max_drawdown",     "Max drawdown of top-50 P&L (← real money)",     ".4f"),
        ("turnover",         "Turnover (daily, 1 − jaccard@top50)",            ".3f"),
    ]:
        md.append(f"## {label}\n")
        md.append("| Path | VAL | H1 | H2 |")
        md.append("|---|---:|---:|---:|")
        for name in results:
            row = [name]
            for s in ("VAL", "H1", "H2"):
                v = results[name].get(s, {}).get(metric)
                row.append(format(v, fmt) if v is not None else "—")
            md.append("| " + " | ".join(row) + " |")
        md.append("")

    # Quick takeaway computation
    md.append("## Quick takeaways\n")
    # Compare each path's mean_y across slices
    for name in results:
        vals = {s: results[name].get(s, {}).get("mean_y") for s in ("VAL", "H1", "H2")}
        if all(v is not None for v in vals.values()):
            spread = max(vals.values()) - min(vals.values())
            md.append(f"- **{name}**: VAL→H1 {(vals['H1']-vals['VAL'])*1e4:+.1f} bps, "
                      f"H1→H2 {(vals['H2']-vals['H1'])*1e4:+.1f} bps, spread {spread*1e4:.1f} bps")

    (OUT_ROOT / "RESULTS.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nwrote {OUT_ROOT / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
