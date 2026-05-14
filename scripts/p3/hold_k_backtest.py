"""Hold-K backtest — pair each entry signal with hypothetical fixed-K-day hold exit.

For each (model, K∈{1,3,5,10,20,40}, slice):
  - top-50 picked each day by model's score_calibrated
  - hold for K days, realize cumulative excess return (compounded vs market)
  - average per-pick → mean realized for that (model, K)
  - hit rate = fraction of (date, stock) pairs with realized > 0

Output:
  - runs/hold_k_backtest/results.json
  - runs/hold_k_backtest/RESULTS.md  with one heat table per slice
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import polars as pl


BUNDLE = Path("data/p3_4070")
OUT = Path("runs/hold_k_backtest")


SLICES = {
    "VAL":              (dt.date(2025, 1, 1),  dt.date(2025, 6, 4)),
    "H1":               (dt.date(2025, 7, 1),  dt.date(2025, 9, 30)),
    "H2":               (dt.date(2025, 10, 1), dt.date(2025, 12, 31)),
    "2026-Q1":          (dt.date(2026, 1, 1),  dt.date(2026, 3, 31)),
    "2026-Q2-partial":  (dt.date(2026, 4, 1),  dt.date(2026, 5, 11)),
}


PATHS = [
    ("Path 1 long",         "runs/sl_path1_long/predictions.parquet",             "score_calibrated"),
    ("Path 4 short (prod)", "runs/sl_path4/predictions.parquet",                  "score_calibrated"),
    ("Path 4 long",         "runs/sl_path_d/predictions.parquet",                 "score_calibrated"),
    ("Path 2 long",         "runs/sl_path2_long/predictions.parquet",             "score_calibrated"),
    ("Path 5 long",         "runs/sl_regime_stack_long/predictions.parquet",      "score_calibrated"),
    ("Hybrid p1L+p4S",      "runs/sl_hybrid_p1long_p4short/predictions.parquet",  "score"),
    ("Wave v1",             "runs/sl_path1_long_wave_v1/predictions.parquet",     "score_calibrated"),
]

KS = (1, 3, 5, 10, 20, 40)


def build_fwd_excess() -> pl.DataFrame:
    """Per-(date, stock) fwd-K cumulative excess for K∈KS."""
    realized = pl.read_parquet(BUNDLE / "realized_returns.parquet").select(
        ["trade_date", "ts_code", "pct_chg_t_plus_1"]
    ).sort(["ts_code", "trade_date"])
    market = pl.read_parquet(BUNDLE / "market_returns.parquet").select(
        ["trade_date", "eq_weight_pct_chg_t_plus_1"]
    ).sort("trade_date")
    for K in KS:
        realized = realized.with_columns(
            pl.col("pct_chg_t_plus_1").log1p()
              .rolling_sum(window_size=K)
              .shift(-(K - 1))
              .exp().sub(1.0)
              .over("ts_code")
              .alias(f"fwd_{K}d")
        )
        market = market.with_columns(
            pl.col("eq_weight_pct_chg_t_plus_1").log1p()
              .rolling_sum(window_size=K)
              .shift(-(K - 1))
              .exp().sub(1.0)
              .alias(f"mkt_fwd_{K}d")
        )
    df = realized.join(market.select(["trade_date"] + [f"mkt_fwd_{K}d" for K in KS]),
                       on="trade_date", how="left")
    for K in KS:
        df = df.with_columns((pl.col(f"fwd_{K}d") - pl.col(f"mkt_fwd_{K}d")).alias(f"excess_{K}d"))
    return df.select(["trade_date", "ts_code"] + [f"excess_{K}d" for K in KS])


def evaluate(preds: pl.DataFrame, score_col: str, fwd: pl.DataFrame, time_start, time_end, top_k=50):
    joined = preds.select(["trade_date", "ts_code", pl.col(score_col).alias("score")]).join(
        fwd, on=["trade_date", "ts_code"], how="inner"
    ).filter((pl.col("trade_date") >= time_start) & (pl.col("trade_date") <= time_end))
    if len(joined) == 0:
        return {K: None for K in KS}
    top = joined.sort(["trade_date", "score"], descending=[False, True]) \
                .group_by("trade_date", maintain_order=True).head(top_k)
    out = {}
    for K in KS:
        col = f"excess_{K}d"
        # Daily mean of top-50's fwd-K excess, then mean across days
        daily = top.group_by("trade_date").agg(pl.col(col).mean().alias("daily_mean"))
        # Drop None rows from daily_mean (slice has no valid forward window)
        daily_clean = daily.drop_nulls("daily_mean")
        if len(daily_clean) == 0:
            out[K] = None
            continue
        mean_val = daily_clean["daily_mean"].mean()
        hit_val = (top[col] > 0).cast(pl.Float64).mean()
        out[K] = {
            "mean_excess": float(mean_val) if mean_val is not None else None,
            "hit_rate":    float(hit_val) if hit_val is not None else None,
            "n_days":      len(daily_clean),
        }
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("[build] fwd K-d excess for K∈{1,3,5,10,20,40}")
    fwd = build_fwd_excess()
    print(f"  rows: {len(fwd):,}")

    results = {}
    for name, path, score_col in PATHS:
        p = Path(path)
        if not p.exists():
            print(f"[skip] {name}")
            continue
        print(f"[eval] {name}")
        preds = pl.read_parquet(p)
        results[name] = {}
        for sname, (ts, te) in SLICES.items():
            results[name][sname] = evaluate(preds, score_col, fwd, ts, te)
            row = results[name][sname]
            parts = []
            for K in KS:
                v = row.get(K)
                if v is not None and v.get("mean_excess") is not None:
                    parts.append(f"K={K}: {v['mean_excess']:+.5f}")
                else:
                    parts.append(f"K={K}: —")
            print(f"  {sname:18}  " + "  ".join(parts))

    (OUT / "results.json").write_text(json.dumps(results, indent=2, default=str))

    # Markdown report
    md = [
        "# Hold-K Backtest — Realized fwd-K Excess by Model × Slice × K",
        f"Generated: {dt.datetime.now().isoformat()}",
        "",
        "For each (model, K, slice): pick top-50 by score_calibrated daily, compute",
        "compounded excess return over forward K days, average across days.",
        "",
        "Best-K identifies each model's optimal hold horizon.",
        "",
    ]
    for sname in SLICES:
        md.append(f"## {sname}")
        md.append("")
        md.append("| Path | " + " | ".join(f"K={K}" for K in KS) + " | best K |")
        md.append("|---" + "|---:" * len(KS) + "|---:|")
        for name in results:
            row = [name]
            best_k = None
            best_v = -np.inf
            for K in KS:
                v = results[name].get(sname, {}).get(K)
                if v is not None and v.get("mean_excess") is not None:
                    me = v["mean_excess"]
                    row.append(f"{me:+.5f}")
                    if me > best_v:
                        best_v = me
                        best_k = K
                else:
                    row.append("—")
            row.append(f"K={best_k}" if best_k else "—")
            md.append("| " + " | ".join(row) + " |")
        md.append("")

    # Cross-model "best K" summary
    md.append("## Best-K summary (which K maximizes each model)")
    md.append("")
    md.append("| Path | " + " | ".join(SLICES.keys()) + " |")
    md.append("|---" + "|---:" * len(SLICES) + "|")
    for name in results:
        row = [name]
        for sname in SLICES:
            best_v = -np.inf
            best_k = None
            for K in KS:
                v = results[name].get(sname, {}).get(K)
                if v is not None and v.get("mean_excess") is not None and v["mean_excess"] > best_v:
                    best_v = v["mean_excess"]
                    best_k = K
            row.append(f"K={best_k}" if best_k else "—")
        md.append("| " + " | ".join(row) + " |")
    md.append("")
    md.append("**Interpretation**: a model's `best K` per slice tells you the optimal")
    md.append("hold horizon. If `best K = 1` → use as next-day signal. If `best K = 20+`")
    md.append("→ pair with swing-trade exit. **Mismatched K is value destruction**.")

    (OUT / "RESULTS.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nwrote {OUT / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
