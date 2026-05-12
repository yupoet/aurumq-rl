"""Time-Slice Eval Matrix v2 — entry-only framing, multi-slice.

Per user (2026-05-13):
  - Entry-only model (exit handled separately) → DROP turnover, cumulative
    P&L, portfolio drawdown. KEEP per-entry metrics.
  - ADD size slices (HS300 / ZZ500 / CSI1000), industry-momentum regime
    slices, money-flow regime slices.
  - EXTEND time slices to 2026-Q1, 2026-Q2-partial.

Outputs:
  - runs/eval_matrix_v2/results.json   (full nested)
  - runs/eval_matrix_v2/RESULTS.md     (markdown report)
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl


BUNDLE = Path("data/p3_4070")
SLICES_V1 = Path("data/p3_4070_slices")          # constituent_calendar + industry_daily_agg
SLICES_V2 = Path("data/p3_4070_slices_v2")       # stock_industry_daily + limit_daily_agg + universe_money_agg
OUT_ROOT = Path("runs/eval_matrix_v2")


# Time slices (capped by realized = 2023-01-03 ~ 2026-05-11)
TIME_SLICES = {
    "VAL":           (dt.date(2025, 1, 1),  dt.date(2025, 6, 4)),
    "H1":            (dt.date(2025, 7, 1),  dt.date(2025, 9, 30)),
    "H2":            (dt.date(2025, 10, 1), dt.date(2025, 12, 31)),
    "2026-Q1":       (dt.date(2026, 1, 1),  dt.date(2026, 3, 31)),
    "2026-Q2-partial": (dt.date(2026, 4, 1), dt.date(2026, 5, 11)),
}


PATHS = [
    ("Path 1 short",        "runs/sl_path1/predictions.parquet",                  "score_calibrated"),
    ("Path 1 long",         "runs/sl_path1_long/predictions.parquet",             "score_calibrated"),
    ("Path 4 short (prod)", "runs/sl_path4/predictions.parquet",                  "score_calibrated"),
    ("Path 4 long",         "runs/sl_path_d/predictions.parquet",                 "score_calibrated"),
    ("Path 2 short",        "runs/sl_path2/predictions.parquet",                  "score_calibrated"),
    ("Path 2 long",         "runs/sl_path2_long/predictions.parquet",             "score_calibrated"),
    ("Path 5 long",         "runs/sl_regime_stack_long/predictions.parquet",      "score_calibrated"),
    ("Hybrid p1L+p4S",      "runs/sl_hybrid_p1long_p4short/predictions.parquet",  "score"),
]


def build_per_entry_metrics(realized: pl.DataFrame, market: pl.DataFrame) -> pl.DataFrame:
    """Compute per-(date, stock) forward-K excess + per-entry max DD/return over fwd-5d window.

    Returns columns:
      trade_date, ts_code,
      excess_1d, excess_3d, excess_5d         (cumulative excess at horizon K)
      entry_max_dd_5d, entry_max_ret_5d        (within-window max/min compounded path vs entry)
    """
    r = realized.select(["trade_date", "ts_code", "pct_chg_t_plus_1"]).sort(["ts_code", "trade_date"])

    # Per-stock daily log returns
    for K in (1, 3, 5):
        r = r.with_columns(
            pl.col("pct_chg_t_plus_1").log1p()
              .rolling_sum(window_size=K)
              .shift(-(K - 1))
              .exp().sub(1.0)
              .over("ts_code")
              .alias(f"fwd_{K}d")
        )

    # For per-entry max DD/return within 5-day window, we need the path of cumulative returns
    # for days 1..5 forward. Build columns ret_t_plus_1, ret_t_plus_2, ..., ret_t_plus_5
    # then compute cumulative product and find max/min.
    for k in range(1, 6):
        r = r.with_columns(
            pl.col("pct_chg_t_plus_1").shift(-(k - 1)).over("ts_code").alias(f"r_p{k}")
        )

    # Compounded path c_p1=1+r_p1, c_p2=c_p1*(1+r_p2), ..., c_p5
    r = r.with_columns([
        (1.0 + pl.col("r_p1")).alias("c_p1"),
    ])
    for k in range(2, 6):
        r = r.with_columns(
            (pl.col(f"c_p{k-1}") * (1.0 + pl.col(f"r_p{k}"))).alias(f"c_p{k}")
        )
    # max_ret_5d = max(c_p1..c_p5) - 1; max_dd_5d = min(c_p1..c_p5) - 1
    r = r.with_columns(
        pl.max_horizontal([pl.col(f"c_p{k}") for k in range(1, 6)]).sub(1.0).alias("entry_max_ret_5d"),
        pl.min_horizontal([pl.col(f"c_p{k}") for k in range(1, 6)]).sub(1.0).alias("entry_max_dd_5d"),
    ).drop([f"r_p{k}" for k in range(1, 6)] + [f"c_p{k}" for k in range(1, 6)])

    # Market fwd K
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
    for K in (1, 3, 5):
        df = df.with_columns((pl.col(f"fwd_{K}d") - pl.col(f"mkt_fwd_{K}d")).alias(f"excess_{K}d"))

    return df.select([
        "trade_date", "ts_code",
        "excess_1d", "excess_3d", "excess_5d",
        "entry_max_dd_5d", "entry_max_ret_5d",
    ])


def load_size_slice() -> pl.DataFrame:
    """Per (trade_date, ts_code) size membership flags."""
    df = pl.read_parquet(SLICES_V1 / "constituent_calendar.parquet").with_columns(
        pl.col("trade_date").cast(pl.Date)
    )
    # Define mutually exclusive size groups
    df = df.with_columns([
        pl.col("in_hs300").alias("size_large"),                                          # HS300
        (pl.col("in_zz500") & ~pl.col("in_hs300")).alias("size_mid"),                    # ZZ500 not HS300
        (pl.col("in_csi1000") & ~pl.col("in_hs300") & ~pl.col("in_zz500")).alias("size_small"),
    ])
    return df.select(["trade_date", "ts_code", "size_large", "size_mid", "size_small"])


def load_money_regime() -> pl.DataFrame:
    """Per trade_date: hot/cold regime by main_force_net_yi rolling-20d sign."""
    df = pl.read_parquet(SLICES_V2 / "universe_money_agg.parquet").with_columns(
        pl.col("trade_date").cast(pl.Date)
    ).select(["trade_date", "main_force_net_yi", "north_net_yi"])
    df = df.sort("trade_date").with_columns([
        pl.col("main_force_net_yi").rolling_mean(window_size=20).alias("mf_net_ma20"),
    ])
    # Hot = top 25% by mf_ma20; cold = bottom 25%
    q75 = df["mf_net_ma20"].quantile(0.75)
    q25 = df["mf_net_ma20"].quantile(0.25)
    df = df.with_columns([
        (pl.col("mf_net_ma20") >= q75).alias("mf_regime_hot"),
        (pl.col("mf_net_ma20") <= q25).alias("mf_regime_cold"),
    ])
    return df.select(["trade_date", "mf_regime_hot", "mf_regime_cold"])


def load_industry_regime() -> pl.DataFrame:
    """Per (trade_date, ts_code): is this stock in a hot/cold industry today?

    Hot industry: today's mean_pct_chg rolling 20d ranks in top-5 (out of 31 L1).
    """
    stock_ind = pl.read_parquet(SLICES_V2 / "stock_industry_daily.parquet").select(
        ["trade_date", "ts_code", "l1_code"]
    ).with_columns(pl.col("trade_date").cast(pl.Date))

    ind_agg = pl.read_parquet(SLICES_V1 / "industry_daily_agg.parquet").select(
        ["trade_date", "l1_code", "mean_pct_chg"]
    ).with_columns(pl.col("trade_date").cast(pl.Date))

    # Industry-level 20d momentum + per-day rank
    ind_agg = ind_agg.sort(["l1_code", "trade_date"]).with_columns(
        pl.col("mean_pct_chg").rolling_mean(window_size=20).over("l1_code").alias("mom_20d")
    )
    ind_agg = ind_agg.sort(["trade_date", "mom_20d"], descending=[False, True]).with_columns(
        pl.cum_count("trade_date").over("trade_date").alias("rank_in_day")
    )
    ind_agg = ind_agg.with_columns([
        (pl.col("rank_in_day") <= 5).alias("ind_hot"),
        (pl.col("rank_in_day") >= 27).alias("ind_cold"),
    ]).select(["trade_date", "l1_code", "ind_hot", "ind_cold"])

    return stock_ind.join(ind_agg, on=["trade_date", "l1_code"], how="left").select(
        ["trade_date", "ts_code", "ind_hot", "ind_cold"]
    )


def evaluate_one(preds_aug: pl.DataFrame, score_col: str, time_start, time_end,
                 universe_filter: pl.Expr | None = None, top_k: int = 50) -> dict:
    df = preds_aug.filter(
        (pl.col("trade_date") >= time_start) & (pl.col("trade_date") <= time_end)
    )
    if universe_filter is not None:
        df = df.filter(universe_filter)
    if len(df) == 0:
        return {"n_days": 0}

    top = (
        df.sort(["trade_date", score_col], descending=[False, True])
        .group_by("trade_date", maintain_order=True)
        .head(top_k)
    )

    daily = top.group_by("trade_date").agg([
        pl.col("excess_1d").mean().alias("daily_excess_1d"),
        pl.col("excess_3d").mean().alias("daily_excess_3d"),
        pl.col("excess_5d").mean().alias("daily_excess_5d"),
        (pl.col("excess_1d") > 0).cast(pl.Float64).mean().alias("daily_T1_hit"),
        (pl.col("excess_3d") > 0).cast(pl.Float64).mean().alias("daily_T3_hit"),
        (pl.col("excess_5d") > 0).cast(pl.Float64).mean().alias("daily_T5_hit"),
        pl.col("entry_max_dd_5d").mean().alias("daily_max_dd_5d"),
        pl.col("entry_max_ret_5d").mean().alias("daily_max_ret_5d"),
        pl.col("ts_code").count().alias("n_picks_today"),
    ])
    n_days = len(daily)
    if n_days == 0:
        return {"n_days": 0}

    return {
        "n_days":           n_days,
        "n_picks_avg":      float(daily["n_picks_today"].mean()),
        "top50_excess_1d":  float(daily["daily_excess_1d"].mean()),
        "top50_excess_3d":  float(daily["daily_excess_3d"].mean()),
        "top50_excess_5d":  float(daily["daily_excess_5d"].mean()),
        "T1_hit":           float(daily["daily_T1_hit"].mean()),
        "T3_hit":           float(daily["daily_T3_hit"].mean()),
        "T5_hit":           float(daily["daily_T5_hit"].mean()),
        "entry_max_dd_5d":  float(daily["daily_max_dd_5d"].mean()),
        "entry_max_ret_5d": float(daily["daily_max_ret_5d"].mean()),
    }


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("[load] realized + market + per-entry forward metrics")
    realized = pl.read_parquet(BUNDLE / "realized_returns.parquet")
    market = pl.read_parquet(BUNDLE / "market_returns.parquet")
    fwd = build_per_entry_metrics(realized, market)
    print(f"  fwd rows: {len(fwd):,}, date range: {fwd['trade_date'].min()} ~ {fwd['trade_date'].max()}")

    print("[load] slice annotations: size + money-regime + industry-regime")
    size = load_size_slice()
    money_regime = load_money_regime()
    ind_regime = load_industry_regime()

    universe_slices = {
        "all":         None,
        "HS300":       pl.col("size_large") == True,
        "ZZ500":       pl.col("size_mid") == True,
        "CSI1000":     pl.col("size_small") == True,
        "MF_hot":      pl.col("mf_regime_hot") == True,
        "MF_cold":     pl.col("mf_regime_cold") == True,
        "Industry_hot": pl.col("ind_hot") == True,
        "Industry_cold": pl.col("ind_cold") == True,
    }

    results = {}
    for name, path, score_col in PATHS:
        p = Path(path)
        if not p.exists():
            print(f"[skip] {name} (missing {p})")
            continue
        print(f"[eval] {name}")
        preds = pl.read_parquet(p)
        preds_aug = (
            preds.join(fwd, on=["trade_date", "ts_code"], how="inner")
            .join(size, on=["trade_date", "ts_code"], how="left")
            .join(money_regime, on="trade_date", how="left")
            .join(ind_regime, on=["trade_date", "ts_code"], how="left")
        )
        results[name] = {}
        for tname, (ts, te) in TIME_SLICES.items():
            results[name][tname] = {}
            for uname, ufilter in universe_slices.items():
                m = evaluate_one(preds_aug, score_col, ts, te, ufilter)
                results[name][tname][uname] = m
            row = results[name][tname]["all"]
            print(f"  {tname:18} all: n={row.get('n_days',0)}  "
                  f"mean_y=excess1d={row.get('top50_excess_1d',0):+.5f} "
                  f"T1={row.get('T1_hit',0):.3f} T3={row.get('T3_hit',0):.3f} "
                  f"max_dd={row.get('entry_max_dd_5d',0):.3f} "
                  f"max_ret={row.get('entry_max_ret_5d',0):.3f}")

    (OUT_ROOT / "results.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"wrote {OUT_ROOT / 'results.json'}")

    # ============ Markdown report ============
    md = [
        "# Time-Slice Evaluation Matrix v2 (entry-only framing)",
        f"Generated: {dt.datetime.now().isoformat()}",
        "",
        "Per user (2026-05-13): entry-only model — DROP turnover / cumulative P&L /",
        "portfolio drawdown (assume separate exit model). KEEP per-entry metrics.",
        "",
        "## Time slices",
    ]
    for s, (a, b) in TIME_SLICES.items():
        md.append(f"- **{s}**: {a} ~ {b}")
    md.append("")
    md.append("## Universe slices")
    md.append("- **all**: full main-board universe")
    md.append("- **HS300**: large-cap (HS300 members)")
    md.append("- **ZZ500**: mid-cap (ZZ500 members not in HS300)")
    md.append("- **CSI1000**: small-cap (CSI1000 members not in HS300/ZZ500)")
    md.append("- **MF_hot/cold**: top/bottom 25% days by main_force_net rolling-20d")
    md.append("- **Industry_hot/cold**: stocks in top-5 / bottom-5 L1 industries by 20d momentum")
    md.append("")

    # Headline: top50_excess_3d × time-slice × all-universe
    for metric, label, fmt in [
        ("top50_excess_1d",  "Top-50 fwd-1d excess",                "+.5f"),
        ("top50_excess_3d",  "Top-50 fwd-3d excess",                "+.5f"),
        ("top50_excess_5d",  "Top-50 fwd-5d excess",                "+.5f"),
        ("T1_hit",           "T1 hit (fwd-1d excess > 0)",          ".3f"),
        ("T3_hit",           "T3 hit (fwd-3d excess > 0)",          ".3f"),
        ("T5_hit",           "T5 hit (fwd-5d excess > 0)",          ".3f"),
        ("entry_max_dd_5d",  "Per-entry max drawdown within 5d",    "+.4f"),
        ("entry_max_ret_5d", "Per-entry max return within 5d (主升浪上限)", "+.4f"),
    ]:
        md.append(f"## {label} — full universe")
        md.append("")
        md.append("| Path | " + " | ".join(TIME_SLICES.keys()) + " |")
        md.append("|---" + "|---:" * len(TIME_SLICES) + "|")
        for name in results:
            row = [name]
            for s in TIME_SLICES.keys():
                v = results[name].get(s, {}).get("all", {}).get(metric)
                row.append(format(v, fmt) if v is not None else "—")
            md.append("| " + " | ".join(row) + " |")
        md.append("")

    # Universe-slice cross-cut on top50_excess_3d for the H2 + 2026-Q1 windows
    for tname in ("H2", "2026-Q1"):
        md.append(f"## Universe slice breakdown — top50_excess_3d × {tname}")
        md.append("")
        md.append("| Path | " + " | ".join(universe_slices.keys()) + " |")
        md.append("|---" + "|---:" * len(universe_slices) + "|")
        for name in results:
            row = [name]
            for u in universe_slices.keys():
                v = results[name].get(tname, {}).get(u, {}).get("top50_excess_3d")
                row.append(f"{v:+.5f}" if v is not None else "—")
            md.append("| " + " | ".join(row) + " |")
        md.append("")

    # Headline take-aways (auto-generated)
    md.append("## Auto take-aways")
    md.append("")
    h2_2026q1 = []
    for name in results:
        v_h2 = results[name].get("H2", {}).get("all", {}).get("top50_excess_3d")
        v_q1 = results[name].get("2026-Q1", {}).get("all", {}).get("top50_excess_3d")
        if v_h2 is not None and v_q1 is not None:
            h2_2026q1.append((name, v_h2, v_q1, v_q1 - v_h2))
    h2_2026q1.sort(key=lambda x: x[1], reverse=True)
    md.append("### Ranking by H2 top50_excess_3d (decreasing)")
    md.append("| Path | H2 | 2026-Q1 | Δ (Q1 - H2) |")
    md.append("|---|---:|---:|---:|")
    for n, h2, q1, d in h2_2026q1:
        md.append(f"| {n} | {h2:+.5f} | {q1:+.5f} | {d*1e4:+.1f} bps |")
    md.append("")

    (OUT_ROOT / "RESULTS.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {OUT_ROOT / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
