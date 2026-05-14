"""Rule-based regime router v1.

Per user 2026-05-13 strategic direction: stop "stack another model", build a
4-dimensional system: Entry family × Hold horizon × Exit model × Regime router.

This script implements the ROUTER component. Inputs daily regime features +
universe-aggregate signals; outputs per-day routing decision:
  (family ∈ {PROXIMITY, WAVE}, hold_K ∈ {1, 3, 5, 10, 20, 40})

Rules (transparent, no training):
  - high_vol + low_breadth  → PROXIMITY + K=1   (volatile, choppy, mean-revert)
  - high_vol + high_breadth → PROXIMITY + K=3   (volatile but broad rally)
  - low_vol + main_force_net>0 + main_force_streak>=3 → PROXIMITY + K=20  (steady bull)
  - low_vol + main_force_net<0                       → WAVE + K=20  (institutional avoiding)
  - sideways (med_vol + low_breadth_trend)           → WAVE + K=20

Then for each (model, slice) pair, simulate routing decisions vs fixed K=1 baseline
to see if router beats single-strategy.

Regime feature inputs:
  - univ_vol_20d
  - cs_dispersion
  - sector_dispersion (from regime_features)
  - main_force_net_yi (from universe_money_agg)
  - main_force_net_yi rolling 5d streak sign
  - north_net_yi
  - lianban_n_ge_4
  - top-5 industry momentum (rotate_into_top5 count from industry_momentum_rotation)
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import polars as pl


SHORT = Path("data/p3_4070")
SLICES_V1 = Path("data/p3_4070_slices")
SLICES_V2 = Path("data/p3_4070_slices_v2")
OUT = Path("runs/regime_router_v1")


def build_regime_table() -> pl.DataFrame:
    """Build per-trade_date regime feature table and routing decision."""
    print("[regime] loading inputs ...")
    regime = pl.read_parquet(SHORT / "regime_features.parquet").select([
        "trade_date", "univ_vol_20d", "cs_dispersion", "sector_dispersion",
        "univ_skew_20d", "market_pct"
    ])
    money = pl.read_parquet(SLICES_V2 / "universe_money_agg.parquet").select([
        "trade_date", "main_force_net_yi", "north_net_yi"
    ]).with_columns(pl.col("trade_date").cast(pl.Date))
    limits = pl.read_parquet(SLICES_V2 / "limit_daily_agg.parquet").select([
        "trade_date", "zt_count", "dt_count", "lianban_n_ge_4"
    ]).with_columns(pl.col("trade_date").cast(pl.Date))
    # industry_momentum_rotation_daily was shipped in paris_reply (P2 in v2 reply)
    ind_rot_path = Path("data/p3_4070/industry_momentum_rotation_daily.parquet")
    if not ind_rot_path.exists():
        ind_rot_path = Path("data/paris_reply_2026-05-13/industry_momentum_rotation_daily.parquet")
    ind_rot = pl.read_parquet(ind_rot_path).select([
        "trade_date", "rotated_into_top5"
    ]).with_columns(pl.col("trade_date").cast(pl.Date)).group_by("trade_date").agg(
        pl.col("rotated_into_top5").cast(pl.Int64).sum().alias("ind_rotation_count")
    )

    df = regime.join(money, on="trade_date", how="left") \
               .join(limits, on="trade_date", how="left") \
               .join(ind_rot, on="trade_date", how="left") \
               .sort("trade_date")

    # Derived signals
    df = df.with_columns([
        # main_force streak: rolling 5d mean of sign(main_force_net)
        pl.col("main_force_net_yi").sign().rolling_mean(window_size=5).alias("mf_streak_5d"),
        # zt_dt ratio: net upward limit pressure
        ((pl.col("zt_count") - pl.col("dt_count")) /
         (pl.col("zt_count") + pl.col("dt_count") + 1.0)).alias("limit_net_ratio"),
    ])

    # === Rule-based router ===
    # Build classification regions on quantiles of regime features (computed on full history)
    vol_q33 = df["univ_vol_20d"].quantile(0.33)
    vol_q67 = df["univ_vol_20d"].quantile(0.67)
    disp_q67 = df["cs_dispersion"].quantile(0.67)

    print(f"  vol q33/q67 = {vol_q33:.4f} / {vol_q67:.4f}")
    print(f"  disp q67    = {disp_q67:.4f}")

    df = df.with_columns([
        # Regime tags
        (pl.col("univ_vol_20d") > vol_q67).alias("vol_hi"),
        (pl.col("univ_vol_20d") < vol_q33).alias("vol_lo"),
        (pl.col("cs_dispersion") > disp_q67).alias("disp_hi"),
        (pl.col("mf_streak_5d") >= 0.6).alias("mf_bull"),
        (pl.col("mf_streak_5d") <= -0.6).alias("mf_bear"),
    ])

    # Decision tree (simple, transparent)
    df = df.with_columns(
        pl.when(pl.col("vol_hi") & pl.col("disp_hi"))   .then(pl.lit("PROXIMITY_K3"))
          .when(pl.col("vol_hi") & ~pl.col("disp_hi"))  .then(pl.lit("PROXIMITY_K1"))
          .when(pl.col("vol_lo") & pl.col("mf_bull"))   .then(pl.lit("PROXIMITY_K20"))
          .when(pl.col("vol_lo") & pl.col("mf_bear"))   .then(pl.lit("WAVE_K20"))
          .otherwise(pl.lit("WAVE_K20"))                 # default: wave + long hold
          .alias("router_decision")
    )

    # Parse decision into (family, K) for downstream use
    df = df.with_columns([
        pl.col("router_decision").str.starts_with("PROXIMITY").alias("use_proximity"),
        pl.when(pl.col("router_decision") == "PROXIMITY_K1").then(1)
          .when(pl.col("router_decision") == "PROXIMITY_K3").then(3)
          .when(pl.col("router_decision") == "PROXIMITY_K20").then(20)
          .when(pl.col("router_decision") == "WAVE_K20").then(20)
          .otherwise(20).alias("hold_K"),
    ])
    return df


def simulate_router(regime_df: pl.DataFrame):
    """Simulate router: per day, pick the routed (family, K) and compute realized excess.

    For each routed day:
      - if family=PROXIMITY: use Path 1 long predictions
      - if family=WAVE: use Wave v2 predictions
      - hold K days; realized cumulative fwd-K excess for top-50 picks
    """
    p1l = pl.read_parquet("runs/sl_path1_long/predictions.parquet").select([
        "trade_date", "ts_code", pl.col("score_calibrated").alias("score_proxy"),
    ])
    wv2 = pl.read_parquet("runs/sl_path1_long_wave_v2/predictions.parquet").select([
        "trade_date", "ts_code", pl.col("score_calibrated").alias("score_wave"),
    ])
    # Realized fwd-1d/3d/5d/20d excess
    realized = pl.read_parquet(SHORT / "realized_returns.parquet").select(
        ["trade_date", "ts_code", "pct_chg_t_plus_1"]
    ).sort(["ts_code", "trade_date"])
    market = pl.read_parquet(SHORT / "market_returns.parquet").select(
        ["trade_date", "eq_weight_pct_chg_t_plus_1"]
    ).sort("trade_date")
    for K in (1, 3, 5, 20):
        realized = realized.with_columns(
            pl.col("pct_chg_t_plus_1").log1p().rolling_sum(window_size=K).shift(-(K-1))
              .exp().sub(1.0).over("ts_code").alias(f"fwd_{K}d")
        )
        market = market.with_columns(
            pl.col("eq_weight_pct_chg_t_plus_1").log1p().rolling_sum(window_size=K).shift(-(K-1))
              .exp().sub(1.0).alias(f"mkt_fwd_{K}d")
        )
    df_r = realized.join(market.select(["trade_date","mkt_fwd_1d","mkt_fwd_3d","mkt_fwd_5d","mkt_fwd_20d"]),
                          on="trade_date", how="left")
    for K in (1, 3, 5, 20):
        df_r = df_r.with_columns((pl.col(f"fwd_{K}d") - pl.col(f"mkt_fwd_{K}d")).alias(f"excess_{K}d"))
    fwd = df_r.select(["trade_date","ts_code","excess_1d","excess_3d","excess_5d","excess_20d"])

    # Join everything: regime → score + realized
    print("[simulate] joining + ranking ...")
    base = regime_df.select(["trade_date", "use_proximity", "hold_K", "router_decision"])
    p1l_aug = base.join(p1l, on="trade_date", how="inner").join(fwd, on=["trade_date","ts_code"], how="inner")
    wv2_aug = base.join(wv2, on="trade_date", how="inner").join(fwd, on=["trade_date","ts_code"], how="inner")

    # For each day, compute top-50 by routed family's score, then realized excess at hold_K
    SLICES = {
        "VAL":              (dt.date(2025, 1, 1),  dt.date(2025, 6, 4)),
        "H1":               (dt.date(2025, 7, 1),  dt.date(2025, 9, 30)),
        "H2":               (dt.date(2025, 10, 1), dt.date(2025, 12, 31)),
        "2026-Q1":          (dt.date(2026, 1, 1),  dt.date(2026, 3, 31)),
        "2026-Q2-partial":  (dt.date(2026, 4, 1),  dt.date(2026, 5, 11)),
    }

    results = {}
    for sname, (ts, te) in SLICES.items():
        d_prox = p1l_aug.filter(pl.col("use_proximity") & (pl.col("trade_date") >= ts) & (pl.col("trade_date") <= te))
        d_wave = wv2_aug.filter(~pl.col("use_proximity") & (pl.col("trade_date") >= ts) & (pl.col("trade_date") <= te))
        # Top-50 by score, take realized at hold_K
        rows = []
        for d, score_col, K_col in ((d_prox, "score_proxy", "hold_K"), (d_wave, "score_wave", "hold_K")):
            if len(d) == 0:
                continue
            top = d.sort(["trade_date", score_col], descending=[False, True]) \
                   .group_by("trade_date", maintain_order=True).head(50)
            # For each day, use that day's hold_K to pick the realized
            for K in (1, 3, 5, 20):
                col = f"excess_{K}d"
                daily = top.filter(pl.col("hold_K") == K).group_by("trade_date").agg(
                    pl.col(col).mean().alias("e")
                ).drop_nulls("e")
                if len(daily) > 0:
                    rows.append({"K": K, "n_days": len(daily), "mean_excess": float(daily["e"].mean())})
        # Also default behavior: ROUTER decides which (family, K) per day
        # We need to compute "realized at hold_K" for each day's actual decision, mixing family
        all_router = pl.concat([
            d_prox.select(["trade_date", "use_proximity", "hold_K",
                           pl.col("score_proxy").alias("score"),
                           pl.col("excess_1d"), pl.col("excess_3d"), pl.col("excess_5d"), pl.col("excess_20d"),
                           "ts_code"]),
            d_wave.select(["trade_date", "use_proximity", "hold_K",
                           pl.col("score_wave").alias("score"),
                           pl.col("excess_1d"), pl.col("excess_3d"), pl.col("excess_5d"), pl.col("excess_20d"),
                           "ts_code"]),
        ])
        top_router = all_router.sort(["trade_date", "score"], descending=[False, True]) \
                                .group_by("trade_date", maintain_order=True).head(50)
        # Each day pick its routed K's realized:
        top_router = top_router.with_columns(
            pl.when(pl.col("hold_K") == 1).then(pl.col("excess_1d"))
              .when(pl.col("hold_K") == 3).then(pl.col("excess_3d"))
              .when(pl.col("hold_K") == 5).then(pl.col("excess_5d"))
              .when(pl.col("hold_K") == 20).then(pl.col("excess_20d"))
              .alias("routed_excess")
        )
        daily_router = top_router.group_by("trade_date").agg(
            pl.col("routed_excess").mean().alias("e"),
            pl.col("hold_K").first().alias("K"),
            pl.col("use_proximity").first().alias("uses_prox"),
        ).drop_nulls("e")
        results[sname] = {
            "n_days_routed": len(daily_router),
            "router_mean_excess": float(daily_router["e"].mean()) if len(daily_router) else None,
            "by_K": rows,
        }
        if len(daily_router):
            print(f"  {sname:18}  n_days={len(daily_router):>3}  "
                  f"router_mean={daily_router['e'].mean():+.5f}  "
                  f"K-distribution: {dict(daily_router.group_by('K').len().sort('K').iter_rows())}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=2, default=str))
    return results


def write_report(regime_df: pl.DataFrame, results: dict):
    OUT.mkdir(parents=True, exist_ok=True)
    # Decision distribution
    dist = regime_df.group_by("router_decision").len().sort("len", descending=True)
    md = [
        "# Rule-based Regime Router v1",
        f"Generated: {dt.datetime.now().isoformat()}",
        "",
        "Per user strategic direction (2026-05-13 afternoon): build Entry-family × Hold-K ×",
        "Exit × Router 4-D system. This is the ROUTER component — rules only, no training.",
        "",
        "## Decision rules",
        "",
        "| Condition | Decision |",
        "|---|---|",
        "| univ_vol_20d > q67 & cs_dispersion > q67 | PROXIMITY + K=3 |",
        "| univ_vol_20d > q67 & cs_dispersion ≤ q67 | PROXIMITY + K=1 |",
        "| univ_vol_20d < q33 & main_force_streak_5d ≥ +0.6 | PROXIMITY + K=20 |",
        "| univ_vol_20d < q33 & main_force_streak_5d ≤ -0.6 | WAVE + K=20 |",
        "| (else) | WAVE + K=20 |",
        "",
        "## Decision distribution (full history)",
        "",
        "| Decision | Days | Pct |",
        "|---|---:|---:|",
    ]
    total = dist["len"].sum()
    for r in dist.iter_rows(named=True):
        md.append(f"| {r['router_decision']} | {r['len']} | {r['len']/total*100:.1f}% |")
    md.append("")
    md.append("## Router-driven realized excess by slice")
    md.append("")
    md.append("| Slice | n_days_routed | router_mean_excess |")
    md.append("|---|---:|---:|")
    for sname, r in results.items():
        v = r["router_mean_excess"]
        md.append(f"| {sname} | {r['n_days_routed']} | {v:+.5f}" if v is not None else f"| {sname} | {r['n_days_routed']} | — |")
    md.append("")
    md.append("## Comparison vs fixed strategies (single-family, fixed K)")
    md.append("")
    md.append("Reference numbers from runs/eval_matrix_v2/RESULTS.md (top-50 fwd-K excess):")
    md.append("")
    md.append("| Slice | Path 1 long fwd-1d | Wave v2 fwd-20d | Router |")
    md.append("|---|---:|---:|---:|")
    fixed_p1l = {
        "VAL": 0.01476, "H1": 0.01451, "H2": 0.01520, "2026-Q1": 0.01733, "2026-Q2-partial": 0.01250,
    }
    fixed_wv2_20 = {
        "VAL": 0.00049, "H1": 0.01285, "H2": 0.00449, "2026-Q1": 0.00124, "2026-Q2-partial": None,
    }
    def fmt(v): return f"{v:+.5f}" if v is not None else "—"
    for sname in results:
        r_val = results[sname]["router_mean_excess"]
        p1l_val = fixed_p1l.get(sname)
        wv2_val = fixed_wv2_20.get(sname)
        md.append(f"| {sname} | {fmt(p1l_val)} | {fmt(wv2_val)} | {fmt(r_val)} |")
    md.append("")
    md.append("**Interpretation**: if router beats single-family in some slices, the rules")
    md.append("are capturing real regime-dependent value. Tune the rule thresholds based on this.")

    (OUT / "RESULTS.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {OUT / 'RESULTS.md'}")


def main():
    regime_df = build_regime_table()
    print()
    print(f"[regime] {len(regime_df)} trade days")
    print(regime_df.group_by("router_decision").len().sort("len", descending=True))
    print()
    results = simulate_router(regime_df)
    print()
    write_report(regime_df, results)


if __name__ == "__main__":
    main()
