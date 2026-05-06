#!/usr/bin/env python3
"""Phase 23 — diagnose every T-1 hit in a run's episode_picks.jsonl.

For each pick where the model selected a stock at T-1 of a real episode,
pull:
- Price path 30 days before t_start through t_peak
- Volume path
- Key factor values at decision day vs cross-section z-scores
- Verdict: was this a clean main-wave eve (consolidation → breakout) or
  a "lucky bounce" (oversold rebound that happened to morph into a wave)?

Output: ``runs/<run_dir>/t1_diagnostic.md`` with one section per hit.
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--picks", type=Path, required=True,
                   help="Path to episode_picks.jsonl")
    p.add_argument("--data-path", type=Path, required=True)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--ckpt-label", default="step224928",
                   help="Filter to one ckpt label, default best")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--days-before", type=int, default=20)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in args.picks.read_text(encoding="utf-8").splitlines() if l]
    rows = [
        r for r in rows
        if r.get("top_k") == args.top_k
        and r.get("ckpt") == args.ckpt_label
        and r.get("proximity_to_episode") == 1
    ]
    print(f"[diag] {len(rows)} T-1 hits in {args.picks} (top_k={args.top_k} ckpt={args.ckpt_label})")
    if not rows:
        return 0

    df = pl.read_parquet(args.data_path)

    # Useful factor list — includes both legacy mf and the cumulative-position
    # signals plus alpha/gtja extremes
    factor_cols = [
        # Cumulative position
        "mfp_elg_buy_ratio_20d", "mfp_lg_buy_ratio_20d",
        "hm_seat_count_30d", "inst_appear_count_60d",
        "mg_buy_30d_ratio", "mg_balance_pct",
        # Recent flow
        "mf_net_1d", "mf_net_3d", "mf_net_5d", "mf_net_10d", "mf_net_20d",
        # Sentiment / chip
        "senti_max_step_60d", "cyq_winning_ratio",
        # Selected alphas
        "alpha_005", "alpha_004", "alpha_001", "gtja_004",
        # Industry
        "ind_relative_strength_60d",
    ]
    factor_cols = [c for c in factor_cols if c in df.columns]

    # Get full unique date list for navigation
    all_dates = sorted(df.select("trade_date").unique().to_series().to_list())

    sections = []
    for idx, r in enumerate(rows, 1):
        date = dt.date.fromisoformat(r["date"])
        code = r["stock_code"]
        ep = r.get("episode") or {}
        t_start_d = dt.date.fromisoformat(ep["t_start_date"]) if ep.get("t_start_date") else None
        t_peak_d = dt.date.fromisoformat(ep["t_peak_date"]) if ep.get("t_peak_date") else None
        peak_ret = ep.get("peak_return", 0.0)
        duration = ep.get("duration", 0)
        max_dd = ep.get("max_dd_during", 0.0)

        # Stock close path: from days_before before decision through t_peak + 5
        date_to_idx = {d: i for i, d in enumerate(all_dates)}
        if date not in date_to_idx:
            continue
        d_idx = date_to_idx[date]
        lo_idx = max(0, d_idx - args.days_before)
        hi_idx = min(len(all_dates) - 1, date_to_idx.get(t_peak_d, d_idx + 25) + 5)
        win_dates = all_dates[lo_idx:hi_idx + 1]
        sub = df.filter(
            (pl.col("ts_code") == code)
            & (pl.col("trade_date").is_in(win_dates))
        ).select(["trade_date", "close", "pct_chg", "vol"]).sort("trade_date")

        # Day-by-day price string with flag
        path_lines: list[str] = []
        first_close = None
        for row in sub.iter_rows(named=True):
            d = row["trade_date"]
            c = row["close"] or 0.0
            pct = (row["pct_chg"] or 0.0)
            vol = row["vol"] or 0.0
            if first_close is None and c > 0:
                first_close = c
            cum = (c / first_close - 1.0) * 100 if first_close else 0.0
            tag = "  "
            if d == date:
                tag = "▲▲"  # decision day
            elif d == t_start_d:
                tag = " ★"
            elif d == t_peak_d:
                tag = " ◆"
            elif pct > 0.095:
                tag = " ↑"   # limit-up
            elif pct < -0.095:
                tag = " ↓"   # limit-down
            path_lines.append(
                f"  {tag} {d}  c={c:7.2f}  pct={pct*100:+6.2f}%  cum={cum:+7.2f}%  vol={vol/1e4:7.1f}万"
            )

        # Factor z-scores for THIS stock on the decision day
        day_df = df.filter(pl.col("trade_date") == date)
        factor_lines: list[str] = []
        for f in factor_cols:
            if f not in day_df.columns:
                continue
            vals = day_df.get_column(f).fill_null(0.0).to_numpy()
            valid = vals[np.isfinite(vals) & (vals != 0)]
            if len(valid) < 10:
                continue
            mu = float(valid.mean())
            sigma = float(valid.std())
            if sigma < 1e-9:
                continue
            target = day_df.filter(pl.col("ts_code") == code).get_column(f)
            if len(target) == 0:
                continue
            v = float(target[0]) if target[0] is not None else 0.0
            if not np.isfinite(v):
                continue
            z = (v - mu) / sigma
            factor_lines.append(f"  {f:<28}  z = {z:+.3f}")

        # Verdict heuristic
        verdict_signals: list[str] = []
        # Look at price action 5-day pre-decision: was it consolidation or trending?
        pre_close = sub.head(min(len(sub), args.days_before + 1))
        if len(pre_close) >= 5:
            last5 = pre_close.tail(5).get_column("close").to_numpy()
            if (last5 > 0).all():
                pct_5d = float(last5[-1] / last5[0] - 1)
                rng_5d = float(last5.max() / last5.min() - 1)
                if pct_5d < -0.05:
                    verdict_signals.append(f"5天累计 {pct_5d:+.1%} (大跌后反弹模式)")
                elif rng_5d < 0.04 and abs(pct_5d) < 0.02:
                    verdict_signals.append(f"5天横盘 (区间 {rng_5d:.1%})")
                else:
                    verdict_signals.append(f"5天 {pct_5d:+.1%} 区间 {rng_5d:.1%}")

        # Decision day relative to t_start: did the launch arrive immediately or after delay?
        if t_start_d and t_peak_d and t_start_d in date_to_idx:
            ts_idx = date_to_idx[t_start_d]
            tp_idx = date_to_idx[t_peak_d]
            # Did price spend most of the early window below t_start price?
            sub_array = sub.to_numpy()
            # find t_start_d in sub
            t_start_close = None
            t_start_offset = None
            for i, row in enumerate(sub.iter_rows(named=True)):
                if row["trade_date"] == t_start_d:
                    t_start_close = row["close"]
                    t_start_offset = i
                    break
            if t_start_close and t_start_offset is not None:
                post = sub.slice(t_start_offset, min(20, len(sub) - t_start_offset))
                post_closes = post.get_column("close").to_numpy()
                # Find first day >= t_start_close + 5%
                breakout_offset = None
                for i, c in enumerate(post_closes):
                    if c > 0 and c >= t_start_close * 1.05:
                        breakout_offset = i
                        break
                if breakout_offset is None:
                    verdict_signals.append("entry 后无 5% 突破日(直线上涨型)")
                elif breakout_offset == 0:
                    verdict_signals.append("entry 当日即 +5% 突破")
                elif breakout_offset <= 2:
                    verdict_signals.append(f"entry 后 {breakout_offset} 天内突破")
                else:
                    # Look for limit-up days in post window
                    post_pct = post.get_column("pct_chg").to_numpy()
                    n_zt = int(np.sum(post_pct > 0.095))
                    n_dt = int(np.sum(post_pct < -0.095))
                    verdict_signals.append(
                        f"entry 后 {breakout_offset} 天才突破 (洗盘 / 间隔)"
                        f"; 中间 {n_zt} 涨停 / {n_dt} 跌停"
                    )

        section = (
            f"### {idx}. {code}  decision={date}  t_start={t_start_d}  "
            f"peak={peak_ret:+.2%}  duration={duration}d  dd={max_dd:.2%}\n\n"
            "**Price path** (▲▲ = decision day, ★ = t_start, ◆ = t_peak, ↑↓ = 涨跌停):\n\n"
            "```\n" + "\n".join(path_lines) + "\n```\n\n"
            "**Factor z-scores at decision day**:\n\n"
            "```\n" + "\n".join(factor_lines) + "\n```\n\n"
            "**Verdict signals**: " + "; ".join(verdict_signals) + "\n"
        )
        sections.append(section)

    out = "# Phase 23A T-1 hits — diagnostic\n\n"
    out += f"Source: `{args.picks}`  ckpt=`{args.ckpt_label}`  top_k={args.top_k}\n\n"
    out += f"Total T-1 hits: {len(rows)}\n\n---\n\n"
    out += "\n\n---\n\n".join(sections)
    args.out.write_text(out, encoding="utf-8")
    print(f"[diag] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
