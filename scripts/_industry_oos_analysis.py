"""Phase 15 industry-stratified OOS analysis.

Loads a trained SB3 zip checkpoint, runs inference on the validation window,
and reports the industry distribution of the top-K picks vs. the universe
baseline. CPU-only by default so it can run alongside a GPU training job.

Usage
-----
    python scripts/_industry_oos_analysis.py \
        --model models/production/phase14c_600k_best_oos.zip \
        --metadata models/production/phase14c_600k_metadata.json \
        --data data/factor_panel_combined_short_2023_2026.parquet \
        --val-start 2025-07-01 --val-end 2026-04-24 \
        --top-k 30 \
        --out runs/phase15_8h_exploration/industry_oos_600k.json

Output: JSON with per-industry pick count, universe count, overweight ratio,
average forward return per industry-date, and a markdown table to stdout.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import defaultdict
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

import numpy as np
import polars as pl
import torch
from stable_baselines3 import PPO

from aurumq_rl.data_loader import (
    FactorPanelLoader,
    UniverseFilter,
    align_panel_to_stock_list,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--metadata", required=True, type=Path)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--val-start", required=True)
    p.add_argument("--val-end", required=True)
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--universe-filter", default="main_board_non_st")
    p.add_argument("--device", default="cpu",
                   help="torch device for inference (default cpu so we don't fight a parallel GPU training job)")
    p.add_argument("--out", required=True, type=Path)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    meta = json.loads(args.metadata.read_text(encoding="utf-8"))
    train_stock_codes = meta["stock_codes"]
    factor_count = meta["factor_count"]
    print(f"[industry_oos] training universe: {len(train_stock_codes)} stocks, {factor_count} factors")

    loader = FactorPanelLoader(parquet_path=args.data)
    panel = loader.load_panel(
        start_date=dt.date.fromisoformat(args.val_start),
        end_date=dt.date.fromisoformat(args.val_end),
        n_factors=factor_count,
        forward_period=10,
        universe_filter=UniverseFilter(args.universe_filter),
    )
    panel = align_panel_to_stock_list(panel, train_stock_codes)
    n_dates, n_stocks, n_factors = panel.factor_array.shape
    print(f"[industry_oos] aligned panel: dates={n_dates} stocks={n_stocks} factors={n_factors}")

    # ts_code,trade_date -> industry_code via raw parquet (industry_code is
    # NOT in the FactorPanel object).
    print(f"[industry_oos] reading industry_code mapping from {args.data}...")
    raw = pl.read_parquet(args.data, columns=["ts_code", "trade_date", "industry_code"])
    raw = raw.filter(
        (pl.col("trade_date") >= dt.date.fromisoformat(args.val_start))
        & (pl.col("trade_date") <= dt.date.fromisoformat(args.val_end))
    )
    # industry_code is per (ts_code, trade_date). For our purposes it's
    # ~stable; take latest seen per ts_code in the val window.
    ind_per_stock = (
        raw.group_by("ts_code")
        .agg(pl.col("industry_code").last())
        .to_pandas()
    )
    ind_map = dict(zip(ind_per_stock["ts_code"], ind_per_stock["industry_code"]))
    train_industry = [ind_map.get(c, "UNKNOWN") for c in train_stock_codes]
    train_industry = ["UNKNOWN" if (i is None or i == "NaN") else i for i in train_industry]
    print(f"[industry_oos] industry mapping: {sum(1 for x in train_industry if x != 'UNKNOWN')}/{n_stocks} resolved")

    # The 600k zip was trained with IndexOnlyRolloutBuffer / GPURolloutBuffer
    # which require cuda even at load. We don't actually need a rollout buffer
    # for inference — override with the SB3 default.
    print(f"[industry_oos] loading model {args.model} on {args.device}...")
    from stable_baselines3.common.buffers import RolloutBuffer
    custom_objects = {"rollout_buffer_class": RolloutBuffer}
    model = PPO.load(str(args.model), device=args.device, custom_objects=custom_objects)
    model.policy.eval()
    # Move policy to chosen device explicitly (SB3 sometimes ignores device on load).
    model.policy.to(args.device)

    panel_t = torch.from_numpy(panel.factor_array).to(args.device)
    valid_t = (
        ~torch.from_numpy(panel.is_st_array).to(args.device)
        & ~torch.from_numpy(panel.is_suspended_array).to(args.device)
    )

    # Per-date scoring + top-k pick
    print(f"[industry_oos] scoring {n_dates} dates...")
    pick_industry_counts: dict[str, int] = defaultdict(int)
    pick_industry_ret_sum: dict[str, float] = defaultdict(float)
    pick_industry_ret_n: dict[str, int] = defaultdict(int)
    universe_industry_counts: dict[str, int] = defaultdict(int)
    pick_codes_per_date: list[list[str]] = []

    with torch.no_grad():
        for t in range(n_dates):
            feats = model.policy.features_extractor(panel_t[t : t + 1])
            scores = model.policy.action_net(feats["per_stock"]).squeeze(-1)[0]  # (n_stocks,)
            mask = valid_t[t]
            # mask out invalid before topk
            scores_masked = scores.clone()
            scores_masked[~mask] = float("-inf")
            topk_idx = torch.topk(scores_masked, args.top_k).indices.cpu().numpy()

            picks = [train_stock_codes[int(j)] for j in topk_idx]
            pick_codes_per_date.append(picks)

            ret_t = panel.return_array[t]
            for j_int in topk_idx.tolist():
                ind = train_industry[j_int]
                pick_industry_counts[ind] += 1
                rj = float(ret_t[j_int])
                if np.isfinite(rj):
                    pick_industry_ret_sum[ind] += rj
                    pick_industry_ret_n[ind] += 1
            valid_idx = mask.cpu().numpy().nonzero()[0]
            for j_int in valid_idx.tolist():
                universe_industry_counts[train_industry[j_int]] += 1

    total_picks = sum(pick_industry_counts.values())
    total_universe = sum(universe_industry_counts.values())
    print(f"[industry_oos] total picks={total_picks} (= {n_dates} * {args.top_k}); universe slots={total_universe}")

    # Build per-industry stats
    rows = []
    all_industries = set(pick_industry_counts) | set(universe_industry_counts)
    for ind in sorted(all_industries):
        pc = pick_industry_counts.get(ind, 0)
        uc = universe_industry_counts.get(ind, 0)
        pick_pct = pc / total_picks if total_picks else 0.0
        uni_pct = uc / total_universe if total_universe else 0.0
        overweight = (pick_pct / uni_pct) if uni_pct > 0 else float("inf")
        n_ret = pick_industry_ret_n.get(ind, 0)
        avg_ret = pick_industry_ret_sum.get(ind, 0.0) / n_ret if n_ret else 0.0
        rows.append({
            "industry": ind,
            "pick_count": pc,
            "universe_slots": uc,
            "pick_pct": pick_pct,
            "universe_pct": uni_pct,
            "overweight_ratio": overweight,
            "avg_forward_return_when_picked": avg_ret,
            "n_picks_with_ret": n_ret,
        })
    rows.sort(key=lambda r: r["pick_count"], reverse=True)

    out_data = {
        "model": str(args.model),
        "val_start": args.val_start,
        "val_end": args.val_end,
        "top_k": args.top_k,
        "n_dates": n_dates,
        "total_picks": total_picks,
        "n_industries_picked": sum(1 for r in rows if r["pick_count"] > 0),
        "n_industries_universe": sum(1 for r in rows if r["universe_slots"] > 0),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[industry_oos] wrote {args.out}")

    # Markdown summary (top 25 by pick_count)
    md_lines = [
        f"# Industry-stratified OOS picks — {Path(args.model).name}",
        "",
        f"- val window: {args.val_start} → {args.val_end} ({n_dates} dates)",
        f"- top-K = {args.top_k}, total picks = {total_picks}",
        f"- universe industries: {out_data['n_industries_universe']}, picked from: {out_data['n_industries_picked']}",
        "",
        "Overweight = pick_pct / universe_pct. >1 means model favors this industry vs cap-weighted universe.",
        "",
        "| industry | picks | uni | pick% | uni% | overweight | avg fwd ret |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows[:30]:
        ow = r["overweight_ratio"]
        ow_s = f"{ow:.2f}x" if ow != float("inf") else "∞"
        md_lines.append(
            f"| {r['industry']} | {r['pick_count']} | {r['universe_slots']} | "
            f"{100*r['pick_pct']:.2f}% | {100*r['universe_pct']:.2f}% | "
            f"{ow_s} | {100*r['avg_forward_return_when_picked']:+.2f}% |"
        )
    md_path = args.out.with_suffix(".md")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[industry_oos] wrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
