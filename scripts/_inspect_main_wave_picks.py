#!/usr/bin/env python3
"""Phase 22 (c) — inspect main_wave_picks.jsonl by industry, period, hit/miss.

Reads ``main_wave_picks.jsonl`` produced by ``_eval_main_wave*.py`` and the
parquet's ``industry_code`` mapping, and prints:

1. Aggregate counts: total picks, hit count by ckpt × top_k.
2. Industry breakdown of picks vs industry hit rate.
3. Per-month breakdown — does the model do better in some months?
4. Score model vs realized hold_return scatter (ranking quality).
5. A few "best hits" and a few "biggest losers" with their context.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import polars as pl


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--picks", required=True, type=Path,
                   help="Path to main_wave_picks.jsonl")
    p.add_argument("--data-path", required=True, type=Path,
                   help="Parquet for industry_code lookup")
    p.add_argument("--top-k", type=int, default=None,
                   help="Filter to one top_k value (default: all)")
    p.add_argument("--out", type=Path, default=None,
                   help="Optional markdown summary out path")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = [json.loads(line) for line in args.picks.read_text(encoding="utf-8").splitlines() if line]
    if args.top_k is not None:
        rows = [r for r in rows if r.get("top_k") == args.top_k]
    if not rows:
        print(f"[err] no rows in {args.picks}", file=sys.stderr)
        return 1

    print(f"[inspect] {len(rows)} pick rows, "
          f"top_k values: {sorted({r['top_k'] for r in rows})}, "
          f"ckpts: {sorted({r['ckpt'] for r in rows})}")

    # ---- Industry lookup ----
    df = pl.read_parquet(args.data_path, columns=["ts_code", "industry_code"]).unique()
    # industry_code can be int or string (申万一级行业 Chinese name); keep as-is.
    industry_lookup: dict[str, str] = {
        row["ts_code"]: str(row["industry_code"]) if row["industry_code"] is not None else "?"
        for row in df.iter_rows(named=True)
    }

    out_lines: list[str] = []

    def _emit(line: str) -> None:
        print(line)
        out_lines.append(line)

    # ---- 1. Per-ckpt × top_k summary ----
    _emit("\n### 1. Aggregate")
    by_key: dict[tuple, list] = defaultdict(list)
    for r in rows:
        if not r.get("label_valid"):
            continue
        by_key[(r["ckpt"], r["top_k"])].append(r)
    for (ckpt, k), rs in sorted(by_key.items()):
        n = len(rs)
        hits = sum(1 for r in rs if r["hit_main_wave"])
        wins = sum(1 for r in rs if r["hold_return"] > 0)
        avg_h = float(np.mean([r["hold_return"] for r in rs]))
        avg_dd = float(np.mean([abs(r["max_drawdown_during_hold"]) for r in rs]))
        _emit(f"  {ckpt:>15} top{k}: n={n} hit={hits/n:.3f} win={wins/n:.3f} "
              f"avg_hold={avg_h:+.4f} avg_dd={avg_dd:.4f}")

    # ---- 2. Industry breakdown ----
    _emit("\n### 2. Top industries (by pick count, then hit rate)")
    industry_picks: dict[str, list] = defaultdict(list)
    for r in rows:
        if not r.get("label_valid"):
            continue
        ind = industry_lookup.get(r["stock_code"], "?")
        industry_picks[ind].append(r)
    sorted_inds = sorted(industry_picks.items(),
                          key=lambda kv: -len(kv[1]))[:15]
    for ind_code, rs in sorted_inds:
        n = len(rs)
        hits = sum(1 for r in rs if r["hit_main_wave"])
        avg_h = float(np.mean([r["hold_return"] for r in rs]))
        _emit(f"  {ind_code!s:>20}: n={n:>4} hit={hits/n:.3f} avg_hold={avg_h:+.4f}")

    # ---- 3. Per-month ----
    _emit("\n### 3. Per-month")
    by_month: dict[str, list] = defaultdict(list)
    for r in rows:
        if not r.get("label_valid") or r.get("date") is None:
            continue
        ym = r["date"][:7]  # 'YYYY-MM'
        by_month[ym].append(r)
    for ym in sorted(by_month):
        rs = by_month[ym]
        n = len(rs)
        hits = sum(1 for r in rs if r["hit_main_wave"])
        avg_h = float(np.mean([r["hold_return"] for r in rs]))
        wins = sum(1 for r in rs if r["hold_return"] > 0)
        _emit(f"  {ym}: n={n:>4} hit={hits/n:.3f} win={wins/n:.3f} avg_hold={avg_h:+.4f}")

    # ---- 4. Ranking quality: model score vs realized hold_return ----
    _emit("\n### 4. Score-vs-return correlation (per ckpt)")
    by_ckpt: dict[str, list] = defaultdict(list)
    for r in rows:
        if not r.get("label_valid"):
            continue
        by_ckpt[r["ckpt"]].append(r)
    for ckpt, rs in sorted(by_ckpt.items()):
        scores = np.array([r["score_model"] for r in rs])
        rets = np.array([r["hold_return"] for r in rs])
        if len(scores) > 1 and scores.std() > 1e-9 and rets.std() > 1e-9:
            corr = float(np.corrcoef(scores, rets)[0, 1])
        else:
            corr = 0.0
        _emit(f"  {ckpt}: corr(score, hold_return) = {corr:+.4f}  (n={len(rs)})")

    # ---- 5. Sample best hits + worst losers ----
    _emit("\n### 5. Sample best 5 hits and worst 5 losers (across all ckpts/top_k)")
    sorted_by_ret = sorted([r for r in rows if r.get("label_valid")],
                           key=lambda r: r["hold_return"], reverse=True)
    best = sorted_by_ret[:5]
    worst = sorted_by_ret[-5:]
    _emit("Best 5:")
    for r in best:
        _emit(f"  {r['date']} {r['stock_code']:11} ind={industry_lookup.get(r['stock_code'], '?')!s:>15} "
              f"score={r['score_model']:+.3f} hold={r['hold_return']:+.4f} "
              f"max_cum={r['max_cum_return_5d']:+.4f} dd={r['max_drawdown_during_hold']:+.4f} "
              f"days={r['holding_days']} hit={'Y' if r['hit_main_wave'] else 'N'}")
    _emit("Worst 5:")
    for r in worst:
        _emit(f"  {r['date']} {r['stock_code']:11} ind={industry_lookup.get(r['stock_code'], '?')!s:>15} "
              f"score={r['score_model']:+.3f} hold={r['hold_return']:+.4f} "
              f"max_cum={r['max_cum_return_5d']:+.4f} dd={r['max_drawdown_during_hold']:+.4f} "
              f"days={r['holding_days']} hit={'Y' if r['hit_main_wave'] else 'N'}")

    # ---- 6. Stocks picked most often ----
    _emit("\n### 6. Stocks picked >= 5 times (concentration check)")
    code_counts = Counter(r["stock_code"] for r in rows if r.get("label_valid"))
    repeats = sorted([(c, n) for c, n in code_counts.items() if n >= 5],
                     key=lambda kv: -kv[1])[:20]
    if not repeats:
        _emit("  (no stock picked 5+ times)")
    for code, n in repeats:
        ind = industry_lookup.get(code, "?")
        _emit(f"  {code:11} ind={ind!s:>15} picked {n} times")

    if args.out is not None:
        args.out.write_text("\n".join(out_lines), encoding="utf-8")
        print(f"\n[inspect] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
