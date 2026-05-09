"""Aggregate Phase 26F/G v3 runs into a markdown scoreboard.

Pass/fail criteria (from PROPOSAL_PHASE26FG_V3.md):
  26C2-v3 sanity: best lift ≥ 1.85× (replicate v2 baseline 1.70× +0.15× margin)
  26F-v3:  median > 26C2-v3 + 0.20×  AND  median > 26F-v2 (2.15×)
  26G-v3:  median > 26F-v3 + 0.10×  (encoder bump unlocks more)

Usage::

    python phase26fg_v3_scoreboard.py \\
        --runs-dir runs --tiers 26C2-v3 26F-v3 26G-v3 \\
        --seeds 42 43 44 \\
        --baseline-c2-v2-median 1.70 --baseline-f-v2-median 2.15 \\
        --out runs/phase26fg_v3_scoreboard.md
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def best_lift_top5(eval_json: dict) -> tuple[float, float, str]:
    rows = eval_json.get("checkpoint_eval_results", []) or eval_json.get("rows", [])
    best = (-1.0, 0.0, "n/a")
    for r in rows:
        if r.get("top_k") != 5:
            continue
        lift = r.get("t1_lift_over_base") or 0.0
        if lift > best[0]:
            best = (float(lift), float(r.get("t_minus_1_hit_rate") or 0.0), r.get("checkpoint_label", "?"))
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path, default=Path("runs"))
    ap.add_argument("--tiers", nargs="+", default=["26C2-v3", "26F-v3", "26G-v3"])
    ap.add_argument("--seeds", nargs="+", default=["42", "43", "44"])
    ap.add_argument("--baseline-c2-v2-median", type=float, default=1.70)
    ap.add_argument("--baseline-f-v2-median", type=float, default=2.15)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows: list[str] = []
    rows.append("# Phase 26 F/G v3 Scoreboard\n")
    rows.append(
        f"v2 baselines: 26C2-v2 median **{args.baseline_c2_v2_median}×**, "
        f"26F-v2 median **{args.baseline_f_v2_median}×**\n\n"
    )
    rows.append("## Per-tier (top_k=5)\n")
    rows.append("| tier | best lift | best hit | best ckpt | median (3 seed) | min | max |")
    rows.append("|---|---:|---:|---|---:|---:|---:|")

    summaries: dict[str, dict] = {}
    for tier in args.tiers:
        per_seed: list[float] = []
        per_hit: list[float] = []
        ckpts: list[str] = []
        for seed in args.seeds:
            run_dir = args.runs_dir / f"{tier}_seed{seed}"
            ev_path = run_dir / "episode_eval.json"
            if not ev_path.exists():
                continue
            ev = json.loads(ev_path.read_text(encoding="utf-8"))
            lift, hit, ckpt = best_lift_top5(ev)
            per_seed.append(lift)
            per_hit.append(hit)
            ckpts.append(ckpt)
        if not per_seed:
            rows.append(f"| {tier} | NO DATA | | | | | |")
            continue
        med = statistics.median(per_seed)
        bidx = max(range(len(per_seed)), key=lambda i: per_seed[i])
        rows.append(
            f"| {tier} | {per_seed[bidx]:.2f}× | {per_hit[bidx]*100:.2f}% | {ckpts[bidx]} | "
            f"{med:.2f}× | {min(per_seed):.2f}× | {max(per_seed):.2f}× |"
        )
        summaries[tier] = {
            "best": per_seed[bidx], "med": med,
            "min": min(per_seed), "max": max(per_seed),
            "n": len(per_seed),
        }

    rows.append("\n## Pass/fail per tier\n")
    rows.append("| tier | criterion | result |")
    rows.append("|---|---|---|")

    c2 = summaries.get("26C2-v3")
    f3 = summaries.get("26F-v3")
    g3 = summaries.get("26G-v3")

    if c2:
        ok = c2["best"] >= 1.85
        rows.append(
            f"| 26C2-v3 | best ≥ 1.85× (sanity replicate v2) | "
            f"{'PASS' if ok else 'FAIL'} ({c2['best']:.2f}×) |"
        )
    if f3 and c2:
        thr_c2 = c2["med"] + 0.20
        ok = f3["med"] > thr_c2 and f3["med"] > args.baseline_f_v2_median
        rows.append(
            f"| 26F-v3 | median > 26C2-v3 + 0.20× ({thr_c2:.2f}×) AND > 26F-v2 ({args.baseline_f_v2_median:.2f}×) | "
            f"{'PASS' if ok else 'REJECT'} (med={f3['med']:.2f}×) |"
        )
    if g3 and f3:
        thr_f = f3["med"] + 0.10
        ok = g3["med"] > thr_f
        rows.append(
            f"| 26G-v3 | median > 26F-v3 + 0.10× ({thr_f:.2f}×) | "
            f"{'PASS' if ok else 'REJECT'} (med={g3['med']:.2f}×) |"
        )

    rows.append("\n## Recommendation\n")
    if g3 and f3 and g3["med"] > f3["med"] + 0.10:
        rows.append("→ **Promote 26G-v3 to production.** Encoder bump 192→96→48 with fp16 panel "
                    "delivered; replace 23A baseline.")
    elif f3 and c2 and f3["med"] > c2["med"] + 0.20 and f3["med"] > args.baseline_f_v2_median:
        rows.append("→ **Promote 26F-v3 to production.** Cleaner v3 panel + events confirmed; "
                    "26G capacity bump did not unlock additional lift.")
    elif c2 and c2["best"] >= 1.85:
        rows.append(f"→ **Stay on 26F-v2 production (med 2.15×).** v3 panel did not add lift; "
                    "sanitizer + formula patches are defensive only at current encoder.")
    else:
        rows.append("→ **Investigate 26C2-v3 baseline failure first.** v3 panel may have introduced "
                    "regression; compare panel hashes vs v2 and check column join order.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(rows), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
