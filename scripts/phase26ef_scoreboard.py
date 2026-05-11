"""Aggregate Phase 26E/F/G runs into a markdown scoreboard.

Reads each ``runs/<tier>_seed<N>/episode_eval.json`` and produces a
single-table summary with median / best T-1 lift per tier vs the 26C2
baseline. Also surfaces convergence step (best ckpt) for each run.

Usage::

    python phase26ef_scoreboard.py \\
        --runs-dir runs \\
        --tiers 26C2 26E 26F 26G \\
        --seeds 42 43 44 \\
        --baseline-lift 2.61 \\
        --out runs/phase26ef_scoreboard.md
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def load_eval_json(run_dir: Path) -> dict:
    p = run_dir / "episode_eval.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def best_lift_top5(eval_json: dict) -> tuple[float, float, str]:
    """Return (best_t1_lift, best_t1_hit, best_ckpt_label) for top_k=5 across all checkpoints."""
    rows = eval_json.get("checkpoint_eval_results", []) or eval_json.get("rows", [])
    best = (-1.0, 0.0, "n/a")
    for r in rows:
        if r.get("top_k") != 5:
            continue
        lift = r.get("t1_lift_over_base") or 0.0
        if lift > best[0]:
            hit = r.get("t_minus_1_hit_rate") or 0.0
            best = (float(lift), float(hit), r.get("checkpoint_label", "?"))
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path, default=Path("runs"))
    ap.add_argument("--tiers", nargs="+", default=["26C2", "26E", "26F", "26G"])
    ap.add_argument("--seeds", nargs="+", default=["42", "43", "44"])
    ap.add_argument("--baseline-lift", type=float, default=2.61)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows: list[str] = []
    rows.append("# Phase 26 E/F/G Scoreboard\n")
    rows.append(f"Baseline (26C2 published): **{args.baseline_lift:.2f}× T-1 lift**\n\n")
    rows.append("## Per-tier summary (top_k=5)\n")
    rows.append(
        "| tier | best lift | best hit | best ckpt | median (3 seeds) | min seed | max seed | vs 26C2 |"
    )
    rows.append(
        "|------|----------:|---------:|-----------|----------------:|--------:|--------:|--------:|"
    )

    tier_summaries: dict[str, dict] = {}
    for tier in args.tiers:
        per_seed_lift: list[float] = []
        per_seed_hit: list[float] = []
        ckpts: list[str] = []
        for seed in args.seeds:
            run_dir = args.runs_dir / f"{tier}_seed{seed}"
            ev = load_eval_json(run_dir)
            if not ev:
                continue
            lift, hit, ckpt = best_lift_top5(ev)
            per_seed_lift.append(lift)
            per_seed_hit.append(hit)
            ckpts.append(ckpt)
        if not per_seed_lift:
            rows.append(f"| {tier} | NO DATA | | | | | | |")
            continue
        med = statistics.median(per_seed_lift)
        best_idx = max(range(len(per_seed_lift)), key=lambda i: per_seed_lift[i])
        best_lift = per_seed_lift[best_idx]
        best_hit = per_seed_hit[best_idx]
        best_ckpt = ckpts[best_idx]
        delta = med - args.baseline_lift
        sign = "+" if delta >= 0 else ""
        rows.append(
            f"| {tier} | {best_lift:.2f}× | {best_hit * 100:.2f}% | {best_ckpt} | "
            f"{med:.2f}× | {min(per_seed_lift):.2f}× | {max(per_seed_lift):.2f}× | {sign}{delta:.2f}× |"
        )
        tier_summaries[tier] = {
            "best_lift": best_lift,
            "best_hit": best_hit,
            "best_ckpt": best_ckpt,
            "median_lift": med,
            "min_lift": min(per_seed_lift),
            "max_lift": max(per_seed_lift),
            "vs_baseline": delta,
            "n_seeds": len(per_seed_lift),
        }

    rows.append("")
    rows.append("## Pass/fail per tier\n")
    rows.append("| tier | criterion | result |")
    rows.append("|------|-----------|--------|")
    for tier in args.tiers:
        s = tier_summaries.get(tier)
        if not s:
            rows.append(f"| {tier} | n/a | NO DATA |")
            continue
        if tier == "26C2":
            ok = s["best_lift"] >= 2.5
            verdict = "PASS (sanity gate)" if ok else "FAIL — baseline cannot reproduce"
            rows.append(f"| 26C2 | best lift ≥ 2.5× | {verdict} |")
        elif tier == "26E":
            ok = s["median_lift"] >= args.baseline_lift - 0.10
            v = "PASS — curated tech non-harmful" if ok else "REJECT — curated tech still hurts"
            rows.append(
                f"| 26E | median ≥ 26C2 - 0.10× ({args.baseline_lift - 0.10:.2f}×) | {v} ({s['median_lift']:.2f}×) |"
            )
        elif tier == "26F":
            ok = (
                s["median_lift"] > args.baseline_lift + 0.10 and s["best_lift"] > args.baseline_lift
            )
            v = "PASS — events add value" if ok else "REJECT — events do not improve"
            rows.append(
                f"| 26F | median > 26C2 + 0.10× AND best > 26C2 best | {v} (med={s['median_lift']:.2f}×) |"
            )
        elif tier == "26G":
            f_med = tier_summaries.get("26F", {}).get("median_lift", 0)
            ok = s["median_lift"] > f_med + 0.10
            v = "PASS — capacity helps" if ok else "REJECT — capacity does not help further"
            rows.append(
                f"| 26G | median > 26F + 0.10× ({f_med + 0.10:.2f}×) | {v} (med={s['median_lift']:.2f}×) |"
            )

    rows.append("")
    rows.append("## Recommendation\n")
    if "26F" in tier_summaries and tier_summaries["26F"]["median_lift"] > args.baseline_lift + 0.10:
        if (
            "26G" in tier_summaries
            and tier_summaries["26G"]["median_lift"] > tier_summaries["26F"]["median_lift"] + 0.10
        ):
            rows.append(
                "→ **Promote 26G to production.** Events + larger encoder both help. "
                "Larger encoder may need more multi-seed runs before final commit."
            )
        else:
            rows.append(
                "→ **Promote 26F to production.** Event-decay tech features add measurable lift; encoder bump unnecessary."
            )
    elif (
        "26E" in tier_summaries
        and tier_summaries["26E"]["median_lift"] >= args.baseline_lift - 0.10
    ):
        rows.append(
            "→ **Stay on 26C2 production. 26E neutral, 26F/G no improvement.** "
            "Curated continuous tech is non-harmful but adds no lift; revisit with deeper feature engineering or different encoder family."
        )
    else:
        rows.append(
            "→ **Stay on 26C2 production.** None of the tiers cleared the bar. "
            "Tech-feature direction is dead-end at current encoder + target definition; "
            "consider event-driven re-rank stage instead of in-encoder integration."
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(rows), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
