"""Aggregate supplement experiments into SUPPLEMENT_RESULTS.md."""
from __future__ import annotations

import json, glob, os, statistics as s
from pathlib import Path


def load_json(p):
    try: return json.loads(Path(p).read_text())
    except FileNotFoundError: return None


def per_run_h1h2(root):
    rows = []
    for p in sorted(glob.glob(f"{root}/*/results.json")):
        d = load_json(p)
        if d:
            rows.append((os.path.basename(os.path.dirname(p)),
                         d['H1']['primary_mean_top50_proximity_excess'],
                         d['H2']['primary_mean_top50_proximity_excess']))
    return rows


def main():
    md = ["# Supplement experiments — RESULTS", "Date: 2026-05-11", ""]

    # Strategy D evals
    md.append("## Strategy D (score-weighted top-50) — re-eval\n")
    md.append("| Path | H1 mean_y | H2 mean_y |")
    md.append("|---|---:|---:|")
    for path, label in (
        ("runs/sl_path1_long/strategy_d_eval.json", "Path 1 long"),
        ("runs/sl_path_d/strategy_d_eval.json", "Path 4 long (Path D)"),
        ("runs/sl_path4/strategy_d_eval.json", "Path 4 short (prod baseline)"),
    ):
        d = load_json(path)
        if d:
            md.append(f"| {label} | {d['H1']['mean_top50_strategyD']:+.6f} | {d['H2']['mean_top50_strategyD']:+.6f} |")
    md.append("")

    # Experiment A: Path 1 ablation learning curve
    md.append("## Experiment A — Path 1 train-window ablation (sample-size learning curve)\n")
    md.append("Best-config (nl31, lr=0.030, mdl=100) trained at four train-windows:\n")
    md.append("| Train window | n train years | H1 mean (3 seeds) | H2 mean (3 seeds) |")
    md.append("|---|---:|---:|---:|")
    # Path 1 short = 2 years
    short_rows = per_run_h1h2('runs/sl_path1')
    short_nl31 = [r for r in short_rows if r[0].startswith('nl31_lr030_mdl100_')]
    if short_nl31:
        h1m = s.mean(r[1] for r in short_nl31); h2m = s.mean(r[2] for r in short_nl31)
        md.append(f"| 2023-2024 (2y, Path 1 short) | 2 | {h1m:+.6f} | {h2m:+.6f} |")
    # Ablation: 3y, 5y
    abl_rows = per_run_h1h2('runs/sl_path1_ablation')
    for prefix, label, years in (
        ("3y_2022", "2022-2024 (3y)", 3),
        ("5y_2020", "2020-2024 (5y)", 5),
    ):
        sub = [r for r in abl_rows if r[0].startswith(prefix)]
        if sub:
            h1m = s.mean(r[1] for r in sub); h2m = s.mean(r[2] for r in sub)
            md.append(f"| {label} | {years} | {h1m:+.6f} | {h2m:+.6f} |")
    # Path 1 long = 7 years
    long_rows = per_run_h1h2('runs/sl_path1_long')
    long_nl31 = [r for r in long_rows if r[0].startswith('nl31_lr030_mdl100_')]
    if long_nl31:
        h1m = s.mean(r[1] for r in long_nl31); h2m = s.mean(r[2] for r in long_nl31)
        md.append(f"| 2018-2024 (7y, Path 1 long) | 7 | {h1m:+.6f} | {h2m:+.6f} |")
    md.append("")
    md.append("**Interpretation**: monotonic ↑ → still on the up-slope, more history helps. Plateau → 7y is enough.\n")

    # Experiment B: Path 4 hyperparam + RAW long
    md.append("## Experiment B — Path 4 hyperparam + RAW long panel (rank-z hypothesis test)\n")
    p4raw = load_json('runs/sl_path4_raw_long/ensemble.json')
    p1long = load_json('runs/sl_path1_long/ensemble.json')
    p4d = load_json('runs/sl_path_d/ensemble.json')
    if p4raw:
        md.append("| Path | H1 | H2 |")
        md.append("|---|---:|---:|")
        def f(d, w): return f"{d[f'ensemble_calibrated_{w}']['primary_mean_top50_proximity_excess']:+.6f}" if d else "—"
        md.append(f"| Path 4 long (rank-z) — Path D   | {f(p4d,'H1')} | {f(p4d,'H2')} |")
        md.append(f"| **Path 4 raw long (this expt)** | {f(p4raw,'H1')} | {f(p4raw,'H2')} |")
        md.append(f"| Path 1 long (raw, original)    | {f(p1long,'H1')} | {f(p1long,'H2')} |")
        md.append("")
        md.append("**Verdict**:")
        md.append("- If Path 4 raw long ≈ Path 1 long → confirmed: rank-z destroys long-panel info.")
        md.append("- If Path 4 raw long ≈ Path D (rank-z) → not the rank-z; hyperparam interaction.")
        md.append("- If Path 4 raw long > both → new winner.\n")

    # Experiment C: Hybrid ensemble
    md.append("## Experiment C — Hybrid (Path 1 long + Path 4 short, equal-weight)\n")
    hyb = load_json('runs/sl_hybrid_p1long_p4short/ensemble.json')
    p4short = load_json('runs/sl_path4/ensemble.json')
    if hyb and p4short and p1long:
        md.append("| Method | H1 | H2 |")
        md.append("|---|---:|---:|")
        md.append(f"| Path 1 long ensemble        | {p1long['ensemble_calibrated_H1']['primary_mean_top50_proximity_excess']:+.6f} | {p1long['ensemble_calibrated_H2']['primary_mean_top50_proximity_excess']:+.6f} |")
        md.append(f"| Path 4 short ensemble       | {p4short['ensemble_calibrated_H1']['primary_mean_top50_proximity_excess']:+.6f} | {p4short['ensemble_calibrated_H2']['primary_mean_top50_proximity_excess']:+.6f} |")
        md.append(f"| **Hybrid (50/50)**          | {hyb['H1_primary']:+.6f} | {hyb['H2_primary']:+.6f} |")
        # Path 5 stack for compare
        p5l = load_json('runs/sl_regime_stack_long/ensemble.json')
        if p5l:
            md.append(f"| Path 5 stack (long bases)   | {p5l['stacking_calibrated_H1']['primary_mean_top50_proximity_excess']:+.6f} | {p5l['stacking_calibrated_H2']['primary_mean_top50_proximity_excess']:+.6f} |")
        md.append("")
        md.append("**Decision rule**: if Hybrid ≥ Path 5 stack, use Hybrid in production (no meta complexity).\n")

    out = Path("runs/sl_overnight_logs/SUPPLEMENT_RESULTS.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
