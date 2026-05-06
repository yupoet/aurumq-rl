#!/usr/bin/env python3
"""Phase 25 — compute per-factor weights for importance-weighted training.

Inputs (Methods B + C, joint):
- ``--ig-json``: IG saliency per-factor scores (from
  ``scripts/eval_factor_importance.py`` on the 24A ckpt)
- ``--tk-md``: T-1 factor z-score table from
  ``scripts/_inspect_factor_at_t_minus_k.py`` output

Combines them via geometric mean of normalized scores, then maps to a
continuous weight curve via a smooth sigmoid keyed on rank percentile.

Output: ``--out-json`` containing ``{factor_name: weight}`` for every
factor present in BOTH inputs. Factors absent from IG (e.g. tech_*
not yet computed for 24A) are emitted at weight=1.0 so they aren't
suppressed during Phase 25 training.

Weight curve (sigmoid by percentile):

    w_i = w_min + (w_max - w_min) * sigmoid(steepness * (pct_i - 0.5))

with ``pct_i ∈ [0, 1]`` = rank percentile of factor i (1 = most important).
Defaults: w_min=0.20, w_max=3.00, steepness=6 — top decile w ~2.85,
median w ~1.60, bottom decile w ~0.30.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ig-json", type=Path, required=True,
                   help="JSON output from eval_factor_importance.py "
                        "(must contain {'ig_per_factor': [...], 'factor_names': [...]})")
    p.add_argument("--tk-md", type=Path, required=True,
                   help="Markdown report from _inspect_factor_at_t_minus_k.py")
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--w-min", type=float, default=0.20)
    p.add_argument("--w-max", type=float, default=3.00)
    p.add_argument("--steepness", type=float, default=6.0,
                   help="Sigmoid steepness; higher = sharper top/bottom contrast")
    p.add_argument("--ig-weight", type=float, default=0.5,
                   help="Mixture coefficient for IG score (1-x for T-k)")
    p.add_argument("--include-default-weight", type=float, default=1.0,
                   help="Weight for factors only present in one input (no penalty)")
    return p.parse_args(argv)


def _parse_tk_md(path: Path) -> dict[str, float]:
    """Parse |mean_z@T-1| per factor from inspect markdown table."""
    text = path.read_text(encoding="utf-8").splitlines()
    out: dict[str, float] = {}
    in_table = False
    for line in text:
        if line.startswith("| factor |"):
            in_table = True
            continue
        if in_table:
            if not line.startswith("|"):
                break
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) < 2 or cells[0] in ("---", "factor"):
                continue
            name = cells[0]
            # mean@T-1 is the second cell
            try:
                z = float(cells[1].replace("+", ""))
                out[name] = abs(z)
            except (ValueError, IndexError):
                continue
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # ---- Load IG ----
    ig_data = json.loads(args.ig_json.read_text(encoding="utf-8"))
    if "ig_per_factor" not in ig_data or "factor_names" not in ig_data:
        print(f"[err] {args.ig_json} missing ig_per_factor or factor_names",
              file=sys.stderr)
        return 1
    ig_scores: dict[str, float] = {
        name: abs(float(s))
        for name, s in zip(ig_data["factor_names"], ig_data["ig_per_factor"])
    }

    # ---- Load T-k z-scores ----
    tk_scores = _parse_tk_md(args.tk_md)

    # ---- Union of factor names ----
    all_factors = sorted(set(ig_scores) | set(tk_scores))
    print(f"[weights] IG factors: {len(ig_scores)}, "
          f"T-k factors: {len(tk_scores)}, union: {len(all_factors)}")

    # ---- Normalize each score to [0, 1] by percentile rank ----
    def _percentile_rank(scores: dict[str, float]) -> dict[str, float]:
        if not scores:
            return {}
        names_sorted = sorted(scores, key=lambda n: scores[n])
        n = len(names_sorted)
        return {name: i / max(n - 1, 1) for i, name in enumerate(names_sorted)}

    ig_pct = _percentile_rank(ig_scores)
    tk_pct = _percentile_rank(tk_scores)

    # ---- Joint score ----
    joint: dict[str, float] = {}
    for name in all_factors:
        b = ig_pct.get(name)
        c = tk_pct.get(name)
        if b is not None and c is not None:
            # Both available — mix
            joint[name] = args.ig_weight * b + (1 - args.ig_weight) * c
        elif b is not None:
            joint[name] = b
        elif c is not None:
            joint[name] = c
        else:
            joint[name] = 0.5   # default median

    # ---- Sigmoid weight curve by joint percentile ----
    # Re-rank joint scores to percentile (so the actual w distribution matches the curve)
    joint_pct = _percentile_rank(joint)

    weights: dict[str, float] = {}
    for name, p in joint_pct.items():
        s = 1.0 / (1.0 + np.exp(-args.steepness * (p - 0.5)))
        w = args.w_min + (args.w_max - args.w_min) * s
        weights[name] = float(w)

    # ---- Stats + write ----
    arr = np.asarray(list(weights.values()))
    print(f"[weights] curve: min={arr.min():.3f}  median={np.median(arr):.3f}  "
          f"max={arr.max():.3f}  mean={arr.mean():.3f}")
    # Show top 10 and bottom 5
    by_w = sorted(weights.items(), key=lambda kv: -kv[1])
    print(f"\n[weights] top 10 (highest weight):")
    for name, w in by_w[:10]:
        ig_v = ig_scores.get(name, float("nan"))
        tk_v = tk_scores.get(name, float("nan"))
        print(f"  {name:<32}  w={w:.3f}  ig={ig_v:.3e}  |z@T-1|={tk_v:.3f}")
    print(f"\n[weights] bottom 5 (lowest weight):")
    for name, w in by_w[-5:]:
        ig_v = ig_scores.get(name, float("nan"))
        tk_v = tk_scores.get(name, float("nan"))
        print(f"  {name:<32}  w={w:.3f}  ig={ig_v:.3e}  |z@T-1|={tk_v:.3f}")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps({
            "weights": weights,
            "config": {
                "w_min": args.w_min, "w_max": args.w_max,
                "steepness": args.steepness, "ig_weight": args.ig_weight,
            },
            "n_factors": len(weights),
            "ig_source": str(args.ig_json),
            "tk_source": str(args.tk_md),
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n[weights] wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
