"""v33 task 1 — NPF anchor cross-paradigm test.

paris 6/3+ deferred research priority, promoted P0 post-v13 (paradigm 3 failed).

Question: on NPF universe, does paradigm 2 (anchor) beat paradigm 1 (proximity / sparse binary)?

Reads existing matrix outputs:
- matrix_v10b_results.json (paradigm 1 proximity, target_y label)
- matrix_v10c_results.json (paradigm 1 binary dense P75)
- matrix_v11_results.json (paradigm 1 binary sparse 0.8% pos)
- matrix_v12_results.json (paradigm 2 anchor α/β)

Outputs apples-to-apples comparison per (panel, horizon).
"""
from __future__ import annotations
import json
from pathlib import Path

import pandas as pd

OUT = Path("data/kronos/outputs/matrix_v33_npf_cross_paradigm")
OUT.mkdir(parents=True, exist_ok=True)


def load(p: Path) -> dict:
    if not p.exists(): return {}
    return json.loads(p.read_text()).get("results", {})


def extract_h2_q1_sharpe(result: dict, K: int = 50):
    """Return (h2_ic, q1_ic, sharpe_h2, sharpe_q1) for fwd20 if present."""
    static = result.get("static") if isinstance(result, dict) else None
    if not static: return None
    try:
        h2 = static.get("H2_2025", {}).get("fwd20", {})
        q1 = static.get("Q1_2026", {}).get("fwd20", {})
        s_h2 = h2.get("sizing", {}).get(str(K), {}).get("sharpe_net", float("nan"))
        s_q1 = q1.get("sizing", {}).get(str(K), {}).get("sharpe_net", float("nan"))
        return {
            "h2_ic": h2.get("ic", float("nan")) * 100,
            "q1_ic": q1.get("ic", float("nan")) * 100,
            "sharpe_h2": s_h2,
            "sharpe_q1": s_q1,
        }
    except Exception:
        return None


def main():
    matrices = {
        "p1_target_y (v10b proximity)": load(Path("data/kronos/outputs/matrix_v10b_results.json")),
        "p1_binary_dense (v10c P75)":   load(Path("data/kronos/outputs/matrix_v10c_results.json")),
        "p1_binary_sparse (v11 0.8%)":  load(Path("data/kronos/outputs/matrix_v11_results.json")),
        "p2_anchor_alpha (v12)":         load(Path("data/kronos/outputs/matrix_v12_results.json")),
        "p2_anchor_beta (v12)":          load(Path("data/kronos/outputs/matrix_v12_results.json")),
    }

    # Collect NPF cells per matrix
    npf_cells = {}
    for matrix_name, results in matrices.items():
        for cell_id, result in results.items():
            if "_NPF_" not in cell_id and not cell_id.endswith("_NPF"):
                continue
            if cell_id.startswith("alpha_") and "p2_anchor_alpha" not in matrix_name: continue
            if cell_id.startswith("beta_") and "p2_anchor_beta" not in matrix_name: continue
            stats = extract_h2_q1_sharpe(result)
            if stats is None: continue
            npf_cells.setdefault(matrix_name, []).append({
                "cell_id": cell_id,
                **stats,
            })

    print("=== NPF cross-paradigm summary ===\n")
    summary_rows = []
    for matrix_name, cells in npf_cells.items():
        cells_sorted = sorted(cells, key=lambda c: (c["h2_ic"] + max(c["q1_ic"], 0)), reverse=True)
        top = cells_sorted[:5]
        print(f"\n--- {matrix_name} ---")
        print(f"{'cell':40s} {'H2 IC%':>8s} {'Q1 IC%':>8s} {'Sharpe_H2':>10s} {'Sharpe_Q1':>10s}")
        for c in top:
            print(f"{c['cell_id']:40s} {c['h2_ic']:+8.3f} {c['q1_ic']:+8.3f} {c['sharpe_h2']:+10.2f} {c['sharpe_q1']:+10.2f}")
        if top:
            best = top[0]
            summary_rows.append({"paradigm_method": matrix_name, "best_cell": best["cell_id"], **{k: v for k, v in best.items() if k != "cell_id"}})

    # Save summary
    out_path = OUT / "npf_cross_paradigm_summary.json"
    out_path.write_text(json.dumps({
        "task": "v33 task 1 — NPF anchor cross-paradigm test",
        "scope": "Compare paradigm 1 (proximity / binary dense / binary sparse) vs paradigm 2 (anchor α/β) on NPF universe",
        "n_cells_per_matrix": {k: len(v) for k, v in npf_cells.items()},
        "summary": summary_rows,
        "raw_cells": npf_cells,
    }, indent=2, default=str))
    print(f"\n[saved] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
