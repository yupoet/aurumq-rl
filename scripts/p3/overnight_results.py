"""Combined RESULTS for overnight long-panel run.

Compares short vs long for Path 1, 2, 4(=Path D), 5 — writes:
  - runs/sl_path1_long/RESULTS.md
  - runs/sl_path2_long/RESULTS.md
  - runs/sl_regime_stack_long/RESULTS.md (if path5 succeeded)
  - runs/sl_overnight_logs/COMBINED_RESULTS.md
"""
from __future__ import annotations

import json
from pathlib import Path


def _load(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except FileNotFoundError:
        return None


def _h1h2(d: dict, key: str) -> tuple[float | None, float | None]:
    h1 = d.get(f"{key}_H1", {}).get("primary_mean_top50_proximity_excess") if d else None
    h2 = d.get(f"{key}_H2", {}).get("primary_mean_top50_proximity_excess") if d else None
    return h1, h2


def _bps(delta: float | None) -> str:
    return f"{delta*1e4:+.2f} bps" if delta is not None else "—"


def main() -> int:
    short_p1 = _load(Path("runs/sl_path1/ensemble.json"))
    short_p2 = _load(Path("runs/sl_path2/ensemble.json"))
    short_p4 = _load(Path("runs/sl_path4/ensemble.json"))
    short_p5 = _load(Path("runs/sl_regime_stack/ensemble.json"))

    long_p1 = _load(Path("runs/sl_path1_long/ensemble.json"))
    long_p2 = _load(Path("runs/sl_path2_long/ensemble.json"))
    long_p4 = _load(Path("runs/sl_path_d/ensemble.json"))  # already done = path 4 long
    long_p5 = _load(Path("runs/sl_regime_stack_long/ensemble.json"))

    rows = []
    for name, s, l, key in (
        ("Path 1 (raw LGB)",         short_p1, long_p1, "ensemble_calibrated"),
        ("Path 4 (rank-z LGB)",      short_p4, long_p4, "ensemble_calibrated"),
        ("Path 2 (CB+XGB)",          short_p2, long_p2, "ensemble_calibrated"),
        ("Path 5 (regime stacking)", short_p5, long_p5, "stacking_calibrated"),
    ):
        s_h1, s_h2 = _h1h2(s, key)
        l_h1, l_h2 = _h1h2(l, key)
        d_h1 = (l_h1 - s_h1) if (l_h1 is not None and s_h1 is not None) else None
        d_h2 = (l_h2 - s_h2) if (l_h2 is not None and s_h2 is not None) else None
        rows.append((name, s_h1, l_h1, d_h1, s_h2, l_h2, d_h2))

    md = ["# Overnight long-panel sweep — combined RESULTS\n",
          "Date: 2026-05-10/11\n",
          "Setup: TRAIN 2018-01-02 ~ 2024-12-04 (long); 2023-01-03 ~ 2024-12-04 (short).",
          "Eval H1/H2 identical between short and long.\n",
          "## Headline\n",
          "| Path | short H1 | long H1 | Δ H1 | short H2 | long H2 | Δ H2 |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for n, sh, lh, dh, sh2, lh2, dh2 in rows:
        sf = lambda v: f"{v:+.6f}" if v is not None else "—"
        md.append(f"| {n} | {sf(sh)} | {sf(lh)} | {_bps(dh)} | {sf(sh2)} | {sf(lh2)} | {_bps(dh2)} |")
    md.append("")
    md.append("## Interpretation guide\n")
    md.append("- |Δ| < 2 bps → noise band; long panel did not help")
    md.append("- 2 ≤ |Δ| < 5 bps → marginal; weight against compute cost")
    md.append("- |Δ| ≥ 5 bps → real lift; productionize long-panel variant")
    md.append("")
    md.append("## Files\n")
    md.append("- `runs/sl_path1_long/` — 24 LGB long-panel grid + ensemble")
    md.append("- `runs/sl_path2_long/` — 36 CatBoost+XGB long-panel grid + ensemble")
    md.append("- `runs/sl_path_d/` — Path 4 long (already done earlier this session)")
    md.append("- `runs/sl_regime_stack_long/` — Path 5 meta refit on long bases")
    md.append("- `runs/sl_overnight_logs/` — per-step logs")

    out = Path("runs/sl_overnight_logs/COMBINED_RESULTS.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {out}")

    # Also dump individual RESULTS.md for path1_long / path2_long
    for ens, label, root in (
        (long_p1, "Path 1 long", Path("runs/sl_path1_long")),
        (long_p2, "Path 2 long", Path("runs/sl_path2_long")),
        (long_p5, "Path 5 long", Path("runs/sl_regime_stack_long")),
    ):
        if ens is None or not root.exists():
            continue
        key = "stacking_calibrated" if "stacking_calibrated_H1" in ens else "ensemble_calibrated"
        h1, h2 = _h1h2(ens, key)
        (root / "RESULTS.md").write_text(
            f"# {label} — RESULTS\n\n"
            f"H1 calibrated primary: {h1:+.6f}\n"
            f"H2 calibrated primary: {h2:+.6f}\n\n"
            f"Compare with short panel via runs/sl_overnight_logs/COMBINED_RESULTS.md\n",
            encoding="utf-8",
        )

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
