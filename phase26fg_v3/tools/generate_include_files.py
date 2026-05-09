"""Compose tier-specific include_columns from your 23A 353-col base.

Same as phase26ef but produces 26C2/F/G v3 names.

Usage::

    cp YOUR_23A_353.txt configs/include_columns_phase23a_353.txt
    python tools/generate_include_files.py

Outputs:
  configs/include_columns_phase26c2_353.txt   = base verbatim
  configs/include_columns_phase26f_361.txt    = base + extras_26f_v3_8cols.txt (8 cols)
  (26G uses 26F's include list — encoder differs only)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def read_lines(p: Path) -> list[str]:
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=Path("configs/include_columns_phase23a_353.txt"))
    ap.add_argument("--out-dir", type=Path, default=Path("configs"))
    a = ap.parse_args()

    if not a.base.exists():
        print(f"[!] base list missing: {a.base}", file=sys.stderr)
        print("    Provide your 23A 353-col include list. Phase 26 handoff doesn't ship it.")
        return 1

    base = read_lines(a.base)
    if len(set(base)) != len(base):
        print(f"[!] base has duplicates: {len(base)} lines, {len(set(base))} unique", file=sys.stderr)
        return 1
    print(f"[base] {a.base}: {len(base)} cols")

    extras = read_lines(a.out_dir / "extras_26f_v3_8cols.txt")
    a.out_dir.mkdir(parents=True, exist_ok=True)

    # 26C2 = base verbatim
    out_c2 = a.out_dir / "include_columns_phase26c2_353.txt"
    out_c2.write_text("\n".join(base) + "\n", encoding="utf-8")
    print(f"[26C2-v3] wrote {out_c2}: {len(base)} cols")

    # 26F = base + 8 extras (2 curated tech + 6 events_decay10)
    list_f = base + [c for c in extras if c not in set(base)]
    out_f = a.out_dir / "include_columns_phase26f_361.txt"
    out_f.write_text("\n".join(list_f) + "\n", encoding="utf-8")
    print(f"[26F-v3]  wrote {out_f}: {len(list_f)} cols (+{len(list_f) - len(base)})")

    if len(list_f) != len(base) + 8:
        print(f"[!] expected base + 8, got {len(list_f)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
