"""Compose tier-specific include_columns files from your 23A 353-col base.

Usage::

    python tools/generate_include_files.py \\
        --base configs/include_columns_phase23a_353.txt

Reads your 23A list (must exist) and emits four output files:

  configs/include_columns_phase26c2_353.txt   = base verbatim (sanity check)
  configs/include_columns_phase26e_355.txt    = base + extras_26e_2cols.txt
  configs/include_columns_phase26f_361.txt    = base + extras_26f_8cols.txt

Aborts non-zero if base has duplicates or wrong row count.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def read_lines(p: Path) -> list[str]:
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base",
        type=Path,
        default=Path("configs/include_columns_phase23a_353.txt"),
        help="Your existing 23A 353-col include list (source of truth).",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("configs"))
    a = ap.parse_args()

    if not a.base.exists():
        print(f"[!] base list missing: {a.base}", file=sys.stderr)
        print(
            "    Provide your 23A 353-col include list. The Phase 26 handoff "
            "doesn't ship it (it lives on RL side)."
        )
        return 1

    base = read_lines(a.base)
    if len(set(base)) != len(base):
        print(f"[!] base has duplicates: {len(base)} lines, {len(set(base))} unique", file=sys.stderr)
        return 1
    print(f"[base] {a.base}: {len(base)} cols")

    extras_26e = read_lines(a.out_dir / "extras_26e_2cols.txt")
    extras_26f = read_lines(a.out_dir / "extras_26f_8cols.txt")

    a.out_dir.mkdir(parents=True, exist_ok=True)

    # 26C2 = base verbatim (rename for clarity)
    out_c2 = a.out_dir / "include_columns_phase26c2_353.txt"
    out_c2.write_text("\n".join(base) + "\n", encoding="utf-8")
    print(f"[26C2] wrote {out_c2}: {len(base)} cols")

    # 26E = base + 2 extras
    list_e = base + [c for c in extras_26e if c not in set(base)]
    out_e = a.out_dir / "include_columns_phase26e_355.txt"
    out_e.write_text("\n".join(list_e) + "\n", encoding="utf-8")
    print(f"[26E]  wrote {out_e}: {len(list_e)} cols (+{len(list_e) - len(base)})")

    # 26F = base + 8 extras
    list_f = base + [c for c in extras_26f if c not in set(base)]
    out_f = a.out_dir / "include_columns_phase26f_361.txt"
    out_f.write_text("\n".join(list_f) + "\n", encoding="utf-8")
    print(f"[26F]  wrote {out_f}: {len(list_f)} cols (+{len(list_f) - len(base)})")

    if len(list_e) != len(base) + 2 or len(list_f) != len(base) + 8:
        print(f"[!] expected base+2 / base+8, got {len(list_e)} / {len(list_f)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
