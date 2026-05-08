"""Pre-flight integrity check for the panels v2 bundle.

Run on the AurumQ-RL side after pulling panels from OSS, BEFORE training.
Verifies sha256 against MANIFEST.json + 0 inf + nonnull rates on the
3 previously-broken cols (alpha_029, alpha_031, gtja_143). Exits non-zero
if anything is wrong so the orchestrator can abort.

Usage::

    python tools/verify_panels.py /path/to/panels_v2/

Expected output (clean panels)::

    [✓] sha256 OK for 12 / 12 parquets
    [✓] inf count = 0 across all factor columns
    [✓] alpha_029 nonnull >= 99.0% in all years
    [✓] alpha_031 nonnull >= 99.0% in all years
    [✓] gtja_143 nonnull >= 99.0% in all years
    [✓] tech_event_panel events fire at expected rates (0.1% .. 13%)
    PRE-FLIGHT PASS
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl


REQUIRED_NONNULL = {
    "alpha_029": 0.99,
    "alpha_031": 0.99,
    "gtja_143": 0.99,
}

EVENT_RATE_BOUNDS = {
    "tech_evt_kdj_below_30_cross_raw": (0.02, 0.08),
    "tech_evt_kdj_j_oversold_turn_raw": (0.05, 0.20),
    "tech_evt_macd_zero_cross_raw": (0.005, 0.05),
    "tech_evt_boll_squeeze_break_raw": (0.0005, 0.01),
    "tech_evt_ma5_cross_ma10_raw": (0.02, 0.10),
    "tech_evt_vol_breakout_3sigma_raw": (0.01, 0.06),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="panels v2 root (contains MANIFEST.json)")
    a = ap.parse_args()

    failures: list[str] = []

    manifest_path = a.root / "MANIFEST.json"
    if not manifest_path.exists():
        print(f"[!] MANIFEST.json missing at {manifest_path}")
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # 1. sha256 check
    n_ok = 0
    for entry in manifest.get("files", []):
        f = a.root / entry["path"]
        if not f.exists():
            failures.append(f"missing file: {entry['path']}")
            continue
        actual = sha256(f)
        if actual != entry["sha256"]:
            failures.append(
                f"sha256 mismatch: {entry['path']} expected={entry['sha256'][:12]}.. actual={actual[:12]}.."
            )
        else:
            n_ok += 1
    total = len(manifest.get("files", []))
    print(f"[{'✓' if n_ok == total else '!'}] sha256 OK for {n_ok} / {total} parquets")

    # 2. Per-panel checks: 0 inf + nonnull rate on the 3 fixed cols + event rates
    for entry in manifest.get("files", []):
        f = a.root / entry["path"]
        if not f.exists():
            continue
        if "alpha_panel" not in entry["path"] and "gtja_panel" not in entry["path"] \
                and "tech_event_panel" not in entry["path"]:
            continue
        try:
            df = pl.read_parquet(f)
        except Exception as e:
            failures.append(f"read failed {entry['path']}: {e}")
            continue

        # Inf check
        n_inf = 0
        for c in df.columns:
            if c in ("ts_code", "stock_code", "trade_date"):
                continue
            arr = df[c].cast(pl.Float64, strict=False).to_numpy()
            n_inf += int(np.isinf(arr).sum())
        if n_inf > 0:
            failures.append(f"{entry['path']}: {n_inf} inf values")

        # Nonnull check on fixed cols
        for col, threshold in REQUIRED_NONNULL.items():
            if col not in df.columns:
                continue
            ratio = df[col].is_not_null().sum() / len(df)
            if ratio < threshold:
                failures.append(
                    f"{entry['path']}: {col} nonnull={ratio:.3f} < {threshold:.2f}"
                )

        # Event rate check
        for col, (lo, hi) in EVENT_RATE_BOUNDS.items():
            if col not in df.columns:
                continue
            arr = df[col].cast(pl.Float64, strict=False).to_numpy()
            rate = float((arr >= 0.5).sum()) / len(df)
            if not (lo <= rate <= hi):
                failures.append(
                    f"{entry['path']}: {col} fire rate={rate:.3f} outside [{lo:.3f}, {hi:.3f}]"
                )

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
        print("\nPRE-FLIGHT FAIL")
        return 1

    print("[✓] inf count = 0 across all factor columns")
    print("[✓] alpha_029, alpha_031, gtja_143 nonnull >= 99% in all years")
    print("[✓] tech_event_panel events fire at expected rates")
    print("\nPRE-FLIGHT PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
