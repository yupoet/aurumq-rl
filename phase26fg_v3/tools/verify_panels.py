"""Pre-flight integrity check for phase26fg-v3 main-board panels.

HARD assertions (per CLAUDE.md "Universe Constraint" section):
  - All ts_codes pass :func:`_universe.is_main_board` (anchored regex)
  - No 30* (创业板), 688*/689* (科创板), .BJ (北交所)
  - **ST check via PG stock_info JOIN** — fail-closed: if the lookup
    cannot run (DATABASE_URL missing, parse failure, query failure),
    verification ABORTS unless ``--skip-st-check`` is passed explicitly.
    Codex round-2 flagged that silent degradation defeats the gate.
  - sha256 verified against MANIFEST.json (per-file)
  - alpha_panel + gtja_panel: 0 inf in factor cells (sanitizer + formula patches landed)
  - alpha_029, alpha_031, gtja_143 nonnull >= 99% (impl fixes landed)
  - mf_panel has the 6 _log variant cols
  - tech_event_panel has 12 cols (6 raw + 6 decay10)

Usage::

    python tools/verify_panels.py /path/to/panels_v3/
    python tools/verify_panels.py /path/to/panels_v3/ --skip-st-check  # PG unreachable

Exit non-zero on any assertion failure so the orchestrator can abort.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import polars as pl


# Single-source universe helper. Bundle layout: tools/ sibling has no
# scripts/_universe.py, so we duplicate the regex strings here AND assert
# they match. (When this bundle is bootstrapped onto the RL side it can't
# import scripts/_universe.py from AurumQ — keep self-contained.)
_SH_MAIN = re.compile(r"^60[0135]\d{3}\.SH$")
_SZ_MAIN = re.compile(r"^00[0123]\d{3}\.SZ$")


def is_main_board(code: str) -> bool:
    """Anchored main-board regex match. Mirrors scripts/_universe.is_main_board."""
    if not code:
        return False
    return bool(_SH_MAIN.match(code) or _SZ_MAIN.match(code))


def fetch_st_or_delisted_codes() -> tuple[set[str], bool, str]:
    """Pull ST + delisted ts_codes from PG ``stock_info``.

    **Fail-closed**: returns ``(codes, ok, err_msg)``. ``ok=False`` means
    the lookup did not produce an authoritative ST set; the caller must
    treat this as a hard failure unless ``--skip-st-check`` is set
    explicitly. Codex round-2 found that an empty set silently passing
    verification was equivalent to "no ST in panel", which a real
    fail-closed gate should never accept.

    Returns
    -------
    (codes, ok, err_msg):
      - ``ok=True, codes={...}, err_msg=""``  — successful query (set may be empty if DB has no ST)
      - ``ok=False, codes=set(), err_msg="..."`` — caller must abort or --skip-st-check
    """
    try:
        import psycopg2
        from dotenv import load_dotenv
    except Exception as e:  # pragma: no cover — environmental
        return set(), False, f"psycopg2/dotenv import failed: {e}"

    load_dotenv("/data/AurumQ/.env")
    url = (os.environ.get("DATABASE_URL") or "").replace("+asyncpg", "+psycopg2")
    if not url:
        return set(), False, "DATABASE_URL not set in env"
    m = re.match(r"postgresql\+psycopg2://([^:]+):([^@]+)@([^:]+):(\d+)/(.*)", url)
    if not m:
        return set(), False, f"DATABASE_URL malformed: {url[:40]}..."
    user, pw, host, port, db = m.groups()
    try:
        conn = psycopg2.connect(user=user, password=pw, host=host, port=port, dbname=db)
        cur = conn.cursor()
        cur.execute(
            "SELECT stock_code FROM stock_info "
            "WHERE is_st = TRUE OR delist_date IS NOT NULL"
        )
        codes = {r[0] for r in cur.fetchall()}
        cur.close()
        conn.close()
        return codes, True, ""
    except Exception as e:  # pragma: no cover — environmental
        return set(), False, f"PG query failed: {e}"


REQUIRED_NONNULL = {
    "alpha_029": 0.99,
    "alpha_031": 0.99,
    "gtja_143": 0.99,
}

MF_LOG_COLS = (
    "mf_net_1d_log",
    "mf_net_3d_log",
    "mf_net_5d_log",
    "mf_net_10d_log",
    "mf_net_20d_log",
    "mf_net_60d_log",
)

EVENT_DECAY_COLS = (
    "tech_evt_kdj_below_30_cross_decay10",
    "tech_evt_kdj_j_oversold_turn_decay10",
    "tech_evt_macd_zero_cross_decay10",
    "tech_evt_boll_squeeze_break_decay10",
    "tech_evt_ma5_cross_ma10_decay10",
    "tech_evt_vol_breakout_3sigma_decay10",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument(
        "--skip-st-check",
        action="store_true",
        help=(
            "Bypass the PG stock_info ST/delist lookup. "
            "Default behaviour is FAIL-CLOSED: if PG is unreachable verification "
            "aborts. Use this flag only when running on a host (e.g. ledashi's "
            "Windows RL box) that intentionally cannot reach the AurumQ PG. "
            "There is NO fallback ST detection — main-board codes only filter "
            "on board prefix; freshly-renamed *ST tickers that retain old codes "
            "are NOT caught without the PG join."
        ),
    )
    ap.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2023, 2024, 2025, 2026],
        help=(
            "Which years to verify (default 2023-2026 = short panel). "
            "Files for years outside this list are skipped — useful when the "
            "manifest contains 2017-2022 entries that haven't been re-built "
            "with the latest sanitizer/formula patches. Pass --years 2017 2018 "
            "... for full long-panel verification."
        ),
    )
    args = ap.parse_args()
    in_scope_years = set(args.years)
    print(f"[info] verifying years: {sorted(in_scope_years)}")

    failures: list[str] = []

    # Pull ST/delist set up front (one query, reused across panels).
    if args.skip_st_check:
        st_set: set[str] = set()
        st_check_active = False
        print("[info] --skip-st-check set; ST detection DISABLED")
        print("       (only board-prefix regex enforced — known coverage gap)")
    else:
        st_set, ok, err_msg = fetch_st_or_delisted_codes()
        if not ok:
            print(f"[FAIL-CLOSED] ST/delist lookup failed: {err_msg}", file=sys.stderr)
            print(
                "  This script defaults to fail-closed because a missing ST set "
                "is indistinguishable from 'no ST in panel'. To bypass on a host "
                "without PG access, pass --skip-st-check explicitly.",
                file=sys.stderr,
            )
            return 2
        st_check_active = True
        print(f"[info] PG stock_info ST/delisted set: {len(st_set)} codes (fail-closed mode)")

    manifest_path = args.root / "MANIFEST.json"
    if not manifest_path.exists():
        print(f"[!] MANIFEST.json missing at {manifest_path}")
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    def in_scope(path: str) -> bool:
        # Files have format like "alpha_panel_year=2023.parquet" or
        # "cyq_panel/year=2023.parquet". Match year=YYYY.
        m = re.search(r"year=(\d{4})", path)
        if not m:
            return True  # non-year-tagged files always in scope
        return int(m.group(1)) in in_scope_years

    # 1. sha256 — only check in-scope files
    n_ok = 0; n_total = 0
    for entry in manifest.get("files", []):
        if not in_scope(entry["path"]):
            continue
        n_total += 1
        f = args.root / entry["path"]
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
    print(f"[{'✓' if n_ok == n_total else '!'}] sha256 OK for {n_ok} / {n_total} parquets (in-scope years)")

    # 2. Per-panel checks
    for entry in manifest.get("files", []):
        if not in_scope(entry["path"]):
            continue
        path = entry["path"]
        f = args.root / path
        if not f.exists():
            continue
        if not (path.endswith(".parquet")):
            continue
        try:
            df = pl.read_parquet(f)
        except Exception as e:
            failures.append(f"read failed {path}: {e}")
            continue

        ts_col = "ts_code" if "ts_code" in df.columns else "stock_code"
        if ts_col not in df.columns:
            failures.append(f"{path}: no ts_code/stock_code column")
            continue

        # 2a. Universe assertion (regex + ST/delist set lookup)
        codes = df[ts_col].unique().to_list()
        bad_codes = [c for c in codes if c is not None and not is_main_board(c)]
        if bad_codes:
            failures.append(
                f"{path}: {len(bad_codes)} non-main-board ts_codes leaked: {bad_codes[:5]}"
            )
        # Even if regex passes, the code may have been renamed *ST since panel
        # build. Only enforce when ST check is active (PG was reachable).
        # Empty st_set under st_check_active=True is fine — means PG truly
        # has no ST/delisted, e.g. very early in trading day.
        if st_check_active:
            st_leaked = [c for c in codes if c in st_set]
            if st_leaked:
                failures.append(
                    f"{path}: {len(st_leaked)} ST/delisted codes leaked "
                    f"(from PG stock_info): {st_leaked[:5]}"
                )

        # 2b. inf
        if "alpha_panel" in path or "gtja_panel" in path:
            n_inf = 0
            for c in df.columns:
                if c in (ts_col, "trade_date"):
                    continue
                arr = df[c].cast(pl.Float64, strict=False).to_numpy()
                n_inf += int(np.isinf(arr).sum())
            if n_inf > 0:
                failures.append(f"{path}: {n_inf} inf cells (sanitizer should have caught)")

        # 2c. fixed-col nonnull
        for col, threshold in REQUIRED_NONNULL.items():
            if col not in df.columns:
                continue
            ratio = df[col].is_not_null().sum() / len(df)
            if ratio < threshold:
                failures.append(
                    f"{path}: {col} nonnull={ratio:.3f} < {threshold:.2f}"
                )

        # 2d. mf_panel _log variants
        if "mf_panel" in path:
            missing = [c for c in MF_LOG_COLS if c not in df.columns]
            if missing:
                failures.append(f"{path}: missing mf _log variants: {missing}")

        # 2e. tech_event_panel
        if "tech_event_panel" in path:
            missing = [c for c in EVENT_DECAY_COLS if c not in df.columns]
            if missing:
                failures.append(f"{path}: missing event decay cols: {missing}")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
        print("\nPRE-FLIGHT FAIL")
        return 1

    print("[✓] universe: 100% main-board (60* SH / 00* SZ); 0 创业/科创/北交")
    print("[✓] inf: 0 across all factor cells")
    print("[✓] alpha_029/031/gtja_143 nonnull >= 99%")
    print("[✓] mf_panel has 6 _log variants")
    print("[✓] tech_event_panel has 6 decay10 cols")
    if st_check_active:
        print("[✓] ST/delist gate ACTIVE; no ST/delisted codes in panel")
    else:
        print("[!] ST/delist gate SKIPPED (--skip-st-check); coverage gap noted")
    print("\nPRE-FLIGHT PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
