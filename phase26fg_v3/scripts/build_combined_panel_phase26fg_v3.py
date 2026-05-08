"""Build combined panel for Phase 26F/G v3 training.

Same join semantics as phase26ef builder, except:
  - panels_root defaults to ``data/panels_v3`` (the v3-FINAL-PATCHED bundle)
  - 26C2 = 353 base only (no extras)
  - 26F  = 26C2 + 2 curated tech (boll_percent, cmf_120d_pct_amt) + 6 events_decay10
  - 26G  shares 26F's panel (encoder differs only)

The output panel is filtered to MAIN BOARD by virtue of pulling from
``panels-main-board-v3`` source. verify_panels.py asserts no leakage.

Usage::

    python scripts/build_combined_panel_phase26fg_v3.py \\
        --tier 26F-v3 --panels-root data/panels_v3 --out data/panel_26f_v3.parquet

Tier names use the ``-v3`` suffix (matches argparse choices on line ~75).
Passing ``--tier 26F`` (no suffix) will fail argparse validation. The
runner ``run_phase26fg_v3_overnight.sh`` calls with the suffix form.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl


TIER_INCLUDE_FILE = {
    "26C2-v3": "configs/include_columns_phase26c2_353.txt",
    "26F-v3":  "configs/include_columns_phase26f_361.txt",
    "26G-v3":  "configs/include_columns_phase26f_361.txt",
}

TIER_EXTRA_TECH_COLS = {
    "26C2-v3": [],
    "26F-v3":  ["tech_boll_percent", "cmf_120d_pct_amt"],
    "26G-v3":  ["tech_boll_percent", "cmf_120d_pct_amt"],
}

TIER_INCLUDE_EVENTS = {
    "26C2-v3": False,
    "26F-v3":  True,
    "26G-v3":  True,
}

EVENT_DECAY_COLS = (
    "tech_evt_kdj_below_30_cross_decay10",
    "tech_evt_kdj_j_oversold_turn_decay10",
    "tech_evt_macd_zero_cross_decay10",
    "tech_evt_boll_squeeze_break_decay10",
    "tech_evt_ma5_cross_ma10_decay10",
    "tech_evt_vol_breakout_3sigma_decay10",
)


def _read_yearly(panels_root: Path, subpath: str, years: list[int]) -> pl.DataFrame:
    parts: list[pl.DataFrame] = []
    for y in years:
        p = panels_root / subpath.format(year=y)
        if not p.exists():
            raise SystemExit(f"missing: {p}")
        parts.append(pl.scan_parquet(p).collect())
    return pl.concat(parts, how="diagonal_relaxed")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=list(TIER_INCLUDE_FILE.keys()), required=True)
    ap.add_argument("--panels-root", type=Path, default=Path("data/panels_v3"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--years", nargs="+", type=int, default=[2023, 2024, 2025, 2026])
    args = ap.parse_args()

    print(f"[26FG-v3 panel] tier={args.tier} panels_root={args.panels_root} years={args.years}")

    # 1. alpha + gtja (with patched factor library)
    alpha = _read_yearly(args.panels_root, "alpha_panel_year={year}.parquet", args.years)
    gtja  = _read_yearly(args.panels_root, "gtja_panel_year={year}.parquet", args.years)
    if "stock_code" in alpha.columns:
        alpha = alpha.rename({"stock_code": "ts_code"})
    if "stock_code" in gtja.columns:
        gtja = gtja.rename({"stock_code": "ts_code"})
    print(f"  alpha: {len(alpha):,}  gtja: {len(gtja):,}")
    panel = alpha.join(gtja, on=["ts_code", "trade_date"], how="inner")

    # 2. mf + other enriched families
    mf = _read_yearly(args.panels_root, "mf_panel_year={year}.parquet", args.years)
    if "stock_code" in mf.columns:
        mf = mf.rename({"stock_code": "ts_code"})
    panel = panel.join(mf, on=["ts_code", "trade_date"], how="left")

    # 2b. enriched_panel (40 cols: fund_/hk_/hm_/ind_/inst_/mfp_/mg_/senti_/sh_)
    # Added 2026-05-08 by paris when v3 alpha/gtja/mf alone didn't cover the
    # full 23A factor set.
    enriched_path = args.panels_root / "enriched_panel_year=2023.parquet"
    if enriched_path.exists():
        enriched = _read_yearly(args.panels_root, "enriched_panel_year={year}.parquet", args.years)
        if "stock_code" in enriched.columns:
            enriched = enriched.rename({"stock_code": "ts_code"})
        panel = panel.join(enriched, on=["ts_code", "trade_date"], how="left")
        print(f"  + enriched_panel: {len(enriched.columns) - 2} cols")

    # 2c. OHLC / meta cols (close, pct_chg, vol, name, is_st, days_since_ipo).
    # v3 panels carry only factor data, so join from combined_short single-file.
    # data_loader needs these for return computation, suspended detection, ST gate.
    proj_root = Path(__file__).resolve().parent.parent.parent
    qe_path = proj_root / "data" / "factor_panel_combined_short_2023_2026.parquet"
    if not qe_path.exists():
        raise SystemExit(f"missing combined_short for OHLC: {qe_path}")
    META_COLS = ["close", "pct_chg", "vol", "name", "is_st", "industry_code", "days_since_ipo"]
    qe = pl.read_parquet(qe_path)
    keep_meta = [c for c in META_COLS if c in qe.columns]
    panel = panel.join(
        qe.select(["ts_code", "trade_date", *keep_meta]),
        on=["ts_code", "trade_date"], how="left",
    )
    print(f"  + OHLC/meta from combined_short: {len(keep_meta)} cols")

    # 3. v1.2 cyq override (replaces any legacy cyq in alpha/gtja namespace)
    cyq = _read_yearly(args.panels_root, "cyq_panel/year={year}.parquet", args.years)
    drop_legacy = [c for c in panel.columns if c.startswith("cyq_") and c in cyq.columns]
    if drop_legacy:
        panel = panel.drop(drop_legacy)
    panel = panel.join(cyq, on=["ts_code", "trade_date"], how="left")

    # 4. tier-specific tech extras (need tech_panel_v1)
    extras = TIER_EXTRA_TECH_COLS[args.tier]
    if extras:
        tech = _read_yearly(args.panels_root, "tech_panel_v1/tech_panel_year={year}.parquet", args.years)
        if "stock_code" in tech.columns:
            tech = tech.rename({"stock_code": "ts_code"})
        cols_present = [c for c in extras if c in tech.columns]
        if len(cols_present) != len(extras):
            raise SystemExit(f"tech_panel_v1 missing {set(extras)-set(cols_present)}")
        panel = panel.join(
            tech.select(["ts_code", "trade_date", *cols_present]),
            on=["ts_code", "trade_date"], how="left",
        )

    # 5. event-decay (26F/G)
    if TIER_INCLUDE_EVENTS[args.tier]:
        ev = _read_yearly(args.panels_root, "tech_event_panel/year={year}.parquet", args.years)
        keep_ev = [c for c in EVENT_DECAY_COLS if c in ev.columns]
        if len(keep_ev) != len(EVENT_DECAY_COLS):
            raise SystemExit(f"tech_event_panel missing {set(EVENT_DECAY_COLS)-set(keep_ev)}")
        panel = panel.join(
            ev.select(["ts_code", "trade_date", *keep_ev]),
            on=["ts_code", "trade_date"], how="left",
        )

    panel = panel.sort(["trade_date", "ts_code"])
    n_dup = (
        panel.group_by(["ts_code", "trade_date"]).len().filter(pl.col("len") > 1).height
    )
    if n_dup:
        sys.exit(f"FAIL: {n_dup} duplicate keys")

    # Universe assertion (codex round-2 finding 5): build script must
    # verify the output universe BEFORE writing the parquet, mirroring
    # the project-wide hard constraint declared in CLAUDE.md and enforced
    # by tools/verify_panels.py. Inline regex (anchored) to keep this
    # script self-contained even when the bundle is shipped to the RL host.
    import re
    sh_main = re.compile(r"^60[0135]\d{3}\.SH$")
    sz_main = re.compile(r"^00[0123]\d{3}\.SZ$")
    codes = panel["ts_code"].unique().to_list()
    bad = [c for c in codes if c is not None and not (sh_main.match(c) or sz_main.match(c))]
    if bad:
        sys.exit(
            f"FAIL: {len(bad)} non-main-board ts_codes leaked into output: "
            f"{bad[:10]} (universe constraint requires 60[0135]*.SH | 00[0123]*.SZ)"
        )
    print(f"[26FG-v3 panel] universe OK: {len(codes)} stocks, all main-board")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    panel.write_parquet(args.out, compression="zstd", compression_level=10)
    sz_gb = args.out.stat().st_size / 1024 / 1024 / 1024
    print(f"[26FG-v3 panel] wrote {args.out}: {len(panel):,} rows × {len(panel.columns)} cols, {sz_gb:.2f} GB")


if __name__ == "__main__":
    main()
