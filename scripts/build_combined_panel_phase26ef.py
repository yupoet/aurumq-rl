"""Build combined panel for Phase 26E/F/G training.

Joins:
  - 23A's exact 353-col base (alpha + gtja + mf + mfp + hm + hk + inst +
    mg + senti + sh + fund + ind + cyq legacy from quotes_enriched)
  - v1.2 cyq_panel override (3 cols: cyq_winning_ratio,
    cyq_concentration_70, cyq_cost_distance) — replaces the legacy
    quotes_enriched.cyq_*
  - **Tier-specific extras:**
      26C2: nothing extra (353 cols total) — reproduces production baseline
      26E:  + tech_boll_percent + cmf_120d_pct_amt (355 cols)
      26F:  26E + 6 tech_evt_*_decay10 columns (361 cols)
      26G:  same as 26F (encoder differs, panel identical)

Pre-requisites
--------------
panels_v2/ must contain:
    alpha_panel_year={2023,2024,2025,2026}.parquet
    gtja_panel_year={2023,2024,2025,2026}.parquet
    cyq_panel/year={2023,2024,2025,2026}.parquet
    tech_panel_v1/                          (for tech_boll_percent, cmf_120d_pct_amt)
    tech_event_panel/year={2023,2024,2025,2026}.parquet
    quotes_enriched/year={2023,2024,2025,2026}.parquet

(matches the OSS bundle layout.)

Usage::

    python scripts/build_combined_panel_phase26ef.py --tier 26F \\
        --panels-root data/panels_v2 \\
        --out data/panel_26f.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl


TIER_INCLUDE_FILE = {
    "26C2": "configs/include_columns_phase26c2_353.txt",
    "26E":  "configs/include_columns_phase26e_355.txt",
    "26F":  "configs/include_columns_phase26f_361.txt",
    "26G":  "configs/include_columns_phase26f_361.txt",
}

TIER_EXTRA_TECH_COLS = {
    "26C2": [],
    "26E": ["tech_boll_percent", "cmf_120d_pct_amt"],
    "26F": ["tech_boll_percent", "cmf_120d_pct_amt"],
    "26G": ["tech_boll_percent", "cmf_120d_pct_amt"],
}

TIER_INCLUDE_EVENTS = {
    "26C2": False,
    "26E":  False,
    "26F":  True,
    "26G":  True,
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
    ap.add_argument("--panels-root", type=Path, default=Path("data/panels_v2"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--years", nargs="+", type=int, default=[2023, 2024, 2025, 2026])
    args = ap.parse_args()

    print(f"[26EF panel] tier={args.tier} panels_root={args.panels_root} years={args.years}")

    # 1. Base: alpha + gtja
    alpha = _read_yearly(args.panels_root, "alpha_panel_year={year}.parquet", args.years)
    gtja  = _read_yearly(args.panels_root, "gtja_panel_year={year}.parquet", args.years)

    # Standardize ts_code column name across panels
    if "stock_code" in alpha.columns:
        alpha = alpha.rename({"stock_code": "ts_code"})
    if "stock_code" in gtja.columns:
        gtja = gtja.rename({"stock_code": "ts_code"})

    print(f"  alpha: {len(alpha):,} rows × {len(alpha.columns)} cols")
    print(f"  gtja:  {len(gtja):,} rows × {len(gtja.columns)} cols")

    panel = alpha.join(gtja, on=["ts_code", "trade_date"], how="inner")
    print(f"  alpha ⨝ gtja: {len(panel):,} rows × {len(panel.columns)} cols")

    # 2. Other 23A factor families from local combined_short single-file
    # (mf, mfp, hm, hk, inst, mg, senti, sh, fund, ind, plus close/pct_chg/vol/name/is_st/days_since_ipo).
    # Paris's reference assumed yearly quotes_enriched/year=*.parquet; on the
    # AurumQ-RL side these all live in factor_panel_combined_short_2023_2026.parquet.
    # Project-root-relative path (not panels_root.parent) so it works
    # regardless of how nested panels_root is.
    project_root = Path(__file__).resolve().parent.parent
    qe_path = project_root / "data" / "factor_panel_combined_short_2023_2026.parquet"
    if not qe_path.exists():
        raise SystemExit(f"missing combined_short panel: {qe_path}")
    qe = pl.read_parquet(qe_path)
    META_COLS = ["close", "pct_chg", "vol", "name", "is_st", "industry_code", "days_since_ipo"]
    keep_qe = [c for c in qe.columns
        if c.startswith(("mf_", "mfp_", "hm_", "hk_", "inst_", "mg_", "senti_", "sh_", "fund_", "ind_"))]
    keep_meta = [c for c in META_COLS if c in qe.columns]
    panel = panel.join(
        qe.select(["ts_code", "trade_date", *keep_qe, *keep_meta]),
        on=["ts_code", "trade_date"], how="left",
    )
    print(f"  + combined_short extras ({len(keep_qe)} factor + {len(keep_meta)} meta): {len(panel):,} × {len(panel.columns)}")

    # 3. v1.2 cyq override
    cyq = _read_yearly(args.panels_root, "cyq_panel/year={year}.parquet", args.years)
    # Drop legacy cyq_* if any from quotes_enriched, then join canonical
    drop_legacy = [c for c in panel.columns if c.startswith("cyq_") and c in cyq.columns]
    if drop_legacy:
        panel = panel.drop(drop_legacy)
    panel = panel.join(cyq, on=["ts_code", "trade_date"], how="left")
    print(f"  + v1.2 cyq override (3 cols): {len(panel):,} × {len(panel.columns)}")

    # 4. Tier-specific tech extras
    extras = TIER_EXTRA_TECH_COLS[args.tier]
    if extras:
        # AurumQ-RL local layout: data/tech_panel_v1/tech_panel/year=*.parquet
        tech_root = project_root / "data" / "tech_panel_v1" / "tech_panel"
        if not tech_root.exists():
            raise SystemExit(f"missing tech_panel_v1: {tech_root}")
        tech_parts = [pl.read_parquet(tech_root / f"year={y}.parquet") for y in args.years
                      if (tech_root / f"year={y}.parquet").exists()]
        tech = pl.concat(tech_parts, how="diagonal_relaxed")
        if "stock_code" in tech.columns:
            tech = tech.rename({"stock_code": "ts_code"})
        cols_present = [c for c in extras if c in tech.columns]
        if len(cols_present) != len(extras):
            missing = set(extras) - set(cols_present)
            raise SystemExit(f"tech_panel_v1 missing tier-{args.tier} cols: {missing}")
        panel = panel.join(
            tech.select(["ts_code", "trade_date", *cols_present]),
            on=["ts_code", "trade_date"], how="left",
        )
        print(f"  + tier-{args.tier} curated tech ({len(cols_present)}): {len(panel):,} × {len(panel.columns)}")

    # 5. Event-decay (only 26F/G)
    if TIER_INCLUDE_EVENTS[args.tier]:
        ev = _read_yearly(args.panels_root, "tech_event_panel/year={year}.parquet", args.years)
        keep_ev = [c for c in EVENT_DECAY_COLS if c in ev.columns]
        if len(keep_ev) != len(EVENT_DECAY_COLS):
            missing = set(EVENT_DECAY_COLS) - set(keep_ev)
            raise SystemExit(f"tech_event_panel missing decay cols: {missing}")
        panel = panel.join(
            ev.select(["ts_code", "trade_date", *keep_ev]),
            on=["ts_code", "trade_date"], how="left",
        )
        print(f"  + tier-{args.tier} event decay ({len(keep_ev)}): {len(panel):,} × {len(panel.columns)}")

    # 6. Sort + dedup safety + write
    panel = panel.sort(["trade_date", "ts_code"])
    n_dup = (
        panel.group_by(["ts_code", "trade_date"])
        .len()
        .filter(pl.col("len") > 1)
        .height
    )
    if n_dup:
        sys.exit(f"FAIL: {n_dup} duplicate keys")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    panel.write_parquet(args.out, compression="zstd", compression_level=10)
    sz_gb = args.out.stat().st_size / 1024 / 1024 / 1024
    print(f"[26EF panel] wrote {args.out}: {len(panel):,} rows × {len(panel.columns)} cols, {sz_gb:.2f} GB")


if __name__ == "__main__":
    main()
