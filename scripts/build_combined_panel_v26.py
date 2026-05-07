#!/usr/bin/env python3
"""Phase 26A — combined panel build.

Inputs:
  - data/factor_panel_combined_short_2023_2026.parquet      (existing 23A base)
  - data/tech_panel_v1/tech_panel/year=*.parquet            (30 new tech_/cmf_/zt_ cols)
  - data/tech_panel_v1/cyq_panel/year=*.parquet             (3 canonical cyq_ cols)
  - data/tech_panel_v1/tech_panel_report/include_columns_v1.txt

Output:
  - data/factor_panel_phase26a_2023_2026.parquet            (373 factor cols + meta)

The cyq_panel is the canonical source for cyq_winning_ratio /
cyq_concentration_70 / cyq_cost_distance — the existing combined_short
versions are dropped before the join. mfp_/senti_/mkt_ deprecated cols
are dropped by allowlisting against include_columns_v1.txt.

Joins are inner on (ts_code, trade_date) — the tech_panel covers
2023-01-03..2026-05-06; the combined_short covers 2023-01-03..2026-04-24,
so the inner join effectively yields combined_short's date range.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


_ROOT = Path(__file__).resolve().parent.parent

# Required + optional metadata cols data_loader needs
META_COLS = ["ts_code", "trade_date", "close", "pct_chg", "vol",
             "name", "is_st", "industry_code", "days_since_ipo"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path,
                   default=_ROOT / "data/factor_panel_combined_short_2023_2026.parquet")
    p.add_argument("--tech-glob", type=str,
                   default=str(_ROOT / "data/tech_panel_v1/tech_panel/year=*.parquet"))
    p.add_argument("--cyq-glob", type=str,
                   default=str(_ROOT / "data/tech_panel_v1/cyq_panel/year=*.parquet"))
    p.add_argument("--include-list", type=Path,
                   default=_ROOT / "data/tech_panel_v1/tech_panel_report/include_columns_v1.txt")
    p.add_argument("--out", type=Path,
                   default=_ROOT / "data/factor_panel_phase26a_2023_2026.parquet")
    p.add_argument("--start-date", default="2023-01-03",
                   help="Inclusive lower bound (must be >= 2023-01-03).")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    include_cols = [l.strip() for l in args.include_list.read_text().splitlines() if l.strip()]
    print(f"[build] include list: {len(include_cols)} columns")

    # Load base, drop the 3 cyq_ that will come canonically from cyq_panel,
    # then keep only meta + cols that survive the include list.
    base_lf = pl.scan_parquet(str(args.base))
    base_schema = base_lf.collect_schema()
    base_cols_present = list(base_schema.names())

    base_cyq_to_drop = [c for c in ("cyq_winning_ratio", "cyq_concentration_70", "cyq_cost_distance")
                        if c in base_cols_present]
    print(f"[build] base parquet: {len(base_cols_present)} cols; dropping non-canonical cyq: {base_cyq_to_drop}")

    keep_meta = [c for c in META_COLS if c in base_cols_present]
    keep_factors = [c for c in include_cols if c in base_cols_present and c not in base_cyq_to_drop]
    print(f"[build] base contributes {len(keep_factors)} factor cols + {len(keep_meta)} meta")

    base_lf = (
        base_lf
        .filter(pl.col("trade_date") >= pl.lit(args.start_date).str.to_date())
        .select(keep_meta + keep_factors)
    )

    # Tech panel
    tech_lf = pl.scan_parquet(args.tech_glob)
    tech_schema = list(tech_lf.collect_schema().names())
    tech_factor_cols = [c for c in include_cols if c in tech_schema]
    print(f"[build] tech panel contributes {len(tech_factor_cols)} cols")
    tech_lf = (
        tech_lf
        .filter(pl.col("trade_date") >= pl.lit(args.start_date).str.to_date())
        .select(["ts_code", "trade_date"] + tech_factor_cols)
    )

    # Cyq panel (canonical)
    cyq_lf = pl.scan_parquet(args.cyq_glob)
    cyq_schema = list(cyq_lf.collect_schema().names())
    cyq_factor_cols = [c for c in include_cols if c in cyq_schema]
    print(f"[build] cyq panel contributes {len(cyq_factor_cols)} cols")
    cyq_lf = (
        cyq_lf
        .filter(pl.col("trade_date") >= pl.lit(args.start_date).str.to_date())
        .select(["ts_code", "trade_date"] + cyq_factor_cols)
    )

    joined_lf = (
        base_lf
        .join(tech_lf, on=["ts_code", "trade_date"], how="inner")
        .join(cyq_lf, on=["ts_code", "trade_date"], how="left")
    )

    print(f"[build] collecting...")
    df = joined_lf.collect()
    print(f"[build] joined: {len(df):,} rows × {len(df.columns)} cols")

    factor_cols_in_df = [c for c in include_cols if c in df.columns]
    missing = [c for c in include_cols if c not in df.columns]
    print(f"[build] include-list cols in panel: {len(factor_cols_in_df)} / {len(include_cols)}")
    if missing:
        print(f"[build] MISSING: {missing}")
        return 1

    final_cols = keep_meta + include_cols
    df = df.select(final_cols)

    dr = df.select(pl.col("trade_date").min().alias("mn"),
                   pl.col("trade_date").max().alias("mx"),
                   pl.col("ts_code").n_unique().alias("nstk")).row(0)
    print(f"[build] dates: {dr[0]} .. {dr[1]}  stocks={dr[2]}")

    from collections import Counter
    prefix_counts = Counter(c.split("_")[0] + "_" for c in include_cols)
    print(f"[build] per-prefix counts: {dict(prefix_counts)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(args.out, compression="zstd")
    size_mb = args.out.stat().st_size / 1e6
    print(f"[build] wrote {args.out}  ({size_mb:.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
