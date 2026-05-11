"""Path 4 — feature engineering: cross-sectional rank-z + outlier clip.

Per spec §4.2:
  1. For each (trade_date, feature column), replace raw value with
     (rank_within_in_universe / N - 0.5) * 2 ∈ [-1, +1].
     Eliminates per-day distribution shift and outlier dominance.
     The single highest-value feature engineering on Chinese alpha factors.
  2. Outlier audit + clip — built-in: rank transform IS the clip
     (any extreme value just becomes the highest rank).

Out-of-universe stocks are EXCLUDED from the rank computation but kept in
the panel with NaN feature values. The downstream training script's
universe filter drops them anyway.

NaN raw values get rank = 0.5 (median-equivalent) so downstream LightGBM's
NaN-handling treats them consistently.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import polars as pl


logger = logging.getLogger(__name__)


def cross_sectional_rank_z(
    panel: pl.DataFrame,
    universe: pl.DataFrame,
    feature_cols: list[str],
    chunk_size: int = 30,
) -> pl.DataFrame:
    """Per-day rank-z transform on each feature column over in-universe stocks.

    Processes ``feature_cols`` in chunks (default 30 columns per pass) so peak
    memory stays bounded for large panels (8M+ rows × 345 features OOMs polars
    if all .with_columns() exprs evaluate at once). Output is identical
    regardless of chunk_size.

    Args
    ----
    panel : pl.DataFrame
        (trade_date, ts_code, *features). May include out-of-universe rows.
    universe : pl.DataFrame
        (trade_date, ts_code, in_universe).
    feature_cols : list[str]
        Columns to transform.
    chunk_size : int
        Number of feature columns to rank per polars pass. Smaller = less peak
        memory, slightly more wall time. 30 fits 8M rows in <8 GB.

    Returns
    -------
    pl.DataFrame
        Same schema as ``panel``, but each ``feature_cols`` column replaced
        with (rank_within_day_in_universe / N_in_universe - 0.5) * 2 ∈ [-1, 1].
        Out-of-universe rows have feature value 0.0 (neutral). NaN raw values
        also get 0.0.
    """
    df = panel.join(universe, on=["trade_date", "ts_code"], how="left")

    n_in_uni = (pl.col("in_universe") == True).cast(pl.Int64).sum().over("trade_date")  # noqa: E712
    denom = pl.when(n_in_uni > 1).then(n_in_uni - 1).otherwise(1)

    # Process in chunks to bound peak memory
    n = len(feature_cols)
    for i in range(0, n, chunk_size):
        batch = feature_cols[i : i + chunk_size]
        rank_exprs = []
        for col in batch:
            masked = pl.when(pl.col("in_universe") == True).then(pl.col(col)).otherwise(None)  # noqa: E712
            rank = masked.rank(method="ordinal").over("trade_date")
            rank_z = (rank.cast(pl.Float32) - 1.0) / denom.cast(pl.Float32) * 2.0 - 1.0
            rank_z = rank_z.fill_null(0.0).cast(pl.Float32)
            rank_exprs.append(rank_z.alias(col))
        df = df.with_columns(rank_exprs)
        logger.info("  rank-z chunk %d/%d: cols [%s..%s]", i // chunk_size + 1,
                    (n + chunk_size - 1) // chunk_size, batch[0], batch[-1])

    return df.drop("in_universe")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="data/p3_4070", type=Path)
    ap.add_argument("--feature-panel-in", default="feature_panel_v3_344.parquet")
    ap.add_argument("--feature-panel-out", default="feature_panel_clean.parquet")
    args = ap.parse_args(argv)

    in_path = args.bundle / args.feature_panel_in
    out_path = args.bundle / args.feature_panel_out

    t0 = time.time()
    logger.info("loading raw panel: %s", in_path)
    panel = pl.read_parquet(in_path)
    feature_cols = [c for c in panel.columns if c not in ("ts_code", "trade_date")]
    logger.info("panel: %d rows × %d features (%.1fs)", len(panel), len(feature_cols), time.time() - t0)

    uni_parts = []
    # Glob all year shards — supports both short (2023-2026) and long (2018+) bundles.
    for p in sorted((args.bundle / "universe_mask").glob("year=*.parquet")):
        uni_parts.append(pl.read_parquet(p).select(["trade_date", "ts_code", "in_universe"]))
    if not uni_parts:
        raise SystemExit(f"no universe_mask shards in {args.bundle}/universe_mask/")
    universe = pl.concat(uni_parts)
    logger.info("universe: %d rows", len(universe))

    t1 = time.time()
    logger.info("computing rank-z (this is the heavy step)...")
    out = cross_sectional_rank_z(panel, universe, feature_cols)
    logger.info("rank-z done in %.1fs (rows=%d)", time.time() - t1, len(out))

    # Sanity stats on a couple features
    for c in feature_cols[:3]:
        s = out[c]
        logger.info("  %s: min=%.4f mean=%.4f max=%.4f", c, s.min(), s.mean(), s.max())

    out.write_parquet(out_path, compression="zstd", compression_level=9)
    logger.info("wrote %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)
    return 0


if __name__ == "__main__":
    sys.exit(main())
