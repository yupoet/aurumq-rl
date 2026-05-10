"""Reusable cross-sectional rank-z transform for SL Path 4 inference.

Extracted from scripts/p3/path4_features.py so the production inference
pipeline (paris-side on Ubuntu) can apply identical preprocessing without
copying the function.

Per-day rank within in-universe stocks → (rank - 1) / (N - 1) * 2 - 1 ∈ [-1, 1].
Out-of-universe rows: 0.0 (neutral). NaN raw values: 0.0.

Critical invariant for inference correctness:
  - Training and inference MUST use the same rank ordering (method="ordinal")
  - Rank is computed PER trade_date over in-universe stocks ONLY
  - Inference for "today" uses today's universe (cannot leak future data)
"""
from __future__ import annotations

import polars as pl


def cross_sectional_rank_z(
    panel: pl.DataFrame,
    universe: pl.DataFrame,
    feature_cols: list[str],
) -> pl.DataFrame:
    """Per-day rank-z transform on each feature column over in-universe stocks.

    Args
    ----
    panel : pl.DataFrame
        Schema: trade_date, ts_code, *features. May include out-of-universe rows.
    universe : pl.DataFrame
        Schema: trade_date, ts_code, in_universe (bool).
    feature_cols : list[str]
        Columns to transform. Other columns pass through unchanged.

    Returns
    -------
    pl.DataFrame
        Same schema as ``panel``, but ``feature_cols`` columns replaced
        with rank-z transformed values ∈ [-1, 1]. NaN raw / out-of-universe
        rows get 0.0 (neutral).
    """
    df = panel.join(universe, on=["trade_date", "ts_code"], how="left")

    rank_exprs = []
    for col in feature_cols:
        masked = pl.when(pl.col("in_universe") == True).then(pl.col(col)).otherwise(None)  # noqa: E712
        rank = masked.rank(method="ordinal").over("trade_date")
        n_in_uni = (pl.col("in_universe") == True).cast(pl.Int64).sum().over("trade_date")  # noqa: E712
        denom = pl.when(n_in_uni > 1).then(n_in_uni - 1).otherwise(1)
        rank_z = (rank.cast(pl.Float32) - 1.0) / denom.cast(pl.Float32) * 2.0 - 1.0
        rank_z = rank_z.fill_null(0.0).cast(pl.Float32)
        rank_exprs.append(rank_z.alias(col))

    out = df.with_columns(rank_exprs).drop("in_universe")
    return out


__all__ = ["cross_sectional_rank_z"]
