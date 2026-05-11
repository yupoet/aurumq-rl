"""Hybrid ensemble: equal-weight average Path 1 long + Path 4 short ensembles."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from p3.path1_eval import H1, H2, evaluate


def main():
    bundle = Path("data/p3_4070")
    p1l = pl.read_parquet("runs/sl_path1_long/predictions.parquet")
    p4s = pl.read_parquet("runs/sl_path4/predictions.parquet")

    p1l_col = "score_calibrated" if "score_calibrated" in p1l.columns else "score_raw"
    p4s_col = "score_calibrated" if "score_calibrated" in p4s.columns else "score_raw"

    p1l = p1l.select(["trade_date", "ts_code", pl.col(p1l_col).alias("p1l_score")])
    p4s = p4s.select(["trade_date", "ts_code", pl.col(p4s_col).alias("p4s_score")])

    hybrid = p1l.join(p4s, on=["trade_date", "ts_code"], how="inner")
    print(f"hybrid rows: {len(hybrid):,}")

    # Equal-weight average
    hybrid = hybrid.with_columns(
        ((pl.col("p1l_score") + pl.col("p4s_score")) / 2.0).alias("score")
    )

    target_y = pl.read_parquet(bundle / "target_y.parquet")
    realized = pl.read_parquet(bundle / "realized_returns.parquet").select(
        ["trade_date", "ts_code", "pct_chg_t_plus_1"]
    )
    market = pl.read_parquet(bundle / "market_returns.parquet").select(
        ["trade_date", "eq_weight_pct_chg_t_plus_1"]
    )

    h1_eval = evaluate(hybrid.select(["trade_date","ts_code","score"]), target_y, realized, market, H1)
    h2_eval = evaluate(hybrid.select(["trade_date","ts_code","score"]), target_y, realized, market, H2)

    summary = {
        "method": "Hybrid equal-weight: (path1_long + path4_short) / 2",
        "H1_primary": h1_eval["primary_mean_top50_proximity_excess"],
        "H2_primary": h2_eval["primary_mean_top50_proximity_excess"],
        "H1": h1_eval,
        "H2": h2_eval,
    }
    out = Path("runs/sl_hybrid_p1long_p4short")
    out.mkdir(parents=True, exist_ok=True)
    (out / "ensemble.json").write_text(json.dumps(summary, indent=2, default=str))
    hybrid.write_parquet(out / "predictions.parquet", compression="zstd", compression_level=10)
    print(f"H1 primary: {h1_eval['primary_mean_top50_proximity_excess']:+.6f}")
    print(f"H2 primary: {h2_eval['primary_mean_top50_proximity_excess']:+.6f}")
    print(f"wrote {out}/ensemble.json + predictions.parquet")


if __name__ == "__main__":
    main()
