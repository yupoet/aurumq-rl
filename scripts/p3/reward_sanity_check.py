"""P3 pre-training reward sanity check (强制).

Per ALGORITHM_SPEC v2 §7. Run once before any training:
    python scripts/p3/reward_sanity_check.py --bundle ./data_p3
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import duckdb
import numpy as np


logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default="./data_p3", help="bundle dir from OSS")
    args = parser.parse_args(argv)

    bundle = Path(args.bundle)
    realized = bundle / "realized_returns.parquet"
    market = bundle / "market_returns.parquet"
    if not realized.exists() or not market.exists():
        logger.error("Missing files in bundle: %s / %s", realized, market)
        return 2

    con = duckdb.connect()
    stats = con.execute(f"""
        SELECT
            AVG(pct_chg_t_plus_1)::FLOAT AS mean_r,
            STDDEV(pct_chg_t_plus_1)::FLOAT AS std_r,
            MIN(pct_chg_t_plus_1)::FLOAT AS min_r,
            MAX(pct_chg_t_plus_1)::FLOAT AS max_r,
            COUNT(*) FILTER (WHERE pct_chg_t_plus_1 = 0)::FLOAT / COUNT(*) AS zero_frac,
            COUNT(*) AS n_rows
        FROM '{realized}'
    """).fetchone()

    market_stats = con.execute(f"""
        SELECT AVG(eq_weight_pct_chg_t_plus_1)::FLOAT, STDDEV(eq_weight_pct_chg_t_plus_1)::FLOAT,
               MIN(n_stocks), MAX(n_stocks), COUNT(*)
        FROM '{market}'
    """).fetchone()

    logger.info("==== Realized returns ====")
    logger.info("  rows: %d", stats[5])
    logger.info("  mean: %.5f  (target: |mean| < 0.001)", stats[0])
    logger.info("  std:  %.5f  (target: 0.020-0.040)", stats[1])
    logger.info("  range: [%.4f, %.4f]  (target: |x| < 0.15)", stats[2], stats[3])
    logger.info("  zero fraction: %.4f  (suspended/no-trade)", stats[4])
    logger.info("==== Market eq-weight ====")
    logger.info("  dates: %d  mean: %.5f  std: %.5f  n_stocks: [%d, %d]",
                market_stats[4], market_stats[0], market_stats[1], market_stats[2], market_stats[3])

    failures = []
    if abs(stats[0]) >= 0.001:
        failures.append(f"mean(r)={stats[0]:.5f} not near zero")
    if not (0.020 <= stats[1] <= 0.040):
        failures.append(f"std(r)={stats[1]:.5f} outside [0.020, 0.040]")
    if not (-0.15 < stats[2] and stats[3] < 0.15):
        failures.append(f"range [{stats[2]:.4f}, {stats[3]:.4f}] has outliers > 15%")
    if stats[4] > 0.10:
        failures.append(f"zero_frac={stats[4]:.4f} > 10% (too many suspended)")

    if failures:
        logger.error("==== SANITY FAILED ====")
        for f in failures:
            logger.error("  - %s", f)
        logger.error("Do NOT start training. Fix data first.")
        return 1

    logger.info("==== SANITY PASSED ====")
    logger.info("OK to start training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
