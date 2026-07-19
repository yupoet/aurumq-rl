"""Blend MASTER-lite scores with the LightGBM base and apply the kill criteria.

Second half of the 2026-07 MASTER experiment (README §12.7). Loads two prediction
parquets — the LGBM base (e.g. path5_long) and the MASTER-lite run — evaluates
base / master / rank-percentile blends on the standard windows, then renders the
pre-registered KEEP/KILL verdict via ``master_lib.kill_criteria_verdict``:

    KEEP  iff the best blend beats the base on Spearman IC without losing on
          top-50 proximity excess in >= 2/3 of the eval windows.

The blend never replaces the base. A KEEP means "MASTER earns a weight in the
production rank blend"; a KILL means the experiment stops (Finding-style
discipline: negative results get recorded, not retried with shifted goalposts).

Cost discipline (§12.7d): every metric block is merged with
``master_lib.cost_metric_block`` (annualized two-sided turnover + top-k T+1
excess net of COST_SPEC_V1), and the verdict requires the blend to hold the
net-of-cost excess in addition to IC / proximity — an IC win that loses net of
trading costs is not a win.

Pre-registered protocol (2026-07-17, before any 4070 run):
  * eval windows: H1 + H2 + W3=2026-01-05:2026-06-30 (pass W3 via
    ``--extra-window W3=2026-01-05:2026-06-30``; 2-window verdicts degenerate
    to 2/2 and are logged with a warning)
  * seeds: 42, 43, 44 — pass all three prediction parquets via repeated
    ``--master-preds``; they are rank-averaged into ONE master score column
    before blending (seed-ensemble, Phase 18 discipline)

Usage::

    python scripts/p3/master_ensemble_eval.py \
        --master-preds runs/master_lite/d64_L8_seed42/predictions.parquet \
        --master-preds runs/master_lite/d64_L8_seed43/predictions.parquet \
        --master-preds runs/master_lite/d64_L8_seed44/predictions.parquet \
        --base-preds runs/sl_path5_long/best/predictions.parquet \
        --bundle data/p3_4070_long \
        --extra-window W3=2026-01-05:2026-06-30 \
        --out runs/master_lite/d64_L8/ensemble
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent))

from p3.master_lib import (
    blend_rank_scores,
    cost_metric_block,
    kill_criteria_verdict,
)
from p3.path1_eval import H1, H2, evaluate

logger = logging.getLogger(__name__)

DEFAULT_BLEND_WEIGHTS = (0.25, 0.50)  # master weight in the (base, master) blend


def parse_windows(specs: list[str]) -> dict[str, tuple[dt.date, dt.date]]:
    """Parse repeated ``name=YYYY-MM-DD:YYYY-MM-DD`` specs into eval windows."""
    windows: dict[str, tuple[dt.date, dt.date]] = {}
    for spec in specs:
        name, _, rng = spec.partition("=")
        lo, _, hi = rng.partition(":")
        if not (name and lo and hi):
            raise SystemExit(f"bad --extra-window {spec!r}, want name=start:end")
        windows[name] = (dt.date.fromisoformat(lo), dt.date.fromisoformat(hi))
    return windows


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
    )
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--master-preds",
        required=True,
        type=Path,
        action="append",
        help="MASTER predictions parquet; repeat for multi-seed runs (rank-averaged)",
    )
    ap.add_argument("--base-preds", required=True, type=Path)
    ap.add_argument("--bundle", default="data/p3_4070", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--blend-weights",
        default=",".join(str(w) for w in DEFAULT_BLEND_WEIGHTS),
        help="comma-separated MASTER weights to try in the (base, master) blend",
    )
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument(
        "--extra-window",
        action="append",
        default=[],
        help="additional eval window as name=YYYY-MM-DD:YYYY-MM-DD (repeatable)",
    )
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    base = pl.read_parquet(args.base_preds).select(["trade_date", "ts_code", "score"])
    seeds = [
        pl.read_parquet(p).select(["trade_date", "ts_code", "score"]) for p in args.master_preds
    ]
    # Multi-seed runs are rank-averaged into ONE master column before blending
    # (equal weights). A single frame passes through unchanged.
    master = seeds[0] if len(seeds) == 1 else blend_rank_scores(seeds, [1.0] * len(seeds))
    logger.info(
        "base: %s rows | master: %s rows (%d seed%s)",
        f"{len(base):,}",
        f"{len(master):,}",
        len(seeds),
        "" if len(seeds) == 1 else "s",
    )

    target_y = pl.read_parquet(args.bundle / "target_y.parquet")
    realized = pl.read_parquet(args.bundle / "realized_returns.parquet").select(
        ["trade_date", "ts_code", "pct_chg_t_plus_1"]
    )
    market = pl.read_parquet(args.bundle / "market_returns.parquet").select(
        ["trade_date", "eq_weight_pct_chg_t_plus_1"]
    )

    windows = {"H1": H1, "H2": H2, **parse_windows(args.extra_window)}
    if len(windows) < 3:
        logger.warning(
            "only %d eval windows: ceil(2/3 * %d) = %d, so the verdict degenerates to "
            "all-windows-must-win; pre-registered protocol adds W3=2026-01-05:2026-06-30 "
            "via --extra-window",
            len(windows),
            len(windows),
            -(-2 * len(windows) // 3),
        )

    def eval_all(preds: pl.DataFrame) -> dict[str, dict]:
        # §12.7d: the standard metric block is merged with the turnover/cost
        # block so the kill criteria can veto on net-of-cost excess.
        return {
            name: {
                **evaluate(preds, target_y, realized, market, w, args.top_k),
                **cost_metric_block(preds, realized, market, w, args.top_k),
            }
            for name, w in windows.items()
        }

    results: dict = {"windows": {k: [str(v[0]), str(v[1])] for k, v in windows.items()}}
    results["base"] = eval_all(base)
    results["master"] = eval_all(master)

    blend_metrics: dict[str, dict[str, dict]] = {}
    for w_master in (float(w) for w in args.blend_weights.split(",")):
        key = f"blend_m{w_master:g}"
        blended = blend_rank_scores([base, master], [1.0 - w_master, w_master])
        blend_metrics[key] = eval_all(blended)
    results["blends"] = blend_metrics

    # Best blend by mean Spearman across windows — chosen among the small
    # pre-declared weight grid, not tuned per window.
    def mean_ic(per_window: dict[str, dict]) -> float:
        return sum(m["spearman"] for m in per_window.values()) / len(per_window)

    best_key = max(blend_metrics, key=lambda k: mean_ic(blend_metrics[k]))
    results["best_blend"] = best_key
    results["n_seeds"] = len(seeds)
    # Self-documenting protocol flag: a verdict produced off-protocol (fewer
    # than 3 windows or 3 seeds) is dev-grade and must not enter README §12.2.
    results["protocol_ok"] = len(windows) >= 3 and len(seeds) >= 3
    if not results["protocol_ok"]:
        logger.warning(
            "OFF-PROTOCOL RUN (windows=%d, seeds=%d; protocol needs >=3 of each): "
            "verdict is dev-grade, do not record it as the pre-registered result",
            len(windows),
            len(seeds),
        )
    results["kill_criteria"] = kill_criteria_verdict(
        results["base"],
        blend_metrics[best_key],
        cost_key="net_mean_topk_excess_t1",
    )

    out_path = args.out / "ensemble_verdict.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))

    header = (
        f"{'window':<10}{'base IC':>10}{'master IC':>11}{best_key + ' IC':>16}"
        f"{'base net':>11}{'blend net':>11}{'blend t/o':>11}"
    )
    logger.info("%s", header)
    for name in windows:
        blend_w = blend_metrics[best_key][name]
        logger.info(
            "%-10s%+10.4f%+11.4f%+16.4f%+11.4f%+11.4f%11.1f",
            name,
            results["base"][name]["spearman"],
            results["master"][name]["spearman"],
            blend_w["spearman"],
            results["base"][name]["net_mean_topk_excess_t1"],
            blend_w["net_mean_topk_excess_t1"],
            blend_w["annualized_two_sided_turnover"],
        )
    verdict = results["kill_criteria"]
    logger.info(
        "VERDICT: %s (%d/%d wins, need %d) -> %s",
        verdict["verdict"],
        verdict["wins"],
        verdict["n_windows"],
        verdict["required_wins"],
        out_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
