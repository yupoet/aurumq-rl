"""Integration tests for scripts/p3/master_ensemble_eval.py (CLI wiring).

Runs main() against a tiny synthetic bundle in tmp_path and asserts the §12.7d
contract: every metric block carries the turnover/cost fields, the verdict is
rendered with the cost veto enabled, and multi-seed --master-preds inputs are
rank-averaged into one master column.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from p3.master_ensemble_eval import main, parse_windows

N_STOCKS = 10
N_DATES = 12
CODES = [f"SYN{i:03d}" for i in range(N_STOCKS)]
# January dates: outside H1/H2 so those default windows stay empty in tests
DATES = [dt.date(2025, 1, 6) + dt.timedelta(days=i) for i in range(N_DATES)]
ANCHORS = DATES[:8]  # anchors need 2 later realized dates for e2/e3
WINDOW_SPEC = "W=2025-01-01:2025-01-31"


def _write_bundle(bundle: Path) -> None:
    bundle.mkdir(parents=True)
    rng = np.random.default_rng(7)
    long_rows = {
        "trade_date": [d for d in DATES for _ in CODES],
        "ts_code": CODES * N_DATES,
    }
    pl.DataFrame({**long_rows, "y": rng.normal(0.0, 0.02, N_DATES * N_STOCKS)}).write_parquet(
        bundle / "target_y.parquet"
    )
    pl.DataFrame(
        {**long_rows, "pct_chg_t_plus_1": rng.normal(0.0, 0.02, N_DATES * N_STOCKS)}
    ).write_parquet(bundle / "realized_returns.parquet")
    pl.DataFrame(
        {"trade_date": DATES, "eq_weight_pct_chg_t_plus_1": rng.normal(0.0, 0.005, N_DATES)}
    ).write_parquet(bundle / "market_returns.parquet")


def _write_preds(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    pl.DataFrame(
        {
            "trade_date": [d for d in ANCHORS for _ in CODES],
            "ts_code": CODES * len(ANCHORS),
            "score": rng.normal(0.0, 1.0, len(ANCHORS) * N_STOCKS),
        }
    ).write_parquet(path)


@pytest.fixture()
def bundle_and_preds(tmp_path: Path) -> dict[str, Path]:
    bundle = tmp_path / "bundle"
    _write_bundle(bundle)
    paths = {"bundle": bundle, "out": tmp_path / "out"}
    for name, seed in (("base", 1), ("m42", 42), ("m43", 43)):
        paths[name] = tmp_path / f"{name}.parquet"
        _write_preds(paths[name], seed)
    return paths


def _run(paths: dict[str, Path], master_flags: list[str]) -> dict:
    argv = [
        *master_flags,
        "--base-preds",
        str(paths["base"]),
        "--bundle",
        str(paths["bundle"]),
        "--out",
        str(paths["out"]),
        "--top-k",
        "3",
        "--extra-window",
        WINDOW_SPEC,
    ]
    assert main(argv) == 0
    return json.loads((paths["out"] / "ensemble_verdict.json").read_text())


def test_verdict_json_carries_cost_fields_and_cost_veto(bundle_and_preds):
    results = _run(bundle_and_preds, ["--master-preds", str(bundle_and_preds["m42"])])
    w_block = results["base"]["W"]
    for key in (
        "spearman",
        "primary_mean_top50_proximity_excess",
        "topk_daily_replaced_frac",
        "annualized_two_sided_turnover",
        "gross_mean_topk_excess_t1",
        "net_mean_topk_excess_t1",
    ):
        assert key in w_block, f"missing {key} in merged metric block"
    # gross data exists in W and costs are strictly positive -> net < gross
    assert w_block["net_mean_topk_excess_t1"] < w_block["gross_mean_topk_excess_t1"]
    verdict = results["kill_criteria"]
    assert verdict["cost_key"] == "net_mean_topk_excess_t1"
    assert verdict["verdict"] in ("KEEP", "KILL")
    assert results["n_seeds"] == 1
    # windows H1/H2 (empty synthetic data) still produce inert blocks, no crash
    assert results["base"]["H1"]["n_rows"] == 0


def test_multi_seed_masters_are_rank_averaged(bundle_and_preds):
    paths = bundle_and_preds
    results = _run(
        paths,
        ["--master-preds", str(paths["m42"]), "--master-preds", str(paths["m43"])],
    )
    assert results["n_seeds"] == 2
    # the ensembled master is a real average: its W-window IC differs from
    # either single seed's unless they tie (vanishingly unlikely under rng)
    single = _run(paths, ["--master-preds", str(paths["m42"])])
    assert results["master"]["W"]["spearman"] != single["master"]["W"]["spearman"]


def test_parse_windows_roundtrip_and_rejects_garbage():
    parsed = parse_windows(["W3=2026-01-05:2026-06-30"])
    assert parsed["W3"] == (dt.date(2026, 1, 5), dt.date(2026, 6, 30))
    with pytest.raises(SystemExit, match="bad --extra-window"):
        parse_windows(["W3=2026-01-05"])
