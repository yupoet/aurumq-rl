"""Tests for Path 1 ensemble + calibration utilities."""
from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from p3.path1_ensemble import (
    calibrate_isotonic,
    pick_top_configs_by_val,
    seed_mean_ensemble,
)


def test_seed_mean_three_identical_predictions():
    """Three identical predictions averaged → unchanged."""
    df_a = pl.DataFrame({
        "trade_date": [dt.date(2025, 1, 1), dt.date(2025, 1, 2)],
        "ts_code": ["S1.SH", "S2.SH"],
        "score": [0.5, 0.7],
    })
    out = seed_mean_ensemble([df_a, df_a.clone(), df_a.clone()])
    assert out["score"].to_list() == pytest.approx([0.5, 0.7])


def test_seed_mean_averages_correctly():
    """Mean of [0.0, 0.5, 1.0] → 0.5 per row."""
    base_keys = pl.DataFrame({
        "trade_date": [dt.date(2025, 1, 1), dt.date(2025, 1, 2)],
        "ts_code": ["S1.SH", "S2.SH"],
    })
    df_a = base_keys.with_columns(pl.lit(0.0).alias("score"))
    df_b = base_keys.with_columns(pl.lit(0.5).alias("score"))
    df_c = base_keys.with_columns(pl.lit(1.0).alias("score"))
    out = seed_mean_ensemble([df_a, df_b, df_c])
    assert out["score"].to_list() == pytest.approx([0.5, 0.5])


def test_isotonic_preserves_order():
    """Isotonic-fit then transform → preserves rank ordering of inputs."""
    rng = np.random.default_rng(0)
    pred_h1 = rng.uniform(0, 1, size=1000)
    actual_h1 = 2 * pred_h1 + rng.normal(0, 0.05, size=1000)
    pred_h2 = rng.uniform(0, 1, size=500)

    calibrator = calibrate_isotonic(pred_h1, actual_h1)
    out_h2 = calibrator(pred_h2)

    assert np.all(np.argsort(pred_h2) == np.argsort(out_h2))


def test_pick_top_3_configs_by_val_primary():
    """Pick top-3 distinct CONFIGS (collapsing seeds) by VAL primary."""
    runs = {
        "nl31_lr030_mdl50_seed42": 0.001,
        "nl31_lr030_mdl50_seed43": 0.0015,
        "nl31_lr030_mdl50_seed44": 0.001,
        "nl63_lr050_mdl100_seed42": 0.003,
        "nl63_lr050_mdl100_seed43": 0.0028,
        "nl63_lr050_mdl100_seed44": 0.0032,
        "nl127_lr050_mdl100_seed42": 0.002,
        "nl127_lr050_mdl100_seed43": 0.0025,
        "nl127_lr050_mdl100_seed44": 0.0022,
        "nl63_lr030_mdl50_seed42": 0.0018,
        "nl63_lr030_mdl50_seed43": 0.0019,
        "nl63_lr030_mdl50_seed44": 0.0017,
    }
    top3 = pick_top_configs_by_val(runs, top_k=3)
    config_names = {n.rsplit("_seed", 1)[0] for n in top3}
    assert "nl63_lr050_mdl100" in config_names
    assert "nl127_lr050_mdl100" in config_names
    assert len(config_names) == 3
