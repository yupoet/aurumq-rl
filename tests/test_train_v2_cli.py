"""Unit tests for the Phase 15 train_v2 CLI helpers.

These tests do not run training; they exercise the schedule helper and the
new argparse flags so a typo or missing wiring is caught before a long
unattended run kicks off.
"""
from __future__ import annotations

import math

import pytest


def test_make_lr_callable_constant():
    from train_v2 import _make_lr_callable

    out = _make_lr_callable(1e-4, "constant", 0.1)
    # Constant returns the bare float (SB3 accepts either a float or a callable).
    assert out == 1e-4


def test_make_lr_callable_linear_endpoints():
    from train_v2 import _make_lr_callable

    sched = _make_lr_callable(1e-4, "linear", 0.1)
    # progress_remaining=1.0 -> initial_lr; =0.0 -> initial * final_frac.
    assert sched(1.0) == pytest.approx(1e-4)
    assert sched(0.0) == pytest.approx(1e-5)
    # Halfway is the linear mean.
    assert sched(0.5) == pytest.approx((1e-4 + 1e-5) / 2)


def test_make_lr_callable_cosine_endpoints():
    from train_v2 import _make_lr_callable

    sched = _make_lr_callable(1e-4, "cosine", 0.1)
    assert sched(1.0) == pytest.approx(1e-4)
    assert sched(0.0) == pytest.approx(1e-5)
    # Cosine half-way == linear half-way for half-cycle (cos(pi/2)=0).
    assert sched(0.5) == pytest.approx((1e-4 + 1e-5) / 2)


def test_make_lr_callable_cosine_monotonic():
    """LR should monotonically decrease as progress_remaining decreases."""
    from train_v2 import _make_lr_callable

    sched = _make_lr_callable(1e-4, "cosine", 0.0)
    prev = float("inf")
    for p_int in range(10, -1, -1):  # 1.0, 0.9, ..., 0.0
        p = p_int / 10.0
        cur = sched(p)
        assert cur <= prev + 1e-12
        prev = cur


def test_make_lr_callable_unknown_mode_raises():
    from train_v2 import _make_lr_callable

    with pytest.raises(ValueError, match="unknown lr-schedule"):
        _make_lr_callable(1e-4, "garbage", 0.1)


def test_parse_args_default_no_phase15_flags():
    from train_v2 import parse_args

    args = parse_args([
        "--total-timesteps", "1000",
        "--data-path", "x.parquet",
        "--start-date", "2023-01-01",
        "--end-date", "2023-01-31",
        "--out-dir", "/tmp/x",
    ])
    assert args.resume_from is None
    assert args.lr_schedule == "constant"
    assert args.lr_final_frac == 0.1
    assert args.drop_factor_prefix is None


def test_parse_args_resume_from():
    from pathlib import Path

    from train_v2 import parse_args

    args = parse_args([
        "--total-timesteps", "300000",
        "--data-path", "x.parquet",
        "--start-date", "2023-01-01",
        "--end-date", "2023-01-31",
        "--out-dir", "/tmp/x",
        "--resume-from", "/tmp/600k.zip",
        "--learning-rate", "3e-5",
    ])
    assert args.resume_from == Path("/tmp/600k.zip")
    assert args.learning_rate == 3e-5


def test_parse_args_lr_schedule_choices():
    from train_v2 import parse_args

    for mode in ("constant", "linear", "cosine"):
        args = parse_args([
            "--total-timesteps", "1000",
            "--data-path", "x.parquet",
            "--start-date", "2023-01-01",
            "--end-date", "2023-01-31",
            "--out-dir", "/tmp/x",
            "--lr-schedule", mode,
            "--lr-final-frac", "0.05",
        ])
        assert args.lr_schedule == mode
        assert args.lr_final_frac == 0.05


def test_parse_args_drop_factor_prefix_multi():
    from train_v2 import parse_args

    args = parse_args([
        "--total-timesteps", "1000",
        "--data-path", "x.parquet",
        "--start-date", "2023-01-01",
        "--end-date", "2023-01-31",
        "--out-dir", "/tmp/x",
        "--drop-factor-prefix", "mkt_", "ind_",
    ])
    assert args.drop_factor_prefix == ["mkt_", "ind_"]
