"""Argparse-level tests for new training flags."""

from __future__ import annotations

import pytest

# scripts/ is added to sys.path by tests/conftest.py
from train import parse_args


def test_parse_default_no_vec_normalize():
    args = parse_args(["--out-dir", "/tmp/x"])
    assert args.vec_normalize is False
    assert args.learning_rate_schedule == "constant"
    assert args.policy_kwargs_json == "{}"


def test_parse_vec_normalize_flag():
    args = parse_args(["--out-dir", "/tmp/x", "--vec-normalize"])
    assert args.vec_normalize is True


def test_parse_lr_schedule_choices():
    for choice in ("constant", "linear", "cosine"):
        args = parse_args(["--out-dir", "/tmp/x", "--learning-rate-schedule", choice])
        assert args.learning_rate_schedule == choice


def test_parse_lr_schedule_rejects_unknown():
    with pytest.raises(SystemExit):
        parse_args(["--out-dir", "/tmp/x", "--learning-rate-schedule", "bogus"])


def test_parse_policy_kwargs_json():
    args = parse_args(
        [
            "--out-dir",
            "/tmp/x",
            "--policy-kwargs-json",
            '{"net_arch": [256, 256]}',
        ]
    )
    assert args.policy_kwargs_json == '{"net_arch": [256, 256]}'
