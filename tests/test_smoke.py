"""End-to-end smoke test: scripts/train.py --smoke-test must produce a
well-formed smoke_summary.json without requiring PyTorch/SB3."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.smoke
def test_train_smoke_test_runs_and_writes_summary(project_root: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "smoke_out"
    cmd = [
        sys.executable,
        str(project_root / "scripts" / "train.py"),
        "--smoke-test",
        "--out-dir",
        str(out_dir),
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, (
        f"smoke test failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    )

    summary_path = out_dir / "smoke_summary.json"
    assert summary_path.exists(), f"smoke_summary.json not created at {summary_path}"

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    assert summary["mode"] == "smoke_test"
    assert summary["status"] == "ok"
    assert summary["n_steps"] > 0

    panel_shape = summary["panel_shape"]
    assert panel_shape["n_dates"] > 0
    assert panel_shape["n_stocks"] > 0
    assert panel_shape["n_factors"] > 0

    metrics_summary = summary["metrics_summary"]
    assert metrics_summary["n_checkpoints"] > 0
    assert metrics_summary["algorithm"] in {"PPO", "A2C", "SAC"}

    metrics_path = out_dir / "training_metrics.jsonl"
    assert metrics_path.exists()
    # metrics file has one JSON record per line
    lines = [line for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == metrics_summary["n_checkpoints"]
