"""C7 regression tests: scripts/infer.py universe/factor alignment.

Production inference must map model scores onto the TRAINING-time stock
universe (``metadata["stock_codes"]``) with the TRAINING-time factor layout
(``metadata["factor_names"]``). The old code flattened today's universe and
pad/truncated the tail of the obs vector, so any universe drift shifted every
stock's factor block by whole rows and attached scores to the wrong codes.

Fixture design (synthetic, adj_factor=1.0):
  training universe  [600001, 600002, 600003, 600004]  (A, B, C, D)
  today's universe   [600001, 600002, 600004, 600005]  (C delisted, E new)
The fake agent returns fixed positional scores [1, 2, 3, 4] so a correct
implementation attaches A=1, B=2, C=3, D=4 — while the old flatten-by-today's
-universe logic gives D the 3.0 that belongs to C and hands E the 4.0.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from types import SimpleNamespace

# scripts/ is added to sys.path by tests/conftest.py
import infer
import numpy as np
import polars as pl
import pytest

TRAIN_CODES = ["600001.SH", "600002.SH", "600003.SH", "600004.SH"]
TODAY_CODES = ["600001.SH", "600002.SH", "600004.SH", "600005.SH"]
FACTOR_NAMES = ["alpha_001", "alpha_002"]
EVAL_DATE = "2024-01-08"
DATES = [datetime.date(2024, 1, d) for d in (2, 3, 4, 5, 8)]


def _write_panel(tmp_path: Path, codes: list[str], extra_factor: bool = False) -> Path:
    """Synthetic parquet: alpha_001 = stock position j, alpha_002 = 10 - j.

    ``extra_factor`` adds ``alpha_000`` (alphabetically FIRST, value -j so its
    z-scores differ from alpha_001's) to simulate panel schema drift: the old
    prefix-discovery truncated to factor_count alphabetically and would pick
    [alpha_000, alpha_001] instead of the trained [alpha_001, alpha_002].
    """
    rows = []
    for d in DATES:
        for j, code in enumerate(codes):
            row = {
                "ts_code": code,
                "trade_date": d,
                "close": 10.0 + j,
                "adj_factor": 1.0,  # no corporate actions in the fixture
                "pct_chg": 0.01,
                "vol": 1000.0,
                "alpha_001": float(j),
                "alpha_002": float(10 - j),
            }
            if extra_factor:
                row["alpha_000"] = float(-j)
            rows.append(row)
    out = tmp_path / "panel.parquet"
    pl.DataFrame(rows).write_parquet(out)
    return out


def _write_model_dir(tmp_path: Path, drop_keys: tuple[str, ...] = ()) -> Path:
    """Fake model dir with metadata.json only (agent itself is monkeypatched)."""
    model_dir = tmp_path / "model"
    model_dir.mkdir(exist_ok=True)
    meta = {
        "algorithm": "PPO",
        "training_timesteps": 1000,
        "obs_shape": [len(TRAIN_CODES) * len(FACTOR_NAMES)],
        "action_shape": [len(TRAIN_CODES)],
        "factor_count": len(FACTOR_NAMES),
        "factor_names": list(FACTOR_NAMES),
        "stock_codes": list(TRAIN_CODES),
    }
    for k in drop_keys:
        meta.pop(k, None)
    (model_dir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    return model_dir


class _FakeAgent:
    """RlAgentInference stand-in: fixed positional scores [1, 2, 3, 4].

    Records the obs it receives so tests can assert the feature layout.
    """

    last_obs: np.ndarray | None = None

    def __init__(self, model_dir: str | Path) -> None:
        meta = json.loads((Path(model_dir) / "metadata.json").read_text(encoding="utf-8"))
        # Mirror RlAgentMetadata's optional C7 fields (None when absent).
        meta.setdefault("factor_names", None)
        meta.setdefault("stock_codes", None)
        self.metadata = SimpleNamespace(**meta)

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32)
        expected = int(np.prod(self.metadata.obs_shape))
        assert obs.size == expected, f"model fed {obs.size} values, expects {expected}"
        _FakeAgent.last_obs = obs.reshape(-1).copy()
        return np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)


@pytest.fixture(autouse=True)
def _patch_agent(monkeypatch):
    monkeypatch.setattr("aurumq_rl.inference.RlAgentInference", _FakeAgent)
    _FakeAgent.last_obs = None


def _run_infer(model_dir: Path, pq: Path, out_path: Path, top_k: int = 4) -> int:
    return infer.main(
        [
            "--model",
            str(model_dir),
            "--data",
            str(pq),
            "--date",
            EVAL_DATE,
            "--top-k",
            str(top_k),
            "--universe-filter",
            "main_board_non_st",
            "--out",
            str(out_path),
        ]
    )


# ---------------------------------------------------------------------------
# 1) Universe drift: scores must attach to the TRAINING codes
# ---------------------------------------------------------------------------


def test_universe_drift_scores_attach_to_correct_stocks(tmp_path):
    pq = _write_panel(tmp_path, TODAY_CODES)
    model_dir = _write_model_dir(tmp_path)
    out_path = tmp_path / "picks.json"

    rc = _run_infer(model_dir, pq, out_path)
    assert rc == 0

    picks = {p["stock_code"]: p["score"] for p in json.loads(out_path.read_text())["picks"]}

    # Positional scores in the TRAINING universe: A=1, B=2, C=3, D=4.
    # Old flatten-today's-universe code mapped [1,2,3,4] onto [A,B,D,E],
    # so D got C's 3.0 and E got D's 4.0.
    assert picks["600004.SH"] == pytest.approx(4.0), "D must get its own score, not C's"
    assert picks["600001.SH"] == pytest.approx(1.0)
    assert picks["600002.SH"] == pytest.approx(2.0)
    # E trades today but was not in the training universe: not a candidate.
    assert "600005.SH" not in picks
    # C is in the training universe but absent today (delisted): the align
    # semantics mark it ST+suspended, so the eligibility mask excludes it.
    assert "600003.SH" not in picks


def test_universe_drift_logs_dropped_count(tmp_path, capsys):
    pq = _write_panel(tmp_path, TODAY_CODES)
    model_dir = _write_model_dir(tmp_path)

    rc = _run_infer(model_dir, pq, tmp_path / "picks.json")
    assert rc == 0
    err = capsys.readouterr().err
    # 1 stock trading today outside the training universe (E), 1 missing (C).
    assert "dropped 1" in err
    assert "missing" in err


# ---------------------------------------------------------------------------
# 2) Factor drift: features selected by metadata factor_names, exact order
# ---------------------------------------------------------------------------


def _zscore(vals: list[float]) -> np.ndarray:
    v = np.asarray(vals, dtype=np.float32)
    return (v - v.mean()) / (v.std() + 1e-8)


def test_factor_drift_extra_column_does_not_shift_features(tmp_path):
    """An alphabetically-early alpha_000 in the panel must NOT displace the
    trained factor layout (old prefix-discovery truncated to factor_count)."""
    pq = _write_panel(tmp_path, TODAY_CODES, extra_factor=True)
    model_dir = _write_model_dir(tmp_path)

    rc = _run_infer(model_dir, pq, tmp_path / "picks.json")
    assert rc == 0
    assert _FakeAgent.last_obs is not None

    obs = _FakeAgent.last_obs.reshape(len(TRAIN_CODES), len(FACTOR_NAMES))

    # Expected: z-scores over TODAY's 4-stock cross-section, realigned to the
    # training order [A, B, C, D] with missing C zero-filled.
    z1 = _zscore([0.0, 1.0, 2.0, 3.0])  # alpha_001 for [A, B, D, E]
    z2 = _zscore([10.0, 9.0, 8.0, 7.0])  # alpha_002 for [A, B, D, E]
    expected = np.array(
        [
            [z1[0], z2[0]],  # A
            [z1[1], z2[1]],  # B
            [0.0, 0.0],  # C missing today -> zero-padded
            [z1[2], z2[2]],  # D
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(obs, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# 3) Missing metadata keys: hard error naming the fix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "drop",
    [("factor_names",), ("stock_codes",), ("factor_names", "stock_codes")],
)
def test_missing_metadata_keys_hard_error(tmp_path, capsys, drop):
    pq = _write_panel(tmp_path, TODAY_CODES)
    model_dir = _write_model_dir(tmp_path, drop_keys=drop)

    rc = _run_infer(model_dir, pq, tmp_path / "picks.json")
    assert rc != 0, "legacy metadata without factor_names/stock_codes must hard-error"
    err = capsys.readouterr().err
    assert "factor_names" in err and "stock_codes" in err
    assert "re-export" in err.lower() or "retrain" in err.lower()


# ---------------------------------------------------------------------------
# 4) Obs-dim mismatch: fail loudly, never pad/truncate
# ---------------------------------------------------------------------------


def test_obs_dim_mismatch_fails_loudly(tmp_path, capsys):
    pq = _write_panel(tmp_path, TODAY_CODES)
    model_dir = _write_model_dir(tmp_path)
    # Corrupt obs_shape so aligned obs (8) cannot match the model (10).
    meta_path = model_dir / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["obs_shape"] = [10]
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    rc = _run_infer(model_dir, pq, tmp_path / "picks.json")
    assert rc != 0, "obs dim mismatch must be a hard error, not silent padding"
    assert "mismatch" in capsys.readouterr().err.lower()


# ---------------------------------------------------------------------------
# 5) Shared helper: alignment + drift stats (used by infer AND eval_backtest)
# ---------------------------------------------------------------------------


def test_align_panel_to_training_universe_stats():
    from aurumq_rl.data_loader import FactorPanel, align_panel_to_training_universe

    n_dates, n_factors = 2, 2
    codes_today = TODAY_CODES
    n_today = len(codes_today)
    panel = FactorPanel(
        factor_array=np.ones((n_dates, n_today, n_factors), dtype=np.float32),
        return_array=np.zeros((n_dates, n_today), dtype=np.float32),
        pct_change_array=np.zeros((n_dates, n_today), dtype=np.float32),
        is_st_array=np.zeros((n_dates, n_today), dtype=np.bool_),
        is_suspended_array=np.zeros((n_dates, n_today), dtype=np.bool_),
        days_since_ipo_array=np.full((n_dates, n_today), 200.0, dtype=np.float32),
        dates=[datetime.date(2024, 1, 2), datetime.date(2024, 1, 3)],
        stock_codes=list(codes_today),
        factor_names=list(FACTOR_NAMES),
        close_array=np.full((n_dates, n_today), 10.0, dtype=np.float32),
    )

    aligned, stats = align_panel_to_training_universe(panel, TRAIN_CODES)

    assert stats == {"kept": 3, "missing": 1, "dropped": 1}
    assert list(aligned.stock_codes) == TRAIN_CODES
    c_idx = TRAIN_CODES.index("600003.SH")
    assert aligned.is_st_array[:, c_idx].all(), "missing stock must be marked ST"
    assert aligned.is_suspended_array[:, c_idx].all(), "missing stock must be suspended"
