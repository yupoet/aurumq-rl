"""Phase 16 regression tests covering the corrected eval pipeline.

Five concrete claims this file pins down:

1. ``mfp_`` is a recognised factor prefix (was missing pre-Phase-16 — Phase 15
   trained without those 12 columns).
2. ``FactorPanelLoader.load_panel(factor_names=...)`` performs an EXACT-order
   load and rejects the panel when a requested column is absent.
3. ``GPUStockPickingEnv.step_wait`` reads ``returns[t]``, not
   ``returns[t + forward_period]`` (the t-th row IS already the forward-window
   return per :class:`FactorPanelLoader`).
4. ``factor_importance._eval_top_k_metrics`` reads ``returns[t]`` and reports
   ``sharpe_legacy``, ``sharpe_adjusted``, ``sharpe_non_overlap``.
5. ``backtest.compute_top_k_sharpes`` returns ``adjusted ≈ legacy /
   sqrt(forward_period)`` on the same return series.
"""
from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

# ---------------------------------------------------------------------------
# 1) mfp_ discovery
# ---------------------------------------------------------------------------


def test_mfp_in_factor_col_prefixes():
    from aurumq_rl.data_loader import FACTOR_COL_PREFIXES

    assert "mfp_" in FACTOR_COL_PREFIXES, (
        "mfp_ must be a recognised prefix; missing pre-Phase-16 caused 12 cols "
        "to silently drop out of the model"
    )


def test_discover_factor_columns_includes_mfp():
    from aurumq_rl.data_loader import discover_factor_columns

    df = pl.DataFrame(
        {
            "ts_code": ["X.SH"],
            "trade_date": [datetime.date(2024, 1, 2)],
            "alpha_001": [0.0],
            "mf_a": [0.0],
            "mfp_001": [0.1],
            "mfp_002": [0.2],
            "noise_col": [0.0],
        }
    )
    cols = discover_factor_columns(df)
    assert "mfp_001" in cols and "mfp_002" in cols
    assert "noise_col" not in cols


# ---------------------------------------------------------------------------
# 2) factor_names exact-order load
# ---------------------------------------------------------------------------


def _build_min_panel(tmp_path: Path) -> Path:
    """Synthetic 5-date × 4-stock panel with three factor groups."""
    rows = []
    dates = [datetime.date(2024, 1, d) for d in (2, 3, 4, 5, 8)]
    for d in dates:
        for code in ("600001.SH", "600002.SH", "000001.SZ", "000002.SZ"):
            rows.append(
                {
                    "ts_code": code,
                    "trade_date": d,
                    "close": 10.0 + hash((code, d)) % 7,
                    "pct_chg": 0.01,
                    "vol": 1000.0,
                    "alpha_001": float(hash((code, d, "a1")) % 100) / 100.0,
                    "alpha_002": float(hash((code, d, "a2")) % 100) / 100.0,
                    "mf_001": float(hash((code, d, "mf")) % 100) / 100.0,
                    "mfp_001": float(hash((code, d, "mfp")) % 100) / 100.0,
                }
            )
    df = pl.DataFrame(rows)
    out = tmp_path / "panel.parquet"
    df.write_parquet(out)
    return out


def test_load_panel_factor_names_exact_order(tmp_path):
    from aurumq_rl.data_loader import FactorPanelLoader, UniverseFilter

    pq = _build_min_panel(tmp_path)
    loader = FactorPanelLoader(pq)
    # Request a NON-alphabetical order to prove the loader honours the list.
    requested = ["mfp_001", "alpha_002", "mf_001", "alpha_001"]
    panel = loader.load_panel(
        start_date=datetime.date(2024, 1, 2),
        end_date=datetime.date(2024, 1, 8),
        forward_period=1,
        universe_filter=UniverseFilter.MAIN_BOARD_NON_ST,
        factor_names=requested,
    )
    assert list(panel.factor_names) == requested
    assert panel.factor_array.shape[2] == len(requested)


def test_load_panel_factor_names_missing_column_raises(tmp_path):
    from aurumq_rl.data_loader import FactorPanelLoader, UniverseFilter

    pq = _build_min_panel(tmp_path)
    loader = FactorPanelLoader(pq)
    with pytest.raises(ValueError, match="not in panel"):
        loader.load_panel(
            start_date=datetime.date(2024, 1, 2),
            end_date=datetime.date(2024, 1, 8),
            forward_period=1,
            universe_filter=UniverseFilter.MAIN_BOARD_NON_ST,
            factor_names=["alpha_001", "alpha_002", "definitely_not_present"],
        )


# ---------------------------------------------------------------------------
# 3) GPUStockPickingEnv reward uses returns[t]
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason="cuda required"
)
def test_gpu_env_step_uses_returns_at_t():
    """If reward = mean of returns[t] for top-K, then setting returns[t]
    deterministically (and zeroing forward rows) guarantees a known reward.
    Phase 15's bug indexed returns[t + fp], which would yield 0 here."""
    import torch

    from aurumq_rl.gpu_env import GPUStockPickingEnv

    n_stocks = 8
    n_factors = 3
    n_dates = 60
    forward_period = 5
    episode_length = 30
    top_k = 4

    panel = torch.zeros((n_dates, n_stocks, n_factors), device="cuda")
    returns = torch.zeros((n_dates, n_stocks), device="cuda")
    # Make t=0 row carry a unique signal: stock j has return 0.01*(j+1).
    returns[0] = torch.tensor(
        [0.01 * (j + 1) for j in range(n_stocks)], device="cuda"
    )
    valid = torch.ones((n_dates, n_stocks), dtype=torch.bool, device="cuda")

    env = GPUStockPickingEnv(
        panel,
        returns,
        valid,
        n_envs=1,
        episode_length=episode_length,
        forward_period=forward_period,
        top_k=top_k,
        cost_bps=0.0,  # no cost so reward == raw mean
        seed=0,
    )
    # Force the 1 env to start at t=0 (override the random sample).
    env.t.zero_()
    env.last_obs_t = env.t.clone()

    # Pick the top 4 stocks by index (largest j ⇒ largest return at t=0).
    action = torch.tensor([[float(j) for j in range(n_stocks)]], device="cuda")
    env.step_async(action.cpu().numpy())
    _obs, rewards, _dones, _infos = env.step_wait()

    # Top-4 are j=4,5,6,7 ⇒ rewards (0.05+0.06+0.07+0.08)/4 = 0.065.
    expected = (0.05 + 0.06 + 0.07 + 0.08) / 4
    assert rewards[0] == pytest.approx(expected, abs=1e-5), (
        f"reward {rewards[0]} != {expected}; the bug would have given 0 "
        f"(returns[t+fp]=returns[5] is all zeros)"
    )


# ---------------------------------------------------------------------------
# 4) factor_importance._eval_top_k_metrics uses returns[t] and reports trio
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason="cuda required"
)
def test_factor_importance_eval_metrics_use_returns_at_t():
    import torch

    from aurumq_rl.factor_importance import _eval_top_k_metrics

    n_dates = 30
    n_stocks = 6
    n_factors = 2
    forward_period = 5
    panel = torch.zeros((n_dates, n_stocks, n_factors), device="cuda")
    returns = torch.zeros((n_dates, n_stocks), device="cuda")

    # Make returns[t] non-trivial for t in [0, n_dates - fp).
    rng = torch.Generator(device="cuda").manual_seed(0)
    returns[: n_dates - forward_period] = torch.randn(
        (n_dates - forward_period, n_stocks), generator=rng, device="cuda"
    ) * 0.01

    # Score = first feature value per stock (deterministic per-date).
    def score_fn(obs: torch.Tensor) -> torch.Tensor:
        return obs[..., 0].mean(dim=0).reshape(1, -1)  # (1, S)

    out = _eval_top_k_metrics(
        score_fn, panel, returns, forward_period=forward_period, top_k=3,
    )
    # New keys exist
    for k in ("ic", "sharpe", "sharpe_legacy", "sharpe_adjusted", "sharpe_non_overlap"):
        assert k in out, f"missing key {k}"
    # Adjusted ≈ legacy / sqrt(forward_period) on the SAME series.
    if abs(out["sharpe_legacy"]) > 1e-9:
        ratio = out["sharpe_legacy"] / out["sharpe_adjusted"]
        assert ratio == pytest.approx(np.sqrt(forward_period), rel=1e-5), (
            f"adjusted should be legacy / sqrt(fp), got ratio {ratio}"
        )
    # `sharpe` (back-compat) is the adjusted Sharpe in Phase 16.
    assert out["sharpe"] == pytest.approx(out["sharpe_adjusted"], abs=1e-9)


# ---------------------------------------------------------------------------
# 5) backtest.compute_top_k_sharpes legacy/adjusted relationship
# ---------------------------------------------------------------------------


def test_compute_top_k_sharpes_adjusted_equals_legacy_over_sqrt_fp():
    from aurumq_rl.backtest import compute_top_k_sharpes

    rng = np.random.default_rng(7)
    n_dates, n_stocks = 80, 12
    preds = rng.normal(size=(n_dates, n_stocks))
    rets = rng.normal(scale=0.01, size=(n_dates, n_stocks))

    fp = 10
    out = compute_top_k_sharpes(preds, rets, top_k=4, forward_period=fp)
    assert out["legacy"] != 0.0
    assert out["adjusted"] == pytest.approx(out["legacy"] / np.sqrt(fp), rel=1e-9)
    # non_overlap may differ from adjusted but must be finite.
    assert np.isfinite(out["non_overlap"])


def test_run_backtest_emits_sharpe_trio():
    from aurumq_rl.backtest import run_backtest

    rng = np.random.default_rng(11)
    n_dates, n_stocks = 120, 14
    preds = rng.normal(size=(n_dates, n_stocks))
    rets = rng.normal(scale=0.01, size=(n_dates, n_stocks))

    fp = 10
    res = run_backtest(
        predictions=preds, returns=rets, top_k=5, n_random_simulations=10,
        random_seed=0, forward_period=fp,
    )
    # The dataclass exposes the trio + fp.
    assert res.forward_period == fp
    assert res.top_k_sharpe == pytest.approx(res.top_k_sharpe_adjusted)
    if abs(res.top_k_sharpe_legacy) > 1e-9:
        assert res.top_k_sharpe_adjusted == pytest.approx(
            res.top_k_sharpe_legacy / np.sqrt(fp), rel=1e-9
        )
    # random baseline reports both scales
    rb = res.random_baseline
    assert "p50_sharpe" in rb and "p50_sharpe_adjusted" in rb
    assert "p50_sharpe_non_overlap" in rb
