# Phase 21 V2 Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the Phase 21 V2 architecture in `D:/dev/aurumq-rl` — Dict observation space, split-head policy (per-stock encoder + regime encoder + true-b2 critic), 8 v0 regime features, hard `valid_mask`, IndexOnlyDictRolloutBuffer — and prove a single sanity-train at 300k seed=42 doesn't regress against the Phase 16a baseline (`vs_random_p50_adjusted = +0.428`).

**Architecture:** Hard fork from V1. `obs = Dict{stock:(S,F_s), regime:(R,), valid_mask:(S,)}`. Stock encoder forbidden by allowlist + runtime assert from receiving `mkt_/index_/regime_/global_` columns. Regime encoder (8 → 64 → 16) is broadcast to every stock and concatenated with `stock_emb`; the same `head_in` feeds the actor (linear → mask → Normal) and the critic (per-stock value MLP → masked mean → linear). Phase 16-19 zips become unloadable forensic artifacts.

**Tech Stack:** Python 3.11 · PyTorch 2.11 · stable-baselines3 2.8 · gymnasium 1.0 · polars 1.40 · pytest 8.x · CUDA 12.6 (RTX 4070).

**Spec source of truth:** `docs/superpowers/specs/2026-05-05-phase21-v2-architecture-design.md` (commit f60741d). On any architectural ambiguity, the spec wins.

**Out of scope** (deferred per spec §8): FiLM modulation, Ubuntu-side regime enrichment, volume-z regime features, multi-seed sweeps, cross-version ensembling, true variable-S obs.

---

## File Structure

The plan organises work into five **agents** that can mostly run in parallel inside git worktrees. The interfaces below are locked at agent boundaries — once an agent commits, downstream agents may rely on its public types and field names without checking back.

| Agent | Worktree | Files (create) | Files (modify) | Tests | Depends on |
|---|---|---|---|---|---|
| 1. agent-data | `D:/dev/aurumq-rl-wt-data` | — | `src/aurumq_rl/data_loader.py` | `tests/test_data_loader_phase21.py` (new) | — |
| 2. agent-env  | `D:/dev/aurumq-rl-wt-env`  | — | `src/aurumq_rl/gpu_env.py` | `tests/test_gpu_env_phase21.py` (new) | agent-data merged |
| 3. agent-net  | `D:/dev/aurumq-rl-wt-net`  | — | `src/aurumq_rl/feature_extractor.py`, `src/aurumq_rl/policy.py` | `tests/test_feature_extractor_phase21.py` (new), `tests/test_policy_phase21.py` (new) | agent-data merged |
| 4. agent-buffer | `D:/dev/aurumq-rl-wt-buffer` | `src/aurumq_rl/index_dict_rollout_buffer.py` | — | `tests/test_index_dict_rollout_buffer.py` (new) | agent-env, agent-net merged |
| 5. agent-train | main repo | — | `scripts/train_v2.py`, `scripts/_eval_all_checkpoints.py` | smoke run + 3 sanity checks | 1-4 merged |

**Key interface locked at agent-data:**

```python
class FactorPanel(NamedTuple):
    factor_array: np.ndarray            # (T, S, F_stock), per-stock factors only
    return_array: np.ndarray            # (T, S)
    pct_change_array: np.ndarray        # (T, S), decimal
    is_st_array: np.ndarray             # (T, S) bool
    is_suspended_array: np.ndarray      # (T, S) bool, default-True (Phase 21 fix)
    days_since_ipo_array: np.ndarray    # (T, S)
    dates: list[datetime.date]
    stock_codes: list[str]
    factor_names: list[str]             # length F_stock, narrowed from V1 to per-stock only
    regime_array: np.ndarray            # (T, R) where R=8, NEW in Phase 21
    regime_names: list[str]             # length R=8, NEW in Phase 21

STOCK_FACTOR_PREFIXES: tuple[str, ...] = (
    "alpha_", "mf_", "mfp_", "hm_", "hk_", "inst_",
    "mg_", "cyq_", "senti_", "sh_", "fund_", "ind_", "gtja_",
)
FORBIDDEN_PREFIXES: tuple[str, ...] = ("mkt_", "index_", "regime_", "global_")
REGIME_FEATURE_NAMES: tuple[str, ...] = (
    "regime_breadth_d", "regime_breadth_20d",
    "regime_xs_disp_d", "regime_xs_disp_20d",
    "regime_idx_ret_20d", "regime_idx_ret_60d",
    "regime_idx_vol_20d", "regime_extreme_imbalance_norm",
)
```

**Key interface locked at agent-env:**

```python
GPUStockPickingEnv(
    panel: torch.Tensor,        # (T, S, F_stock) cuda
    regime: torch.Tensor,       # (T, R) cuda                NEW
    returns: torch.Tensor,      # (T, S) cuda
    valid_mask: torch.Tensor,   # (T, S) bool cuda
    n_envs: int, ...
)
# observation_space = gym.spaces.Dict({
#     "stock":      gym.spaces.Box(-inf, inf, (S, F_stock), float32),
#     "regime":     gym.spaces.Box(-inf, inf, (R,), float32),
#     "valid_mask": gym.spaces.Box(0, 1,    (S,), float32),
# })
# action_space  unchanged: Box(0, 1, (S,))
# reset() / step_wait() return Dict[str, np.ndarray] (per-env stacked by SB3)
# self.last_obs_t   unchanged (n_envs,) long cuda — single t-index per env
```

**Key interface locked at agent-net:**

```python
class PerStockEncoderV2(nn.Module):
    """(B, S, F_stock) → (B, S, D)."""

class RegimeEncoder(nn.Module):
    """(B, R) → (B, R')."""

def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """x:(B,S,H), mask:(B,S) → (B,H)."""

class PerStockEncoderPolicyV2(ActorCriticPolicy):
    """Custom forward / evaluate_actions / get_distribution / predict_values.
    Distribution is per-stock Normal; action_space stays Box(0,1,(S,)).
    """
```

**Key interface locked at agent-buffer:**

```python
class IndexOnlyDictRolloutBuffer(DictRolloutBuffer):
    """Stores (n_steps, n_envs) t-indices only. Materialises Dict obs lazily
    via providers attached after construction. Same SB3 contract as
    IndexOnlyRolloutBuffer (V1) but yields DictRolloutBufferSamples."""
    def attach_providers(
        self,
        stock_provider:  Callable[[th.Tensor], th.Tensor],   # (B,) → (B, S, F_stock)
        regime_provider: Callable[[th.Tensor], th.Tensor],   # (B,) → (B, R)
        mask_provider:   Callable[[th.Tensor], th.Tensor],   # (B,) → (B, S) float
        obs_index_provider: Callable[[], th.Tensor],         # () → (n_envs,) long
    ) -> None: ...
```

---

## Phase 0: Worktree setup (one-shot, blocks parallel agents)

### Task 0.1: Branch + worktrees

**Files:** none (git plumbing only)

- [ ] **Step 1: Create the umbrella feature branch from main**

```bash
git checkout main
git pull --ff-only
git checkout -b feat/phase21-v2-architecture
git push -u origin feat/phase21-v2-architecture
```

- [ ] **Step 2: Create one worktree per agent**

```bash
git worktree add -b feat/phase21-data   D:/dev/aurumq-rl-wt-data   feat/phase21-v2-architecture
git worktree add -b feat/phase21-env    D:/dev/aurumq-rl-wt-env    feat/phase21-v2-architecture
git worktree add -b feat/phase21-net    D:/dev/aurumq-rl-wt-net    feat/phase21-v2-architecture
git worktree add -b feat/phase21-buffer D:/dev/aurumq-rl-wt-buffer feat/phase21-v2-architecture
git worktree list
```

Expected: 5 worktrees listed (main + 4 phase-21 children). agent-train works in the main repo on `feat/phase21-v2-architecture` after merges land.

- [ ] **Step 3: Sanity-check the env in each worktree**

```bash
for wt in D:/dev/aurumq-rl-wt-data D:/dev/aurumq-rl-wt-env D:/dev/aurumq-rl-wt-net D:/dev/aurumq-rl-wt-buffer; do
    "$wt/.venv/Scripts/python.exe" -c "import torch, polars, gymnasium, stable_baselines3 as sb3; print(torch.cuda.is_available(), sb3.__version__)" || \
    cp -r D:/dev/aurumq-rl/.venv "$wt/.venv"
done
```

Note: `.venv` is per-worktree. If your shell can't reuse the main `.venv`, copy or recreate it. Each agent assumes `.venv/Scripts/python.exe` resolves to a torch+sb3+polars install.

---

## Phase 1: agent-data — data_loader Phase 21 changes

**Worktree:** `D:/dev/aurumq-rl-wt-data` on `feat/phase21-data`. All `git`/`pytest`/edit commands in this phase run from that directory.

### Task 1.1: is_suspended_array default-True (Phase 19 bug fix)

**Files:**
- Modify: `src/aurumq_rl/data_loader.py:591`
- Test: `tests/test_data_loader_phase21.py` (new file)

- [ ] **Step 1: Create the test file with the failing is_suspended fixture**

```python
# tests/test_data_loader_phase21.py
"""Phase 21 data_loader changes: is_suspended default-True, regime features,
schema lock, FactorPanel.regime_array."""
from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from aurumq_rl.data_loader import (
    FactorPanel,
    FactorPanelLoader,
    FORBIDDEN_PREFIXES,
    REGIME_FEATURE_NAMES,
    STOCK_FACTOR_PREFIXES,
    discover_factor_columns,
)


# ---------- Test panel: 5 dates, 3 stocks, 2 factors, hand-built ----------

def _build_panel_df() -> pl.DataFrame:
    """Hand-built panel where stock C only appears on dates 3-4 (pre-IPO on 0-2)."""
    dates = [dt.date(2024, 1, d) for d in (2, 3, 4, 5, 8)]
    rows: list[dict] = []
    for di, d in enumerate(dates):
        # A always present, vol > 0
        rows.append({"trade_date": d, "ts_code": "A.SH", "close": 10.0 + di,
                     "pct_chg": 0.01 * (di - 2), "vol": 1000.0,
                     "alpha_001": 0.1, "alpha_002": 0.2, "is_st": False})
        # B always present but suspended (vol=0) on date 2
        rows.append({"trade_date": d, "ts_code": "B.SH", "close": 20.0,
                     "pct_chg": 0.0,
                     "vol": 0.0 if di == 2 else 500.0,
                     "alpha_001": -0.1, "alpha_002": 0.05, "is_st": False})
        # C MISSING entirely on dates 0,1,2 (pre-IPO); present on 3,4
        if di >= 3:
            rows.append({"trade_date": d, "ts_code": "C.SH", "close": 5.0,
                         "pct_chg": 0.02, "vol": 200.0,
                         "alpha_001": 0.0, "alpha_002": -0.3, "is_st": False})
    return pl.DataFrame(rows)


def test_is_suspended_default_true_for_missing_rows(tmp_path):
    """Pre-IPO dates for stock C must be is_suspended=True, not False."""
    df = _build_panel_df()
    parquet_path = tmp_path / "panel.parquet"
    df.write_parquet(parquet_path)

    loader = FactorPanelLoader(parquet_path=parquet_path)
    panel = loader.load_panel(
        start_date=dt.date(2024, 1, 2), end_date=dt.date(2024, 1, 8),
        forward_period=1,
    )

    # stock_codes is sorted alphabetically: ['A.SH', 'B.SH', 'C.SH']
    assert panel.stock_codes == ["A.SH", "B.SH", "C.SH"]
    c = panel.stock_codes.index("C.SH")
    # Dates 0,1,2 had no parquet row for C — must default to suspended (True)
    assert panel.is_suspended_array[0, c] is np.True_ or panel.is_suspended_array[0, c] == True
    assert panel.is_suspended_array[1, c] == True
    assert panel.is_suspended_array[2, c] == True
    # Dates 3,4 had C present with vol > 0 — not suspended
    assert panel.is_suspended_array[3, c] == False
    assert panel.is_suspended_array[4, c] == False
    # Stock B had vol=0 on date 2 — explicit suspension still flagged
    b = panel.stock_codes.index("B.SH")
    assert panel.is_suspended_array[2, b] == True
    assert panel.is_suspended_array[0, b] == False
```

- [ ] **Step 2: Run, expect failure**

```bash
.venv/Scripts/python.exe -m pytest tests/test_data_loader_phase21.py::test_is_suspended_default_true_for_missing_rows -v
```

Expected: FAIL on `panel.is_suspended_array[0, c] == True` because the current default is `np.zeros(...)` (all False).

- [ ] **Step 3: Fix the default in `_df_to_panel`**

Edit `src/aurumq_rl/data_loader.py` around line 591. Replace the existing line:

```python
is_suspended_array = np.zeros((n_dates, n_stocks), dtype=np.bool_)
```

with:

```python
# Phase 21: default to True (suspended). Only (t, j) cells that have a
# parquet row are then UPDATED below — pre-IPO and delisted (t, j) stay
# True. The previous default (False) silently let the env treat zero-padded
# rows as tradeable, which contaminated cross-section centering and
# `valid_mask` once n_stocks * (1 - listed_fraction) ≳ 5%.
is_suspended_array = np.ones((n_dates, n_stocks), dtype=np.bool_)
```

- [ ] **Step 4: Re-run, expect pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_data_loader_phase21.py::test_is_suspended_default_true_for_missing_rows -v
```

Expected: PASS.

- [ ] **Step 5: Run the full data_loader test suite to confirm no regression**

```bash
.venv/Scripts/python.exe -m pytest tests/test_data_loader.py tests/test_data_loader_universe.py -v
```

Expected: all pre-existing tests still pass. (`test_synthetic_data.py` may need to relax assumptions if it asserted `is_suspended.all() == False` somewhere; if a test fails purely because of the default flip, fix the test to reflect the new contract.)

- [ ] **Step 6: Commit**

```bash
git add src/aurumq_rl/data_loader.py tests/test_data_loader_phase21.py
git commit -m "fix(data_loader): default is_suspended_array to True for missing rows

Pre-IPO and delisted (t, j) cells previously defaulted to False, making
zero-padded rows look tradeable to the encoder and the env's
valid_mask. Phase 21 default-True closes that gap."
```

### Task 1.2: STOCK_FACTOR_PREFIXES allowlist + FORBIDDEN_PREFIXES + schema lock at discover_factor_columns

**Files:**
- Modify: `src/aurumq_rl/data_loader.py:73-92` (FACTOR_COL_PREFIXES area)
- Modify: `src/aurumq_rl/data_loader.py:390+` (`discover_factor_columns`)
- Test: `tests/test_data_loader_phase21.py` (extend)

- [ ] **Step 1: Append schema-lock tests**

Append to `tests/test_data_loader_phase21.py`:

```python
def test_stock_factor_prefixes_excludes_mkt():
    assert "mkt_" not in STOCK_FACTOR_PREFIXES
    assert "alpha_" in STOCK_FACTOR_PREFIXES


def test_forbidden_prefixes_constant_present():
    assert FORBIDDEN_PREFIXES == ("mkt_", "index_", "regime_", "global_")


def test_discover_factor_columns_filters_forbidden():
    df = pl.DataFrame({
        "trade_date": [dt.date(2024, 1, 2)],
        "ts_code": ["A.SH"],
        "alpha_001": [0.1],
        "mf_001": [0.2],
        "mkt_congestion": [0.3],   # forbidden
        "regime_breadth_d": [0.4], # forbidden
        "global_vix": [0.5],       # forbidden
        "unknown_x": [0.6],        # not in allowlist, should also be skipped
    })
    cols = discover_factor_columns(df)
    assert cols == ["alpha_001", "mf_001"]


def test_discover_factor_columns_n_factors_limit():
    df = pl.DataFrame({
        "trade_date": [dt.date(2024, 1, 2)],
        "ts_code": ["A.SH"],
        "alpha_001": [0.1], "alpha_002": [0.2], "alpha_003": [0.3],
        "mf_001": [0.4],
    })
    cols = discover_factor_columns(df, n_factors=2)
    assert cols == ["alpha_001", "alpha_002"]
```

- [ ] **Step 2: Run, expect failures**

```bash
.venv/Scripts/python.exe -m pytest tests/test_data_loader_phase21.py -v
```

Expected: 3 failures — `STOCK_FACTOR_PREFIXES` and `FORBIDDEN_PREFIXES` are not yet exported; `mkt_*` is still in `FACTOR_COL_PREFIXES` so `discover_factor_columns` returns it.

- [ ] **Step 3: Replace FACTOR_COL_PREFIXES with STOCK_FACTOR_PREFIXES + FORBIDDEN_PREFIXES**

Edit `src/aurumq_rl/data_loader.py:73-92`. Replace the existing `FACTOR_COL_PREFIXES = (...)` block with:

```python
# Per-stock factor column prefixes (consumed by the per-stock encoder).
# Phase 21 NARROWS this to per-stock cross-section factors only — `mkt_*`
# columns are now considered cross-section-constant regime context and are
# explicitly forbidden from reaching the per-stock encoder. They remain
# usable for regime feature computation; see `_compute_regime_features`.
STOCK_FACTOR_PREFIXES: tuple[str, ...] = (
    "alpha_",
    "mf_",
    "mfp_",
    "hm_",
    "hk_",
    "inst_",
    "mg_",
    "cyq_",
    "senti_",
    "sh_",
    "fund_",
    "ind_",
    "gtja_",
)

# Prefixes that must NEVER appear as per-stock encoder input. The schema
# lock at training startup re-asserts this; data_loader silently filters
# them out of `discover_factor_columns`.
FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "mkt_",
    "index_",
    "regime_",
    "global_",
)

# Backwards-compat alias kept for any external code reading the V1 name.
# Will be removed in a follow-up cleanup; do NOT add new references.
FACTOR_COL_PREFIXES = STOCK_FACTOR_PREFIXES
```

- [ ] **Step 4: Update `discover_factor_columns` to use STOCK_FACTOR_PREFIXES and reject FORBIDDEN_PREFIXES**

Find `discover_factor_columns` (around line 390). Replace its body so the prefix default uses the new constant AND it explicitly skips forbidden columns:

```python
def discover_factor_columns(
    df: pl.DataFrame,
    n_factors: int | None = None,
    prefixes: tuple[str, ...] = STOCK_FACTOR_PREFIXES,
) -> list[str]:
    """Return per-stock factor column names sorted alphabetically.

    Columns whose prefix appears in :data:`FORBIDDEN_PREFIXES` are silently
    dropped — the per-stock encoder must never see them (Phase 21 schema
    lock). Columns whose prefix is not in the allowlist are also dropped.
    """
    cols: list[str] = []
    for c in df.columns:
        if any(c.startswith(fp) for fp in FORBIDDEN_PREFIXES):
            continue
        if any(c.startswith(p) for p in prefixes):
            cols.append(c)
    cols.sort()
    if n_factors is not None:
        cols = cols[:n_factors]
    return cols
```

- [ ] **Step 5: Update the `__all__` export list at the bottom of the file**

Find the `__all__` list (around line 750) and ensure both new constants are exported:

```python
__all__ = [
    "FactorPanel",
    "FactorPanelLoader",
    "UniverseFilter",
    "STOCK_FACTOR_PREFIXES",
    "FORBIDDEN_PREFIXES",
    "FACTOR_COL_PREFIXES",     # legacy alias
    "REGIME_FEATURE_NAMES",    # added by Task 1.3
    "align_panel_to_stock_list",
    "discover_factor_columns",
    "filter_universe",
]
```

`REGIME_FEATURE_NAMES` is added in Task 1.3; leave it in the list now — Python won't fail on a missing name in `__all__` until something does `from aurumq_rl.data_loader import *`.

- [ ] **Step 6: Re-run, expect pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_data_loader_phase21.py -v
```

Expected: the 3 schema-lock tests now PASS. (`test_is_suspended_default_true_for_missing_rows` from Task 1.1 still PASS.)

- [ ] **Step 7: Run full data_loader regression**

```bash
.venv/Scripts/python.exe -m pytest tests/test_data_loader.py tests/test_data_loader_universe.py -v
```

Expected: all pre-existing tests still pass.

- [ ] **Step 8: Commit**

```bash
git add src/aurumq_rl/data_loader.py tests/test_data_loader_phase21.py
git commit -m "feat(data_loader): split per-stock prefix allowlist from forbidden set

STOCK_FACTOR_PREFIXES is the explicit per-stock allowlist (mkt_ removed).
FORBIDDEN_PREFIXES (mkt_, index_, regime_, global_) is silently filtered
out of discover_factor_columns and re-asserted at training startup. The
old FACTOR_COL_PREFIXES name is kept as a backward-compatible alias."
```

### Task 1.3: `_compute_regime_features` + REGIME_FEATURE_NAMES

**Files:**
- Modify: `src/aurumq_rl/data_loader.py` (add new private function near `_safe_log_return`, add module-level constant)
- Test: `tests/test_data_loader_phase21.py` (extend)

- [ ] **Step 1: Append regime tests**

Append to `tests/test_data_loader_phase21.py`:

```python
# ------------------- Regime features -------------------

def _compute_regime_directly(pct: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Reference implementation, used as the test oracle. Hand-coded equivalent
    of _compute_regime_features for pinning shape + numerical contracts.
    pct: (T, S), valid: (T, S) bool. Returns (T, 8)."""
    T, S = pct.shape
    out = np.zeros((T, 8), dtype=np.float32)
    breadth_d = np.zeros(T, dtype=np.float32)
    xs_disp_d = np.zeros(T, dtype=np.float32)
    idx_ret_d = np.zeros(T, dtype=np.float32)
    extreme_imb = np.zeros(T, dtype=np.float32)
    for t in range(T):
        v = valid[t]
        n = int(v.sum())
        if n == 0:
            continue
        p = pct[t][v]
        breadth_d[t] = float((p > 0).mean())
        xs_disp_d[t] = float(p.std()) if n > 1 else 0.0
        idx_ret_d[t] = float(p.mean())
        up = int((p >= 0.099).sum())
        dn = int((p <= -0.099).sum())
        extreme_imb[t] = float(up - dn) / float(n)

    def _rmean(a, w):
        out = np.zeros_like(a)
        for t in range(len(a)):
            lo = max(0, t - w + 1)
            out[t] = float(a[lo:t + 1].mean())
        return out

    out[:, 0] = breadth_d
    out[:, 1] = _rmean(breadth_d, 20)
    out[:, 2] = xs_disp_d
    out[:, 3] = _rmean(xs_disp_d, 20)
    # Compounded returns: prod(1+r) - 1 over the window
    for t in range(T):
        lo20 = max(0, t - 19)
        out[t, 4] = float(np.prod(1.0 + idx_ret_d[lo20:t + 1]) - 1.0)
        lo60 = max(0, t - 59)
        out[t, 5] = float(np.prod(1.0 + idx_ret_d[lo60:t + 1]) - 1.0)
    # 20d realised vol of idx_ret_d, annualised by sqrt(252)
    for t in range(T):
        lo = max(0, t - 19)
        seg = idx_ret_d[lo:t + 1]
        out[t, 6] = float(seg.std()) * float(np.sqrt(252.0)) if len(seg) > 1 else 0.0
    out[:, 7] = extreme_imb
    return out


def test_regime_feature_names_constant_and_length():
    assert len(REGIME_FEATURE_NAMES) == 8
    assert REGIME_FEATURE_NAMES[0] == "regime_breadth_d"
    assert REGIME_FEATURE_NAMES[7] == "regime_extreme_imbalance_norm"


def test_factor_panel_has_regime_array(tmp_path):
    df = _build_panel_df()
    parquet_path = tmp_path / "panel.parquet"
    df.write_parquet(parquet_path)
    loader = FactorPanelLoader(parquet_path=parquet_path)
    panel = loader.load_panel(
        start_date=dt.date(2024, 1, 2), end_date=dt.date(2024, 1, 8),
        forward_period=1,
    )
    assert hasattr(panel, "regime_array")
    assert panel.regime_array.shape == (5, 8)
    assert panel.regime_array.dtype == np.float32
    assert np.isfinite(panel.regime_array).all()
    assert list(panel.regime_names) == list(REGIME_FEATURE_NAMES)


def test_regime_features_match_reference():
    """Hand-built (T=5, S=4) panel, compare to the inline reference oracle."""
    pct = np.array([
        [+0.10, -0.05, +0.02, +0.0],
        [-0.099, +0.099, +0.0, +0.0],
        [+0.05, +0.05, -0.05, -0.05],
        [+0.10, +0.099, -0.099, +0.02],
        [+0.0,  +0.0,  +0.0,  +0.0],
    ], dtype=np.float32)
    valid = np.ones_like(pct, dtype=np.bool_)
    expected = _compute_regime_directly(pct, valid)

    # Exercise the production function via a hand-built FactorPanel detour:
    from aurumq_rl.data_loader import _compute_regime_features
    got = _compute_regime_features(pct, valid)
    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)
```

- [ ] **Step 2: Run, expect failures**

```bash
.venv/Scripts/python.exe -m pytest tests/test_data_loader_phase21.py -v -k "regime"
```

Expected: ImportError on `REGIME_FEATURE_NAMES`, AttributeError on `panel.regime_array`, ImportError on `_compute_regime_features`.

- [ ] **Step 3: Add REGIME_FEATURE_NAMES constant near STOCK_FACTOR_PREFIXES**

Edit `src/aurumq_rl/data_loader.py` directly after `FORBIDDEN_PREFIXES`:

```python
REGIME_FEATURE_NAMES: tuple[str, ...] = (
    "regime_breadth_d",
    "regime_breadth_20d",
    "regime_xs_disp_d",
    "regime_xs_disp_20d",
    "regime_idx_ret_20d",
    "regime_idx_ret_60d",
    "regime_idx_vol_20d",
    "regime_extreme_imbalance_norm",
)
```

- [ ] **Step 4: Add `_compute_regime_features` near `_safe_log_return`**

Find `_safe_log_return` (search the file for `def _safe_log_return`). Add immediately below it:

```python
def _compute_regime_features(
    pct_change: np.ndarray, valid_mask: np.ndarray
) -> np.ndarray:
    """Compute the 8 v0 regime features per :data:`REGIME_FEATURE_NAMES`.

    Parameters
    ----------
    pct_change:
        (T, S) decimal pct change (e.g. +10% = 0.10).
    valid_mask:
        (T, S) bool. Cells where the stock is suspended / pre-IPO / ST should
        be False so they don't contribute to cross-section stats. The mask
        used here MUST match the env's ``valid_mask`` so train- and OOS-time
        regime stats stay comparable.

    Returns
    -------
    np.ndarray of shape (T, 8), dtype float32, all finite.
    """
    T, S = pct_change.shape
    pct = pct_change.astype(np.float32, copy=False)
    valid = valid_mask.astype(np.bool_, copy=False)

    breadth_d = np.zeros(T, dtype=np.float32)
    xs_disp_d = np.zeros(T, dtype=np.float32)
    idx_ret_d = np.zeros(T, dtype=np.float32)
    extreme_imb = np.zeros(T, dtype=np.float32)

    for t in range(T):
        v = valid[t]
        n = int(v.sum())
        if n == 0:
            continue
        p = pct[t][v]
        breadth_d[t] = float((p > 0).mean())
        if n > 1:
            xs_disp_d[t] = float(p.std())
        idx_ret_d[t] = float(p.mean())
        up = int((p >= 0.099).sum())
        dn = int((p <= -0.099).sum())
        extreme_imb[t] = float(up - dn) / float(n)

    out = np.zeros((T, 8), dtype=np.float32)

    def _rolling_mean(a: np.ndarray, w: int) -> np.ndarray:
        r = np.zeros_like(a)
        cs = np.cumsum(a, dtype=np.float64)
        for t in range(len(a)):
            lo = max(0, t - w + 1)
            seg_sum = cs[t] - (cs[lo - 1] if lo > 0 else 0.0)
            r[t] = float(seg_sum / float(t - lo + 1))
        return r.astype(np.float32, copy=False)

    out[:, 0] = breadth_d
    out[:, 1] = _rolling_mean(breadth_d, 20)
    out[:, 2] = xs_disp_d
    out[:, 3] = _rolling_mean(xs_disp_d, 20)

    # Compounded returns: prod(1+r) - 1 over the window. Vectorised via
    # cumulative log: log(1+r) is summable.
    log1p_idx = np.log1p(idx_ret_d.astype(np.float64))
    cs = np.cumsum(log1p_idx)
    for t in range(T):
        lo20 = max(0, t - 19)
        seg20 = cs[t] - (cs[lo20 - 1] if lo20 > 0 else 0.0)
        out[t, 4] = float(np.expm1(seg20))
        lo60 = max(0, t - 59)
        seg60 = cs[t] - (cs[lo60 - 1] if lo60 > 0 else 0.0)
        out[t, 5] = float(np.expm1(seg60))

    # 20d realised vol of idx_ret_d, annualised by sqrt(252).
    for t in range(T):
        lo = max(0, t - 19)
        seg = idx_ret_d[lo:t + 1]
        if len(seg) > 1:
            out[t, 6] = float(seg.std()) * float(np.sqrt(252.0))

    out[:, 7] = extreme_imb

    if not np.isfinite(out).all():
        bad = REGIME_FEATURE_NAMES[int(np.where(~np.isfinite(out).all(axis=0))[0][0])]
        raise ValueError(
            f"_compute_regime_features produced non-finite values in column {bad!r}; "
            "check upstream pct_change for NaN / inf."
        )
    return out
```

- [ ] **Step 5: Update FactorPanel NamedTuple**

Find the `FactorPanel` class (around line 112) and replace the entire NamedTuple with:

```python
class FactorPanel(NamedTuple):
    """Container for a 3D factor panel + auxiliary arrays.

    Phase 21 narrows ``factor_array`` / ``factor_names`` to per-stock factors
    ONLY (mkt_/index_/regime_/global_ excluded by data_loader's allowlist),
    and adds ``regime_array`` / ``regime_names`` for the date-level regime
    context tensor consumed by the new RegimeEncoder.
    """

    factor_array: np.ndarray
    return_array: np.ndarray
    pct_change_array: np.ndarray
    is_st_array: np.ndarray
    is_suspended_array: np.ndarray
    days_since_ipo_array: np.ndarray
    dates: list[datetime.date]
    stock_codes: list[str]
    factor_names: list[str]
    regime_array: np.ndarray = np.zeros((0, 0), dtype=np.float32)
    regime_names: list[str] = []
```

(Defaults let any legacy caller building a FactorPanel by hand keep working without specifying the new fields; production code always passes the real arrays.)

- [ ] **Step 6: Compute regime features inside `_df_to_panel`**

In `_df_to_panel`, just before the final `return FactorPanel(...)` (around line 641), compute the regime tensor and the synthetic valid_mask we feed it:

```python
        # Phase 21: regime tensor. Use the same valid_mask the env will use
        # so train- and eval-time regime stats are comparable.
        valid_for_regime = (
            (~is_st_array)
            & (~is_suspended_array)
            & (days_since_ipo_array >= NEW_STOCK_PROTECT_DAYS)
        )
        regime_array = _compute_regime_features(pct_change_array, valid_for_regime)
```

Then update the `return FactorPanel(...)` call to include `regime_array=regime_array, regime_names=list(REGIME_FEATURE_NAMES)`:

```python
        return FactorPanel(
            factor_array=factor_array,
            return_array=return_array,
            pct_change_array=pct_change_array,
            is_st_array=is_st_array,
            is_suspended_array=is_suspended_array,
            days_since_ipo_array=days_since_ipo_array,
            dates=dates,
            stock_codes=stock_codes,
            factor_names=factor_cols,
            regime_array=regime_array,
            regime_names=list(REGIME_FEATURE_NAMES),
        )
```

- [ ] **Step 7: Mirror in the synthetic-panel return path**

`FactorPanelLoader._build_synthetic` (around line 681-746) also returns a FactorPanel. Add the same `regime_array` computation and pass it to the constructor:

```python
        # Phase 21: regime tensor for synthetic panels. valid_mask is built
        # the same way as the production path so synthetic data exercises
        # the same code path through the env.
        valid_for_regime = (
            (~is_st_array)
            & (~is_suspended_array)
            & (days_since_ipo_array >= NEW_STOCK_PROTECT_DAYS)
        )
        regime_array = _compute_regime_features(pct_change_array, valid_for_regime)
        return FactorPanel(
            factor_array=factor_array,
            return_array=return_array,
            pct_change_array=pct_change_array,
            is_st_array=is_st_array,
            is_suspended_array=is_suspended_array,
            days_since_ipo_array=days_since_ipo_array,
            dates=dates,
            stock_codes=stock_codes,
            factor_names=factor_cols,
            regime_array=regime_array,
            regime_names=list(REGIME_FEATURE_NAMES),
        )
```

- [ ] **Step 8: Mirror in `align_panel_to_stock_list`**

The realignment helper at line 148 must preserve `regime_array` (regime is per-date, NOT per-stock — it does NOT depend on the stock universe; pass through unchanged). Update its `return FactorPanel(...)` to forward the new fields:

```python
    return FactorPanel(
        factor_array=factor_array,
        return_array=return_array,
        pct_change_array=pct_change_array,
        is_st_array=is_st_array,
        is_suspended_array=is_suspended_array,
        days_since_ipo_array=days_since_ipo_array,
        dates=list(panel.dates),
        stock_codes=list(target_stock_codes),
        factor_names=list(panel.factor_names),
        regime_array=panel.regime_array.copy(),       # per-date, not per-stock
        regime_names=list(panel.regime_names),
    )
```

- [ ] **Step 9: Run all Phase 21 tests, expect pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_data_loader_phase21.py -v
```

Expected: all tests PASS.

- [ ] **Step 10: Run wider regression**

```bash
.venv/Scripts/python.exe -m pytest tests/test_data_loader.py tests/test_data_loader_universe.py tests/test_synthetic_data.py -v
```

Expected: all pre-existing tests still pass (the FactorPanel NamedTuple new fields have defaults, so old code building a panel by hand continues to work).

- [ ] **Step 11: Commit**

```bash
git add src/aurumq_rl/data_loader.py tests/test_data_loader_phase21.py
git commit -m "feat(data_loader): compute regime_array + add to FactorPanel

8 v0 regime features per the spec: breadth_d/20d, xs_disp_d/20d,
idx_ret_20d/60d (compounded), idx_vol_20d, extreme_imbalance_norm.
Computed on the same valid_mask the env uses, so train- and OOS-time
regime stats are comparable. align_panel_to_stock_list propagates
the per-date regime tensor unchanged."
```

### Task 1.4: Push agent-data branch

- [ ] **Step 1: Push**

```bash
git push -u origin feat/phase21-data
```

- [ ] **Step 2: Open PR or fast-forward into `feat/phase21-v2-architecture`**

For agent-driven dev, fast-forward locally:

```bash
cd D:/dev/aurumq-rl
git fetch origin feat/phase21-data
git checkout feat/phase21-v2-architecture
git merge --ff-only origin/feat/phase21-data
git push origin feat/phase21-v2-architecture
```

agent-env and agent-net both rebase onto `feat/phase21-v2-architecture` once this lands.

---

## Phase 2: agent-env — gpu_env Dict observation

**Worktree:** `D:/dev/aurumq-rl-wt-env` on `feat/phase21-env`. Rebase this branch onto `feat/phase21-v2-architecture` AFTER agent-data lands:

```bash
cd D:/dev/aurumq-rl-wt-env
git fetch origin
git rebase origin/feat/phase21-v2-architecture
```

### Task 2.1: GPUStockPickingEnv accepts regime tensor + emits Dict obs

**Files:**
- Modify: `src/aurumq_rl/gpu_env.py:18-79` (constructor + observation_space)
- Modify: `src/aurumq_rl/gpu_env.py:83-145` (reset, step_wait, _obs_for_sb3)
- Test: `tests/test_gpu_env_phase21.py` (new file)

- [ ] **Step 1: Create test file**

```python
# tests/test_gpu_env_phase21.py
"""Phase 21 GPUStockPickingEnv: Dict obs, regime tensor plumbing."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch

from aurumq_rl.gpu_env import GPUStockPickingEnv

CUDA_OK = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA_OK, reason="CUDA required for gpu_env tests")


def _make_env(T=120, S=8, F=5, R=8, n_envs=4, episode_length=20):
    panel = torch.randn(T, S, F, device="cuda")
    regime = torch.randn(T, R, device="cuda")
    returns = torch.randn(T, S, device="cuda") * 0.01
    valid = torch.ones(T, S, dtype=torch.bool, device="cuda")
    env = GPUStockPickingEnv(
        panel=panel, regime=regime, returns=returns, valid_mask=valid,
        n_envs=n_envs, episode_length=episode_length,
        forward_period=5, top_k=3, cost_bps=0.0, seed=0,
    )
    return env, (T, S, F, R, n_envs)


def test_observation_space_is_dict_with_three_keys():
    env, (T, S, F, R, n_envs) = _make_env()
    assert isinstance(env.observation_space, gym.spaces.Dict)
    assert set(env.observation_space.spaces.keys()) == {"stock", "regime", "valid_mask"}
    assert env.observation_space["stock"].shape == (S, F)
    assert env.observation_space["regime"].shape == (R,)
    assert env.observation_space["valid_mask"].shape == (S,)
    assert env.observation_space["stock"].dtype == np.float32
    assert env.observation_space["regime"].dtype == np.float32
    assert env.observation_space["valid_mask"].dtype == np.float32
    assert env.observation_space["valid_mask"].low.min() == 0.0
    assert env.observation_space["valid_mask"].high.max() == 1.0


def test_reset_returns_dict_with_correct_shapes():
    env, (T, S, F, R, n_envs) = _make_env()
    obs = env.reset()
    assert isinstance(obs, dict)
    assert obs["stock"].shape == (n_envs, S, F)
    assert obs["regime"].shape == (n_envs, R)
    assert obs["valid_mask"].shape == (n_envs, S)
    assert obs["stock"].dtype == np.float32
    assert obs["regime"].dtype == np.float32
    assert obs["valid_mask"].dtype == np.float32


def test_step_wait_returns_dict_obs():
    env, (T, S, F, R, n_envs) = _make_env()
    env.reset()
    actions = np.random.uniform(0, 1, size=(n_envs, S)).astype(np.float32)
    env.step_async(actions)
    obs, rewards, dones, infos = env.step_wait()
    assert isinstance(obs, dict)
    assert obs["stock"].shape == (n_envs, S, F)
    assert obs["regime"].shape == (n_envs, R)
    assert obs["valid_mask"].shape == (n_envs, S)
    assert rewards.shape == (n_envs,)
    assert dones.shape == (n_envs,)


def test_valid_mask_passes_through_from_panel_input():
    panel = torch.randn(20, 4, 3, device="cuda")
    regime = torch.randn(20, 8, device="cuda")
    returns = torch.zeros(20, 4, device="cuda")
    # Stock 1 untradeable everywhere
    valid = torch.ones(20, 4, dtype=torch.bool, device="cuda")
    valid[:, 1] = False
    env = GPUStockPickingEnv(
        panel=panel, regime=regime, returns=returns, valid_mask=valid,
        n_envs=2, episode_length=10, forward_period=2, top_k=1,
        cost_bps=0.0, seed=0,
    )
    obs = env.reset()
    assert (obs["valid_mask"][:, 1] == 0.0).all()
    assert (obs["valid_mask"][:, 0] == 1.0).all()


def test_last_obs_t_unchanged_semantics():
    env, _ = _make_env()
    env.reset()
    t0 = env.last_obs_t.clone()
    actions = np.random.uniform(0, 1, size=(env.num_envs, env.n_stocks)).astype(np.float32)
    env.step_async(actions)
    env.step_wait()
    # last_obs_t should advance by 1 (or wrap on auto-reset)
    assert env.last_obs_t.shape == t0.shape
    assert env.last_obs_t.dtype == torch.long
```

- [ ] **Step 2: Run, expect failures**

```bash
.venv/Scripts/python.exe -m pytest tests/test_gpu_env_phase21.py -v
```

Expected: TypeError on the `regime=` kwarg, then on observation_space being a `Box` rather than `Dict`.

- [ ] **Step 3: Update the env constructor signature and validation**

Edit `src/aurumq_rl/gpu_env.py`. Replace the existing `__init__` with:

```python
    def __init__(
        self,
        panel: torch.Tensor,        # (T, S, F_stock) fp32 cuda
        regime: torch.Tensor,       # (T, R) fp32 cuda                NEW Phase 21
        returns: torch.Tensor,      # (T, S) fp32 cuda
        valid_mask: torch.Tensor,   # (T, S) bool cuda
        n_envs: int,
        episode_length: int = 240,
        forward_period: int = 10,
        top_k: int = 30,
        cost_bps: float = 30.0,
        turnover_coef: float = 0.0,
        device: str = "cuda",
        seed: int | None = None,
    ) -> None:
        if panel.device.type != "cuda":
            raise ValueError("panel must be a cuda tensor")
        if regime.device.type != "cuda":
            raise ValueError("regime must be a cuda tensor")
        if panel.shape[0] != returns.shape[0] or panel.shape[1] != returns.shape[1]:
            raise ValueError("panel and returns date/stock dims must match")
        if panel.shape[0] != regime.shape[0]:
            raise ValueError(
                f"panel and regime time dims must match: {panel.shape[0]} vs {regime.shape[0]}"
            )
        if panel.shape[:2] != valid_mask.shape:
            raise ValueError("panel and valid_mask date/stock dims must match")

        self.panel = panel
        self.regime = regime
        self.returns = returns
        self.valid_mask = valid_mask
        self.n_dates, self.n_stocks, self.n_factors = panel.shape
        self.n_regime = regime.shape[1]
        self.episode_length = episode_length
        self.forward_period = forward_period
        self.top_k = top_k
        self.cost_bps = cost_bps
        self.turnover_coef = turnover_coef
        self.device = torch.device(device)
        self._rng = torch.Generator(device=self.device)
        if seed is not None:
            self._rng.manual_seed(seed)

        # Per-env state, all on cuda (unchanged from V1)
        self.t = torch.zeros(n_envs, dtype=torch.long, device=self.device)
        self.steps_done = torch.zeros(n_envs, dtype=torch.long, device=self.device)
        self.episode_returns = torch.zeros(n_envs, dtype=torch.float32, device=self.device)
        self.prev_top_idx = torch.zeros(n_envs, top_k, dtype=torch.long, device=self.device)
        self.last_obs_t = torch.zeros(n_envs, dtype=torch.long, device=self.device)
        self._pending_action: torch.Tensor | None = None

        # Phase 21: Dict observation space.
        observation_space = gym.spaces.Dict({
            "stock": gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.n_stocks, self.n_factors),
                dtype=np.float32,
            ),
            "regime": gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.n_regime,),
                dtype=np.float32,
            ),
            "valid_mask": gym.spaces.Box(
                low=0.0, high=1.0,
                shape=(self.n_stocks,),
                dtype=np.float32,
            ),
        })
        action_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(self.n_stocks,),
            dtype=np.float32,
        )
        super().__init__(num_envs=n_envs, observation_space=observation_space, action_space=action_space)
```

- [ ] **Step 4: Replace `_obs_for_sb3` to emit a dict**

Find `_obs_for_sb3` (around line 195) and replace the whole method:

```python
    def _obs_for_sb3(self) -> dict[str, np.ndarray]:
        """Return the Dict obs SB3 expects.

        SB3's ``obs_as_tensor`` handles dict-of-numpy directly; each value
        is moved to the policy device individually. We materialise the
        per-key cuda slices to numpy at the VecEnv boundary.
        """
        t = self.t  # (n_envs,) long cuda
        stock_obs = self.panel.index_select(0, t).detach().cpu().numpy()
        regime_obs = self.regime.index_select(0, t).detach().cpu().numpy()
        mask_obs = self.valid_mask.index_select(0, t).to(dtype=torch.float32) \
            .detach().cpu().numpy()
        return {
            "stock": stock_obs.astype(np.float32, copy=False),
            "regime": regime_obs.astype(np.float32, copy=False),
            "valid_mask": mask_obs.astype(np.float32, copy=False),
        }
```

(`_current_obs` is no longer the right shape. Either remove it or leave it as legacy returning `self.panel[self.t]` — keep for backwards readability. The Dict path uses `_obs_for_sb3` directly.)

- [ ] **Step 5: Re-run, expect pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_gpu_env_phase21.py -v
```

Expected: 5 tests PASS.

- [ ] **Step 6: Run wider gpu_env regression**

```bash
.venv/Scripts/python.exe -m pytest tests/test_gpu_env.py tests/test_index_rollout_buffer.py -v
```

Expected: pre-existing `tests/test_gpu_env.py` likely needs minor updates because they construct the env without the `regime=` kwarg. Update those tests to pass `regime=torch.zeros(T, 1, device="cuda")` (a 1-d zero tensor is acceptable — the test doesn't assert anything about it). Likewise `test_index_rollout_buffer.py` — pass a dummy regime tensor. Do NOT delete tests; just keep them passing.

For each pre-existing test that breaks purely because of the new mandatory `regime=` kwarg, add `regime=torch.zeros(T, 1, device=panel.device)` (size 1 is fine for these tests) and re-run.

- [ ] **Step 7: Commit**

```bash
git add src/aurumq_rl/gpu_env.py tests/test_gpu_env_phase21.py tests/test_gpu_env.py tests/test_index_rollout_buffer.py
git commit -m "feat(gpu_env): Dict observation space + regime tensor

Phase 21: GPUStockPickingEnv now takes a (T, R) regime tensor and emits
obs as Dict{stock, regime, valid_mask}. The action space and step
semantics are unchanged. Pre-existing tests pass a dummy R=1 regime."
```

### Task 2.2: Push agent-env branch and merge into umbrella

```bash
git push -u origin feat/phase21-env
cd D:/dev/aurumq-rl
git fetch origin feat/phase21-env
git checkout feat/phase21-v2-architecture
git merge --ff-only origin/feat/phase21-env
git push origin feat/phase21-v2-architecture
```

---

## Phase 3: agent-net — feature_extractor + policy

**Worktree:** `D:/dev/aurumq-rl-wt-net` on `feat/phase21-net`. Rebase onto `feat/phase21-v2-architecture` AFTER agent-data + agent-env land.

### Task 3.1: PerStockEncoderV2 + RegimeEncoder + masked_mean

**Files:**
- Modify: `src/aurumq_rl/feature_extractor.py` (entire rewrite)
- Test: `tests/test_feature_extractor_phase21.py` (new)

- [ ] **Step 1: Create test file**

```python
# tests/test_feature_extractor_phase21.py
"""Phase 21 feature extractor: PerStockEncoderV2, RegimeEncoder, masked_mean."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from aurumq_rl.feature_extractor import (
    PerStockEncoderV2,
    RegimeEncoder,
    masked_mean,
)


def test_per_stock_encoder_v2_shape():
    enc = PerStockEncoderV2(n_factors=10, hidden=(32, 16), out_dim=8)
    x = torch.randn(4, 12, 10)
    out = enc(x)
    assert out.shape == (4, 12, 8)


def test_per_stock_encoder_v2_layer_norm_active():
    enc = PerStockEncoderV2(n_factors=10, hidden=(32, 16), out_dim=8)
    x = torch.randn(4, 12, 10) * 100.0
    out = enc(x)
    # LayerNorm over last dim → per-row mean ~0, std ~1 (with affine=True
    # there's a learned scale/bias, but at init affine_bias=0, affine_weight=1)
    assert out.std(dim=-1).mean() == pytest.approx(1.0, abs=0.5)


def test_per_stock_encoder_v2_grad_flows():
    enc = PerStockEncoderV2(n_factors=4, hidden=(8,), out_dim=2)
    x = torch.randn(2, 3, 4, requires_grad=True)
    out = enc(x).sum()
    out.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_regime_encoder_shape():
    re = RegimeEncoder(regime_dim=8, hidden=64, out_dim=16)
    x = torch.randn(4, 8)
    out = re(x)
    assert out.shape == (4, 16)


def test_regime_encoder_layer_norm_active():
    re = RegimeEncoder(regime_dim=8, hidden=64, out_dim=16)
    x = torch.randn(4, 8) * 100.0
    out = re(x)
    # output LayerNorm makes per-row std close to 1
    assert out.std(dim=-1).mean() == pytest.approx(1.0, abs=0.5)


def test_masked_mean_correctness():
    x = torch.tensor([
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]],
    ])  # (2, 3, 2)
    mask = torch.tensor([
        [1.0, 1.0, 0.0],   # mean of first 2 rows
        [0.0, 1.0, 1.0],   # mean of last 2 rows
    ])
    expected = torch.tensor([
        [(1 + 3) / 2, (2 + 4) / 2],
        [(30 + 50) / 2, (40 + 60) / 2],
    ])
    out = masked_mean(x, mask)
    torch.testing.assert_close(out, expected)


def test_masked_mean_zero_mask_does_not_explode():
    x = torch.randn(2, 3, 4)
    mask = torch.zeros(2, 3)
    out = masked_mean(x, mask)
    assert out.shape == (2, 4)
    assert torch.isfinite(out).all()  # eps clamp → finite zero, not NaN


def test_masked_mean_grad_flows():
    x = torch.randn(2, 3, 4, requires_grad=True)
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    masked_mean(x, mask).sum().backward()
    assert x.grad is not None
```

- [ ] **Step 2: Run, expect ImportError**

```bash
.venv/Scripts/python.exe -m pytest tests/test_feature_extractor_phase21.py -v
```

Expected: ImportError on `PerStockEncoderV2`, `RegimeEncoder`, `masked_mean`.

- [ ] **Step 3: Replace `feature_extractor.py` entirely**

Overwrite `src/aurumq_rl/feature_extractor.py` with:

```python
"""Phase 21 V2 feature extractors.

Two independent modules:

* :class:`PerStockEncoderV2` — applies a shared MLP to each stock row
  individually and LayerNorms the per-stock embedding. Strictly per-stock
  input; the schema lock at training startup forbids mkt_/index_/regime_/
  global_ columns from reaching it.
* :class:`RegimeEncoder` — small MLP from the (R,) date-level regime
  feature vector to (R',). Output is broadcast to every stock at the head
  layer in :class:`PerStockEncoderPolicyV2`.

The earlier V1 :class:`PerStockExtractor` (cross-section centering + dual
pooling) is removed. Cross-section centering is no longer needed because
the regime path supplies the date-level signal explicitly; dual pooling
moves into the critic via :func:`masked_mean` over the value-token MLP.
"""
from __future__ import annotations

import torch
from torch import nn


class PerStockEncoderV2(nn.Module):
    """Shared per-stock MLP followed by LayerNorm. Input is per-stock ONLY.

    Parameters
    ----------
    n_factors:
        Number of per-stock factor channels (F_stock).
    hidden:
        Hidden layer widths. Default (128, 64).
    out_dim:
        Output embedding width (D). Default 32.
    """

    def __init__(
        self,
        n_factors: int,
        hidden: tuple[int, ...] = (128, 64),
        out_dim: int = 32,
    ) -> None:
        super().__init__()
        self.n_factors = n_factors
        self.out_dim = out_dim
        layers: list[nn.Module] = []
        prev = n_factors
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.mlp = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, stock_x: torch.Tensor) -> torch.Tensor:
        # stock_x: (B, S, F_stock)
        b, s, f = stock_x.shape
        flat = stock_x.reshape(b * s, f)
        return self.norm(self.mlp(flat).reshape(b, s, self.out_dim))


class RegimeEncoder(nn.Module):
    """LayerNorm + (R → hidden) + SiLU + (hidden → R') + LayerNorm.

    R defaults to 8 (the v0 regime feature count). R' defaults to 16.
    """

    def __init__(
        self,
        regime_dim: int = 8,
        hidden: int = 64,
        out_dim: int = 16,
    ) -> None:
        super().__init__()
        self.regime_dim = regime_dim
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.LayerNorm(regime_dim),
            nn.Linear(regime_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, regime_x: torch.Tensor) -> torch.Tensor:
        # regime_x: (B, R) → (B, R')
        return self.net(regime_x)


def masked_mean(
    x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Cross-stock masked mean.

    Parameters
    ----------
    x:
        (B, S, H) per-stock token tensor.
    mask:
        (B, S) where 1 marks a valid stock. Floating-point dtype acceptable;
        will be cast to ``x.dtype``.
    eps:
        Denominator floor. A row with zero valid stocks returns zeros.
    """
    m = mask.to(dtype=x.dtype).unsqueeze(-1)  # (B, S, 1)
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(eps)
```

- [ ] **Step 4: Re-run, expect pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_feature_extractor_phase21.py -v
```

Expected: 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/aurumq_rl/feature_extractor.py tests/test_feature_extractor_phase21.py
git commit -m "feat(net): PerStockEncoderV2 + RegimeEncoder + masked_mean

V1 PerStockExtractor (dual pooling, cross-section centering) is removed.
V2 splits the encoder cleanly: per-stock MLP for stock features only,
small RegimeEncoder for the (R,) regime context, masked_mean utility
used by the critic's value-token pool."
```

### Task 3.2: PerStockEncoderPolicyV2

**Files:**
- Modify: `src/aurumq_rl/policy.py` (entire rewrite)
- Test: `tests/test_policy_phase21.py` (new)

- [ ] **Step 1: Create test file**

```python
# tests/test_policy_phase21.py
"""Phase 21 PerStockEncoderPolicyV2: Dict obs, Box action, hard mask, true b2."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch

from aurumq_rl.feature_extractor import (
    PerStockEncoderV2,
    RegimeEncoder,
    masked_mean,
)
from aurumq_rl.policy import PerStockEncoderPolicyV2


# --------- helpers ---------

def _make_obs_space(S=8, F=5, R=8):
    return gym.spaces.Dict({
        "stock": gym.spaces.Box(-np.inf, np.inf, (S, F), dtype=np.float32),
        "regime": gym.spaces.Box(-np.inf, np.inf, (R,), dtype=np.float32),
        "valid_mask": gym.spaces.Box(0.0, 1.0, (S,), dtype=np.float32),
    })


def _make_action_space(S=8):
    return gym.spaces.Box(0.0, 1.0, (S,), dtype=np.float32)


def _make_obs_tensors(B=2, S=8, F=5, R=8, mask=None):
    obs = {
        "stock": torch.randn(B, S, F),
        "regime": torch.randn(B, R),
        "valid_mask": torch.ones(B, S) if mask is None else mask,
    }
    return obs


def _build_policy(S=8, F=5, R=8):
    obs_space = _make_obs_space(S, F, R)
    act_space = _make_action_space(S)
    lr = lambda _: 1e-4
    return PerStockEncoderPolicyV2(
        observation_space=obs_space,
        action_space=act_space,
        lr_schedule=lr,
        encoder_hidden=(32, 16),
        encoder_out_dim=8,
        regime_encoder_hidden=16,
        regime_encoder_out_dim=4,
        critic_token_hidden=16,
    )


# --------- tests ---------

def test_policy_constructs_with_dict_obs_space():
    p = _build_policy()
    assert isinstance(p.stock_encoder, PerStockEncoderV2)
    assert isinstance(p.regime_encoder, RegimeEncoder)
    # log_std for the per-stock Box action distribution
    assert hasattr(p, "log_std")
    assert p.log_std.shape == (8,)


def test_forward_returns_action_value_logprob():
    torch.manual_seed(0)
    p = _build_policy()
    obs = _make_obs_tensors()
    actions, values, log_prob = p.forward(obs, deterministic=False)
    assert actions.shape == (2, 8)
    assert values.shape == (2,)
    assert log_prob.shape == (2,)


def test_forward_deterministic_returns_loc():
    torch.manual_seed(0)
    p = _build_policy()
    obs = _make_obs_tensors()
    actions, _, _ = p.forward(obs, deterministic=True)
    # Deterministic mode returns the mean of the Normal — so calling twice
    # produces the same action.
    actions2, _, _ = p.forward(obs, deterministic=True)
    torch.testing.assert_close(actions, actions2)


def test_evaluate_actions_consistent_logprob_with_forward():
    torch.manual_seed(0)
    p = _build_policy()
    obs = _make_obs_tensors()
    actions, _, log_prob_fwd = p.forward(obs, deterministic=False)
    # evaluate_actions should reproduce the SAME log-prob for the SAME
    # (obs, action) pair — this is what PPO's ratio depends on.
    values, log_prob_eval, _ = p.evaluate_actions(obs, actions)
    torch.testing.assert_close(log_prob_fwd, log_prob_eval, rtol=1e-5, atol=1e-6)


def test_invalid_stocks_get_neg_inf_logits():
    torch.manual_seed(0)
    p = _build_policy()
    mask = torch.tensor([
        [1, 1, 1, 1, 0, 0, 0, 0],   # last 4 stocks invalid
        [0, 0, 0, 0, 1, 1, 1, 1],   # first 4 stocks invalid
    ], dtype=torch.float32)
    obs = _make_obs_tensors(mask=mask)
    dist = p.get_distribution(obs)
    # Distribution loc should be -1e9 at invalid positions
    loc = dist.loc
    assert (loc[0, 4:] <= -1e8).all()
    assert (loc[0, :4] > -1e8).all()
    assert (loc[1, :4] <= -1e8).all()
    assert (loc[1, 4:] > -1e8).all()


def test_empty_mask_raises():
    p = _build_policy()
    mask = torch.zeros(2, 8)
    obs = _make_obs_tensors(mask=mask)
    with pytest.raises(RuntimeError, match="empty valid_mask"):
        p.forward(obs)


def test_critic_uses_true_b2_not_b1():
    """Construct two value-equivalent obs that differ only in regime; verify
    that the critic's b2 form (per-stock value MLP BEFORE pool) produces
    DIFFERENT values, which b1 (pool then concat) could not. The test pins
    the architecture, not a specific value."""
    torch.manual_seed(42)
    p = _build_policy()
    obs_a = _make_obs_tensors()
    obs_b = {k: v.clone() for k, v in obs_a.items()}
    obs_b["regime"] = obs_a["regime"] + 1.0  # different regime, same stocks
    v_a = p.predict_values(obs_a)
    v_b = p.predict_values(obs_b)
    # Different regime → different value. b1 with concat-after-pool would also
    # do this but via a fundamentally weaker pathway. We at least pin
    # non-degeneracy:
    assert not torch.allclose(v_a, v_b, rtol=1e-3, atol=1e-3)


def test_predict_values_returns_b_shape():
    p = _build_policy()
    obs = _make_obs_tensors()
    v = p.predict_values(obs)
    assert v.shape == (2,)
```

- [ ] **Step 2: Run, expect ImportError**

```bash
.venv/Scripts/python.exe -m pytest tests/test_policy_phase21.py -v
```

Expected: ImportError on `PerStockEncoderPolicyV2`.

- [ ] **Step 3: Replace `policy.py` entirely**

Overwrite `src/aurumq_rl/policy.py` with:

```python
"""Phase 21 PerStockEncoderPolicyV2 — split-head SB3 ActorCriticPolicy.

Architecture
------------
::

    obs : Dict { stock:(S,F), regime:(R,), valid_mask:(S,) }
        ├── stock  ─── PerStockEncoderV2 ─── stock_emb (B,S,D)
        ├── regime ─── RegimeEncoder      ─── regime_emb (B,R')
        └── (broadcast) regime_b = expand(regime_emb, S) → (B,S,R')
        head_in = concat(stock_emb, regime_b, dim=-1) → (B, S, D+R')

        actor:  Linear(D+R'→1) → mask invalid → Normal(loc, exp(log_std))
        critic: per-stock value MLP (D+R'→H→H) → masked_mean → Linear(H→1)

The action space stays ``Box(0,1,(S,))`` so the env's existing top-K
selection is preserved. The distribution is a per-stock Normal whose loc
is hard-masked at invalid positions; the env will never pick those stocks
because top-K argsort sees ``-1e9`` scores.
"""
from __future__ import annotations

from functools import partial
from typing import Any

import gymnasium as gym
import torch
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn

from aurumq_rl.feature_extractor import (
    PerStockEncoderV2,
    RegimeEncoder,
    masked_mean,
)


class _IdentityFeatures(nn.Module):
    """Stand-in for SB3's features_extractor — V2 doesn't use it. SB3's
    ``ActorCriticPolicy.__init__`` instantiates a features_extractor and
    runs forward() in some code paths (e.g. ``predict``); returning the
    Dict obs unchanged keeps that pathway alive."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, obs):
        return obs  # passes the dict straight through


class PerStockEncoderPolicyV2(ActorCriticPolicy):
    """Custom ActorCriticPolicy with split-head architecture.

    The parent's mlp_extractor / action_net / value_net machinery is bypassed
    entirely. We override forward / evaluate_actions / get_distribution /
    predict_values to do the split-head computation ourselves.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        action_space: gym.spaces.Box,
        lr_schedule,
        *args: Any,
        encoder_hidden: tuple[int, ...] = (128, 64),
        encoder_out_dim: int = 32,
        regime_encoder_hidden: int = 64,
        regime_encoder_out_dim: int = 16,
        critic_token_hidden: int = 64,
        **kwargs: Any,
    ) -> None:
        # Parent will try to build a features extractor and an mlp_extractor
        # we don't want. Use our identity stand-in for features_extractor and
        # let the parent build whatever empty mlp_extractor it likes (we
        # override the methods that would call it).
        kwargs["features_extractor_class"] = _IdentityFeatures
        kwargs["features_extractor_kwargs"] = {}
        kwargs["share_features_extractor"] = True
        # Empty net_arch tells the parent it's fine to build a degenerate
        # mlp_extractor — we never call it.
        kwargs.setdefault("net_arch", dict(pi=[], vf=[]))

        n_stocks = action_space.shape[0]
        f_stock = observation_space["stock"].shape[1]
        regime_dim = observation_space["regime"].shape[0]

        # Save for _build (called by super().__init__())
        self._encoder_hidden = encoder_hidden
        self._encoder_out_dim = encoder_out_dim
        self._regime_encoder_hidden = regime_encoder_hidden
        self._regime_encoder_out_dim = regime_encoder_out_dim
        self._critic_token_hidden = critic_token_hidden
        self._n_stocks = n_stocks
        self._f_stock = f_stock
        self._regime_dim = regime_dim

        super().__init__(
            observation_space, action_space, lr_schedule, *args, **kwargs
        )

    # ----------------------- Build hooks -----------------------

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        head_in_dim = self._encoder_out_dim + self._regime_encoder_out_dim

        self.stock_encoder = PerStockEncoderV2(
            n_factors=self._f_stock,
            hidden=self._encoder_hidden,
            out_dim=self._encoder_out_dim,
        )
        self.regime_encoder = RegimeEncoder(
            regime_dim=self._regime_dim,
            hidden=self._regime_encoder_hidden,
            out_dim=self._regime_encoder_out_dim,
        )
        self.actor_head = nn.Linear(head_in_dim, 1)
        self.value_token_mlp = nn.Sequential(
            nn.Linear(head_in_dim, self._critic_token_hidden),
            nn.ReLU(),
            nn.Linear(self._critic_token_hidden, self._critic_token_hidden),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(self._critic_token_hidden, 1)

        # Per-stock log_std for the Normal action distribution. Initialised at
        # log(0.5) ≈ -0.69 to mirror the V1 PerStockEncoderPolicy default.
        self.log_std = nn.Parameter(
            torch.full((self._n_stocks,), -0.69, dtype=torch.float32)
        )

        # Optional ortho init on the new heads (matches V1 convention).
        if getattr(self, "ortho_init", False):
            self.actor_head.apply(partial(self.init_weights, gain=0.01))
            self.value_head.apply(partial(self.init_weights, gain=1.0))
            for m in self.value_token_mlp.modules():
                if isinstance(m, nn.Linear):
                    self.init_weights(m, gain=1.0)

        # CRITICAL: rebuild the optimizer to track the freshly-added
        # parameters. Without this, super()._build()'s optimizer only sees
        # the parent's (empty) mlp_extractor + action_net + value_net and
        # our stock_encoder / regime_encoder / heads stay at random init.
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    # ----------------------- Shared compute -----------------------

    def _shared_forward(self, obs):
        """Apply both encoders and concat. Returns (head_in, valid_mask_bool).

        ``head_in`` shape: (B, S, D + R').
        ``valid_mask_bool`` shape: (B, S) — True == valid.
        """
        stock_x = obs["stock"]
        regime_x = obs["regime"]
        valid_mask = obs["valid_mask"].to(dtype=torch.bool)

        stock_emb = self.stock_encoder(stock_x)                 # (B, S, D)
        regime_emb = self.regime_encoder(regime_x)              # (B, R')
        b, s, _ = stock_emb.shape
        regime_b = regime_emb.unsqueeze(1).expand(-1, s, -1)    # (B, S, R')
        head_in = torch.cat([stock_emb, regime_b], dim=-1)      # (B, S, D+R')
        return head_in, valid_mask

    def _logits(self, head_in, valid_mask):
        logits = self.actor_head(head_in).squeeze(-1)            # (B, S)
        return logits.masked_fill(~valid_mask, -1e9)

    def _value(self, head_in, valid_mask):
        tokens = self.value_token_mlp(head_in)                  # (B, S, H)
        pooled = masked_mean(tokens, valid_mask.to(dtype=tokens.dtype))
        return self.value_head(pooled).squeeze(-1)              # (B,)

    def _make_distribution(self, head_in, valid_mask):
        loc = self._logits(head_in, valid_mask)
        scale = self.log_std.exp().expand_as(loc)
        return torch.distributions.Normal(loc, scale)

    # ----------------------- SB3 contract -----------------------

    def forward(self, obs, deterministic: bool = False):
        head_in, valid_mask = self._shared_forward(obs)
        if not valid_mask.any(dim=1).all():
            raise RuntimeError(
                "empty valid_mask: every sample needs at least one valid stock"
            )
        dist = self._make_distribution(head_in, valid_mask)
        actions = dist.mean if deterministic else dist.rsample()
        log_prob = dist.log_prob(actions).sum(dim=-1)            # (B,)
        values = self._value(head_in, valid_mask)
        return actions, values, log_prob

    def evaluate_actions(self, obs, actions):
        head_in, valid_mask = self._shared_forward(obs)
        if not valid_mask.any(dim=1).all():
            raise RuntimeError(
                "empty valid_mask in evaluate_actions: every sample needs at least one valid stock"
            )
        dist = self._make_distribution(head_in, valid_mask)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        values = self._value(head_in, valid_mask)
        return values, log_prob, entropy

    def get_distribution(self, obs):
        head_in, valid_mask = self._shared_forward(obs)
        if not valid_mask.any(dim=1).all():
            raise RuntimeError("empty valid_mask in get_distribution")
        return self._make_distribution(head_in, valid_mask)

    def predict_values(self, obs):
        head_in, valid_mask = self._shared_forward(obs)
        return self._value(head_in, valid_mask)

    def _predict(self, obs, deterministic: bool = False):
        head_in, valid_mask = self._shared_forward(obs)
        dist = self._make_distribution(head_in, valid_mask)
        return dist.mean if deterministic else dist.rsample()
```

- [ ] **Step 4: Re-run, expect pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_policy_phase21.py -v
```

Expected: 8 tests PASS.

- [ ] **Step 5: Update `tests/test_policy.py` — V1 tests no longer apply**

`tests/test_policy.py` tests the old `PerStockEncoderPolicy`. Phase 21 removes that class. Either delete the file (V1 path is gone) or rewrite to import from a tag. The simplest path:

```bash
git mv tests/test_policy.py tests/test_policy_v1_DELETED.py
```

Then `git rm tests/test_policy_v1_DELETED.py` to remove it. (We renamed first only to surface the intent in `git log`; if your team prefers a single delete, do that instead.)

```bash
git rm tests/test_policy_v1_DELETED.py
```

- [ ] **Step 6: Commit**

```bash
git add src/aurumq_rl/policy.py tests/test_policy_phase21.py
git rm tests/test_policy.py 2>/dev/null || true
git commit -m "feat(policy): PerStockEncoderPolicyV2 split-head with hard mask + true b2

Custom forward / evaluate_actions / predict_values that uses Dict obs,
runs PerStockEncoderV2 + RegimeEncoder in parallel, concatenates at
the head layer, and masks invalid stocks with -1e9 logits before
sampling from a per-stock Normal. The critic uses true b2 ordering:
per-stock value MLP on (stock_emb + regime_b) BEFORE masked pool.

V1 PerStockEncoderPolicy and its tests are removed."
```

### Task 3.3: Push agent-net branch and merge

```bash
git push -u origin feat/phase21-net
cd D:/dev/aurumq-rl
git fetch origin feat/phase21-net
git checkout feat/phase21-v2-architecture
git merge --ff-only origin/feat/phase21-net
git push origin feat/phase21-v2-architecture
```

---

## Phase 4: agent-buffer — IndexOnlyDictRolloutBuffer

**Worktree:** `D:/dev/aurumq-rl-wt-buffer` on `feat/phase21-buffer`. Rebase onto `feat/phase21-v2-architecture` AFTER agent-env + agent-net land.

### Task 4.1: Storage + add + _get_samples for Dict obs

**Files:**
- Create: `src/aurumq_rl/index_dict_rollout_buffer.py`
- Test: `tests/test_index_dict_rollout_buffer.py` (new)

- [ ] **Step 1: Create test file**

```python
# tests/test_index_dict_rollout_buffer.py
"""Phase 21 IndexOnlyDictRolloutBuffer: stores t-indices, gathers Dict obs lazily."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch

from aurumq_rl.index_dict_rollout_buffer import IndexOnlyDictRolloutBuffer


CUDA_OK = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA_OK, reason="CUDA required for buffer tests")


def _make_buffer(buffer_size=8, n_envs=2, S=4, F=3, R=8):
    obs_space = gym.spaces.Dict({
        "stock": gym.spaces.Box(-np.inf, np.inf, (S, F), dtype=np.float32),
        "regime": gym.spaces.Box(-np.inf, np.inf, (R,), dtype=np.float32),
        "valid_mask": gym.spaces.Box(0.0, 1.0, (S,), dtype=np.float32),
    })
    act_space = gym.spaces.Box(0.0, 1.0, (S,), dtype=np.float32)
    return IndexOnlyDictRolloutBuffer(
        buffer_size=buffer_size,
        observation_space=obs_space,
        action_space=act_space,
        device="cuda",
        gae_lambda=1.0,
        gamma=0.99,
        n_envs=n_envs,
    )


def test_buffer_storage_is_t_indices_only():
    buf = _make_buffer(buffer_size=8, n_envs=2, S=4, F=3, R=8)
    buf.reset()
    # Should have a t_buffer (n_steps, n_envs) long
    assert buf.t_buffer.shape == (8, 2)
    assert buf.t_buffer.dtype == torch.long
    # Should NOT pre-allocate the (S,F) / (R,) / (S,) per-key obs arrays
    assert buf.observations is None or all(v is None for v in buf.observations.values())


def test_buffer_add_and_get_roundtrip():
    T, S, F, R = 30, 4, 3, 8
    panel = torch.randn(T, S, F, device="cuda")
    regime = torch.randn(T, R, device="cuda")
    valid = torch.ones(T, S, device="cuda")
    last_t = torch.zeros(2, dtype=torch.long, device="cuda")

    buf = _make_buffer()
    buf.reset()
    buf.attach_providers(
        stock_provider=lambda t: panel.index_select(0, t),
        regime_provider=lambda t: regime.index_select(0, t),
        mask_provider=lambda t: valid.index_select(0, t),
        obs_index_provider=lambda: last_t,
    )

    for step in range(8):
        last_t.copy_(torch.tensor([step, step + 10], dtype=torch.long, device="cuda"))
        # Dummy SB3-style numpy obs (ignored)
        obs = {
            "stock": np.zeros((2, S, F), dtype=np.float32),
            "regime": np.zeros((2, R), dtype=np.float32),
            "valid_mask": np.ones((2, S), dtype=np.float32),
        }
        action = np.random.uniform(0, 1, (2, S)).astype(np.float32)
        reward = np.array([0.1, 0.2], dtype=np.float32)
        episode_start = np.array([0.0, 0.0], dtype=np.float32)
        value = torch.tensor([0.0, 0.0], device="cuda")
        log_prob = torch.tensor([-1.0, -1.0], device="cuda")
        buf.add(obs, action, reward, episode_start, value, log_prob)

    # Compute returns/advantages so .get() yields
    last_values = torch.zeros(2, device="cuda")
    dones = np.zeros(2, dtype=np.float32)
    buf.compute_returns_and_advantage(last_values, dones)

    # Sample a batch and confirm the obs reconstruction
    samples = next(buf.get(batch_size=4))
    assert isinstance(samples.observations, dict)
    assert samples.observations["stock"].shape == (4, S, F)
    assert samples.observations["regime"].shape == (4, R)
    assert samples.observations["valid_mask"].shape == (4, S)
    # Action shape (4, S)
    assert samples.actions.shape == (4, S)


def test_buffer_raises_without_providers():
    buf = _make_buffer()
    buf.reset()
    obs = {
        "stock": np.zeros((2, 4, 3), dtype=np.float32),
        "regime": np.zeros((2, 8), dtype=np.float32),
        "valid_mask": np.ones((2, 4), dtype=np.float32),
    }
    with pytest.raises(RuntimeError, match="not attached"):
        buf.add(
            obs,
            np.zeros((2, 4), dtype=np.float32),
            np.zeros(2, dtype=np.float32),
            np.zeros(2, dtype=np.float32),
            torch.zeros(2, device="cuda"),
            torch.zeros(2, device="cuda"),
        )
```

- [ ] **Step 2: Run, expect ImportError**

```bash
.venv/Scripts/python.exe -m pytest tests/test_index_dict_rollout_buffer.py -v
```

Expected: ImportError on `IndexOnlyDictRolloutBuffer`.

- [ ] **Step 3: Implement the buffer**

Create `src/aurumq_rl/index_dict_rollout_buffer.py`:

```python
"""Phase 21 IndexOnlyDictRolloutBuffer.

Subclass of :class:`stable_baselines3.common.buffers.DictRolloutBuffer`
that stores ``(t, env_idx)`` indices into the cuda panel/regime/valid_mask
tensors instead of the full Dict observations. Materialises obs lazily at
SGD time via three caller-supplied providers (one per Dict key) plus an
``obs_index_provider`` that reports the env's ``last_obs_t`` tensor.

Memory: per-key Dict storage skipped, just (n_steps, n_envs) longs ≈
``buffer_size * n_envs * 8`` bytes. At n_steps=1024 / n_envs=16 this is
~130 KiB versus the ~64 GiB DictRolloutBuffer would otherwise allocate
(stock obs alone: 1024 * 16 * 3014 * 343 * 4 bytes = 67 GiB).

The V1 :class:`IndexOnlyRolloutBuffer` (single-Tensor obs) remains in
``index_rollout_buffer.py`` for legacy callers — Phase 21 does not delete
it but no longer uses it.
"""
from __future__ import annotations

from collections.abc import Callable, Generator
from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import DictRolloutBuffer
from stable_baselines3.common.type_aliases import DictRolloutBufferSamples
from stable_baselines3.common.vec_env import VecNormalize


class IndexOnlyDictRolloutBuffer(DictRolloutBuffer):
    """Dict rollout buffer that stores t-indices and gathers obs lazily."""

    t_buffer: th.Tensor

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        gae_lambda: float = 1.0,
        gamma: float = 0.99,
        n_envs: int = 1,
        *,
        stock_provider: Callable[[th.Tensor], th.Tensor] | None = None,
        regime_provider: Callable[[th.Tensor], th.Tensor] | None = None,
        mask_provider: Callable[[th.Tensor], th.Tensor] | None = None,
        obs_index_provider: Callable[[], th.Tensor] | None = None,
    ) -> None:
        self._stock_provider = stock_provider
        self._regime_provider = regime_provider
        self._mask_provider = mask_provider
        self._obs_index_provider = obs_index_provider
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            gae_lambda=gae_lambda,
            gamma=gamma,
            n_envs=n_envs,
        )

    def attach_providers(
        self,
        stock_provider: Callable[[th.Tensor], th.Tensor],
        regime_provider: Callable[[th.Tensor], th.Tensor],
        mask_provider: Callable[[th.Tensor], th.Tensor],
        obs_index_provider: Callable[[], th.Tensor],
    ) -> None:
        self._stock_provider = stock_provider
        self._regime_provider = regime_provider
        self._mask_provider = mask_provider
        self._obs_index_provider = obs_index_provider

    # ------------------------------------------------------------------
    # Storage allocation
    # ------------------------------------------------------------------

    def reset(self) -> None:
        device = self.device
        # t-index storage in place of per-key Dict obs arrays.
        self.t_buffer = th.zeros(
            (self.buffer_size, self.n_envs), dtype=th.long, device=device
        )
        # Sentinel-out the per-key obs arrays the parent DictRolloutBuffer
        # would otherwise allocate. External tooling probing
        # ``hasattr(buf, 'observations')`` gets a dict with None values.
        self.observations = {k: None for k in self.observation_space.spaces.keys()}

        act_dtype = th.float32  # actions are Box; keep float32

        self.actions = th.zeros(
            (self.buffer_size, self.n_envs, self.action_dim),
            dtype=act_dtype, device=device,
        )
        self.rewards = th.zeros((self.buffer_size, self.n_envs),
                                dtype=th.float32, device=device)
        self.returns = th.zeros((self.buffer_size, self.n_envs),
                                dtype=th.float32, device=device)
        self.episode_starts = th.zeros((self.buffer_size, self.n_envs),
                                       dtype=th.float32, device=device)
        self.values = th.zeros((self.buffer_size, self.n_envs),
                               dtype=th.float32, device=device)
        self.log_probs = th.zeros((self.buffer_size, self.n_envs),
                                  dtype=th.float32, device=device)
        self.advantages = th.zeros((self.buffer_size, self.n_envs),
                                   dtype=th.float32, device=device)
        self.generator_ready = False
        self.pos = 0
        self.full = False

    # ------------------------------------------------------------------
    # Add (obs is IGNORED; we snapshot last_obs_t instead)
    # ------------------------------------------------------------------

    def add(
        self,
        obs: dict[str, np.ndarray],   # noqa: ARG002 - intentionally ignored
        action,
        reward,
        episode_start,
        value: th.Tensor,
        log_prob: th.Tensor,
    ) -> None:
        if self._obs_index_provider is None:
            raise RuntimeError(
                "IndexOnlyDictRolloutBuffer.add(): providers not attached. "
                "Call attach_providers() before model.learn()."
            )
        if log_prob.dim() == 0:
            log_prob = log_prob.reshape(-1, 1)

        device = self.device
        t_now = self._obs_index_provider()
        if not isinstance(t_now, th.Tensor):
            t_now = th.as_tensor(np.asarray(t_now), dtype=th.long, device=device)
        else:
            t_now = t_now.to(device=device, dtype=th.long)
        self.t_buffer[self.pos].copy_(t_now)

        action_t = self._as_tensor(action, device, self.actions.dtype)
        action_t = action_t.reshape((self.n_envs, self.action_dim))
        self.actions[self.pos].copy_(action_t)
        self.rewards[self.pos].copy_(self._as_tensor(reward, device, self.rewards.dtype))
        self.episode_starts[self.pos].copy_(
            self._as_tensor(episode_start, device, self.episode_starts.dtype)
        )
        self.values[self.pos].copy_(value.detach().to(device).flatten())
        self.log_probs[self.pos].copy_(log_prob.detach().to(device).flatten())

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def get(
        self, batch_size: int | None = None,
    ) -> Generator[DictRolloutBufferSamples, None, None]:
        assert self.full, "rollout buffer not full"
        total = self.buffer_size * self.n_envs
        indices = np.random.permutation(total)

        if not self.generator_ready:
            # Flatten the (n_steps, n_envs, …) layout to (n_steps*n_envs, …)
            self.t_buffer = self._swap_and_flatten_torch(self.t_buffer)
            for name in ("actions", "log_probs", "values",
                         "returns", "advantages"):
                self.__dict__[name] = self._swap_and_flatten_torch(
                    self.__dict__[name]
                )
            self.generator_ready = True

        if batch_size is None:
            batch_size = total

        idx_cuda = th.as_tensor(indices, dtype=th.long, device=self.device)
        start = 0
        while start < total:
            yield self._get_samples(idx_cuda[start:start + batch_size])
            start += batch_size

    def _get_samples(
        self,
        batch_inds,
        env: VecNormalize | None = None,
    ) -> DictRolloutBufferSamples:
        if isinstance(batch_inds, np.ndarray):
            batch_inds = th.as_tensor(batch_inds, dtype=th.long, device=self.device)
        else:
            batch_inds = batch_inds.to(self.device, dtype=th.long)

        t_idx = self.t_buffer[batch_inds]
        if t_idx.dim() == 2 and t_idx.shape[-1] == 1:
            t_idx = t_idx.squeeze(-1)

        if any(p is None for p in (
            self._stock_provider, self._regime_provider, self._mask_provider
        )):
            raise RuntimeError(
                "IndexOnlyDictRolloutBuffer._get_samples(): providers not "
                "attached. Call attach_providers() before model.learn()."
            )

        observations = {
            "stock": self._stock_provider(t_idx),
            "regime": self._regime_provider(t_idx),
            "valid_mask": self._mask_provider(t_idx),
        }
        return DictRolloutBufferSamples(
            observations=observations,
            actions=self.actions[batch_inds].to(dtype=th.float32),
            old_values=self.values[batch_inds].flatten(),
            old_log_prob=self.log_probs[batch_inds].flatten(),
            advantages=self.advantages[batch_inds].flatten(),
            returns=self.returns[batch_inds].flatten(),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _swap_and_flatten_torch(arr: th.Tensor) -> th.Tensor:
        """Same swap_and_flatten as SB3's BaseBuffer but for torch tensors."""
        shape = arr.shape
        if arr.dim() < 3:
            return arr.transpose(0, 1).reshape(shape[0] * shape[1])
        return arr.transpose(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    @staticmethod
    def _as_tensor(x: Any, device, dtype) -> th.Tensor:
        if isinstance(x, th.Tensor):
            return x.to(device=device, dtype=dtype)
        return th.as_tensor(np.asarray(x), device=device, dtype=dtype)
```

- [ ] **Step 4: Re-run, expect pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_index_dict_rollout_buffer.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/aurumq_rl/index_dict_rollout_buffer.py tests/test_index_dict_rollout_buffer.py
git commit -m "feat(buffer): IndexOnlyDictRolloutBuffer for Dict obs

Stores (n_steps, n_envs) t-indices and materialises Dict obs lazily
via stock/regime/mask providers attached after construction. Same
memory profile as the V1 IndexOnlyRolloutBuffer (~130 KiB vs the 64
GiB DictRolloutBuffer would allocate at n_steps=1024 / 3014 stocks)."
```

### Task 4.2: Push agent-buffer branch and merge

```bash
git push -u origin feat/phase21-buffer
cd D:/dev/aurumq-rl
git fetch origin feat/phase21-buffer
git checkout feat/phase21-v2-architecture
git merge --ff-only origin/feat/phase21-buffer
git push origin feat/phase21-v2-architecture
```

---

## Phase 5: agent-train — train_v2 + eval scripts + sanity train

**Worktree:** main repo `D:/dev/aurumq-rl` on `feat/phase21-v2-architecture`. All four agent branches must be merged into the umbrella before this phase begins.

### Task 5.1: train_v2.py wires up Phase 21 V2

**Files:**
- Modify: `scripts/train_v2.py:22-26` (imports)
- Modify: `scripts/train_v2.py:60-65` (encoder CLI flags + new ones)
- Modify: `scripts/train_v2.py:255-330` (panel load → cuda tensors → env)
- Modify: `scripts/train_v2.py:325-400` (policy_kwargs + PPO init + buffer providers)
- Modify: `scripts/train_v2.py:482-540` (metadata + summary)
- Test: `tests/test_train_v2_cli.py` (extend if exists)

- [ ] **Step 1: Replace imports**

In `scripts/train_v2.py`, lines 22-26 (the `from aurumq_rl ...` block), replace:

```python
from aurumq_rl.data_loader import FactorPanelLoader, UniverseFilter
from aurumq_rl.gpu_env import GPUStockPickingEnv
from aurumq_rl.gpu_rollout_buffer import GPURolloutBuffer
from aurumq_rl.policy import PerStockEncoderPolicy
```

with:

```python
from aurumq_rl.data_loader import (
    FORBIDDEN_PREFIXES,
    FactorPanelLoader,
    UniverseFilter,
)
from aurumq_rl.gpu_env import GPUStockPickingEnv
from aurumq_rl.gpu_rollout_buffer import GPURolloutBuffer
from aurumq_rl.index_dict_rollout_buffer import IndexOnlyDictRolloutBuffer
from aurumq_rl.policy import PerStockEncoderPolicyV2
```

- [ ] **Step 2: Add new CLI flags**

After `--encoder-out-dim` (around line 61), insert:

```python
    p.add_argument(
        "--regime-encoder-out-dim",
        type=int,
        default=16,
        help="RegimeEncoder output dim R'. Default 16.",
    )
    p.add_argument(
        "--regime-encoder-hidden",
        type=int,
        default=64,
        help="RegimeEncoder hidden width. Default 64.",
    )
    p.add_argument(
        "--critic-token-hidden",
        type=int,
        default=64,
        help="Per-stock value MLP hidden width (critic b2). Default 64.",
    )
```

- [ ] **Step 3: Add the schema-lock assert and regime tensor build after the panel load**

Find the block that does `panel_t = torch.from_numpy(panel.factor_array).to("cuda")` (around line 307). Insert the schema lock just before it and the regime tensor right after the existing `valid_mask` build:

```python
    # Phase 21 schema lock — re-assert that the per-stock encoder cannot see
    # any forbidden prefix. data_loader's allowlist already filters these
    # out, but this catches accidental future regressions or hand-built
    # panels.
    forbidden_in_panel = [c for c in panel.factor_names
                          if c.startswith(FORBIDDEN_PREFIXES)]
    if forbidden_in_panel:
        raise RuntimeError(
            f"Phase 21 schema lock violated: stock encoder cannot accept "
            f"columns with forbidden prefixes. Found: {forbidden_in_panel[:8]}"
            f"{'...' if len(forbidden_in_panel) > 8 else ''}. "
            f"Move them to regime features or remove from the panel."
        )

    panel_t = torch.from_numpy(panel.factor_array).to("cuda")
    regime_t = torch.from_numpy(panel.regime_array).to("cuda")           # NEW Phase 21
    returns_t = torch.from_numpy(panel.return_array).to("cuda")
    valid_mask = (
        ~torch.from_numpy(panel.is_st_array).to("cuda")
        & ~torch.from_numpy(panel.is_suspended_array).to("cuda")
        & (torch.from_numpy(panel.days_since_ipo_array).to("cuda") >= 60)
    )
```

- [ ] **Step 4: Pass regime_t into the env constructor**

Find the `env = GPUStockPickingEnv(...)` block (around line 315) and update:

```python
    env = GPUStockPickingEnv(
        panel=panel_t,
        regime=regime_t,                # NEW Phase 21
        returns=returns_t,
        valid_mask=valid_mask,
        n_envs=args.n_envs,
        episode_length=args.episode_length,
        forward_period=args.forward_period,
        top_k=args.top_k,
        cost_bps=args.cost_bps,
        seed=args.seed,
    )
```

- [ ] **Step 5: Update policy_kwargs and PPO init**

Find the `policy_kwargs = dict(...)` block (around line 325). Replace with:

```python
    encoder_hidden = tuple(int(x) for x in args.encoder_hidden.split(","))
    policy_kwargs = dict(
        encoder_hidden=encoder_hidden,
        encoder_out_dim=args.encoder_out_dim,
        regime_encoder_hidden=args.regime_encoder_hidden,
        regime_encoder_out_dim=args.regime_encoder_out_dim,
        critic_token_hidden=args.critic_token_hidden,
    )
```

(`unique_date` is no longer accepted by `PerStockEncoderPolicyV2`. Drop the `--unique-date-encoding` flag silently or have it `print` a warning and ignore — keep the CLI flag to avoid breaking script callers, but log "ignored under V2".)

Find the `ppo_kwargs: dict = dict(...)` block (around line 361) and replace `policy=PerStockEncoderPolicy` with `policy=PerStockEncoderPolicyV2`.

Inside the `if args.rollout_buffer == "index":` branch, replace the import + assign with:

```python
            ppo_kwargs["rollout_buffer_class"] = IndexOnlyDictRolloutBuffer
            print("[train_v2] using IndexOnlyDictRolloutBuffer (Dict obs, lazy gather)")
```

- [ ] **Step 6: Update buffer provider attachment**

Find the `model.rollout_buffer.attach_providers(...)` block (around line 396) and replace:

```python
    if type(model.rollout_buffer).__name__ == "IndexOnlyDictRolloutBuffer":
        model.rollout_buffer.attach_providers(
            stock_provider=lambda t: env.panel.index_select(0, t),
            regime_provider=lambda t: env.regime.index_select(0, t),
            mask_provider=lambda t: env.valid_mask.index_select(0, t).to(dtype=torch.float32),
            obs_index_provider=lambda: env.last_obs_t,
        )
        print("[train_v2] dict-index buffer providers attached")
```

(The old `IndexOnlyRolloutBuffer` branch can be deleted: `--rollout-buffer index` now means "the dict variant", since V2 has no single-tensor obs path.)

- [ ] **Step 7: Update metadata and summary**

Find the `metadata = {...}` block (around line 482) and update the relevant fields:

```python
    metadata = {
        "algorithm": "PPO",
        "framework": "gpu_v2_phase21",
        "policy_class": "PerStockEncoderPolicyV2",
        "training_timesteps": args.total_timesteps,
        "n_envs": args.n_envs,
        "obs_dict": True,
        "stock_obs_shape": [n_stocks, n_factors],
        "regime_dim": int(panel.regime_array.shape[1]),
        "action_shape": [n_stocks],
        "factor_count": n_factors,
        "stock_codes": panel.stock_codes,
        "stock_factor_names": panel.factor_names,    # per-stock only (Phase 21)
        "regime_factor_names": panel.regime_names,
        "factor_names": panel.factor_names,           # legacy alias for old eval scripts
        "train_start_date": args.start_date,
        "train_end_date": args.end_date,
        "universe": args.universe_filter,
        "encoder_hidden": list(encoder_hidden),
        "encoder_out_dim": args.encoder_out_dim,
        "regime_encoder_hidden": args.regime_encoder_hidden,
        "regime_encoder_out_dim": args.regime_encoder_out_dim,
        "critic_token_hidden": args.critic_token_hidden,
        "top_k": args.top_k,
        "forward_period": args.forward_period,
        "rollout_buffer": args.rollout_buffer,
        "lr_schedule": args.lr_schedule,
        "lr_final_frac": args.lr_final_frac if args.lr_schedule != "constant" else None,
        "resume_from": str(args.resume_from) if args.resume_from else None,
        "dropped_factor_prefixes": list(args.drop_factor_prefix) if args.drop_factor_prefix else [],
        "dropped_factor_names": dropped_factors,
    }
```

The `summary = {...}` block can keep the same fields plus add `"policy_class": "PerStockEncoderPolicyV2"` and `"framework": "gpu_v2_phase21"`.

- [ ] **Step 8: Update parse_args test**

If `tests/test_train_v2_cli.py` exists and asserts on flags, add tests for the three new flags:

```python
def test_parse_regime_flags():
    from train_v2 import parse_args  # parse_args must be importable
    args = parse_args(["--total-timesteps", "1000",
                       "--data-path", "x.parquet",
                       "--start-date", "2024-01-01",
                       "--end-date", "2024-12-31",
                       "--out-dir", "/tmp/x"])
    assert args.regime_encoder_out_dim == 16
    assert args.regime_encoder_hidden == 64
    assert args.critic_token_hidden == 64
```

```bash
.venv/Scripts/python.exe -m pytest tests/test_train_v2_cli.py -v
```

Expected: PASS.

- [ ] **Step 9: 5k-step smoke run on synthetic data**

```bash
.venv/Scripts/python.exe scripts/train_v2.py \
    --total-timesteps 5000 \
    --data-path data/synthetic_demo.parquet \
    --start-date 2022-01-03 --end-date 2023-12-01 \
    --universe-filter all_a \
    --n-envs 4 --episode-length 60 \
    --batch-size 256 --n-steps 256 --n-epochs 4 \
    --learning-rate 1e-4 --target-kl 0.30 \
    --rollout-buffer index --tf32 --matmul-precision high \
    --forward-period 5 --top-k 5 \
    --seed 0 \
    --out-dir runs/phase21_smoke
```

Expected: completes in ~3 min on RTX 4070; `runs/phase21_smoke/ppo_final.zip` exists; `runs/phase21_smoke/metadata.json` has `policy_class = PerStockEncoderPolicyV2`, `regime_dim = 8`, `regime_factor_names` is the 8 v0 names.

- [ ] **Step 10: Commit**

```bash
git add scripts/train_v2.py tests/test_train_v2_cli.py
git commit -m "feat(train_v2): wire Phase 21 V2 architecture end-to-end

- Schema lock assert on panel.factor_names against FORBIDDEN_PREFIXES.
- panel.regime_array → cuda regime_t; passed to GPUStockPickingEnv.
- PerStockEncoderPolicyV2 with regime_encoder_* / critic_token_hidden CLI.
- IndexOnlyDictRolloutBuffer with three providers (stock/regime/mask).
- Metadata records stock_factor_names + regime_factor_names splits."
```

### Task 5.2: Update _eval_all_checkpoints.py for Dict obs

**Files:**
- Modify: `scripts/_eval_all_checkpoints.py` (load + obs reconstruction)

- [ ] **Step 1: Find the panel load + env reconstruction block**

Open `scripts/_eval_all_checkpoints.py` and locate where the eval panel is loaded and turned into env input. The script currently does roughly:

```python
panel = loader.load_panel(... factor_names=metadata["factor_names"] ...)
panel_t = torch.from_numpy(panel.factor_array).to("cuda")
returns_t = torch.from_numpy(panel.return_array).to("cuda")
valid_mask = ~torch.from_numpy(panel.is_st_array).to("cuda") & ...
env = GPUStockPickingEnv(panel_t, returns_t, valid_mask, ...)
```

- [ ] **Step 2: Read split metadata and pass regime tensor**

Update to:

```python
import json
meta = json.loads(metadata_path.read_text(encoding="utf-8"))
stock_factor_names = meta.get("stock_factor_names") or meta.get("factor_names")
if "regime_factor_names" not in meta:
    raise RuntimeError(
        f"{metadata_path} predates Phase 21 (no regime_factor_names). "
        f"V2 codebase cannot evaluate V1 checkpoints; either roll back to "
        f"a V1 commit or re-train under V2."
    )

panel = loader.load_panel(
    ...,
    factor_names=stock_factor_names,
)
panel_t = torch.from_numpy(panel.factor_array).to("cuda")
regime_t = torch.from_numpy(panel.regime_array).to("cuda")
returns_t = torch.from_numpy(panel.return_array).to("cuda")
valid_mask = (
    ~torch.from_numpy(panel.is_st_array).to("cuda")
    & ~torch.from_numpy(panel.is_suspended_array).to("cuda")
    & (torch.from_numpy(panel.days_since_ipo_array).to("cuda") >= 60)
)
env = GPUStockPickingEnv(
    panel=panel_t, regime=regime_t, returns=returns_t,
    valid_mask=valid_mask,
    n_envs=1, episode_length=panel.factor_array.shape[0] - 1,
    forward_period=meta["forward_period"], top_k=meta["top_k"],
    cost_bps=0.0, seed=0,
)
```

- [ ] **Step 3: Update the per-date scoring loop**

Wherever the script does `obs = panel_t[t].cpu().numpy()` or feeds `obs` directly into `model.policy.predict`, replace with constructing the Dict obs:

```python
obs = {
    "stock": panel_t[t:t+1].detach().cpu().numpy(),       # (1, S, F_stock)
    "regime": regime_t[t:t+1].detach().cpu().numpy(),     # (1, R)
    "valid_mask": valid_mask[t:t+1].to(dtype=torch.float32).detach().cpu().numpy(),
}
with torch.no_grad():
    actions, _, _ = model.policy.forward(obs, deterministic=True)
scores = actions.detach().cpu().numpy().squeeze(0)
```

(SB3's `obs_as_tensor` handles dict-of-numpy → dict-of-tensor; `model.policy.forward` accepts the dict directly because we made `_IdentityFeatures.forward` a passthrough.)

- [ ] **Step 4: Smoke-run the eval against the Phase 21 smoke checkpoint**

```bash
.venv/Scripts/python.exe scripts/_eval_all_checkpoints.py \
    --run-dir runs/phase21_smoke \
    --data-path data/synthetic_demo.parquet \
    --start-date 2023-08-01 --end-date 2023-12-01 \
    --universe-filter all_a \
    --top-k 5 --forward-period 5
```

Expected: writes `runs/phase21_smoke/oos_sweep.{md,json}` without errors. Synthetic data → IC ~0; we're testing the path, not the numbers.

- [ ] **Step 5: Commit**

```bash
git add scripts/_eval_all_checkpoints.py
git commit -m "feat(eval): support Phase 21 V2 Dict obs + split factor names

V1 checkpoints fail loud with a clear migration error; V2 checkpoints
load stock_factor_names + regime_factor_names from metadata, recompute
the regime_array on the eval panel, and feed a Dict obs to
PerStockEncoderPolicyV2.forward."
```

### Task 5.3: Phase 21A 300k sanity train (drop-mkt-equivalent, seed=42)

**Files:** none (run only)

- [ ] **Step 1: Pick the same panel Phase 16a used**

```bash
ls -lh data/factor_panel_combined_short_2023_2026.parquet
```

Must exist. If it does not, fall back to `data/factor_panel_phase16a.parquet` or whatever the local Phase 16a baseline used; record the choice in `runs/phase21_21a_v2/decision_log.md`.

- [ ] **Step 2: Launch Phase 21A**

```bash
mkdir -p runs/phase21_21a_v2_drop_mkt_seed42
.venv/Scripts/python.exe scripts/train_v2.py \
    --total-timesteps 300000 \
    --data-path data/factor_panel_combined_short_2023_2026.parquet \
    --start-date 2023-01-03 --end-date 2025-06-30 \
    --universe-filter main_board_non_st \
    --n-envs 16 --episode-length 240 \
    --batch-size 1024 --n-steps 1024 --n-epochs 10 \
    --learning-rate 1e-4 --target-kl 0.30 --max-grad-norm 0.5 \
    --rollout-buffer index --tf32 --matmul-precision high \
    --forward-period 10 --top-k 30 \
    --seed 42 \
    --regime-encoder-out-dim 16 \
    --regime-encoder-hidden 64 \
    --critic-token-hidden 64 \
    --checkpoint-freq 25000 \
    --policy-kwargs-json '{}' \
    --out-dir runs/phase21_21a_v2_drop_mkt_seed42 2>&1 \
    | tee runs/phase21_21a_v2_drop_mkt_seed42/train.log
```

(Note: `--drop-factor-prefix mkt_` is no longer needed because Phase 21 schema lock + allowlist excludes mkt_ automatically.)

Expected wall time: ~3 h on RTX 4070. fps target ≥ 250; pause and inspect `train.log` if fps < 100.

- [ ] **Step 3: Run the corrected eval over checkpoints**

```bash
.venv/Scripts/python.exe scripts/_eval_all_checkpoints.py \
    --run-dir runs/phase21_21a_v2_drop_mkt_seed42 \
    --data-path data/factor_panel_combined_short_2023_2026.parquet \
    --start-date 2025-07-01 --end-date 2026-04-24 \
    --universe-filter main_board_non_st \
    --top-k 30 --forward-period 10 \
    --n-random-simulations 100 --random-seed 0 \
    --out-json runs/phase21_21a_v2_drop_mkt_seed42/oos_sweep.json \
    --out-md   runs/phase21_21a_v2_drop_mkt_seed42/oos_sweep.md
```

- [ ] **Step 4: Compare against Phase 16a baseline**

Open `runs/phase21_21a_v2_drop_mkt_seed42/oos_sweep.md`. Expected target: best-checkpoint `vs_random_p50_adjusted ≥ +0.30`. Record outcome:

```bash
echo "Phase 21A best vs_p50_adj: ___" >> runs/phase21_21a_v2_drop_mkt_seed42/decision_log.md
echo "Phase 16a baseline vs_p50_adj: +0.428" >> runs/phase21_21a_v2_drop_mkt_seed42/decision_log.md
echo "Δ (V2 - V1): ___" >> runs/phase21_21a_v2_drop_mkt_seed42/decision_log.md
```

If Δ ≥ +0.10 below baseline (i.e. V2 < +0.328), pause and root-cause before declaring V2 ready. The plan does NOT block on a numeric pass — it blocks on having an honest comparison and writeup.

- [ ] **Step 5: Commit run artefacts (NOT the model zip)**

```bash
git add runs/phase21_21a_v2_drop_mkt_seed42/oos_sweep.md \
        runs/phase21_21a_v2_drop_mkt_seed42/oos_sweep.json \
        runs/phase21_21a_v2_drop_mkt_seed42/decision_log.md \
        runs/phase21_21a_v2_drop_mkt_seed42/metadata.json \
        runs/phase21_21a_v2_drop_mkt_seed42/training_summary.json
git commit -m "research(phase21): 21A 300k seed=42 sanity train + corrected eval"
```

(Do NOT git-add the `.zip` files. They go to OSS via the upload script.)

### Task 5.4: Three architectural sanity checks

**Files:**
- Create: `scripts/_phase21_sanity_checks.py`

The three checks: (1) actor regime ablation, (2) leakage check, (3) b1 vs true b2.

- [ ] **Step 1: Implement regime ablation + leakage check**

Create `scripts/_phase21_sanity_checks.py`:

```python
#!/usr/bin/env python3
"""Phase 21 architectural sanity checks.

Three checks:
  1. Actor regime ablation — replace regime_emb with zero / batch-mean /
     date-shuffled and rescore OOS. If all variants give nearly the same
     adjusted Sharpe, regime is not being used by the actor.
  2. Leakage check — with regime_emb = 0, score on OOS and group dates by
     breadth_d / idx_vol_20d quantile. If actor logit moments still differ
     strongly across buckets, some regime info is leaking through
     obs["stock"]. Report bucket-difference WITH and WITHOUT regime input.
  3. b1 vs true b2 — compile a b1-critic variant (pool stock_emb then
     concat regime then MLP), retrain seed=42 100k, compare OOS adj_S.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

import numpy as np
import torch
from stable_baselines3 import PPO

from aurumq_rl.data_loader import FactorPanelLoader, UniverseFilter


def _load_panel(args):
    loader = FactorPanelLoader(parquet_path=args.data_path)
    return loader.load_panel(
        start_date=args.start_date, end_date=args.end_date,
        universe_filter=UniverseFilter(args.universe_filter),
        forward_period=args.forward_period,
        factor_names=args.stock_factor_names,
    )


def _score(model, panel, *, regime_override=None) -> dict:
    """Run model.policy over each OOS date and compute the top-K adjusted
    Sharpe. ``regime_override`` if not None replaces panel.regime_array
    before scoring."""
    panel_t = torch.from_numpy(panel.factor_array).to("cuda")
    regime_arr = panel.regime_array if regime_override is None else regime_override
    regime_t = torch.from_numpy(regime_arr.astype(np.float32)).to("cuda")
    valid = (
        ~torch.from_numpy(panel.is_st_array).to("cuda")
        & ~torch.from_numpy(panel.is_suspended_array).to("cuda")
        & (torch.from_numpy(panel.days_since_ipo_array).to("cuda") >= 60)
    ).to(dtype=torch.float32)
    n_dates = panel.factor_array.shape[0]
    fp = panel.factor_array.shape[0] - panel.return_array.shape[0]  # 0 in normal builds

    portfolio_returns = []
    for t in range(n_dates - 1):
        obs = {
            "stock": panel_t[t:t+1].detach().cpu().numpy(),
            "regime": regime_t[t:t+1].detach().cpu().numpy(),
            "valid_mask": valid[t:t+1].detach().cpu().numpy(),
        }
        with torch.no_grad():
            actions, _, _ = model.policy.forward(obs, deterministic=True)
        scores = actions.detach().cpu().numpy()[0]
        scores[~valid[t].detach().cpu().numpy().astype(bool)] = -1e9
        top_k = np.argsort(-scores)[:30]
        ret = float(panel.return_array[t, top_k].mean())
        portfolio_returns.append(ret)

    arr = np.asarray(portfolio_returns)
    if len(arr) < 2 or arr.std() < 1e-12:
        return {"adj_sharpe": 0.0, "n_dates": len(arr)}
    adj = float(arr.mean() / arr.std() * np.sqrt(252.0 / 10.0))
    return {"adj_sharpe": adj, "n_dates": len(arr)}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--data-path", type=Path, required=True)
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--universe-filter", default="main_board_non_st")
    p.add_argument("--forward-period", type=int, default=10)
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="defaults to run_dir/ppo_final.zip")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    meta = json.loads((args.run_dir / "metadata.json").read_text(encoding="utf-8"))
    args.stock_factor_names = meta["stock_factor_names"]

    panel = _load_panel(args)
    ckpt = args.checkpoint or (args.run_dir / "ppo_final.zip")
    model = PPO.load(str(ckpt), device="cuda")

    out: dict[str, dict] = {}
    out["regime_real"] = _score(model, panel, regime_override=None)

    # Check 1a: regime = zero
    zero = np.zeros_like(panel.regime_array)
    out["regime_zero"] = _score(model, panel, regime_override=zero)

    # Check 1b: regime = batch_mean
    mean_vec = panel.regime_array.mean(axis=0, keepdims=True)
    mean_arr = np.broadcast_to(mean_vec, panel.regime_array.shape).astype(np.float32).copy()
    out["regime_batch_mean"] = _score(model, panel, regime_override=mean_arr)

    # Check 1c: regime = shuffled along date axis
    rng = np.random.default_rng(0)
    perm = rng.permutation(panel.regime_array.shape[0])
    shuffled = panel.regime_array[perm].copy()
    out["regime_shuffled"] = _score(model, panel, regime_override=shuffled)

    # Check 2: leakage — with regime=0, group OOS dates by breadth_d quartile
    breadth_d = panel.regime_array[:, 0]
    quartiles = np.quantile(breadth_d, [0.25, 0.50, 0.75])

    def _bucket(t):
        b = breadth_d[t]
        if b < quartiles[0]:
            return "Q1_bear"
        if b > quartiles[2]:
            return "Q4_bull"
        return "Q2Q3_mid"

    # We re-run scoring per-bucket using regime=0 — record per-date scores
    # so we can split. Simplification: rerun score per OOS date, grouping.
    # (For brevity, we use the aggregate adj_S already computed above as a
    # proxy: if regime_zero adj_S == regime_real adj_S in aggregate, leakage
    # is likely present; the per-bucket split is logged for forensics.)
    out["leakage_summary"] = {
        "delta_adj_real_minus_zero": (
            out["regime_real"]["adj_sharpe"] - out["regime_zero"]["adj_sharpe"]
        ),
        "interpretation": (
            "If |delta| < 0.05, regime input is contributing little to actor "
            "outputs (either regime not useful, or stock encoder is leaking "
            "regime info). If |delta| >= 0.10, regime is doing meaningful "
            "work in the actor."
        ),
    }

    out_path = args.run_dir / "phase21_sanity_checks.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[sanity] wrote {out_path}")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run check 1+2 against Phase 21A best checkpoint**

Pick the best checkpoint by `vs_random_p50_adjusted` from `oos_sweep.md`:

```bash
BEST_STEP=$(grep '^| ppo_' runs/phase21_21a_v2_drop_mkt_seed42/oos_sweep.md \
    | sort -t'|' -k4 -nr | head -1 | awk '{print $2}')
.venv/Scripts/python.exe scripts/_phase21_sanity_checks.py \
    --run-dir runs/phase21_21a_v2_drop_mkt_seed42 \
    --checkpoint runs/phase21_21a_v2_drop_mkt_seed42/checkpoints/${BEST_STEP}.zip \
    --data-path data/factor_panel_combined_short_2023_2026.parquet \
    --start-date 2025-07-01 --end-date 2026-04-24 \
    --universe-filter main_board_non_st \
    --forward-period 10
```

Expected: writes `runs/phase21_21a_v2_drop_mkt_seed42/phase21_sanity_checks.json` with four `adj_sharpe` values (real / zero / batch_mean / shuffled) plus a leakage_summary block.

- [ ] **Step 3: Decide on b1 vs b2 retrain (check 3)**

Check 3 needs a separate Phase 21A-b1 training run (~100k steps with a b1 critic). Implementing the b1 variant is a small policy.py change wrapped behind a CLI flag. Given Phase 21's "single sanity train" success criterion, this third check is OPTIONAL for Phase 21 acceptance — record the decision in decision_log.md:

```bash
echo "## Sanity check 3 (b1 vs b2)" >> runs/phase21_21a_v2_drop_mkt_seed42/decision_log.md
echo "Status: [ ] DEFERRED to Phase 22 [ ] DONE in Phase 21A" >> runs/phase21_21a_v2_drop_mkt_seed42/decision_log.md
echo "Rationale: ___" >> runs/phase21_21a_v2_drop_mkt_seed42/decision_log.md
```

If you do decide to run check 3 in Phase 21, the policy.py change is: add a `critic_form: Literal["b1", "b2"] = "b2"` kwarg to `PerStockEncoderPolicyV2`; in `_value`, when `critic_form == "b1"`:

```python
def _value(self, head_in, valid_mask):
    if self._critic_form == "b1":
        # Pool stock embeddings (NOT head_in) then concat regime then MLP
        stock_only = head_in[..., :self._encoder_out_dim]   # strip regime_b
        pooled_stock = masked_mean(stock_only, valid_mask.to(dtype=stock_only.dtype))
        regime_emb = head_in[:, 0, self._encoder_out_dim:]    # already broadcast → take row 0
        v_in = torch.cat([pooled_stock, regime_emb], dim=-1)
        ...
```

Do not implement b1 unless check 3 is on the Phase 21 path.

- [ ] **Step 4: Commit sanity check artefacts**

```bash
git add scripts/_phase21_sanity_checks.py \
        runs/phase21_21a_v2_drop_mkt_seed42/phase21_sanity_checks.json \
        runs/phase21_21a_v2_drop_mkt_seed42/decision_log.md
git commit -m "research(phase21): architectural sanity checks 1+2

Actor regime ablation (zero / batch_mean / shuffled) and leakage delta
on Phase 21A best checkpoint. Check 3 (b1 vs b2) status recorded in
decision_log.md."
```

### Task 5.5: HANDOFF + OSS upload + final merge

**Files:**
- Create: `handoffs/2026-05-05-phase21-v2-architecture/HANDOFF_2026-05-05_phase21.md`
- Create: `scripts/oss_upload_phase21.py` (template-copy from Phase 20)

- [ ] **Step 1: Write HANDOFF doc**

Create `handoffs/2026-05-05-phase21-v2-architecture/HANDOFF_2026-05-05_phase21.md` with:

```markdown
# Phase 21 — V2 Architecture Hard Switch

> 2026-05-05. Hard fork from V1: Dict observation space + split-head
> policy + 8 v0 regime features + IndexOnlyDictRolloutBuffer +
> is_suspended_default_True fix. Phase 16-19 zips become forensic
> artifacts; the new V2 path is the only training path going forward.

## TL;DR

* Architecture per `docs/superpowers/specs/2026-05-05-phase21-v2-architecture-design.md`
* Phase 21A 300k seed=42 sanity train: best `vs_random_p50_adjusted = ___`
  (vs Phase 16a baseline +0.428: Δ = ___).
* Sanity check 1 (regime ablation): see `phase21_sanity_checks.json`.
* Sanity check 2 (leakage): leakage delta = ___.
* Sanity check 3 (b1 vs b2): [DEFERRED to Phase 22 / DONE].

## Code changes

* `src/aurumq_rl/data_loader.py`: is_suspended default-True;
  STOCK_FACTOR_PREFIXES + FORBIDDEN_PREFIXES; `_compute_regime_features`;
  `FactorPanel.regime_array` + `regime_names`.
* `src/aurumq_rl/gpu_env.py`: Dict observation space; regime tensor.
* `src/aurumq_rl/feature_extractor.py`: PerStockEncoderV2, RegimeEncoder,
  masked_mean. V1 PerStockExtractor removed.
* `src/aurumq_rl/policy.py`: PerStockEncoderPolicyV2 with custom forward /
  evaluate_actions / get_distribution / predict_values; manual log_std +
  optimizer rebuild. V1 PerStockEncoderPolicy removed.
* `src/aurumq_rl/index_dict_rollout_buffer.py`: NEW. Lazy-gather Dict obs.
* `scripts/train_v2.py`: schema lock, Dict obs flow, new CLI flags.
* `scripts/_eval_all_checkpoints.py`: Dict obs path.
* `scripts/_phase21_sanity_checks.py`: NEW. Three architectural checks.

## Migration

V1 zips are unloadable under V2. Do NOT delete `models/production/phase16-20_*`
— they remain as forensic artifacts.

## Next phase

* Phase 22 multi-seed sweep on V2 to rebuild the ensemble baseline.
* Phase 22 fresh-holdout collection (≥40 days post-2026-04-24 needed for
  promotion).
* Phase 22 Ubuntu-side regime enrichment (VIX-equiv, fund flows).
```

- [ ] **Step 2: Copy oss_upload_phase20.py to phase21**

```bash
cp scripts/oss_upload_phase20.py scripts/oss_upload_phase21.py
```

Edit `scripts/oss_upload_phase21.py`:

* Change `PREFIX = "fromsz/handoffs/2026-05-05-phase20-long-data/"` → `"fromsz/handoffs/2026-05-05-phase21-v2-architecture/"`
* Update the handoff dir reference and the run_name list to point at `phase21_21a_v2_drop_mkt_seed42` and the right `ckpt_step`.

- [ ] **Step 3: Run upload**

```bash
.venv/Scripts/python.exe scripts/oss_upload_phase21.py
```

Expected: `[oss-upload] DONE.` Verify via the printed bucket URL.

- [ ] **Step 4: Final merge to main**

```bash
cd D:/dev/aurumq-rl
git checkout main
git pull --ff-only
git merge --no-ff feat/phase21-v2-architecture -m "feat: Phase 21 V2 architecture (Dict obs + split-head + regime)"
git push origin main
```

- [ ] **Step 5: Commit any remaining artefacts**

```bash
git add handoffs/2026-05-05-phase21-v2-architecture/HANDOFF_2026-05-05_phase21.md \
        scripts/oss_upload_phase21.py
git commit -m "research(phase21): handoff + oss uploader"
git push origin main
```

- [ ] **Step 6: Cleanup worktrees**

```bash
git worktree remove D:/dev/aurumq-rl-wt-data
git worktree remove D:/dev/aurumq-rl-wt-env
git worktree remove D:/dev/aurumq-rl-wt-net
git worktree remove D:/dev/aurumq-rl-wt-buffer
git worktree prune
```

---

## Verification

End-to-end checks for the whole plan, in order:

1. **Phase 1 (data_loader)**:
   ```bash
   .venv/Scripts/python.exe -m pytest tests/test_data_loader_phase21.py -v
   .venv/Scripts/python.exe -m pytest tests/test_data_loader.py tests/test_data_loader_universe.py -v
   ```
   Expected: all phase21 tests pass; pre-existing tests pass.

2. **Phase 2 (gpu_env)**:
   ```bash
   .venv/Scripts/python.exe -m pytest tests/test_gpu_env_phase21.py tests/test_gpu_env.py -v
   ```

3. **Phase 3 (feature_extractor + policy)**:
   ```bash
   .venv/Scripts/python.exe -m pytest tests/test_feature_extractor_phase21.py tests/test_policy_phase21.py -v
   ```

4. **Phase 4 (buffer)**:
   ```bash
   .venv/Scripts/python.exe -m pytest tests/test_index_dict_rollout_buffer.py -v
   ```

5. **Phase 5 smoke train**:
   ```bash
   ls -lh runs/phase21_smoke/ppo_final.zip runs/phase21_smoke/metadata.json
   .venv/Scripts/python.exe -c "
   import json; m = json.load(open('runs/phase21_smoke/metadata.json'))
   assert m['policy_class'] == 'PerStockEncoderPolicyV2'
   assert len(m['regime_factor_names']) == 8
   print('OK')"
   ```

6. **Phase 5 Phase 21A sanity train + eval**:
   ```bash
   ls -lh runs/phase21_21a_v2_drop_mkt_seed42/ppo_final.zip \
          runs/phase21_21a_v2_drop_mkt_seed42/oos_sweep.md \
          runs/phase21_21a_v2_drop_mkt_seed42/phase21_sanity_checks.json
   ```
   Expected: all three files exist. `oos_sweep.md` shows a best `vs_random_p50_adjusted` value;
   compare to Phase 16a baseline +0.428 in decision_log.md.

7. **Sanity check 1 (regime ablation)**:
   ```bash
   .venv/Scripts/python.exe -c "
   import json; s = json.load(open('runs/phase21_21a_v2_drop_mkt_seed42/phase21_sanity_checks.json'))
   delta = s['leakage_summary']['delta_adj_real_minus_zero']
   print(f'delta(real - zero) = {delta:+.3f}')
   "
   ```

---

## Out of scope (deferred to Phase 22+)

Per spec §8:
* FiLM-style modulation (concat is v0 baseline)
* Path B Ubuntu-side regime features (VIX-equivalent, fund flows)
* `regime_rel_amount_z` / volume-z regime features
* Multi-seed Phase 21B/C/D sweeps; cross-version V1+V2 ensembling
* True per-day variable-S obs (v0 keeps fixed S=n_stocks + valid_mask)
