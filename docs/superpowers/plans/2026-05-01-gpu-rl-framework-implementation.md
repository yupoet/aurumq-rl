# GPU-Vectorised RL Training Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the GPU-vectorised stock-picking RL training framework specified in `docs/superpowers/specs/2026-05-01-gpu-rl-framework-design.md`: `GPUStockPickingEnv` (panel as cuda tensors, SB3 VecEnv), `PerStockEncoderPolicy` (Deep-Sets per-stock encoder ~50K params), `factor_importance.py` (Integrated Gradients + per-date cross-section permutation), web `FactorImportancePanel`, and `train_v2.py` integrating all four. Targets fps 1000-3000, GPU mean util ≥ 70 %, all 343 factors used.

**Architecture:** Single-process VecEnv with panel resident on GPU + per-stock encoder policy with shared MLP across stocks (mathematically permutation-equivariant for action, invariant for value). bf16 autocast for forward/backward. Five parallel agents in git worktrees (env, policy, importance, web, train) — env / policy / importance / web are independent; train waits for env+policy+importance to land. TDD throughout.

**Tech Stack:** PyTorch 2.11 + cu126 · stable-baselines3 2.8 · gymnasium · polars 1.40 · Next.js 16 (Turbopack) · Tailwind v4 · recharts · pytest

---

## File Structure

### New files (Python)

| Path | Owner agent | Responsibility |
|---|---|---|
| `src/aurumq_rl/gpu_env.py` | agent-env | `GPUStockPickingEnv` (SB3 VecEnv subclass, panel on cuda) |
| `src/aurumq_rl/feature_extractor.py` | agent-policy | `PerStockExtractor` (BaseFeaturesExtractor subclass, returns dict {per_stock, pooled}) |
| `src/aurumq_rl/policy.py` | agent-policy | `PerStockEncoderPolicy` (ActorCriticPolicy subclass with custom forward/_predict/evaluate_actions/predict_values) |
| `src/aurumq_rl/factor_importance.py` | agent-importance | Integrated Gradients + permutation importance pure functions |
| `scripts/train_v2.py` | agent-train | Training entry point using new env+policy |
| `scripts/eval_factor_importance.py` | agent-importance | Post-training CLI: writes `runs/<id>/factor_importance.json` |
| `tests/test_gpu_env.py` | agent-env | Numerical-equivalence + GPU residency + auto-reset |
| `tests/test_policy.py` | agent-policy | Equivariance / invariance / param count / bf16 |
| `tests/test_factor_importance.py` | agent-importance | Synthetic identifiability + determinism |
| `tests/_synthetic_panel.py` | agent-env (shared) | Tiny synthetic FactorPanel fixture used by all three test modules |

### New files (Web)

| Path | Owner agent | Responsibility |
|---|---|---|
| `web/components/FactorImportancePanel.tsx` | agent-web | Bar chart by group + per-factor saliency heatmap |

### Modified files

| Path | Owner agent | Change |
|---|---|---|
| `web/lib/runs-shared.ts` | agent-web | Add `FactorImportance`, `FactorGroupImportance` interfaces |
| `web/lib/runs.ts` | agent-web | Add `readFactorImportance(id)` server-side helper |
| `web/app/api/runs/[...id]/route.ts` | agent-web | Add `?part=factor-importance` branch |
| `web/app/runs/[...id]/page.tsx` | agent-web | Read + render `<FactorImportancePanel>` when present |

### Untouched but referenced

- `src/aurumq_rl/env.py` — fallback baseline (StockPickingEnv stays unmodified)
- `src/aurumq_rl/data_loader.py` — `load_panel()` returns `FactorPanel` NamedTuple with `factor_array, return_array, pct_change_array, is_st_array, is_suspended_array, days_since_ipo_array, stock_codes, factor_names`. Used by all agents to build test fixtures and the train_v2 pipeline.
- `src/aurumq_rl/price_limits.py::compute_dynamic_limits()` — used by agent-env to build `valid_mask`.
- `scripts/train.py` — legacy baseline trainer, untouched.
- `scripts/eval_backtest.py` — works against any model with valid `metadata.json`; the new policy must save the same metadata fields.

---

## Phase 0: Repo prep (one-shot, blocks everything)

**Files:** `tests/_synthetic_panel.py` (NEW), `tests/conftest.py` (MODIFY)

This phase creates the shared test fixture that agent-env, agent-policy, and agent-importance all depend on. Done first on `main` so worktrees inherit it.

### Task 0.1: Create the shared synthetic-panel fixture

- [ ] **Step 1: Write `tests/_synthetic_panel.py`**

```python
"""Synthetic FactorPanel fixtures shared across GPU-framework tests.

Why a separate module (not just conftest fixtures): the same
construction is needed by gpu_env, policy, and factor_importance
tests, plus by some smoke scripts. Keeping it as a plain module
makes it importable from anywhere.
"""
from __future__ import annotations

import datetime as _dt

import numpy as np

from aurumq_rl.data_loader import FactorPanel


def make_synthetic_panel(
    n_dates: int = 60,
    n_stocks: int = 50,
    n_factors: int = 20,
    seed: int = 0,
    plant_true_factor: bool = False,
    true_factor_index: int = 0,
    true_factor_strength: float = 0.5,
) -> FactorPanel:
    """Build a small in-memory FactorPanel for tests.

    All factor values ~ N(0, 1). Returns are pure noise unless
    ``plant_true_factor=True`` — in that case
    ``returns[t, s] = strength * factors[t, s, true_factor_index] + noise``,
    so a working factor-importance module must rank that factor first.
    """
    rng = np.random.default_rng(seed)
    factor_array = rng.standard_normal((n_dates, n_stocks, n_factors)).astype(np.float32)
    base_returns = rng.standard_normal((n_dates, n_stocks)).astype(np.float32) * 0.02
    if plant_true_factor:
        signal = true_factor_strength * factor_array[..., true_factor_index]
        return_array = (base_returns + signal).astype(np.float32)
    else:
        return_array = base_returns
    pct_change_array = return_array.copy()
    is_st_array = np.zeros((n_dates, n_stocks), dtype=np.bool_)
    is_suspended_array = np.zeros((n_dates, n_stocks), dtype=np.bool_)
    days_since_ipo_array = np.full((n_dates, n_stocks), 1000, dtype=np.float32)
    base_date = _dt.date(2024, 1, 2)
    dates = [base_date + _dt.timedelta(days=i) for i in range(n_dates)]
    stock_codes = [f"SYN{i:04d}.SH" for i in range(n_stocks)]
    factor_names = [f"alpha_{i:03d}" if i < n_factors // 2 else f"gtja_{i:03d}"
                    for i in range(n_factors)]
    return FactorPanel(
        factor_array=factor_array,
        return_array=return_array,
        pct_change_array=pct_change_array,
        is_st_array=is_st_array,
        is_suspended_array=is_suspended_array,
        days_since_ipo_array=days_since_ipo_array,
        dates=dates,
        stock_codes=stock_codes,
        factor_names=factor_names,
    )
```

- [ ] **Step 2: Verify the fixture imports cleanly**

```bash
.venv/Scripts/python.exe -c "from tests._synthetic_panel import make_synthetic_panel; p = make_synthetic_panel(); print(p.factor_array.shape, p.return_array.shape, len(p.stock_codes), len(p.factor_names))"
```
Expected output: `(60, 50, 20) (60, 50) 50 20`

- [ ] **Step 3: Commit**

```bash
git add tests/_synthetic_panel.py
git commit -m "test(framework): add shared synthetic-panel fixture for GPU framework tests"
```

### Task 0.2: Verify CUDA + bf16 are usable

- [ ] **Step 1: Run a one-liner to confirm cuda + bf16**

```bash
.venv/Scripts/python.exe -c "import torch; print('cuda', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0)); x = torch.zeros(10, dtype=torch.bfloat16, device='cuda'); print('bf16 cuda ok', x.dtype, x.device)"
```
Expected: `cuda True`, `device NVIDIA GeForce RTX 4070`, `bf16 cuda ok torch.bfloat16 cuda:0`. If cuda is False, **stop the plan** — implementation is GPU-only.

### Task 0.3: Create five worktrees off `main`

- [ ] **Step 1: Make worktrees**

```bash
git -C /d/dev/aurumq-rl worktree add /d/dev/aurumq-rl-wt-env       -b feat/gpu-framework-env
git -C /d/dev/aurumq-rl worktree add /d/dev/aurumq-rl-wt-policy    -b feat/gpu-framework-policy
git -C /d/dev/aurumq-rl worktree add /d/dev/aurumq-rl-wt-importance -b feat/gpu-framework-importance
git -C /d/dev/aurumq-rl worktree add /d/dev/aurumq-rl-wt-web       -b feat/gpu-framework-web
# agent-train works on main directly after env/policy/importance merge
```

- [ ] **Step 2: Verify**

```bash
git -C /d/dev/aurumq-rl worktree list
```
Expected: 5 entries (main + 4 worktrees).

---

## Phase 1: agent-env — `GPUStockPickingEnv`

**Worktree:** `D:/dev/aurumq-rl-wt-env` (branch `feat/gpu-framework-env`)
**Files created:** `src/aurumq_rl/gpu_env.py`, `tests/test_gpu_env.py`

### Task 1.1: Write the skeleton class with all SB3 VecEnv abstract methods stubbed

**Files:**
- Create: `src/aurumq_rl/gpu_env.py`

- [ ] **Step 1: Write `src/aurumq_rl/gpu_env.py` skeleton**

```python
"""GPU-vectorised stock-picking environment.

Inherits from stable_baselines3.common.vec_env.VecEnv (NOT
gymnasium.vector.VectorEnv). All n_envs share a single panel
tensor on cuda; per-env state is a time-index vector also on
cuda. step_wait() is a single batched tensor op.

See docs/superpowers/specs/2026-05-01-gpu-rl-framework-design.md §5.
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.vec_env import VecEnv


class GPUStockPickingEnv(VecEnv):
    """Single-process VecEnv with the panel resident on cuda."""

    def __init__(
        self,
        panel: torch.Tensor,        # (T, S, F) fp32 cuda
        returns: torch.Tensor,      # (T, S)    fp32 cuda
        valid_mask: torch.Tensor,   # (T, S)    bool cuda
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
        if panel.shape[0] != returns.shape[0] or panel.shape[1] != returns.shape[1]:
            raise ValueError("panel and returns date/stock dims must match")
        if panel.shape[:2] != valid_mask.shape:
            raise ValueError("panel and valid_mask date/stock dims must match")

        self.panel = panel
        self.returns = returns
        self.valid_mask = valid_mask
        self.n_dates, self.n_stocks, self.n_factors = panel.shape
        self.episode_length = episode_length
        self.forward_period = forward_period
        self.top_k = top_k
        self.cost_bps = cost_bps
        self.turnover_coef = turnover_coef
        self.device = torch.device(device)
        self._rng = torch.Generator(device=self.device)
        if seed is not None:
            self._rng.manual_seed(seed)

        # Per-env state, all on cuda
        self.t = torch.zeros(n_envs, dtype=torch.long, device=self.device)
        self.steps_done = torch.zeros(n_envs, dtype=torch.long, device=self.device)
        self.episode_returns = torch.zeros(n_envs, dtype=torch.float32, device=self.device)
        self.prev_top_idx = torch.zeros(n_envs, top_k, dtype=torch.long, device=self.device)
        self._pending_action: torch.Tensor | None = None

        observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.n_stocks, self.n_factors),
            dtype=np.float32,
        )
        action_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(self.n_stocks,),
            dtype=np.float32,
        )
        super().__init__(num_envs=n_envs, observation_space=observation_space, action_space=action_space)

    # SB3 VecEnv abstract methods --------------------------------------

    def reset(self):
        self._sample_starts(torch.ones(self.num_envs, dtype=torch.bool, device=self.device))
        self.steps_done.zero_()
        self.episode_returns.zero_()
        self.prev_top_idx.zero_()
        return self._current_obs()

    def step_async(self, actions):
        self._pending_action = self._coerce_action(actions)

    def step_wait(self):
        raise NotImplementedError("filled in Task 1.4")

    def close(self) -> None:
        pass

    def get_attr(self, attr_name: str, indices=None):
        # Most common SB3 internal asks: 'render_mode', 'spec'
        if attr_name in {"render_mode", "spec"}:
            return [None] * self._indices_count(indices)
        raise NotImplementedError(f"get_attr({attr_name!r}) not supported")

    def set_attr(self, attr_name: str, value, indices=None) -> None:
        raise NotImplementedError(f"set_attr({attr_name!r}) not supported")

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        raise NotImplementedError(f"env_method({method_name!r}) not supported")

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self._indices_count(indices)

    def seed(self, seed=None):
        if seed is not None:
            self._rng.manual_seed(seed)
        return [seed] * self.num_envs

    # Helpers ----------------------------------------------------------

    def _coerce_action(self, actions):
        if isinstance(actions, np.ndarray):
            return torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        return actions.to(self.device, dtype=torch.float32)

    def _current_obs(self) -> torch.Tensor:
        return self.panel[self.t]

    def _sample_starts(self, mask: torch.Tensor) -> None:
        max_start = self.n_dates - self.episode_length - self.forward_period
        if max_start <= 0:
            raise ValueError(
                f"panel too short: n_dates={self.n_dates} episode_length="
                f"{self.episode_length} forward_period={self.forward_period}"
            )
        new_starts = torch.randint(
            low=0, high=max_start + 1,
            size=(int(mask.sum().item()),),
            generator=self._rng, device=self.device,
        )
        self.t[mask] = new_starts

    def _indices_count(self, indices) -> int:
        if indices is None:
            return self.num_envs
        if isinstance(indices, int):
            return 1
        return len(indices)
```

- [ ] **Step 2: Verify it imports**

```bash
cd /d/dev/aurumq-rl-wt-env
.venv/Scripts/python.exe -c "from aurumq_rl.gpu_env import GPUStockPickingEnv; print(GPUStockPickingEnv.__mro__)"
```
Expected: prints MRO including `VecEnv`. Module loads with no error.

- [ ] **Step 3: Commit**

```bash
git add src/aurumq_rl/gpu_env.py
git commit -m "feat(gpu-env): scaffold GPUStockPickingEnv (SB3 VecEnv subclass)"
```

### Task 1.2: First test — panel residency on cuda

**Files:**
- Create: `tests/test_gpu_env.py`

- [ ] **Step 1: Write `tests/test_gpu_env.py` (failing test only)**

```python
"""Tests for src/aurumq_rl/gpu_env.py."""
from __future__ import annotations

import numpy as np
import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")

from aurumq_rl.gpu_env import GPUStockPickingEnv
from tests._synthetic_panel import make_synthetic_panel


def _panel_to_cuda(syn, device="cuda"):
    panel = torch.from_numpy(syn.factor_array).to(device)
    returns = torch.from_numpy(syn.return_array).to(device)
    valid_mask = torch.ones(panel.shape[:2], dtype=torch.bool, device=device)
    return panel, returns, valid_mask


@cuda
def test_env_residency_on_cuda():
    syn = make_synthetic_panel(n_dates=60, n_stocks=50, n_factors=20)
    panel, returns, valid_mask = _panel_to_cuda(syn)
    env = GPUStockPickingEnv(panel, returns, valid_mask, n_envs=4)
    assert env.panel.device.type == "cuda"
    assert env.returns.device.type == "cuda"
    assert env.valid_mask.device.type == "cuda"
    assert env.t.device.type == "cuda"
    assert env.num_envs == 4
```

- [ ] **Step 2: Run, expect pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_gpu_env.py::test_env_residency_on_cuda -v
```
Expected: PASSED. (The init code already enforces cuda residency.)

- [ ] **Step 3: Commit**

```bash
git add tests/test_gpu_env.py
git commit -m "test(gpu-env): assert panel/returns/valid_mask residency on cuda"
```

### Task 1.3: reset() returns valid obs shape and types

> **NOTE from Phase 1 retro:** SB3 2.8's `obs_as_tensor` doesn't accept
> `torch.Tensor` (only `np.ndarray` and `dict`). `reset()` and
> `step_wait()` MUST materialise the cuda obs to numpy at the VecEnv
> boundary. Internal panel stays on cuda; only the return value is
> numpy. The test below uses numpy assertions to match that reality.
> Also use a smaller `episode_length`/`forward_period` so the default
> `n_dates=60` panel from `make_synthetic_panel` has a valid start
> window.

- [ ] **Step 1: Add test**

Append to `tests/test_gpu_env.py`:

```python
@cuda
def test_reset_returns_correct_shape_and_dtype():
    syn = make_synthetic_panel()
    panel, returns, valid_mask = _panel_to_cuda(syn)
    env = GPUStockPickingEnv(panel, returns, valid_mask, n_envs=3,
                             episode_length=30, forward_period=5, seed=42)
    obs = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (3, 50, 20)         # (n_envs, n_stocks, n_factors)
    assert obs.dtype == np.float32
    # Each env got an independently sampled start
    starts = env.t.cpu().tolist()
    assert all(0 <= s for s in starts)
    # Internal panel still cuda-resident
    assert env.panel.device.type == "cuda"
```

- [ ] **Step 2: Run, expect pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_gpu_env.py::test_reset_returns_correct_shape_and_dtype -v
```
Expected: PASSED. (The reset implemented in 1.1 already covers this.)

- [ ] **Step 3: Commit**

```bash
git add tests/test_gpu_env.py
git commit -m "test(gpu-env): cover reset() obs shape/dtype/cuda residency"
```

### Task 1.4: Implement step_wait + first behavioural test

- [ ] **Step 1: Add a failing test for step_wait** (append to `tests/test_gpu_env.py`)

```python
@cuda
def test_step_returns_obs_rewards_dones_infos():
    syn = make_synthetic_panel(n_dates=120)
    panel, returns, valid_mask = _panel_to_cuda(syn)
    env = GPUStockPickingEnv(panel, returns, valid_mask, n_envs=2, episode_length=50,
                             forward_period=5, top_k=10, cost_bps=0.0, seed=0)
    env.reset()
    actions = np.random.default_rng(0).standard_normal((2, 50)).astype(np.float32)
    env.step_async(actions)
    obs, rewards, dones, infos = env.step_wait()

    assert isinstance(obs, np.ndarray) and obs.shape == (2, 50, 20) and obs.dtype == np.float32
    assert isinstance(rewards, np.ndarray) and rewards.shape == (2,) and rewards.dtype == np.float32
    assert isinstance(dones, np.ndarray) and dones.shape == (2,) and dones.dtype == bool
    assert isinstance(infos, list) and len(infos) == 2
    # Internal panel still cuda
    assert env.panel.device.type == "cuda"
```

- [ ] **Step 2: Run, expect fail**

```bash
.venv/Scripts/python.exe -m pytest tests/test_gpu_env.py::test_step_returns_obs_rewards_dones_infos -v
```
Expected: FAIL with `NotImplementedError("filled in Task 1.4")`.

- [ ] **Step 3: Implement `step_wait` in `src/aurumq_rl/gpu_env.py`**

Replace the body of `step_wait`:

```python
    def step_wait(self):
        assert self._pending_action is not None, "step_async must be called before step_wait"
        action = self._pending_action
        self._pending_action = None

        # 1. mask invalid stocks (they can never enter top-K)
        action = action.masked_fill(~self.valid_mask[self.t], float("-inf"))
        # 2. top-K
        top_idx = torch.topk(action, k=self.top_k, dim=-1).indices  # (n_envs, K)
        # 3. forward returns gathered for the K picked stocks
        fwd_t = (self.t + self.forward_period).clamp(max=self.n_dates - 1)
        fwd_rets = self.returns[fwd_t].gather(1, top_idx)            # (n_envs, K)
        rewards = fwd_rets.mean(dim=-1) - self.cost_bps / 1e4
        # 4. turnover penalty (Jaccard-style)
        if self.turnover_coef > 0.0:
            overlap = torch.zeros_like(rewards)
            for i in range(self.num_envs):
                overlap[i] = float(
                    len(set(top_idx[i].tolist()) & set(self.prev_top_idx[i].tolist()))
                )
            jaccard_dist = 1.0 - overlap / float(self.top_k)
            rewards = rewards - self.turnover_coef * jaccard_dist
        self.prev_top_idx = top_idx
        self.episode_returns += rewards

        # 5. advance time
        self.t = self.t + 1
        self.steps_done = self.steps_done + 1
        dones = self.steps_done >= self.episode_length

        # 6. for done envs, build SB3 episode info, then auto-reset
        infos: list[dict] = [{} for _ in range(self.num_envs)]
        if bool(dones.any().item()):
            for i in dones.nonzero(as_tuple=True)[0].tolist():
                infos[i]["episode"] = {
                    "r": float(self.episode_returns[i].item()),
                    "l": int(self.steps_done[i].item()),
                }
            self._reset_done_envs(dones)

        obs = self._current_obs()
        return obs, rewards.detach().cpu().numpy().astype(np.float32), dones.detach().cpu().numpy(), infos

    def _reset_done_envs(self, dones: torch.Tensor) -> None:
        self._sample_starts(dones)
        self.steps_done = torch.where(
            dones,
            torch.zeros_like(self.steps_done),
            self.steps_done,
        )
        self.episode_returns = torch.where(
            dones,
            torch.zeros_like(self.episode_returns),
            self.episode_returns,
        )
        # Zero prev_top_idx for done envs only
        self.prev_top_idx[dones] = 0
```

- [ ] **Step 4: Run test, expect pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_gpu_env.py::test_step_returns_obs_rewards_dones_infos -v
```
Expected: PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/aurumq_rl/gpu_env.py tests/test_gpu_env.py
git commit -m "feat(gpu-env): implement step_wait with auto-reset and episode info"
```

### Task 1.5: Auto-reset semantics test

- [ ] **Step 1: Add a test that drives the env past episode_length and checks that done envs received the `episode` info dict and a fresh start**

Append to `tests/test_gpu_env.py`:

```python
@cuda
def test_auto_reset_on_episode_end():
    syn = make_synthetic_panel(n_dates=60)
    panel, returns, valid_mask = _panel_to_cuda(syn)
    env = GPUStockPickingEnv(panel, returns, valid_mask, n_envs=1, episode_length=5,
                             forward_period=2, top_k=5, seed=0)
    env.reset()
    initial_t = env.t.clone()
    actions = np.zeros((1, 50), dtype=np.float32)
    last_info = None
    last_done = False
    for _ in range(6):
        env.step_async(actions)
        _, _, dones, infos = env.step_wait()
        if dones[0]:
            last_done = True
            last_info = infos[0]
            break
    assert last_done, "episode_length=5 should fire done within 6 steps"
    assert "episode" in last_info, "done env must populate info['episode']"
    assert {"r", "l"} <= last_info["episode"].keys()
    assert env.steps_done[0].item() == 0, "steps_done resets after auto-reset"
    # New start was sampled (very likely different)
    assert env.t[0].item() != initial_t[0].item()
```

- [ ] **Step 2: Run, expect pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_gpu_env.py::test_auto_reset_on_episode_end -v
```
Expected: PASSED.

- [ ] **Step 3: Commit**

```bash
git add tests/test_gpu_env.py
git commit -m "test(gpu-env): cover auto-reset + episode info on done"
```

### Task 1.6: SB3 VecEnv contract smoke test

- [ ] **Step 1: Add tests for VecEnv methods SB3 PPO actually calls**

Append to `tests/test_gpu_env.py`:

```python
@cuda
def test_vecenv_required_methods():
    syn = make_synthetic_panel()
    panel, returns, valid_mask = _panel_to_cuda(syn)
    env = GPUStockPickingEnv(panel, returns, valid_mask, n_envs=2)
    assert env.get_attr("render_mode") == [None, None]
    assert env.env_is_wrapped(object) == [False, False]
    env.close()
```

- [ ] **Step 2: Run, expect pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_gpu_env.py::test_vecenv_required_methods -v
```
Expected: PASSED.

- [ ] **Step 3: Commit**

```bash
git add tests/test_gpu_env.py
git commit -m "test(gpu-env): cover SB3 VecEnv contract surface (get_attr/env_is_wrapped/close)"
```

### Task 1.7: Smoke run with SB3 PPO

- [ ] **Step 1: Add a 1-rollout PPO smoke test** to verify SB3 actually accepts our VecEnv

Append to `tests/test_gpu_env.py`:

```python
@cuda
def test_sb3_ppo_one_rollout():
    """Smoke: SB3 PPO can collect one rollout against our VecEnv without crashing."""
    from stable_baselines3 import PPO

    syn = make_synthetic_panel(n_dates=120, n_stocks=20, n_factors=8)
    panel, returns, valid_mask = _panel_to_cuda(syn)
    env = GPUStockPickingEnv(panel, returns, valid_mask, n_envs=4, episode_length=30,
                             forward_period=5, top_k=4, seed=0)
    model = PPO("MlpPolicy", env, n_steps=64, batch_size=32, n_epochs=1,
                verbose=0, device="cuda")
    model.learn(total_timesteps=256)
    # If we got here, collect_rollouts + train both worked
```

- [ ] **Step 2: Run, expect pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_gpu_env.py::test_sb3_ppo_one_rollout -v -s
```
Expected: PASSED. If it fails, the failure mode tells us which SB3 surface we're missing — fix in `gpu_env.py` and re-run.

- [ ] **Step 3: Commit**

```bash
git add tests/test_gpu_env.py
git commit -m "test(gpu-env): smoke that SB3 PPO collect_rollouts works against the new env"
```

### Task 1.8: Push the env branch

- [ ] **Step 1: Verify all tests pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_gpu_env.py -v
```
Expected: 5 passed.

- [ ] **Step 2: Push branch**

```bash
git push -u origin feat/gpu-framework-env
```

agent-env is done. Branch ready to merge to main.

---

## Phase 2: agent-policy — `PerStockEncoderPolicy`

**Worktree:** `D:/dev/aurumq-rl-wt-policy` (branch `feat/gpu-framework-policy`)
**Files created:** `src/aurumq_rl/feature_extractor.py`, `src/aurumq_rl/policy.py`, `tests/test_policy.py`

> **NOTE from Phase 1 retro:** The cuda obs is materialised to numpy at the
> VecEnv boundary (see updated spec §5.4). SB3 will hand the policy an
> `obs` tensor that's already converted from numpy and moved to cuda by
> `obs_as_tensor`. So `PerStockExtractor.forward(obs)` receives a cuda
> float32 tensor of shape `(B, n_stocks, n_factors)` — same as the spec
> said, just travelling through numpy in the middle. No code change here,
> just a head's-up.

### Task 2.1: PerStockExtractor scaffold

**Files:**
- Create: `src/aurumq_rl/feature_extractor.py`

- [ ] **Step 1: Write the extractor**

```python
"""Per-stock features extractor with shared MLP across stocks (Deep Sets)."""
from __future__ import annotations

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class PerStockExtractor(BaseFeaturesExtractor):
    """Apply a shared MLP per stock; return both per-stock and pooled features.

    Returned features are a TensorDict-like dict (we use a regular dict because
    SB3 doesn't insist on a Tensor return — see PerStockEncoderPolicy.forward).

    Output keys:
      - "per_stock":  (B, n_stocks, out_dim) — used by action head
      - "pooled":     (B,         out_dim) — mean-pool across stocks, used by value head
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        hidden: tuple[int, ...] = (128, 64),
        out_dim: int = 32,
    ):
        # observation_space.shape = (n_stocks, n_factors)
        n_stocks, n_factors = observation_space.shape
        # SB3's BaseFeaturesExtractor stores features_dim — set it to per-stock out_dim
        # (we won't use SB3's mlp_extractor split anyway, see policy.py)
        super().__init__(observation_space, features_dim=out_dim)
        self.n_stocks = n_stocks
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

    def forward(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        # obs: (B, n_stocks, n_factors)
        b, s, f = obs.shape
        flat = obs.reshape(b * s, f)
        encoded = self.mlp(flat).reshape(b, s, self.out_dim)
        pooled = encoded.mean(dim=1)
        return {"per_stock": encoded, "pooled": pooled}
```

- [ ] **Step 2: Verify import**

```bash
cd /d/dev/aurumq-rl-wt-policy
.venv/Scripts/python.exe -c "from aurumq_rl.feature_extractor import PerStockExtractor; import gymnasium as gym; e = PerStockExtractor(gym.spaces.Box(-1,1,(50,20))); print(sum(p.numel() for p in e.parameters()))"
```
Expected: prints a parameter count under 100,000 (e.g. `~5K`).

- [ ] **Step 3: Commit**

```bash
git add src/aurumq_rl/feature_extractor.py
git commit -m "feat(policy): add PerStockExtractor (Deep Sets shared MLP)"
```

### Task 2.2: Test extractor permutation equivariance + shapes

**Files:**
- Create: `tests/test_policy.py`

- [ ] **Step 1: Write test**

```python
"""Tests for src/aurumq_rl/policy.py and feature_extractor.py."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")

from aurumq_rl.feature_extractor import PerStockExtractor


def _obs_space(n_stocks=50, n_factors=20):
    return gym.spaces.Box(-np.inf, np.inf, (n_stocks, n_factors), dtype=np.float32)


def test_extractor_output_shapes():
    ext = PerStockExtractor(_obs_space(50, 20), hidden=(128, 64), out_dim=32)
    obs = torch.randn(4, 50, 20)
    out = ext(obs)
    assert out["per_stock"].shape == (4, 50, 32)
    assert out["pooled"].shape == (4, 32)


def test_extractor_param_count_under_budget():
    ext = PerStockExtractor(_obs_space(3014, 343), hidden=(128, 64), out_dim=32)
    n_params = sum(p.numel() for p in ext.parameters())
    assert n_params <= 100_000, f"extractor has {n_params} params, budget is 100K"


def test_extractor_permutation_equivariance_per_stock():
    ext = PerStockExtractor(_obs_space(50, 20), hidden=(64,), out_dim=8)
    ext.eval()
    obs = torch.randn(2, 50, 20)
    pi = torch.randperm(50)
    out_a = ext(obs[:, pi])["per_stock"]
    out_b = ext(obs)["per_stock"][:, pi]
    assert torch.allclose(out_a, out_b, atol=1e-5)


def test_extractor_pool_invariance():
    ext = PerStockExtractor(_obs_space(50, 20), hidden=(64,), out_dim=8)
    ext.eval()
    obs = torch.randn(2, 50, 20)
    pi = torch.randperm(50)
    pooled_a = ext(obs[:, pi])["pooled"]
    pooled_b = ext(obs)["pooled"]
    assert torch.allclose(pooled_a, pooled_b, atol=1e-5)
```

- [ ] **Step 2: Run, expect pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_policy.py -v
```
Expected: 4 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/test_policy.py
git commit -m "test(policy): cover PerStockExtractor shapes / param count / equivariance / pool invariance"
```

### Task 2.3: PerStockEncoderPolicy scaffold

**Files:**
- Create: `src/aurumq_rl/policy.py`

- [ ] **Step 1: Write the policy**

```python
"""PerStockEncoderPolicy — SB3 ActorCriticPolicy with Deep-Sets feature
extraction. Permutation-equivariant action head, permutation-invariant
value head.

We override forward / _predict / evaluate_actions / predict_values
because SB3's default ActorCriticPolicy assumes a flat features tensor
and an mlp_extractor that splits it into (latent_pi, latent_vf). Our
extractor returns a dict and the two heads need different shapes.
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import torch
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn

from aurumq_rl.feature_extractor import PerStockExtractor


class _IdentityMlpExtractor(nn.Module):
    """No-op standin for SB3's mlp_extractor — we don't use the split."""

    def __init__(self, features_dim: int) -> None:
        super().__init__()
        self.latent_dim_pi = features_dim
        self.latent_dim_vf = features_dim

    def forward(self, features):  # not actually called
        return features, features

    def forward_actor(self, features):
        return features

    def forward_critic(self, features):
        return features


class PerStockEncoderPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        *args: Any,
        encoder_hidden: tuple[int, ...] = (128, 64),
        encoder_out_dim: int = 32,
        value_hidden: tuple[int, ...] = (64,),
        **kwargs: Any,
    ):
        # Hand the kwargs to SB3 with a custom features_extractor
        kwargs["features_extractor_class"] = PerStockExtractor
        kwargs["features_extractor_kwargs"] = {
            "hidden": encoder_hidden, "out_dim": encoder_out_dim,
        }
        kwargs["share_features_extractor"] = True
        # Disable SB3's mlp_extractor splits with empty net_arch
        kwargs.setdefault("net_arch", dict(pi=[], vf=[]))
        self._encoder_out_dim = encoder_out_dim
        self._value_hidden = value_hidden
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = _IdentityMlpExtractor(features_dim=self._encoder_out_dim)

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        # Override action_net and value_net with per-stock-aware heads
        n_stocks = self.action_space.shape[0]
        self.action_net = nn.Linear(self._encoder_out_dim, 1)  # per-stock score
        layers: list[nn.Module] = []
        prev = self._encoder_out_dim
        for h in self._value_hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.value_net = nn.Sequential(*layers)
        # Re-init action distribution + log_std
        self.action_dist = DiagGaussianDistribution(n_stocks)
        self.log_std = nn.Parameter(torch.full((n_stocks,), -0.69, dtype=torch.float32))  # ~log(0.5)

    def _features(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.features_extractor(obs)

    def forward(self, obs, deterministic: bool = False):
        feats = self._features(obs)
        scores = self.action_net(feats["per_stock"]).squeeze(-1)  # (B, S)
        values = self.value_net(feats["pooled"]).squeeze(-1)      # (B,)
        distribution = self.action_dist.proba_distribution(scores, self.log_std)
        actions = distribution.get_actions(deterministic=deterministic)
        log_probs = distribution.log_prob(actions)
        return actions, values, log_probs

    def evaluate_actions(self, obs, actions):
        feats = self._features(obs)
        scores = self.action_net(feats["per_stock"]).squeeze(-1)
        values = self.value_net(feats["pooled"]).squeeze(-1)
        distribution = self.action_dist.proba_distribution(scores, self.log_std)
        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return values, log_probs, entropy

    def predict_values(self, obs):
        feats = self._features(obs)
        return self.value_net(feats["pooled"]).squeeze(-1)

    def _predict(self, obs, deterministic: bool = False):
        feats = self._features(obs)
        scores = self.action_net(feats["per_stock"]).squeeze(-1)
        distribution = self.action_dist.proba_distribution(scores, self.log_std)
        return distribution.get_actions(deterministic=deterministic)
```

- [ ] **Step 2: Verify import**

```bash
.venv/Scripts/python.exe -c "from aurumq_rl.policy import PerStockEncoderPolicy; print('ok')"
```
Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add src/aurumq_rl/policy.py
git commit -m "feat(policy): add PerStockEncoderPolicy with per-stock action head + pooled value head"
```

### Task 2.4: Test policy forward + equivariance

- [ ] **Step 1: Add tests** to `tests/test_policy.py`:

```python
from aurumq_rl.policy import PerStockEncoderPolicy


def _make_policy(n_stocks=50, n_factors=20, lr=1e-3):
    obs_space = _obs_space(n_stocks, n_factors)
    act_space = gym.spaces.Box(0.0, 1.0, (n_stocks,), dtype=np.float32)
    policy = PerStockEncoderPolicy(
        obs_space, act_space, lr_schedule=lambda _: lr,
        encoder_hidden=(64,), encoder_out_dim=16, value_hidden=(32,),
    )
    return policy


def test_policy_param_count_under_budget():
    policy = _make_policy(n_stocks=3014, n_factors=343)
    n_params = sum(p.numel() for p in policy.parameters())
    assert n_params <= 100_000, f"policy has {n_params} params, budget is 100K"


def test_policy_forward_shapes():
    policy = _make_policy(n_stocks=50, n_factors=20)
    obs = torch.randn(4, 50, 20)
    actions, values, log_probs = policy(obs)
    assert actions.shape == (4, 50)
    assert values.shape == (4,)
    assert log_probs.shape == (4,)


def test_policy_action_equivariance_in_eval():
    """Action mean must permute the same way the input does."""
    policy = _make_policy(n_stocks=50, n_factors=20)
    policy.eval()
    obs = torch.randn(3, 50, 20)
    pi = torch.randperm(50)
    # Forward with permuted input vs forward then permute output
    feats_a = policy._features(obs[:, pi])
    feats_b = policy._features(obs)
    scores_a = policy.action_net(feats_a["per_stock"]).squeeze(-1)
    scores_b = policy.action_net(feats_b["per_stock"]).squeeze(-1)
    assert torch.allclose(scores_a, scores_b[:, pi], atol=1e-5)


def test_policy_value_invariance():
    policy = _make_policy(n_stocks=50, n_factors=20)
    policy.eval()
    obs = torch.randn(3, 50, 20)
    pi = torch.randperm(50)
    v_a = policy.predict_values(obs[:, pi])
    v_b = policy.predict_values(obs)
    assert torch.allclose(v_a, v_b, atol=1e-5)


@cuda
def test_policy_bf16_autocast_finite():
    policy = _make_policy(n_stocks=50, n_factors=20).to("cuda")
    obs = torch.randn(2, 50, 20, device="cuda")
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        actions, values, log_probs = policy(obs)
    assert torch.isfinite(actions).all()
    assert torch.isfinite(values).all()
    assert torch.isfinite(log_probs).all()
```

- [ ] **Step 2: Run, expect pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_policy.py -v
```
Expected: 9 passed (4 from earlier + 5 here).

- [ ] **Step 3: Commit**

```bash
git add tests/test_policy.py
git commit -m "test(policy): cover forward shapes / equivariance / invariance / bf16 finite"
```

### Task 2.5: Push

```bash
git push -u origin feat/gpu-framework-policy
```

agent-policy done.

---

## Phase 3: agent-importance — Integrated Gradients + permutation importance

**Worktree:** `D:/dev/aurumq-rl-wt-importance` (branch `feat/gpu-framework-importance`)
**Files created:** `src/aurumq_rl/factor_importance.py`, `tests/test_factor_importance.py`, `scripts/eval_factor_importance.py`

### Task 3.1: Pure-function module

**Files:**
- Create: `src/aurumq_rl/factor_importance.py`

- [ ] **Step 1: Write the module**

```python
"""Factor-importance attribution: Integrated Gradients + permutation.

Both functions are pure and torch-only; they take a callable
``score_fn(panel) -> (B, n_stocks)`` so they don't depend on SB3
internals — the caller (eval_factor_importance.py) wraps a trained
policy into the right closure.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable

import numpy as np
import torch


def integrated_gradients(
    score_fn: Callable[[torch.Tensor], torch.Tensor],
    panel_batch: torch.Tensor,                  # (B, n_stocks, n_factors)
    n_alpha_steps: int = 50,
    baseline: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-factor saliency = average |IG| across batch and stocks.

    Returns a 1D tensor of length n_factors.
    """
    if baseline is None:
        baseline = torch.zeros_like(panel_batch)
    delta = (panel_batch - baseline).detach()
    saliency_sum = torch.zeros(panel_batch.shape[-1], device=panel_batch.device)
    for k in range(n_alpha_steps):
        alpha = (k + 0.5) / n_alpha_steps
        x = (baseline + alpha * (panel_batch - baseline)).detach().requires_grad_(True)
        scores = score_fn(x)               # (B, n_stocks)
        scores.sum().backward()
        # |grad| × Δ, averaged over batch and stocks
        attribution = (x.grad * delta).abs().mean(dim=(0, 1))
        saliency_sum += attribution
    return (saliency_sum / n_alpha_steps).detach()


def per_date_cross_section_shuffle(
    panel: torch.Tensor,                        # (T, S, F)
    cols: list[int],
    seed: int,
) -> torch.Tensor:
    """Return a copy of `panel` where the columns in `cols` are
    independently permuted across the stock axis on each date.
    Preserves time-series + per-date marginal; breaks cross-section
    ranking. See spec §7.
    """
    out = panel.clone()
    g = torch.Generator(device=panel.device)
    g.manual_seed(seed)
    T, S, _ = panel.shape
    for t in range(T):
        perm = torch.randperm(S, generator=g, device=panel.device)
        out[t, :, cols] = panel[t, perm][:, cols]
    return out


def permutation_importance(
    score_fn: Callable[[torch.Tensor], torch.Tensor],
    val_panel: torch.Tensor,                    # (T, S, F) cuda fp32
    val_returns: torch.Tensor,                  # (T, S)    cuda fp32
    factor_names: list[str],
    forward_period: int = 10,
    top_k: int = 30,
    n_seeds: int = 5,
    base_seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Per-prefix ΔIC + ΔSharpe via per-date cross-section shuffle.

    Returns a dict keyed by prefix (e.g. "alpha", "gtja", "mfp", ...)
    with statistics over n_seeds shuffles.
    """
    cols_by_prefix: dict[str, list[int]] = defaultdict(list)
    for i, name in enumerate(factor_names):
        prefix = name.split("_", 1)[0] if "_" in name else name
        cols_by_prefix[prefix].append(i)

    baseline_metrics = _eval_top_k_metrics(
        score_fn, val_panel, val_returns, forward_period, top_k,
    )
    out: dict[str, dict[str, float]] = {}
    for prefix, cols in cols_by_prefix.items():
        ic_drops, sharpe_drops = [], []
        for seed in range(base_seed, base_seed + n_seeds):
            shuffled = per_date_cross_section_shuffle(val_panel, cols, seed)
            m = _eval_top_k_metrics(
                score_fn, shuffled, val_returns, forward_period, top_k,
            )
            ic_drops.append(baseline_metrics["ic"] - m["ic"])
            sharpe_drops.append(baseline_metrics["sharpe"] - m["sharpe"])
        out[prefix] = {
            "n_factors": len(cols),
            "n_seeds": n_seeds,
            "ic_drop_mean": float(np.mean(ic_drops)),
            "ic_drop_std": float(np.std(ic_drops, ddof=1) if len(ic_drops) > 1 else 0.0),
            "sharpe_drop_mean": float(np.mean(sharpe_drops)),
            "sharpe_drop_std": float(np.std(sharpe_drops, ddof=1) if len(sharpe_drops) > 1 else 0.0),
        }
    return out


def _eval_top_k_metrics(
    score_fn: Callable[[torch.Tensor], torch.Tensor],
    panel: torch.Tensor,
    returns: torch.Tensor,
    forward_period: int,
    top_k: int,
) -> dict[str, float]:
    """Score every date independently, build top-K portfolio, return IC + annualised Sharpe."""
    T = panel.shape[0]
    valid_T = T - forward_period
    if valid_T <= 0:
        return {"ic": 0.0, "sharpe": 0.0}
    portfolio_returns = []
    ics = []
    with torch.no_grad():
        for t in range(valid_T):
            obs = panel[t : t + 1]                         # (1, S, F)
            scores = score_fn(obs)[0]                       # (S,)
            top_idx = torch.topk(scores, top_k).indices
            r = returns[t + forward_period][top_idx].mean()
            portfolio_returns.append(r.item())
            # IC: corr(scores, future returns)
            f = returns[t + forward_period]
            mask = torch.isfinite(scores) & torch.isfinite(f)
            if mask.sum() < 2 or scores[mask].std().item() < 1e-12:
                continue
            c = torch.corrcoef(torch.stack([scores[mask], f[mask]]))[0, 1].item()
            if np.isfinite(c):
                ics.append(c)
    if len(portfolio_returns) < 2:
        return {"ic": 0.0, "sharpe": 0.0}
    arr = np.asarray(portfolio_returns)
    s = arr.std(ddof=1)
    sharpe = float(arr.mean() / s * np.sqrt(252)) if s > 1e-12 else 0.0
    ic = float(np.mean(ics)) if ics else 0.0
    return {"ic": ic, "sharpe": sharpe}


__all__ = ["integrated_gradients", "permutation_importance", "per_date_cross_section_shuffle"]
```

- [ ] **Step 2: Verify import**

```bash
cd /d/dev/aurumq-rl-wt-importance
.venv/Scripts/python.exe -c "from aurumq_rl.factor_importance import integrated_gradients, permutation_importance; print('ok')"
```
Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add src/aurumq_rl/factor_importance.py
git commit -m "feat(importance): IG + per-date cross-section permutation importance"
```

### Task 3.2: Tests with synthetic identifiability

**Files:**
- Create: `tests/test_factor_importance.py`

- [ ] **Step 1: Write tests**

```python
"""Tests for src/aurumq_rl/factor_importance.py."""
from __future__ import annotations

import numpy as np
import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")

from aurumq_rl.factor_importance import (
    integrated_gradients,
    per_date_cross_section_shuffle,
    permutation_importance,
)


def _linear_score_fn(weights: torch.Tensor):
    """Closure: scores = obs @ weights, where weights is (n_factors,)."""
    def _fn(obs: torch.Tensor) -> torch.Tensor:
        # obs: (B, n_stocks, n_factors)
        return (obs * weights).sum(dim=-1)
    return _fn


def test_ig_recovers_planted_weight_direction():
    """A linear score with weight=10 on factor 3 should saliency-rank factor 3 first."""
    n_factors = 6
    weights = torch.zeros(n_factors)
    weights[3] = 10.0
    score_fn = _linear_score_fn(weights)
    panel_batch = torch.randn(4, 12, n_factors)
    sal = integrated_gradients(score_fn, panel_batch, n_alpha_steps=20)
    assert sal.shape == (n_factors,)
    assert sal.argmax().item() == 3


def test_per_date_cross_section_shuffle_preserves_marginal():
    panel = torch.randn(8, 30, 5)
    shuffled = per_date_cross_section_shuffle(panel, cols=[1, 3], seed=0)
    # Shape preserved
    assert shuffled.shape == panel.shape
    # Untouched columns identical
    assert torch.allclose(shuffled[:, :, 0], panel[:, :, 0])
    assert torch.allclose(shuffled[:, :, 2], panel[:, :, 2])
    assert torch.allclose(shuffled[:, :, 4], panel[:, :, 4])
    # Per-date marginal of touched cols preserved (sorted values match per date)
    for t in range(panel.shape[0]):
        for c in (1, 3):
            assert torch.allclose(
                torch.sort(shuffled[t, :, c]).values,
                torch.sort(panel[t, :, c]).values,
            )


def test_per_date_shuffle_deterministic_with_seed():
    panel = torch.randn(4, 20, 5)
    a = per_date_cross_section_shuffle(panel, [1, 3], seed=42)
    b = per_date_cross_section_shuffle(panel, [1, 3], seed=42)
    assert torch.equal(a, b)


def test_permutation_importance_identifies_planted_group():
    """Plant the signal in alpha_003 only — `alpha` group must dominate."""
    n_dates, n_stocks, n_factors = 30, 40, 8
    rng = torch.Generator().manual_seed(0)
    panel = torch.randn(n_dates, n_stocks, n_factors, generator=rng)
    weights = torch.zeros(n_factors)
    weights[3] = 5.0
    # Synth returns: depend on factor_3 of each stock, plus noise
    returns = (panel * weights).sum(dim=-1) + 0.01 * torch.randn(n_dates, n_stocks, generator=rng)
    factor_names = [f"alpha_{i:03d}" if i < 4 else f"gtja_{i:03d}" for i in range(n_factors)]
    score_fn = _linear_score_fn(weights)
    out = permutation_importance(
        score_fn, panel, returns, factor_names=factor_names,
        forward_period=2, top_k=5, n_seeds=3, base_seed=0,
    )
    assert "alpha" in out
    assert "gtja" in out
    # alpha group should have strictly larger ic_drop than gtja
    assert out["alpha"]["ic_drop_mean"] > out["gtja"]["ic_drop_mean"]
```

- [ ] **Step 2: Run, expect pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_factor_importance.py -v
```
Expected: 4 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/test_factor_importance.py
git commit -m "test(importance): IG planted weight + permutation marginal/determinism/group identification"
```

### Task 3.3: `eval_factor_importance.py` CLI

**Files:**
- Create: `scripts/eval_factor_importance.py`

- [ ] **Step 1: Write the CLI**

```python
#!/usr/bin/env python3
"""Post-training: load a PerStockEncoderPolicy, run IG + permutation
importance on the OOS panel, write runs/<id>/factor_importance.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

import numpy as np
import torch
from stable_baselines3 import PPO

from aurumq_rl.data_loader import FactorPanelLoader, UniverseFilter
from aurumq_rl.factor_importance import integrated_gradients, permutation_importance


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--data-path", required=True, type=Path)
    p.add_argument("--val-start", required=True)
    p.add_argument("--val-end", required=True)
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--forward-period", type=int, default=10)
    p.add_argument("--universe-filter", default="main_board_non_st")
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--ig-alpha-steps", type=int, default=50)
    p.add_argument("--ig-batch-size", type=int, default=8)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    meta = json.loads((args.run_dir / "metadata.json").read_text(encoding="utf-8"))
    n_factors = int(meta["factor_count"])

    loader = FactorPanelLoader(parquet_path=args.data_path)
    panel = loader.load_panel(
        start_date=dt.date.fromisoformat(args.val_start),
        end_date=dt.date.fromisoformat(args.val_end),
        n_factors=n_factors,
        forward_period=args.forward_period,
        universe_filter=UniverseFilter(args.universe_filter),
    )

    panel_t = torch.from_numpy(panel.factor_array).to("cuda")
    returns_t = torch.from_numpy(panel.return_array).to("cuda")

    final_model_path = next(args.run_dir.glob("*_final.zip"))
    model = PPO.load(str(final_model_path), device="cuda")
    model.policy.eval()

    def score_fn(obs: torch.Tensor) -> torch.Tensor:
        feats = model.policy.features_extractor(obs)
        return model.policy.action_net(feats["per_stock"]).squeeze(-1)

    # IG: take a stratified sample of dates for the batch
    sample_idx = np.linspace(0, panel_t.shape[0] - 1, args.ig_batch_size).astype(int)
    ig_batch = panel_t[sample_idx]   # (B, S, F)
    saliency = integrated_gradients(score_fn, ig_batch, n_alpha_steps=args.ig_alpha_steps).cpu().numpy()
    saliency_per_factor = {name: float(saliency[i]) for i, name in enumerate(panel.factor_names)}

    # Permutation
    perm_out = permutation_importance(
        score_fn=score_fn,
        val_panel=panel_t,
        val_returns=returns_t,
        factor_names=panel.factor_names,
        forward_period=args.forward_period,
        top_k=args.top_k,
        n_seeds=args.n_seeds,
    )

    # Aggregate saliency by prefix
    by_prefix: dict[str, list[float]] = {}
    for name, s in saliency_per_factor.items():
        prefix = name.split("_", 1)[0] if "_" in name else name
        by_prefix.setdefault(prefix, []).append(s)
    for prefix, drops in perm_out.items():
        sals = by_prefix.get(prefix, [])
        if sals:
            drops["saliency_mean"] = float(np.mean(sals))
            drops["saliency_max"] = float(np.max(sals))
            drops["saliency_std"] = float(np.std(sals, ddof=1) if len(sals) > 1 else 0.0)

    output = {
        "method": "integrated_gradients_v1+permutation_v1",
        "panel": str(args.data_path),
        "val_window": f"{args.val_start}..{args.val_end}",
        "saliency_per_factor": saliency_per_factor,
        "importance_per_group": perm_out,
    }
    out_path = args.run_dir / "factor_importance.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[importance] wrote {out_path}")
    print("[importance] top groups by ic_drop_mean:")
    ranked = sorted(perm_out.items(), key=lambda kv: -kv[1]["ic_drop_mean"])
    for prefix, m in ranked[:5]:
        print(f"  {prefix:8s}  ic_drop={m['ic_drop_mean']:+.4f}  n_factors={m['n_factors']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Verify import (cannot run without a real run dir, that's fine)**

```bash
.venv/Scripts/python.exe scripts/eval_factor_importance.py --help
```
Expected: argparse usage text printed.

- [ ] **Step 3: Commit and push**

```bash
git add scripts/eval_factor_importance.py
git commit -m "feat(importance): eval_factor_importance.py CLI writes factor_importance.json"
git push -u origin feat/gpu-framework-importance
```

agent-importance done.

---

## Phase 4: agent-web — `FactorImportancePanel`

**Worktree:** `D:/dev/aurumq-rl-wt-web` (branch `feat/gpu-framework-web`)

### Task 4.1: Add the shared types

**Files:**
- Modify: `web/lib/runs-shared.ts`

- [ ] **Step 1: Append to `web/lib/runs-shared.ts`**

```typescript
export interface FactorGroupImportance {
  n_factors: number;
  n_seeds: number;
  ic_drop_mean: number;
  ic_drop_std: number;
  sharpe_drop_mean: number;
  sharpe_drop_std: number;
  saliency_mean?: number;
  saliency_max?: number;
  saliency_std?: number;
}

export interface FactorImportance {
  method: string;
  panel: string;
  val_window: string;
  saliency_per_factor: Record<string, number>;
  importance_per_group: Record<string, FactorGroupImportance>;
}
```

- [ ] **Step 2: Verify TypeScript still type-checks**

```bash
cd /d/dev/aurumq-rl-wt-web/web && npx tsc --noEmit
```
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add web/lib/runs-shared.ts
git commit -m "feat(web): add FactorImportance / FactorGroupImportance shared types"
```

### Task 4.2: Server-side reader + API route

**Files:**
- Modify: `web/lib/runs.ts`
- Modify: `web/app/api/runs/[...id]/route.ts`

- [ ] **Step 1: Add `readFactorImportance` to `web/lib/runs.ts`** — append at the bottom, before any closing comment block:

```typescript
export async function readFactorImportance(id: string): Promise<unknown | null> {
  const p = path.join(RUNS_DIR, ...id.split("/"), "factor_importance.json");
  try {
    return JSON.parse(await fs.readFile(p, "utf-8"));
  } catch {
    return null;
  }
}
```

Re-export the type from runs-shared at the top of the file (where the other re-exports are):

```typescript
export type { FactorImportance, FactorGroupImportance } from "./runs-shared";
```

- [ ] **Step 2: Add `?part=factor-importance` branch to the API route** at `web/app/api/runs/[...id]/route.ts`. Locate the existing `if (part === "gpu")` branch and add a sibling:

```typescript
  if (part === "factor-importance") {
    return NextResponse.json(await readFactorImportance(decoded));
  }
```

(Add `readFactorImportance` to the import at top of the file.)

- [ ] **Step 3: Probe the route works** (start the dev server if not running, then):

```bash
curl -sS "http://localhost:3000/api/runs/ppo_smoke_r3_lr1e4?part=factor-importance" | head -3
```
Expected: `null` (because the json file doesn't exist for older runs) — the endpoint exists and doesn't 500.

- [ ] **Step 4: Commit**

```bash
git add web/lib/runs.ts web/app/api/runs/\[...id\]/route.ts
git commit -m "feat(web): readFactorImportance + ?part=factor-importance API branch"
```

### Task 4.3: `FactorImportancePanel` component

**Files:**
- Create: `web/components/FactorImportancePanel.tsx`

- [ ] **Step 1: Write the panel**

```typescript
"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { FactorImportance } from "@/lib/runs-shared";

const COLORS = [
  "#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899",
  "#14b8a6", "#a855f7", "#f97316", "#22c55e", "#0ea5e9", "#eab308",
];

export function FactorImportancePanel({ data }: { data: FactorImportance }) {
  const groupRows = Object.entries(data.importance_per_group)
    .map(([prefix, m]) => ({
      prefix,
      ic_drop_mean: m.ic_drop_mean,
      ic_drop_std: m.ic_drop_std,
      sharpe_drop_mean: m.sharpe_drop_mean,
      n_factors: m.n_factors,
      saliency_mean: m.saliency_mean ?? 0,
    }))
    .sort((a, b) => b.ic_drop_mean - a.ic_drop_mean);

  // Top 12 most-salient individual factors for the second chart
  const topFactors = Object.entries(data.saliency_per_factor)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 12)
    .map(([name, sal]) => ({ name, saliency: sal }));

  return (
    <section className="rounded-xl border border-zinc-200 dark:border-zinc-800 p-5 space-y-6">
      <header className="flex items-baseline justify-between">
        <h2 className="text-lg font-semibold">Factor importance</h2>
        <span className="text-xs text-zinc-500 font-mono">{data.method}</span>
      </header>

      <div className="min-w-0">
        <h3 className="text-xs text-zinc-500 mb-1">Per-group IC drop (permutation)</h3>
        <ResponsiveContainer width="100%" height={Math.max(180, groupRows.length * 24)}>
          <BarChart data={groupRows} layout="vertical" margin={{ top: 4, right: 12, bottom: 4, left: 32 }}>
            <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
            <XAxis type="number" tick={{ fontSize: 10 }} />
            <YAxis type="category" dataKey="prefix" tick={{ fontSize: 11, fontFamily: "monospace" }} width={64} />
            <Tooltip contentStyle={{ fontSize: 12 }} formatter={(v: number) => v.toFixed(4)} />
            <Bar dataKey="ic_drop_mean" isAnimationActive={false}>
              {groupRows.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="min-w-0">
        <h3 className="text-xs text-zinc-500 mb-1">Top-12 individual factor saliency (Integrated Gradients)</h3>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={topFactors} layout="vertical" margin={{ top: 4, right: 12, bottom: 4, left: 80 }}>
            <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
            <XAxis type="number" tick={{ fontSize: 10 }} />
            <YAxis type="category" dataKey="name" tick={{ fontSize: 10, fontFamily: "monospace" }} width={84} />
            <Tooltip contentStyle={{ fontSize: 12 }} formatter={(v: number) => v.toFixed(4)} />
            <Bar dataKey="saliency" fill="#10b981" isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <p className="text-xs text-zinc-500">
        IC drop = permutation-importance per-date cross-section shuffle (preserves time, breaks ranking).
        Saliency = Integrated Gradients average |∂score/∂factor| over a stratified panel sample.
      </p>
    </section>
  );
}
```

- [ ] **Step 2: Verify it type-checks**

```bash
cd /d/dev/aurumq-rl-wt-web/web && npx tsc --noEmit
```
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add web/components/FactorImportancePanel.tsx
git commit -m "feat(web): FactorImportancePanel (per-group IC drop bar + per-factor saliency bar)"
```

### Task 4.4: Wire into the run-detail page

**Files:**
- Modify: `web/app/runs/[...id]/page.tsx`

- [ ] **Step 1: Add the import + read + render** in `page.tsx`. Find the `Promise.all` block that already reads summary/metrics/etc. and add a new entry. Then render the panel below the existing GPU panel:

```typescript
import { FactorImportancePanel } from "@/components/FactorImportancePanel";
import type { FactorImportance } from "@/lib/runs-shared";

// in the Promise.all destructure, append readFactorImportance:
const [summary, metrics, backtest, series, gpu, fi, live, initialOffset] =
  await Promise.all([
    readSummary(decoded),
    readMetricsJsonl(decoded),
    readBacktest(decoded) as Promise<BacktestData | null>,
    readBacktestSeries(decoded) as Promise<BacktestSeriesData | null>,
    readGpuJsonl(decoded),
    readFactorImportance(decoded) as Promise<FactorImportance | null>,
    isRunLive(decoded),
    metricsJsonlSize(decoded),
  ]);

// in the JSX, after {gpu.length > 0 && <GpuMetricsPanel data={gpu} />}:
{fi != null && <FactorImportancePanel data={fi} />}
```

(Add `readFactorImportance` to the existing import from `@/lib/runs`.)

- [ ] **Step 2: Probe a run page in dev** (server should already be running):

```bash
curl -sS "http://localhost:3000/runs/ppo_smoke_r3_lr1e4" -o /dev/null -w "%{http_code}\n"
```
Expected: `200`.

- [ ] **Step 3: Commit + push**

```bash
git add web/app/runs/\[...id\]/page.tsx
git commit -m "feat(web): render FactorImportancePanel on run-detail page"
git push -u origin feat/gpu-framework-web
```

agent-web done.

---

## Phase 5: agent-train — integration in `train_v2.py`

**Worktree:** `main` (after env+policy+importance branches merge)
**Files created:** `scripts/train_v2.py`

### Task 5.1: Merge prerequisite branches

- [ ] **Step 1: Merge env / policy / importance branches into main**

```bash
cd /d/dev/aurumq-rl
git checkout main
git merge --no-ff feat/gpu-framework-env       -m "merge feat/gpu-framework-env"
git merge --no-ff feat/gpu-framework-policy    -m "merge feat/gpu-framework-policy"
git merge --no-ff feat/gpu-framework-importance -m "merge feat/gpu-framework-importance"
git merge --no-ff feat/gpu-framework-web       -m "merge feat/gpu-framework-web"
```

- [ ] **Step 2: Run all tests**

```bash
.venv/Scripts/python.exe -m pytest tests/test_gpu_env.py tests/test_policy.py tests/test_factor_importance.py -v
```
Expected: all green (≥ 13 passed).

### Task 5.2: Write `scripts/train_v2.py`

**Files:**
- Create: `scripts/train_v2.py`

- [ ] **Step 1: Write the trainer**

```python
#!/usr/bin/env python3
"""GPU-vectorised training entry. Wraps PPO + GPUStockPickingEnv +
PerStockEncoderPolicy. Loads panel once on cuda. Auto-runs factor
importance after training.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from aurumq_rl.data_loader import FactorPanelLoader, UniverseFilter
from aurumq_rl.gpu_env import GPUStockPickingEnv
from aurumq_rl.policy import PerStockEncoderPolicy


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--total-timesteps", type=int, required=True)
    p.add_argument("--data-path", type=Path, required=True)
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--universe-filter", default="main_board_non_st")
    p.add_argument("--n-factors", type=int, default=None,
                   help="default None = use all available factor cols")
    p.add_argument("--n-envs", type=int, default=12)
    p.add_argument("--episode-length", type=int, default=240)
    p.add_argument("--forward-period", type=int, default=10)
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--cost-bps", type=float, default=30.0)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--n-steps", type=int, default=1024)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--target-kl", type=float, default=0.20)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--encoder-hidden", default="128,64",
                   help="comma-separated layer sizes for the per-stock MLP hidden layers")
    p.add_argument("--encoder-out-dim", type=int, default=32)
    p.add_argument("--checkpoint-freq", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train_v2] loading panel from {args.data_path} ({args.start_date}..{args.end_date})...")
    loader = FactorPanelLoader(parquet_path=args.data_path)
    panel = loader.load_panel(
        start_date=dt.date.fromisoformat(args.start_date),
        end_date=dt.date.fromisoformat(args.end_date),
        n_factors=args.n_factors,
        forward_period=args.forward_period,
        universe_filter=UniverseFilter(args.universe_filter),
    )
    n_dates, n_stocks, n_factors = panel.factor_array.shape
    print(f"[train_v2] panel: dates={n_dates} stocks={n_stocks} factors={n_factors}")

    panel_t = torch.from_numpy(panel.factor_array).to("cuda")
    returns_t = torch.from_numpy(panel.return_array).to("cuda")
    valid_mask = (
        ~torch.from_numpy(panel.is_st_array).to("cuda")
        & ~torch.from_numpy(panel.is_suspended_array).to("cuda")
        & (torch.from_numpy(panel.days_since_ipo_array).to("cuda") >= 60)
    )

    env = GPUStockPickingEnv(
        panel_t, returns_t, valid_mask,
        n_envs=args.n_envs,
        episode_length=args.episode_length,
        forward_period=args.forward_period,
        top_k=args.top_k,
        cost_bps=args.cost_bps,
        seed=args.seed,
    )

    encoder_hidden = tuple(int(x) for x in args.encoder_hidden.split(","))
    policy_kwargs = dict(
        encoder_hidden=encoder_hidden,
        encoder_out_dim=args.encoder_out_dim,
    )

    model = PPO(
        policy=PerStockEncoderPolicy,
        env=env,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        target_kl=args.target_kl,
        max_grad_norm=args.max_grad_norm,
        verbose=1,
        seed=args.seed,
        device="cuda",
        policy_kwargs=policy_kwargs,
    )

    callbacks = []
    if args.checkpoint_freq > 0:
        cp_freq_per_env = max(args.checkpoint_freq // args.n_envs, 1)
        callbacks.append(CheckpointCallback(
            save_freq=cp_freq_per_env,
            save_path=str(args.out_dir / "checkpoints"),
            name_prefix="ppo",
        ))

    print(f"[train_v2] training for {args.total_timesteps:,} steps (n_envs={args.n_envs})...")
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks or None)

    final_path = args.out_dir / "ppo_final.zip"
    model.save(str(final_path))
    print(f"[train_v2] final model saved: {final_path}")

    metadata = {
        "algorithm": "PPO",
        "framework": "gpu_v2",
        "policy_class": "PerStockEncoderPolicy",
        "training_timesteps": args.total_timesteps,
        "n_envs": args.n_envs,
        "obs_shape": [n_stocks, n_factors],
        "action_shape": [n_stocks],
        "factor_count": n_factors,
        "stock_codes": panel.stock_codes,
        "factor_names": panel.factor_names,
        "train_start_date": args.start_date,
        "train_end_date": args.end_date,
        "universe": args.universe_filter,
        "encoder_hidden": list(encoder_hidden),
        "encoder_out_dim": args.encoder_out_dim,
        "top_k": args.top_k,
        "forward_period": args.forward_period,
    }
    (args.out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"[train_v2] metadata saved: {args.out_dir / 'metadata.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Verify import**

```bash
.venv/Scripts/python.exe scripts/train_v2.py --help
```
Expected: argparse usage printed.

- [ ] **Step 3: Commit**

```bash
git add scripts/train_v2.py
git commit -m "feat(train_v2): GPU-vectorised PPO training entry with PerStockEncoderPolicy"
```

### Task 5.3: 50k smoke run on combined SHORT panel

> **Phase 5 retro patch:** `n_steps=1024` was the original plan but causes
> SB3 `RolloutBuffer.reset()` to try allocating
> `(1024, 12, 3014, 343)` fp32 = **47 GiB host RAM**, which fails on
> a 64 GB box due to fragmentation. The fix is to shrink `n_steps` to
> 128 → buffer ≈ 6 GiB, fits comfortably. Per-PPO-update transitions
> drop from 12,288 to 1,536 — smaller but adequate for pipeline
> validation. If smoke OOS Sharpe is materially worse than the R3
> baseline (`+3.301 ± 20%`), the proper fix is GPURolloutBuffer
> (deferred plan), not bumping `n_steps` back up at this scale.

- [ ] **Step 1: Run the smoke**

```bash
mkdir -p runs/smoke_v2_50k
.venv/Scripts/python.exe scripts/train_v2.py \
    --total-timesteps 50000 \
    --data-path data/factor_panel_combined_short_2023_2026.parquet \
    --start-date 2023-01-03 --end-date 2025-06-30 \
    --universe-filter main_board_non_st \
    --n-envs 12 --episode-length 240 \
    --batch-size 512 --n-steps 128 --n-epochs 10 \
    --learning-rate 1e-4 --target-kl 0.20 --max-grad-norm 0.5 \
    --out-dir runs/smoke_v2_50k 2>&1 | tee runs/smoke_v2_50k.log
```

Expected: completes without crash; produces `runs/smoke_v2_50k/ppo_final.zip` and `metadata.json`.

- [ ] **Step 2: Measure fps from the log**

```bash
grep -E "fps " runs/smoke_v2_50k.log | tail -5
```

Acceptance: fps ≥ 500. If < 500 with this `n_steps=128` config, the
default RolloutBuffer's CPU↔GPU round-trip is the dominant cost.
Promote `GPURolloutBuffer` from "deferred" to next plan (cuda-resident
buffer, also forces n_steps ≤ ~160 due to 12 GB VRAM ceiling, but
eliminates the 50 GB-per-rollout PCIe traffic).

### Task 5.4: Backtest + factor importance on the smoke

- [ ] **Step 1: Run backtest**

```bash
.venv/Scripts/python.exe scripts/eval_backtest.py \
    --run-dir runs/smoke_v2_50k \
    --data-path data/factor_panel_combined_short_2023_2026.parquet \
    --val-start 2025-07-01 --val-end 2026-04-24 \
    --top-k 30 --universe-filter main_board_non_st 2>&1 | tail -10
```

Expected: writes `runs/smoke_v2_50k/backtest.json`. Logs IC, top30 Sharpe vs random p50.

- [ ] **Step 2: Run factor importance**

```bash
.venv/Scripts/python.exe scripts/eval_factor_importance.py \
    --run-dir runs/smoke_v2_50k \
    --data-path data/factor_panel_combined_short_2023_2026.parquet \
    --val-start 2025-07-01 --val-end 2026-04-24 \
    --top-k 30 --universe-filter main_board_non_st \
    --n-seeds 3 2>&1 | tail -20
```

Expected: writes `runs/smoke_v2_50k/factor_importance.json`. Logs top 5 groups by ic_drop_mean.

- [ ] **Step 3: Open the dashboard run page**

```bash
curl -sS "http://localhost:3000/runs/smoke_v2_50k" -o /dev/null -w "%{http_code}\n"
```
Expected: `200`. Open in browser to visually verify `FactorImportancePanel` renders.

- [ ] **Step 4: Sanity-check numbers**

Open `runs/smoke_v2_50k/backtest.json` and confirm:
- `top_k_sharpe` is finite
- `random_baseline.p50_sharpe` is also finite

The 50k smoke does not have to beat random p50 — pipeline correctness is the test, not metric quality. Acceptance per spec §12 #5: OOS Sharpe within ±20 % of R3's `+3.301`.

- [ ] **Step 5: Commit the smoke artifacts via run notes**

```bash
git add runs/smoke_v2_50k.log     # only the log; the run dir itself is gitignored
git commit -m "smoke(train_v2): 50k step e2e on combined SHORT panel — fps + IC + factor importance recorded"
```

### Task 5.5: Push and finalise

- [ ] **Step 1: Push main**

```bash
git push origin main
```

- [ ] **Step 2: Cleanup worktrees**

```bash
git worktree remove /d/dev/aurumq-rl-wt-env --force
git worktree remove /d/dev/aurumq-rl-wt-policy --force
git worktree remove /d/dev/aurumq-rl-wt-importance --force
git worktree remove /d/dev/aurumq-rl-wt-web --force
```

- [ ] **Step 3: Append to TRAINING_HISTORY.md**

Open `docs/TRAINING_HISTORY.md` and add a new "Phase 7 — GPU framework first smoke" entry under Section B, capturing the smoke fps, GPU util, OOS metrics, and any deviations from the design. Commit:

```bash
git add docs/TRAINING_HISTORY.md
git commit -m "docs(history): Phase 7 GPU framework first 50k smoke results"
git push origin main
```

Framework v1 is shippable. Next user decision: kick off the 5M overnight on `train_v2.py`.

---

## Verification

End-to-end checklist:

1. **Phase 0 fixture import** — `tests/_synthetic_panel.py::make_synthetic_panel()` returns a valid FactorPanel.
2. **Phase 1 env tests** — `pytest tests/test_gpu_env.py -v` → 5+ passed including `test_sb3_ppo_one_rollout`.
3. **Phase 2 policy tests** — `pytest tests/test_policy.py -v` → 9+ passed including bf16 finite + equivariance.
4. **Phase 3 importance tests** — `pytest tests/test_factor_importance.py -v` → 4 passed.
5. **Phase 4 web** — TypeScript `npx tsc --noEmit` clean; `/api/runs/<id>?part=factor-importance` returns valid JSON when file exists, `null` otherwise; dashboard renders `FactorImportancePanel` for runs with the file.
6. **Phase 5 integration** — `train_v2.py --total-timesteps 50000` completes; produces `ppo_final.zip` + `metadata.json`; `eval_backtest.py` works against the new metadata; `eval_factor_importance.py` writes `factor_importance.json`; dashboard renders all panels.

Acceptance criteria from spec §12 (must hold at end of Phase 5):
- All five agents' tests pass and merge cleanly. ✓
- Smoke E2E succeeds and writes `factor_importance.json`. ✓
- Smoke fps ≥ 500. (If not, escalate to GPURolloutBuffer.)
- GPU mean util ≥ 70 %. (Read from `gpu.jsonl`.)
- `FactorImportancePanel` renders. ✓
- OOS Sharpe within ±20 % of R3 `+3.301`.

---

## Out of scope (deferred plans)

- `GPURolloutBuffer` subclass (P1 if smoke fps < 500). Separate plan.
- LONG-panel training (needs fp16 panel or out-of-core streaming). Separate plan.
- Multi-GPU distributed training. Separate plan.
- Drop-one-group ablation matrix. Use IG + permutation results to pick top-3 candidates first; ablation can be a follow-up.
- Migration of legacy runs to the v2 metadata schema. Old runs continue to work with the legacy code path.
