# Phase 13 — PPO SGD Perf Probe

**Date:** 2026-05-01
**Branch:** `feat/phase13-perf-probe`
**Status:** Implementation

## Motivation

Phase 11 (bf16 + batch=2048) didn't help; Phase 12 (target_kl=0.10) caused
freeze. We are explicitly stopping blind hyperparameter tuning and instead
profiling what is actually slow inside PPO's SGD step. The goal of Phase 13
is to **locate** the bottleneck — not fix it. The fix is Phase 14.

## Scope (this phase)

Code-only instrumentation. **Does not change training semantics.** Adds:

1. `src/aurumq_rl/profiler_utils.py` — module-level perf-probe helpers:
   `cuda_stage()` context manager (with `cuda.synchronize()` bracketing for
   accurate timing), `_sgd_stage_times` dict, `print_sgd_stage_times`,
   `print_tensor_meta`, `diagnose_bottleneck`, `recommend_next_phase`.

2. `src/aurumq_rl/ppo_profiled.py` — `ProfiledPPO(PPO)` subclass that
   overrides `train()`. Identical to SB3's PPO when `profile_sgd=False`.
   When True, wraps each SGD stage in `cuda_stage(...)`:
   - `batch_get_or_index`
   - `obs_materialize_or_gather`
   - `cpu_to_gpu_copy`
   - `contiguous_or_clone`
   - `forward_eval_actions`
   - `loss_build`
   - `zero_grad`
   - `backward`
   - `optimizer_step`
   - `cuda_tail_sync`

3. `scripts/train_v2.py` — adds CLI flags:
   - `--run-name`
   - `--profile-sgd`
   - `--profile-sgd-minibatches` (default 20)
   - `--profile-torch-profiler` (default 0)
   - `--profile-memory`
   - `--profile-print-every` (default 10)
   - `--profile-output-dir` (default `<out_dir>/profiler`)

4. `tests/test_profiler_utils.py` — 5–6 unit tests for the helpers.

## Heuristics for `diagnose_bottleneck`

Each flag is independent (multiple may fire). Conservative thresholds: an
event must contribute ≥10% of iter time (or appear in profiler top-10) to
be flagged.

- `cpu_to_gpu_copy_bottleneck`: `cpu_to_gpu_copy` mean ≥ 5% of iter, OR
  profiler top-10 contains `aten::to` / `aten::copy_` / `cudaMemcpyAsync`.
- `gather_index_bottleneck`: `contiguous_or_clone` + `obs_materialize_or_gather`
  ≥ 10% of iter, OR profiler top-10 contains `aten::index` /
  `aten::index_select` / `aten::gather`.
- `forward_dominated`: `forward_eval_actions` ≥ 40% of iter.
- `backward_dominated`: `backward` ≥ 40% of iter.
- `optimizer_dominated`: `optimizer_step` ≥ 15% of iter.
- `python_overhead_dominated`: `cuda_tail_sync` < 5% AND no single stage
  ≥ 30%; suggests Python/autograd/launch overhead.

## Recommendations from `recommend_next_phase`

- CPU→GPU copy: Phase 14 — GPU-resident rollout/index buffer.
- Gather/index: Phase 14A — sorted/block-shuffled minibatch.
- Date duplication: Phase 14B — unique-date market encoding.
- Linear/GEMM: Phase 14C — TF32 only.
- Python overhead: Phase 14D — torch.compile encoder.

## Out of scope

- Running the smoke (controller does that).
- Merging to main.
- Any change to PPO hyperparameters.
- Any change to existing tests.

## Acceptance

- All existing tests pass (33).
- 5–6 new tests for profiler_utils pass.
- `train_v2.py --help` shows new flags.
- Branch pushed to origin as `feat/phase13-perf-probe`.
