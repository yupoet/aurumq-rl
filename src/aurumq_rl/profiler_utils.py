"""Centralised perf-probe helpers for PPO SGD profiling (Phase 13).

This module is consumed by both ``ppo_profiled.ProfiledPPO`` (which wraps
each SGD stage in :func:`cuda_stage`) and ``scripts/train_v2.py`` (which
prints/saves the summary at the end of training).

Design notes:

- Stage timings live in a module-level :class:`collections.defaultdict`
  keyed by stage name. We use module-level state because (a) the
  ``train()`` body inside SB3's PPO is hard to thread instance state
  through cleanly, and (b) only one PPO trainer instance is alive at a
  time inside our process.
- Every :func:`cuda_stage` call brackets the timed region with
  ``torch.cuda.synchronize()``. This is **non-negotiable** for accurate
  cuda timing — without it, async kernel queueing makes ``backward()``
  appear nearly free while the actual work is still in flight.
- The helpers are deliberately dependency-free (only ``torch`` and the
  stdlib) so they can be imported even on CPU-only test boxes; they
  silently no-op the synchronize calls when ``cuda.is_available()`` is
  False.

This module does not change PPO training semantics. It only adds
instrumentation. The caller decides whether to enable it via the
``enabled`` flag on :func:`cuda_stage` (see ``ProfiledPPO.profile_sgd``).
"""
from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any

import torch

# Module-level dict, keyed by stage name → list of millisecond floats. We
# use a defaultdict so the first ``append`` creates the bucket cleanly
# without needing to pre-register every stage name.
_sgd_stage_times: dict[str, list[float]] = defaultdict(list)


def reset_sgd_stage_times() -> None:
    """Clear all recorded stage timings. Call at the start of a fresh run."""
    _sgd_stage_times.clear()


@contextmanager
def cuda_stage(name: str, enabled: bool = True):
    """Context manager that times a stage with cuda.synchronize() bracketing.

    The synchronize() calls are CRITICAL for accurate cuda timing — without
    them, a fast-launching async kernel queue would record near-zero time
    for backward() while the actual work is still in flight.

    When ``enabled`` is False the context is a no-op and adds no entry to
    the timings dict, so the production hot path is unaffected.

    Args:
        name: Stable identifier for this stage (e.g. ``"backward"``).
        enabled: When False, skip timing entirely (fast path).
    """
    if not enabled:
        yield
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _sgd_stage_times[name].append((time.perf_counter() - t0) * 1000.0)


def print_sgd_stage_times(prefix: str = "  ", last_n: int = 20) -> None:
    """Print mean/last ms for each recorded stage over last_n samples.

    Stable order: prints in insertion order (Python 3.7+ dict semantics),
    which matches the order the SGD loop wraps stages.

    Args:
        prefix: String prepended to each line (for indenting in summaries).
        last_n: Number of most-recent samples to average. Older entries
            are kept in the dict but not included in this view.
    """
    if not _sgd_stage_times:
        return
    print(f"{prefix}=== PPO SGD stage timing (last_n={last_n}) ===")
    for name, values in _sgd_stage_times.items():
        if not values:
            continue
        xs = values[-last_n:]
        mean_ms = sum(xs) / len(xs)
        last_ms = xs[-1]
        total_ms = sum(values)
        print(
            f"{prefix}{name:30s} mean={mean_ms:7.2f} ms  "
            f"last={last_ms:7.2f} ms  n={len(values):4d}  "
            f"total={total_ms / 1000:6.2f} s"
        )


def get_sgd_stage_times() -> dict[str, list[float]]:
    """Return a *copy* of the recorded times for downstream summarising."""
    return {k: list(v) for k, v in _sgd_stage_times.items()}


def print_tensor_meta(name: str, x: Any) -> None:
    """One-shot tensor metadata dump used at the top of the run.

    Prints shape, dtype, device, contiguity, stride, numel, and bytes.
    Falls back to a friendly "not a tensor" line for non-tensor inputs
    so the caller doesn't have to type-guard each access.
    """
    if not isinstance(x, torch.Tensor):
        print(f"  {name}: not a tensor, type={type(x).__name__}")
        return
    print(
        f"  {name}.shape={tuple(x.shape)}  dtype={x.dtype}  "
        f"device={x.device}  contig={x.is_contiguous()}  "
        f"stride={x.stride()}  numel={x.numel():,}  "
        f"bytes={x.numel() * x.element_size() / 1024 ** 3:.4f} GiB"
    )


def _stage_mean(stage_times: dict[str, list[float]], name: str) -> float:
    """Return the mean ms of stage ``name``, or 0.0 if absent/empty."""
    xs = stage_times.get(name) or []
    if not xs:
        return 0.0
    return sum(xs) / len(xs)


def _profiler_top_names_lower(
    profiler_top_ops_cuda: list[tuple[str, float]] | None,
    top_k: int = 10,
) -> list[str]:
    """Lowercased op names from the top-k of the profiler output.

    Used by the heuristic flags so we don't have to repeat the
    case-folding logic in each branch.
    """
    if not profiler_top_ops_cuda:
        return []
    return [name.lower() for name, _ in profiler_top_ops_cuda[:top_k]]


def diagnose_bottleneck(
    stage_times: dict[str, list[float]],
    profiler_top_ops_cuda: list[tuple[str, float]] | None = None,
) -> dict[str, Any]:
    """Heuristic bottleneck classifier reading from stage timings + (optional)
    torch.profiler top ops. Returns a structured dict the caller renders.

    Heuristics (each independent — multiple can be True):

    - ``cpu_to_gpu_copy_bottleneck``:
      cpu_to_gpu_copy mean ms > 5% of total iter ms, OR profiler shows
      ``aten::to`` / ``aten::copy_`` / ``cudamemcpyasync`` in top 10 self_cuda.
    - ``gather_index_bottleneck``:
      ``contiguous_or_clone`` OR ``obs_materialize_or_gather`` > 10% of iter,
      OR ``aten::index`` / ``index_select`` / ``gather`` in profiler top 10.
    - ``forward_dominated``:
      ``forward_eval_actions`` > 40% of iter ms.
    - ``backward_dominated``:
      ``backward`` > 40% of iter ms.
    - ``optimizer_dominated``:
      ``optimizer_step`` > 15% of iter ms (would be unusual; flag).
    - ``python_overhead_dominated``:
      ``cuda_tail_sync`` < 5% AND no single stage > 30%; suggests Python /
      autograd / kernel launch overhead is the killer.

    The returned dict has one boolean per heuristic, plus an ``evidence``
    sub-dict with the supporting numbers (so the caller can render a
    human-readable summary without re-deriving the percentages).
    """
    # Empty input → safe default with all flags False. The caller (or
    # the recommender) is responsible for handling this case gracefully.
    if not stage_times:
        return {
            "cpu_to_gpu_copy_bottleneck": False,
            "gather_index_bottleneck": False,
            "forward_dominated": False,
            "backward_dominated": False,
            "optimizer_dominated": False,
            "python_overhead_dominated": False,
            "evidence": {
                "iter_total_ms": 0.0,
                "stage_pct": {},
                "profiler_top_names": [],
                "note": "no stage timings recorded",
            },
        }

    # We treat "iter_total" as the sum of stage means. This is an
    # approximation (overlapping cuda kernels can inflate it slightly),
    # but it's stable across runs and good enough for flagging.
    means = {name: _stage_mean(stage_times, name) for name in stage_times}
    iter_total_ms = sum(means.values())
    if iter_total_ms <= 0:
        # Defensive: empty samples or NaN floats. Return safe default.
        return {
            "cpu_to_gpu_copy_bottleneck": False,
            "gather_index_bottleneck": False,
            "forward_dominated": False,
            "backward_dominated": False,
            "optimizer_dominated": False,
            "python_overhead_dominated": False,
            "evidence": {
                "iter_total_ms": 0.0,
                "stage_pct": {},
                "profiler_top_names": [],
                "note": "iter_total_ms == 0; nothing to diagnose",
            },
        }

    pct = {name: 100.0 * m / iter_total_ms for name, m in means.items()}
    profiler_names = _profiler_top_names_lower(profiler_top_ops_cuda, top_k=10)

    def any_in_profiler(needles: tuple[str, ...]) -> bool:
        return any(any(needle in n for needle in needles) for n in profiler_names)

    cpu_gpu_pct = pct.get("cpu_to_gpu_copy", 0.0)
    cpu_gpu_in_profiler = any_in_profiler(
        ("aten::to", "aten::copy_", "cudamemcpyasync")
    )
    cpu_to_gpu_copy_bottleneck = (cpu_gpu_pct > 5.0) or cpu_gpu_in_profiler

    gather_pct = pct.get("contiguous_or_clone", 0.0) + pct.get(
        "obs_materialize_or_gather", 0.0
    )
    gather_in_profiler = any_in_profiler(
        ("aten::index", "aten::index_select", "aten::gather")
    )
    gather_index_bottleneck = (gather_pct > 10.0) or gather_in_profiler

    forward_dominated = pct.get("forward_eval_actions", 0.0) > 40.0
    backward_dominated = pct.get("backward", 0.0) > 40.0
    optimizer_dominated = pct.get("optimizer_step", 0.0) > 15.0

    cuda_tail_pct = pct.get("cuda_tail_sync", 0.0)
    largest_pct = max(pct.values()) if pct else 0.0
    python_overhead_dominated = (cuda_tail_pct < 5.0) and (largest_pct < 30.0)

    return {
        "cpu_to_gpu_copy_bottleneck": cpu_to_gpu_copy_bottleneck,
        "gather_index_bottleneck": gather_index_bottleneck,
        "forward_dominated": forward_dominated,
        "backward_dominated": backward_dominated,
        "optimizer_dominated": optimizer_dominated,
        "python_overhead_dominated": python_overhead_dominated,
        "evidence": {
            "iter_total_ms": iter_total_ms,
            "stage_pct": pct,
            "profiler_top_names": profiler_names,
            "cpu_gpu_in_profiler": cpu_gpu_in_profiler,
            "gather_in_profiler": gather_in_profiler,
            "largest_stage_pct": largest_pct,
            "cuda_tail_pct": cuda_tail_pct,
        },
    }


def recommend_next_phase(diag: dict[str, Any]) -> str:
    """Return a single string recommending Phase 14 direction based on diag.

    The recommendations match the user's spec:

    - Case 1 (CPU→GPU copy): "Phase 14 = GPU-resident rollout/index buffer"
    - Case 2 (gather/index): "Phase 14A = sorted/block-shuffled minibatch"
    - Case 3 (date dup): "Phase 14B = unique-date market encoding"
    - Case 5 (Linear/GEMM): "Phase 14C = TF32 only"
    - Case 6 (Python overhead): "Phase 14D = torch.compile encoder"

    The caller passes diag plus duplicate_factor info. Returns the most
    likely top recommendation as a string. If multiple cases apply, lists
    the top 2 in priority order.

    Note: case 3 (date duplication) cannot be inferred from
    :func:`diagnose_bottleneck` alone; it requires the date-dup factor
    measured inside the SGD loop. The caller can mention it separately
    if observed; here we surface only the heuristics derivable from the
    stage timings.
    """
    if not diag:
        return (
            "No diagnosis available (empty stage timings). "
            "Re-run with --profile-sgd and --profile-sgd-minibatches >= 10."
        )

    evidence = diag.get("evidence", {})
    note = evidence.get("note")
    if note:
        return f"No diagnosis available ({note}). Re-run --profile-sgd."

    # Build (priority, case_string) pairs. Lower priority = stronger signal.
    candidates: list[tuple[int, str]] = []

    if diag.get("cpu_to_gpu_copy_bottleneck"):
        candidates.append((
            1,
            "Phase 14 = GPU-resident rollout/index buffer "
            "(cpu_to_gpu_copy stage or aten::to/aten::copy_ dominates).",
        ))
    if diag.get("gather_index_bottleneck"):
        candidates.append((
            2,
            "Phase 14A = sorted/block-shuffled minibatch "
            "(panel gather / index_select / index dominates).",
        ))
    if diag.get("forward_dominated"):
        candidates.append((
            3,
            "Phase 14C = TF32 only / Linear-fast-path "
            "(forward_eval_actions > 40% of iter — Linear/GEMM-bound).",
        ))
    if diag.get("backward_dominated"):
        candidates.append((
            4,
            "Phase 14C = TF32 only / Linear-fast-path "
            "(backward > 40% of iter — autograd is GEMM-bound).",
        ))
    if diag.get("optimizer_dominated"):
        candidates.append((
            5,
            "Investigate optimizer.step() (Adam moment update is "
            ">15% of iter — unusual; check parameter count).",
        ))
    if diag.get("python_overhead_dominated"):
        candidates.append((
            6,
            "Phase 14D = torch.compile encoder "
            "(no single stage dominates; Python / kernel launch overhead).",
        ))

    if not candidates:
        return (
            "No clear bottleneck flagged by heuristics. "
            "Inspect raw stage_times percentages and the chrome trace "
            "(profile_output_dir/sgd_trace.json) for a manual diagnosis."
        )

    candidates.sort(key=lambda x: x[0])
    if len(candidates) == 1:
        return candidates[0][1]
    # Top 2 in priority order
    top = candidates[:2]
    return (
        f"Top candidate: {top[0][1]} "
        f"Secondary: {top[1][1]}"
    )
