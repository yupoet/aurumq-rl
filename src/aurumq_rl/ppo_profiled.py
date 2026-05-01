"""ProfiledPPO — instrumented subclass of SB3's PPO (Phase 13).

Wraps each meaningful stage of the SGD step in
:func:`aurumq_rl.profiler_utils.cuda_stage`. When ``profile_sgd`` is
False, behaviour is **identical** to plain ``stable_baselines3.PPO``.

Why a subclass instead of a hook? SB3's ``PPO.train()`` does not provide
hook points; the cleanest way to insert per-stage timing is to copy the
loop body and add ``cuda_stage(...)`` brackets. This is a faithful
mechanical translation of SB3 v2's ``train()`` (commit shipped with the
.venv) — the only behavioural change is that we may early-exit the
profiler instrumentation after ``profile_sgd_minibatches`` minibatches,
but the underlying training keeps running normally.

The profiler instrumentation adds a few hundred microseconds of overhead
per minibatch (the synchronize() calls). For a 20-minibatch profile this
is negligible. We do not leave the synchronize() calls on for the entire
run, because that would bias the reported fps.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance
from torch.nn import functional as F
from torch.profiler import ProfilerActivity, profile, record_function

from aurumq_rl.profiler_utils import (
    cuda_stage,
    print_sgd_stage_times,
    print_tensor_meta,
    reset_sgd_stage_times,
)


class ProfiledPPO(PPO):
    """PPO subclass that times each SGD stage with cuda.synchronize() bracketing.

    Extra kwargs (all ignored when ``profile_sgd=False``):

    Args:
        profile_sgd: Master switch. When False this class is
            behaviourally identical to ``stable_baselines3.PPO``.
        profile_sgd_minibatches: Number of minibatches to instrument
            with stage timers. After this many, instrumentation stops
            but training continues normally.
        profile_torch_profiler_n: If > 0, additionally wrap the first N
            minibatches in ``torch.profiler.profile`` and dump tables +
            chrome trace to ``profile_output_dir``.
        profile_memory: Forwarded to torch.profiler. Off by default.
        profile_print_every: Print running stage summary every N
            minibatches.
        profile_output_dir: Where to save profiler outputs (chrome trace,
            top-ops tables). If None, profiler outputs are skipped even
            if ``profile_torch_profiler_n > 0``.
    """

    def __init__(
        self,
        *args: Any,
        profile_sgd: bool = False,
        profile_sgd_minibatches: int = 20,
        profile_torch_profiler_n: int = 0,
        profile_memory: bool = False,
        profile_print_every: int = 10,
        profile_output_dir: Path | None = None,
        **kwargs: Any,
    ) -> None:
        # Strip profile_* kwargs from the SB3 super().__init__ call —
        # PPO.__init__ would raise TypeError for unknown kwargs otherwise.
        super().__init__(*args, **kwargs)
        self._profile_sgd = bool(profile_sgd)
        self._profile_sgd_minibatches = int(profile_sgd_minibatches)
        self._profile_torch_profiler_n = int(profile_torch_profiler_n)
        self._profile_memory = bool(profile_memory)
        self._profile_print_every = max(1, int(profile_print_every))
        self._profile_output_dir = (
            Path(profile_output_dir) if profile_output_dir is not None else None
        )
        self._profile_sgd_seen = 0
        # Capture the profiler-derived "top ops" tables on first profile;
        # train_v2.py reads these out for the perf_summary.json.
        self._profile_top_cuda: list[tuple[str, float]] = []
        self._profile_top_cpu: list[tuple[str, float]] = []
        self._profile_top_cuda_mem: list[tuple[str, int]] = []

    # ------------------------------------------------------------------
    # train() — copied from SB3 PPO.train() and wrapped with cuda_stage()
    # ------------------------------------------------------------------

    def train(self) -> None:  # noqa: C901 (mirrors SB3's structure)
        """Update policy using the currently gathered rollout buffer.

        This is a faithful copy of ``stable_baselines3.PPO.train()`` with
        :func:`cuda_stage` brackets added around each meaningful stage.
        When ``self._profile_sgd`` is False the brackets are no-ops.
        """
        if self._profile_sgd:
            # Fresh profile every call to train(); the caller can also
            # call reset_sgd_stage_times() externally but doing it here
            # makes ProfiledPPO self-contained.
            reset_sgd_stage_times()
            self._profile_sgd_seen = 0

        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses: list[float] = []
        pg_losses: list[float] = []
        value_losses: list[float] = []
        clip_fractions: list[float] = []

        continue_training = True

        # Optional torch.profiler over the first N minibatches.
        prof_ctx = None
        if (
            self._profile_sgd
            and self._profile_torch_profiler_n > 0
            and self._profile_output_dir is not None
        ):
            activities = [ProfilerActivity.CPU]
            if th.cuda.is_available():
                activities.append(ProfilerActivity.CUDA)
            prof_ctx = profile(
                activities=activities,
                record_shapes=True,
                profile_memory=self._profile_memory,
                with_stack=True,
            )
            prof_ctx.__enter__()

        try:
            for epoch in range(self.n_epochs):
                approx_kl_divs: list[float] = []

                # We need to time the generator's __next__ explicitly,
                # not just the body. A naive ``for rollout_data in
                # buffer.get(...):`` hides the per-batch indexing cost
                # in the loop control. Drive the generator manually.
                gen = self.rollout_buffer.get(self.batch_size)
                while True:
                    enabled = (
                        self._profile_sgd
                        and self._profile_sgd_seen < self._profile_sgd_minibatches
                    )

                    # ---- batch_get_or_index ---------------------------
                    with cuda_stage("batch_get_or_index", enabled):
                        with record_function(
                            "ppo_sgd_minibatch_stage_batch_get_or_index"
                        ):
                            try:
                                rollout_data = next(gen)
                            except StopIteration:
                                rollout_data = None
                    if rollout_data is None:
                        break

                    # One-shot tensor metadata dump on the very first
                    # minibatch we instrument. Helps confirm dtype / device
                    # / contiguity assumptions before reading the timing.
                    if enabled and self._profile_sgd_seen == 0:
                        print("[profile] tensor metadata (first minibatch):")
                        print_tensor_meta(
                            "rollout_data.observations", rollout_data.observations
                        )
                        print_tensor_meta(
                            "rollout_data.actions", rollout_data.actions
                        )
                        print_tensor_meta(
                            "rollout_data.advantages", rollout_data.advantages
                        )
                        print_tensor_meta(
                            "rollout_data.old_log_prob", rollout_data.old_log_prob
                        )
                        print_tensor_meta(
                            "rollout_data.old_values", rollout_data.old_values
                        )
                        buf = self.rollout_buffer
                        if hasattr(buf, "t_buffer"):
                            print_tensor_meta(
                                "rollout_buffer.t_buffer", buf.t_buffer
                            )
                        print_tensor_meta(
                            "rollout_buffer.actions", buf.actions
                        )
                        # Try to pull the panel reference out of the
                        # IndexOnlyRolloutBuffer's obs_provider closure.
                        try:
                            provider = getattr(buf, "_obs_provider", None)
                            if provider is not None and hasattr(provider, "__closure__"):
                                cell = provider.__closure__[0]  # type: ignore[index]
                                panel_ref = cell.cell_contents
                                # The closure may capture either the env
                                # or the panel directly. Try both.
                                if isinstance(panel_ref, th.Tensor):
                                    print_tensor_meta(
                                        "obs_provider.panel", panel_ref
                                    )
                                elif hasattr(panel_ref, "panel"):
                                    print_tensor_meta(
                                        "obs_provider.env.panel", panel_ref.panel
                                    )
                                else:
                                    print(
                                        "  obs_provider closure: "
                                        f"type={type(panel_ref).__name__} "
                                        "(no panel tensor surfaced)"
                                    )
                        except Exception as exc:  # noqa: BLE001
                            print(f"  obs_provider closure: introspection failed ({exc})")

                    # Approximate date-duplication check — sample bs
                    # random t-indices from the buffer's t_buffer (this
                    # is approximate; recovering the exact slice would
                    # require subclassing _get_samples).
                    if (
                        enabled
                        and self._profile_sgd_seen < 3
                        and hasattr(self.rollout_buffer, "t_buffer")
                    ):
                        buffer = self.rollout_buffer
                        t_buffer_flat = (
                            buffer.t_buffer
                            if buffer.t_buffer.dim() == 2
                            else buffer.t_buffer.view(-1)
                        )
                        bs = rollout_data.observations.shape[0]
                        sample_idx = th.randperm(
                            t_buffer_flat.numel(), device=t_buffer_flat.device
                        )[:bs]
                        t_idx = t_buffer_flat.view(-1)[sample_idx]
                        unique_dates = th.unique(t_idx).numel()
                        dup_factor = bs / max(unique_dates, 1)
                        print(
                            f"  [date-dup] minibatch sample bs={bs} "
                            f"unique_dates={unique_dates} "
                            f"dup_factor={dup_factor:.2f}"
                        )

                    # ---- obs_materialize_or_gather --------------------
                    # For IndexOnlyRolloutBuffer this is where panel[t]
                    # would fire — but in our buffer the gather happens
                    # inside _get_samples (already counted under
                    # batch_get_or_index). We still time observation
                    # access here as a sanity check; if .observations is
                    # already a fully-materialised tensor this stage will
                    # be ~free. The two combined tell us whether the
                    # gather is the bottleneck.
                    with cuda_stage("obs_materialize_or_gather", enabled):
                        with record_function(
                            "ppo_sgd_minibatch_stage_obs_materialize_or_gather"
                        ):
                            obs_for_forward = rollout_data.observations

                    # ---- cpu_to_gpu_copy ------------------------------
                    # Most cleanly: any time obs / actions / advantages
                    # need a host->device hop, it would happen here. With
                    # GPU rollout buffers this should be near-zero; with
                    # SB3's default buffer it would be the dominant cost.
                    actions = rollout_data.actions
                    with cuda_stage("cpu_to_gpu_copy", enabled):
                        with record_function(
                            "ppo_sgd_minibatch_stage_cpu_to_gpu_copy"
                        ):
                            if (
                                hasattr(obs_for_forward, "device")
                                and obs_for_forward.device != self.device
                            ):
                                obs_for_forward = obs_for_forward.to(self.device)
                            if (
                                hasattr(actions, "device")
                                and actions.device != self.device
                            ):
                                actions = actions.to(self.device)

                    if isinstance(self.action_space, spaces.Discrete):
                        actions = actions.long().flatten()

                    # ---- contiguous_or_clone --------------------------
                    # Empty in our hot path (we don't call .contiguous()
                    # / .clone() inside SGD), but kept as a non-noisy
                    # marker so the percentage shows up at ~0% as
                    # evidence rather than as silently absent.
                    with cuda_stage("contiguous_or_clone", enabled):
                        with record_function(
                            "ppo_sgd_minibatch_stage_contiguous_or_clone"
                        ):
                            pass

                    # ---- forward_eval_actions -------------------------
                    with cuda_stage("forward_eval_actions", enabled):
                        with record_function(
                            "ppo_sgd_minibatch_stage_forward_eval_actions"
                        ):
                            values, log_prob, entropy = self.policy.evaluate_actions(
                                obs_for_forward, actions
                            )
                    values = values.flatten()

                    advantages = rollout_data.advantages
                    if self.normalize_advantage and len(advantages) > 1:
                        advantages = (advantages - advantages.mean()) / (
                            advantages.std() + 1e-8
                        )

                    # ---- loss_build -----------------------------------
                    with cuda_stage("loss_build", enabled):
                        with record_function(
                            "ppo_sgd_minibatch_stage_loss_build"
                        ):
                            ratio = th.exp(log_prob - rollout_data.old_log_prob)
                            policy_loss_1 = advantages * ratio
                            policy_loss_2 = advantages * th.clamp(
                                ratio, 1 - clip_range, 1 + clip_range
                            )
                            policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                            pg_losses.append(policy_loss.item())
                            clip_fraction = th.mean(
                                (th.abs(ratio - 1) > clip_range).float()
                            ).item()
                            clip_fractions.append(clip_fraction)

                            if self.clip_range_vf is None:
                                values_pred = values
                            else:
                                values_pred = rollout_data.old_values + th.clamp(
                                    values - rollout_data.old_values,
                                    -clip_range_vf,
                                    clip_range_vf,
                                )
                            value_loss = F.mse_loss(rollout_data.returns, values_pred)
                            value_losses.append(value_loss.item())

                            if entropy is None:
                                entropy_loss = -th.mean(-log_prob)
                            else:
                                entropy_loss = -th.mean(entropy)
                            entropy_losses.append(entropy_loss.item())

                            loss = (
                                policy_loss
                                + self.ent_coef * entropy_loss
                                + self.vf_coef * value_loss
                            )

                    # KL early-stop check — outside the timed loss block
                    # because it's a logging-only no_grad pass.
                    with th.no_grad():
                        log_ratio = log_prob - rollout_data.old_log_prob
                        approx_kl_div = (
                            th.mean((th.exp(log_ratio) - 1) - log_ratio)
                            .cpu()
                            .numpy()
                        )
                        approx_kl_divs.append(approx_kl_div)

                    if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                        continue_training = False
                        if self.verbose >= 1:
                            print(
                                f"Early stopping at step {epoch} due to reaching "
                                f"max kl: {approx_kl_div:.2f}"
                            )
                        break

                    # ---- zero_grad ------------------------------------
                    with cuda_stage("zero_grad", enabled):
                        with record_function(
                            "ppo_sgd_minibatch_stage_zero_grad"
                        ):
                            self.policy.optimizer.zero_grad()

                    # ---- backward -------------------------------------
                    with cuda_stage("backward", enabled):
                        with record_function(
                            "ppo_sgd_minibatch_stage_backward"
                        ):
                            loss.backward()

                    # ---- optimizer_step -------------------------------
                    with cuda_stage("optimizer_step", enabled):
                        with record_function(
                            "ppo_sgd_minibatch_stage_optimizer_step"
                        ):
                            th.nn.utils.clip_grad_norm_(
                                self.policy.parameters(), self.max_grad_norm
                            )
                            self.policy.optimizer.step()

                    # ---- cuda_tail_sync -------------------------------
                    # Catches any straggler kernels that were queued but
                    # not finished by the previous synchronize() calls.
                    # On a healthy run this should be near-zero; if it's
                    # large, something is queueing async work between
                    # the timed stages.
                    with cuda_stage("cuda_tail_sync", enabled):
                        with record_function(
                            "ppo_sgd_minibatch_stage_cuda_tail_sync"
                        ):
                            if th.cuda.is_available():
                                th.cuda.synchronize()

                    if enabled:
                        self._profile_sgd_seen += 1
                        if self._profile_sgd_seen % self._profile_print_every == 0:
                            print_sgd_stage_times(
                                prefix="  [profile] ",
                                last_n=self._profile_print_every,
                            )

                self._n_updates += 1
                if not continue_training:
                    break
        finally:
            if prof_ctx is not None:
                prof_ctx.__exit__(None, None, None)
                self._dump_profiler(prof_ctx)

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten(),
        )

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())  # type: ignore[possibly-undefined]
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)  # type: ignore[possibly-undefined]

    # ------------------------------------------------------------------
    # Profiler dump helpers
    # ------------------------------------------------------------------

    def _dump_profiler(self, prof_ctx: profile) -> None:
        """Save chrome trace and capture top-N tables for later JSON dump."""
        if self._profile_output_dir is None:
            return
        out = self._profile_output_dir
        out.mkdir(parents=True, exist_ok=True)
        try:
            trace_path = out / "sgd_trace.json"
            prof_ctx.export_chrome_trace(str(trace_path))
            print(f"[profile] chrome trace saved: {trace_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"[profile] chrome trace export failed: {exc}")

        # Top tables — store on self for train_v2.py to read out and
        # round-trip into perf_summary.json. Use try/except per table so
        # that a torch version mismatch on memory tables doesn't kill
        # the cuda/cpu tables.
        try:
            self._profile_top_cuda = self._key_avg_to_pairs(
                prof_ctx, sort_by="self_cuda_time_total"
            )
            print(
                "[profile] captured top "
                f"{len(self._profile_top_cuda)} cuda ops"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[profile] cuda top-table capture failed: {exc}")
        try:
            self._profile_top_cpu = self._key_avg_to_pairs(
                prof_ctx, sort_by="self_cpu_time_total"
            )
            print(
                "[profile] captured top "
                f"{len(self._profile_top_cpu)} cpu ops"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[profile] cpu top-table capture failed: {exc}")
        if self._profile_memory:
            try:
                self._profile_top_cuda_mem = self._key_avg_to_pairs_int(
                    prof_ctx, sort_by="self_cuda_memory_usage"
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[profile] cuda mem top-table capture failed: {exc}")

        # Plain-text dumps for human reading
        try:
            txt_path = out / "sgd_top_ops.txt"
            with txt_path.open("w", encoding="utf-8") as fh:
                fh.write("=== Top-40 self_cuda_time_total ===\n")
                fh.write(
                    prof_ctx.key_averages().table(
                        sort_by="self_cuda_time_total", row_limit=40
                    )
                )
                fh.write("\n\n=== Top-40 self_cpu_time_total ===\n")
                fh.write(
                    prof_ctx.key_averages().table(
                        sort_by="self_cpu_time_total", row_limit=40
                    )
                )
                if self._profile_memory:
                    fh.write("\n\n=== Top-40 self_cuda_memory_usage ===\n")
                    fh.write(
                        prof_ctx.key_averages().table(
                            sort_by="self_cuda_memory_usage", row_limit=40
                        )
                    )
            print(f"[profile] top-ops text dump saved: {txt_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"[profile] top-ops text dump failed: {exc}")

    @staticmethod
    def _key_avg_to_pairs(
        prof_ctx: profile, sort_by: str, row_limit: int = 40
    ) -> list[tuple[str, float]]:
        """Convert profiler key_averages → list of (name, microseconds) pairs."""
        avgs = prof_ctx.key_averages()
        # Each entry exposes .key (name) and .self_cuda_time_total / etc.
        # We attempt to read the requested sort column reflectively.
        attr_map = {
            "self_cuda_time_total": "self_cuda_time_total",
            "self_cpu_time_total": "self_cpu_time_total",
        }
        attr = attr_map.get(sort_by, sort_by)
        pairs: list[tuple[str, float]] = []
        for entry in avgs:
            try:
                value = float(getattr(entry, attr, 0.0) or 0.0)
            except (TypeError, ValueError):
                value = 0.0
            pairs.append((str(entry.key), value))
        pairs.sort(key=lambda p: p[1], reverse=True)
        return pairs[:row_limit]

    @staticmethod
    def _key_avg_to_pairs_int(
        prof_ctx: profile, sort_by: str, row_limit: int = 40
    ) -> list[tuple[str, int]]:
        """Same as :meth:`_key_avg_to_pairs` but for integer fields (memory)."""
        avgs = prof_ctx.key_averages()
        pairs: list[tuple[str, int]] = []
        for entry in avgs:
            try:
                value = int(getattr(entry, sort_by, 0) or 0)
            except (TypeError, ValueError):
                value = 0
            pairs.append((str(entry.key), value))
        pairs.sort(key=lambda p: p[1], reverse=True)
        return pairs[:row_limit]


def write_perf_summary_json(
    out_path: Path,
    *,
    config: dict[str, Any],
    stage_times: dict[str, list[float]],
    top_cuda_ops: list[tuple[str, float]],
    top_cpu_ops: list[tuple[str, float]],
    diagnosis: dict[str, Any],
    recommendation: str,
) -> None:
    """Write a machine-readable perf_summary.json next to perf_summary.txt."""
    payload = {
        "config": config,
        "stage_times_ms": stage_times,
        "top_cuda_ops_us": top_cuda_ops,
        "top_cpu_ops_us": top_cpu_ops,
        "diagnosis": {
            k: (v if not isinstance(v, dict) else _coerce_jsonable(v))
            for k, v in diagnosis.items()
        },
        "recommendation": recommendation,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _coerce_jsonable(d: dict[str, Any]) -> dict[str, Any]:
    """Coerce evidence sub-dict values to JSON-friendly types."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            out[k] = v
        elif isinstance(v, dict):
            out[k] = _coerce_jsonable(v)
        elif isinstance(v, (list, tuple)):
            out[k] = list(v)
        else:
            out[k] = str(v)
    return out
