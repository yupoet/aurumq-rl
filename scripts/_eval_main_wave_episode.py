#!/usr/bin/env python3
"""Phase 23 episode-based eval — V1 ckpt path.

Evaluates a trained V1 ``PerStockEncoderPolicy`` checkpoint against the new
**entry-only** main-wave criterion: did the model pick a stock at T-1 to
T-3 of a real main-wave episode?

Output (all under ``<run-dir>``):
- ``episode_eval.json`` / ``.md`` — aggregate metrics per checkpoint × top_k
- ``episode_picks.jsonl`` — per-(date, rank, stock) detail rows

Metrics replace Phase 22's exit-coupled hold_return metrics with:
- ``t_minus_1_hit_rate``: P(picked stock has episode with t_start = decision_day + 1)
- ``t_minus_3_hit_rate``: same with t_start ∈ {decision+1, +2, +3}
- ``avg_peak_return_of_hits`` / ``avg_duration_of_hits`` / ``avg_smoothness_of_hits``
- ``proximity_distribution`` (T-1 vs T-2 vs T-3 vs miss)
- ``daily_t_minus_1_precision``: at least one of top_K is at T-1 on a given day
- ``eval_score_v23``: composite

Usage::

    .venv/Scripts/python.exe scripts/_eval_main_wave_episode.py \\
        --run-dir runs/phase23a_episode_seed42 \\
        --data-path data/factor_panel_combined_short_2023_2026.parquet \\
        --val-start 2025-07-01 --val-end 2026-04-24 \\
        --top-k 3 5

Loads V1 single-tensor obs ckpts. For V2 Dict-obs ckpts, see the
``_eval_main_wave.py`` variant on the phase21 branch (not used in Phase 23).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np
import polars as pl
import torch
from stable_baselines3 import PPO

from aurumq_rl.data_loader import (
    FactorPanelLoader,
    UniverseFilter,
    align_panel_to_stock_list,
)
from aurumq_rl.gpu_env import GPUStockPickingEnv  # noqa: F401
from aurumq_rl.gpu_rollout_buffer import GPURolloutBuffer
from aurumq_rl.index_rollout_buffer import IndexOnlyRolloutBuffer
from aurumq_rl.main_wave_episodes import (
    EpisodeConfig, find_main_wave_episodes, episodes_summary,
)
from aurumq_rl.main_wave_target_labels import (
    TargetConfig, build_target_quality, quality_of_episode,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--data-path", type=Path, required=True)
    p.add_argument("--val-start", required=True)
    p.add_argument("--val-end", required=True)
    p.add_argument("--universe-filter", default="main_board_non_st")
    p.add_argument("--top-k", type=int, nargs="+", default=[3, 5])
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-suffix", default="")
    return p.parse_args(argv)


def _list_checkpoints(run_dir: Path, single: Path | None) -> list[tuple[int, Path]]:
    if single is not None:
        name = single.stem
        step = -1
        if name.startswith("ppo_") and name.endswith("_steps"):
            try:
                step = int(name.split("_")[1])
            except (ValueError, IndexError):
                step = -1
        return [(step, single)]
    out: list[tuple[int, Path]] = []
    final = run_dir / "ppo_final.zip"
    if final.exists():
        out.append((-1, final))
    cp_dir = run_dir / "checkpoints"
    if cp_dir.exists():
        for p in sorted(cp_dir.glob("ppo_*_steps.zip")):
            try:
                step = int(p.stem.split("_")[1])
                out.append((step, p))
            except (ValueError, IndexError):
                continue
    return out


def _load_v1_ckpt(ckpt_path: Path, device: str):
    custom_objects = {
        "rollout_buffer_class": IndexOnlyRolloutBuffer,
        "GPURolloutBuffer": GPURolloutBuffer,
        "IndexOnlyRolloutBuffer": IndexOnlyRolloutBuffer,
    }
    return PPO.load(str(ckpt_path), device=device, custom_objects=custom_objects)


def _score_v1(model, panel_t: torch.Tensor, n_dates: int) -> np.ndarray:
    policy_device = next(model.policy.parameters()).device
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for t in range(n_dates):
            obs_tensor = panel_t[t : t + 1].to(policy_device)
            actions, _, _ = model.policy.forward(obs_tensor, deterministic=True)
            scores.append(actions.detach().cpu().numpy().squeeze(0))
    return np.stack(scores, axis=0)


def _select_top_k(
    pred: np.ndarray, eligible_mask: np.ndarray, top_k: int,
) -> list[np.ndarray]:
    T, _ = pred.shape
    out: list[np.ndarray] = []
    masked = pred.copy()
    masked[~eligible_mask] = -np.inf
    for t in range(T):
        if not np.isfinite(masked[t]).any():
            out.append(np.array([], dtype=np.int64))
            continue
        order = np.argsort(-masked[t])
        eligible_count = int(np.isfinite(masked[t]).sum())
        k_eff = min(top_k, eligible_count)
        out.append(order[:k_eff].astype(np.int64))
    return out


def _to_array(piv: pl.DataFrame, codes: list[str]) -> np.ndarray:
    existing = {c for c in piv.columns if c != "trade_date"}
    arrs = []
    for code in codes:
        if code in existing:
            col = piv.get_column(code).fill_null(0.0).to_numpy()
        else:
            col = np.zeros(piv.height, dtype=np.float32)
        arrs.append(col.astype(np.float32, copy=False))
    return np.stack(arrs, axis=1)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # ---- Metadata ----
    meta_path = args.run_dir / "metadata.json"
    if not meta_path.exists():
        print(f"[err] {meta_path} not found", file=sys.stderr)
        return 2
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    factor_names = meta.get("factor_names")
    if factor_names is None:
        print(f"[err] {meta_path} has no factor_names", file=sys.stderr)
        return 3
    train_stock_codes = meta["stock_codes"]
    forward_period = int(meta.get("forward_period", 10))

    ckpts = _list_checkpoints(args.run_dir, args.checkpoint)
    if not ckpts:
        print(f"[err] no checkpoints", file=sys.stderr)
        return 4
    print(f"[ep_eval] {len(ckpts)} checkpoint(s)")

    # ---- Panel ----
    loader = FactorPanelLoader(parquet_path=args.data_path)
    panel = loader.load_panel(
        start_date=dt.date.fromisoformat(args.val_start),
        end_date=dt.date.fromisoformat(args.val_end),
        universe_filter=UniverseFilter(args.universe_filter),
        forward_period=forward_period,
        factor_names=factor_names,
    )
    panel = align_panel_to_stock_list(panel, train_stock_codes)
    n_dates, n_stocks, n_factors = panel.factor_array.shape
    print(f"[ep_eval] panel: dates={n_dates} stocks={n_stocks} factors={n_factors}")

    # ---- Recover absolute close / vol from parquet for episode scan ----
    df = pl.read_parquet(args.data_path)
    df = df.filter(
        (pl.col("trade_date") >= dt.date.fromisoformat(args.val_start))
        & (pl.col("trade_date") <= dt.date.fromisoformat(args.val_end))
    )
    close_arr = _to_array(
        df.select(["trade_date", "ts_code", "close"])
          .pivot(values="close", index="trade_date", on="ts_code")
          .sort("trade_date"),
        train_stock_codes,
    )
    vol_arr = _to_array(
        df.select(["trade_date", "ts_code", "vol"])
          .pivot(values="vol", index="trade_date", on="ts_code")
          .sort("trade_date"),
        train_stock_codes,
    )
    if close_arr.shape[0] != n_dates:
        # date filter may emit slightly different rowcount; align by panel.dates
        piv_dates = df.select("trade_date").unique().sort("trade_date").get_column("trade_date").to_list()
        d2r = {d: i for i, d in enumerate(piv_dates)}
        idx = [d2r[d] for d in panel.dates if d in d2r]
        close_arr = close_arr[idx]
        vol_arr = vol_arr[idx]

    valid_basic = (
        (~panel.is_st_array)
        & (~panel.is_suspended_array)
        & (panel.days_since_ipo_array >= 60)
    )

    # ---- Run episode scanner on the EVAL window ----
    ep_cfg = EpisodeConfig()
    print("[ep_eval] scanning OOS episodes...")
    episodes = find_main_wave_episodes(close_arr, vol_arr, valid_basic, ep_cfg)
    summary = episodes_summary(episodes)
    n_eps = len(episodes)
    print(f"[ep_eval] {n_eps:,} episodes; "
          f"avg peak={summary['avg_peak_return']:+.4f}, "
          f"avg duration={summary['avg_duration']:.1f}, "
          f"avg max_dd={summary['avg_max_dd']:.4f}")

    # ---- Build target_quality + proximity arrays ----
    tg_cfg = TargetConfig()
    targets = build_target_quality(episodes, n_dates, n_stocks, tg_cfg)
    n_t1 = int((targets.proximity == 1).sum())
    n_t13 = int((targets.proximity > 0).sum())
    label_valid = valid_basic & (close_arr > 0)
    n_label_valid = int(label_valid.sum())
    base_t1 = n_t1 / max(n_label_valid, 1)
    base_t13 = n_t13 / max(n_label_valid, 1)
    print(f"[ep_eval] T-1 cells: {n_t1:,} / {n_label_valid:,} "
          f"(base rate {100*base_t1:.3f}%)")
    print(f"[ep_eval] T-1..T-3 cells: {n_t13:,} (base rate {100*base_t13:.3f}%)")

    # ---- Eligibility for SELECTION (basic + amount_ma20 > 1e8) ----
    # Compute 20d rolling amount mean
    from aurumq_rl.main_wave_episodes import _rolling_mean_2d
    amount = close_arr * vol_arr
    amount_ma20 = _rolling_mean_2d(amount, 20)
    liquid = amount_ma20 > ep_cfg.amount_ma_min
    eligible = valid_basic & liquid

    # ---- Per-checkpoint eval ----
    panel_t = torch.from_numpy(panel.factor_array).to(args.device)
    suffix = args.out_suffix
    picks_path = args.run_dir / f"episode_picks{suffix}.jsonl"
    picks_writer = picks_path.open("w", encoding="utf-8")

    results: list[dict] = []
    for step, ckpt_path in ckpts:
        label = f"step{step}" if step >= 0 else "final"
        print(f"\n[ep_eval] === {label} ===")
        try:
            model = _load_v1_ckpt(ckpt_path, args.device)
        except Exception as e:
            print(f"  [skip] {e!r}")
            continue
        model.policy.eval(); model.policy.to(args.device)
        try:
            preds = _score_v1(model, panel_t, n_dates)
        except Exception as e:
            print(f"  [skip] scoring: {e!r}")
            continue
        finally:
            del model
            torch.cuda.empty_cache()

        for top_k in args.top_k:
            selected = _select_top_k(preds, eligible, top_k)
            metrics = _aggregate(
                selected, targets, episodes, panel, top_k,
                ep_cfg=ep_cfg, label=label, base_t1=base_t1, base_t13=base_t13,
            )
            metrics["checkpoint"] = str(ckpt_path)
            metrics["checkpoint_label"] = label
            metrics["step"] = step
            results.append(metrics)
            print(
                f"  top{top_k}: T1_hit={metrics['t_minus_1_hit_rate']:.4f} "
                f"(base={base_t1:.4f}, lift={metrics['t_minus_1_hit_rate']/max(base_t1,1e-9):.2f}x)  "
                f"T13_hit={metrics['t_minus_3_hit_rate']:.4f}  "
                f"avg_peak_hit={metrics['avg_peak_return_of_hits']:+.4f}  "
                f"avg_dur={metrics['avg_duration_of_hits']:.1f}  "
                f"daily_T1_prec={metrics['daily_t_minus_1_precision']:.4f}  "
                f"eval_score={metrics['eval_score_v23']:+.4f}"
            )
            _dump_picks(picks_writer, panel, selected, preds, targets, episodes,
                        label=label, top_k=top_k)

    picks_writer.close()

    out_json_path = args.run_dir / f"episode_eval{suffix}.json"
    out_md_path = args.run_dir / f"episode_eval{suffix}.md"
    out_json_path.write_text(
        json.dumps({
            "data_path": str(args.data_path),
            "val_start": args.val_start, "val_end": args.val_end,
            "n_dates": n_dates, "n_stocks": n_stocks,
            "n_episodes": n_eps,
            "episode_summary": summary,
            "n_t1_cells": n_t1, "n_t13_cells": n_t13,
            "n_label_valid": n_label_valid,
            "base_t1_rate": base_t1, "base_t13_rate": base_t13,
            "rows": results,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    out_md_path.write_text(_render_md(results, base_t1, base_t13), encoding="utf-8")
    print(f"\n[ep_eval] wrote {out_json_path}")
    print(f"[ep_eval] wrote {out_md_path}")
    print(f"[ep_eval] wrote {picks_path}")
    return 0


def _aggregate(
    selected_per_date, targets, episodes, panel, top_k,
    ep_cfg, label, base_t1: float, base_t13: float,
) -> dict:
    """Per-checkpoint aggregate metrics."""
    n_dates = targets.target_quality.shape[0]
    proximities: list[int] = []           # 0 (miss), 1, 2, 3
    peak_returns: list[float] = []
    durations: list[int] = []
    smoothnesses: list[float] = []
    daily_has_t1: list[bool] = []
    n_picks_total = 0

    for t, idx in enumerate(selected_per_date):
        if idx is None or len(idx) == 0:
            continue
        n_picks_total += len(idx)
        any_t1 = False
        for j in idx.tolist():
            prox = int(targets.proximity[t, j])
            proximities.append(prox)
            if prox == 1:
                any_t1 = True
            if prox > 0:
                ep_id = int(targets.episode_id_at_proximity[t, j])
                if 0 <= ep_id < len(episodes):
                    e = episodes[ep_id]
                    peak_returns.append(e.peak_return)
                    durations.append(e.duration)
                    smoothness = 1.0 - min(e.max_dd_during / max(e.peak_return, 1e-6), 1.0)
                    smoothnesses.append(smoothness)
        daily_has_t1.append(any_t1)

    if n_picks_total == 0:
        return {
            "n_picks": 0, "top_k": top_k,
            "t_minus_1_hit_rate": 0.0, "t_minus_3_hit_rate": 0.0,
            "avg_peak_return_of_hits": 0.0, "avg_duration_of_hits": 0.0,
            "avg_smoothness_of_hits": 0.0, "daily_t_minus_1_precision": 0.0,
            "proximity_t1_count": 0, "proximity_t2_count": 0,
            "proximity_t3_count": 0, "miss_count": 0,
            "eval_score_v23": 0.0, "n_hits": 0,
        }

    prox_arr = np.asarray(proximities)
    t1_count = int((prox_arr == 1).sum())
    t2_count = int((prox_arr == 2).sum())
    t3_count = int((prox_arr == 3).sum())
    miss_count = int((prox_arr == 0).sum())
    t1_rate = t1_count / n_picks_total
    t13_rate = (t1_count + t2_count + t3_count) / n_picks_total

    avg_peak = float(np.mean(peak_returns)) if peak_returns else 0.0
    avg_dur = float(np.mean(durations)) if durations else 0.0
    avg_smooth = float(np.mean(smoothnesses)) if smoothnesses else 0.0
    daily_prec = float(np.mean(daily_has_t1)) if daily_has_t1 else 0.0

    eval_score = (
        0.35 * t1_rate
        + 0.20 * t13_rate
        + 0.20 * float(np.tanh(avg_peak / 0.20))
        + 0.10 * float(np.tanh((avg_dur - 5) / 5))
        + 0.10 * avg_smooth
        + 0.05 * daily_prec
    )

    return {
        "top_k": top_k,
        "n_picks": n_picks_total,
        "n_hits": int(t1_count + t2_count + t3_count),
        "t_minus_1_hit_rate": t1_rate,
        "t_minus_3_hit_rate": t13_rate,
        "t1_lift_over_base": t1_rate / max(base_t1, 1e-9),
        "t13_lift_over_base": t13_rate / max(base_t13, 1e-9),
        "avg_peak_return_of_hits": avg_peak,
        "avg_duration_of_hits": avg_dur,
        "avg_smoothness_of_hits": avg_smooth,
        "daily_t_minus_1_precision": daily_prec,
        "proximity_t1_count": t1_count,
        "proximity_t2_count": t2_count,
        "proximity_t3_count": t3_count,
        "miss_count": miss_count,
        "eval_score_v23": eval_score,
    }


def _dump_picks(fp, panel, selected_per_date, preds, targets, episodes,
                label: str, top_k: int) -> None:
    for t, idx in enumerate(selected_per_date):
        if idx is None or len(idx) == 0:
            continue
        date_iso = panel.dates[t].isoformat()
        for rank, j in enumerate(idx.tolist(), start=1):
            prox = int(targets.proximity[t, j])
            ep_id = int(targets.episode_id_at_proximity[t, j])
            ep_data = None
            if 0 <= ep_id < len(episodes):
                e = episodes[ep_id]
                ep_data = {
                    "t_start_date": (panel.dates[e.t_start].isoformat()
                                     if 0 <= e.t_start < len(panel.dates) else None),
                    "t_peak_date": (panel.dates[e.t_peak].isoformat()
                                    if 0 <= e.t_peak < len(panel.dates) else None),
                    "peak_return": float(e.peak_return),
                    "duration": int(e.duration),
                    "max_dd_during": float(e.max_dd_during),
                }
            row = {
                "ckpt": label, "top_k": top_k, "date": date_iso,
                "rank": rank, "stock_code": panel.stock_codes[j],
                "score_model": float(preds[t, j]),
                "target_quality": float(targets.target_quality[t, j]),
                "proximity_to_episode": prox,
                "is_t_minus_1": prox == 1,
                "episode": ep_data,
            }
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def _render_md(results: list[dict], base_t1: float, base_t13: float) -> str:
    if not results:
        return "# episode_eval — no rows\n"
    cols = ["checkpoint_label", "top_k", "t_minus_1_hit_rate", "t_minus_3_hit_rate",
            "t1_lift_over_base", "avg_peak_return_of_hits", "avg_duration_of_hits",
            "avg_smoothness_of_hits", "daily_t_minus_1_precision", "eval_score_v23",
            "n_picks", "n_hits"]
    head = "# Phase 23 episode eval (V1)\n\n"
    head += f"Base rates: T-1 = **{base_t1:.4f}**, T-1..T-3 = **{base_t13:.4f}**\n\n"
    head += "| " + " | ".join(cols) + " |\n"
    head += "| " + " | ".join("---" for _ in cols) + " |\n"
    rows: list[str] = []
    for r in results:
        cells = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float):
                cells.append(f"{v:+.4f}" if "rate" in c or "score" in c or "return" in c
                             else f"{v:.3f}")
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return head + "\n".join(rows) + "\n"


if __name__ == "__main__":
    sys.exit(main())
