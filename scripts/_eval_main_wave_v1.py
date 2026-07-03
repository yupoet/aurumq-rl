#!/usr/bin/env python3
"""Phase 22 main-wave eval — V1 architecture variant.

Same scoring criteria as ``_eval_main_wave.py`` (the V2 variant on the
phase21 branch), but loads a V1 ``PerStockEncoderPolicy`` checkpoint and
feeds it the V1 single-tensor observation (n_stocks, n_factors). No
regime tensor, no Dict obs, no V2 custom_objects.

Used to baseline V1 production checkpoints (Phase 16a etc.) under the
main-wave criterion before Phase 22 reward redesign.

Output
------
- ``<run-dir>/main_wave_eval.json``
- ``<run-dir>/main_wave_eval.md``
- ``<run-dir>/main_wave_picks.jsonl``
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
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
    pivot_adjusted_close,
)
from aurumq_rl.gpu_env import GPUStockPickingEnv  # noqa: F401  custom_objects
from aurumq_rl.gpu_rollout_buffer import GPURolloutBuffer
from aurumq_rl.index_rollout_buffer import IndexOnlyRolloutBuffer
from aurumq_rl.main_wave_labels import (
    MainWaveConfig,
    aggregate_eval_metrics,
    compute_main_wave_labels,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--data-path", type=Path, required=True)
    p.add_argument("--val-start", required=True)
    p.add_argument("--val-end", required=True)
    p.add_argument("--universe-filter", default="main_board_non_st")
    p.add_argument("--top-k", type=int, nargs="+", default=[3, 5])
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Single ckpt; default = ppo_final.zip + every ppo_*_steps.zip")
    p.add_argument("--device", default="cuda")
    p.add_argument("--metadata-from", type=Path, default=None,
                   help="If --run-dir doesn't have metadata.json, point to a "
                        "donor metadata (e.g. models/production/...)")
    p.add_argument("--out-suffix", default="",
                   help="Append to output filenames, e.g. '_phase16a' to write "
                        "main_wave_eval_phase16a.json")
    p.add_argument("--hold-window", type=int, default=5)
    p.add_argument("--vol-window", type=int, default=20)
    p.add_argument("--sigma-multiplier", type=float, default=2.0)
    p.add_argument("--absolute-threshold", type=float, default=0.06)
    p.add_argument("--max-adverse-limit", type=float, default=0.05)
    p.add_argument("--amount-ma-window", type=int, default=20)
    p.add_argument("--amount-ma-min", type=float, default=1e8)
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
    """V1 PPO.load. We supply IndexOnlyRolloutBuffer + GPURolloutBuffer as
    custom_objects in case the saved zip references a buffer class that
    pickle can't otherwise resolve."""
    custom_objects = {
        "rollout_buffer_class": IndexOnlyRolloutBuffer,
        "GPURolloutBuffer": GPURolloutBuffer,
        "IndexOnlyRolloutBuffer": IndexOnlyRolloutBuffer,
    }
    return PPO.load(str(ckpt_path), device=device, custom_objects=custom_objects)


def _score_v1(model, panel_t: torch.Tensor, n_dates: int) -> np.ndarray:
    """V1 single-tensor obs path. panel_t: (T, S, F) cuda."""
    policy_device = next(model.policy.parameters()).device
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for t in range(n_dates):
            obs_tensor = panel_t[t : t + 1].to(policy_device)  # (1, S, F)
            actions, _, _ = model.policy.forward(obs_tensor, deterministic=True)
            scores.append(actions.detach().cpu().numpy().squeeze(0))
    return np.stack(scores, axis=0)


def _select_top_k(
    pred: np.ndarray,                   # (T, S)
    eligible_mask: np.ndarray,           # (T, S) bool
    top_k: int,
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


def _to_array(pivoted: pl.DataFrame, codes: list[str]) -> np.ndarray:
    existing = [c for c in pivoted.columns if c != "trade_date"]
    existing_set = set(existing)
    arrs: list[np.ndarray] = []
    for code in codes:
        if code in existing_set:
            col = pivoted.get_column(code).fill_null(0.0).to_numpy()
        else:
            col = np.zeros(pivoted.height, dtype=np.float32)
        arrs.append(col.astype(np.float32, copy=False))
    return np.stack(arrs, axis=1)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = MainWaveConfig(
        hold_window=args.hold_window,
        vol_window=args.vol_window,
        sigma_multiplier=args.sigma_multiplier,
        absolute_threshold=args.absolute_threshold,
        max_adverse_limit=args.max_adverse_limit,
        amount_ma_window=args.amount_ma_window,
        amount_ma_min=args.amount_ma_min,
    )

    # ---- Metadata ----
    meta_src = args.metadata_from or (args.run_dir / "metadata.json")
    if not meta_src.exists():
        print(f"[err] metadata not found at {meta_src}", file=sys.stderr)
        return 2
    meta = json.loads(meta_src.read_text(encoding="utf-8"))
    factor_names = meta.get("factor_names")
    if factor_names is None:
        print(f"[err] {meta_src} has no 'factor_names' (V1 metadata key)", file=sys.stderr)
        return 3
    train_stock_codes = meta["stock_codes"]
    forward_period = int(meta.get("forward_period", 10))

    ckpts = _list_checkpoints(args.run_dir, args.checkpoint)
    if not ckpts:
        print(f"[err] no checkpoints found", file=sys.stderr)
        return 4
    print(f"[main_wave_v1] {len(ckpts)} checkpoint(s) to evaluate")

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
    print(f"[main_wave_v1] panel: dates={n_dates} stocks={n_stocks} factors={n_factors}")

    # ---- Recover absolute close / vol / pct from parquet ----
    df = pl.read_parquet(args.data_path)
    df = df.filter(
        (pl.col("trade_date") >= dt.date.fromisoformat(args.val_start))
        & (pl.col("trade_date") <= dt.date.fromisoformat(args.val_end))
    )
    close_df = (df.select(["trade_date", "ts_code", "close"])
                  .pivot(values="close", index="trade_date", on="ts_code")
                  .sort("trade_date"))
    vol_df = (df.select(["trade_date", "ts_code", "vol"])
                .pivot(values="vol", index="trade_date", on="ts_code")
                .sort("trade_date"))
    pct_df = (df.select(["trade_date", "ts_code", "pct_chg"])
                .pivot(values="pct_chg", index="trade_date", on="ts_code")
                .sort("trade_date"))
    close_arr = _to_array(close_df, train_stock_codes)
    vol_arr = _to_array(vol_df, train_stock_codes)
    pct_arr = _to_array(pct_df, train_stock_codes)

    if close_arr.shape[0] != n_dates:
        date_to_row = {d: i for i, d in enumerate(close_df.get_column("trade_date").to_list())}
        idx = [date_to_row[d] for d in panel.dates if d in date_to_row]
        close_arr = close_arr[idx]
        vol_arr = vol_arr[idx]
        pct_arr = pct_arr[idx]

    # C1: label/MA computation runs on ADJUSTED close (close * adj_factor
    # when the parquet carries it); raw close_arr stays for the RAW
    # amount/liquidity gate.
    adj_close_arr = pivot_adjusted_close(df, train_stock_codes, panel.dates)
    amount_arr = close_arr * vol_arr

    # ---- valid_mask_basic ----
    valid_basic = (
        (~panel.is_st_array)
        & (~panel.is_suspended_array)
        & (panel.days_since_ipo_array >= 60)
    )

    # ---- Compute labels ----
    print("[main_wave_v1] computing main-wave labels...")
    labels = compute_main_wave_labels(
        close=adj_close_arr, pct_chg=pct_arr, vol=vol_arr,
        valid_mask_basic=valid_basic, cfg=cfg, amount=amount_arr,
    )
    n_eligible_total = int(labels.entry_eligible_mask.sum())
    n_label_valid = int(labels.label_valid_mask.sum())
    n_hit_total = int(labels.hit_main_wave.sum())
    print(f"[main_wave_v1] eligible (t,j): {n_eligible_total:,}")
    print(f"[main_wave_v1] label-valid (t,j): {n_label_valid:,}")
    print(f"[main_wave_v1] hit_main_wave (t,j): {n_hit_total:,}  "
          f"(base rate: {100.0 * n_hit_total / max(n_label_valid, 1):.2f}%)")

    # ---- Eval each ckpt ----
    panel_t = torch.from_numpy(panel.factor_array).to(args.device)
    eligible_for_selection = labels.entry_eligible_mask & labels.label_valid_mask

    suffix = args.out_suffix
    picks_path = args.run_dir / f"main_wave_picks{suffix}.jsonl"
    picks_writer = picks_path.open("w", encoding="utf-8")

    results: list[dict] = []
    for step, ckpt_path in ckpts:
        label = f"step{step}" if step >= 0 else "final"
        print(f"\n[main_wave_v1] === {label} ===")
        try:
            model = _load_v1_ckpt(ckpt_path, args.device)
        except Exception as e:
            print(f"  [skip] load failed: {e!r}")
            continue
        model.policy.eval()
        model.policy.to(args.device)

        try:
            preds = _score_v1(model, panel_t, n_dates)
        except Exception as e:
            print(f"  [skip] scoring failed: {e!r}")
            continue
        finally:
            del model
            torch.cuda.empty_cache()

        for top_k in args.top_k:
            selected = _select_top_k(preds, eligible_for_selection, top_k)
            metrics = aggregate_eval_metrics(labels, selected, cfg)
            metrics["checkpoint"] = str(ckpt_path)
            metrics["checkpoint_label"] = label
            metrics["step"] = step
            metrics["top_k"] = top_k
            results.append(metrics)

            print(
                f"  top{top_k}: hit={metrics['main_wave_hit_rate']:.3f} "
                f"win={metrics['basic_win_rate']:.3f} "
                f"avg_hold={metrics['avg_hold_return']:+.4f} "
                f"avg_score={metrics['avg_main_wave_score']:.1f} "
                f"avg_dd={metrics['avg_max_drawdown']:.4f} "
                f"payoff={metrics['payoff_ratio']:.2f} "
                f"days={metrics['avg_holding_days']:.2f} "
                f"daily_prec={metrics['top_k_daily_precision']:.3f} "
                f"eval_score={metrics['eval_score']:+.4f}"
            )

            for t, idx_arr in enumerate(selected):
                if idx_arr is None or len(idx_arr) == 0:
                    continue
                date_iso = panel.dates[t].isoformat()
                for rank, j in enumerate(idx_arr.tolist(), start=1):
                    entry_d_idx = int(labels.entry_day_idx[t, j])
                    exit_d_idx = int(labels.exit_day_idx[t, j])
                    row = {
                        "ckpt": label,
                        "top_k": top_k,
                        "date": date_iso,
                        "entry_date": (
                            panel.dates[entry_d_idx].isoformat()
                            if 0 <= entry_d_idx < len(panel.dates) else None
                        ),
                        "exit_date": (
                            panel.dates[exit_d_idx].isoformat()
                            if 0 <= exit_d_idx < len(panel.dates) else None
                        ),
                        "rank": rank,
                        "stock_code": panel.stock_codes[j],
                        "score_model": float(preds[t, j]),
                        "main_wave_score": float(labels.main_wave_score[t, j]),
                        "hit_main_wave": bool(labels.hit_main_wave[t, j]),
                        # NOTE: since the C1 fix, entry/exit prices are the
                        # ADJUSTED close (close * adj_factor) whenever the
                        # parquet carries adj_factor — correct for return
                        # math, but NOT the raw quoted price. Field names
                        # are kept for downstream picks.jsonl consumers.
                        "entry_price": float(labels.entry_price[t, j]),
                        "exit_price": float(labels.exit_price[t, j]),
                        "holding_days": int(labels.holding_days[t, j]),
                        "hold_return": float(labels.hold_return[t, j]),
                        "max_cum_return_5d": float(labels.max_cum_return_5d[t, j]),
                        "max_drawdown_during_hold": float(labels.max_drawdown_during_hold[t, j]),
                        "max_adverse_excursion": float(labels.max_adverse_excursion[t, j]),
                        "amount_ma20": float(labels.amount_ma20[t, j]),
                        "below_ma_state": bool(labels.below_ma_state[t, j]),
                        "label_valid": bool(labels.label_valid_mask[t, j]),
                        "entry_price_proxy": "next_close",
                        "path_uses_close_only": True,
                    }
                    picks_writer.write(json.dumps(row, ensure_ascii=False) + "\n")
    picks_writer.close()

    out_json = args.run_dir / f"main_wave_eval{suffix}.json"
    out_md = args.run_dir / f"main_wave_eval{suffix}.md"
    out_json.write_text(
        json.dumps({
            "config": cfg.__dict__,
            "data_path": str(args.data_path),
            "val_start": args.val_start,
            "val_end": args.val_end,
            "n_dates": n_dates, "n_stocks": n_stocks, "n_factors": n_factors,
            "n_eligible_cells": n_eligible_total,
            "n_label_valid_cells": n_label_valid,
            "n_hit_cells": n_hit_total,
            "rows": results,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if results:
        cols = [
            "checkpoint_label", "top_k", "main_wave_hit_rate", "basic_win_rate",
            "avg_hold_return", "avg_main_wave_score", "avg_max_drawdown",
            "payoff_ratio", "avg_holding_days", "top_k_daily_precision",
            "eval_score", "n_picks",
        ]
        head = "# Phase 22 main-wave eval (V1)\n\n"
        head += "| " + " | ".join(cols) + " |\n"
        head += "| " + " | ".join("---" for _ in cols) + " |\n"
        rows = []
        for r in results:
            cells = []
            for c in cols:
                v = r.get(c, "")
                if isinstance(v, float):
                    cells.append(
                        f"{v:+.4f}" if any(k in c for k in
                                            ("rate", "ratio", "score", "return", "drawdown"))
                        else f"{v:.3f}"
                    )
                else:
                    cells.append(str(v))
            rows.append("| " + " | ".join(cells) + " |")
        out_md.write_text(head + "\n".join(rows) + "\n", encoding="utf-8")
    print(f"\n[main_wave_v1] wrote {out_json}")
    print(f"[main_wave_v1] wrote {out_md}")
    print(f"[main_wave_v1] wrote {picks_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
