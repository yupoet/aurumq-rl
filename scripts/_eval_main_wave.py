#!/usr/bin/env python3
"""Phase 22 main-wave eval — runs a trained ckpt on an OOS window and scores
its picks under the "main-wave eve" criterion (NOT 10-day forward Sharpe).

Output
------
- ``<run-dir>/main_wave_eval.json``  : aggregate metrics per checkpoint
- ``<run-dir>/main_wave_eval.md``    : same as a markdown table
- ``<run-dir>/main_wave_picks.jsonl``: per-(date, rank, stock) detail rows

Usage
-----

    .venv/Scripts/python.exe scripts/_eval_main_wave.py \\
        --run-dir runs/phase21_21a_v2_drop_mkt_seed42 \\
        --data-path data/factor_panel_combined_short_2023_2026.parquet \\
        --val-start 2025-07-01 --val-end 2026-04-24 \\
        --top-k 5 \\
        --universe-filter main_board_non_st

The eval does NOT modify training, env, or reward — only inference scoring +
new label-based metrics. See spec at docs/superpowers/specs/2026-05-05-phase21-v2-architecture-design.md
for the architecture context; the main-wave criterion is documented in this
script's module docstring + ``src/aurumq_rl/main_wave_labels.py``.
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
from stable_baselines3.common.utils import obs_as_tensor

from aurumq_rl.data_loader import (
    FactorPanelLoader,
    UniverseFilter,
    align_panel_to_stock_list,
)
from aurumq_rl.gpu_env import GPUStockPickingEnv  # noqa: F401  custom_objects
from aurumq_rl.gpu_rollout_buffer import GPURolloutBuffer
from aurumq_rl.index_dict_rollout_buffer import IndexOnlyDictRolloutBuffer
from aurumq_rl.main_wave_labels import (
    MainWaveConfig,
    aggregate_eval_metrics,
    compute_main_wave_labels,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--data-path", type=Path, required=True)
    p.add_argument("--val-start", required=True)
    p.add_argument("--val-end", required=True)
    p.add_argument("--universe-filter", default="main_board_non_st")
    p.add_argument(
        "--top-k",
        type=int,
        nargs="+",
        default=[3, 5],
        help="One or more top-K values to evaluate (default 3 5).",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Single checkpoint to eval (default: every ppo_*_steps.zip + ppo_final.zip in <run-dir>).",
    )
    p.add_argument("--device", default="cuda")
    # MainWaveConfig overrides — most users won't change these
    p.add_argument("--hold-window", type=int, default=5)
    p.add_argument("--vol-window", type=int, default=20)
    p.add_argument("--sigma-multiplier", type=float, default=2.0)
    p.add_argument("--absolute-threshold", type=float, default=0.06)
    p.add_argument("--max-adverse-limit", type=float, default=0.05)
    p.add_argument("--amount-ma-window", type=int, default=20)
    p.add_argument("--amount-ma-min", type=float, default=1e8)
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _list_checkpoints(run_dir: Path, single: Path | None) -> list[tuple[int, Path]]:
    """Return list of (step, path); step=-1 for ppo_final.zip."""
    if single is not None:
        # Try to extract step from filename
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


def _load_ckpt(ckpt_path: Path, device: str):
    """Load with custom_objects so V2 (Dict-obs) buffers deserialize."""
    custom_objects = {
        "rollout_buffer_class": IndexOnlyDictRolloutBuffer,
        "IndexOnlyDictRolloutBuffer": IndexOnlyDictRolloutBuffer,
        "GPURolloutBuffer": GPURolloutBuffer,
    }
    return PPO.load(str(ckpt_path), device=device, custom_objects=custom_objects)


def _build_dict_obs(
    panel_t: torch.Tensor,
    regime_t: torch.Tensor,
    valid_mask_t: torch.Tensor,
    t: int,
) -> dict[str, np.ndarray]:
    """V2 Dict observation. (1, S, F) / (1, R) / (1, S)."""
    return {
        "stock": panel_t[t : t + 1].detach().cpu().numpy(),
        "regime": regime_t[t : t + 1].detach().cpu().numpy(),
        "valid_mask": valid_mask_t[t : t + 1].to(dtype=torch.float32).detach().cpu().numpy(),
    }


def _score_all_dates(model, panel_t, regime_t, valid_mask_t, n_dates: int) -> np.ndarray:
    """Run model.policy.forward on every date; return (n_dates, n_stocks) preds."""
    policy_device = next(model.policy.parameters()).device
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for t in range(n_dates):
            obs_np = _build_dict_obs(panel_t, regime_t, valid_mask_t, t)
            obs_tensor = obs_as_tensor(obs_np, policy_device)
            actions, _, _ = model.policy.forward(obs_tensor, deterministic=True)
            scores.append(actions.detach().cpu().numpy().squeeze(0))
    return np.stack(scores, axis=0)


def _select_top_k(
    pred: np.ndarray,                   # (T, S) raw model scores
    eligible_mask: np.ndarray,           # (T, S) bool — basic & liquid & not below_ma
    top_k: int,
) -> list[np.ndarray]:
    """Per-date list of (≤ top_k) selected stock indices, eligible only."""
    T, S = pred.shape
    out: list[np.ndarray] = []
    masked = pred.copy()
    masked[~eligible_mask] = -np.inf
    for t in range(T):
        # Stocks with -inf are ineligible; if fewer than top_k eligible,
        # take all eligible (variable-length).
        if not np.isfinite(masked[t]).any():
            out.append(np.array([], dtype=np.int64))
            continue
        order = np.argsort(-masked[t])
        # Keep only finite (eligible) entries
        eligible_count = int(np.isfinite(masked[t]).sum())
        k_eff = min(top_k, eligible_count)
        out.append(order[:k_eff].astype(np.int64))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


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

    # ---- Metadata + checkpoints ----
    meta_path = args.run_dir / "metadata.json"
    if not meta_path.exists():
        print(f"[err] {meta_path} not found", file=sys.stderr)
        return 2
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if "regime_factor_names" not in meta:
        print(
            f"[err] {meta_path} predates Phase 21 (no regime_factor_names). "
            f"Phase 22 main-wave eval is currently V2-only.",
            file=sys.stderr,
        )
        return 3
    stock_factor_names = meta.get("stock_factor_names") or meta.get("factor_names")
    train_stock_codes = meta["stock_codes"]
    forward_period = int(meta.get("forward_period", 10))

    ckpts = _list_checkpoints(args.run_dir, args.checkpoint)
    if not ckpts:
        print(f"[err] no checkpoints in {args.run_dir}", file=sys.stderr)
        return 4
    print(f"[main_wave] {len(ckpts)} checkpoint(s) to evaluate")

    # ---- Panel ----
    loader = FactorPanelLoader(parquet_path=args.data_path)
    panel = loader.load_panel(
        start_date=dt.date.fromisoformat(args.val_start),
        end_date=dt.date.fromisoformat(args.val_end),
        universe_filter=UniverseFilter(args.universe_filter),
        forward_period=forward_period,
        factor_names=stock_factor_names,
    )
    panel = align_panel_to_stock_list(panel, train_stock_codes)
    n_dates, n_stocks, n_factors = panel.factor_array.shape
    print(f"[main_wave] panel: dates={n_dates} stocks={n_stocks} factors={n_factors}")

    # ---- Recover raw close / vol / pct_chg from the parquet (NOT from
    # ----  `panel.factor_array` which is z-scored). We need ABSOLUTE prices.
    df = pl.read_parquet(args.data_path)
    df = df.filter(
        (pl.col("trade_date") >= dt.date.fromisoformat(args.val_start))
        & (pl.col("trade_date") <= dt.date.fromisoformat(args.val_end))
    )
    # Pivot: rows=trade_date, cols=ts_code
    close_df = (
        df.select(["trade_date", "ts_code", "close"])
          .pivot(values="close", index="trade_date", on="ts_code")
          .sort("trade_date")
    )
    vol_df = (
        df.select(["trade_date", "ts_code", "vol"])
          .pivot(values="vol", index="trade_date", on="ts_code")
          .sort("trade_date")
    )
    pct_df = (
        df.select(["trade_date", "ts_code", "pct_chg"])
          .pivot(values="pct_chg", index="trade_date", on="ts_code")
          .sort("trade_date")
    )
    # Re-order columns to match train_stock_codes (zero-fill missing)
    def _to_array(pivoted: pl.DataFrame, codes: list[str]) -> np.ndarray:
        # The pivoted df has trade_date as first column then one per stock.
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

    close_arr = _to_array(close_df, train_stock_codes)        # (T, S)
    vol_arr = _to_array(vol_df, train_stock_codes)
    pct_arr = _to_array(pct_df, train_stock_codes)
    # If we got fewer rows than panel dates (date-filter quirk), trim.
    # The panel.dates is the canonical eval calendar.
    if close_arr.shape[0] != n_dates:
        # Filter close_arr rows to match panel.dates by date.
        date_to_row = {d: i for i, d in enumerate(close_df.get_column("trade_date").to_list())}
        idx = [date_to_row[d] for d in panel.dates if d in date_to_row]
        close_arr = close_arr[idx]
        vol_arr = vol_arr[idx]
        pct_arr = pct_arr[idx]
        print(f"[main_wave] re-aligned close/vol/pct to panel.dates: shape={close_arr.shape}")

    # ---- valid_mask_basic from panel ----
    valid_basic = (
        (~panel.is_st_array)
        & (~panel.is_suspended_array)
        & (panel.days_since_ipo_array >= 60)
    )

    # ---- Compute main-wave labels ONCE (independent of checkpoint) ----
    print("[main_wave] computing main-wave labels...")
    labels = compute_main_wave_labels(
        close=close_arr, pct_chg=pct_arr, vol=vol_arr,
        valid_mask_basic=valid_basic, cfg=cfg,
    )
    n_eligible_total = int(labels.entry_eligible_mask.sum())
    n_label_valid = int(labels.label_valid_mask.sum())
    n_hit_total = int(labels.hit_main_wave.sum())
    print(f"[main_wave] eligible (t,j) cells: {n_eligible_total:,}")
    print(f"[main_wave] label-valid (t,j) cells: {n_label_valid:,}")
    print(f"[main_wave] hit_main_wave (t,j) cells: {n_hit_total:,}  "
          f"(base rate: {100.0 * n_hit_total / max(n_label_valid, 1):.2f}%)")

    # ---- GPU tensors for inference ----
    panel_t = torch.from_numpy(panel.factor_array).to(args.device)
    regime_t = torch.from_numpy(panel.regime_array).to(args.device)
    valid_mask_t = torch.from_numpy(valid_basic).to(args.device)

    # Eligibility for SELECTION (basic & liquid & not below_ma & has path)
    eligible_for_selection = labels.entry_eligible_mask & labels.label_valid_mask

    # ---- Per-checkpoint eval ----
    results: list[dict] = []
    picks_writer = (args.run_dir / "main_wave_picks.jsonl").open("w", encoding="utf-8")

    for step, ckpt_path in ckpts:
        label = f"step{step}" if step >= 0 else "final"
        print(f"\n[main_wave] === {label} ===")
        try:
            model = _load_ckpt(ckpt_path, args.device)
        except Exception as e:
            print(f"  [skip] failed to load: {e!r}")
            continue
        model.policy.eval()
        model.policy.to(args.device)

        try:
            preds = _score_all_dates(model, panel_t, regime_t, valid_mask_t, n_dates)
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

            # Print summary
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

            # Dump per-pick rows
            _dump_picks(
                picks_writer, panel, labels, selected, preds,
                ckpt_label=label, top_k=top_k,
            )

    picks_writer.close()

    # ---- Write summary outputs ----
    out_json = args.run_dir / "main_wave_eval.json"
    out_md = args.run_dir / "main_wave_eval.md"
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
    out_md.write_text(_render_md(results), encoding="utf-8")
    print(f"\n[main_wave] wrote {out_json}")
    print(f"[main_wave] wrote {out_md}")
    print(f"[main_wave] wrote {args.run_dir / 'main_wave_picks.jsonl'}")
    return 0


def _dump_picks(
    fp,
    panel,
    labels,
    selected_per_date,
    preds: np.ndarray,
    ckpt_label: str,
    top_k: int,
) -> None:
    """One JSONL row per (date, rank, stock)."""
    for t, idx_arr in enumerate(selected_per_date):
        if idx_arr is None or len(idx_arr) == 0:
            continue
        date_iso = panel.dates[t].isoformat()
        for rank, j in enumerate(idx_arr.tolist(), start=1):
            entry_d_idx = int(labels.entry_day_idx[t, j])
            exit_d_idx = int(labels.exit_day_idx[t, j])
            row = {
                "ckpt": ckpt_label,
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
                "entry_price_proxy": "next_close",  # documents the open-price approximation
                "path_uses_close_only": True,        # documents the high/low approximation
            }
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def _render_md(results: list[dict]) -> str:
    if not results:
        return "# main_wave_eval\n\nNo results.\n"
    cols = [
        "checkpoint_label", "top_k", "main_wave_hit_rate", "basic_win_rate",
        "avg_hold_return", "avg_main_wave_score", "avg_max_drawdown",
        "payoff_ratio", "avg_holding_days", "top_k_daily_precision", "eval_score",
        "n_picks",
    ]
    head = "# Phase 22 main-wave eval\n\n"
    head += "Header columns:\n"
    head += "- `main_wave_hit_rate`: P(hit_main_wave == True | picked)\n"
    head += "- `basic_win_rate`:     P(hold_return > 0 | picked)\n"
    head += "- `avg_hold_return`:    mean realized hold return per pick (decimal)\n"
    head += "- `avg_main_wave_score`: mean composite score (0..100, can be negative)\n"
    head += "- `avg_max_drawdown`:   mean |max DD| during hold (positive)\n"
    head += "- `payoff_ratio`:       avg(positive_hold) / avg(|negative_hold|)\n"
    head += "- `eval_score`:         composite ranking score (see main_wave_labels.py)\n\n"
    head += "| " + " | ".join(cols) + " |\n"
    head += "| " + " | ".join("---" for _ in cols) + " |\n"
    rows: list[str] = []
    for r in results:
        cells = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float):
                cells.append(f"{v:+.4f}" if "rate" in c or "ratio" in c or "score" in c
                             or "return" in c or "drawdown" in c
                             else f"{v:.3f}")
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return head + "\n".join(rows) + "\n"


if __name__ == "__main__":
    sys.exit(main())
