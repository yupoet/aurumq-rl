"""Evaluate every SB3 checkpoint in a run dir on the same OOS window.

Loops through ``<run-dir>/checkpoints/ppo_*_steps.zip`` plus the optional
``<run-dir>/ppo_final.zip``, runs the same backtest as scripts/eval_backtest.py,
and writes a single compact summary file.

Usage
-----
    python scripts/_eval_all_checkpoints.py \
        --run-dir runs/phase15a_14c_fine_ckpt_700k \
        --data-path data/factor_panel_combined_short_2023_2026.parquet \
        --val-start 2025-07-01 --val-end 2026-04-24 \
        --top-k 30

Outputs ``<run-dir>/oos_sweep.json`` and ``<run-dir>/oos_sweep.md``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import re
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer

from aurumq_rl.backtest import run_backtest_with_series
from aurumq_rl.data_loader import (
    FactorPanelLoader,
    UniverseFilter,
    align_panel_to_stock_list,
)

_CKPT_RE = re.compile(r"ppo_(\d+)_steps\.zip$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--data-path", required=True, type=Path)
    p.add_argument("--val-start", required=True)
    p.add_argument("--val-end", required=True)
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--universe-filter", default="main_board_non_st")
    p.add_argument("--n-random-simulations", type=int, default=100)
    p.add_argument("--device", default="cuda",
                   help="cuda or cpu; cpu lets the eval coexist with a GPU training job")
    return p.parse_args()


def _list_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    out: list[tuple[int, Path]] = []
    ckpt_dir = run_dir / "checkpoints"
    if ckpt_dir.exists():
        for p in ckpt_dir.glob("ppo_*_steps.zip"):
            m = _CKPT_RE.search(p.name)
            if m:
                out.append((int(m.group(1)), p))
    final = run_dir / "ppo_final.zip"
    if final.exists():
        out.append((-1, final))  # -1 = final marker, sorted last
    out.sort()
    return out


def main() -> int:
    args = parse_args()

    meta_path = args.run_dir / "metadata.json"
    if not meta_path.exists():
        print(f"[eval] {meta_path} not found", file=sys.stderr)
        return 1
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    train_stock_codes = meta["stock_codes"]
    factor_count = meta["factor_count"]
    train_factor_names = meta.get("factor_names")
    if isinstance(train_factor_names, list) and train_factor_names:
        train_factor_names = [str(c) for c in train_factor_names]
        print(f"[eval] factor_names from metadata: {len(train_factor_names)} cols")
    else:
        train_factor_names = None
        print("[eval] WARN: metadata.json has no factor_names; falling back to "
              "factor_count — column ORDER may shift if a new prefix was added.")
    forward_period = int(meta.get("forward_period", 10))
    print(f"[eval] forward_period={forward_period}")

    checkpoints = _list_checkpoints(args.run_dir)
    if not checkpoints:
        print(f"[eval] no checkpoints in {args.run_dir}", file=sys.stderr)
        return 2
    print(f"[eval] {len(checkpoints)} checkpoints to evaluate")

    # Load + align panel ONCE.
    loader = FactorPanelLoader(parquet_path=args.data_path)
    panel = loader.load_panel(
        start_date=dt.date.fromisoformat(args.val_start),
        end_date=dt.date.fromisoformat(args.val_end),
        n_factors=factor_count if train_factor_names is None else None,
        forward_period=forward_period,
        universe_filter=UniverseFilter(args.universe_filter),
        factor_names=train_factor_names,
    )
    panel = align_panel_to_stock_list(panel, train_stock_codes)
    n_dates, n_stocks, n_factors = panel.factor_array.shape
    print(f"[eval] panel: dates={n_dates} stocks={n_stocks} factors={n_factors}")

    panel_t = torch.from_numpy(panel.factor_array).to(args.device)

    custom_objects = {"rollout_buffer_class": RolloutBuffer}

    rows = []
    for step, ckpt_path in checkpoints:
        try:
            model = PPO.load(str(ckpt_path), device=args.device,
                             custom_objects=custom_objects)
            model.policy.eval()
            model.policy.to(args.device)
            scores = []
            with torch.no_grad():
                for t in range(n_dates):
                    feats = model.policy.features_extractor(panel_t[t : t + 1])
                    s = model.policy.action_net(feats["per_stock"]).squeeze(-1)
                    scores.append(s[0].detach().cpu().numpy())
            preds = np.stack(scores, axis=0)
            result, _series = run_backtest_with_series(
                predictions=preds,
                returns=panel.return_array,
                dates=panel.dates,
                top_k=args.top_k,
                n_random_simulations=args.n_random_simulations,
                random_seed=0,
                forward_period=forward_period,
            )
            label = f"{step}" if step >= 0 else "final"
            rb = result.random_baseline
            rand_p50_adj = rb.get("p50_sharpe_adjusted", 0.0)
            rand_p50_legacy = rb.get("p50_sharpe", 0.0)
            rand_p50_nov = rb.get("p50_sharpe_non_overlap", 0.0)
            rows.append({
                "step": step,
                "label": label,
                "checkpoint": str(ckpt_path),
                "ic": result.ic,
                "ic_ir": result.ic_ir,
                # adjusted is the primary metric; keep all three so we can
                # compare across regimes without re-running.
                "top_k_sharpe_adjusted": result.top_k_sharpe_adjusted,
                "top_k_sharpe_legacy": result.top_k_sharpe_legacy,
                "top_k_sharpe_non_overlap": result.top_k_sharpe_non_overlap,
                "top_k_cumret": result.top_k_cumret,
                "random_p50_sharpe_adjusted": rand_p50_adj,
                "random_p95_sharpe_adjusted": rb.get("p95_sharpe_adjusted", 0.0),
                "random_p50_sharpe_legacy": rand_p50_legacy,
                "random_p50_sharpe_non_overlap": rand_p50_nov,
                "vs_random_p50_adjusted": result.top_k_sharpe_adjusted - rand_p50_adj,
                "vs_random_p50_non_overlap": result.top_k_sharpe_non_overlap - rand_p50_nov,
                "forward_period": forward_period,
            })
            print(f"[eval] {label:>7s}: IC={result.ic:+.4f} "
                  f"adj_S={result.top_k_sharpe_adjusted:+.3f} "
                  f"vs p50_adj={rows[-1]['vs_random_p50_adjusted']:+.3f} "
                  f"non_overlap={result.top_k_sharpe_non_overlap:+.3f}")
        except Exception as e:
            label = f"{step}" if step >= 0 else "final"
            print(f"[eval] {label}: FAILED - {e!r}")
            rows.append({"step": step, "label": label, "error": repr(e)})
        finally:
            # Each PPO.load allocates a fresh RolloutBuffer of size
            # (n_steps, n_envs, *obs_shape) = ~64 GiB host RAM for our
            # 1024 * 16 * 3014 * 343 * float32. Without explicit GC the
            # second iteration's allocation hits MemoryError. Free the
            # model and force a collection between iterations.
            try:
                del model
            except UnboundLocalError:
                pass
            gc.collect()
            if args.device == "cuda":
                torch.cuda.empty_cache()

    out_json = args.run_dir / "oos_sweep.json"
    out_json.write_text(json.dumps({
        "run_dir": str(args.run_dir),
        "val_start": args.val_start,
        "val_end": args.val_end,
        "top_k": args.top_k,
        "n_dates": n_dates,
        "forward_period": forward_period,
        "rows": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[eval] wrote {out_json}")

    # Markdown table
    valid = [r for r in rows if "error" not in r]
    if not valid:
        print("[eval] no valid rows; skipping md")
        return 0
    # Phase 16: rank by adjusted Sharpe (the corrected metric).
    best = max(valid, key=lambda r: r["top_k_sharpe_adjusted"])
    md = [
        f"# OOS sweep — {args.run_dir.name}",
        "",
        f"- val window: {args.val_start} → {args.val_end} ({n_dates} dates, "
        f"forward_period={forward_period})",
        f"- top-K = {args.top_k}",
        f"- best (by adjusted Sharpe): step={best['label']}  "
        f"adj_Sharpe={best['top_k_sharpe_adjusted']:+.3f}  "
        f"vs random p50 adj={best['vs_random_p50_adjusted']:+.3f}",
        "",
        "Adjusted Sharpe = `mean / std * sqrt(252 / forward_period)`. "
        "Non-overlap subsamples every fp-th day. Legacy `sqrt(252)` is "
        "shown only for backwards comparison.",
        "",
        "| step | IC | IR | adj Sharpe | vs random p50 adj | non-overlap | legacy |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in valid:
        md.append(
            f"| {r['label']} | {r['ic']:+.4f} | {r['ic_ir']:+.3f} | "
            f"{r['top_k_sharpe_adjusted']:+.3f} | "
            f"{r['vs_random_p50_adjusted']:+.3f} | "
            f"{r['top_k_sharpe_non_overlap']:+.3f} | "
            f"{r['top_k_sharpe_legacy']:+.3f} |"
        )
    out_md = args.run_dir / "oos_sweep.md"
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"[eval] wrote {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
