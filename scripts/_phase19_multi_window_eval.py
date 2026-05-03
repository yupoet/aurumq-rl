"""Phase 19 Stage B — multi-window OOS validation.

Evaluates fixed candidate model + ensembles on:
  1. 3 quarter blocks (Q3'25, Q4'25, Q1+Apr'26)
  2. each calendar month
  3. 60-trading-day rolling windows, step 20

Reuses Phase 18 cached score matrices in reports/phase18_6h/_scores_cache_*.npy
(regenerated if missing).

Outputs reports/phase19_validation/fixed_window_eval.{md,json}.
"""
from __future__ import annotations

import datetime as dt
import gc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer

from aurumq_rl.backtest import compute_top_k_sharpes, compute_ic, compute_ic_ir, random_baseline
from aurumq_rl.data_loader import (
    FactorPanelLoader,
    UniverseFilter,
    align_panel_to_stock_list,
)


DATA = str(ROOT / "data" / "factor_panel_combined_short_2023_2026.parquet")
VAL_START = dt.date(2025, 7, 1)
VAL_END = dt.date(2026, 4, 24)
TOP_K = 30
FORWARD_PERIOD = 10
DEVICE = "cuda"
CACHE_DIR = ROOT / "reports" / "phase18_6h"
OUT_DIR = ROOT / "reports" / "phase19_validation"

MEMBERS = {
    "16a": ROOT / "models" / "production" / "phase16_16a_drop_mkt_best.zip",
    "17b": ROOT / "models" / "production" / "phase17_17b_drop_mkt_seed1_best.zip",
    "17d": ROOT / "models" / "production" / "phase17_17d_drop_mkt_seed3_best.zip",
    "p18s4": ROOT / "models" / "phase18" / "phase18_18a_drop_mkt_seed4_best.zip",
    "p18s5": ROOT / "models" / "phase18" / "phase18_18b_drop_mkt_seed5_best.zip",
    "p18s6": ROOT / "models" / "phase18" / "phase18_18c_drop_mkt_seed6_best.zip",
}
CACHE_LABELS = {  # Phase 18 used "1818a" etc; map to canonical Phase 19 labels
    "16a": "16a", "17b": "17b", "17d": "17d",
    "p18s4": "1818a", "p18s5": "1818b", "p18s6": "1818c",
}


def _load_canonical_panel():
    meta = json.loads((ROOT / "models" / "production" / "phase16_16a_drop_mkt_best_metadata.json").read_text(encoding="utf-8"))
    factor_names = list(meta["factor_names"])
    stock_codes = list(meta["stock_codes"])
    loader = FactorPanelLoader(parquet_path=DATA)
    panel = loader.load_panel(
        start_date=VAL_START, end_date=VAL_END,
        n_factors=None, forward_period=FORWARD_PERIOD,
        universe_filter=UniverseFilter("main_board_non_st"),
        factor_names=factor_names,
    )
    panel = align_panel_to_stock_list(panel, stock_codes)
    return panel


def _score_member(zip_path: Path, panel_t: torch.Tensor) -> np.ndarray:
    custom_objects = {"rollout_buffer_class": RolloutBuffer}
    model = PPO.load(str(zip_path), device=DEVICE, custom_objects=custom_objects)
    model.policy.eval(); model.policy.to(DEVICE)
    rows = []
    with torch.no_grad():
        for t in range(panel_t.shape[0]):
            feats = model.policy.features_extractor(panel_t[t : t + 1])
            s = model.policy.action_net(feats["per_stock"]).squeeze(-1)
            rows.append(s[0].detach().cpu().numpy())
    out = np.stack(rows, axis=0)
    del model; gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return out


def _per_date_z(scores: np.ndarray) -> np.ndarray:
    out = np.full_like(scores, np.nan, dtype=np.float64)
    for t in range(scores.shape[0]):
        row = scores[t]
        m = np.isfinite(row)
        if m.sum() < 2:
            continue
        std = row[m].std(ddof=1)
        if std < 1e-12:
            continue
        out[t] = np.where(m, (row - row[m].mean()) / std, np.nan)
    return out


def _per_date_rank(scores: np.ndarray) -> np.ndarray:
    out = np.full_like(scores, np.nan, dtype=np.float64)
    for t in range(scores.shape[0]):
        row = scores[t]
        m = np.isfinite(row)
        if m.sum() < 2:
            continue
        ranks = np.full_like(row, np.nan, dtype=np.float64)
        order = np.argsort(np.argsort(row[m]))
        ranks_finite = (order + 0.5) / float(m.sum())
        ranks[m] = ranks_finite
        out[t] = ranks
    return out


def _aggregate(scores_dict: dict[str, np.ndarray], labels: list[str], agg: str) -> np.ndarray:
    matrices = [scores_dict[l] for l in labels]
    if agg == "single":
        return matrices[0].astype(np.float64)
    if agg == "zmean":
        zs = np.stack([_per_date_z(m) for m in matrices], axis=0)
        return np.where(np.isnan(np.nanmean(zs, axis=0)), -np.inf, np.nanmean(zs, axis=0)).astype(np.float64)
    if agg == "zmedian":
        zs = np.stack([_per_date_z(m) for m in matrices], axis=0)
        return np.where(np.isnan(np.nanmedian(zs, axis=0)), -np.inf, np.nanmedian(zs, axis=0)).astype(np.float64)
    if agg == "rankmean":
        rs = np.stack([_per_date_rank(m) for m in matrices], axis=0)
        return np.where(np.isnan(np.nanmean(rs, axis=0)), -np.inf, np.nanmean(rs, axis=0)).astype(np.float64)
    raise ValueError(agg)


def _eval_window(predictions: np.ndarray, returns: np.ndarray, top_k: int, fp: int,
                 idxs: np.ndarray) -> dict | None:
    """Evaluate a contiguous window described by row indices (already filtered
    to exclude trailing fp rows for the global panel; idxs are POSITIONAL into
    the validation panel, must already be eval-able)."""
    if len(idxs) < 2 * fp:
        return None
    p = predictions[idxs]
    r = returns[idxs]
    sharpes = compute_top_k_sharpes(p, r, top_k=top_k, forward_period=fp)
    ic = compute_ic(p, r)
    ic_ir = compute_ic_ir(p, r)
    rb = random_baseline(r, top_k=top_k, n_simulations=50, seed=0, forward_period=fp)
    return {
        "n_dates": int(len(idxs)),
        "top_k_sharpe_adjusted": sharpes["adjusted"],
        "top_k_sharpe_legacy": sharpes["legacy"],
        "top_k_sharpe_non_overlap": sharpes["non_overlap"],
        "ic": ic,
        "ic_ir": ic_ir,
        "random_p50_sharpe_adjusted": rb["p50_sharpe_adjusted"],
        "vs_random_p50_adjusted": sharpes["adjusted"] - rb["p50_sharpe_adjusted"],
    }


def _windows(dates: list[dt.date], fp: int) -> dict[str, list[dict]]:
    """Build the windows per Phase 19 spec. Returns dict: window_name -> [{label, idxs}]"""
    n = len(dates)
    cap = n - fp  # last evaluable index
    eval_idxs = np.arange(0, cap)
    windows = {"quarters": [], "months": [], "rolling60": []}

    # 1) 3 quarter blocks
    quarter_specs = [
        ("2025-Q3 (Jul-Sep)", dt.date(2025, 7, 1), dt.date(2025, 9, 30)),
        ("2025-Q4 (Oct-Dec)", dt.date(2025, 10, 1), dt.date(2025, 12, 31)),
        ("2026-Q1+Apr (Jan-Apr24)", dt.date(2026, 1, 1), dt.date(2026, 4, 24)),
    ]
    for label, start, end in quarter_specs:
        idxs = np.array([
            i for i in eval_idxs
            if start <= dates[i] <= end
        ])
        windows["quarters"].append({"label": label, "idxs": idxs.tolist()})

    # 2) months
    month_keys = sorted({(d.year, d.month) for d in dates[:cap]})
    for y, m in month_keys:
        idxs = np.array([
            i for i in eval_idxs
            if dates[i].year == y and dates[i].month == m
        ])
        if len(idxs) > 0:
            windows["months"].append({
                "label": f"{y:04d}-{m:02d}",
                "idxs": idxs.tolist(),
            })

    # 3) rolling 60-trading-day windows, step 20
    win_size, step = 60, 20
    start_i = 0
    while start_i + win_size <= cap:
        idxs = np.arange(start_i, start_i + win_size)
        # window label = first..last date
        windows["rolling60"].append({
            "label": f"{dates[idxs[0]].isoformat()}..{dates[idxs[-1]].isoformat()}",
            "idxs": idxs.tolist(),
        })
        start_i += step
    return windows


def main() -> int:
    print(f"[phase19_mw] loading panel {VAL_START}..{VAL_END}")
    panel = _load_canonical_panel()
    n_dates, n_stocks, n_factors = panel.factor_array.shape
    print(f"[phase19_mw] panel: dates={n_dates} stocks={n_stocks} factors={n_factors}")
    panel_t = torch.from_numpy(panel.factor_array).to(DEVICE)

    # Load or score each member.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scores: dict[str, np.ndarray] = {}
    for label, zp in MEMBERS.items():
        cache_label = CACHE_LABELS.get(label, label)
        cache_path = CACHE_DIR / f"_scores_cache_{cache_label}.npy"
        if cache_path.exists():
            arr = np.load(cache_path)
            if arr.shape == (n_dates, n_stocks):
                print(f"  [cache] {label} ({cache_label}): loaded {arr.shape}")
                scores[label] = arr
                continue
        print(f"  [score] {label}: scoring...")
        scores[label] = _score_member(zp, panel_t)

    # Variants we evaluate.
    variants = [
        ("single_16a", "single", ["16a"]),
        ("single_p18s4", "single", ["p18s4"]),
        ("ens_rankmean6", "rankmean", ["16a", "17b", "17d", "p18s4", "p18s5", "p18s6"]),
        ("ens_zmean6", "zmean", ["16a", "17b", "17d", "p18s4", "p18s5", "p18s6"]),
        ("ens_zmedian6", "zmedian", ["16a", "17b", "17d", "p18s4", "p18s5", "p18s6"]),
    ]
    aggregated = {name: _aggregate(scores, members, agg) for name, agg, members in variants}

    # Windows.
    windows = _windows(panel.dates, FORWARD_PERIOD)

    # Evaluate every variant on every window.
    results: dict[str, dict[str, list]] = {}
    for vname, _, _ in variants:
        results[vname] = {"quarters": [], "months": [], "rolling60": []}
        for wtype, wlist in windows.items():
            for w in wlist:
                idxs = np.asarray(w["idxs"], dtype=np.int64)
                if len(idxs) < 2 * FORWARD_PERIOD:
                    continue
                row = _eval_window(aggregated[vname], panel.return_array, TOP_K, FORWARD_PERIOD, idxs)
                if row is None:
                    continue
                row["label"] = w["label"]
                results[vname][wtype].append(row)
        print(f"  [done] {vname}: {sum(len(v) for v in results[vname].values())} windows")

    # Write JSON
    out = {
        "val_start": VAL_START.isoformat(),
        "val_end": VAL_END.isoformat(),
        "n_eval_dates": int(n_dates - FORWARD_PERIOD),
        "forward_period": FORWARD_PERIOD,
        "top_k": TOP_K,
        "results": results,
    }
    out_json = OUT_DIR / "fixed_window_eval.json"
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[phase19_mw] wrote {out_json}")

    # Build markdown.
    md = ["# Phase 19 Stage B — fixed-window OOS validation", ""]
    md.append(f"OOS panel: {VAL_START} → {VAL_END} ({n_dates} dates total, {n_dates - FORWARD_PERIOD} eval-able after fp=10).")
    md.append("Primary metric: `vs_random_p50_adjusted`. Rankings within each window are by this metric.")
    md.append("")

    # Quarter blocks
    md.append("## 1. Quarter blocks")
    md.append("")
    md.append("| variant | " + " | ".join([f"{q['label']} adj_S" for q in windows["quarters"]]) + " | "
              + " | ".join([f"{q['label']} vs_p50" for q in windows["quarters"]]) + " | "
              + " | ".join([f"{q['label']} IC" for q in windows["quarters"]]) + " |")
    md.append("|---|" + "|".join(["---:"] * (3 * len(windows["quarters"]))) + "|")
    for vname, _, _ in variants:
        cells = [vname]
        for q in windows["quarters"]:
            row = next((r for r in results[vname]["quarters"] if r["label"] == q["label"]), None)
            cells.append(f"{row['top_k_sharpe_adjusted']:+.3f}" if row else "n/a")
        for q in windows["quarters"]:
            row = next((r for r in results[vname]["quarters"] if r["label"] == q["label"]), None)
            cells.append(f"{row['vs_random_p50_adjusted']:+.3f}" if row else "n/a")
        for q in windows["quarters"]:
            row = next((r for r in results[vname]["quarters"] if r["label"] == q["label"]), None)
            cells.append(f"{row['ic']:+.4f}" if row else "n/a")
        md.append("| " + " | ".join(cells) + " |")
    md.append("")

    # Per-month
    md.append("## 2. Per-month (vs_random_p50_adjusted)")
    md.append("")
    months = sorted({m["label"] for v in results.values() for m in v["months"]})
    md.append("| month | " + " | ".join([v[0] for v in variants]) + " | n_days |")
    md.append("|---|" + "|".join(["---:"] * (len(variants) + 1)) + "|")
    for month in months:
        cells = [month]
        n_days = 0
        for vname, _, _ in variants:
            row = next((r for r in results[vname]["months"] if r["label"] == month), None)
            cells.append(f"{row['vs_random_p50_adjusted']:+.3f}" if row else "n/a")
            if row:
                n_days = max(n_days, row["n_dates"])
        cells.append(str(n_days))
        md.append("| " + " | ".join(cells) + " |")
    md.append("")

    # Rolling-60 summary stats
    md.append("## 3. Rolling 60-day, step 20 (summary stats per variant)")
    md.append("")
    md.append("| variant | n_windows | mean vs_p50 | median | min | max | IC pos rate | win rate vs random |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for vname, _, _ in variants:
        rows = results[vname]["rolling60"]
        if not rows:
            md.append(f"| {vname} | 0 | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        vs_arr = np.array([r["vs_random_p50_adjusted"] for r in rows])
        ic_arr = np.array([r["ic"] for r in rows])
        ic_pos = float((ic_arr > 0).mean())
        win = float((vs_arr > 0).mean())
        md.append(
            f"| {vname} | {len(rows)} | {vs_arr.mean():+.3f} | {np.median(vs_arr):+.3f} | "
            f"{vs_arr.min():+.3f} | {vs_arr.max():+.3f} | {ic_pos:.3f} | {win:.3f} |"
        )

    out_md = OUT_DIR / "fixed_window_eval.md"
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"[phase19_mw] wrote {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
