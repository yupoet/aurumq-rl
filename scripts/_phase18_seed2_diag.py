"""Phase 18 Stage C — seed=2 failure diagnostics.

Compares Phase17C (seed=2, failed) vs Phase17D (seed=3, healthy) vs
Phase16a (seed=42, baseline) on:

1. training trajectory (training_metrics.jsonl)
2. monthly OOS adjusted Sharpe + IC
3. daily top-30 pick overlap matrix
4. score behavior (per-date std)

Outputs a single markdown report. Used by the Phase 18 orchestrator
in Stage C.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import gc

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer

from aurumq_rl.data_loader import (
    FactorPanelLoader,
    UniverseFilter,
    align_panel_to_stock_list,
)


DATA_PATH = ROOT / "data" / "factor_panel_combined_short_2023_2026.parquet"
VAL_START = dt.date(2025, 7, 1)
VAL_END = dt.date(2026, 4, 24)
TOP_K = 30
FORWARD_PERIOD = 10
DEVICE = "cuda"


# Source runs (from prior phases) for trajectory data + score generation.
SUBJECTS = {
    "16a (seed=42)": {
        "run_dir": ROOT / "runs" / "phase16a_fixed_drop_mkt_300k",
        "model_zip": ROOT / "models" / "production" / "phase16_16a_drop_mkt_best.zip",
        "metadata": ROOT / "models" / "production" / "phase16_16a_drop_mkt_best_metadata.json",
        "color": "16a",
    },
    "17C (seed=2 FAIL)": {
        "run_dir": ROOT / "runs" / "phase17_17c_drop_mkt_seed2",
        "model_zip": ROOT / "models" / "production" / "phase17_17c_drop_mkt_seed2_best.zip",
        "metadata": ROOT / "models" / "production" / "phase17_17c_drop_mkt_seed2_best_metadata.json",
        "color": "17C",
    },
    "17D (seed=3)": {
        "run_dir": ROOT / "runs" / "phase17_17d_drop_mkt_seed3",
        "model_zip": ROOT / "models" / "production" / "phase17_17d_drop_mkt_seed3_best.zip",
        "metadata": ROOT / "models" / "production" / "phase17_17d_drop_mkt_seed3_best_metadata.json",
        "color": "17D",
    },
}


def _load_trajectory(run_dir: Path) -> list[dict]:
    p = run_dir / "training_metrics.jsonl"
    if not p.exists():
        return []
    out: list[dict] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _summary_traj(rows: list[dict]) -> dict[str, float]:
    """Summarise iter-level training_metrics. Pulls out rolling mean of
    approx_kl, value_loss, entropy at last iter and a few midpoints."""
    if not rows:
        return {"n_rows": 0}
    last = rows[-1]
    out = {
        "n_rows": len(rows),
        "final_timestep": float(last.get("timestep", 0)),
        "final_approx_kl": float(last.get("extra", {}).get("train/approx_kl", float("nan"))),
        "final_value_loss": float(last.get("extra", {}).get("train/value_loss", float("nan"))),
        "final_explained_var": float(last.get("extra", {}).get("train/explained_variance", float("nan"))),
        "final_clip_frac": float(last.get("extra", {}).get("train/clip_fraction", float("nan"))),
    }
    return out


def _score_model(model_zip: Path, panel_t: torch.Tensor) -> np.ndarray:
    custom_objects = {"rollout_buffer_class": RolloutBuffer}
    model = PPO.load(str(model_zip), device=DEVICE, custom_objects=custom_objects)
    model.policy.eval()
    model.policy.to(DEVICE)
    rows: list[np.ndarray] = []
    with torch.no_grad():
        for t in range(panel_t.shape[0]):
            feats = model.policy.features_extractor(panel_t[t : t + 1])
            s = model.policy.action_net(feats["per_stock"]).squeeze(-1)
            rows.append(s[0].detach().cpu().numpy())
    out = np.stack(rows, axis=0)
    del model
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return out


def _per_month_metrics(predictions: np.ndarray, returns: np.ndarray, dates: list[dt.date],
                       fp: int, top_k: int) -> list[dict]:
    """Compute per-month adjusted Sharpe, mean top-K return, IC."""
    n_dates = predictions.shape[0]
    months: dict[str, list[int]] = {}
    for i, d in enumerate(dates):
        if i >= n_dates - fp:  # exclude trailing rows with no forward return
            continue
        key = f"{d.year:04d}-{d.month:02d}"
        months.setdefault(key, []).append(i)
    out = []
    for month, idxs in sorted(months.items()):
        port = []
        ics = []
        for t in idxs:
            p, r = predictions[t], returns[t]
            mask = np.isfinite(p) & np.isfinite(r)
            if mask.sum() < top_k:
                continue
            order = np.argsort(-p[mask])[:top_k]
            port.append(float(r[mask][order].mean()))
            if mask.sum() >= 2:
                std_p = p[mask].std()
                std_r = r[mask].std()
                if std_p > 1e-12 and std_r > 1e-12:
                    c = np.corrcoef(p[mask], r[mask])[0, 1]
                    if np.isfinite(c):
                        ics.append(float(c))
        if len(port) < 2:
            out.append({"month": month, "n_days": len(idxs), "adj_sharpe": float("nan"),
                        "mean_top_k": float("nan"), "ic_mean": float("nan")})
            continue
        arr = np.asarray(port)
        std = arr.std(ddof=1)
        adj_sharpe = float(arr.mean() / std * np.sqrt(252 / fp)) if std > 1e-12 else float("nan")
        out.append({
            "month": month, "n_days": len(idxs),
            "adj_sharpe": adj_sharpe,
            "mean_top_k": float(arr.mean()),
            "ic_mean": float(np.mean(ics)) if ics else float("nan"),
        })
    return out


def _pick_overlap(scores_a: np.ndarray, scores_b: np.ndarray, top_k: int) -> dict[str, float]:
    """Daily top-K Jaccard between two score matrices."""
    n_dates = min(scores_a.shape[0], scores_b.shape[0])
    overlaps = []
    for t in range(n_dates):
        a, b = scores_a[t], scores_b[t]
        ma = np.isfinite(a)
        mb = np.isfinite(b)
        if ma.sum() < top_k or mb.sum() < top_k:
            continue
        # Argsort then take top
        idx_a = set(np.argsort(-np.where(ma, a, -np.inf))[:top_k].tolist())
        idx_b = set(np.argsort(-np.where(mb, b, -np.inf))[:top_k].tolist())
        inter = len(idx_a & idx_b)
        union = len(idx_a | idx_b)
        overlaps.append(inter / max(union, 1))
    arr = np.asarray(overlaps)
    return {
        "n_days": int(len(arr)),
        "mean_jaccard": float(arr.mean()) if len(arr) else float("nan"),
        "min_jaccard": float(arr.min()) if len(arr) else float("nan"),
        "max_jaccard": float(arr.max()) if len(arr) else float("nan"),
    }


def _score_daily_stats(scores: np.ndarray) -> dict[str, float]:
    """Per-date stats: mean of std, fraction of days where std < 1e-6 (saturation)."""
    finite_mask = np.isfinite(scores)
    stds = []
    sat_count = 0
    for t in range(scores.shape[0]):
        m = finite_mask[t]
        if m.sum() < 2:
            continue
        s = scores[t][m].std()
        stds.append(float(s))
        if s < 1e-6:
            sat_count += 1
    return {
        "mean_daily_std": float(np.mean(stds)) if stds else float("nan"),
        "median_daily_std": float(np.median(stds)) if stds else float("nan"),
        "saturation_days_frac": sat_count / max(len(stds), 1),
    }


def main(out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load validation panel using 16a's factor_names + stock_codes (canonical).
    canonical_meta = json.loads(SUBJECTS["16a (seed=42)"]["metadata"].read_text(encoding="utf-8"))
    factor_names = list(canonical_meta["factor_names"])
    stock_codes = list(canonical_meta["stock_codes"])
    print(f"[diag] loading panel...")
    loader = FactorPanelLoader(parquet_path=DATA_PATH)
    panel = loader.load_panel(
        start_date=VAL_START, end_date=VAL_END,
        n_factors=None, forward_period=FORWARD_PERIOD,
        universe_filter=UniverseFilter("main_board_non_st"),
        factor_names=factor_names,
    )
    panel = align_panel_to_stock_list(panel, stock_codes)
    n_dates, n_stocks, n_factors = panel.factor_array.shape
    print(f"[diag] panel: dates={n_dates} stocks={n_stocks} factors={n_factors}")
    panel_t = torch.from_numpy(panel.factor_array).to(DEVICE)

    # 2. Score all 3 subjects + collect trajectory summaries.
    scores: dict[str, np.ndarray] = {}
    traj: dict[str, dict] = {}
    daily_stats: dict[str, dict] = {}
    monthly: dict[str, list[dict]] = {}
    for label, info in SUBJECTS.items():
        print(f"[diag] scoring {label}...")
        scores[label] = _score_model(info["model_zip"], panel_t)
        traj[label] = _summary_traj(_load_trajectory(info["run_dir"]))
        daily_stats[label] = _score_daily_stats(scores[label])
        monthly[label] = _per_month_metrics(
            scores[label], panel.return_array, panel.dates,
            fp=FORWARD_PERIOD, top_k=TOP_K,
        )

    # 3. Pairwise pick overlap (16a/17C/17D).
    overlap = {
        "17C_vs_16a": _pick_overlap(scores["17C (seed=2 FAIL)"], scores["16a (seed=42)"], TOP_K),
        "17C_vs_17D": _pick_overlap(scores["17C (seed=2 FAIL)"], scores["17D (seed=3)"], TOP_K),
        "17D_vs_16a": _pick_overlap(scores["17D (seed=3)"], scores["16a (seed=42)"], TOP_K),
    }

    # 4. Build report.
    md = ["# Phase 18 — seed=2 failure diagnostics", ""]
    md.append(f"OOS window: {VAL_START} → {VAL_END} ({n_dates} dates, fp={FORWARD_PERIOD}). "
              f"Universe: {n_stocks} stocks, factors: {n_factors}.")
    md.append("")

    # Trajectory summary
    md.append("## 1. Training trajectory summary (last logged iter)")
    md.append("")
    md.append("| run | n_rows | final timestep | approx_kl | value_loss | explained_var | clip_frac |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for label, t in traj.items():
        if not t or t.get("n_rows", 0) == 0:
            md.append(f"| {label} | 0 | n/a | n/a | n/a | n/a | n/a |")
            continue
        md.append(
            f"| {label} | {t['n_rows']} | {t['final_timestep']:.0f} | "
            f"{t['final_approx_kl']:.5f} | {t['final_value_loss']:.5f} | "
            f"{t['final_explained_var']:.4f} | {t['final_clip_frac']:.3f} |"
        )
    md.append("")

    # Score behavior
    md.append("## 2. Per-date score behavior")
    md.append("")
    md.append("| run | mean daily std | median daily std | saturation days frac |")
    md.append("|---|---:|---:|---:|")
    for label, s in daily_stats.items():
        md.append(
            f"| {label} | {s['mean_daily_std']:.5f} | {s['median_daily_std']:.5f} | "
            f"{s['saturation_days_frac']:.3f} |"
        )
    md.append("")

    # Pick overlap
    md.append("## 3. Daily top-30 pick overlap (Jaccard)")
    md.append("")
    md.append("| pair | n_days | mean | min | max |")
    md.append("|---|---:|---:|---:|---:|")
    for pair, o in overlap.items():
        md.append(
            f"| {pair} | {o['n_days']} | {o['mean_jaccard']:.3f} | "
            f"{o['min_jaccard']:.3f} | {o['max_jaccard']:.3f} |"
        )
    md.append("")

    # Per-month metrics
    md.append("## 4. Per-month corrected metrics (adjusted Sharpe / mean top-K return / IC)")
    md.append("")
    months = sorted({m["month"] for rows in monthly.values() for m in rows})
    md.append("| month | " + " | ".join([f"{label} adj_S" for label in monthly]) + " | "
              + " | ".join([f"{label} mean_top_k" for label in monthly]) + " | "
              + " | ".join([f"{label} IC" for label in monthly]) + " |")
    md.append("|---|" + "|".join(["---:"] * (3 * len(monthly))) + "|")
    for month in months:
        cells = [month]
        for label in monthly:
            row = next((r for r in monthly[label] if r["month"] == month), None)
            cells.append(f"{row['adj_sharpe']:+.3f}" if row else "n/a")
        for label in monthly:
            row = next((r for r in monthly[label] if r["month"] == month), None)
            cells.append(f"{row['mean_top_k']*100:+.2f}%" if row else "n/a")
        for label in monthly:
            row = next((r for r in monthly[label] if r["month"] == month), None)
            cells.append(f"{row['ic_mean']:+.4f}" if row else "n/a")
        md.append("| " + " | ".join(cells) + " |")
    md.append("")

    # Heuristic classification.
    md.append("## 5. Classification of seed=2 failure")
    md.append("")
    sd17c = daily_stats["17C (seed=2 FAIL)"]
    sd16a = daily_stats["16a (seed=42)"]
    ovr_17c_16a = overlap["17C_vs_16a"]["mean_jaccard"]
    ovr_17d_16a = overlap["17D_vs_16a"]["mean_jaccard"]

    classification = []
    # Heuristic 1: training collapse (saturation high or std very low)
    if sd17c["saturation_days_frac"] > 0.3 or sd17c["mean_daily_std"] < 0.1 * sd16a["mean_daily_std"]:
        classification.append("optimization failure (score saturation / very low daily std)")
    # Heuristic 2: pick overlap with 16a much lower than 17D's
    if ovr_17c_16a < ovr_17d_16a - 0.05:
        classification.append(f"orthogonal policy (mean Jaccard with 16a "
                              f"{ovr_17c_16a:.3f} vs 17D's {ovr_17d_16a:.3f})")
    # Heuristic 3: monthly Sharpe concentration
    monthly17c = monthly["17C (seed=2 FAIL)"]
    monthly17d = monthly["17D (seed=3)"]
    bad_months_17c = sum(1 for r in monthly17c if (r["adj_sharpe"] is not None and not np.isnan(r["adj_sharpe"]) and r["adj_sharpe"] < 0))
    if monthly17c and bad_months_17c >= len(monthly17c) // 2:
        classification.append(f"persistent monthly weakness ({bad_months_17c}/{len(monthly17c)} months negative)")
    if not classification:
        classification.append("inconclusive (no single signal dominant)")
    md.append("Heuristic flags raised by this diagnostic:")
    for c in classification:
        md.append(f"- {c}")

    # Final classification (one of the categories from instructions §3.C)
    md.append("")
    md.append("**Classification (best fit from §3.C categories):**")
    cat = "inconclusive"
    if any("saturation" in c or "very low" in c for c in classification):
        cat = "optimization failure"
    elif any("orthogonal policy" in c for c in classification):
        cat = "bad but stable local optimum"
    elif any("monthly weakness" in c for c in classification):
        cat = "OOS date/industry concentration problem"
    md.append(f"- {cat}")

    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"[diag] wrote {out_path}")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "reports/phase18_6h/seed2_failure_diagnostics.md"
    main(Path(out))
