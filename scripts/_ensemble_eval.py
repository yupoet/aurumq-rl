"""Phase 18 ensemble evaluator.

Given a list of (label, model_zip, metadata.json) members, score each on
the validation panel, aggregate per-date scores via z-score-mean,
z-score-median, or percentile-rank-mean, run top-K backtest with Phase 16
corrected metrics, and report a unified ranking.

Usage
-----
    python scripts/_ensemble_eval.py \
        --data-path data/factor_panel_combined_short_2023_2026.parquet \
        --val-start 2025-07-01 --val-end 2026-04-24 \
        --top-k 30 --forward-period 10 \
        --device cuda \
        --output-dir reports/phase18_6h \
        --member 16a:models/production/phase16_16a_drop_mkt_best.zip \
        --member 17d:models/production/phase17_17d_drop_mkt_seed3_best.zip \
        --member 17b:models/production/phase17_17b_drop_mkt_seed1_best.zip \
        --variant ens3_zmean:zmean:16a,17d,17b \
        --variant ens3_zmedian:zmedian:16a,17d,17b \
        --variant ens3_rankmean:rankmean:16a,17d,17b \
        --variant ens2_16a_17d_zmean:zmean:16a,17d \
        --suffix stage_a

Writes:
    reports/phase18_6h/ensemble_eval_<suffix>.json
    reports/phase18_6h/ensemble_eval_<suffix>.md

Assumes models share train_stock_codes (Phase 16+17 trained on same data
+ universe). Validates this and warns if not. Each member's score matrix
is aligned to the validation panel BEFORE aggregation.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", required=True, type=Path)
    p.add_argument("--val-start", required=True)
    p.add_argument("--val-end", required=True)
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--forward-period", type=int, default=10)
    p.add_argument("--universe-filter", default="main_board_non_st")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--suffix", default="default")
    p.add_argument("--member", action="append", required=True,
                   help="label:model_zip[:metadata_json] (repeat for each member)")
    p.add_argument("--variant", action="append", required=True,
                   help="name:agg:label1,label2,... (repeat); agg in {zmean,zmedian,rankmean}")
    p.add_argument("--n-random-simulations", type=int, default=100)
    return p.parse_args()


def _load_panel_aligned(args, factor_names: list[str], stock_codes: list[str]):
    """Load panel using exact-order factor_names from one canonical member, then
    align to stock_codes. We assume all members were trained on the same data
    with the same universe filter, so factor_names + stock_codes from any
    member align all of them. We log a hash to confirm."""
    loader = FactorPanelLoader(parquet_path=args.data_path)
    panel = loader.load_panel(
        start_date=dt.date.fromisoformat(args.val_start),
        end_date=dt.date.fromisoformat(args.val_end),
        n_factors=None,
        forward_period=args.forward_period,
        universe_filter=UniverseFilter(args.universe_filter),
        factor_names=factor_names,
    )
    panel = align_panel_to_stock_list(panel, stock_codes)
    return panel


def _score_member(zip_path: Path, panel_t: torch.Tensor, device: str) -> np.ndarray:
    """Run the policy over every date; return (n_dates, n_stocks) score matrix."""
    custom_objects = {"rollout_buffer_class": RolloutBuffer}
    model = PPO.load(str(zip_path), device=device, custom_objects=custom_objects)
    model.policy.eval()
    model.policy.to(device)
    n_dates = panel_t.shape[0]
    rows: list[np.ndarray] = []
    with torch.no_grad():
        for t in range(n_dates):
            feats = model.policy.features_extractor(panel_t[t : t + 1])
            s = model.policy.action_net(feats["per_stock"]).squeeze(-1)
            rows.append(s[0].detach().cpu().numpy())
    out = np.stack(rows, axis=0)
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def _per_date_z(scores: np.ndarray) -> np.ndarray:
    """Per-date z-score: (s - mean) / std over finite stocks each day."""
    out = np.full_like(scores, np.nan, dtype=np.float64)
    for t in range(scores.shape[0]):
        row = scores[t]
        m = np.isfinite(row)
        if m.sum() < 2:
            continue
        std = row[m].std(ddof=1)
        if std < 1e-12:
            # degenerate: fall back to rank
            order = row.argsort().argsort().astype(np.float64)
            n = float(len(row))
            r = (order + 0.5) / n
            r[~m] = np.nan
            # centre & scale to ~unit
            r_finite = r[m]
            out[t] = (r - r_finite.mean()) / (r_finite.std(ddof=1) + 1e-12)
            continue
        out[t] = np.where(m, (row - row[m].mean()) / std, np.nan)
    return out


def _per_date_rank(scores: np.ndarray) -> np.ndarray:
    """Per-date percentile in [0, 1]. NaN-safe."""
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


def _aggregate(per_member: dict[str, np.ndarray], labels: list[str], agg: str) -> np.ndarray:
    """Combine per-member score matrices (already (n_dates, n_stocks))."""
    matrices = [per_member[lbl] for lbl in labels]
    stacked = np.stack(matrices, axis=0)  # (M, T, S)
    if agg == "zmean":
        zs = np.stack([_per_date_z(m) for m in matrices], axis=0)
        with np.errstate(invalid="ignore"):
            out = np.nanmean(zs, axis=0)
    elif agg == "zmedian":
        zs = np.stack([_per_date_z(m) for m in matrices], axis=0)
        with np.errstate(invalid="ignore"):
            out = np.nanmedian(zs, axis=0)
    elif agg == "rankmean":
        rs = np.stack([_per_date_rank(m) for m in matrices], axis=0)
        with np.errstate(invalid="ignore"):
            out = np.nanmean(rs, axis=0)
    else:
        raise ValueError(f"unknown agg {agg!r}")
    # Convert NaN to -inf so they cannot be picked by topK; then convert to
    # float32 finite for backtest helpers. NaN finiteness is checked in
    # backtest.compute_top_k_sharpes via np.isfinite.
    out = np.where(np.isnan(out), -np.inf, out).astype(np.float64)
    return out


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Parse members. Use maxsplit=1 so Windows absolute paths like
    # ``D:\foo\bar.zip`` survive. Optional ``::metadata_json`` suffix is
    # detected via a dedicated separator that won't appear in real paths.
    members: list[tuple[str, Path, Path]] = []
    for spec in args.member:
        if "::" in spec:
            head, meta_str = spec.split("::", 1)
            label, zip_str = head.split(":", 1)
            meta_path = Path(meta_str)
        else:
            label, zip_str = spec.split(":", 1)
            zp = Path(zip_str)
            # auto-derive: <zip>_metadata.json (matches our convention)
            meta_path = Path(str(zp).replace(".zip", "_metadata.json"))
        members.append((label, Path(zip_str), Path(meta_path)))
    print(f"[ensemble] {len(members)} members:")
    for lbl, zp, mp in members:
        exists_z = "OK" if zp.exists() else "MISSING"
        exists_m = "OK" if mp.exists() else "MISSING"
        print(f"  {lbl:>6s}  zip={exists_z}  meta={exists_m}  {zp.name}")

    # Take canonical factor_names + stock_codes from first member with metadata.
    canonical_meta = None
    for _, _, mp in members:
        if mp.exists():
            canonical_meta = json.loads(mp.read_text(encoding="utf-8"))
            break
    if canonical_meta is None:
        raise SystemExit("no member has a metadata.json; cannot align panel")
    canonical_factor_names = list(canonical_meta["factor_names"])
    canonical_stocks = list(canonical_meta["stock_codes"])

    # Sanity: warn if other members' metadata disagree on factor_names or stock_codes
    for label, _, mp in members:
        if not mp.exists():
            print(f"  [warn] {label}: metadata.json missing; assuming canonical layout")
            continue
        m = json.loads(mp.read_text(encoding="utf-8"))
        if list(m["factor_names"]) != canonical_factor_names:
            print(f"  [warn] {label}: factor_names DIFFER from canonical "
                  f"(len {len(m['factor_names'])} vs {len(canonical_factor_names)})")
        if list(m["stock_codes"]) != canonical_stocks:
            print(f"  [warn] {label}: stock_codes DIFFER from canonical "
                  f"(len {len(m['stock_codes'])} vs {len(canonical_stocks)})")

    print(f"[ensemble] canonical: {len(canonical_factor_names)} factors, "
          f"{len(canonical_stocks)} stocks")

    panel = _load_panel_aligned(args, canonical_factor_names, canonical_stocks)
    n_dates, n_stocks, n_factors = panel.factor_array.shape
    print(f"[ensemble] panel: dates={n_dates} stocks={n_stocks} factors={n_factors}")

    panel_t = torch.from_numpy(panel.factor_array).to(args.device)

    # Score each member once, cache to disk for re-use across variants.
    per_member: dict[str, np.ndarray] = {}
    for label, zp, _ in members:
        if not zp.exists():
            print(f"  [skip] {label}: model missing at {zp}")
            continue
        cache_path = args.output_dir / f"_scores_cache_{label}.npy"
        if cache_path.exists():
            arr = np.load(cache_path)
            if arr.shape == (n_dates, n_stocks):
                print(f"  [cache] {label}: loaded {arr.shape}")
                per_member[label] = arr
                continue
        print(f"  [score] {label}: scoring {n_dates} dates...")
        scores = _score_member(zp, panel_t, args.device)
        np.save(cache_path, scores)
        per_member[label] = scores
    if not per_member:
        raise SystemExit("no member produced scores; abort")

    # Also include each individual model as a degenerate "ensemble" so the
    # report has the same metric definition for singletons as for ensembles.
    variants: list[tuple[str, str, list[str]]] = []
    for label in per_member:
        variants.append((f"single_{label}", "passthrough", [label]))
    for spec in args.variant:
        parts = spec.split(":")
        if len(parts) != 3:
            raise SystemExit(f"--variant spec must be name:agg:l1,l2,...; got {spec}")
        name, agg, label_csv = parts
        labels = [s.strip() for s in label_csv.split(",") if s.strip()]
        unknown = [lb for lb in labels if lb not in per_member]
        if unknown:
            print(f"  [skip] variant {name}: missing members {unknown}")
            continue
        variants.append((name, agg, labels))

    # Evaluate each variant.
    rows = []
    for name, agg, labels in variants:
        if agg == "passthrough":
            preds = per_member[labels[0]].astype(np.float64)
        else:
            preds = _aggregate(per_member, labels, agg)
        result, _series = run_backtest_with_series(
            predictions=preds,
            returns=panel.return_array,
            dates=panel.dates,
            top_k=args.top_k,
            n_random_simulations=args.n_random_simulations,
            random_seed=0,
            forward_period=args.forward_period,
        )
        rb = result.random_baseline
        rand_p50_adj = rb.get("p50_sharpe_adjusted", 0.0)
        row = {
            "variant": name,
            "aggregation": agg,
            "members": labels,
            "ic": result.ic,
            "ic_ir": result.ic_ir,
            "top_k_sharpe_adjusted": result.top_k_sharpe_adjusted,
            "top_k_sharpe_legacy": result.top_k_sharpe_legacy,
            "top_k_sharpe_non_overlap": result.top_k_sharpe_non_overlap,
            "top_k_cumret": result.top_k_cumret,
            "random_p50_sharpe_adjusted": rand_p50_adj,
            "random_p95_sharpe_adjusted": rb.get("p95_sharpe_adjusted", 0.0),
            "vs_random_p50_adjusted": result.top_k_sharpe_adjusted - rand_p50_adj,
            "vs_random_p50_non_overlap": result.top_k_sharpe_non_overlap - rb.get("p50_sharpe_non_overlap", 0.0),
            "n_dates": result.n_dates,
            "forward_period": args.forward_period,
        }
        rows.append(row)
        print(
            f"  [eval] {name:30s} adj_S={result.top_k_sharpe_adjusted:+.3f} "
            f"vs_p50_adj={row['vs_random_p50_adjusted']:+.3f} "
            f"non_overlap={result.top_k_sharpe_non_overlap:+.3f} "
            f"IC={result.ic:+.4f}"
        )

    # Sort by vs_random_p50_adjusted, descending.
    rows_sorted = sorted(rows, key=lambda r: -r["vs_random_p50_adjusted"])
    out_json = args.output_dir / f"ensemble_eval_{args.suffix}.json"
    out_json.write_text(json.dumps({
        "val_start": args.val_start,
        "val_end": args.val_end,
        "top_k": args.top_k,
        "forward_period": args.forward_period,
        "rows": rows_sorted,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    # Markdown
    md = [
        f"# Ensemble eval — {args.suffix}",
        "",
        f"- val: {args.val_start} → {args.val_end} ({n_dates} dates, fp={args.forward_period})",
        f"- top-K = {args.top_k}; ranked by `vs_random_p50_adjusted` (Phase 16 corrected metric).",
        f"- Phase 16a baseline: adj_S=+1.593, vs_p50_adj=+0.428, non_overlap=+1.112, IC=+0.0143.",
        "",
        "| variant | aggregation | members | adj S | vs p50 adj | non-overlap | IC | Δ vs 16a |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    P16_VS = 0.428
    for r in rows_sorted:
        members_str = "+".join(r["members"]) if len(r["members"]) > 1 else r["members"][0]
        delta = r["vs_random_p50_adjusted"] - P16_VS
        md.append(
            f"| {r['variant']} | {r['aggregation']} | {members_str} | "
            f"{r['top_k_sharpe_adjusted']:+.3f} | "
            f"{r['vs_random_p50_adjusted']:+.3f} | "
            f"{r['top_k_sharpe_non_overlap']:+.3f} | "
            f"{r['ic']:+.4f} | {delta:+.3f} |"
        )
    out_md = args.output_dir / f"ensemble_eval_{args.suffix}.md"
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"[ensemble] wrote {out_json} and {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
