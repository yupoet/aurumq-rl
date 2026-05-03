"""Phase 19 Stage D — ensemble ablation + seed=4 forensics."""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import polars as pl

from aurumq_rl.backtest import compute_top_k_sharpes, compute_ic, random_baseline
from aurumq_rl.data_loader import (
    FactorPanelLoader,
    UniverseFilter,
    align_panel_to_stock_list,
)


DATA = str(ROOT / "data" / "factor_panel_combined_short_2023_2026.parquet")
VAL_START = dt.date(2025, 7, 1)
VAL_END = dt.date(2026, 4, 24)
TOP_K = 30
HOLD_DAYS = 10
CACHE_DIR = ROOT / "reports" / "phase18_6h"
OUT_DIR = ROOT / "reports" / "phase19_validation"

ALL_SEEDS = {
    "16a": "16a", "17b": "17b", "17c": "17c", "17d": "17d",
    "p18s4": "1818a", "p18s5": "1818b", "p18s6": "1818c",
}
ELIGIBLE_6 = ["16a", "17b", "17d", "p18s4", "p18s5", "p18s6"]


def _per_date_rank(scores):
    out = np.full_like(scores, np.nan, dtype=np.float64)
    for t in range(scores.shape[0]):
        row = scores[t]
        m = np.isfinite(row)
        if m.sum() < 2:
            continue
        ranks = np.full_like(row, np.nan, dtype=np.float64)
        order = np.argsort(np.argsort(row[m]))
        ranks[m] = (order + 0.5) / float(m.sum())
        out[t] = ranks
    return out


def _aggregate_rankmean(scores_dict, labels):
    rs = np.stack([_per_date_rank(scores_dict[l]) for l in labels], axis=0)
    rm = np.nanmean(rs, axis=0)
    return np.where(np.isnan(rm), -np.inf, rm).astype(np.float64)


def _eval_simple(predictions, returns, top_k, fp):
    sharpes = compute_top_k_sharpes(predictions, returns, top_k=top_k, forward_period=fp)
    ic = compute_ic(predictions, returns)
    rb = random_baseline(returns, top_k=top_k, n_simulations=50, seed=0, forward_period=fp)
    return {
        "top_k_sharpe_adjusted": sharpes["adjusted"],
        "top_k_sharpe_non_overlap": sharpes["non_overlap"],
        "ic": ic,
        "random_p50_sharpe_adjusted": rb["p50_sharpe_adjusted"],
        "vs_random_p50_adjusted": sharpes["adjusted"] - rb["p50_sharpe_adjusted"],
    }


def _topk_picks(scores_t: np.ndarray, top_k: int) -> list[int]:
    s = scores_t.copy()
    s[~np.isfinite(s)] = -np.inf
    order = np.argsort(-s)[:top_k]
    return [int(j) for j in order]


def main() -> int:
    print("[phase19_d] loading panel + scores")
    meta = json.loads((ROOT / "models" / "production" / "phase16_16a_drop_mkt_best_metadata.json").read_text(encoding="utf-8"))
    factor_names = list(meta["factor_names"])
    stock_codes = list(meta["stock_codes"])
    loader = FactorPanelLoader(parquet_path=DATA)
    panel = loader.load_panel(
        start_date=VAL_START, end_date=VAL_END,
        n_factors=None, forward_period=HOLD_DAYS,
        universe_filter=UniverseFilter("main_board_non_st"),
        factor_names=factor_names,
    )
    panel = align_panel_to_stock_list(panel, stock_codes)
    n_dates = panel.factor_array.shape[0]

    scores = {}
    for label, cache_label in ALL_SEEDS.items():
        p = CACHE_DIR / f"_scores_cache_{cache_label}.npy"
        if p.exists():
            scores[label] = np.load(p)

    # ===== Ablation =====
    ablations = []
    # Baseline rankmean6
    base = _aggregate_rankmean(scores, ELIGIBLE_6)
    ablations.append({"variant": "rankmean6 (baseline)", "members": ELIGIBLE_6, **_eval_simple(base, panel.return_array, TOP_K, HOLD_DAYS)})
    # Leave-one-out
    for leave in ELIGIBLE_6:
        members = [l for l in ELIGIBLE_6 if l != leave]
        agg = _aggregate_rankmean(scores, members)
        ablations.append({"variant": f"rankmean5_minus_{leave}", "members": members, **_eval_simple(agg, panel.return_array, TOP_K, HOLD_DAYS)})
    # Top 3 strong seeds
    top3 = ["p18s4", "p18s5", "p18s6"]
    ablations.append({"variant": "rankmean3_top3_strong", "members": top3, **_eval_simple(_aggregate_rankmean(scores, top3), panel.return_array, TOP_K, HOLD_DAYS)})
    # Add seed=2 (sensitivity)
    if "17c" in scores:
        members = ELIGIBLE_6 + ["17c"]
        ablations.append({"variant": "rankmean7_with_seed2_17c", "members": members, **_eval_simple(_aggregate_rankmean(scores, members), panel.return_array, TOP_K, HOLD_DAYS)})
    # rankmean8 (everything we have including failures)
    if "17c" in scores:
        members = list(scores.keys())
        ablations.append({"variant": "rankmean_all_loaded", "members": members, **_eval_simple(_aggregate_rankmean(scores, members), panel.return_array, TOP_K, HOLD_DAYS)})

    # Sort
    ablations_sorted = sorted(ablations, key=lambda r: -r["vs_random_p50_adjusted"])
    print("[phase19_d] ablation results:")
    for r in ablations_sorted:
        print(f"  {r['variant']:36s}  vs_p50_adj={r['vs_random_p50_adjusted']:+.3f}  IC={r['ic']:+.4f}")

    out_obj = {"ablations": ablations_sorted, "val_start": VAL_START.isoformat(), "val_end": VAL_END.isoformat()}
    (OUT_DIR / "ensemble_ablation.json").write_text(json.dumps(out_obj, indent=2, ensure_ascii=False), encoding="utf-8")

    md_lines = ["# Phase 19 Stage D-1 — ensemble ablation", ""]
    md_lines.append(f"OOS panel: {VAL_START} → {VAL_END} ({n_dates} dates, fp={HOLD_DAYS}). Aggregation: per-date percentile rank-mean.")
    md_lines.append("")
    md_lines.append("| variant | members | adj S | vs p50 adj | non-overlap | IC | Δ vs base |")
    md_lines.append("|---|---|---:|---:|---:|---:|---:|")
    base_vs = next(r for r in ablations_sorted if r["variant"] == "rankmean6 (baseline)")["vs_random_p50_adjusted"]
    for r in ablations_sorted:
        members_str = "+".join(r["members"]) if isinstance(r["members"], list) else str(r["members"])
        delta = r["vs_random_p50_adjusted"] - base_vs
        md_lines.append(
            f"| {r['variant']} | {members_str} | "
            f"{r['top_k_sharpe_adjusted']:+.3f} | {r['vs_random_p50_adjusted']:+.3f} | "
            f"{r['top_k_sharpe_non_overlap']:+.3f} | {r['ic']:+.4f} | {delta:+.3f} |"
        )
    (OUT_DIR / "ensemble_ablation.md").write_text("\n".join(md_lines), encoding="utf-8")
    print("[phase19_d] wrote ensemble_ablation.{md,json}")

    # ===== seed=4 forensics =====
    print("[phase19_d] seed=4 forensics")

    # Load industry mapping per stock
    raw = pl.read_parquet(DATA, columns=["ts_code", "trade_date", "industry_code"])
    raw = raw.filter(
        (pl.col("trade_date") >= VAL_START) & (pl.col("trade_date") <= VAL_END)
    )
    ind_map_df = raw.group_by("ts_code").agg(pl.col("industry_code").last())
    ind_map = dict(zip(ind_map_df["ts_code"].to_list(), ind_map_df["industry_code"].to_list()))
    train_industry = [ind_map.get(c, "UNKNOWN") for c in stock_codes]
    train_industry = ["UNKNOWN" if (i is None or str(i) == "NaN") else str(i) for i in train_industry]

    # Top-30 daily picks for: single seed=4, ens6-baseline, ens5-without-seed4
    p18s4_scores = scores["p18s4"]
    ens6_scores = base
    members_no_seed4 = [l for l in ELIGIBLE_6 if l != "p18s4"]
    ens5_scores = _aggregate_rankmean(scores, members_no_seed4)

    def _picks_per_date(score_mat):
        return [_topk_picks(score_mat[t], TOP_K) for t in range(n_dates)]

    picks_seed4 = _picks_per_date(p18s4_scores)
    picks_ens6 = _picks_per_date(ens6_scores)
    picks_ens5 = _picks_per_date(ens5_scores)

    # Jaccard per date: seed4 vs ens6
    jacc_s4_ens6 = []
    jacc_ens5_ens6 = []
    for t in range(n_dates - HOLD_DAYS):
        a = set(picks_seed4[t]); b = set(picks_ens6[t]); c = set(picks_ens5[t])
        if a or b:
            jacc_s4_ens6.append(len(a & b) / max(len(a | b), 1))
        if c or b:
            jacc_ens5_ens6.append(len(c & b) / max(len(c | b), 1))
    jacc_s4_ens6_arr = np.asarray(jacc_s4_ens6)
    jacc_ens5_ens6_arr = np.asarray(jacc_ens5_ens6)

    # Per-month adjusted Sharpe for the 3 candidates and difference
    months = {}
    for t in range(n_dates - HOLD_DAYS):
        d = panel.dates[t]
        key = f"{d.year:04d}-{d.month:02d}"
        months.setdefault(key, []).append(t)

    def _eval_window(score_mat, idxs):
        if len(idxs) < 2:
            return None
        p = score_mat[idxs]
        r = panel.return_array[idxs]
        sh = compute_top_k_sharpes(p, r, top_k=TOP_K, forward_period=HOLD_DAYS)
        return sh["adjusted"]

    monthly_table = []
    total_lift_seed4_vs_ens5 = 0.0
    monthly_lifts = []
    for key, idxs in sorted(months.items()):
        s_seed4 = _eval_window(p18s4_scores, idxs)
        s_ens6 = _eval_window(ens6_scores, idxs)
        s_ens5 = _eval_window(ens5_scores, idxs)
        if any(x is None for x in (s_seed4, s_ens6, s_ens5)):
            continue
        lift = s_ens6 - s_ens5  # how much seed=4 inclusion helped
        monthly_lifts.append(lift)
        total_lift_seed4_vs_ens5 += lift
        monthly_table.append({"month": key, "n_days": len(idxs), "seed4_adj": s_seed4,
                              "ens6_adj": s_ens6, "ens5_no_seed4_adj": s_ens5,
                              "lift_from_seed4": lift})

    # Industry concentration for seed=4 picks
    ind_count_seed4 = {}
    total_picks_seed4 = 0
    for t_idxs in picks_seed4[: n_dates - HOLD_DAYS]:
        for j in t_idxs:
            ind = train_industry[j]
            ind_count_seed4[ind] = ind_count_seed4.get(ind, 0) + 1
            total_picks_seed4 += 1
    ind_count_ens6 = {}
    total_picks_ens6 = 0
    for t_idxs in picks_ens6[: n_dates - HOLD_DAYS]:
        for j in t_idxs:
            ind = train_industry[j]
            ind_count_ens6[ind] = ind_count_ens6.get(ind, 0) + 1
            total_picks_ens6 += 1
    top_ind_s4 = sorted(ind_count_seed4.items(), key=lambda kv: -kv[1])[:10]
    top_ind_ens6 = sorted(ind_count_ens6.items(), key=lambda kv: -kv[1])[:10]

    # Seed=4 contribution concentration: how much of total lift comes from
    # any single month?
    if monthly_lifts:
        max_month_lift = max(monthly_lifts)
        max_month_share = max_month_lift / total_lift_seed4_vs_ens5 if abs(total_lift_seed4_vs_ens5) > 1e-9 else 0.0
    else:
        max_month_lift = 0.0
        max_month_share = 0.0

    forensics_obj = {
        "jaccard_seed4_vs_ens6_mean": float(jacc_s4_ens6_arr.mean()),
        "jaccard_ens5_vs_ens6_mean": float(jacc_ens5_ens6_arr.mean()),
        "monthly_lifts": monthly_table,
        "total_lift_from_seed4": total_lift_seed4_vs_ens5,
        "max_single_month_lift": max_month_lift,
        "max_single_month_share_of_total_lift": max_month_share,
        "industry_top10_seed4": [{"industry": k, "picks": v, "pct": v/max(total_picks_seed4,1)} for k, v in top_ind_s4],
        "industry_top10_ens6": [{"industry": k, "picks": v, "pct": v/max(total_picks_ens6,1)} for k, v in top_ind_ens6],
    }
    (OUT_DIR / "seed4_forensics.json").write_text(json.dumps(forensics_obj, indent=2, ensure_ascii=False), encoding="utf-8")

    md = ["# Phase 19 Stage D-2 — seed=4 forensics", ""]
    md.append("Question: how much of the rankmean6 ensemble lift over rankmean5-without-seed4 is attributable to seed=4? Is seed=4 a single-month/single-industry concentration?")
    md.append("")
    md.append("## Daily top-30 Jaccard")
    md.append("")
    md.append(f"- seed=4 vs ens6 baseline: mean Jaccard = {jacc_s4_ens6_arr.mean():.3f}")
    md.append(f"- ens5 (without seed=4) vs ens6 baseline: mean Jaccard = {jacc_ens5_ens6_arr.mean():.3f}")
    md.append(f"- (seed=4 alone shares only {jacc_s4_ens6_arr.mean()*100:.1f}% of ens6's daily picks; ens5-without-4 shares {jacc_ens5_ens6_arr.mean()*100:.1f}%.)")
    md.append("")
    md.append("## Per-month lift from including seed=4 (ens6 − ens5)")
    md.append("")
    md.append("| month | n_days | seed=4 alone adj_S | ens6 adj_S | ens5 (no seed4) adj_S | lift from seed=4 |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for r in monthly_table:
        md.append(f"| {r['month']} | {r['n_days']} | {r['seed4_adj']:+.3f} | {r['ens6_adj']:+.3f} | {r['ens5_no_seed4_adj']:+.3f} | {r['lift_from_seed4']:+.3f} |")
    md.append("")
    md.append(f"Total lift from seed=4 (sum over months): {total_lift_seed4_vs_ens5:+.3f}")
    md.append(f"Max single-month lift: {max_month_lift:+.3f}")
    md.append(f"Max single-month share of total lift: {max_month_share*100:.1f}%")
    md.append("")
    md.append("## Industry concentration of seed=4 vs ens6 daily picks")
    md.append("")
    md.append("Top 10 industries by pick count (validation window):")
    md.append("")
    md.append("| seed=4 industry | seed=4 picks | seed=4 pct | | ens6 industry | ens6 picks | ens6 pct |")
    md.append("|---|---:|---:|---|---|---:|---:|")
    for i in range(max(len(top_ind_s4), len(top_ind_ens6))):
        a = top_ind_s4[i] if i < len(top_ind_s4) else (None, 0)
        b = top_ind_ens6[i] if i < len(top_ind_ens6) else (None, 0)
        a_str = f"| {a[0] or 'n/a'} | {a[1]} | {a[1]/max(total_picks_seed4,1)*100:.2f}% |" if a[0] else "| | | |"
        b_str = f"| {b[0] or 'n/a'} | {b[1]} | {b[1]/max(total_picks_ens6,1)*100:.2f}% |" if b[0] else "| | | |"
        md.append(a_str + " " + b_str)
    md.append("")
    md.append("## Verdict")
    md.append("")
    if max_month_share >= 0.35:
        md.append(f"⚠ seed=4's lift is concentrated: {max_month_share*100:.1f}% from a single month (threshold = 35%). Phase 18 ensemble confidence should be downgraded.")
    else:
        md.append(f"✓ seed=4's lift is distributed across months (max single-month share = {max_month_share*100:.1f}% < 35% threshold). The Phase 18 ensemble's reliance on seed=4 is structurally healthy, not a single-period accident.")

    (OUT_DIR / "seed4_forensics.md").write_text("\n".join(md), encoding="utf-8")
    print("[phase19_d] wrote seed4_forensics.{md,json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
