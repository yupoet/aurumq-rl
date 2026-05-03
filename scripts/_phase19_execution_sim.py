"""Phase 19 Stage C — execution constraint simulation.

For each candidate model/ensemble:
  - signal day t -> T+1 entry on close[t+1] (proxy for next-day open)
  - 10-day hold -> exit on close[t+11]
  - filters at entry: skip ST, suspended (vol=0), IPO<60 days, limit-up at t+1
  - limit-down at exit: defer up to 5 days; if still locked, exit at the
    last available close
  - costs: 30 / 60 / 100 bps round-trip (entry + exit) applied to gross
  - two re-balancing modes:
    1. non-overlap: rebalance every 10 trading days
    2. daily 10-sleeve: 10 parallel portfolios offset by 1 day each
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch  # noqa: F401

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
MAX_DEFER = 5
NEW_STOCK_PROTECT = 60
LIMIT_UP_THRESHOLD = 0.099
LIMIT_DN_THRESHOLD = -0.099
COSTS_BPS_ROUND_TRIP = (30, 60, 100)
CACHE_DIR = ROOT / "reports" / "phase18_6h"
OUT_DIR = ROOT / "reports" / "phase19_validation"

MEMBERS_FOR_AGG = {
    "16a": "16a", "17b": "17b", "17d": "17d",
    "p18s4": "1818a", "p18s5": "1818b", "p18s6": "1818c",
}


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


def _build_aggregations(scores: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {
        "single_16a": scores["16a"].astype(np.float64),
        "single_p18s4": scores["p18s4"].astype(np.float64),
    }
    labels = ["16a", "17b", "17d", "p18s4", "p18s5", "p18s6"]
    rs = np.stack([_per_date_rank(scores[l]) for l in labels], axis=0)
    rm = np.nanmean(rs, axis=0)
    out["ens_rankmean6"] = np.where(np.isnan(rm), -np.inf, rm).astype(np.float64)
    return out


def _eligible_at_entry(panel, t_entry, j) -> bool:
    if t_entry >= panel.is_st_array.shape[0]:
        return False
    if panel.is_st_array[t_entry, j]:
        return False
    if panel.is_suspended_array[t_entry, j]:
        return False
    if panel.days_since_ipo_array[t_entry, j] < NEW_STOCK_PROTECT:
        return False
    pc = float(panel.pct_change_array[t_entry, j])
    if not np.isfinite(pc):
        return False
    if pc >= LIMIT_UP_THRESHOLD:
        return False
    return True


def _exit_close_index(panel, t_exit_target, j):
    n_dates = panel.is_st_array.shape[0]
    for d in range(MAX_DEFER + 1):
        ti = t_exit_target + d
        if ti >= n_dates:
            return min(t_exit_target, n_dates - 1)
        if panel.is_suspended_array[ti, j]:
            continue
        pc = float(panel.pct_change_array[ti, j])
        if np.isfinite(pc) and pc <= LIMIT_DN_THRESHOLD:
            continue
        return ti
    return min(t_exit_target + MAX_DEFER, n_dates - 1)


def _trade_return(panel, t_signal, j):
    n_dates = panel.is_st_array.shape[0]
    t_entry = t_signal + 1
    t_exit_target = t_entry + HOLD_DAYS
    if t_entry >= n_dates or t_exit_target >= n_dates:
        return None, {"dropout": "panel_end"}
    if not _eligible_at_entry(panel, t_entry, j):
        return None, {"dropout": "entry_filter"}
    t_exit_actual = _exit_close_index(panel, t_exit_target, j)
    if t_exit_actual is None:
        return None, {"dropout": "no_exit"}
    s = 0.0
    for ti in range(t_entry + 1, t_exit_actual + 1):
        pc = float(panel.pct_change_array[ti, j])
        if not np.isfinite(pc):
            return None, {"dropout": "missing_pct"}
        pc = max(min(pc, 0.105), -0.105)
        s += np.log(1.0 + pc)
    deferred = t_exit_actual - t_exit_target
    return s, {"dropout": None, "deferred_days": deferred,
               "hold_days": t_exit_actual - t_entry}


def _topk_picks(scores_t, panel, t_entry, top_k):
    s = scores_t.copy()
    s[~np.isfinite(s)] = -np.inf
    order = np.argsort(-s)
    out = []
    for j in order:
        if len(out) >= top_k:
            break
        if not np.isfinite(scores_t[j]):
            continue
        if _eligible_at_entry(panel, t_entry, int(j)):
            out.append(int(j))
    return out


def _simulate_non_overlap(panel, scores, top_k, cost_bps, rebalance_step=HOLD_DAYS):
    n_dates = panel.return_array.shape[0]
    eval_cap = n_dates - HOLD_DAYS - 1
    if eval_cap <= 0:
        return {"error": "panel too short"}
    period_returns = []
    failed_entries = 0
    deferred_exits = 0
    total_attempts = 0
    turnovers = []
    prev_picks = set()
    for t_signal in range(0, eval_cap, rebalance_step):
        picks = _topk_picks(scores[t_signal], panel, t_signal + 1, top_k)
        if not picks:
            continue
        gross_log = []
        for j in picks:
            r, info = _trade_return(panel, t_signal, j)
            total_attempts += 1
            if r is None:
                failed_entries += 1
                continue
            gross_log.append(r)
            if info.get("deferred_days", 0) > 0:
                deferred_exits += 1
        if not gross_log:
            continue
        gross = float(np.mean(gross_log))
        cost = (cost_bps / 10000.0) * 1.0
        net = gross - cost
        period_returns.append(net)
        cur = set(picks)
        if prev_picks:
            inter = len(prev_picks & cur)
            turnover = 1.0 - inter / float(len(cur))
            turnovers.append(turnover)
        prev_picks = cur
    if len(period_returns) < 2:
        return {"error": "too few rebalances"}
    arr = np.asarray(period_returns)
    sharpe_adj = float(arr.mean() / arr.std(ddof=1) * np.sqrt(252.0 / HOLD_DAYS)) if arr.std(ddof=1) > 1e-12 else 0.0
    eq = np.exp(np.cumsum(arr))
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    return {
        "n_rebalances": int(len(arr)),
        "post_cost_mean_per_period": float(arr.mean()),
        "post_cost_std_per_period": float(arr.std(ddof=1)),
        "post_cost_sharpe_adjusted": sharpe_adj,
        "post_cost_total_log_return": float(arr.sum()),
        "post_cost_max_drawdown": float(dd.min()),
        "failed_entries_total": int(failed_entries),
        "total_entry_attempts": int(total_attempts),
        "failed_entry_rate": float(failed_entries / max(total_attempts, 1)),
        "deferred_exits_total": int(deferred_exits),
        "deferred_exit_rate": float(deferred_exits / max(total_attempts - failed_entries, 1)),
        "avg_turnover": float(np.mean(turnovers)) if turnovers else 0.0,
    }


def _simulate_daily_10sleeve(panel, scores, top_k, cost_bps, hold_days=HOLD_DAYS):
    sleeve_results = []
    for offset in range(hold_days):
        n_dates = panel.return_array.shape[0]
        eval_cap = n_dates - hold_days - 1
        if eval_cap <= 0:
            continue
        period_returns = []
        for t_signal in range(offset, eval_cap, hold_days):
            picks = _topk_picks(scores[t_signal], panel, t_signal + 1, top_k)
            if not picks:
                continue
            gross_log = []
            for j in picks:
                r, _ = _trade_return(panel, t_signal, j)
                if r is None:
                    continue
                gross_log.append(r)
            if not gross_log:
                continue
            gross = float(np.mean(gross_log))
            cost = cost_bps / 10000.0
            period_returns.append(gross - cost)
        if len(period_returns) >= 2:
            arr = np.asarray(period_returns)
            sleeve_results.append(arr)
    if not sleeve_results:
        return {"error": "no sleeves"}
    flat = np.concatenate(sleeve_results)
    sharpe_adj = float(flat.mean() / flat.std(ddof=1) * np.sqrt(252.0 / hold_days)) if flat.std(ddof=1) > 1e-12 else 0.0
    eq = np.exp(np.cumsum(flat))
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    return {
        "n_sleeves": len(sleeve_results),
        "n_period_returns_pooled": int(len(flat)),
        "post_cost_mean_per_period": float(flat.mean()),
        "post_cost_std_per_period": float(flat.std(ddof=1)),
        "post_cost_sharpe_adjusted": sharpe_adj,
        "post_cost_max_drawdown": float(dd.min()),
    }


def _random_baseline_sim(panel, top_k, cost_bps, n_sims=20, seed=0):
    rng = np.random.default_rng(seed)
    n_dates, n_stocks = panel.return_array.shape
    sharpes = []
    for _ in range(n_sims):
        s = rng.normal(size=(n_dates, n_stocks))
        out = _simulate_non_overlap(panel, s, top_k, cost_bps)
        if "error" in out:
            continue
        sharpes.append(out["post_cost_sharpe_adjusted"])
    if not sharpes:
        return {"mean_sharpe": float("nan"), "p50_sharpe": float("nan"), "p95_sharpe": float("nan")}
    a = np.asarray(sharpes)
    return {
        "n_sims": int(len(sharpes)),
        "mean_sharpe": float(a.mean()),
        "p50_sharpe": float(np.percentile(a, 50)),
        "p95_sharpe": float(np.percentile(a, 95)),
    }


def main() -> int:
    print(f"[phase19_exec] loading panel {VAL_START}..{VAL_END}")
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
    n_dates, n_stocks, _ = panel.factor_array.shape
    print(f"[phase19_exec] panel: dates={n_dates} stocks={n_stocks}")

    scores = {}
    for label, cache_label in MEMBERS_FOR_AGG.items():
        p = CACHE_DIR / f"_scores_cache_{cache_label}.npy"
        if not p.exists():
            print(f"[error] missing {p}")
            return 1
        scores[label] = np.load(p)
        print(f"  [cache] {label}: {scores[label].shape}")

    aggregated = _build_aggregations(scores)

    rb_60 = _random_baseline_sim(panel, TOP_K, 60, n_sims=20, seed=0)
    print(f"[phase19_exec] random baseline (60bps) p50 adj_S = {rb_60['p50_sharpe']:+.3f}")

    rows = []
    for label, sc in aggregated.items():
        for cost in COSTS_BPS_ROUND_TRIP:
            non_ov = _simulate_non_overlap(panel, sc, TOP_K, cost)
            d10 = _simulate_daily_10sleeve(panel, sc, TOP_K, cost)
            rows.append({
                "candidate": label,
                "cost_bps_round_trip": cost,
                "non_overlap": non_ov,
                "daily_10sleeve": d10,
            })
            print(
                f"  [{label:>16s}  {cost:>3d}bps]  non_overlap adj_S={non_ov.get('post_cost_sharpe_adjusted', float('nan')):+.3f}  "
                f"failed={non_ov.get('failed_entry_rate', 0):.3f}  defer={non_ov.get('deferred_exit_rate', 0):.3f}  "
                f"DD={non_ov.get('post_cost_max_drawdown', 0):.3f}  TO={non_ov.get('avg_turnover', 0):.3f}"
            )

    out_obj = {
        "val_start": VAL_START.isoformat(),
        "val_end": VAL_END.isoformat(),
        "top_k": TOP_K,
        "hold_days": HOLD_DAYS,
        "max_defer": MAX_DEFER,
        "limit_thresholds": [LIMIT_DN_THRESHOLD, LIMIT_UP_THRESHOLD],
        "costs_bps_round_trip": list(COSTS_BPS_ROUND_TRIP),
        "random_baseline_60bps": rb_60,
        "rows": rows,
    }
    out_json = OUT_DIR / "execution_sim.json"
    out_json.write_text(json.dumps(out_obj, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[phase19_exec] wrote {out_json}")

    md = ["# Phase 19 Stage C — execution constraint simulation", ""]
    md.append(f"OOS panel: {VAL_START} → {VAL_END} (199 dates, fp={HOLD_DAYS}, top-K={TOP_K}).")
    md.append("")
    md.append("Conventions:")
    md.append(f"- T+1 entry: pick at signal day t, buy at close[t+1] (proxy for next-day open).")
    md.append(f"- {HOLD_DAYS}-day hold: sell at close[t+{HOLD_DAYS+1}].")
    md.append(f"- Filters at entry: skip ST, suspended (vol=0), IPO < {NEW_STOCK_PROTECT} days, day-1 limit-up (pct >= {LIMIT_UP_THRESHOLD:+.3f}).")
    md.append(f"- Limit-down at exit: defer up to {MAX_DEFER} days; if still locked, sell at last attempted close.")
    md.append("- Cost = round-trip total bps applied once per trade as a log-return subtraction.")
    md.append("- Aggregated as equal-weight log-return per rebalance, annualised by sqrt(252/hold_days) = sqrt(25.2).")
    md.append(f"- Random-pick baseline (60bps, 20 sims): p50 adj_S = {rb_60['p50_sharpe']:+.3f}")
    md.append("")
    md.append("## Non-overlap (rebalance every 10 days)")
    md.append("")
    md.append("| candidate | cost bps | n_rebal | adj S | total log ret | max DD | failed entry | deferred exit | turnover |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        no = r["non_overlap"]
        if "error" in no:
            md.append(f"| {r['candidate']} | {r['cost_bps_round_trip']} | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        md.append(
            f"| {r['candidate']} | {r['cost_bps_round_trip']} | {no['n_rebalances']} | "
            f"{no['post_cost_sharpe_adjusted']:+.3f} | {no['post_cost_total_log_return']:+.3f} | "
            f"{no['post_cost_max_drawdown']:.3f} | {no['failed_entry_rate']:.3f} | "
            f"{no['deferred_exit_rate']:.3f} | {no['avg_turnover']:.3f} |"
        )
    md.append("")
    md.append("## Daily 10-sleeve (10 parallel portfolios offset by 1 day)")
    md.append("")
    md.append("| candidate | cost bps | n_sleeves | n_pooled | adj S | max DD |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for r in rows:
        d = r["daily_10sleeve"]
        if "error" in d:
            md.append(f"| {r['candidate']} | {r['cost_bps_round_trip']} | 0 | 0 | n/a | n/a |")
            continue
        md.append(
            f"| {r['candidate']} | {r['cost_bps_round_trip']} | {d['n_sleeves']} | "
            f"{d['n_period_returns_pooled']} | {d['post_cost_sharpe_adjusted']:+.3f} | "
            f"{d['post_cost_max_drawdown']:.3f} |"
        )

    out_md = OUT_DIR / "execution_sim.md"
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"[phase19_exec] wrote {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
