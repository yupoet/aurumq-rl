"""Kronos cross-eval v2 — adds H/I/I_high triggers using paris's raw MACD/KDJ data.

Builds on v1 (`kronos_crosseval.py`) outputs:
- Reuses `crosseval_predictions.parquet` (6 score sources × 8.7M rows)
- Re-runs dynamic-exit simulation with 4 new triggers added
- Reuses `crosseval_results.json` static section (identical, triggers don't affect static)
- Outputs `crosseval_v2_*.{json,md}` with 11 triggers and unchanged static

New triggers:
  H_macd_death  (macd_line < macd_signal AND prev_macd_line > prev_macd_signal)
  H_macd_hist_neg (above + macd_hist <= 0, stricter)
  I_kdj_death   (kdj_k < kdj_d AND prev_kdj_k > prev_kdj_d)
  I_kdj_high_death (above + kdj_d.shift(1) > 70, only high-position)

Source data: oss://ledashi-oss/aurumq-rl/handoffs/2026-05-14-paris-macd-kdj-raw/
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

PRED_PATH = "data/kronos/outputs/crosseval_predictions.parquet"
V1_RESULTS = "data/kronos/outputs/crosseval_results.json"
PANEL_CLOSE = "data/p3_4070_long/stock_close_volume_daily.parquet"
TECH_LINES = "D:/dev/aurumq-handoffs/inbox/2026-05-14-paris-macd-kdj-raw/tech_lines_daily.parquet"

WINDOWS = {
    "H1_2025": (pd.Timestamp("2025-01-01").date(), pd.Timestamp("2025-06-30").date()),
    "H2_2025": (pd.Timestamp("2025-07-01").date(), pd.Timestamp("2025-12-31").date()),
    "Q1_2026": (pd.Timestamp("2026-01-01").date(), pd.Timestamp("2026-03-31").date()),
}
K_MAX = 30

TRIGGERS = (
    "A_stop_5pct", "B_stop_3pct", "C_vol_drop", "D_vol_or_stop",
    "E_trail_5pct", "F_trend_break", "G_K_max_only",
    "H_macd_death", "H_macd_hist_neg",
    "I_kdj_death", "I_kdj_high_death",
)

SCORES = ("paris_t3_baseline", "ledashi_sl_final",
          "ledashi_lgb_baseline", "ledashi_lgb_kronos",
          "kronos_raw_fwd5", "kronos_raw_fwd20")


def _dt(df):
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    return df


def build_per_stock(panel_cv: pd.DataFrame, tech_df: pd.DataFrame) -> dict:
    """Merge panel + tech_lines per stock, return dict[ts_code] -> dict of np arrays."""
    print("[per-stock] merging panel + tech_lines ...", flush=True)
    t = time.time()
    # Left-join tech onto panel (missing dates -> NaN, will be False-trigger)
    merged = panel_cv.merge(tech_df, on=["ts_code", "trade_date"], how="left")
    print(f"  merged: {len(merged):,} rows  ({time.time()-t:.0f}s)")

    print("[per-stock] building arrays ...", flush=True)
    out: dict = {}
    for sym, g in merged.groupby("ts_code"):
        n = len(g)
        if n < K_MAX + 5:
            continue
        out[sym] = {
            "dates": g["trade_date"].to_numpy(),
            "adj_close": g["adj_close"].to_numpy(),
            "vol": g["vol"].to_numpy(),
            "pct_chg": g["pct_chg"].to_numpy(),
            "ma20_vol": g["ma20_vol"].to_numpy(),
            "ma5_close": g["ma5_close"].to_numpy(),
            "macd_line": g["macd_line"].to_numpy(),
            "macd_signal": g["macd_signal"].to_numpy(),
            "macd_hist": g["macd_hist"].to_numpy(),
            "kdj_k": g["kdj_k"].to_numpy(),
            "kdj_d": g["kdj_d"].to_numpy(),
        }
    print(f"  {len(out)} stocks indexed  ({time.time()-t:.0f}s)")
    return out


def simulate(per_stock: dict) -> pd.DataFrame:
    """Per-stock vectorized simulation of 11 triggers.

    Returns dataframe with (ts_code, trade_date) + per-trigger _realized_ret, _holding_days, _triggered cols."""
    t = time.time()
    print("[sim] simulating 11 triggers ...", flush=True)
    rows = []

    for sym, arrs in per_stock.items():
        dates = arrs["dates"]
        adj_close = arrs["adj_close"]
        n = len(dates)

        # Absolute trigger arrays (per-day boolean, indexed by absolute t)
        # C: vol_drop = vol > 2*MA20(vol) AND pct_chg < -3%
        C_abs = (arrs["vol"] > 2.0 * arrs["ma20_vol"]) & (arrs["pct_chg"] < -0.03)
        C_abs = np.nan_to_num(C_abs, nan=False).astype(bool)

        # F: trend_break = close < MA5
        F_abs = arrs["adj_close"] < arrs["ma5_close"]
        F_abs = np.nan_to_num(F_abs, nan=False).astype(bool)

        # H_macd_death: macd_line crosses below macd_signal
        macd = arrs["macd_line"]
        sig = arrs["macd_signal"]
        hist = arrs["macd_hist"]
        H_abs = np.zeros(n, dtype=bool)
        H_abs[1:] = (macd[:-1] > sig[:-1]) & (macd[1:] <= sig[1:])
        H_abs = H_abs & ~np.isnan(macd) & ~np.isnan(sig)

        H_hist_abs = H_abs & (hist <= 0)
        H_hist_abs = H_hist_abs & ~np.isnan(hist)

        # I_kdj_death: K crosses below D
        K = arrs["kdj_k"]
        D = arrs["kdj_d"]
        I_abs = np.zeros(n, dtype=bool)
        I_abs[1:] = (K[:-1] > D[:-1]) & (K[1:] <= D[1:])
        I_abs = I_abs & ~np.isnan(K) & ~np.isnan(D)

        # I_high: above + D.shift(1) > 70
        I_high_abs = np.zeros(n, dtype=bool)
        I_high_abs[1:] = I_abs[1:] & (D[:-1] > 70)
        I_high_abs = I_high_abs & ~np.isnan(D)

        # Valid entry indices
        max_entry = n - K_MAX - 1
        if max_entry < 0:
            continue
        entries = np.arange(0, max_entry + 1)
        forward_idx = entries[:, None] + np.arange(1, K_MAX + 1)[None, :]
        forward_close = adj_close[forward_idx]
        entry_close = adj_close[entries][:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            cum_ret = forward_close / entry_close - 1.0
        cum_ret = np.where(np.isfinite(cum_ret), cum_ret, 0.0)
        peak_ret = np.maximum.accumulate(cum_ret, axis=1)

        A_f = cum_ret < -0.05
        B_f = cum_ret < -0.03
        C_f = C_abs[forward_idx]
        D_f = A_f | C_f
        E_f = (peak_ret - cum_ret) > 0.05
        F_f = F_abs[forward_idx]
        G_f = np.zeros_like(A_f)
        H_f = H_abs[forward_idx]
        Hh_f = H_hist_abs[forward_idx]
        I_f = I_abs[forward_idx]
        Ih_f = I_high_abs[forward_idx]

        firmap = {
            "A_stop_5pct": A_f, "B_stop_3pct": B_f, "C_vol_drop": C_f,
            "D_vol_or_stop": D_f, "E_trail_5pct": E_f, "F_trend_break": F_f,
            "G_K_max_only": G_f, "H_macd_death": H_f, "H_macd_hist_neg": Hh_f,
            "I_kdj_death": I_f, "I_kdj_high_death": Ih_f,
        }

        rec = {"ts_code": np.full(len(entries), sym, dtype=object),
               "trade_date": dates[entries]}
        idx_range = np.arange(len(entries))
        for trig, fired in firmap.items():
            any_fired = fired.any(axis=1)
            first_idx = np.where(any_fired, fired.argmax(axis=1), K_MAX - 1)
            realized = cum_ret[idx_range, first_idx].astype(np.float32)
            holding = (first_idx + 1).astype(np.int8)
            rec[f"{trig}_realized_ret"] = realized
            rec[f"{trig}_holding_days"] = holding
            rec[f"{trig}_triggered"] = any_fired
        rows.append(pd.DataFrame(rec))

    out = pd.concat(rows, ignore_index=True)
    print(f"  {len(out):,} (entry_date, ts_code) rows  ({time.time()-t:.0f}s)")
    return out


def dyn_agg(df, score_col, trigger) -> dict:
    rc = f"{trigger}_realized_ret"; hc = f"{trigger}_holding_days"; tc = f"{trigger}_triggered"
    sub = df.dropna(subset=[score_col, rc])
    if len(sub) == 0:
        return {"mean_realized_ret": float("nan"), "mean_holding_days": float("nan"),
                "pct_triggered": float("nan"), "n_trades": 0,
                "top50_mean_ret": float("nan"), "top50_sharpe": float("nan")}
    mean_ret = float(sub[rc].mean())
    mean_hold = float(sub[hc].mean())
    pct_trig = float(sub[tc].mean())
    n_trades = int(len(sub))
    daily = []
    for _, g in sub.groupby("trade_date"):
        if len(g) < 50: continue
        daily.append(float(g.nlargest(50, score_col)[rc].mean()))
    if len(daily) >= 2:
        arr = np.asarray(daily); sd = arr.std(ddof=1)
        top50_mean = float(arr.mean())
        ann = np.sqrt(252.0 / max(mean_hold, 1.0))
        top50_sharpe = float(arr.mean() / sd * ann) if sd > 1e-9 else 0.0
    else:
        top50_mean = float("nan"); top50_sharpe = float("nan")
    return {
        "mean_realized_ret": mean_ret,
        "mean_holding_days": mean_hold,
        "pct_triggered": pct_trig,
        "n_trades": n_trades,
        "top50_mean_ret": top50_mean,
        "top50_sharpe": top50_sharpe,
    }


def main() -> int:
    t_total = time.time()
    print(f"[load] scores from {PRED_PATH}", flush=True)
    scores = pd.read_parquet(PRED_PATH)
    scores["trade_date"] = pd.to_datetime(scores["trade_date"]).dt.date
    print(f"  scores: {len(scores):,} rows × {scores.shape[1]} cols")

    print(f"[load] close+vol panel", flush=True)
    p = pd.read_parquet(PANEL_CLOSE, columns=["ts_code", "trade_date", "close", "adj_factor", "volume"])
    p = _dt(p).sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    latest = p.groupby("ts_code")["adj_factor"].transform("last")
    p["adj_close"] = (p["close"] * p["adj_factor"] / latest).astype(np.float32)
    p["vol"] = p["volume"].astype(np.float32)
    p["pct_chg"] = (p.groupby("ts_code")["adj_close"].pct_change()).astype(np.float32)
    p["ma20_vol"] = (p.groupby("ts_code")["vol"].transform(
        lambda x: x.rolling(20, min_periods=5).mean())).astype(np.float32)
    p["ma5_close"] = (p.groupby("ts_code")["adj_close"].transform(
        lambda x: x.rolling(5, min_periods=2).mean())).astype(np.float32)
    p = p.drop(columns=["close", "adj_factor", "volume"])

    print(f"[load] tech_lines from paris ({TECH_LINES})", flush=True)
    tech = pd.read_parquet(TECH_LINES)
    tech["trade_date"] = pd.to_datetime(tech["trade_date"]).dt.date
    print(f"  tech: {len(tech):,} rows  range {tech.trade_date.min()} ~ {tech.trade_date.max()}")

    per_stock = build_per_stock(p, tech)
    del p, tech

    exits = simulate(per_stock)
    del per_stock

    print("\n[merge] scores + exits ...")
    exits = scores.merge(exits, on=["ts_code", "trade_date"], how="inner")
    print(f"  rows after merge: {len(exits):,}")
    del scores

    print("\n[agg] per (score, window, trigger) ...")
    dyn_results = {}
    for s in SCORES:
        dyn_results[s] = {}
        for wname, (ws, we) in WINDOWS.items():
            sub = exits[(exits["trade_date"] >= ws) & (exits["trade_date"] <= we)]
            dyn_results[s][wname] = {}
            for trig in TRIGGERS:
                dyn_results[s][wname][trig] = dyn_agg(sub, s, trig)

    # Load v1 results for static section
    v1 = json.loads(Path(V1_RESULTS).read_text())
    out = {
        "config": {**v1.get("config", {}),
                   "dynamic_triggers": list(TRIGGERS),
                   "dynamic_K_max": K_MAX,
                   "tech_lines_source": "oss://ledashi-oss/aurumq-rl/handoffs/2026-05-14-paris-macd-kdj-raw/",
                   "v2_notes": "Added H/I triggers using paris-shipped raw MACD/KDJ"},
        "static": v1.get("static", {}),
        "dynamic": dyn_results,
        "total_time_s": time.time() - t_total,
    }
    Path("data/kronos/outputs/crosseval_v2_results.json").write_text(
        json.dumps(out, indent=2, default=str))

    # Build markdown table (dynamic only — static unchanged from v1)
    lines = ["# Kronos cross-eval v2 — adds H/I/I_high MACD+KDJ triggers", ""]
    lines.append(f"_Generated {time.strftime('%Y-%m-%d %H:%M Asia/Shanghai')}, "
                 f"runtime {time.time()-t_total:.0f}s_")
    lines.append("")
    lines.append("**Static fwdK results unchanged from v1** (see `crosseval_table.md`).")
    lines.append("")
    lines.append("## Dynamic-exit triggers (K_max=30, 11 triggers)")
    lines.append("")
    lines.append("**top-50 annualized Sharpe (Sharpe × sqrt(252/mean_holding_days))**")
    lines.append("")

    short_names = {
        "A_stop_5pct": "A", "B_stop_3pct": "B", "C_vol_drop": "C", "D_vol_or_stop": "D",
        "E_trail_5pct": "E", "F_trend_break": "F", "G_K_max_only": "G",
        "H_macd_death": "H_macd", "H_macd_hist_neg": "H_hist",
        "I_kdj_death": "I_kdj", "I_kdj_high_death": "I_high",
    }

    for wname in WINDOWS:
        lines.append(f"### {wname}")
        h = "| score | " + " | ".join(short_names[t] for t in TRIGGERS) + " |"
        sep = "|" + "---|" * (len(TRIGGERS) + 1)
        lines.append(h); lines.append(sep)
        for s in SCORES:
            row = [s]
            for trig in TRIGGERS:
                v = dyn_results[s][wname][trig].get("top50_sharpe", float("nan"))
                row.append(f"{v:+.2f}" if v == v else "n/a")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    lines.append("**top-50 mean realized return per trade × 100 (%, NOT annualized)**")
    lines.append("")
    for wname in WINDOWS:
        lines.append(f"### {wname}")
        h = "| score | " + " | ".join(short_names[t] for t in TRIGGERS) + " |"
        sep = "|" + "---|" * (len(TRIGGERS) + 1)
        lines.append(h); lines.append(sep)
        for s in SCORES:
            row = [s]
            for trig in TRIGGERS:
                v = dyn_results[s][wname][trig].get("top50_mean_ret", float("nan"))
                row.append(f"{v*100:+.2f}" if v == v else "n/a")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    lines.append("**mean holding days per trigger** (universe-wide, not just top-50)")
    lines.append("")
    for wname in WINDOWS:
        lines.append(f"### {wname}")
        h = "| trigger | mean_hold |"
        sep = "|---|---|"
        lines.append(h); lines.append(sep)
        for trig in TRIGGERS:
            v = dyn_results[SCORES[2]][wname][trig].get("mean_holding_days", float("nan"))
            lines.append(f"| {short_names[trig]} ({trig}) | {v:.1f} |" if v == v else f"| {trig} | n/a |")
        lines.append("")

    lines.append("**% trades that fired trigger (vs K_max forced exit)**")
    lines.append("")
    for wname in WINDOWS:
        lines.append(f"### {wname}")
        h = "| trigger | pct_triggered |"
        sep = "|---|---|"
        lines.append(h); lines.append(sep)
        for trig in TRIGGERS:
            v = dyn_results[SCORES[2]][wname][trig].get("pct_triggered", float("nan"))
            lines.append(f"| {short_names[trig]} ({trig}) | {v*100:.1f}% |" if v == v else f"| {trig} | n/a |")
        lines.append("")

    lines.append("---")
    lines.append("## New v2 trigger definitions")
    lines.append("")
    lines.append("- **H_macd_death**: `macd_line[t-1] > macd_signal[t-1] AND macd_line[t] <= macd_signal[t]` (上方下穿)")
    lines.append("- **H_macd_hist_neg**: H_macd_death AND `macd_hist[t] <= 0` (额外确认翻负)")
    lines.append("- **I_kdj_death**: `kdj_k[t-1] > kdj_d[t-1] AND kdj_k[t] <= kdj_d[t]` (任意位置死叉)")
    lines.append("- **I_kdj_high_death**: I_kdj_death AND `kdj_d[t-1] > 70` (仅高位死叉)")
    lines.append("")
    lines.append("Data source: `oss://ledashi-oss/aurumq-rl/handoffs/2026-05-14-paris-macd-kdj-raw/tech_lines_daily.parquet` (paris ship, 2024-12-02 ~ 2026-05-13, MAIN_BOARD 3003 stocks).")
    Path("data/kronos/outputs/crosseval_v2_table.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[saved] crosseval_v2_results.json + crosseval_v2_table.md")
    print(f"\n[done] total {time.time()-t_total:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
