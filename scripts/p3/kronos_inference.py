"""Kronos batched inference — predict fwd-5 and fwd-20 close per (date, stock).

Uses KronosPredictor.predict_batch for ~64x speedup vs sequential. Inference
target dates cover VAL + H1 + H2 + 2026-Q1 + 2026-Q2-partial (paris eval windows).

Output per paris experiment plan schema:
  trade_date | ts_code | pred_close_fwd5 | pred_close_fwd20 |
  pred_return_fwd5 | pred_return_fwd20 | pred_volume_fwd5_avg | pred_sample_count
"""
from __future__ import annotations

import datetime as dt
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

KRONOS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "kronos" / "repo"
sys.path.insert(0, str(KRONOS_DIR / "finetune"))
sys.path.insert(0, str(KRONOS_DIR))

os.environ["HF_HUB_OFFLINE"] = "1"


def load_full_panel():
    """Load full stock OHLCV panel from paris stock_close_volume_daily, with synth OHL."""
    print("[load] reading paris stock_close_volume_daily ...")
    df = pd.read_parquet("data/p3_4070_long/stock_close_volume_daily.parquet")

    # adj close
    latest_adj = df.sort_values(["ts_code", "trade_date"]).groupby("ts_code")["adj_factor"].last()
    df["adj_factor_latest"] = df["ts_code"].map(latest_adj)
    df["adj_close"] = df["close"] * df["adj_factor"] / df["adj_factor_latest"]
    df["adj_vwap"] = df["vwap"] * df["adj_factor"] / df["adj_factor_latest"]

    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    df["prev_close"] = df.groupby("ts_code")["adj_close"].shift(1)
    df["open"]  = df["prev_close"].fillna(df["adj_close"]).astype(np.float32)
    df["high"]  = df[["adj_close", "prev_close", "adj_vwap"]].max(axis=1).fillna(df["adj_close"]).astype(np.float32)
    df["low"]   = df[["adj_close", "prev_close", "adj_vwap"]].min(axis=1).fillna(df["adj_close"]).astype(np.float32)
    df["close"] = df["adj_close"].astype(np.float32)
    df["vol"]   = df["volume"].astype(np.float32)
    df["amt"]   = df["amount"].astype(np.float64)
    df["datetime"] = pd.to_datetime(df["trade_date"])

    # Per-stock dict {sym: numpy arrays}
    stocks = {}
    for sym, sub in df.groupby("ts_code"):
        n = len(sub)
        if n < 110:
            continue
        stocks[sym] = {
            "datetime": sub["datetime"].values,
            "ohlcv": sub[["open", "high", "low", "close", "vol", "amt"]].values.astype(np.float32),
        }
    print(f"  {len(stocks)} symbols loaded")
    return stocks


def main():
    from model.kronos import Kronos, KronosTokenizer, KronosPredictor
    from config import Config
    config = Config()

    stocks = load_full_panel()

    print(f"[model] loading fine-tuned tokenizer + predictor ...")
    tok = KronosTokenizer.from_pretrained(config.finetuned_tokenizer_path)
    model = Kronos.from_pretrained(config.finetuned_predictor_path)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"  device: {device}")
    predictor = KronosPredictor(model, tok, device=device, max_context=512)
    predictor.tokenizer.eval(); predictor.model.eval()

    L = config.lookback_window   # 90
    P = config.predict_window    # 20
    SAMPLE = 4
    BATCH = 32                   # empirically: 4070 saturates at BATCH=32; 128 dropped to 5 t/s

    # Inference target dates: 2025-01-01 ~ 2026-05-12 (paris eval windows)
    # For each target_date, find stocks with ≥L days before AND ≥P days after.
    start = pd.Timestamp("2025-01-01")
    end = pd.Timestamp("2026-05-12")

    # Build per-stock: index of dates in target range
    per_stock_targets = {}
    for sym, info in stocks.items():
        dates = info["datetime"]
        idxs = []
        for i in range(L, len(dates) - P):
            if dates[i] >= start.to_datetime64() and dates[i] <= end.to_datetime64():
                idxs.append(i)
        if idxs:
            per_stock_targets[sym] = idxs

    # Flatten to (sym, idx) tuples
    all_tasks = [(s, i) for s, idxs in per_stock_targets.items() for i in idxs]
    print(f"[inference] {len(all_tasks):,} (sym, target_date) tasks (pre-resume)")

    # Resume from partial parquet if present
    out_rows = []
    partial_path = Path("data/kronos/outputs/kronos_predictions_partial.parquet")
    if partial_path.exists():
        partial = pd.read_parquet(partial_path)
        print(f"[resume] partial parquet: {len(partial):,} rows already computed")
        done = set(zip(partial["ts_code"].astype(str), partial["trade_date"].astype(str)))
        out_rows = partial.to_dict("records")

        def _target_date_str(sym, i):
            return pd.Timestamp(stocks[sym]["datetime"][i - 1]).date().isoformat()

        before = len(all_tasks)
        all_tasks = [(s, i) for s, i in all_tasks if (s, _target_date_str(s, i)) not in done]
        print(f"[resume] {before - len(all_tasks):,} skipped, {len(all_tasks):,} remaining")

    print(f"  batches of {BATCH} = {len(all_tasks)//BATCH + 1:,} forward passes")
    t0 = time.time()
    for batch_start in range(0, len(all_tasks), BATCH):
        batch = all_tasks[batch_start:batch_start + BATCH]
        df_list, x_ts_list, y_ts_list, meta = [], [], [], []
        for sym, t_idx in batch:
            info = stocks[sym]
            x_arr = info["ohlcv"][t_idx - L:t_idx]
            y_dates = info["datetime"][t_idx:t_idx + P]
            x_dates = info["datetime"][t_idx - L:t_idx]
            target_date = info["datetime"][t_idx - 1]
            last_close = float(x_arr[-1, 3])
            df_b = pd.DataFrame(x_arr, columns=["open","high","low","close","volume","amount"])
            df_list.append(df_b)
            x_ts_list.append(pd.Series(pd.to_datetime(x_dates)))
            y_ts_list.append(pd.Series(pd.to_datetime(y_dates)))
            meta.append((sym, pd.Timestamp(target_date).date(), last_close))

        try:
            preds = predictor.predict_batch(
                df_list=df_list, x_timestamp_list=x_ts_list, y_timestamp_list=y_ts_list,
                pred_len=P, T=1.0, top_p=0.9, sample_count=SAMPLE, verbose=False,
            )
        except Exception as e:
            print(f"  ERR batch {batch_start}: {e}")
            continue

        for (sym, td, last_close), pred in zip(meta, preds):
            c5 = float(pred["close"].iloc[4]) if len(pred) >= 5 else np.nan
            c20 = float(pred["close"].iloc[19]) if len(pred) >= 20 else np.nan
            v5 = float(pred["volume"].iloc[:5].mean()) if len(pred) >= 5 else np.nan
            out_rows.append({
                "trade_date": td, "ts_code": sym,
                "pred_close_fwd5": c5, "pred_close_fwd20": c20,
                "pred_return_fwd5":  (c5/last_close - 1.0) if last_close > 0 and not np.isnan(c5) else np.nan,
                "pred_return_fwd20": (c20/last_close - 1.0) if last_close > 0 and not np.isnan(c20) else np.nan,
                "pred_volume_fwd5_avg": v5,
                "pred_sample_count": SAMPLE,
            })

        n = batch_start + len(batch)
        if n % (BATCH * 20) == 0:
            el = time.time() - t0
            rate = n / el
            eta = (len(all_tasks) - n) / rate / 60
            print(f"  [{n:>7,}/{len(all_tasks):,}] {rate:.0f} tasks/s  ETA {eta:.0f} min", flush=True)
            # Save partial
            pd.DataFrame(out_rows).to_parquet(
                "data/kronos/outputs/kronos_predictions_partial.parquet",
                compression="zstd",
            )

    out_df = pd.DataFrame(out_rows)
    Path("data/kronos/outputs").mkdir(parents=True, exist_ok=True)
    out_path = Path("data/kronos/outputs/kronos_predictions_daily.parquet")
    out_df.to_parquet(out_path, compression="zstd", compression_level=9)
    print(f"\n[done] wrote {out_path}: {len(out_df):,} rows in {time.time()-t0:.0f}s")
    print(f"  pred_return_fwd5:  mean={out_df['pred_return_fwd5'].mean():+.4f}  std={out_df['pred_return_fwd5'].std():.4f}")
    print(f"  pred_return_fwd20: mean={out_df['pred_return_fwd20'].mean():+.4f}  std={out_df['pred_return_fwd20'].std():.4f}")


if __name__ == "__main__":
    main()
