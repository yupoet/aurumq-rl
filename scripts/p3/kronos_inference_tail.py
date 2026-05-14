"""Kronos tail inference — extends predictions from 2026-04-09 to 2026-05-12.

Existing parquet (`data/kronos/outputs/kronos_predictions_daily.parquet`, 963K rows,
2024-11-08 ~ 2026-04-08) was bounded by `range(L, len(dates) - P)` which cut off the
last P=20 days of the panel. To cover up to last real date 2026-05-12, we:

1. Extend each stock's `datetime` array with N=30 synthetic future trading days
   (only the timestamps are needed by Kronos for time-feature embedding; close values
   are predicted, not used as input)
2. Loosen filter: accept t_idx where TARGET date (= dates[t_idx - 1]) is in window,
   not dates[t_idx]
3. Skip rows already in existing parquet (resume)

Output: appends new rows to existing `kronos_predictions_daily.parquet`.
Then a separate copy operation will land it in the outbox for OSS overwrite.
"""
from __future__ import annotations

import os
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

TAIL_START = pd.Timestamp("2026-04-09")
TAIL_END   = pd.Timestamp("2026-05-12")
FUTURE_PAD_DAYS = 30   # extend each stock's datetime by 30 synthetic trading days


def load_full_panel_with_future_pad():
    """Load panel and extend each stock's datetime/ohlcv with future synthetic days."""
    print("[load] reading paris stock_close_volume_daily ...")
    df = pd.read_parquet("data/p3_4070_long/stock_close_volume_daily.parquet")
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

    # Build global trading calendar from union of all stocks' dates
    all_dates = sorted(df["datetime"].unique())
    last_real_date = all_dates[-1]
    last_real_np = np.datetime64(last_real_date, "ns")
    # Pad N synthetic future trading days (skip weekends, ignore CN holidays for simplicity)
    future_dates = []
    d = pd.Timestamp(last_real_date) + pd.Timedelta(days=1)
    while len(future_dates) < FUTURE_PAD_DAYS:
        if d.weekday() < 5:
            future_dates.append(np.datetime64(d, "ns"))
        d += pd.Timedelta(days=1)
    future_dates_arr = np.array(future_dates, dtype="datetime64[ns]")
    print(f"  panel last real date: {last_real_date}, padded {len(future_dates)} synthetic future days")

    stocks = {}
    for sym, sub in df.groupby("ts_code"):
        n = len(sub)
        if n < 110:
            continue
        # Extend datetime + ohlcv with NaN pad for future
        dt_arr = np.concatenate([sub["datetime"].values, future_dates_arr])
        ohlcv_arr = sub[["open","high","low","close","vol","amt"]].values.astype(np.float32)
        pad_ohlcv = np.full((FUTURE_PAD_DAYS, 6), np.nan, dtype=np.float32)
        ohlcv_arr = np.concatenate([ohlcv_arr, pad_ohlcv], axis=0)
        stocks[sym] = {
            "datetime": dt_arr,
            "ohlcv": ohlcv_arr,
            "n_real": n,            # index < n_real → real data; ≥ n_real → synthetic future
        }
    print(f"  {len(stocks)} symbols indexed (each padded with {FUTURE_PAD_DAYS} synthetic future days)")
    return stocks, last_real_np


def main():
    from model.kronos import Kronos, KronosTokenizer, KronosPredictor
    from config import Config
    config = Config()

    stocks, last_real_np = load_full_panel_with_future_pad()

    print(f"[model] loading fine-tuned tokenizer + predictor ...")
    tok = KronosTokenizer.from_pretrained(config.finetuned_tokenizer_path)
    model = Kronos.from_pretrained(config.finetuned_predictor_path)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"  device: {device}")
    predictor = KronosPredictor(model, tok, device=device, max_context=512)
    predictor.tokenizer.eval(); predictor.model.eval()

    L = config.lookback_window
    P = config.predict_window
    SAMPLE = 4
    BATCH = 32

    start_np = np.datetime64(TAIL_START, "ns")
    end_np   = np.datetime64(TAIL_END, "ns")
    print(f"[tail] target_date window: {TAIL_START.date()} ~ {TAIL_END.date()}")

    # Build per-stock target idxs: TARGET = dates[t_idx - 1], so we want
    #   dates[t_idx - 1] in [start, end] (target date in window)
    #   AND dates[t_idx - 1] <= last_real_date (target is a real date, not synthetic)
    per_stock_targets = {}
    for sym, info in stocks.items():
        dates = info["datetime"]
        idxs = []
        # t_idx range: need t_idx - L >= 0 (lookback intact) and t_idx + P <= len(dates) (y_dates intact)
        max_t = len(dates) - P
        for i in range(L, max_t):
            tgt = dates[i - 1]
            if tgt > last_real_np: continue   # synthetic target — skip
            if tgt >= start_np and tgt <= end_np:
                # Also ensure lookback ohlcv is all real (i.e. i - 1 < info["n_real"])
                if i - 1 < info["n_real"]:
                    idxs.append(i)
        if idxs:
            per_stock_targets[sym] = idxs

    all_tasks = [(s, i) for s, idxs in per_stock_targets.items() for i in idxs]
    print(f"[tail] {len(all_tasks):,} (sym, target_date) tasks before resume")

    # Resume from existing daily parquet — skip already-done rows
    existing_path = Path("data/kronos/outputs/kronos_predictions_daily.parquet")
    existing = None
    if existing_path.exists():
        existing = pd.read_parquet(existing_path)
        existing["trade_date"] = pd.to_datetime(existing["trade_date"]).dt.date
        done = set(zip(existing["ts_code"].astype(str), existing["trade_date"].astype(str)))
        print(f"[resume] existing daily parquet: {len(existing):,} rows")

        def _target_date_str(sym, i):
            return pd.Timestamp(stocks[sym]["datetime"][i - 1]).date().isoformat()

        before = len(all_tasks)
        all_tasks = [(s, i) for s, i in all_tasks if (s, _target_date_str(s, i)) not in done]
        print(f"[resume] {before - len(all_tasks):,} skipped, {len(all_tasks):,} remaining")

    if not all_tasks:
        print("[tail] nothing to do — all target_dates already covered.")
        return

    print(f"  batches of {BATCH} = {len(all_tasks)//BATCH + 1:,} forward passes")
    new_rows = []
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
            new_rows.append({
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
            print(f"  [{n:>6,}/{len(all_tasks):,}] {rate:.0f} tasks/s  ETA {eta:.0f} min", flush=True)
            # Save partial to a separate file
            pd.DataFrame(new_rows).to_parquet(
                "data/kronos/outputs/kronos_predictions_tail_partial.parquet",
                compression="zstd",
            )

    new_df = pd.DataFrame(new_rows)
    print(f"\n[tail] computed {len(new_df):,} new rows in {time.time()-t0:.0f}s")
    print(f"  pred_return_fwd5  mean={new_df['pred_return_fwd5'].mean():+.4f} std={new_df['pred_return_fwd5'].std():.4f}")
    print(f"  pred_return_fwd20 mean={new_df['pred_return_fwd20'].mean():+.4f} std={new_df['pred_return_fwd20'].std():.4f}")

    # Append to existing daily parquet and overwrite
    if existing is not None:
        merged = pd.concat([existing, new_df], ignore_index=True)
    else:
        merged = new_df
    out_path = Path("data/kronos/outputs/kronos_predictions_daily.parquet")
    merged.to_parquet(out_path, compression="zstd", compression_level=9)
    print(f"[done] merged daily parquet: {len(merged):,} rows -> {out_path}")
    print(f"  trade_date range: {merged['trade_date'].min()} ~ {merged['trade_date'].max()}")


if __name__ == "__main__":
    main()
