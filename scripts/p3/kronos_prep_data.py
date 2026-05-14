"""Prep paris's stock_close_volume_daily.parquet into Kronos QlibDataset pickle format.

Kronos expects {symbol: DataFrame} dict with columns [datetime, open, high, low, close,
vol, amt]. paris only ships [close, volume, amount, vwap, adj_factor], so synthesize OHL:
  open[t] = close[t-1]   (yesterday's close)
  high[t] = max(close[t], close[t-1], vwap[t])
  low[t]  = min(close[t], close[t-1], vwap[t])

Output:
  data/kronos/processed/train_data.pkl
  data/kronos/processed/val_data.pkl

Date splits (per paris's REPLY):
  train: 2018-01-02 ~ 2024-12-31
  val:   2024-09-01 ~ 2025-06-30 (overlap with train for lookback)
"""
from __future__ import annotations

import datetime as dt
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd


OUT_DIR = Path("data/kronos/processed")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[load] reading paris stock_close_volume_daily.parquet ...")
    df = pd.read_parquet("data/p3_4070_long/stock_close_volume_daily.parquet")
    print(f"  rows: {len(df):,}  range: {df['trade_date'].min()} ~ {df['trade_date'].max()}")

    # Use adj-close for training (more meaningful for return prediction across splits)
    print("[adj] computing adj_close = close * adj_factor / latest_adj_factor ...")
    latest_adj = df.sort_values(["ts_code", "trade_date"]).groupby("ts_code")["adj_factor"].last()
    df["adj_factor_latest"] = df["ts_code"].map(latest_adj)
    df["adj_close"] = df["close"] * df["adj_factor"] / df["adj_factor_latest"]
    df["adj_vwap"] = df["vwap"] * df["adj_factor"] / df["adj_factor_latest"]

    # Sort by stock, date
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    # Synthesize OHL from close + vwap, per-stock
    print("[synth OHL] from close + vwap chain ...")
    df["prev_close"] = df.groupby("ts_code")["adj_close"].shift(1)
    df["open"]  = df["prev_close"].fillna(df["adj_close"])
    df["high"]  = df[["adj_close", "prev_close", "adj_vwap"]].max(axis=1).fillna(df["adj_close"])
    df["low"]   = df[["adj_close", "prev_close", "adj_vwap"]].min(axis=1).fillna(df["adj_close"])
    df["close"] = df["adj_close"]

    # Use 'vol' / 'amt' to match Kronos config naming
    df["vol"] = df["volume"]
    df["amt"] = df["amount"]

    # datetime col (Kronos expects 'datetime')
    df["datetime"] = pd.to_datetime(df["trade_date"])

    # Filter to needed cols + date range
    feature_cols = ["datetime", "open", "high", "low", "close", "vol", "amt"]
    df = df[["ts_code"] + feature_cols].dropna()

    # Group by ts_code
    print("[group] pivoting to {symbol: DataFrame} ...")
    symbols = df["ts_code"].unique()
    print(f"  {len(symbols)} symbols")

    train_data = {}
    val_data = {}
    train_start = pd.Timestamp("2018-01-02")
    train_end = pd.Timestamp("2024-12-31")
    val_start = pd.Timestamp("2024-09-01")
    val_end = pd.Timestamp("2025-06-30")

    # Use only stocks with at least 200 days in training window (avoid tiny series)
    n_kept = 0
    keep_cols = ["datetime", "open", "high", "low", "close", "vol", "amt"]
    # Force numpy-backed dtypes to avoid pyarrow string crash on Windows
    df_clean = df[["ts_code"] + keep_cols].astype({
        "open": "float32", "high": "float32", "low": "float32",
        "close": "float32", "vol": "float32", "amt": "float64",
    })
    df_clean["datetime"] = pd.to_datetime(df_clean["datetime"]).values.astype("datetime64[ns]")

    for sym in symbols:
        sub = df_clean[df_clean["ts_code"] == sym]
        train_sub = sub[(sub["datetime"] >= train_start) & (sub["datetime"] <= train_end)]
        val_sub = sub[(sub["datetime"] >= val_start) & (sub["datetime"] <= val_end)]
        if len(train_sub) >= 200 and len(val_sub) >= 50:
            # Convert via numpy dict→DataFrame to strip any pyarrow-backed column refs
            def to_clean(s):
                return pd.DataFrame({
                    "datetime": s["datetime"].values,
                    "open":  s["open"].values.astype(np.float32),
                    "high":  s["high"].values.astype(np.float32),
                    "low":   s["low"].values.astype(np.float32),
                    "close": s["close"].values.astype(np.float32),
                    "vol":   s["vol"].values.astype(np.float32),
                    "amt":   s["amt"].values.astype(np.float64),
                })
            train_data[sym] = to_clean(train_sub)
            val_data[sym] = to_clean(val_sub)
            n_kept += 1

    print(f"  kept {n_kept} symbols with enough data")
    print(f"  train_data sample sizes: avg {np.mean([len(v) for v in train_data.values()]):.0f}, "
          f"min {min(len(v) for v in train_data.values())}, "
          f"max {max(len(v) for v in train_data.values())}")
    print(f"  val_data sample sizes: avg {np.mean([len(v) for v in val_data.values()]):.0f}")

    # Save
    with open(OUT_DIR / "train_data.pkl", "wb") as f:
        pickle.dump(train_data, f)
    with open(OUT_DIR / "val_data.pkl", "wb") as f:
        pickle.dump(val_data, f)
    print(f"  saved to {OUT_DIR}")

    # Also save a "test" dataset for downstream inference (2025-07-01 ~ 2026-04-30)
    test_start = pd.Timestamp("2025-04-01")  # need 90d lookback before test_start_for_eval = 2025-07-01
    test_data = {}
    n_kept_test = 0
    for sym in symbols:
        sub = df_clean[df_clean["ts_code"] == sym]
        test_sub = sub[sub["datetime"] >= test_start]
        if len(test_sub) >= 100:
            test_data[sym] = pd.DataFrame({
                "datetime": test_sub["datetime"].values,
                "open":  test_sub["open"].values.astype(np.float32),
                "high":  test_sub["high"].values.astype(np.float32),
                "low":   test_sub["low"].values.astype(np.float32),
                "close": test_sub["close"].values.astype(np.float32),
                "vol":   test_sub["vol"].values.astype(np.float32),
                "amt":   test_sub["amt"].values.astype(np.float64),
            })
            n_kept_test += 1
    with open(OUT_DIR / "test_data.pkl", "wb") as f:
        pickle.dump(test_data, f)
    print(f"  test data: {n_kept_test} symbols")


if __name__ == "__main__":
    main()
