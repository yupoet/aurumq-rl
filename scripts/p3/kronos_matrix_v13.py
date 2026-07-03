"""Matrix v13 — paradigm 3 Kronos sequence-to-event anchor matrix.

paradigm 3 = sequence model (Kronos transformer) producing dense embeddings of
(stock, date) OHLCV history, then LGB classifier head learns anchor T pre-wave binary.

Architecture (post paris ACK_v30 + ledashi ACK-of-ACK):
  1. Reuse fine-tuned `aurumq_predictor_small` (90d lookback) — SKIP Phase 1 explicit pre-train
  2. Extract encoder hidden state at last position for 60d + 120d sequence windows
  3. Concat to 1536-dim embedding (768 + 768) per (stock, anchor_date)
  4. Concat 1-dim log(free_float_mv) → 1537-dim total feature
  5. Train 22 LGB heads: 21 main cells + 1 null-embedding control
  6. Eval H2_2025 + Q1_2026 same eval_cell as v12 paradigm 2

Cells (22 total):
  - 18 = 6 univ × 3 anchor × α 5-cond baseline
  - 3  = 1 univ (MAIN_BOARD only) × 3 anchor × β PELT-hybrid
  - 1  = null-embedding control (random 1536-dim, MAIN_BOARD T-3 α label)

CRITICAL: D-1 leakage guard — embedding(D) = encoder(OHLCV[D-seq_len : D-1])
  Enforced by kronos_matrix_v13_lib.build_lookback_window (the real extraction
  slice) + d1_leakage_selfcheck, unit-tested in tests/p3/test_kronos_matrix_v13.py.

Usage:
  # Phase 2: extract embeddings (~3-4h GPU)
  python scripts/p3/kronos_matrix_v13.py --phase 2

  # Phase 3: train LGB heads (~1.5h CPU)
  python scripts/p3/kronos_matrix_v13.py --phase 3

  # Phase 4: eval (H2_2025 + Q1_2026, ~1h)
  python scripts/p3/kronos_matrix_v13.py --phase 4

  # Dry run (validate paths + label loading, no GPU):
  python scripts/p3/kronos_matrix_v13.py --phase 2 --dry-run
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_HANDOFF_INBOX = os.environ.get("AURUMQ_HANDOFF_INBOX", "data/handoffs/inbox")

sys.path.insert(0, str(Path(__file__).parent))
from kronos_matrix_v10 import (
    UNIVERSES, TRAIN_START, TRAIN_END, WINDOWS, HORIZONS, COST_ROUND_TRIP,
    TOP_K_LIST, _dt, load_universes, filter_universe,
    compute_realized_and_exits, eval_cell,
)
from kronos_matrix_v12 import (
    LGB_BINARY_PARAMS, ANCHORS, FIX_INBOX,
    ANCHOR_DIR_BETA, ANCHOR_DIR_ALPHA, load_anchor_label,
)
try:  # package import (tests: p3.kronos_matrix_v13) — share one module object with p3.kronos_matrix_v13_lib
    from .kronos_matrix_v13_lib import (
        EMBARGO_DAYS, build_lookback_window, d1_leakage_selfcheck,
        date_embargo_split, is_cell_done, select_cell_embeddings, v13_artifact_path,
    )
except ImportError:  # direct script execution: python scripts/p3/kronos_matrix_v13.py
    from kronos_matrix_v13_lib import (
        EMBARGO_DAYS, build_lookback_window, d1_leakage_selfcheck,
        date_embargo_split, is_cell_done, select_cell_embeddings, v13_artifact_path,
    )

KRONOS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "kronos" / "repo"
sys.path.insert(0, str(KRONOS_DIR / "finetune"))
sys.path.insert(0, str(KRONOS_DIR))

# --- v13 specific config ---
SEQ_LEN_SHORT = 60
SEQ_LEN_LONG = 120
EMB_DIM_PER_SEQ = 512  # d_model for Kronos-small (verified 5/19 smoke test: 24.7M params)
EMB_DIM_TOTAL = EMB_DIM_PER_SEQ * 2  # 1024, concat 60d + 120d
FEATURE_DIM = EMB_DIM_TOTAL + 1  # 1025, +1 for log(free_float_mv)

OUT_DIR = Path("data/kronos/outputs/matrix_v13")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = OUT_DIR.parent  # data/kronos/outputs — phase-3 checkpoint + phase-4 results json
EMB_DIR = OUT_DIR / "embeddings"
EMB_DIR.mkdir(parents=True, exist_ok=True)

# v13 cell spec — 23 cells (22 main + 1 base-model control per paris ACK v13 smoke §3)
CELL_SPEC = []
# 18 = 6 univ × 3 anchor × α
for univ in UNIVERSES:
    for anchor in ANCHORS:
        CELL_SPEC.append({"univ": univ, "anchor": anchor, "spec": "alpha", "is_null": False, "is_base": False})
# 3 = MAIN_BOARD × 3 anchor × β
for anchor in ANCHORS:
    CELL_SPEC.append({"univ": "MAIN_BOARD", "anchor": anchor, "spec": "beta", "is_null": False, "is_base": False})
# 1 = null-embedding control (MAIN_BOARD T3 α with random emb, paris ACK §Q1)
CELL_SPEC.append({"univ": "MAIN_BOARD", "anchor": "T3", "spec": "alpha", "is_null": True, "is_base": False})
# 1 = raw HF base-model control (paris ACK v13 smoke §3, MAIN_BOARD T3 α with raw NeoQuasar/Kronos-small)
CELL_SPEC.append({"univ": "MAIN_BOARD", "anchor": "T3", "spec": "alpha", "is_null": False, "is_base": True})

# Paths
FREE_FLOAT_MV_PATH = Path(_HANDOFF_INBOX) / "2026-05-19-paris-ack-v30-kronos-paradigm3" / "fundamentals_free_float_mv_2022_2026.parquet"
OHLCV_PATH = Path("data/p3_4070_long/stock_close_volume_daily.parquet")


def cell_id(spec: dict) -> str:
    """Stable ID for a v13 cell."""
    suffix = ""
    if spec.get("is_null"):
        suffix = "_NULL"
    elif spec.get("is_base"):
        suffix = "_BASE"
    return f"{spec['spec']}_{spec['anchor']}_{spec['univ']}{suffix}"


# =============================================================================
# Phase 2: embedding extraction
# =============================================================================

def load_ohlcv_panel() -> pd.DataFrame:
    """Load full A股 OHLCV with adj-close + synth OHL (per kronos_inference.py)."""
    print("[load] reading stock_close_volume_daily ...")
    df = pd.read_parquet(OHLCV_PATH)
    latest_adj = df.sort_values(["ts_code", "trade_date"]).groupby("ts_code")["adj_factor"].last()
    df["adj_factor_latest"] = df["ts_code"].map(latest_adj)
    df["adj_close"] = df["close"] * df["adj_factor"] / df["adj_factor_latest"]
    df["adj_vwap"] = df["vwap"] * df["adj_factor"] / df["adj_factor_latest"]
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    df["prev_close"] = df.groupby("ts_code")["adj_close"].shift(1)
    df["open"] = df["prev_close"].fillna(df["adj_close"]).astype(np.float32)
    df["high"] = df[["adj_close", "prev_close", "adj_vwap"]].max(axis=1).fillna(df["adj_close"]).astype(np.float32)
    df["low"] = df[["adj_close", "prev_close", "adj_vwap"]].min(axis=1).fillna(df["adj_close"]).astype(np.float32)
    df["close"] = df["adj_close"].astype(np.float32)
    df["vol"] = df["volume"].astype(np.float32)
    df["amt"] = df["amount"].astype(np.float64)
    return df[["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amt"]]


CLIP = 5.0  # match Kronos.predict default


def normalize_window(x: np.ndarray) -> np.ndarray:
    """Per-window z-score normalize + clip to ±CLIP (matches KronosPredictor.predict)."""
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    x_norm = (x - x_mean) / (x_std + 1e-5)
    return np.clip(x_norm, -CLIP, CLIP).astype(np.float32)


def build_stamp(dates: pd.DatetimeIndex) -> np.ndarray:
    """Build temporal stamp [seq_len, 5] = [minute, hour, weekday, day, month] (daily → minute/hour=0)."""
    stamp = np.zeros((len(dates), 5), dtype=np.int64)
    stamp[:, 0] = 0  # minute
    stamp[:, 1] = 0  # hour
    stamp[:, 2] = dates.weekday
    stamp[:, 3] = dates.day
    stamp[:, 4] = dates.month
    return stamp


def extract_kronos_hidden_batch(model, tok, ohlcv_batch: np.ndarray, stamp_batch: np.ndarray, device: str):
    """Run Kronos transformer forward on a batch, return mean-pooled hidden state.

    Args:
        ohlcv_batch: [batch, seq_len, 6] float32, pre-normalized + clipped
        stamp_batch: [batch, seq_len, 5] int64 (minute, hour, weekday, day, month)

    Returns:
        np.ndarray [batch, 768] — mean-pooled post-norm transformer hidden state.

    Reuses tokenizer.encode(half=True) → (s1_ids, s2_ids), then custom transformer
    forward up to line 263 of kronos.py (post-norm), skipping the dual-head decode.
    """
    import torch

    x_tensor = torch.from_numpy(ohlcv_batch).to(device)
    stamp_tensor = torch.from_numpy(stamp_batch).to(device)

    with torch.no_grad():
        # half=True → returns (s1_ids, s2_ids) tuple, same as auto_regressive_inference
        s1_ids, s2_ids = tok.encode(x_tensor, half=True)

        # Custom forward — emulate kronos.py:Kronos.forward lines 254-263 (post-norm), skip head
        x_emb = model.embedding([s1_ids, s2_ids])
        x_emb = x_emb + model.time_emb(stamp_tensor)
        x_emb = model.token_drop(x_emb)
        for layer in model.transformer:
            x_emb = layer(x_emb, key_padding_mask=None)
        x_emb = model.norm(x_emb)  # [batch, seq_len, d_model]

        # Mean-pool over sequence (alt: x_emb[:, -1, :] = last token; mean is more robust)
        emb = x_emb.mean(dim=1)  # [batch, d_model]
        return emb.float().cpu().numpy()


def collect_eval_window_pairs(static_sets, pit_dfs, ohlcv) -> pd.DataFrame:
    """Materialize (ts_code, trade_date) pairs for full univ × eval window scoring.

    Returns dedup'd pairs across all 6 universes × H2_2025 + Q1_2026 eval dates.
    """
    # Get trading calendar from OHLCV in eval range
    eval_start = pd.Timestamp("2025-07-01")
    eval_end = pd.Timestamp("2026-03-31")
    cal = ohlcv[(ohlcv["trade_date"] >= eval_start.date()) &
                (ohlcv["trade_date"] <= eval_end.date())]
    eval_dates = sorted(set(cal["trade_date"].tolist()))
    print(f"  eval dates: {len(eval_dates)} trading days in [{eval_start.date()}, {eval_end.date()}]")

    pairs_set = set()
    # Static universes
    for univ_name, stock_set in static_sets.items():
        for stock in stock_set:
            for d in eval_dates:
                pairs_set.add((stock, d))
    # PIT universes (CSI500, CSI1000) — column is `stock_code` not `ts_code` in membership parquets
    for univ_name, pit_df in pit_dfs.items():
        code_col = "stock_code" if "stock_code" in pit_df.columns else "ts_code"
        sub = pit_df[(pit_df["trade_date"] >= eval_start.date()) &
                     (pit_df["trade_date"] <= eval_end.date())]
        for sc, td in zip(sub[code_col].values, sub["trade_date"].values):
            pairs_set.add((sc, td))

    pairs = pd.DataFrame(list(pairs_set), columns=["ts_code", "trade_date"])
    print(f"  collected {len(pairs):,} unique (univ × stock × eval_date) pairs")
    return pairs


def phase2_extract_embeddings(dry_run: bool = False, smoke: bool = False, eval_window: bool = False, base_model: bool = False):
    """Extract 60d + 120d Kronos embeddings for all (stock, date) pairs needed by 22 cells.

    Strategy:
      1. Identify unique (ts_code, anchor_date) pairs across all 22 cell labels
      2. For each pair, extract emb_60 = encoder(OHLCV[date-60 : date-1])
                              emb_120 = encoder(OHLCV[date-120 : date-1])
      3. Save chunked parquets to OUT_DIR/embeddings/
    """
    print(f"\n[phase 2] embedding extraction (dry_run={dry_run})")
    t_total = time.time()

    # Step 1: collect pairs — either label pairs (train) OR eval-window pairs (scoring)
    static_sets, pit_dfs = load_universes(UNIVERSES)
    if base_model:
        # Base-model variant: only MAIN_BOARD T3 alpha pairs (cheap control per paris ACK §3)
        print("\n[phase 2.1] collecting MAIN_BOARD T3 α pairs for base-model control")
        label_df = load_anchor_label("MAIN_BOARD", "T3", "alpha")
        pairs = label_df[["ts_code", "trade_date"]].drop_duplicates().reset_index(drop=True)
        if eval_window:
            # also include MAIN_BOARD × eval-window pairs for scoring
            ohlcv_for_cal = pd.read_parquet(OHLCV_PATH, columns=["trade_date"])
            eval_pairs = collect_eval_window_pairs({"MAIN_BOARD": static_sets["MAIN_BOARD"]}, {}, ohlcv_for_cal)
            pairs = pd.concat([pairs, eval_pairs], ignore_index=True).drop_duplicates().reset_index(drop=True)
    elif eval_window:
        print("\n[phase 2.1] collecting eval-window scoring pairs (univ × stock × eval_date)")
        # Need OHLCV for trading calendar
        ohlcv_for_cal = pd.read_parquet(OHLCV_PATH, columns=["trade_date"])
        pairs = collect_eval_window_pairs(static_sets, pit_dfs, ohlcv_for_cal)
    else:
        print("\n[phase 2.1] collecting unique (ts_code, anchor_date) pairs across cells")
        all_pairs = []
        for spec in CELL_SPEC:
            if spec.get("is_null") or spec.get("is_base"):
                continue  # null cell uses random; base cell handled separately
            label_df = load_anchor_label(spec["univ"], spec["anchor"], spec["spec"])
            if label_df is None:
                print(f"  [SKIP] {cell_id(spec)}: no labels"); continue
            all_pairs.append(label_df[["ts_code", "trade_date"]].drop_duplicates())
        pairs = pd.concat(all_pairs, ignore_index=True).drop_duplicates().reset_index(drop=True)
    print(f"  total unique (ts_code, date) pairs: {len(pairs):,}")

    if smoke:
        # Smoke test: 1/10 sampled, single year subset for fast end-to-end validation
        pairs = pairs.sample(frac=0.1, random_state=42).reset_index(drop=True)
        print(f"  [SMOKE] subsampled to {len(pairs):,} pairs (1/10) for pipeline validation")

    # Additionally collect eval-window pairs (H2_2025 + Q1_2026): all stocks × all eval dates per univ
    # We need to SCORE full cross-section at eval, so all (stock, date) pairs in eval windows.
    # Approach: load OHLCV panel head once → for each univ, intersect stocks × eval dates.
    eval_dates = []
    for win_name, (start, end) in WINDOWS.items():
        if win_name in ("H2_2025", "Q1_2026"):  # paris production primary
            # generate trading days in [start, end] — use full A股 trade calendar from OHLCV
            eval_dates.append((start, end))
    # (Phase 2 implementation will materialize all (univ × stock × eval_date) pairs from OHLCV index)
    print(f"  eval windows: {eval_dates} — Phase 2 will add full-univ scoring pairs")
    # estimated eval pairs: ~5500 stocks × ~400 eval days = ~2.2M per universe × 6 = ~13M raw,
    # dedup → ~5M unique (stock, date) across all 6 universes.

    if dry_run:
        print(f"[phase 2 dry-run] would extract {len(pairs):,} × 2 seq_len = {len(pairs)*2:,} embeddings")
        print(f"  estimated GPU time: {len(pairs) * 2 / 3200 / 60:.1f} min @ 3200 samples/sec")
        print(f"  estimated storage: {len(pairs) * EMB_DIM_TOTAL * 4 / 1e9:.1f} GB float32")
        return

    # Step 2: load Kronos checkpoints
    print("\n[phase 2.2] loading Kronos tokenizer + predictor ...")
    import torch
    from model.kronos import Kronos, KronosTokenizer
    from config import Config
    config = Config()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"  device: {device}")
    if base_model:
        # Raw HF base (paris ACK v13 smoke §4 — start re-pretrain from base, not fine-tuned)
        print("  using RAW HF NeoQuasar checkpoints (NOT 5/13-5/16 fine-tuned)")
        tok = KronosTokenizer.from_pretrained(config.pretrained_tokenizer_path)
        model = Kronos.from_pretrained(config.pretrained_predictor_path)
    else:
        tok = KronosTokenizer.from_pretrained(config.finetuned_tokenizer_path)
        model = Kronos.from_pretrained(config.finetuned_predictor_path)
    tok.to(device).eval()
    model.to(device).eval()
    print(f"  model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    print(f"  d_model: {model.d_model}")
    assert model.d_model == EMB_DIM_PER_SEQ, f"d_model mismatch: {model.d_model} vs {EMB_DIM_PER_SEQ}"

    # Step 3: load OHLCV panel
    print("\n[phase 2.3] loading OHLCV panel ...")
    ohlcv = load_ohlcv_panel()
    # Index by (ts_code) for fast slicing
    ohlcv_by_code = {sym: sub.reset_index(drop=True) for sym, sub in ohlcv.groupby("ts_code")}
    print(f"  {len(ohlcv_by_code):,} symbols, {len(ohlcv):,} total rows")

    # Step 4: D-1 leakage guard self-check (M20) — asserts on build_lookback_window,
    # the SAME function the extraction loop below uses, over a strictly increasing
    # synthetic series where any off-by-one is detectable. Also unit-tested in
    # tests/p3/test_kronos_matrix_v13.py.
    print("\n[phase 2.4] D-1 leakage guard self-check ...")
    d1_leakage_selfcheck(SEQ_LEN_SHORT)
    d1_leakage_selfcheck(SEQ_LEN_LONG)
    print(f"  [pass] build_lookback_window is strictly D-1 for seq_len {SEQ_LEN_SHORT} + {SEQ_LEN_LONG}")

    # Step 5: batched extraction loop
    print("\n[phase 2.5] batched embedding extraction ...")
    BATCH = 64
    extract_seq_lens = (SEQ_LEN_SHORT, SEQ_LEN_LONG)
    skipped_no_window = 0
    out_records: list[dict] = []  # accumulated rows: (ts_code, trade_date, emb_60, emb_120)
    skipped_pairs: list[tuple] = []

    # Pre-build per-stock date index for fast position lookup
    ohlcv_date_idx = {sym: {d: i for i, d in enumerate(sub["trade_date"].values)}
                       for sym, sub in ohlcv_by_code.items()}

    # Group pairs by ts_code for cache-friendly access
    pairs_sorted = pairs.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    # Build all (sym, anchor_idx) tasks across both seq_lens, validate windows exist
    tasks: list[tuple[str, int, pd.Timestamp]] = []  # (sym, anchor_date_idx_in_stock_panel, anchor_date)
    for _, row in pairs_sorted.iterrows():
        sym = row["ts_code"]; d = row["trade_date"]
        if sym not in ohlcv_date_idx: skipped_pairs.append((sym, d, "no_stock")); continue
        idx = ohlcv_date_idx[sym].get(d)
        if idx is None: skipped_pairs.append((sym, d, "no_date")); continue
        # Need SEQ_LEN_LONG rows BEFORE anchor_date: window = [idx-LONG, idx) → idx >= LONG
        # suffices (the old `idx - 1 < LONG` demanded one extra bar and silently dropped
        # each stock's first eligible anchor date — M20 boundary off-by-one).
        if idx < SEQ_LEN_LONG: skipped_pairs.append((sym, d, "insufficient_history")); continue
        tasks.append((sym, idx, d))
    print(f"  {len(tasks):,} valid (sym, anchor) tasks ({len(skipped_pairs):,} skipped)")
    if skipped_pairs[:3]:
        print(f"  skip reasons sample: {skipped_pairs[:3]}")

    # Process in batches — extract 60d + 120d per task
    OHLCV_COLS = ["open", "high", "low", "close", "vol", "amt"]
    t_extract = time.time()
    for batch_start in range(0, len(tasks), BATCH):
        batch_tasks = tasks[batch_start: batch_start + BATCH]
        # Build 60d batch + 120d batch
        win_60 = np.zeros((len(batch_tasks), SEQ_LEN_SHORT, 6), dtype=np.float32)
        win_120 = np.zeros((len(batch_tasks), SEQ_LEN_LONG, 6), dtype=np.float32)
        stamp_60 = np.zeros((len(batch_tasks), SEQ_LEN_SHORT, 5), dtype=np.int64)
        stamp_120 = np.zeros((len(batch_tasks), SEQ_LEN_LONG, 5), dtype=np.int64)
        for bi, (sym, idx, d) in enumerate(batch_tasks):
            sub = ohlcv_by_code[sym]
            # STRICT D-1 cutoff via the guarded pure function (M20): rows
            # [idx - seq_len, idx) — ends at idx-1, never includes the anchor day.
            sub_vals = sub[OHLCV_COLS].values
            sub_dates = sub["trade_date"].values
            x_60, dates_60 = build_lookback_window(sub_vals, sub_dates, idx, SEQ_LEN_SHORT)
            x_120, dates_120 = build_lookback_window(sub_vals, sub_dates, idx, SEQ_LEN_LONG)
            win_60[bi] = normalize_window(x_60.astype(np.float32))
            win_120[bi] = normalize_window(x_120.astype(np.float32))
            stamp_60[bi] = build_stamp(pd.to_datetime(dates_60))
            stamp_120[bi] = build_stamp(pd.to_datetime(dates_120))

        emb_60 = extract_kronos_hidden_batch(model, tok, win_60, stamp_60, device)
        emb_120 = extract_kronos_hidden_batch(model, tok, win_120, stamp_120, device)
        emb_concat = np.concatenate([emb_60, emb_120], axis=1)  # [batch, 1536]

        for bi, (sym, idx, d) in enumerate(batch_tasks):
            out_records.append({
                "ts_code": sym,
                "trade_date": d,
                "emb": emb_concat[bi].astype(np.float16),  # half-precision to halve storage
            })

        if (batch_start // BATCH) % 50 == 0:
            elapsed = time.time() - t_extract
            done = batch_start + len(batch_tasks)
            tput = done / elapsed if elapsed > 0 else 0
            eta_min = (len(tasks) - done) / tput / 60 if tput > 0 else 0
            print(f"    [{done:,}/{len(tasks):,}] {tput:.0f} samples/sec, ETA {eta_min:.1f} min", flush=True)

    print(f"\n[phase 2 done] {len(out_records):,} embeddings extracted in {time.time()-t_extract:.0f}s")
    # Save to single parquet (with float16 emb stored as bytes)
    out_df = pd.DataFrame(out_records)
    smoke_sfx = "_smoke" if smoke else ""
    eval_sfx = "_eval" if eval_window else ""
    base_sfx = "_base" if base_model else ""
    out_path = OUT_DIR / f"embeddings_{SEQ_LEN_SHORT}d_{SEQ_LEN_LONG}d{base_sfx}{eval_sfx}{smoke_sfx}.parquet"
    # Convert emb (np.ndarray per row) to fixed-width binary column for compact storage
    out_df["emb_bytes"] = out_df["emb"].apply(lambda x: x.tobytes())
    out_df = out_df.drop(columns=["emb"])
    out_df.to_parquet(out_path, compression="zstd", index=False)
    print(f"  saved {out_path} ({out_path.stat().st_size/1e9:.2f} GB)")
    return out_path


# =============================================================================
# Phase 3: train 22 LGB heads
# =============================================================================

def load_free_float_mv() -> pd.DataFrame:
    """Load paris fundamentals_free_float_mv parquet, return df with log_size feature."""
    print("[load] free_float_mv ...")
    df = pd.read_parquet(FREE_FLOAT_MV_PATH, columns=["stock_code", "trade_date", "free_float_mv"])
    df = df.rename(columns={"stock_code": "ts_code"})
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    # log1p, fill NULL with median per trade_date
    df["log_size"] = np.log1p(df["free_float_mv"].fillna(df["free_float_mv"].median())).astype(np.float32)
    return df[["ts_code", "trade_date", "log_size"]]


def decode_emb_parquet(emb_path: Path) -> pd.DataFrame:
    """Load embedding parquet and decode float16 bytes → 1024 cols emb_0..emb_1023."""
    print(f"  loading {emb_path}")
    df = pd.read_parquet(emb_path)
    n_rows = len(df)
    # Decode emb_bytes → matrix [N, 1024] float16 → float32 for LGB
    emb_matrix = np.zeros((n_rows, EMB_DIM_TOTAL), dtype=np.float32)
    for i, b in enumerate(df["emb_bytes"].values):
        emb_matrix[i] = np.frombuffer(b, dtype=np.float16).astype(np.float32)
    # Build emb DataFrame once (avoid fragmentation from per-column insert)
    emb_cols = [f"emb_{j}" for j in range(EMB_DIM_TOTAL)]
    emb_df = pd.DataFrame(emb_matrix, columns=emb_cols, index=df.index)
    out = pd.concat([df[["ts_code", "trade_date"]].reset_index(drop=True),
                     emb_df.reset_index(drop=True)], axis=1)
    out["trade_date"] = pd.to_datetime(out["trade_date"])
    return out


def phase3_train_lgb_heads(smoke: bool = False):
    """Train 22 LGB classifier heads on 1025-dim features (1024 Kronos + 1 log_size)."""
    import lightgbm as lgb

    print("\n[phase 3] train LGB heads on Kronos embeddings")
    t_total = time.time()

    static_sets, pit_dfs = load_universes(UNIVERSES)
    size_df = load_free_float_mv()

    # Load embedding parquet (smoke: 17K, full: 192K + 5M eval)
    suffix = "_smoke" if smoke else ""
    label_emb_path = OUT_DIR / f"embeddings_{SEQ_LEN_SHORT}d_{SEQ_LEN_LONG}d{suffix}.parquet"
    if not label_emb_path.exists():
        raise FileNotFoundError(f"Phase 2 output missing: {label_emb_path}. Run --phase 2 first.")
    label_embs = decode_emb_parquet(label_emb_path)
    print(f"  embeddings loaded: {len(label_embs):,} rows × {EMB_DIM_TOTAL} dims")

    # Optional: load eval embeddings if exist (Phase 4 will need these for scoring)
    eval_emb_path = OUT_DIR / f"embeddings_{SEQ_LEN_SHORT}d_{SEQ_LEN_LONG}d_eval{suffix}.parquet"
    eval_embs = decode_emb_parquet(eval_emb_path) if eval_emb_path.exists() else None
    if eval_embs is not None:
        print(f"  eval embeddings loaded: {len(eval_embs):,} rows")
    else:
        print(f"  [WARN] no eval embeddings ({eval_emb_path.name}) — Phase 4 scoring will skip")

    feat_cols = [f"emb_{j}" for j in range(EMB_DIM_TOTAL)] + ["log_size"]
    matrix_v13_partial = {}
    # M19: smoke runs get their own checkpoint (and pred parquets below) so a
    # smoke run can never make a full run silently skip every cell.
    CHECKPOINT = v13_artifact_path("checkpoint", smoke, out_dir=OUT_DIR, results_dir=RESULTS_DIR)
    if CHECKPOINT.exists():
        matrix_v13_partial = json.loads(CHECKPOINT.read_text())
        n_done = sum(1 for v in matrix_v13_partial.values() if is_cell_done(v))
        print(f"  [resume] {len(matrix_v13_partial)} cells in checkpoint "
              f"({n_done} done; skipped entries are retried)")

    # Optional base-model embeddings (paris ACK v13 smoke §3 3-way control).
    # C9: *_BASE cells must train AND score on base-model embeddings — scoring
    # them on the fine-tuned eval embeddings invalidated the 3-way control.
    # Only decoded when a *_BASE cell is actually pending.
    base_embs = base_eval_embs = None
    base_pending = any(s.get("is_base") and not is_cell_done(matrix_v13_partial.get(cell_id(s)))
                       for s in CELL_SPEC)
    if base_pending:
        base_emb_path = OUT_DIR / f"embeddings_{SEQ_LEN_SHORT}d_{SEQ_LEN_LONG}d_base{suffix}.parquet"
        base_embs = decode_emb_parquet(base_emb_path) if base_emb_path.exists() else None
        if base_embs is not None:
            print(f"  base-model embeddings loaded: {len(base_embs):,} rows (train source for *_BASE cells)")
        base_eval_emb_path = OUT_DIR / f"embeddings_{SEQ_LEN_SHORT}d_{SEQ_LEN_LONG}d_base_eval{suffix}.parquet"
        base_eval_embs = decode_emb_parquet(base_eval_emb_path) if base_eval_emb_path.exists() else None
        if base_eval_embs is not None:
            print(f"  base-model eval embeddings loaded: {len(base_eval_embs):,} rows (scoring source for *_BASE cells)")

    available_embs = {"label": label_embs}
    if eval_embs is not None:
        available_embs["eval"] = eval_embs
    if base_embs is not None:
        available_embs["base"] = base_embs
    if base_eval_embs is not None:
        available_embs["base_eval"] = base_eval_embs

    # C9 fail-fast: validate every pending cell's embedding sources BEFORE any LGB
    # fit — a pending *_BASE cell without base/base_eval parquets raises here with
    # the exact phase-2 command, instead of wasting hours of training first.
    for spec_dict in CELL_SPEC:
        if not is_cell_done(matrix_v13_partial.get(cell_id(spec_dict))):
            select_cell_embeddings(spec_dict, available_embs)

    for spec_dict in CELL_SPEC:
        cid = cell_id(spec_dict)
        prev_entry = matrix_v13_partial.get(cid)
        if is_cell_done(prev_entry):
            print(f"  [skip] {cid} done in checkpoint"); continue
        if prev_entry is not None:
            print(f"  [retry] {cid} previously skipped: {prev_entry}")

        # Load anchor labels
        if spec_dict.get("is_null") or spec_dict.get("is_base"):
            # Control cells reuse alpha_T3_MAIN_BOARD labels (same train set as real cell for fair compare)
            label_df = load_anchor_label("MAIN_BOARD", "T3", "alpha")
        else:
            label_df = load_anchor_label(spec_dict["univ"], spec_dict["anchor"], spec_dict["spec"])
        if label_df is None or len(label_df) == 0:
            print(f"  [SKIP] {cid}: no labels"); continue
        label_df = _dt(label_df)
        label_df["trade_date"] = pd.to_datetime(label_df["trade_date"])  # match label_embs dtype
        label_df = label_df.rename(columns={"y_binary": "y"} if "y_binary" in label_df.columns else {})

        # Choose embeddings source per cell type (C9: base cells train on base
        # embeddings AND score on base-model eval embeddings, never fine-tuned ones)
        train_key, eval_key = select_cell_embeddings(spec_dict, available_embs)
        embs_to_use = available_embs[train_key]
        eval_frame = available_embs.get(eval_key)

        # Join labels with embeddings + size
        joined = label_df.merge(embs_to_use, on=["ts_code", "trade_date"], how="inner")
        joined = joined.merge(size_df, on=["ts_code", "trade_date"], how="left")
        joined["log_size"] = joined["log_size"].fillna(joined["log_size"].median())
        if len(joined) < 500:
            print(f"  [SKIP] {cid}: joined train rows {len(joined)} < 500")
            matrix_v13_partial[cid] = {"skipped": True, "n_train": len(joined)}
            CHECKPOINT.write_text(json.dumps(matrix_v13_partial, indent=2, default=str))
            continue

        # Null cell: replace emb cols with random (paris ACK Q1 control)
        if spec_dict.get("is_null"):
            rng = np.random.default_rng(42)
            joined[feat_cols[:EMB_DIM_TOTAL]] = rng.standard_normal(
                (len(joined), EMB_DIM_TOTAL)
            ).astype(np.float32)

        # Train window split
        train = joined[(joined["trade_date"] >= pd.Timestamp(TRAIN_START)) &
                       (joined["trade_date"] <= pd.Timestamp(TRAIN_END))].reset_index(drop=True)
        if len(train) < 500:
            print(f"  [SKIP] {cid}: train window rows {len(train)} < 500")
            matrix_v13_partial[cid] = {"skipped": True, "n_train": len(train)}
            CHECKPOINT.write_text(json.dumps(matrix_v13_partial, indent=2, default=str))
            continue
        # M4: early-stopping split by DATE with a trading-day embargo. The old
        # row-position tail(10%) on the per-year-concat (unsorted) frame put same-date
        # cross-sectional twins in fit+val AND shared forward label event windows
        # (anchor horizon + ~20-25d wave extension) across the boundary.
        try:
            train_fit, val = date_embargo_split(train, val_frac=0.10, embargo_days=EMBARGO_DAYS)
        except ValueError as e:
            print(f"  [SKIP] {cid}: split failed — {e}")
            matrix_v13_partial[cid] = {"skipped": "split_failed", "n_train": len(train)}
            CHECKPOINT.write_text(json.dumps(matrix_v13_partial, indent=2, default=str))
            continue
        if len(train_fit) < 500 or len(val) == 0:
            print(f"  [SKIP] {cid}: post-embargo fit rows {len(train_fit)} < 500 or empty val ({len(val)})")
            matrix_v13_partial[cid] = {"skipped": "embargo_split_too_small",
                                       "n_train_fit": len(train_fit), "n_val": len(val)}
            CHECKPOINT.write_text(json.dumps(matrix_v13_partial, indent=2, default=str))
            continue
        pos_rate = (train_fit["y"] > 0).mean()

        t = time.time()
        model = lgb.LGBMClassifier(**LGB_BINARY_PARAMS)
        model.fit(
            train_fit[feat_cols], train_fit["y"],
            eval_set=[(val[feat_cols], val["y"])],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        best_iter = model.best_iteration_ or LGB_BINARY_PARAMS["n_estimators"]
        train_time = time.time() - t

        # Score eval windows (if eval embeddings exist). C9: eval_frame is the
        # cell-appropriate source (base_eval for *_BASE cells); M19: smoke preds
        # are suffixed so phase 4 never evaluates smoke-trained predictions.
        pred_path = v13_artifact_path("pred", smoke, out_dir=OUT_DIR, results_dir=RESULTS_DIR, cid=cid)
        if eval_frame is not None:
            eval_with_size = eval_frame.merge(size_df, on=["ts_code", "trade_date"], how="left")
            eval_with_size["log_size"] = eval_with_size["log_size"].fillna(eval_with_size["log_size"].median())
            # Coerce trade_date to date (object) so PIT merge in filter_universe works
            # (eval_with_size has datetime64[s], pit_dfs has date object)
            eval_with_size["trade_date"] = pd.to_datetime(eval_with_size["trade_date"]).dt.date
            # Apply universe filter
            univ_name = "MAIN_BOARD" if (spec_dict.get("is_null") or spec_dict.get("is_base")) else spec_dict["univ"]
            eval_filtered = filter_universe(eval_with_size[["ts_code", "trade_date"] + feat_cols],
                                             univ_name, static_sets, pit_dfs)
            if spec_dict.get("is_null"):
                # Null cell: same random replacement for inference (deterministic via same seed pattern)
                rng = np.random.default_rng(42 + 1)  # different seed for eval
                eval_filtered[feat_cols[:EMB_DIM_TOTAL]] = rng.standard_normal(
                    (len(eval_filtered), EMB_DIM_TOTAL)
                ).astype(np.float32)
            preds = model.predict_proba(eval_filtered[feat_cols])[:, 1].astype(np.float32)
            pred_df = eval_filtered[["ts_code", "trade_date"]].copy()
            pred_df["score"] = preds
            pred_df.to_parquet(pred_path, compression="zstd")

        matrix_v13_partial[cid] = {
            "n_train": int(len(train_fit)),
            "n_val": int(len(val)),
            "pos_rate": float(pos_rate),
            "best_iter": int(best_iter),
            "train_time_s": train_time,
            "has_pred": pred_path.exists(),
        }
        print(f"  {cid}: train {train_time:.0f}s n={len(train_fit):,} pos_rate={pos_rate:.3f} iter={best_iter}",
              flush=True)
        CHECKPOINT.write_text(json.dumps(matrix_v13_partial, indent=2, default=str))
        del model, joined; gc.collect()

    print(f"\n[phase 3 done] {len(matrix_v13_partial)} cells in {time.time()-t_total:.0f}s")
    return matrix_v13_partial


# =============================================================================
# Phase 4: eval H2_2025 + Q1_2026
# =============================================================================

def phase4_eval_cells(smoke: bool = False):
    """Eval all 22 cells on H2_2025 + Q1_2026 with K10/K50 × fwd5/fwd20.

    Reuses kronos_matrix_v10.eval_cell() for static IC + dyn-exit + Sharpe NET.
    M19: smoke runs read/write _smoke-suffixed artifacts only.
    """
    print(f"\n[phase 4] eval 22 cells{' (smoke)' if smoke else ''}")
    t_total = time.time()
    realized, exits = compute_realized_and_exits()

    matrix_v13_results = {}
    for spec in CELL_SPEC:
        cid = cell_id(spec)
        pred_path = v13_artifact_path("pred", smoke, out_dir=OUT_DIR, results_dir=RESULTS_DIR, cid=cid)
        if not pred_path.exists():
            print(f"  [SKIP] {cid}: no pred parquet"); continue
        pred_df = pd.read_parquet(pred_path)
        result = eval_cell(pred_df, realized, exits, adaptive_gating=None)
        matrix_v13_results[cid] = result
        # Print key metrics
        r_h2 = result["static"]["H2_2025"]["fwd20"]
        r_q1 = result["static"]["Q1_2026"]["fwd20"]
        ic_h2 = r_h2["ic"] * 100
        ic_q1 = r_q1["ic"] * 100
        sn_h2 = r_h2["sizing"].get("50", {}).get("sharpe_net", float("nan"))
        sn_q1 = r_q1["sizing"].get("50", {}).get("sharpe_net", float("nan"))
        print(f"  {cid}: H2 IC={ic_h2:+.2f}% S_NET={sn_h2:+.2f} | Q1 IC={ic_q1:+.2f}% S_NET={sn_q1:+.2f}")

    # Save matrix_v13_results{_smoke}.json (M19)
    out_path = v13_artifact_path("results", smoke, out_dir=OUT_DIR, results_dir=RESULTS_DIR)
    out_path.write_text(json.dumps({
        "config": {
            "tier": "v13 — paradigm 3 Kronos sequence-to-event anchor matrix",
            "cells": 22,
            "lgb_params": LGB_BINARY_PARAMS,
            "feature_dim": FEATURE_DIM,
            "seq_lens": [SEQ_LEN_SHORT, SEQ_LEN_LONG],
        },
        "results": matrix_v13_results,
        "total_time_s": time.time() - t_total,
    }, indent=2, default=str))
    print(f"\n[saved] {out_path} ({len(matrix_v13_results)} cells)")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, choices=[2, 3, 4], required=True)
    parser.add_argument("--dry-run", action="store_true", help="validate inputs without GPU")
    parser.add_argument("--smoke", action="store_true", help="1/10 sample smoke test (paris ACK §R4)")
    parser.add_argument("--eval-window", action="store_true", help="Phase 2: extract eval-window scoring pairs (full univ × eval dates) instead of label pairs")
    parser.add_argument("--base-model", action="store_true", help="Phase 2: use RAW HF NeoQuasar/Kronos-small (not 5/13 fine-tuned) for 3-way control per paris ACK §3")
    args = parser.parse_args()

    print(f"=== Matrix v13 — Phase {args.phase} ===")
    n_main = sum(1 for s in CELL_SPEC if not s.get("is_null") and not s.get("is_base"))
    n_null = sum(1 for s in CELL_SPEC if s.get("is_null"))
    n_base = sum(1 for s in CELL_SPEC if s.get("is_base"))
    print(f"Cells: {len(CELL_SPEC)} ({n_main} main + {n_null} null + {n_base} base-model control)")
    print(f"Feature dim: {FEATURE_DIM} (= {EMB_DIM_TOTAL} Kronos + 1 log_size)")
    print(f"Sequence lengths: {SEQ_LEN_SHORT}d + {SEQ_LEN_LONG}d (combined)")
    print(f"Train window: {TRAIN_START} ~ {TRAIN_END}")
    print(f"Eval windows: {list(WINDOWS.keys())}")
    print()

    if args.phase == 2:
        phase2_extract_embeddings(dry_run=args.dry_run, smoke=args.smoke,
                                    eval_window=args.eval_window, base_model=args.base_model)
    elif args.phase == 3:
        phase3_train_lgb_heads(smoke=args.smoke)
    elif args.phase == 4:
        phase4_eval_cells(smoke=args.smoke)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
