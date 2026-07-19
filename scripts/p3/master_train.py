"""MASTER-lite — market-guided cross-sectional transformer on the p3 factor panel.

2026-07 course correction (README §12.7): on daily factor panels, GBDT remains the
cost-effective SOTA; the ONE deep-model direction with credible published gains is
cross-sectional structure modeling — MASTER (AAAI 2024, arXiv:2312.15235) reports
~+25 % IC over XGBoost on CSI300/CSI800 via intra-stock temporal attention +
inter-stock cross attention + market-guided feature gating. This script is a
deliberately simplified single-GPU (RTX 4070 12 GB) reproduction of that idea on
our own panel/label, meant to produce ONE extra low-correlation score column for
rank-blending with the LightGBM base (path5_long) — never to replace it.

Simplifications vs the paper (documented, intentional):
  * Market status vector = cross-sectional mean of the z-scored panel at the
    anchor date (the paper uses index/market feature series). Our panel already
    carries mkt_* columns, so the mean vector subsumes them.
  * Gating is a single sigmoid layer over factors; the paper gates with a
    softmax-rescaled selection. Sigmoid keeps gradients dense at bring-up.
  * One temporal encoder layer + one cross-stock encoder layer (12 GB budget).

Pre-registered kill criteria (master_lib.kill_criteria_verdict): the best rank
blend with the LGBM base must beat the base on Spearman IC without losing on
top-50 proximity excess in >= 2/3 of eval windows, else the experiment is KILLED
(same policy that now governs the Kronos embedding track).

Usage (on the 4070 box)::

    python scripts/p3/master_train.py \
        --bundle data/p3_4070_long \
        --feature-panel feature_panel_v3_344_pruned.parquet \
        --out runs/master_lite/d64_L8_seed42 \
        --train-start 2023-01-03 --train-end 2025-06-30 \
        --seq-len 8 --d-model 64 --epochs 30 --seed 42

    # then evaluate + blend vs the LGBM base:
    python scripts/p3/master_ensemble_eval.py \
        --master-preds runs/master_lite/d64_L8_seed42/predictions.parquet \
        --base-preds runs/sl_path5_long/best/predictions.parquet \
        --bundle data/p3_4070_long --out runs/master_lite/d64_L8_seed42/ensemble

Smoke test (CPU, synthetic-scale)::

    python scripts/p3/master_train.py --bundle data/p3_4070 --out /tmp/master_smoke \
        --train-start 2025-01-02 --train-end 2025-06-30 \
        --device cpu --epochs 2 --max-stocks 200 --d-model 16
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent))

from p3.master_lib import (
    STD_FLOOR,
    build_sequence_windows,
    cs_zscore_panel,
    daily_rank_ic,
    train_val_split_with_embargo,
)
from p3.path1_eval import H1, H2, evaluate

logger = logging.getLogger(__name__)

FEATURE_PANEL_FNAME = "feature_panel_v3_344.parquet"
NON_FEATURE_COLS = {"ts_code", "trade_date", "y", "in_universe"}


# ------------------------------------------------------------------ #
# Data
# ------------------------------------------------------------------ #


def load_panel(bundle: Path, feature_panel: str) -> tuple[pl.DataFrame, list[str]]:
    """Load the feature panel with y attached, universe-filtered.

    Mirrors path1_train's loading contract: if the panel already carries a
    pre-joined ``y`` column (long-panel DuckDB pipeline) universe filtering and
    target join are both skipped; otherwise join ``universe_mask`` shards and
    ``target_y.parquet`` from the bundle.
    """
    df = pl.read_parquet(bundle / feature_panel)
    if "y" not in df.columns:
        uni_parts = [
            pl.read_parquet(p).select(["trade_date", "ts_code", "in_universe"])
            for p in sorted((bundle / "universe_mask").glob("year=*.parquet"))
        ]
        if uni_parts:
            uni = pl.concat(uni_parts)
            df = df.join(uni, on=["trade_date", "ts_code"], how="left").filter(
                pl.col("in_universe") == True  # noqa: E712
            )
        target = pl.read_parquet(bundle / "target_y.parquet").select(["trade_date", "ts_code", "y"])
        df = df.join(target, on=["trade_date", "ts_code"], how="left")
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    return df, feature_cols


def densify(
    df: pl.DataFrame, feature_cols: list[str]
) -> tuple[np.ndarray, np.ndarray, list[dt.date], list[str]]:
    """Pivot the long panel to dense [D, N, F] features + [D, N] labels.

    Memory note: 750 dates x 3000 stocks x 344 factors x float32 ~= 3.1 GB host
    RAM. Restrict the date range BEFORE calling this (the CLI does).
    """
    dates = sorted(df["trade_date"].unique().to_list())
    codes = sorted(df["ts_code"].unique().to_list())
    d_idx = {d: i for i, d in enumerate(dates)}
    c_idx = {c: i for i, c in enumerate(codes)}

    x = np.full((len(dates), len(codes), len(feature_cols)), np.nan, dtype=np.float32)
    y = np.full((len(dates), len(codes)), np.nan, dtype=np.float32)

    di = np.fromiter((d_idx[d] for d in df["trade_date"].to_list()), dtype=np.int64, count=len(df))
    ci = np.fromiter((c_idx[c] for c in df["ts_code"].to_list()), dtype=np.int64, count=len(df))
    x[di, ci] = df.select(feature_cols).to_numpy().astype(np.float32)
    y[di, ci] = df["y"].to_numpy().astype(np.float32)
    return x, y, dates, codes


def masked_market_vector(x_anchor: np.ndarray, present_row: np.ndarray) -> np.ndarray:
    """Cross-sectional mean feature vector over present stocks only.

    After ``cs_zscore_panel`` absent stocks sit at exactly 0; averaging them in
    shrinks the market-state vector toward 0 in proportion to how many names
    are absent that day. Falls back to the all-rows mean when nothing is
    present (degenerate warmup dates).
    """
    if present_row.any():
        return x_anchor[present_row].mean(axis=0)
    return x_anchor.mean(axis=0)


def zscore_labels_per_date(y: np.ndarray, present: np.ndarray) -> np.ndarray:
    """Z-score labels per date over present stocks; absent cells left untouched.

    Raw proximity labels keep their cross-sectional dispersion, letting
    high-dispersion dates dominate the MSE loss; per-date normalization makes
    every anchor date contribute comparably (scores are only consumed as a
    within-date ranking downstream). Std is floored at ``STD_FLOOR`` so a
    constant cross-section maps to 0 instead of exploding.
    """
    masked = np.where(present, y, np.nan)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        warnings.filterwarnings("ignore", message="Degrees of freedom <= 0")
        mean = np.nanmean(masked, axis=1, keepdims=True)
        std = np.nanstd(masked, axis=1, keepdims=True)
    mean = np.nan_to_num(mean, nan=0.0)
    std = np.maximum(np.nan_to_num(std, nan=0.0), STD_FLOOR)
    z = (y - mean) / std
    return np.where(present, z, y).astype(np.float32)


# ------------------------------------------------------------------ #
# Model (torch imported lazily so --help works on CPU-only boxes)
# ------------------------------------------------------------------ #


def build_model(n_factors: int, d_model: int, n_heads: int, dropout: float):
    import torch.nn as nn

    class MasterLite(nn.Module):
        """Market gate -> per-stock temporal encoder -> cross-stock attention -> score."""

        def __init__(self) -> None:
            super().__init__()
            self.gate = nn.Sequential(nn.Linear(n_factors, n_factors), nn.Sigmoid())
            self.proj = nn.Linear(n_factors, d_model)
            self.temporal = nn.TransformerEncoderLayer(
                d_model,
                n_heads,
                dim_feedforward=4 * d_model,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.cross = nn.TransformerEncoderLayer(
                d_model,
                n_heads,
                dim_feedforward=4 * d_model,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.head = nn.Linear(d_model, 1)

        def forward(self, x, market, pad_mask=None):
            # x: [N, L, F] one anchor date; market: [F] present-stock mean at
            # anchor; pad_mask: [N] bool, True = absent stock (excluded from
            # cross-stock attention keys, src_key_padding_mask convention).
            if pad_mask is not None and bool(pad_mask.all()):
                pad_mask = None  # degenerate all-absent date: avoid NaN attention
            g = self.gate(market)  # [F] market-guided feature gate
            h = self.proj(x * g)  # [N, L, d]
            h = self.temporal(h)[:, -1, :]  # [N, d] last-step summary
            h = self.cross(
                h.unsqueeze(0),
                src_key_padding_mask=None if pad_mask is None else pad_mask.unsqueeze(0),
            ).squeeze(0)  # attention across present stocks
            return self.head(h).squeeze(-1)  # [N]

    return MasterLite()


# ------------------------------------------------------------------ #
# Train / predict
# ------------------------------------------------------------------ #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--bundle", default="data/p3_4070", type=Path)
    p.add_argument("--feature-panel", default=FEATURE_PANEL_FNAME)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--train-start", required=True, help="ISO date")
    p.add_argument(
        "--train-end",
        required=True,
        help="ISO date; val + 30d embargo are carved from the tail of this window",
    )
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument(
        "--patience",
        type=int,
        default=5,
        help="early-stop after this many epochs without val-IC improvement",
    )
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument(
        "--accum-dates",
        type=int,
        default=4,
        help="gradient accumulation: one optimizer step per this many anchor dates",
    )
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--embargo-days", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--label-norm",
        choices=("zscore", "none"),
        default="zscore",
        help="per-date cross-sectional z-score of the training labels (loss only; "
        "val IC and eval blocks always use raw y)",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-amp", action="store_true", help="disable bf16 autocast")
    p.add_argument(
        "--max-stocks",
        type=int,
        default=None,
        help="debug: keep only the first N stocks (sorted ts_code)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
    )
    args = parse_args(argv)

    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA not available — pass --device cpu for a smoke run")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    train_lo = dt.date.fromisoformat(args.train_start)
    train_hi = dt.date.fromisoformat(args.train_end)

    # 1. Load + densify. Keep seq_len-1 warmup dates before train_lo, and
    # everything after train_hi so eval-window anchors have their sequences.
    t0 = time.time()
    df, feature_cols = load_panel(args.bundle, args.feature_panel)
    logger.info(
        "panel: %s rows, %s factor cols (%.0fs)",
        f"{len(df):,}",
        len(feature_cols),
        time.time() - t0,
    )
    if args.max_stocks:
        keep = sorted(df["ts_code"].unique().to_list())[: args.max_stocks]
        df = df.filter(pl.col("ts_code").is_in(keep))

    x, y, dates, codes = densify(df, feature_cols)
    del df
    # Presence must be captured BEFORE cs_zscore_panel fills NaN with 0.
    feat_present = ~np.isnan(x).all(axis=2)  # [D, N] stock has any feature this date
    x = cs_zscore_panel(x)
    present = ~np.isnan(y)  # [D, N] stock has a label at this date
    y_raw = np.nan_to_num(y, nan=0.0)  # reporting/eval always sees raw labels
    y = zscore_labels_per_date(y_raw, present) if args.label_norm == "zscore" else y_raw
    logger.info("dense panel: D=%d N=%d F=%d (%.0fs)", *x.shape, time.time() - t0)

    windows = build_sequence_windows(len(dates), args.seq_len)
    anchor_dates = [dates[w[-1]] for w in windows]

    train_pool = [i for i, d in enumerate(anchor_dates) if train_lo <= d <= train_hi]
    if len(train_pool) < 20:
        raise SystemExit(f"only {len(train_pool)} anchor dates in train window — check dates")
    train_ds, val_ds = train_val_split_with_embargo(
        [anchor_dates[i] for i in train_pool], args.val_frac, args.embargo_days
    )
    train_ids = [i for i in train_pool if anchor_dates[i] in set(train_ds)]
    val_ids = [i for i in train_pool if anchor_dates[i] in set(val_ds)]
    logger.info(
        "anchors: train=%d val=%d (embargo=%dd)", len(train_ids), len(val_ids), args.embargo_days
    )

    # 2. Model
    model = build_model(len(feature_cols), args.d_model, args.n_heads, args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("MasterLite: %s params, device=%s", f"{n_params:,}", device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    use_amp = device.type == "cuda" and not args.no_amp

    x_t = torch.from_numpy(x)  # stays on host; per-date slices moved to GPU
    y_t = torch.from_numpy(y)
    present_t = torch.from_numpy(present)

    def forward_date(i: int):
        w = windows[i]
        anchor = w[-1]
        xb = x_t[w].permute(1, 0, 2).to(device, non_blocking=True)  # [N, L, F]
        market = torch.from_numpy(masked_market_vector(x[anchor], feat_present[anchor])).to(device)
        pad = torch.from_numpy(~(feat_present[anchor] | present[anchor])).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            return model(xb, market, pad_mask=pad)

    def val_ic() -> float:
        model.eval()
        rows = []
        with torch.no_grad():
            for i in val_ids:
                scores = forward_date(i).float().cpu().numpy()
                mask = present[windows[i][-1]]
                rows.append(
                    pl.DataFrame(
                        {
                            "trade_date": [anchor_dates[i]] * int(mask.sum()),
                            "ts_code": [codes[j] for j in np.flatnonzero(mask)],
                            "score": scores[mask],
                            "actual_y": y_raw[windows[i][-1]][mask],
                        }
                    )
                )
        model.train()
        return daily_rank_ic(pl.concat(rows)) if rows else 0.0

    # 3. Train loop — one anchor date per forward, grad accumulation across dates.
    best_ic, best_epoch, bad_epochs = -np.inf, -1, 0
    history = []
    rng = np.random.default_rng(args.seed)
    for epoch in range(args.epochs):
        order = rng.permutation(train_ids)
        epoch_loss, n_steps = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for step, i in enumerate(order, start=1):
            scores = forward_date(i)
            anchor = windows[i][-1]
            mask = present_t[anchor].to(device)
            if int(mask.sum()) < 10:
                continue
            target = y_t[anchor].to(device)
            loss = ((scores - target)[mask] ** 2).mean() / args.accum_dates
            loss.backward()
            epoch_loss += float(loss) * args.accum_dates
            n_steps += 1
            if step % args.accum_dates == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
        ic = val_ic()
        history.append(
            {"epoch": epoch, "train_mse": epoch_loss / max(n_steps, 1), "val_daily_ic": ic}
        )
        logger.info("epoch %d: train_mse=%.6f val_ic=%+.4f", epoch, history[-1]["train_mse"], ic)
        if ic > best_ic:
            best_ic, best_epoch, bad_epochs = ic, epoch, 0
            torch.save(model.state_dict(), args.out / "model_best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                logger.info(
                    "early stop at epoch %d (best=%d val_ic=%+.4f)", epoch, best_epoch, best_ic
                )
                break

    # 4. Predict every anchor date AFTER the train window (true OOS) + val block.
    model.load_state_dict(
        torch.load(args.out / "model_best.pt", map_location=device, weights_only=True)
    )
    model.eval()
    predict_ids = [i for i, d in enumerate(anchor_dates) if d > train_hi] + val_ids
    rows = []
    with torch.no_grad():
        for i in sorted(set(predict_ids)):
            scores = forward_date(i).float().cpu().numpy()
            anchor = windows[i][-1]
            mask = feat_present[anchor] | present[anchor]
            rows.append(
                pl.DataFrame(
                    {
                        "trade_date": [anchor_dates[i]] * int(mask.sum()),
                        "ts_code": [codes[j] for j in np.flatnonzero(mask)],
                        "score": scores[mask].astype(np.float64),
                    }
                )
            )
    predictions = pl.concat(rows)
    predictions.write_parquet(args.out / "predictions.parquet")
    logger.info(
        "predictions: %s rows -> %s", f"{len(predictions):,}", args.out / "predictions.parquet"
    )

    # 5. Standard H1/H2 metric blocks when the bundle carries the eval files.
    metrics: dict = {
        "config": {k: str(v) for k, v in vars(args).items()},
        "n_params": n_params,
        "best_epoch": best_epoch,
        "best_val_daily_ic": best_ic,
        "history": history,
    }
    eval_files = ["target_y.parquet", "realized_returns.parquet", "market_returns.parquet"]
    if all((args.bundle / f).exists() for f in eval_files):
        target_y = pl.read_parquet(args.bundle / "target_y.parquet")
        realized = pl.read_parquet(args.bundle / "realized_returns.parquet").select(
            ["trade_date", "ts_code", "pct_chg_t_plus_1"]
        )
        market = pl.read_parquet(args.bundle / "market_returns.parquet").select(
            ["trade_date", "eq_weight_pct_chg_t_plus_1"]
        )
        for name, window in {"H1": H1, "H2": H2}.items():
            metrics[name] = evaluate(predictions, target_y, realized, market, window)
            logger.info("%s: %s", name, json.dumps(metrics[name]))
    else:
        logger.warning(
            "bundle lacks %s — skipping H1/H2 eval, run master_ensemble_eval later", eval_files
        )

    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    logger.info("done: best val_ic=%+.4f (epoch %d) -> %s", best_ic, best_epoch, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
