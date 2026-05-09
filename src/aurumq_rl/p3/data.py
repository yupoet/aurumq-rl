"""P3 — Load OSS bundle into aligned (T, S) numpy arrays.

The bundle (downloaded from OSS to local dir) contains:
    feature_panel_v3_344.parquet
    baseline_predictions.parquet
    realized_returns.parquet
    market_returns.parquet
    labels/labels_A_t3_year=*.parquet
    universe_mask/year=*.parquet

This loads all into one MarketPanel-like container with extra arrays
specific to PPO residual training.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import duckdb
import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class P3Bundle:
    """Aligned (T, S) arrays for residual PPO training."""

    trade_dates: list[date]                # length T
    ts_codes: list[str]                    # length S, stable order

    # Features (P0 schema_hash 5e71e158e331)
    feature_panel: np.ndarray              # (T, S, F=345) float32
    feature_cols: list[str]

    # Baseline (P2 v2 ensemble)
    p_baseline: np.ndarray                 # (T, S) float32 ∈ [0, 1]
    rank_pct_baseline: np.ndarray          # (T, S) float32 ∈ [0, 1]

    # Reward inputs (option β: realized excess return)
    realized_pct_t_plus_1: np.ndarray      # (T, S) float32
    market_pct_t_plus_1: np.ndarray        # (T,) float32

    # Universe + labels
    in_universe: np.ndarray                # (T, S) bool
    label_t3: np.ndarray                   # (T, S) int8

    # Schema sanity
    schema_hash: str = "5e71e158e331"

    @property
    def n_dates(self) -> int:
        return len(self.trade_dates)

    @property
    def n_stocks(self) -> int:
        return len(self.ts_codes)

    @property
    def n_features(self) -> int:
        return self.feature_panel.shape[2]


def load_bundle(bundle_dir: Path | str, verify_manifest: bool = True) -> P3Bundle:
    """Load all parquets from bundle_dir into aligned numpy arrays.

    Stocks ordering is the union of in-universe ts_codes across all dates,
    sorted lexicographically. Cells where (date, stock) is out of universe
    are filled with NaN (features) / False (universe) / 0 (labels) /
    NaN (returns). The PPO env masks these out at step time.
    """
    bundle_dir = Path(bundle_dir)
    if verify_manifest:
        manifest_path = bundle_dir / "MANIFEST.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"MANIFEST.json missing in {bundle_dir}")
        with manifest_path.open() as f:
            manifest = json.load(f)
        logger.info("Loading P3 bundle: schema_hash=%s, total %.2f GB",
                    manifest.get("feature_schema_hash"), manifest.get("total_gb", 0.0))

    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='4GB'")
    con.execute("PRAGMA threads=3")

    # 1. Universe mask: discover (T, S) coordinates
    logger.info("  fetching universe mask...")
    uni_glob = str(bundle_dir / "universe_mask" / "year=*.parquet")
    uni_df = con.execute(f"""
        SELECT trade_date, ts_code, in_universe FROM '{uni_glob}'
        ORDER BY trade_date, ts_code
    """).fetch_arrow_table().to_pandas()
    trade_dates = sorted(uni_df["trade_date"].unique())
    ts_codes = sorted(uni_df["ts_code"].unique())
    T, S = len(trade_dates), len(ts_codes)
    date_idx = {d: i for i, d in enumerate(trade_dates)}
    code_idx = {c: i for i, c in enumerate(ts_codes)}
    logger.info("  T=%d S=%d (universe shape)", T, S)

    in_universe = np.zeros((T, S), dtype=bool)
    for d, c, u in zip(uni_df["trade_date"].values, uni_df["ts_code"].values, uni_df["in_universe"].values):
        in_universe[date_idx[d], code_idx[c]] = bool(u)

    # 2. Feature panel
    logger.info("  fetching feature panel (this is the heavy one)...")
    feat_path = bundle_dir / "feature_panel_v3_344.parquet"
    cols_meta = con.execute(f"DESCRIBE SELECT * FROM '{feat_path}' LIMIT 0").fetchall()
    feature_cols = [r[0] for r in cols_meta if r[0] not in ("ts_code", "trade_date")]
    F = len(feature_cols)
    logger.info("  F=%d features", F)

    feature_panel = np.full((T, S, F), np.nan, dtype=np.float32)
    cols_select = ", ".join(f'"{c}"' for c in feature_cols)
    arr = con.execute(f"""
        SELECT trade_date, ts_code, {cols_select} FROM '{feat_path}'
    """).fetch_arrow_table()
    n_loaded = 0
    dates_a = arr.column("trade_date").to_numpy(zero_copy_only=False)
    codes_a = arr.column("ts_code").to_pylist()
    feat_arrays = [arr.column(c).to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
                   for c in feature_cols]
    for k in range(arr.num_rows):
        d = dates_a[k]
        i = date_idx.get(d)
        if i is None:
            continue
        j = code_idx.get(codes_a[k])
        if j is None:
            continue
        for f_idx, fa in enumerate(feat_arrays):
            feature_panel[i, j, f_idx] = fa[k]
        n_loaded += 1
    logger.info("  feature panel: %d cells loaded", n_loaded)
    del arr, feat_arrays, dates_a, codes_a

    # 3. Baseline predictions
    logger.info("  fetching baseline predictions...")
    p_baseline = np.full((T, S), np.nan, dtype=np.float32)
    rank_pct = np.full((T, S), np.nan, dtype=np.float32)
    base_arr = con.execute(f"""
        SELECT trade_date, ts_code, p_t3_baseline, rank_pct_baseline
        FROM '{bundle_dir / 'baseline_predictions.parquet'}'
    """).fetch_arrow_table()
    bd = base_arr.column("trade_date").to_numpy(zero_copy_only=False)
    bc = base_arr.column("ts_code").to_pylist()
    bp = base_arr.column("p_t3_baseline").to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
    br = base_arr.column("rank_pct_baseline").to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
    for k in range(base_arr.num_rows):
        i = date_idx.get(bd[k])
        if i is None: continue
        j = code_idx.get(bc[k])
        if j is None: continue
        p_baseline[i, j] = bp[k]
        rank_pct[i, j] = br[k]

    # 4. Realized returns
    logger.info("  fetching realized returns...")
    realized = np.full((T, S), np.nan, dtype=np.float32)
    r_arr = con.execute(f"""
        SELECT trade_date, ts_code, pct_chg_t_plus_1
        FROM '{bundle_dir / 'realized_returns.parquet'}'
    """).fetch_arrow_table()
    rd = r_arr.column("trade_date").to_numpy(zero_copy_only=False)
    rc = r_arr.column("ts_code").to_pylist()
    rp = r_arr.column("pct_chg_t_plus_1").to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
    for k in range(r_arr.num_rows):
        i = date_idx.get(rd[k])
        if i is None: continue
        j = code_idx.get(rc[k])
        if j is None: continue
        realized[i, j] = rp[k]

    # 5. Market returns
    logger.info("  fetching market returns...")
    market = np.full(T, np.nan, dtype=np.float32)
    m_arr = con.execute(f"""
        SELECT trade_date, eq_weight_pct_chg_t_plus_1
        FROM '{bundle_dir / 'market_returns.parquet'}'
    """).fetch_arrow_table()
    md = m_arr.column("trade_date").to_numpy(zero_copy_only=False)
    mp = m_arr.column("eq_weight_pct_chg_t_plus_1").to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
    for k in range(m_arr.num_rows):
        i = date_idx.get(md[k])
        if i is None: continue
        market[i] = mp[k]

    # 6. Labels (A_t3 only, others available but not used for training)
    logger.info("  fetching labels A_t3...")
    label_t3 = np.zeros((T, S), dtype=np.int8)
    label_glob = str(bundle_dir / "labels" / "labels_A_t3_year=*.parquet")
    l_arr = con.execute(f"""
        SELECT trade_date, ts_code, y FROM '{label_glob}'
    """).fetch_arrow_table()
    ld = l_arr.column("trade_date").to_numpy(zero_copy_only=False)
    lc = l_arr.column("ts_code").to_pylist()
    ly = l_arr.column("y").to_numpy().astype(np.int8)
    for k in range(l_arr.num_rows):
        i = date_idx.get(ld[k])
        if i is None: continue
        j = code_idx.get(lc[k])
        if j is None: continue
        label_t3[i, j] = ly[k]

    return P3Bundle(
        trade_dates=trade_dates,
        ts_codes=ts_codes,
        feature_panel=feature_panel,
        feature_cols=feature_cols,
        p_baseline=p_baseline,
        rank_pct_baseline=rank_pct,
        realized_pct_t_plus_1=realized,
        market_pct_t_plus_1=market,
        in_universe=in_universe,
        label_t3=label_t3,
    )
