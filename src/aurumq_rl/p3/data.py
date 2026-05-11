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

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import duckdb
import numpy as np


logger = logging.getLogger(__name__)


# ── Bundle cache (avoid the 8-min row-by-row Python loop on every load) ──
# The slow path iterates 3.78M rows × 345 features = 1.3B Python-level
# scalar assignments to fill the (T, S, F) numpy array. Pure interpreter
# overhead, no I/O. Caching the resulting numpy arrays to a single .npz
# drops subsequent loads to ~5-10s.
#
# Cache key: sha256 of MANIFEST.sha256.txt (which itself fingerprints all
# source parquets). When upstream re-uploads the bundle, MANIFEST changes,
# cache key changes, slow path runs once again.
#
# File: <bundle_dir>/_bundle_cache.npz (~6 GB uncompressed; gitignored via
# data/p3_4070/ glob in .gitignore). Worth the disk because the loader is
# called by every train/eval/parity-check entry point.
def _bundle_cache_key(bundle_dir: Path) -> str:
    sha_path = bundle_dir / "MANIFEST.sha256.txt"
    if not sha_path.exists():
        return ""
    return hashlib.sha256(sha_path.read_bytes()).hexdigest()[:16]


def _bundle_cache_path(bundle_dir: Path) -> Path:
    return bundle_dir / "_bundle_cache.npz"


def _try_load_cache(bundle_dir: Path) -> "P3Bundle | None":
    cache_path = _bundle_cache_path(bundle_dir)
    if not cache_path.exists():
        return None
    expected_key = _bundle_cache_key(bundle_dir)
    try:
        t0 = time.time()
        z = np.load(cache_path, allow_pickle=True)
        if str(z["cache_key"]) != expected_key:
            logger.info("bundle cache key mismatch (manifest changed); ignoring stale cache")
            return None
        b = P3Bundle(
            trade_dates=z["trade_dates"].tolist(),
            ts_codes=z["ts_codes"].tolist(),
            feature_panel=z["feature_panel"],
            feature_cols=z["feature_cols"].tolist(),
            p_baseline=z["p_baseline"],
            rank_pct_baseline=z["rank_pct_baseline"],
            realized_pct_t_plus_1=z["realized_pct_t_plus_1"],
            market_pct_t_plus_1=z["market_pct_t_plus_1"],
            in_universe=z["in_universe"],
            label_t3=z["label_t3"],
            schema_hash=str(z["schema_hash"]),
        )
        logger.info("bundle loaded from cache in %.1fs (cache=%s)",
                    time.time() - t0, cache_path.name)
        return b
    except Exception as e:
        logger.warning("bundle cache load failed: %s; falling back to slow path", e)
        return None


def _save_cache(bundle_dir: Path, b: "P3Bundle") -> None:
    cache_path = _bundle_cache_path(bundle_dir)
    try:
        t0 = time.time()
        # Write to tmp then rename, so an interrupted save doesn't leave a
        # half-written .npz that future loads would crash on.
        # NOTE: np.savez auto-appends .npz to the path argument if it doesn't
        # already end in .npz. Use a name that ALREADY ends in .npz (e.g.
        # *_tmp.npz, not *.npz.tmp) so the file numpy actually writes matches
        # the path we'll rename — earlier code wrote *.npz.tmp.npz silently
        # while the rename target *.npz.tmp didn't exist.
        tmp = cache_path.with_name(cache_path.stem + "_tmp.npz")
        np.savez(
            tmp,
            cache_key=_bundle_cache_key(bundle_dir),
            trade_dates=np.array(b.trade_dates, dtype=object),
            ts_codes=np.array(b.ts_codes, dtype=object),
            feature_panel=b.feature_panel,
            feature_cols=np.array(b.feature_cols, dtype=object),
            p_baseline=b.p_baseline,
            rank_pct_baseline=b.rank_pct_baseline,
            realized_pct_t_plus_1=b.realized_pct_t_plus_1,
            market_pct_t_plus_1=b.market_pct_t_plus_1,
            in_universe=b.in_universe,
            label_t3=b.label_t3,
            schema_hash=b.schema_hash,
        )
        tmp.replace(cache_path)
        size_gb = cache_path.stat().st_size / (1 << 30)
        logger.info("bundle cache saved in %.1fs (%.2f GB at %s)",
                    time.time() - t0, size_gb, cache_path.name)
    except Exception as e:
        logger.warning("bundle cache save failed: %s (training proceeds without cache)", e)


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

    # Fast path: try cache first (sha-keyed against MANIFEST.sha256.txt).
    # Slow path below runs only on first call after a fresh OSS pull.
    cached = _try_load_cache(bundle_dir)
    if cached is not None:
        return cached

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
    # Date keys must match the universe's date_idx (built from to_pandas() which
    # yields datetime.date). arrow.to_numpy() on date32 yields numpy.datetime64,
    # which is NOT hash-equal to datetime.date — so date_idx.get(d) silently
    # returns None for every row, and 0 cells get loaded. Use to_pylist() to
    # get datetime.date objects throughout.
    dates_a = arr.column("trade_date").to_pylist()
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
    bd = base_arr.column("trade_date").to_pylist()
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
    rd = r_arr.column("trade_date").to_pylist()
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
    md = m_arr.column("trade_date").to_pylist()
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
    ld = l_arr.column("trade_date").to_pylist()
    lc = l_arr.column("ts_code").to_pylist()
    ly = l_arr.column("y").to_numpy().astype(np.int8)
    for k in range(l_arr.num_rows):
        i = date_idx.get(ld[k])
        if i is None: continue
        j = code_idx.get(lc[k])
        if j is None: continue
        label_t3[i, j] = ly[k]

    bundle = P3Bundle(
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
    _save_cache(bundle_dir, bundle)
    return bundle
