#!/usr/bin/env bash
# Path D — long panel (2018-2024) retrain pipeline.
#
# Triggered AFTER paris ships `oss://ledashi-oss/.../2026-05-10-long-panel-extension/`
# containing:
#   - feature_panel_v3_344.parquet      (full 2018-2026 — paris rebuilt)
#   - realized_returns_2018_2022.parquet
#   - market_returns_2018_2022.parquet
#   - universe_mask_2018_2022.parquet  (or 4 yearly shards year=2018..2022)
#   - tech_*_year={2017..2022}.parquet  (paris's tech_* backfill, optional)
#
# Pipeline:
#   1. Pull paris's data into data/p3_4070_long/
#   2. Concat realized + market with existing 2023-2025
#   3. Move universe_mask 2018-2022 shards into data/p3_4070_long/universe_mask/
#   4. Merge tech_* if shipped
#   5. Recompute target_y over the long window (path1_target.py)
#   6. Apply rank-z (path4_features.py) → feature_panel_clean_long.parquet
#   7. Run Path 4 grid (36 LightGBM × 60s = ~40 min) on long panel
#   8. Ensemble + RESULTS
#   9. Diff vs short panel
set -uo pipefail
cd /d/dev/aurumq-rl

PY=".venv/Scripts/python.exe"
LONG_BUNDLE="data/p3_4070_long"
SHORT_BUNDLE="data/p3_4070"
LOG_DIR="runs/sl_path_d_logs"
mkdir -p "$LOG_DIR"
LONG_OSS_PREFIX="aurumq-rl/handoffs/2026-05-10-long-panel-extension/"

step() {
    local name=$1; shift
    local log="$LOG_DIR/$name.log"
    echo "[$(date +%H:%M:%S)] BEGIN $name (log: $log)"
    if "$@" > "$log" 2>&1; then
        echo "[$(date +%H:%M:%S)] DONE  $name"
    else
        echo "[$(date +%H:%M:%S)] FAIL  $name (rc=$?, see $log)"
    fi
}

# 0. Sanity: paris's data must already be pulled
if [ ! -f "$LONG_BUNDLE/feature_panel_v3_344.parquet" ]; then
    echo "[$(date +%H:%M:%S)] ERROR: $LONG_BUNDLE/feature_panel_v3_344.parquet missing"
    echo "    Pull from OSS first: bash scripts/p3/_pull_long_panel.sh"
    exit 1
fi

# 1. Concat realized + market with existing 2023-2025
if [ ! -f "$LONG_BUNDLE/realized_returns.parquet" ]; then
    step "concat_realized_market" $PY -c "
import polars as pl
from pathlib import Path
LONG = Path(r'$LONG_BUNDLE')
SHORT = Path(r'$SHORT_BUNDLE')

# realized_returns
old = pl.read_parquet(LONG / 'realized_returns_2018_2022.parquet')
new = pl.read_parquet(SHORT / 'realized_returns.parquet')
combined = pl.concat([old, new]).sort(['trade_date', 'ts_code']).unique(subset=['trade_date', 'ts_code'])
combined.write_parquet(LONG / 'realized_returns.parquet', compression='zstd', compression_level=9)
print(f'realized: {len(combined):,} rows ({combined[\"trade_date\"].min()}..{combined[\"trade_date\"].max()})')

# market_returns
oldm = pl.read_parquet(LONG / 'market_returns_2018_2022.parquet')
newm = pl.read_parquet(SHORT / 'market_returns.parquet')
combinedm = pl.concat([oldm, newm]).sort('trade_date').unique(subset=['trade_date'])
combinedm.write_parquet(LONG / 'market_returns.parquet', compression='zstd', compression_level=9)
print(f'market: {len(combinedm):,} rows ({combinedm[\"trade_date\"].min()}..{combinedm[\"trade_date\"].max()})')
"
fi

# 2. Universe mask: copy from paris + symlink existing 2023-2025
if [ ! -d "$LONG_BUNDLE/universe_mask" ]; then
    step "merge_universe_mask" $PY -c "
import shutil
from pathlib import Path
LONG = Path(r'$LONG_BUNDLE')
SHORT = Path(r'$SHORT_BUNDLE')
out = LONG / 'universe_mask'
out.mkdir(parents=True, exist_ok=True)
# Copy 2023-2025 from short
for year in (2023, 2024, 2025, 2026):
    src = SHORT / 'universe_mask' / f'year={year}.parquet'
    if src.exists():
        shutil.copy(src, out / f'year={year}.parquet')
# Move 2018-2022 from paris's drop
for f in LONG.glob('universe_mask_year=*.parquet'):
    shutil.move(str(f), str(out / f.name.replace('universe_mask_', '')))
# Or single-file 2018-2022
sf = LONG / 'universe_mask_2018_2022.parquet'
if sf.exists():
    import polars as pl
    df = pl.read_parquet(sf)
    df = df.with_columns(pl.col('trade_date').dt.year().alias('_year'))
    for y in df['_year'].unique().to_list():
        shard = df.filter(pl.col('_year') == y).drop('_year')
        shard.write_parquet(out / f'year={y}.parquet', compression='zstd', compression_level=9)
    sf.unlink()
print('universe_mask:', sorted([p.name for p in out.glob('*.parquet')]))
"
fi

# 3. Recompute target_y over long window
if [ ! -f "$LONG_BUNDLE/target_y.parquet" ]; then
    step "recompute_target_y" $PY scripts/p3/path1_target.py --bundle "$LONG_BUNDLE"
fi

# 4. Rank-z on PRUNED long panel (226 cols, drops 119 SHAP-zero features)
#    to keep peak memory under ~10 GB for 5.6M-row joins on Windows.
if [ ! -f "$LONG_BUNDLE/feature_panel_clean.parquet" ]; then
    step "rank_z_long" $PY scripts/p3/path4_features.py \
        --bundle "$LONG_BUNDLE" \
        --feature-panel-in feature_panel_v3_344_pruned.parquet
fi

# 4b. Pre-join feature_panel + universe (in_uni) + target_y via DuckDB
#     (polars eager join OOMs on Windows for 5.6M-row × 226-col joins).
#     Output: feature_target_long.parquet (~4 GB, 5.3M rows × 229 cols).
if [ ! -f "$LONG_BUNDLE/feature_target_long.parquet" ]; then
    step "duckdb_prejoin" $PY -c "
import duckdb
con = duckdb.connect()
con.execute('PRAGMA memory_limit=\"30GB\"')
con.execute('PRAGMA threads=8')
con.execute('''
    COPY (
      SELECT f.*, t.y FROM \"$LONG_BUNDLE/feature_panel_clean.parquet\" f
      JOIN (SELECT trade_date, ts_code FROM read_parquet([\"$LONG_BUNDLE/universe_mask/year=*.parquet\"]) WHERE in_universe = true) u
        ON u.trade_date = f.trade_date AND u.ts_code = f.ts_code
      JOIN \"$LONG_BUNDLE/target_y.parquet\" t
        ON t.trade_date = f.trade_date AND t.ts_code = f.ts_code
    )
    TO \"$LONG_BUNDLE/feature_target_long.parquet\"
    (FORMAT PARQUET, COMPRESSION ZSTD, COMPRESSION_LEVEL 9)
''')
print('duckdb pre-join done')
"
fi

# 5. Run Path 4 grid on the pre-joined long bundle. n_jobs=8 to bound thread
#    memory; --feature-panel points at the joined file (path1_train auto-detects
#    'y' column and skips re-joining).
if [ "$(ls runs/sl_path_d/*/results.json 2>/dev/null | wc -l)" -lt 36 ]; then
    step "path_d_grid" $PY scripts/p3/path1_grid.py \
        --bundle "$LONG_BUNDLE" \
        --feature-panel feature_target_long.parquet \
        --out-root runs/sl_path_d \
        --train-start 2018-01-02 \
        --train-end 2024-12-04 \
        --num-iterations 2000 \
        --n-jobs 8
fi

# 7. Ensemble
if [ ! -f runs/sl_path_d/ensemble.json ]; then
    step "path_d_ensemble" $PY scripts/p3/path1_ensemble.py \
        --bundle "$LONG_BUNDLE" \
        --runs-root runs/sl_path_d \
        --out-root runs/sl_path_d \
        --top-k-configs 3
fi

# 8. Print headline diff vs sl_path4
echo "=========================================="
$PY -c "
import json
short = json.load(open('runs/sl_path4/ensemble.json'))
long = json.load(open('runs/sl_path_d/ensemble.json'))

def fmt(b): return f'{b[\"primary_mean_top50_proximity_excess\"]:+.6f}'

print('Path 4 (short panel 2023-2024):')
print(f'  H1: {fmt(short[\"ensemble_calibrated_H1\"])}')
print(f'  H2: {fmt(short[\"ensemble_calibrated_H2\"])}')
print('Path D (long panel 2018-2024):')
print(f'  H1: {fmt(long[\"ensemble_calibrated_H1\"])}')
print(f'  H2: {fmt(long[\"ensemble_calibrated_H2\"])}')
delta_h1 = long['ensemble_calibrated_H1']['primary_mean_top50_proximity_excess'] - short['ensemble_calibrated_H1']['primary_mean_top50_proximity_excess']
delta_h2 = long['ensemble_calibrated_H2']['primary_mean_top50_proximity_excess'] - short['ensemble_calibrated_H2']['primary_mean_top50_proximity_excess']
print(f'Delta long − short:')
print(f'  H1: {delta_h1:+.6f} ({delta_h1*1e4:+.2f} bps)')
print(f'  H2: {delta_h2:+.6f} ({delta_h2*1e4:+.2f} bps)')
"

echo "[$(date +%H:%M:%S)] DONE"
