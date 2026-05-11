#!/usr/bin/env bash
# Overnight: Path 1 long + Path 2 long + Path 5 long-base meta + RESULTS.
# ETA: ~5-7h on Windows + 3-parallel × 8-thread.
#
# Steps:
#  1. DuckDB pre-join feature_panel_v3_344 (raw) → feature_target_long_raw.parquet
#  2. Path 1 long grid (24 runs, parallel) → runs/sl_path1_long/
#  3. Path 2 long grid (36 runs, parallel) → runs/sl_path2_long/
#  4. Ensembles for both
#  5. Path 5 long-base meta refit (uses 1+4+2 long preds)
#  6. Combined RESULTS — short vs long for all paths
set -uo pipefail
cd /d/dev/aurumq-rl

# Lockfile — refuse to start a second master in parallel
LOCK="/tmp/overnight_long_paths.lock"
if [ -f "$LOCK" ]; then
    other_pid=$(cat "$LOCK")
    if kill -0 "$other_pid" 2>/dev/null; then
        echo "[$(date +%H:%M:%S)] ABORT: another master alive (pid=$other_pid). Exiting."
        exit 99
    fi
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

PY=".venv/Scripts/python.exe"
LONG="data/p3_4070_long"
LOG_DIR="runs/sl_overnight_logs"
mkdir -p "$LOG_DIR"

step() {
    local name=$1; shift
    local log="$LOG_DIR/$name.log"
    echo "[$(date +%H:%M:%S)] BEGIN $name (log: $log)"
    if "$@" > "$log" 2>&1; then
        echo "[$(date +%H:%M:%S)] DONE  $name"
        return 0
    else
        local rc=$?
        echo "[$(date +%H:%M:%S)] FAIL  $name (rc=$rc, see $log)"
        return $rc
    fi
}

# 1. Pre-join raw long panel for Path 1
if [ ! -f "$LONG/feature_target_long_raw.parquet" ]; then
    step "duckdb_prejoin_raw" $PY -c "
import duckdb
con = duckdb.connect()
con.execute('PRAGMA memory_limit=\"30GB\"')
con.execute('PRAGMA threads=8')
con.execute('''
    COPY (
      SELECT f.*, t.y FROM \"$LONG/feature_panel_v3_344.parquet\" f
      JOIN (SELECT trade_date, ts_code FROM read_parquet([\"$LONG/universe_mask/year=*.parquet\"]) WHERE in_universe = true) u
        ON u.trade_date = f.trade_date AND u.ts_code = f.ts_code
      JOIN \"$LONG/target_y.parquet\" t
        ON t.trade_date = f.trade_date AND t.ts_code = f.ts_code
    )
    TO \"$LONG/feature_target_long_raw.parquet\"
    (FORMAT PARQUET, COMPRESSION ZSTD, COMPRESSION_LEVEL 9)
''')
print('duckdb raw pre-join done')
" || exit 1
fi

# 2. Path 1 long grid — RAW 345-col panel needs ~25 GB/worker. Sequential
#    (1 worker × n_jobs=16 = 16 cores). ~24 × 6 min = ~2.4h.
step "path1_long_grid" $PY scripts/p3/path_d_grid_parallel.py \
    --bundle "$LONG" \
    --feature-panel feature_target_long_raw.parquet \
    --out-root runs/sl_path1_long \
    --train-start 2018-01-02 --train-end 2024-12-04 \
    --num-iterations 2000 --n-jobs 16 --workers 2

# 3. Path 2 long grid (3 parallel × n_jobs=8, ~3-4h)
step "path2_long_grid" $PY scripts/p3/path2_grid_parallel.py \
    --bundle "$LONG" \
    --feature-panel feature_target_long.parquet \
    --out-root runs/sl_path2_long \
    --train-start 2018-01-02 --train-end 2024-12-04 \
    --num-iterations 2000 --n-jobs 8 --workers 3

# 4. Ensembles
# Need baseline_predictions in long bundle; we copied it earlier for Path D.
if [ ! -f "$LONG/baseline_predictions.parquet" ]; then
    cp data/p3_4070/baseline_predictions.parquet "$LONG/baseline_predictions.parquet"
fi
step "path1_long_ensemble" $PY scripts/p3/path1_ensemble.py \
    --bundle "$LONG" --runs-root runs/sl_path1_long --out-root runs/sl_path1_long --top-k-configs 3
step "path2_long_ensemble" $PY scripts/p3/path1_ensemble.py \
    --bundle "$LONG" --runs-root runs/sl_path2_long --out-root runs/sl_path2_long --top-k-configs 3

# 5. Path 5 long-base meta refit (use long base preds + short regime features)
# path5 expects --paths to be names of dirs under runs/ — alias path_d as sl_path4_long
if [ ! -L runs/sl_path4_long ]; then
    cmd //c "mklink /J runs\\sl_path4_long runs\\sl_path_d" 2>/dev/null || \
        ln -s sl_path_d runs/sl_path4_long 2>/dev/null || \
        cp -r runs/sl_path_d runs/sl_path4_long
fi
step "path5_long_meta" $PY scripts/p3/path5_regime_stacking.py \
    --bundle data/p3_4070 \
    --paths sl_path1_long sl_path4_long sl_path2_long \
    --out runs/sl_regime_stack_long || true

# 6. Combined RESULTS
step "combined_results" $PY scripts/p3/overnight_results.py

echo ""
echo "[$(date +%H:%M:%S)] OVERNIGHT MAIN DONE — chaining supplement experiments"

# Chain supplement experiments (Path 1 ablation + Path 4 raw long + Hybrid)
bash scripts/p3/supplement_overnight.sh
echo "[$(date +%H:%M:%S)] OVERNIGHT DONE — see runs/sl_overnight_logs/ + runs/sl_path*_long/RESULTS.md + SUPPLEMENT_RESULTS.md"
