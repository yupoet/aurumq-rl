#!/usr/bin/env bash
# Supplement overnight: 3 experiments after main overnight finishes.
#   A. Path 1 ablation (5y + 3y windows) — sample-size learning curve
#   B. Path 4 hyperparam + raw input on long panel — hypothesis test
#   C. Hybrid Path 1 long + Path 4 short ensemble
# + Strategy D eval on Path 1 long predictions
set -uo pipefail
cd /d/dev/aurumq-rl

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
    else
        local rc=$?
        echo "[$(date +%H:%M:%S)] FAIL  $name (rc=$rc, see $log)"
    fi
}

echo "[$(date +%H:%M:%S)] === SUPPLEMENT START ==="

# 0. Strategy D eval on Path 1 long predictions (~5 min, depends on path1_long_ensemble done)
if [ -f runs/sl_path1_long/predictions.parquet ]; then
    step "supp_strategyD_path1_long" $PY scripts/p3/supplement_strategy_d_eval.py \
        --bundle "$LONG" \
        --predictions runs/sl_path1_long/predictions.parquet \
        --label "Path 1 long"
fi
if [ -f runs/sl_path_d/predictions.parquet ]; then
    step "supp_strategyD_path_d" $PY scripts/p3/supplement_strategy_d_eval.py \
        --bundle "$LONG" \
        --predictions runs/sl_path_d/predictions.parquet \
        --label "Path 4 long (Path D)"
fi

# A. Ablation: train Path 1 best-config at 2020-2024 + 2022-2024 (~30 min)
step "supp_path1_ablation" $PY scripts/p3/supplement_ablation_grid.py \
    --bundle "$LONG" \
    --feature-panel feature_target_long_raw.parquet \
    --out-root runs/sl_path1_ablation \
    --n-jobs 16 --workers 2

# B. Hypothesis test: Path 4 hyperparam + RAW long panel input (~1.5h)
#    Reuse path_d_grid_parallel.py with raw panel + new out dir
step "supp_path4_raw_long" $PY scripts/p3/path_d_grid_parallel.py \
    --bundle "$LONG" \
    --feature-panel feature_target_long_raw.parquet \
    --out-root runs/sl_path4_raw_long \
    --train-start 2018-01-02 --train-end 2024-12-04 \
    --num-iterations 2000 --n-jobs 16 --workers 2

# B-ensemble
if [ ! -f runs/sl_path4_raw_long/ensemble.json ]; then
    step "supp_path4_raw_long_ensemble" $PY scripts/p3/path1_ensemble.py \
        --bundle "$LONG" --runs-root runs/sl_path4_raw_long --out-root runs/sl_path4_raw_long --top-k-configs 3
fi

# C. Hybrid Path 1 long + Path 4 short (~5 min, no model retrain)
step "supp_hybrid_p1long_p4short" $PY scripts/p3/supplement_hybrid_ensemble.py

# Final: write SUPPLEMENT_RESULTS.md
step "supp_results" $PY scripts/p3/supplement_results.py

echo "[$(date +%H:%M:%S)] === SUPPLEMENT DONE ==="
