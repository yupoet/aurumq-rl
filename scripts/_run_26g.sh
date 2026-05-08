#!/usr/bin/env bash
# Run 26G tier (encoder 256,128 / out_dim 64) with reduced batch_size to avoid VRAM blowup.
# 26F panel reused.
set -euo pipefail

PYTHON="${PYTHON:-D:/dev/aurumq-rl/.venv/Scripts/python.exe}"
PANEL="${PANEL:-data/panel_26f.parquet}"
INC="${INC:-configs/include_columns_phase26f_361.txt}"
SEEDS="${SEEDS:-42 43 44}"
BATCH="${BATCH:-384}"

mkdir -p runs/phase26ef_logs

run() {
    local seed=$1
    local out=runs/26G_seed${seed}
    local log=runs/phase26ef_logs/26G_seed${seed}.log
    if [ -f "$out/episode_eval.json" ]; then
        echo "[$(date +%H:%M:%S)] SKIP 26G seed=$seed (already done)"
        return 0
    fi
    echo "[$(date +%H:%M:%S)] BEGIN 26G seed=$seed enc=256,128 out=64 batch=$BATCH"
    "$PYTHON" scripts/train_v2.py \
        --total-timesteps 300000 \
        --data-path "$PANEL" \
        --start-date 2023-01-03 --end-date 2025-06-30 \
        --universe-filter main_board_non_st \
        --include-columns-file "$INC" \
        --n-envs 16 --n-steps 128 \
        --forward-period 5 --top-k 5 \
        --encoder-hidden 256,128 --encoder-out-dim 64 \
        --learning-rate 1e-4 \
        --rollout-buffer index \
        --reward-mode main_wave_target \
        --tf32 --matmul-precision high \
        --batch-size "$BATCH" \
        --seed "$seed" \
        --checkpoint-freq 50000 \
        --out-dir "$out" \
        > "$log" 2>&1
    echo "[$(date +%H:%M:%S)] EVAL 26G seed=$seed"
    "$PYTHON" scripts/_eval_main_wave_episode.py \
        --run-dir "$out" \
        --data-path "$PANEL" \
        --val-start 2025-07-01 --val-end 2026-04-24 \
        --top-k 3 5 --universe-filter main_board_non_st \
        >> "$log" 2>&1
    if [ "$seed" = "42" ]; then
        echo "[$(date +%H:%M:%S)] IG 26G seed=42"
        "$PYTHON" scripts/eval_factor_importance.py \
            --run-dir "$out" \
            --data-path "$PANEL" \
            --val-start 2025-07-01 --val-end 2026-04-24 \
            --top-k 5 --forward-period 5 --n-seeds 3 \
            >> "$log" 2>&1 || echo "(IG failed, non-blocking)"
    fi
    echo "[$(date +%H:%M:%S)] DONE 26G seed=$seed"
}

t0=$(date +%s)
for s in $SEEDS; do
    run "$s" || echo "FAILED 26G seed=$s — see $log"
done
echo "[$(date +%H:%M:%S)] 26G all seeds done in $(( $(date +%s) - t0 ))s"
