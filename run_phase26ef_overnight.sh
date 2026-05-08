#!/usr/bin/env bash
#
# Phase 26E/F/G overnight training orchestrator
# ----------------------------------------------
#
# Runs 4 tiers × 3 seeds = 12 train+eval cycles. Total ~3-4 GPU hours on RTX 4070.
#
# Pre-flight checklist
# --------------------
# 1. Pull AurumQ-RL latest (commit 036371a or later)
# 2. Pull panels v2 from OSS:
#      ossutil cp -r oss://ledashi-oss-sgp/aurumq-rl/handoffs/2026-05-08-phase26ef-tech-events/panels/ ./data/panels_v2/
# 3. Pull this script + companion configs:
#      ossutil cp -r oss://ledashi-oss-sgp/aurumq-rl/handoffs/2026-05-08-phase26ef-tech-events/ ./
# 4. Verify pre-flight integrity:
#      python tools/verify_panels.py ./data/panels_v2/
#    (expects sha256 match per MANIFEST.json + 0 inf + 100% nonnull on alpha_029/031, gtja_143)
# 5. Build combined panels for each tier (~10 min each on local SSD):
#      python scripts/build_combined_panel_phase26ef.py --tier 26C2 --out data/panel_26c2.parquet
#      python scripts/build_combined_panel_phase26ef.py --tier 26E  --out data/panel_26e.parquet
#      python scripts/build_combined_panel_phase26ef.py --tier 26F  --out data/panel_26f.parquet
#      # 26G uses the same panel as 26F (only encoder differs)
#
# Run
# ---
#   bash run_phase26ef_overnight.sh        # all 4 tiers × 3 seeds
#   bash run_phase26ef_overnight.sh 26F    # one tier × 3 seeds
#
# Pass / fail criteria (decided after all 12 runs complete)
# ---------------------------------------------------------
# - 26C2 baseline: must reproduce >= 2.5× T-1 lift (≥1 of 3 seeds). Sanity gate.
# - 26E (curated tech): pass = median lift >= 26C2 - 0.10× across seeds.
# - 26F (events decay): pass = median lift > 26C2 + 0.10× AND best seed > 26C2 best.
# - 26G (bigger encoder): pass = median lift > 26F + 0.10×; otherwise reject capacity.
#
# Output
# ------
# Each run writes runs/<tier>_seed<N>/ with:
#   - ppo_final.zip
#   - episode_eval.{json,md}
#   - factor_importance.json
#   - training_metrics.jsonl
# Final scoreboard: runs/phase26ef_scoreboard.md (auto-aggregated)
#
# If anything degrades vs 26C2 — production stays on 26C2 unchanged.

set -euo pipefail

# ── Config ──────────────────────────────────────────────────────────────
PYTHON="${PYTHON:-.venv/Scripts/python.exe}"   # Windows AurumQ-RL default
SEEDS="${SEEDS:-42 43 44}"                      # 3 seeds for noise control
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-300000}"
TRAIN_START="${TRAIN_START:-2023-01-03}"
TRAIN_END="${TRAIN_END:-2025-06-30}"
EVAL_START="${EVAL_START:-2025-07-01}"
EVAL_END="${EVAL_END:-2026-04-24}"
UNIVERSE="${UNIVERSE:-main_board_non_st}"

# Tier definition ─────────────────────────────────────────────────────────
# Each tier: tier_id, panel_path, include_columns_file, encoder_hidden, encoder_out_dim
declare -A PANEL=(
    [26C2]=data/panel_26c2.parquet
    [26E]=data/panel_26e.parquet
    [26F]=data/panel_26f.parquet
    [26G]=data/panel_26f.parquet  # same panel as 26F
)
declare -A INCLUDE_FILE=(
    [26C2]=configs/include_columns_phase26c2_353.txt
    [26E]=configs/include_columns_phase26e_355.txt
    [26F]=configs/include_columns_phase26f_361.txt
    [26G]=configs/include_columns_phase26f_361.txt
)
declare -A ENCODER_HIDDEN=(
    [26C2]=128,64
    [26E]=128,64
    [26F]=128,64
    [26G]=256,128
)
declare -A ENCODER_OUT_DIM=(
    [26C2]=32
    [26E]=32
    [26F]=32
    [26G]=64
)

TIERS_TO_RUN=("$@")
if [ ${#TIERS_TO_RUN[@]} -eq 0 ]; then
    TIERS_TO_RUN=(26C2 26E 26F 26G)
fi

mkdir -p runs/phase26ef_logs

run_one() {
    local tier=$1
    local seed=$2
    local out_dir="runs/${tier}_seed${seed}"
    local log="runs/phase26ef_logs/${tier}_seed${seed}.log"

    if [ -d "$out_dir/eval" ] && [ -f "$out_dir/episode_eval.json" ]; then
        echo "[$(date +%H:%M:%S)] SKIP $tier seed=$seed (already done)"
        return 0
    fi

    echo "[$(date +%H:%M:%S)] BEGIN $tier seed=$seed enc=${ENCODER_HIDDEN[$tier]} out=${ENCODER_OUT_DIM[$tier]}"

    "$PYTHON" scripts/train_v2.py \
        --total-timesteps "$TOTAL_TIMESTEPS" \
        --data-path "${PANEL[$tier]}" \
        --start-date "$TRAIN_START" \
        --end-date "$TRAIN_END" \
        --universe-filter "$UNIVERSE" \
        --include-columns-file "${INCLUDE_FILE[$tier]}" \
        --n-envs 16 --n-steps 128 \
        --forward-period 5 --top-k 5 \
        --encoder-hidden "${ENCODER_HIDDEN[$tier]}" \
        --encoder-out-dim "${ENCODER_OUT_DIM[$tier]}" \
        --learning-rate 1e-4 \
        --rollout-buffer index \
        --reward-mode main_wave_target \
        --tf32 --matmul-precision high \
        --seed "$seed" \
        --checkpoint-freq 50000 \
        --out-dir "$out_dir" \
        > "$log" 2>&1

    echo "[$(date +%H:%M:%S)] EVAL $tier seed=$seed"
    "$PYTHON" scripts/_eval_main_wave_episode.py \
        --run-dir "$out_dir" \
        --data-path "${PANEL[$tier]}" \
        --val-start "$EVAL_START" --val-end "$EVAL_END" \
        --top-k 3 5 --universe-filter "$UNIVERSE" \
        >> "$log" 2>&1

    # IG / permutation only on best-of-tier (run on seed=42 always)
    if [ "$seed" = "42" ]; then
        echo "[$(date +%H:%M:%S)] IG $tier seed=42"
        "$PYTHON" scripts/eval_factor_importance.py \
            --run-dir "$out_dir" \
            --data-path "${PANEL[$tier]}" \
            --val-start "$EVAL_START" --val-end "$EVAL_END" \
            --top-k 5 --forward-period 5 --n-seeds 3 \
            >> "$log" 2>&1 || echo "  (IG failed, non-blocking)"
    fi

    echo "[$(date +%H:%M:%S)] DONE $tier seed=$seed"
}

t0=$(date +%s)
for tier in "${TIERS_TO_RUN[@]}"; do
    for seed in $SEEDS; do
        run_one "$tier" "$seed" || {
            echo "FAILED $tier seed=$seed (continuing — see log)"
            continue
        }
    done
done

# Aggregate scoreboard
"$PYTHON" scripts/phase26ef_scoreboard.py \
    --runs-dir runs \
    --tiers "${TIERS_TO_RUN[@]}" \
    --seeds $SEEDS \
    --baseline-lift 2.61 \
    --out runs/phase26ef_scoreboard.md

elapsed=$(( $(date +%s) - t0 ))
echo
echo "Phase 26E/F/G overnight done in ${elapsed}s"
echo "Scoreboard: runs/phase26ef_scoreboard.md"
echo
echo "Upload results back to OSS for paris audit:"
echo "  ossutil cp -r runs/26{C2,E,F,G}_seed{42,43,44}/ \\"
echo "    oss://ledashi-oss/fromsz/handoffs/2026-05-08-phase26ef-results/"
