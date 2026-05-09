#!/usr/bin/env bash
#
# Phase 26F/G v3 overnight — 3 tier × 3 seed = 9 train+eval cycles
# on v3-FINAL-PATCHED main-board-only panels.
#
# Pre-flight
# ----------
# 1. Pull AurumQ-RL latest (commit 5e92594 or later — formula safe_div + safe_pow_clip)
# 2. Apply patch_train_v2_fp16.diff to enable --panel-dtype fp16:
#      cd AurumQ-RL && git apply --check phase26fg_v3/scripts/patch_train_v2_fp16.diff && \
#                       git apply phase26fg_v3/scripts/patch_train_v2_fp16.diff
# 3. Pull main_board v3 panels from OSS:
#      ossutil cp -r oss://ledashi-oss-sgp/aurumq-rl/handoffs/2026-05-08-panels-main-board-v3/ ./phase26fg_v3/
#      mkdir -p data/panels_v3 && cp -r phase26fg_v3/panels/* data/panels_v3/
#      # tech_panel_v1 (for tech_boll_percent + cmf_120d_pct_amt) from earlier bundle:
#      ossutil cp -r oss://ledashi-oss-sgp/aurumq-rl/handoffs/2026-05-07-tech-panel-v1/tech_panel_v1/ data/panels_v3/tech_panel_v1/
# 4. Verify integrity (HARD assertion main board only):
#      python phase26fg_v3/tools/verify_panels.py ./data/panels_v3/
# 5. Generate include lists from your 23A 353-col base:
#      cp YOUR_23A_353_LIST.txt phase26fg_v3/configs/include_columns_phase23a_353.txt
#      python phase26fg_v3/tools/generate_include_files.py
# 6. Build 2 combined panels (note: tier names use the -v3 suffix):
#      python phase26fg_v3/scripts/build_combined_panel_phase26fg_v3.py --tier 26C2-v3 --out data/panel_26c2_v3.parquet
#      python phase26fg_v3/scripts/build_combined_panel_phase26fg_v3.py --tier 26F-v3  --out data/panel_26f_v3.parquet
#      # 26G-v3 shares 26F-v3's panel (only encoder differs)
#
# Run
# ---
#   bash phase26fg_v3/run_phase26fg_v3_overnight.sh                # all 3 tiers × 3 seeds
#   bash phase26fg_v3/run_phase26fg_v3_overnight.sh 26F-v3         # one tier × 3 seeds
#
# Pass/fail (auto-rendered in scoreboard.md):
#   26C2-v3: best lift ≥ 1.85× (sanity, replicate v2 baseline)
#   26F-v3:  median > 26C2-v3 + 0.20× AND median > 2.15× (v2 26F median)
#   26G-v3:  median > 26F-v3 + 0.10×
#
# Total wall-clock target: ~3-4 GPU hours on RTX 4070.

set -euo pipefail

# ── Config ──────────────────────────────────────────────────────────────
PYTHON="${PYTHON:-.venv/Scripts/python.exe}"   # Windows AurumQ-RL default
SEEDS="${SEEDS:-42 43 44}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-300000}"
TRAIN_START="${TRAIN_START:-2023-01-03}"
TRAIN_END="${TRAIN_END:-2025-06-30}"
EVAL_START="${EVAL_START:-2025-07-01}"
EVAL_END="${EVAL_END:-2026-04-24}"
UNIVERSE="${UNIVERSE:-main_board_non_st}"

# Path to this bundle root (so scoreboard / tools resolve regardless of cwd)
BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Tier definitions
declare -A PANEL=(
    [26C2-v3]=data/panel_26c2_v3.parquet
    [26F-v3]=data/panel_26f_v3.parquet
    [26G-v3]=data/panel_26f_v3.parquet
)
declare -A INCLUDE_FILE=(
    [26C2-v3]="$BUNDLE_DIR/configs/include_columns_phase26c2_v3_336.txt"
    [26F-v3]="$BUNDLE_DIR/configs/include_columns_phase26f_v3_344.txt"
    [26G-v3]="$BUNDLE_DIR/configs/include_columns_phase26f_v3_344.txt"
)
declare -A ENCODER_HIDDEN=(
    [26C2-v3]=128,64
    [26F-v3]=128,64
    [26G-v3]=192,96
)
declare -A ENCODER_OUT_DIM=(
    [26C2-v3]=32
    [26F-v3]=32
    [26G-v3]=48
)
declare -A PANEL_DTYPE=(
    [26C2-v3]=fp32
    [26F-v3]=fp32
    [26G-v3]=fp16     # save ~1.5 GB VRAM on 4070 to fit 192,96 encoder
)

TIERS_TO_RUN=("$@")
if [ ${#TIERS_TO_RUN[@]} -eq 0 ]; then
    TIERS_TO_RUN=(26C2-v3 26F-v3 26G-v3)
fi

# Reject unknown tier names early — common typo: '26C2' instead of '26C2-v3'.
for tier in "${TIERS_TO_RUN[@]}"; do
    if [ -z "${PANEL[$tier]+x}" ]; then
        echo "ERROR: unknown tier '$tier'. Valid: 26C2-v3, 26F-v3, 26G-v3" >&2
        exit 2
    fi
done

mkdir -p runs/phase26fg_v3_logs

# Returns 0 on full success, 1 on any train/eval failure (IG is best-effort).
run_one() {
    local tier=$1
    local seed=$2
    local out_dir="runs/${tier}_seed${seed}"
    local log="runs/phase26fg_v3_logs/${tier}_seed${seed}.log"

    if [ -f "$out_dir/episode_eval.json" ]; then
        echo "[$(date +%H:%M:%S)] SKIP $tier seed=$seed (already done)"
        return 0
    fi

    echo "[$(date +%H:%M:%S)] BEGIN $tier seed=$seed enc=${ENCODER_HIDDEN[$tier]} out=${ENCODER_OUT_DIM[$tier]} panel=${PANEL_DTYPE[$tier]}"

    # ── train ──────────────────────────────────────────────
    # Capture rc explicitly: under `if ! cmd`, $? is the rc of `! cmd`
    # (always 0 on the failure branch), not of cmd itself. Use a
    # set-pipefail-friendly capture so the diagnostic is honest.
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
        --panel-dtype "${PANEL_DTYPE[$tier]}" \
        --learning-rate 1e-4 \
        --rollout-buffer index \
        --reward-mode main_wave_target \
        --tf32 --matmul-precision high \
        --seed "$seed" \
        --checkpoint-freq 50000 \
        --out-dir "$out_dir" \
        > "$log" 2>&1
    rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "[$(date +%H:%M:%S)] FAIL train $tier seed=$seed (rc=$rc, see $log)"
        return 1
    fi

    # ── eval ───────────────────────────────────────────────
    echo "[$(date +%H:%M:%S)] EVAL $tier seed=$seed"
    "$PYTHON" scripts/_eval_main_wave_episode.py \
        --run-dir "$out_dir" \
        --data-path "${PANEL[$tier]}" \
        --val-start "$EVAL_START" --val-end "$EVAL_END" \
        --top-k 3 5 --universe-filter "$UNIVERSE" \
        >> "$log" 2>&1
    rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "[$(date +%H:%M:%S)] FAIL eval $tier seed=$seed (rc=$rc, see $log)"
        return 1
    fi

    # Sanity: the eval step must have produced episode_eval.json. If not,
    # don't claim DONE — let the outer loop log a FAILURE.
    if [ ! -f "$out_dir/episode_eval.json" ]; then
        echo "[$(date +%H:%M:%S)] FAIL $tier seed=$seed: episode_eval.json missing after eval"
        return 1
    fi

    # ── IG / permutation (only seed=42, best-effort, NON-blocking) ─────
    if [ "$seed" = "42" ]; then
        echo "[$(date +%H:%M:%S)] IG $tier seed=42"
        "$PYTHON" scripts/eval_factor_importance.py \
            --run-dir "$out_dir" \
            --data-path "${PANEL[$tier]}" \
            --val-start "$EVAL_START" --val-end "$EVAL_END" \
            --top-k 5 --forward-period 5 --n-seeds 3 \
            >> "$log" 2>&1
        rc=$?
        if [ "$rc" -ne 0 ]; then
            echo "[$(date +%H:%M:%S)] WARN IG failed for $tier seed=42 (rc=$rc, non-blocking)"
        fi
    fi

    echo "[$(date +%H:%M:%S)] DONE $tier seed=$seed"
    return 0
}

t0=$(date +%s)
declare -i failures=0
for tier in "${TIERS_TO_RUN[@]}"; do
    for seed in $SEEDS; do
        if ! run_one "$tier" "$seed"; then
            failures=$((failures + 1))
            echo "FAILED $tier seed=$seed (continuing — see runs/phase26fg_v3_logs/)"
        fi
    done
done

# ── Aggregate scoreboard (only when ALL runs succeeded) ────────────────
# Codex round-2 finding: scoreboard runs even if some seeds failed, then
# computes median over an incomplete set and prints a "promote tier X"
# recommendation. Block this — fail loudly + leave scoreboard absent so
# the operator sees the gap.
elapsed=$(( $(date +%s) - t0 ))
if [ "$failures" -gt 0 ]; then
    echo
    echo "[INCOMPLETE] $failures train/eval cycle(s) failed. Scoreboard NOT generated"
    echo "             — recommendations require all seeds. See logs in runs/phase26fg_v3_logs/"
    echo "Wall-clock: ${elapsed}s"
    exit "$failures"
fi

"$PYTHON" "$BUNDLE_DIR/scripts/phase26fg_v3_scoreboard.py" \
    --runs-dir runs \
    --tiers "${TIERS_TO_RUN[@]}" \
    --seeds $SEEDS \
    --baseline-c2-v2-median 1.70 \
    --baseline-f-v2-median 2.15 \
    --out runs/phase26fg_v3_scoreboard.md

echo
echo "Phase 26F/G v3 overnight done in ${elapsed}s. failures=0"
echo "Scoreboard: runs/phase26fg_v3_scoreboard.md"
echo
echo "Upload back to OSS for paris audit:"
echo "  ossutil cp -r runs/26{C2,F,G}-v3_seed{42,43,44}/ \\"
echo "    runs/phase26fg_v3_scoreboard.md \\"
echo "    runs/phase26fg_v3_logs/ \\"
echo "    oss://ledashi-oss/fromsz/handoffs/2026-05-08-phase26fg-v3-results/"

exit 0
