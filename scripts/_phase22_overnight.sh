#!/usr/bin/env bash
# Phase 22 overnight pipeline: A (running) → B → C, with eval + inspect after each.
# Writes a final comparison handoff at the end. Designed to run unattended.

set -u
PYTHON=D:/dev/aurumq-rl/.venv/Scripts/python.exe
DATA=data/factor_panel_combined_short_2023_2026.parquet
LOG=runs/_phase22_overnight.log

echo "[overnight] $(date) start" | tee -a $LOG

# ---------- Helpers ----------

wait_for_train() {
    local rundir=$1
    local label=$2
    echo "[overnight] $(date) waiting for $label train to finish..." | tee -a $LOG
    until grep -qE "(\[train_v2\] final model saved|Traceback|Killed|RuntimeError|CUDA out of memory)" "$rundir/train.log" 2>/dev/null; do
        sleep 60
    done
    if grep -q "\[train_v2\] final model saved" "$rundir/train.log"; then
        echo "[overnight] $(date) $label DONE" | tee -a $LOG
        return 0
    fi
    echo "[overnight] $(date) $label FAILED" | tee -a $LOG
    tail -20 "$rundir/train.log" | tee -a $LOG
    return 1
}

eval_run() {
    local rundir=$1
    local label=$2
    echo "[overnight] $(date) eval $label..." | tee -a $LOG
    "$PYTHON" scripts/_eval_main_wave_v1.py \
        --run-dir "$rundir" \
        --data-path "$DATA" \
        --val-start 2025-07-01 --val-end 2026-04-24 \
        --universe-filter main_board_non_st \
        --top-k 3 5 \
        > "$rundir/eval.log" 2>&1 || {
            echo "[overnight] $label eval failed" | tee -a $LOG
            tail -20 "$rundir/eval.log" | tee -a $LOG
        }
    "$PYTHON" scripts/_inspect_main_wave_picks.py \
        --picks "$rundir/main_wave_picks.jsonl" \
        --data-path "$DATA" \
        --top-k 3 \
        --out "$rundir/inspect_top3.md" \
        > "$rundir/inspect.log" 2>&1 || true
    echo "[overnight] $(date) $label eval+inspect done" | tee -a $LOG
}

train_run() {
    local rundir=$1
    local seed=$2
    local top_k=$3
    local total_steps=$4
    local label=$5
    rm -rf "$rundir"
    mkdir -p "$rundir"
    echo "[overnight] $(date) launching $label (seed=$seed, top_k=$top_k, steps=$total_steps)..." | tee -a $LOG
    "$PYTHON" scripts/train_v2.py \
        --total-timesteps "$total_steps" \
        --data-path "$DATA" \
        --start-date 2023-01-03 --end-date 2025-06-30 \
        --universe-filter main_board_non_st \
        --n-envs 16 --episode-length 240 \
        --batch-size 1024 --n-steps 1024 --n-epochs 10 \
        --learning-rate 1e-4 --target-kl 0.30 --max-grad-norm 0.5 \
        --rollout-buffer index --tf32 --matmul-precision high \
        --forward-period 5 --top-k "$top_k" \
        --reward-mode main_wave_hold \
        --drop-factor-prefix mkt_ \
        --checkpoint-freq 25000 \
        --seed "$seed" \
        --out-dir "$rundir" \
        > "$rundir/train.log" 2>&1
}

# ---------- Pipeline ----------

# Phase 22A is already running (started in prior turn); just wait for it.
wait_for_train runs/phase22a_main_wave_v1_seed42 22A
eval_run runs/phase22a_main_wave_v1_seed42 22A

# Phase 22B: same config, seed=1 — robustness check
train_run runs/phase22b_main_wave_v1_seed1 1 5 300000 22B
wait_for_train runs/phase22b_main_wave_v1_seed1 22B
eval_run runs/phase22b_main_wave_v1_seed1 22B

# Phase 22C: top_k=3, seed=42 — concentration check (200k since smaller K converges faster)
train_run runs/phase22c_topk3_seed42 42 3 200000 22C
wait_for_train runs/phase22c_topk3_seed42 22C
eval_run runs/phase22c_topk3_seed42 22C

# ---------- Final comparison ----------
echo "[overnight] $(date) writing comparison summary..." | tee -a $LOG
"$PYTHON" - <<'PY' | tee -a $LOG
"""Quick comparison summary of A/B/C + Phase 16a baseline."""
import json
from pathlib import Path

ROOT = Path("runs")
runs = [
    ("Phase 16a (V1 forward_10d, prod)", ROOT / "phase16a_fixed_drop_mkt_300k", "step224928"),
    ("Phase 21A (V2 forward_10d)",       ROOT / "phase21_21a_v2_drop_mkt_seed42", "step149952"),
    ("Phase 22A (V1 main_wave seed42)",  ROOT / "phase22a_main_wave_v1_seed42", None),
    ("Phase 22B (V1 main_wave seed1)",   ROOT / "phase22b_main_wave_v1_seed1", None),
    ("Phase 22C (V1 main_wave topk=3)",  ROOT / "phase22c_topk3_seed42", None),
]
print(f"\n{'Run':<40} {'topK':>4} {'best_step':>10} {'hit_rate':>9} {'win_rate':>9} {'avg_hold':>10} {'avg_dd':>8} {'eval_score':>11}")
print("-" * 110)
for name, rundir, want_step in runs:
    fp = rundir / "main_wave_eval.json"
    if not fp.exists():
        print(f"{name:<40} (no eval found at {fp})")
        continue
    data = json.loads(fp.read_text(encoding="utf-8"))
    rows = data.get("rows", [])
    if not rows:
        print(f"{name:<40} (empty rows)")
        continue
    # Pick best by eval_score per top_k
    by_topk = {}
    for r in rows:
        k = r.get("top_k")
        if want_step is not None and r.get("checkpoint_label") != want_step:
            continue
        cur = by_topk.get(k)
        if cur is None or r["eval_score"] > cur["eval_score"]:
            by_topk[k] = r
    for k in sorted(by_topk):
        r = by_topk[k]
        print(f"{name:<40} {r['top_k']:>4} {r['checkpoint_label']:>10} "
              f"{r['main_wave_hit_rate']:>9.4f} {r['basic_win_rate']:>9.4f} "
              f"{r['avg_hold_return']:>+10.4f} {r['avg_max_drawdown']:>8.4f} "
              f"{r['eval_score']:>+11.4f}")
print(f"\nUniverse base rate (random pick): ~5.72% hit_main_wave")
PY

echo "[overnight] $(date) DONE" | tee -a $LOG
