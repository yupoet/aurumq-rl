#!/usr/bin/env bash
# Phase 25 overnight: wait 25A → eval → 25B → eval → 25C → eval →
# permutation importance on best → cross-phase comparison.
#
# Continues unattended until ~8am tomorrow.

set -u
PYTHON=D:/dev/aurumq-rl/.venv/Scripts/python.exe
DATA=data/factor_panel_combined_short_2023_2026.parquet
WEIGHTS=runs/phase25_factor_weights.json
LOG=runs/_phase25_overnight.log

echo "[overnight] $(date) Phase 25 overnight start" | tee -a $LOG

wait_for_train() {
    local rundir=$1
    local label=$2
    echo "[overnight] $(date) waiting for $label train..." | tee -a $LOG
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
    echo "[overnight] $(date) eval $label ..." | tee -a $LOG
    "$PYTHON" scripts/_eval_main_wave_episode.py \
        --run-dir "$rundir" --data-path "$DATA" \
        --val-start 2025-07-01 --val-end 2026-04-24 \
        --top-k 3 5 \
        > "$rundir/episode_eval.log" 2>&1 || {
            echo "[overnight] $label eval FAILED" | tee -a $LOG
            tail -20 "$rundir/episode_eval.log" | tee -a $LOG
        }
    if [[ -f "$rundir/episode_eval.json" ]]; then
        local best_label=$("$PYTHON" -c "
import json
d = json.load(open(r'$rundir/episode_eval.json'))
rows = [r for r in d['rows'] if r['top_k'] == 5]
if rows:
    print(max(rows, key=lambda r: r['t_minus_1_hit_rate'])['checkpoint_label'])
else:
    print('final')
")
        echo "[overnight] $label best ckpt by T-1: $best_label" | tee -a $LOG
        "$PYTHON" scripts/_diagnose_t1_hits.py \
            --picks "$rundir/episode_picks.jsonl" \
            --data-path "$DATA" \
            --top-k 5 --ckpt-label "$best_label" \
            --out "$rundir/t1_diagnostic.md" \
            --days-before 20 \
            > "$rundir/diag.log" 2>&1 || true
    fi
    echo "[overnight] $(date) $label eval+diag done" | tee -a $LOG
}

train_run() {
    local rundir=$1
    local seed=$2
    local label=$3
    rm -rf "$rundir"
    mkdir -p "$rundir"
    echo "[overnight] $(date) launching $label (seed=$seed) ..." | tee -a $LOG
    "$PYTHON" scripts/train_v2.py \
        --total-timesteps 300000 \
        --data-path "$DATA" \
        --start-date 2023-01-03 --end-date 2025-06-30 \
        --universe-filter main_board_non_st \
        --n-envs 16 --episode-length 240 \
        --batch-size 1024 --n-steps 1024 --n-epochs 10 \
        --learning-rate 1e-4 --target-kl 0.30 --max-grad-norm 0.5 \
        --rollout-buffer index --tf32 --matmul-precision high \
        --forward-period 5 --top-k 5 \
        --reward-mode main_wave_target \
        --add-technical-factors \
        --factor-weights-json "$WEIGHTS" \
        --drop-factor-prefix mkt_ \
        --checkpoint-freq 25000 \
        --seed "$seed" \
        --out-dir "$rundir" \
        > "$rundir/train.log" 2>&1
}

# --- Phase 25A: already running. Wait + eval. ---
wait_for_train runs/phase25a_weighted_seed42 25A
eval_run runs/phase25a_weighted_seed42 25A

# --- Phase 25B: seed=1 ---
train_run runs/phase25b_weighted_seed1 1 25B
wait_for_train runs/phase25b_weighted_seed1 25B
eval_run runs/phase25b_weighted_seed1 25B

# --- Phase 25C: seed=2 ---
train_run runs/phase25c_weighted_seed2 2 25C
wait_for_train runs/phase25c_weighted_seed2 25C
eval_run runs/phase25c_weighted_seed2 25C

# --- Cross-phase comparison ---
echo "[overnight] $(date) writing cross-phase comparison..." | tee -a $LOG
"$PYTHON" - <<'PY' | tee -a $LOG
import json
runs = [
    ('Phase 16a (V1 forward_10d, prod)',  'runs/phase16a_fixed_drop_mkt_300k'),
    ('Phase 22A (V1 main_wave_hold s42)', 'runs/phase22a_main_wave_v1_seed42'),
    ('Phase 22C (V1 main_wave_hold tk3)', 'runs/phase22c_topk3_seed42'),
    ('Phase 23A (target s42)',            'runs/phase23a_episode_seed42'),
    ('Phase 24A (target+tech s42)',       'runs/phase24a_tech_seed42'),
    ('Phase 25A (target+tech+weights s42)', 'runs/phase25a_weighted_seed42'),
    ('Phase 25B (s1)',                    'runs/phase25b_weighted_seed1'),
    ('Phase 25C (s2)',                    'runs/phase25c_weighted_seed2'),
]
print(f'\n{"Run":<48} {"topK":>5} {"best_step":>10} {"T1_hit":>8} {"T1_lift":>9} {"T13_hit":>9} {"avg_peak":>10} {"daily_T1":>10} {"eval_v23":>10}')
print('-' * 130)
for name, rundir in runs:
    fp = f'{rundir}/episode_eval.json'
    try:
        d = json.load(open(fp))
    except FileNotFoundError:
        print(f'{name:<48} (no eval)')
        continue
    rows = d.get('rows', [])
    by_topk = {}
    for r in rows:
        k = r.get('top_k')
        cur = by_topk.get(k)
        if cur is None or r['t_minus_1_hit_rate'] > cur['t_minus_1_hit_rate']:
            by_topk[k] = r
    for k in sorted(by_topk):
        r = by_topk[k]
        print(f'{name:<48} {r["top_k"]:>5} {r["checkpoint_label"]:>10} '
              f'{r["t_minus_1_hit_rate"]:>8.4f} {r["t1_lift_over_base"]:>7.2f}x  '
              f'{r["t_minus_3_hit_rate"]:>9.4f} {r["avg_peak_return_of_hits"]:>+10.4f} '
              f'{r["daily_t_minus_1_precision"]:>10.4f} {r["eval_score_v23"]:>+10.4f}')
print('\nBase rate: T-1 = 0.0089')
PY

# --- Permutation importance on best Phase 25 ckpt (Method A end-verification) ---
echo "[overnight] $(date) computing best Phase 25 run by T-1 hit..." | tee -a $LOG
BEST_INFO=$("$PYTHON" -c "
import json, glob
best = (None, None, None, -1)  # (rundir, ckpt_label, ckpt_path, t1)
for d in ['runs/phase25a_weighted_seed42', 'runs/phase25b_weighted_seed1', 'runs/phase25c_weighted_seed2']:
    fp = f'{d}/episode_eval.json'
    try:
        ev = json.load(open(fp))
    except FileNotFoundError:
        continue
    for r in ev['rows']:
        if r['top_k'] != 5: continue
        if r['t_minus_1_hit_rate'] > best[3]:
            label = r['checkpoint_label']
            if label == 'final':
                ckpt = f'{d}/ppo_final.zip'
            else:
                step = label.replace('step', '')
                ckpt = f'{d}/checkpoints/ppo_{step}_steps.zip'
            best = (d, label, ckpt, r['t_minus_1_hit_rate'])
print(f'{best[0]}|{best[1]}|{best[2]}|{best[3]}')
")
IFS='|' read -r BEST_DIR BEST_LABEL BEST_CKPT BEST_T1 <<< "$BEST_INFO"
echo "[overnight] best Phase 25: $BEST_DIR ckpt=$BEST_LABEL T1=$BEST_T1" | tee -a $LOG

if [[ -f "$BEST_CKPT" ]]; then
    echo "[overnight] $(date) running permutation importance on $BEST_DIR ..." | tee -a $LOG
    "$PYTHON" scripts/eval_factor_importance.py \
        --run-dir "$BEST_DIR" \
        --data-path "$DATA" \
        --val-start 2025-07-01 --val-end 2026-04-24 \
        --top-k 5 \
        --checkpoint "$BEST_CKPT" \
        --out-json "$BEST_DIR/factor_importance_full.json" \
        --ig-alpha-steps 50 --ig-batch-size 16 \
        --n-seeds 3 \
        > "$BEST_DIR/factor_importance_full.log" 2>&1 || {
            echo "[overnight] permutation FAILED" | tee -a $LOG
            tail -20 "$BEST_DIR/factor_importance_full.log" | tee -a $LOG
        }
fi

echo "[overnight] $(date) Phase 25 overnight DONE" | tee -a $LOG
