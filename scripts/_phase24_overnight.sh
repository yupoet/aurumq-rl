#!/usr/bin/env bash
# Phase 24 overnight: 3-seed pipeline (A=42, B=1, C=2) with technical factors.
# Each: train → episode eval → T-1 diagnostic. Final cross-phase compare.

set -u
PYTHON=D:/dev/aurumq-rl/.venv/Scripts/python.exe
DATA=data/factor_panel_combined_short_2023_2026.parquet
LOG=runs/_phase24_overnight.log

echo "[overnight] $(date) Phase 24 start" | tee -a $LOG

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
        --drop-factor-prefix mkt_ \
        --checkpoint-freq 25000 \
        --seed "$seed" \
        --out-dir "$rundir" \
        > "$rundir/train.log" 2>&1
    return $?
}

eval_run() {
    local rundir=$1
    local label=$2
    echo "[overnight] $(date) eval $label ..." | tee -a $LOG
    "$PYTHON" scripts/_eval_main_wave_episode.py \
        --run-dir "$rundir" \
        --data-path "$DATA" \
        --val-start 2025-07-01 --val-end 2026-04-24 \
        --top-k 3 5 \
        > "$rundir/episode_eval.log" 2>&1 || {
            echo "[overnight] $label eval failed" | tee -a $LOG
            tail -20 "$rundir/episode_eval.log" | tee -a $LOG
        }
    # Find best step by T-1 hit rate
    if [[ -f "$rundir/episode_eval.json" ]]; then
        local best_label=$("$PYTHON" -c "
import json
d = json.load(open('$rundir/episode_eval.json'))
rows = [r for r in d['rows'] if r['top_k'] == 5]
if rows:
    best = max(rows, key=lambda r: r['t_minus_1_hit_rate'])
    print(best['checkpoint_label'])
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
}

# Phase 24A: seed=42
train_run runs/phase24a_tech_seed42 42 24A
eval_run runs/phase24a_tech_seed42 24A

# Phase 24B: seed=1
train_run runs/phase24b_tech_seed1 1 24B
eval_run runs/phase24b_tech_seed1 24B

# Phase 24C: seed=2
train_run runs/phase24c_tech_seed2 2 24C
eval_run runs/phase24c_tech_seed2 24C

# Final cross-phase comparison
echo "[overnight] $(date) writing comparison summary..." | tee -a $LOG
"$PYTHON" - <<'PY' | tee -a $LOG
import json
runs = [
    ('Phase 16a (V1 forward_10d, prod)',  'runs/phase16a_fixed_drop_mkt_300k'),
    ('Phase 22A (V1 main_wave_hold s42)', 'runs/phase22a_main_wave_v1_seed42'),
    ('Phase 22B (V1 main_wave_hold s1)',  'runs/phase22b_main_wave_v1_seed1'),
    ('Phase 22C (V1 main_wave_hold tk3)', 'runs/phase22c_topk3_seed42'),
    ('Phase 23A (target seed=42)',        'runs/phase23a_episode_seed42'),
    ('Phase 24A (target+tech s42)',       'runs/phase24a_tech_seed42'),
    ('Phase 24B (target+tech s1)',        'runs/phase24b_tech_seed1'),
    ('Phase 24C (target+tech s2)',        'runs/phase24c_tech_seed2'),
]
print(f'\n{"Run":<40} {"topK":>5} {"best_step":>10} {"T1_hit":>8} {"T1_lift":>8} {"T13_hit":>8} {"avg_peak":>9} {"avg_dur":>8} {"daily_T1":>9} {"eval_v23":>10}')
print('-' * 130)
for name, rundir in runs:
    fp = f'{rundir}/episode_eval.json'
    try:
        d = json.load(open(fp))
    except FileNotFoundError:
        print(f'{name:<40} (no episode_eval.json)')
        continue
    rows = d.get('rows', [])
    by_topk = {}
    for r in rows:
        k = r.get('top_k')
        cur = by_topk.get(k)
        # rank by T-1 specifically (user's primary metric)
        if cur is None or r['t_minus_1_hit_rate'] > cur['t_minus_1_hit_rate']:
            by_topk[k] = r
    for k in sorted(by_topk):
        r = by_topk[k]
        print(f'{name:<40} {r["top_k"]:>5} {r["checkpoint_label"]:>10} '
              f'{r["t_minus_1_hit_rate"]:>8.4f} {r["t1_lift_over_base"]:>7.2f}x '
              f'{r["t_minus_3_hit_rate"]:>8.4f} {r["avg_peak_return_of_hits"]:>+9.4f} '
              f'{r["avg_duration_of_hits"]:>8.2f} {r["daily_t_minus_1_precision"]:>9.4f} '
              f'{r["eval_score_v23"]:>+10.4f}')
print(f'\nBase rate: T-1 = 0.0089 (0.89%)')
PY

echo "[overnight] $(date) DONE" | tee -a $LOG
