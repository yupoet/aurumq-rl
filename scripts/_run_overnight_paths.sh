#!/usr/bin/env bash
# Unattended overnight orchestrator — chains:
#   Path 4 ensemble → Path 2 grid → Path 2 ensemble → Path 3 grid → final ensemble → OSS upload
# Assumes Path 4 grid (sl_path4) already completed.
#
# Each step is independent + idempotent (skip when output exists). Failures
# in one step log + continue to the next, since downstream steps may still
# work on partial results.
set -uo pipefail
cd /d/dev/aurumq-rl

PY=".venv/Scripts/python.exe"
LOG_DIR="runs/sl_overnight_logs"
mkdir -p "$LOG_DIR"

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

# 0. Wait for Path 4 grid to finish (poll runs/sl_path4 results.json count)
echo "[$(date +%H:%M:%S)] waiting for Path 4 grid (need 36 results.json) ..."
while [ "$(ls runs/sl_path4/*/results.json 2>/dev/null | wc -l)" -lt 36 ]; do
    n=$(ls runs/sl_path4/*/results.json 2>/dev/null | wc -l)
    echo "[$(date +%H:%M:%S)]   path4 grid: $n / 36 done; sleep 60"
    sleep 60
done
echo "[$(date +%H:%M:%S)] Path 4 grid complete (36/36) — chain begins"

# 1. Path 4 ensemble (LightGBM on rank-z panel)
if [ ! -f runs/sl_path4/ensemble.json ]; then
    step "path4_ensemble" $PY scripts/p3/path1_ensemble.py \
        --bundle data/p3_4070 \
        --runs-root runs/sl_path4 \
        --out-root runs/sl_path4 \
        --top-k-configs 3
fi

# 2. Path 4 RESULTS.md
if [ -f runs/sl_path4/ensemble.json ] && [ ! -f runs/sl_path4/RESULTS.md ]; then
    step "path4_results" $PY -c "
import json
from datetime import date
d = json.load(open('runs/sl_path4/ensemble.json'))
def fmt(b): return f'{b[\"primary_mean_top50_proximity_excess\"]:+.6f}'
def fmt_sp(b): return f'{b[\"spearman\"]:+.4f}'
def fmt_hit(b): return f'{b[\"top50_T1_hit_rate\"]*100:.2f}%'
def fmt_ece(b): return f'{b[\"ece_10bin\"]:.5f}'
md = [
    '# SL Path 4 — LightGBM β-regression on rank-z features',
    '',
    f'**Date**: {date.today().isoformat()}',
    '**Spec**: docs/superpowers/specs/2026-05-09-sl-ensemble-training-design.md §4.2',
    '**Foundation**: feature_panel_clean.parquet (cross-sectional rank-z transform)',
    '',
    '## Headline',
    '',
    '| Window | Metric | Path4 raw | Path4 cal | Paris baseline | Δ |',
    '|---|---|---:|---:|---:|---:|',
    f'| H1 | primary | {fmt(d[\"ensemble_raw_H1\"])} | {fmt(d[\"ensemble_calibrated_H1\"])} | {fmt(d[\"paris_baseline_H1\"])} | {d[\"ensemble_calibrated_H1\"][\"primary_mean_top50_proximity_excess\"]-d[\"paris_baseline_H1\"][\"primary_mean_top50_proximity_excess\"]:+.6f} |',
    f'| H1 | spearman | {fmt_sp(d[\"ensemble_raw_H1\"])} | {fmt_sp(d[\"ensemble_calibrated_H1\"])} | {fmt_sp(d[\"paris_baseline_H1\"])} | — |',
    f'| H1 | top50_T1_hit | {fmt_hit(d[\"ensemble_raw_H1\"])} | {fmt_hit(d[\"ensemble_calibrated_H1\"])} | {fmt_hit(d[\"paris_baseline_H1\"])} | — |',
    f'| H1 | ECE | {fmt_ece(d[\"ensemble_raw_H1\"])} | {fmt_ece(d[\"ensemble_calibrated_H1\"])} | {fmt_ece(d[\"paris_baseline_H1\"])} | — |',
    f'| H2 | primary | {fmt(d[\"ensemble_raw_H2\"])} | {fmt(d[\"ensemble_calibrated_H2\"])} | {fmt(d[\"paris_baseline_H2\"])} | {d[\"ensemble_calibrated_H2\"][\"primary_mean_top50_proximity_excess\"]-d[\"paris_baseline_H2\"][\"primary_mean_top50_proximity_excess\"]:+.6f} |',
    f'| H2 | spearman | {fmt_sp(d[\"ensemble_raw_H2\"])} | {fmt_sp(d[\"ensemble_calibrated_H2\"])} | {fmt_sp(d[\"paris_baseline_H2\"])} | — |',
    f'| H2 | top50_T1_hit | {fmt_hit(d[\"ensemble_raw_H2\"])} | {fmt_hit(d[\"ensemble_calibrated_H2\"])} | {fmt_hit(d[\"paris_baseline_H2\"])} | — |',
    f'| H2 | ECE | {fmt_ece(d[\"ensemble_raw_H2\"])} | {fmt_ece(d[\"ensemble_calibrated_H2\"])} | {fmt_ece(d[\"paris_baseline_H2\"])} | — |',
    '',
    '## Chosen runs (top-3 configs × 3 seeds)',
    '',
]
for n in d['chosen_runs']:
    md.append(f'- {n}')
md.append('')
md.append('## vs Path 1 (raw v3_344 panel)')
md.append('')
md.append('Read runs/sl_path1/ensemble.json + this file side-by-side.')
md.append('Cross-path final ensemble combines both: runs/sl_final/RESULTS.md.')
open('runs/sl_path4/RESULTS.md', 'w', encoding='utf-8').write('\n'.join(md))
print('wrote runs/sl_path4/RESULTS.md')
"
fi

# 3. Path 2 grid (catboost + xgboost on rank-z panel)
if [ ! -f runs/sl_path2/ensemble.json ]; then
    if ! ls runs/sl_path2/*/results.json > /dev/null 2>&1 || [ "$(ls runs/sl_path2/*/results.json 2>/dev/null | wc -l)" -lt 36 ]; then
        step "path2_grid" $PY scripts/p3/path2_grid.py \
            --bundle data/p3_4070 \
            --feature-panel feature_panel_clean.parquet \
            --out-root runs/sl_path2 \
            --num-iterations 2000
    fi
fi

# 4. Path 2 ensemble
if [ ! -f runs/sl_path2/ensemble.json ]; then
    step "path2_ensemble" $PY scripts/p3/path1_ensemble.py \
        --bundle data/p3_4070 \
        --runs-root runs/sl_path2 \
        --out-root runs/sl_path2 \
        --top-k-configs 4
fi

# 5. Path 2 RESULTS.md (similar to Path 4's)
if [ -f runs/sl_path2/ensemble.json ] && [ ! -f runs/sl_path2/RESULTS.md ]; then
    step "path2_results" $PY -c "
import json
from datetime import date
d = json.load(open('runs/sl_path2/ensemble.json'))
def fmt(b): return f'{b[\"primary_mean_top50_proximity_excess\"]:+.6f}'
def fmt_sp(b): return f'{b[\"spearman\"]:+.4f}'
def fmt_hit(b): return f'{b[\"top50_T1_hit_rate\"]*100:.2f}%'
def fmt_ece(b): return f'{b[\"ece_10bin\"]:.5f}'
md = [
    '# SL Path 2 — CatBoost + XGBoost ensemble (rank-z features)',
    '',
    f'**Date**: {date.today().isoformat()}',
    '**Spec**: docs/superpowers/specs/2026-05-09-sl-ensemble-training-design.md §4.3',
    '**Foundation**: rank-z panel (Path 4)',
    '',
    '## Headline',
    '',
    '| Window | Metric | Path2 raw | Path2 cal | Paris | Δ |',
    '|---|---|---:|---:|---:|---:|',
    f'| H1 | primary | {fmt(d[\"ensemble_raw_H1\"])} | {fmt(d[\"ensemble_calibrated_H1\"])} | {fmt(d[\"paris_baseline_H1\"])} | {d[\"ensemble_calibrated_H1\"][\"primary_mean_top50_proximity_excess\"]-d[\"paris_baseline_H1\"][\"primary_mean_top50_proximity_excess\"]:+.6f} |',
    f'| H1 | spearman | {fmt_sp(d[\"ensemble_raw_H1\"])} | {fmt_sp(d[\"ensemble_calibrated_H1\"])} | {fmt_sp(d[\"paris_baseline_H1\"])} | — |',
    f'| H1 | top50_T1_hit | {fmt_hit(d[\"ensemble_raw_H1\"])} | {fmt_hit(d[\"ensemble_calibrated_H1\"])} | {fmt_hit(d[\"paris_baseline_H1\"])} | — |',
    f'| H1 | ECE | {fmt_ece(d[\"ensemble_raw_H1\"])} | {fmt_ece(d[\"ensemble_calibrated_H1\"])} | {fmt_ece(d[\"paris_baseline_H1\"])} | — |',
    f'| H2 | primary | {fmt(d[\"ensemble_raw_H2\"])} | {fmt(d[\"ensemble_calibrated_H2\"])} | {fmt(d[\"paris_baseline_H2\"])} | {d[\"ensemble_calibrated_H2\"][\"primary_mean_top50_proximity_excess\"]-d[\"paris_baseline_H2\"][\"primary_mean_top50_proximity_excess\"]:+.6f} |',
    f'| H2 | spearman | {fmt_sp(d[\"ensemble_raw_H2\"])} | {fmt_sp(d[\"ensemble_calibrated_H2\"])} | {fmt_sp(d[\"paris_baseline_H2\"])} | — |',
    f'| H2 | top50_T1_hit | {fmt_hit(d[\"ensemble_raw_H2\"])} | {fmt_hit(d[\"ensemble_calibrated_H2\"])} | {fmt_hit(d[\"paris_baseline_H2\"])} | — |',
    f'| H2 | ECE | {fmt_ece(d[\"ensemble_raw_H2\"])} | {fmt_ece(d[\"ensemble_calibrated_H2\"])} | {fmt_ece(d[\"paris_baseline_H2\"])} | — |',
    '',
    '## Chosen runs (CatBoost + XGBoost mix)',
    '',
]
for n in d['chosen_runs']:
    md.append(f'- {n}')
open('runs/sl_path2/RESULTS.md', 'w', encoding='utf-8').write('\n'.join(md))
print('wrote runs/sl_path2/RESULTS.md')
"
fi

# 6. Path 3 grid (TabNet on GPU)
if [ ! -f runs/sl_path3/ensemble.json ]; then
    if ! ls runs/sl_path3/*/results.json > /dev/null 2>&1 || [ "$(ls runs/sl_path3/*/results.json 2>/dev/null | wc -l)" -lt 8 ]; then
        step "path3_grid" $PY scripts/p3/path3_grid.py \
            --bundle data/p3_4070 \
            --feature-panel feature_panel_clean.parquet \
            --out-root runs/sl_path3
    fi
fi

# 7. Path 3 ensemble
if [ ! -f runs/sl_path3/ensemble.json ] && ls runs/sl_path3/*/results.json > /dev/null 2>&1; then
    step "path3_ensemble" $PY scripts/p3/path1_ensemble.py \
        --bundle data/p3_4070 \
        --runs-root runs/sl_path3 \
        --out-root runs/sl_path3 \
        --top-k-configs 2
fi

# 8. Final cross-path ensemble (Path 1 + 4 + 2 + 3 if present)
if [ ! -f runs/sl_final/ensemble.json ]; then
    step "final_ensemble" $PY scripts/p3/final_ensemble.py \
        --bundle data/p3_4070 \
        --paths sl_path1 sl_path4 sl_path2 sl_path3 \
        --top-k-configs-per-path 3 \
        --out runs/sl_final
fi

# 9. Final RESULTS.md
if [ -f runs/sl_final/ensemble.json ] && [ ! -f runs/sl_final/RESULTS.md ]; then
    step "final_results" $PY -c "
import json
from datetime import date
d = json.load(open('runs/sl_final/ensemble.json'))
def fmt(b): return f'{b[\"primary_mean_top50_proximity_excess\"]:+.6f}'
def fmt_sp(b): return f'{b[\"spearman\"]:+.4f}'
def fmt_hit(b): return f'{b[\"top50_T1_hit_rate\"]*100:.2f}%'
def fmt_ece(b): return f'{b[\"ece_10bin\"]:.5f}'
md = [
    '# SL Final — cross-path rank-mean ensemble',
    '',
    f'**Date**: {date.today().isoformat()}',
    f'**Paths used**: {\", \".join(d[\"paths_used\"])}',
    f'**Calibration rows on H1**: {d[\"n_calibration_rows_H1\"]}',
    '',
    '## Multi-path scoreboard (H1 + H2)',
    '',
    '| Source | H1 primary | H1 spearman | H1 T1_hit | H2 primary | H2 spearman | H2 T1_hit |',
    '|---|---:|---:|---:|---:|---:|---:|',
]
md.append(f'| paris baseline | {fmt(d[\"paris_baseline_H1\"])} | {fmt_sp(d[\"paris_baseline_H1\"])} | {fmt_hit(d[\"paris_baseline_H1\"])} | {fmt(d[\"paris_baseline_H2\"])} | {fmt_sp(d[\"paris_baseline_H2\"])} | {fmt_hit(d[\"paris_baseline_H2\"])} |')
for p, e in d['per_path_ensemble'].items():
    md.append(f'| {p} (path-only) | {fmt(e[\"H1\"])} | {fmt_sp(e[\"H1\"])} | {fmt_hit(e[\"H1\"])} | {fmt(e[\"H2\"])} | {fmt_sp(e[\"H2\"])} | {fmt_hit(e[\"H2\"])} |')
md.append(f'| **FINAL_raw** | **{fmt(d[\"final_ensemble_raw_H1\"])}** | {fmt_sp(d[\"final_ensemble_raw_H1\"])} | {fmt_hit(d[\"final_ensemble_raw_H1\"])} | **{fmt(d[\"final_ensemble_raw_H2\"])}** | {fmt_sp(d[\"final_ensemble_raw_H2\"])} | {fmt_hit(d[\"final_ensemble_raw_H2\"])} |')
md.append(f'| **FINAL_cal** | **{fmt(d[\"final_ensemble_calibrated_H1\"])}** | {fmt_sp(d[\"final_ensemble_calibrated_H1\"])} | {fmt_hit(d[\"final_ensemble_calibrated_H1\"])} | **{fmt(d[\"final_ensemble_calibrated_H2\"])}** | {fmt_sp(d[\"final_ensemble_calibrated_H2\"])} | {fmt_hit(d[\"final_ensemble_calibrated_H2\"])} |')
md.append('')
md.append('## Chosen runs per path (top-3 configs each)')
md.append('')
for p, names in d['chosen_per_path'].items():
    md.append(f'### {p} ({len(names)} runs)')
    for n in names:
        md.append(f'- {n}')
    md.append('')
open('runs/sl_final/RESULTS.md', 'w', encoding='utf-8').write('\n'.join(md))
print('wrote runs/sl_final/RESULTS.md')
"
fi

# 10. Upload to OSS
step "oss_upload" $PY scripts/oss_upload_sl_paths_4_2.py

echo ""
echo "[$(date +%H:%M:%S)] === overnight orchestrator DONE ==="
echo "  Path 4: $(ls runs/sl_path4/*/results.json 2>/dev/null | wc -l) runs"
echo "  Path 2: $(ls runs/sl_path2/*/results.json 2>/dev/null | wc -l) runs"
echo "  Path 3: $(ls runs/sl_path3/*/results.json 2>/dev/null | wc -l) runs"
echo "  RESULTS files:"
for f in runs/sl_path4/RESULTS.md runs/sl_path2/RESULTS.md runs/sl_path3/RESULTS.md runs/sl_final/RESULTS.md; do
    [ -f "$f" ] && echo "    $f"
done
