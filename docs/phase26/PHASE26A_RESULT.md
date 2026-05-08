# Phase 26A — real tech factors result

> 2026-05-07. Train: 300k PPO main_wave_target reward, seed 42, 373-col panel.
> OOS eval window: 2025-07-01 .. 2026-04-24 (same as 23A baseline).

## TL;DR — Phase 26A is a regression vs 23A

| | Phase 23A baseline | Phase 26A |
|---|---|---|
| Factors | 353 (10 of which are dead) | **373** (343 retained + 30 new tech_/cmf_/zt_) |
| Best T-1 lift | **2.38×** (step199936, top_k=5) | **1.36×** (step100k+150k+final, top_k=5) |
| Best T-1 hit | 2.11% | 1.21% |
| Best eval_score_v23 | 0.3690 | 0.3642 |
| Worst-checkpoint T-1 lift | 0.76× | **0.38×** (step50k @ top_k=3, BELOW random) |
| Median T-1 lift across 7 ckpts (top5) | ~1.5× | 1.25× |

**Verdict:** Adding 30 upstream-computed tech factors (tech_/cmf_/zt_) on top of
the 23A baseline drops T-1 hit lift by ~43%. The model gets *worse*, not better.

## Why — IG saliency + permutation importance

Permutation importance (negative `ic_drop_mean` = factor is *hurting* the model
by injecting noise the policy then routes attention to):

| group | n | ic_drop | sal_mean | verdict |
|---|---:|---:|---:|---|
| **alpha_** | 105 | **+0.0075** | 7.3e-10 | strong, load-bearing |
| **gtja_** | 191 | **+0.0039** | 5.7e-10 | strong, load-bearing |
| mfp_ | 4 | +0.0013 | 7.6e-10 | useful |
| inst_ | 3 | +0.0012 | 2.8e-10 | useful |
| mg_ | 3 | +0.0002 | 2.1e-10 | marginal+ |
| senti_ | 1 | +0.0001 | 4.9e-10 | marginal+ |
| ind_ | 2 | -0.0000 | 5.6e-10 | neutral |
| fund_ | 4 | -0.0001 | 4.3e-10 | neutral |
| **cmf_** (new) | 4 | **-0.0002** | 8.0e-10 | borderline harmful |
| sh_ | 2 | -0.0002 | 6.7e-11 | borderline harmful |
| mf_ | 14 | -0.0004 | 5.1e-10 | harmful |
| **zt_** (new) | 6 | **-0.0005** | 1.8e-10 | harmful |
| hk_ | 5 | -0.0008 | 1.4e-10 | harmful |
| cyq_ | 3 | -0.0015 | 6.7e-10 | harmful |
| **tech_** (new) | 20 | **-0.0022** | 7.2e-10 | actively harmful |
| **hm_** | 6 | **-0.0042** | 2.3e-10 | **most harmful** |

The 30 new factors net to **-0.0029** IC drop. Worse, several previously-useful
groups (mf_/hk_/sh_/cyq_/hm_) flipped to negative IC drop in 26A, suggesting
the added tech_* factors didn't just add noise — they *displaced* attention
from genuinely useful signals.

### tech_ subgroup detail

Only 2 of the 20 tech_/cmf_ factors crossed the |sal| ≥ 1e-9 threshold:

- `cmf_120d_pct_amt` (1.06e-9)
- `tech_boll_percent` (1.15e-9)

The other 28 are all WEAK. The full ranking is in
`runs/phase26a_episode_seed42/factor_importance.json::saliency_per_factor`.

## Hypothesis — why does adding "good factors" hurt?

The HANDOFF survival report shows tech_ factors are clean (PASS / minor WARN).
Cross-correlation report shows |ρ| ≤ 0.85 with base. Statistically they're
"healthy." But:

1. **Cross-section z-score amplifies noise in low-signal factors.** `tech_kdj_k_minus_d`
   has |abs_p99| = 20.3 — after z-score the tail flattens but the model sees
   meaningful variance day-to-day. With 20 such weak signals, the encoder
   spends capacity on them instead of refining the alpha/gtja attention.

2. **Tech factors are momentum-coincident, not predictive.** Phase 22 already
   noted that classical TA on close prices reflects "where the price is" not
   "where it's going." Episodes are detected at first up-day; tech factors at
   T-1 mostly reflect the *prior* downtrend, not the imminent breakout.

3. **The Per-Stock Encoder's MLP has a fixed 128→64→32 capacity.** Adding 30
   columns to a 343-col input with the same capacity dilutes per-factor
   attention. Phase 25A saw the same thing with 391 cols; Phase 26A confirms
   it at 373.

## What to try next

Options for Phase 26B (in priority order):

1. **26B-baseline-on-new-panel:** Run Phase 23A's 343-col config on the new
   combined panel (no tech_/cmf_/zt_, but use canonical cyq_ from cyq_panel
   instead of the old quotes_enriched cyq_*). Pure A/B test of "is the
   canonical cyq data better than the old."

2. **26B-prune-harmful:** Phase 26A factors minus the 30 tech_/cmf_/zt_ minus
   harmful legacy groups (hm_, hk_, mf_, cyq_). Net 343 - 6 - 5 - 14 - 3 = 315
   cols. Tests whether 23A's slightly-positive groups are actually load-
   bearing or also dilution.

3. **26B-only-strong:** alpha_+gtja_+mfp_+inst_+mg_+senti_+ind_+fund_ = 105+191+4+3+3+1+2+4 = 313 cols. Strict "only ic_drop ≥ 0" set.

4. **26B-bigger-encoder:** Same 373-col 26A but encoder_hidden=256,128 (4× capacity).
   Tests dilution-vs-capacity hypothesis.

**Recommendation: do 26B-baseline-on-new-panel first** — it isolates the
canonical-cyq A/B and rules out the panel-level data change as the regression
cause before any factor-level intervention.

## Production verdict

**Stick with Phase 23A as production.** Phase 26A's 373-col model is
materially worse on the same OOS window. Do NOT switch to 26A.

## Files

- `runs/phase26a_episode_seed42/ppo_final.zip` — final 300k checkpoint
- `runs/phase26a_episode_seed42/episode_eval.{json,md}` — full OOS T-1 hit table
- `runs/phase26a_episode_seed42/factor_importance.json` — IG + permutation
- `data/factor_panel_phase26a_2023_2026.parquet` — 4.26M × 382 combined panel (8.1 GB)
- `data/tech_panel_v1/` — original 17-file OSS bundle from upstream

## Reproduce

```bash
# Combined panel (one-shot, ~5 min)
.venv/Scripts/python.exe scripts/build_combined_panel_v26.py

# Train (300k, ~16 min on RTX 4070)
.venv/Scripts/python.exe scripts/train_v2.py \
    --total-timesteps 300000 \
    --data-path data/factor_panel_phase26a_2023_2026.parquet \
    --start-date 2023-01-03 --end-date 2024-12-31 \
    --universe-filter main_board_non_st \
    --include-columns-file data/tech_panel_v1/tech_panel_report/include_columns_v1.txt \
    --n-envs 16 --n-steps 128 \
    --forward-period 5 --top-k 5 \
    --encoder-hidden 128,64 --encoder-out-dim 32 \
    --learning-rate 1e-4 \
    --rollout-buffer index \
    --reward-mode main_wave_target \
    --tf32 --matmul-precision high \
    --seed 42 \
    --checkpoint-freq 50000 \
    --out-dir runs/phase26a_episode_seed42

# Eval (~30s)
.venv/Scripts/python.exe scripts/_eval_main_wave_episode.py \
    --run-dir runs/phase26a_episode_seed42 \
    --data-path data/factor_panel_phase26a_2023_2026.parquet \
    --val-start 2025-07-01 --val-end 2026-04-24 \
    --top-k 3 5 --universe-filter main_board_non_st

# IG + permutation (~20-30 min)
.venv/Scripts/python.exe scripts/eval_factor_importance.py \
    --run-dir runs/phase26a_episode_seed42 \
    --data-path data/factor_panel_phase26a_2023_2026.parquet \
    --val-start 2025-07-01 --val-end 2026-04-24 \
    --top-k 5 --forward-period 5 --n-seeds 3
```
