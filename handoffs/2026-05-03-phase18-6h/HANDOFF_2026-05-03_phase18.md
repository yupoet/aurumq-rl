# Phase 18 — 6h Unattended Ensemble + Seed Sweep

> 2026-05-03. Goal: harden the drop_mkt regime via multi-seed ensemble
> and diagnose the Phase 17C seed=2 failure. **No factor drops beyond
> mkt_**. Phase 16a stays as production. Phase 18 strong candidate is
> the rank-mean ensemble of 6 eligible drop_mkt seeds.

## TL;DR

* **Phase 18 strong candidate: `ens_final_rankmean`** — rank-mean of 6
  eligible drop_mkt seeds (16a, 17B, 17D, 18A, 18B, 18C).
  - **vs_random_p50_adjusted = +0.711** (Phase 16a baseline +0.428, **Δ = +0.283**)
  - **IC = +0.0278** (1.94× the 16a baseline)
  - non-overlap Sharpe = +1.938 (1.74× the 16a baseline)
  - All three gates passed: ≥+0.10 vs_p50_adj, ≥+0.010 IC, healthy non-overlap.
* **Single-model winner: phase18_18a_drop_mkt_seed4_best.zip** — vs_p50_adj
  +0.752, **+0.324** above 16a. Per Phase 18 §4, single-seed wins are NOT
  promoted alone; included as the strongest ensemble member instead.
* **Seed robustness on drop_mkt is REAL**: 6/8 seeds beat random p50, mean
  +0.352, median +0.388. Two failures (seed=2, seed=7) both excluded by
  the eligibility gate.
* **Recommendation: KEEP Phase 16a as production**; advance the 6-member
  rank-mean ensemble as a **Phase 18 candidate** to be validated against
  a fresh post-2026-04 holdout + realistic execution constraints.

## 1. Pre-existing baseline (the bar)

| metric | Phase 16a (drop mkt, seed=42, 300k, step 224928) |
|---|---:|
| adj Sharpe | +1.593 |
| vs random p50 adj | **+0.428** |
| non-overlap Sharpe | +1.112 |
| IC | +0.0143 |

## 2. Run list (exact configs)

All Phase 17/18 runs use identical training config (see `runs/phase18_6h/orchestrator.py:common_train_flags`); the only change is `--seed`.

```
--n-envs 16 --episode-length 240
--batch-size 1024 --n-steps 1024 --n-epochs 10
--learning-rate 1e-4 --target-kl 0.30 --max-grad-norm 0.5
--rollout-buffer index --tf32 --matmul-precision high
--unique-date-encoding --checkpoint-freq 25000
--forward-period 10 --top-k 30
--total-timesteps 300000
--drop-factor-prefix mkt_
```

| run | seed | model |
|---|---:|---|
| Phase16a | 42 | `models/production/phase16_16a_drop_mkt_best.zip` |
| Phase17B | 1 | `models/production/phase17_17b_drop_mkt_seed1_best.zip` |
| Phase17C | 2 | `models/production/phase17_17c_drop_mkt_seed2_best.zip` ← failed seed |
| Phase17D | 3 | `models/production/phase17_17d_drop_mkt_seed3_best.zip` |
| **Phase18A** | **4** | `models/phase18/phase18_18a_drop_mkt_seed4_best.zip` ← BIG WIN |
| **Phase18B** | **5** | `models/phase18/phase18_18b_drop_mkt_seed5_best.zip` ← BIG WIN |
| Phase18C | 6 | `models/phase18/phase18_18c_drop_mkt_seed6_best.zip` |
| Phase18D | 7 | `models/phase18/phase18_18d_drop_mkt_seed7_best.zip` ← failed seed |

## 3. Corrected eval table (every model + every ensemble)

OOS window: 2025-07-01 → 2026-04-24 (199 dates, fp=10, top-K=30). Primary metric: `vs_random_p50_adjusted`. See `reports/phase18_6h/final_ranking.md` for the full table.

### Singles (8 seeds)

| seed | run | best step | adj S | vs p50 adj | non-overlap | IC | gate |
|---:|---|---:|---:|---:|---:|---:|---|
| **4** | 18A | 124960 | **+1.917** | **+0.752** | +1.497 | +0.0169 | BIG WIN |
| **5** | 18B | 49984 | **+1.761** | **+0.596** | +1.368 | +0.0175 | BIG WIN |
| 6 | 18C | 299904 | +1.634 | +0.469 | +2.298 | +0.0111 | beat 16a |
| 42 | 16a | 224928 | +1.593 | +0.428 | +1.112 | +0.0143 | baseline |
| 3 | 17D | 24992 | +1.514 | +0.348 | +1.690 | +0.0087 | beat random |
| 1 | 17B | 174944 | +1.446 | +0.280 | +0.759 | +0.0049 | beat random |
| 7 | 18D | 174944 | +1.169 | +0.004 | +1.250 | -0.0002 | FAIL |
| 2 | 17C | 224928 | +1.105 | -0.060 | +1.754 | -0.0025 | FAIL |

### Ensembles

| variant | members | adj S | vs p50 adj | non-overlap | IC | Δ vs 16a |
|---|---|---:|---:|---:|---:|---:|
| **ens_final_rankmean** | 6 (16a+17b+17d+18A+18B+18C) | +1.877 | **+0.711** | +1.938 | +0.0278 | **+0.283** |
| ens_final_zmedian | 6 | +1.788 | +0.623 | +1.625 | +0.0244 | +0.195 |
| ens_final_zmean | 6 | +1.726 | +0.561 | +1.531 | +0.0278 | +0.133 |
| ens4_with_seed2_rankmean (Stage A) | 16a+17b+17d+17c | +1.687 | +0.521 | +1.697 | +0.0117 | +0.093 |
| ens2_16a_17d_zmean (Stage A) | 16a+17d | +1.635 | +0.470 | +1.102 | +0.0154 | +0.042 |
| ens3_zmean (Stage A) | 16a+17b+17d | +1.588 | +0.422 | +1.676 | +0.0160 | -0.006 |

## 4. Seed distribution stats (8 seeds)

| stat | value |
|---|---:|
| mean vs_p50_adj | +0.352 |
| median vs_p50_adj | +0.388 |
| std vs_p50_adj | 0.288 |
| min vs_p50_adj | -0.060 |
| max vs_p50_adj | +0.752 |
| **win rate vs random p50** | **6/8 = 75%** |
| **IC positive rate** | **6/8 = 75%** |

The mean (+0.352) is below the median (+0.388) because of the two failing seeds (2 and 7). With those excluded, the 6-eligible-seed mean is **+0.479**, slightly above the Phase 16a single-seed result. Confirms that: a 6-seed median + ensembling protects against the ~25% per-seed failure rate.

## 5. Best deployable ensemble config

```text
Members (6 eligible drop_mkt seeds):
  16a   (seed=42) D:\dev\aurumq-rl\models\production\phase16_16a_drop_mkt_best.zip
  17b   (seed=1)  D:\dev\aurumq-rl\models\production\phase17_17b_drop_mkt_seed1_best.zip
  17d   (seed=3)  D:\dev\aurumq-rl\models\production\phase17_17d_drop_mkt_seed3_best.zip
  18A   (seed=4)  D:\dev\aurumq-rl\models\phase18\phase18_18a_drop_mkt_seed4_best.zip
  18B   (seed=5)  D:\dev\aurumq-rl\models\phase18\phase18_18b_drop_mkt_seed5_best.zip
  18C   (seed=6)  D:\dev\aurumq-rl\models\phase18\phase18_18c_drop_mkt_seed6_best.zip

Aggregation: per-date percentile-rank mean
  rank[m,t,:] = (argsort(argsort(scores[m,t,:])) + 0.5) / n_finite
  ensemble[t,:] = nanmean over m of rank[m,t,:]
  top-K from ensemble[t,:].argsort()[-K:]

Score normalization rationale:
  rank-mean is robust to scale differences across model heads (one model
  may produce raw scores at 1e-3 scale, another at 1e-1 — z-score handles
  that, but rank handles it AND outliers, AND degenerate-near-zero std
  days. With 6 members of which one is +0.752 and three are +0.28-0.43,
  rank-mean weights the strong member appropriately without letting its
  raw scale dominate.)

Excluded from deployable ensemble:
  17c (seed=2): vs_p50_adj=-0.060, IC=-0.0025 — fails gate
  18D (seed=7): vs_p50_adj=+0.004, IC=-0.0002 — fails gate

Excluded singletons-with-low-non-overlap (gate non_overlap > +0.70):
  17b's non-overlap is +0.759, just above the gate; included.
  No other seed fails on non-overlap.

Metrics on this ensemble (199-day OOS):
  adj S (sqrt(252/fp)):       +1.877
  vs random p50 adj:          +0.711  (target was +0.10, achieved +0.283)
  non-overlap Sharpe:         +1.938
  IC:                         +0.0278
  legacy Sharpe (sqrt(252)):  +5.937 (informational, do NOT decide on this)
```

## 6. seed=2 failure classification

**Best fit category (per §3.C): OOS date / month concentration problem.**

Mechanical heuristics flagged "inconclusive" — none of training-loop / score-saturation / single bad signal was dominant. Manual reading of the same data:

* training-loop healthy (final approx_kl 0.018, value_loss 0.020, explained_var 0.97 — same range as 16a / 17D)
* score-scale slightly compressed (mean daily std 0.0012 vs 17D's 0.0055) but NOT saturated (0% saturation days). 16a's std is even lower (0.0008) and yet 16a is the BEST single model; so std alone is not the determining factor.
* **pick overlap is tiny among ALL pairs** — 17C/16a Jaccard mean 0.007, 17D/16a 0.010, 17C/17D 0.003. Different seeds find essentially disjoint top-30 sets. **This is the ensemble's lift mechanism**, not a 17C-specific failure.
* per-month: 17C is competitive in 7/10 months. Concentrated weakness in 2025-09 (-0.82), 2025-11 (-3.60), 2026-04 (+1.34 vs 16a's +5.13). Aggregate -0.060 vs_p50_adj is accumulation of those three.

**Implication for Phase 18 strategy**: ensembling is the right mitigation rather than "fix seed=2's training". The 6-member ensemble's +0.711 vs_p50_adj is direct evidence that diluting seed-specific bad-month draws works.

Full diagnostic table in `reports/phase18_6h/seed2_failure_diagnostics.md`.

## 7. Promotion language

| condition | rule | met? |
|---|---|---|
| no ensemble/seed beats Phase16a by +0.05 | KEEP 16a, report seed CI only | (no — many beat) |
| ensemble beats +0.05 but not +0.10 | mark Phase18 candidate, not production | (no — beats more) |
| **ensemble beats +0.10 AND passes IC + non-overlap + monthly checks** | **recommend ensemble runtime, but require holdout / live validation** | **✓** |
| single new seed beats +0.10 | do NOT promote alone; include in ensemble | ✓ (seed=4) |
| seed=2-like failures continue | exclude from deployable ensemble | ✓ (excluded seeds 2 + 7) |

The strong-candidate gate fires: `ens_final_rankmean` Δ = +0.283 ≫ +0.10, IC = +0.0278 ≫ +0.010, non-overlap = +1.938 healthy.

**But the instructions explicitly note (§0.6) that single-OOS-window optimum is not production Sharpe.** Production deployment of `ens_final_rankmean` requires:

1. holdout validation on dates after 2026-04-24
2. realistic transaction-cost model (current cost_bps=30 is a placeholder)
3. T+1 / 涨跌停 / ST execution constraint verification
4. live-style portfolio-sizing simulation

So the **explicit recommendation** is:

* **KEEP Phase 16a as the live production model.**
* **Track `ens_final_rankmean` as a parallel "Phase 18 candidate" track**, deployed only after a fresh post-2026-04 holdout passes the same gates.

## 8. What was skipped due to time

* **Stage D (checkpoint-diversity ensembles)**: not run. Hard-stop budget reached after Stage C diagnostics. Phase 18 already has strong evidence that single-seed-checkpoint diversity is huge (Jaccard 0.003-0.010); checkpoint-from-same-seed diversity is a smaller follow-up.
* **Per-month-stratified ensemble validation**: included monthly Sharpe in seed=2 diagnostics for the 3 subjects, but did not run a per-month statistical test against random across all 6 ensemble members.
* **Stage B refresh after seed=4**: subprocess invocation crashed on Windows-absolute-path parsing in `--member`. Fix landed before seed=5 finished; subsequent stages worked. The Stage A standalone CPU run also produced these numbers, so no data lost.

## 9. Phase 19 / next-session recommendations

1. **Holdout validation against post-2026-04-24 dates.** This is the single most important next step before any production move.
2. **Realistic transaction-cost modelling** with the ensemble — single-model cost_bps was 30 in training; multi-model rebalancing may have higher turnover.
3. **More seeds (target 10-12 total)** to reduce the standard error of the seed-distribution mean. Current 8 seeds give std=0.288; 12 would give std≈0.235.
4. **Investigate seed=4's +0.752 outlier**: per-month profile, top-K composition, etc. — is it a robust strong model or a lucky single OOS sample?
5. **Phase 18C investigation**: best ckpt at step 299904 (final) is unusual. Most other Phase 17/18 runs peak earlier. Worth checking if extending 18C to 450k changes anything (a "17E-style extension").
6. **Do not** chase further factor drops based on importance signals alone. Phase 17A already proved that signal does NOT transfer through retrain.

## 10. Artifacts

```
runs/phase18_6h/
  decision_log.md            (event log; richest)
  orchestrator.py            (the runner)
  orchestrator_stdout.log
  phase18_18a-d_*.train.log + .eval.log
  ensemble_eval_*.log

runs/phase18_18[a-d]_drop_mkt_seed[4-7]/
  ppo_final.zip
  checkpoints/ppo_*_steps.zip
  oos_sweep.{md,json}
  metadata.json (factor_names + forward_period)
  training_summary.json

reports/phase18_6h/
  ensemble_eval.{md,json}              (canonical = final)
  ensemble_eval_stage_a.{md,json}
  ensemble_eval_stage_b_after_seed5.{md,json}
  ensemble_eval_stage_b_after_seed6.{md,json}
  ensemble_eval_stage_b_after_seed7.{md,json}
  ensemble_eval_final.{md,json}
  final_ranking.md
  seed2_failure_diagnostics.md
  _scores_cache_*.npy   (per-member score matrices, regenerable)

models/phase18/
  phase18_18a_drop_mkt_seed4_best.zip      ← seed=4 (BIG WIN, single best)
  phase18_18a_drop_mkt_seed4_best_metadata.json
  phase18_18b_drop_mkt_seed5_best.zip      ← seed=5
  phase18_18b_drop_mkt_seed5_best_metadata.json
  phase18_18c_drop_mkt_seed6_best.zip      ← seed=6
  phase18_18c_drop_mkt_seed6_best_metadata.json
  phase18_18d_drop_mkt_seed7_best.zip      ← seed=7 (failure case, forensic only)
  phase18_18d_drop_mkt_seed7_best_metadata.json

handoffs/2026-05-03-phase18-6h/
  PHASE18_6H_UNATTENDED_INSTRUCTIONS.md   (input from previous session)
  HANDOFF_2026-05-03_phase18.md           (this file)
```
