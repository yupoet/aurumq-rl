# Phase 17 — 7h Unattended Sweep (Corrected Eval, Seed Robustness)

> 2026-05-03 overnight session. Goal: validate Phase 16's robust-anti-helpful
> candidates (cyq + inst) under retrain transfer; and seed-robustness Phase 16's
> drop_mkt baseline. Both done. Multiple Phase 17 findings overturn the naïve
> reading of importance signals.

## TL;DR

* **cyq+inst drop hypothesis FAILED.** 17A (drop mkt+cyq+inst, seed=42) lost
  Phase 16a by **−0.732 adjusted Sharpe** and dropped below random p50
  (vs_p50_adj = −0.304). The Phase 16 importance signal of "robust negative
  cyq/inst" did NOT transfer to retrain — confirmed once more that
  permutation importance is **conditional on the trained policy**, not a
  causal claim about the features.
* **Phase 16a winner is reproducible across seeds, with caveats.** 4-seed
  sweep of drop_mkt: 3 of 4 seeds beat random p50 adj. Mean +0.249 vs
  random p50 adj, median +0.314. **seed=42 is at the upper end of the
  band** (+0.428); seed=2 is the only failure (−0.060). The drop_mkt
  configuration is a real signal, but seed-sensitive.
* **gtja / alpha / mfp keep their Phase 16 verdicts.** No re-rerun
  contradicted that they are positive contributors in well-trained
  policies.
* **Phase 17 winner**: 17E (drop_mkt seed=42 450k) at step 224928,
  bit-equivalent to Phase 16a. **No new champion**. The 450k extension
  past 300k oscillates between +0.9 and +1.49 adj — never returns to
  the +1.593 peak. Confirms 224928 is the global ceiling for this
  regime/seed.
* **Production recommendation**: KEEP Phase 16a as production baseline. Do
  NOT promote any Phase 17 model to default deployment. The Phase 17
  evidence supports the Phase 16a numbers but does not exceed them.

## 1. Reuse-of-Phase-16-baseline reminder

| metric | Phase 16a (drop mkt, seed=42, 300k) |
|---|---:|
| best step | 224928 |
| adj Sharpe | **+1.593** |
| vs random p50 adj | **+0.428** |
| non-overlap Sharpe | +1.112 |
| IC | +0.0143 |
| model | `models/production/phase16_16a_drop_mkt_best.zip` |

This is the bar Phase 17 was supposed to clear. None of the Phase 17 models
do. The closest match is 17D (seed=3) at +1.514 / +0.348.

## 2. drop mkt+cyq+inst FAILED to retrain-transfer (17A)

| metric | 17A (drop mkt+cyq+inst, seed=42) | Phase 16a | Δ |
|---|---:|---:|---:|
| best step | final (300k-ish) | 224928 | |
| adj Sharpe | +0.861 | +1.593 | **−0.732** |
| vs random p50 adj | **−0.304** | +0.428 | **−0.732** |
| non-overlap | +0.876 | +1.112 | −0.236 |
| IC | +0.0038 | +0.0143 | −0.011 |
| ckpts above adj +1.4 | 0 | 4 | |

**All 13 checkpoints have IC ≤ +0.005 and adj Sharpe ≤ +0.86**, with most
of the trajectory below random p50 adj. The "best" checkpoint is the
final.zip (the worst-quality candidate by Phase 16's pattern).

### Why the Phase 16 importance signal did NOT transfer

The Phase 16a importance gave **cyq sharpe_drop = −0.142 ± 0.044** and
**inst = −0.115 ± 0.030** — both rock-solid robustly anti-helpful by the
std/|mean|<0.5 bar. By the same evidence pattern that lifted drop_mkt
above the original Phase 14 baseline, dropping cyq+inst should have
lifted further. It did the opposite.

Two possible explanations, neither definitive:

1. **cyq + inst together are only 6 columns.** Removing them shifts the
   model's effective input to 347 columns (vs 353 in 16a). Such a small
   change should not destroy the policy. But the 17A trajectory shows
   the policy **never finds the same OOS-good basin** that 16a did —
   final IC barely positive, max adj Sharpe at the final checkpoint.
   This is consistent with the dropped features carrying small but
   systemically necessary information that the encoder builds on.
2. **Phase 16a importance is conditional on a working policy.** When the
   16a checkpoint scored well at OOS, permuting cyq/inst hurt the score
   pattern, so the drop looked anti-helpful. But that's only "if you
   had a 16a-class policy and removed the noise, you'd score higher" —
   it does NOT say "if you train without those features from scratch,
   you'll find a 16a-class policy". 17A is direct evidence that the
   second statement is false.

This is the same conditional-importance trap that flipped Phase 15's
"gtja anti-helpful" to Phase 16's "gtja load-bearing": importance signs
cannot be trusted to transfer to retrain. Only retrain itself confirms.

## 3. seed robustness on drop_mkt (Phase 16a config)

4-seed sweep of `drop mkt only, 300k` runs:

| seed | run | best step | adj S | vs p50 adj | IC | non-overlap | rank |
|---:|---|---:|---:|---:|---:|---:|---:|
| **42** | Phase 16a | 224928 | **+1.593** | **+0.428** | +0.0143 | +1.112 | 1 |
| 3 | 17D | 24992 | +1.514 | +0.348 | +0.0087 | +1.690 | 2 |
| 1 | 17B | 174944 | +1.446 | +0.280 | +0.0049 | +0.759 | 3 |
| 2 | 17C | 224928 | +1.105 | −0.060 | −0.0025 | +1.754 | 4 |

Statistics across the 4 seeds:

* **adj Sharpe**: mean +1.415, median +1.480, range [+1.105, +1.593], spread 0.488
* **vs random p50 adj**: mean **+0.249**, median +0.314, range [−0.060, +0.428]
* **3 of 4 seeds beat random p50 adj.** seed=2 is the lone failure.
* IC: 3 of 4 positive; seed=2 slightly negative.

The drop_mkt regime is REAL signal, not seed-specific lucky strike. But:

* **seed=42 is at the upper end** of its seed-noise band; expect seed
  sweeps to give a median ~ +0.31 vs p50 adj, not the +0.428 that 16a
  reports.
* **seed=2's failure is concerning.** ~25% chance of NOT beating random
  is high enough that production deployment of a single-seed model is
  risky. A 3-seed-median ensemble would be safer; that's a Phase 18
  recommendation.

Spike check: 17D's best is at **24992 (the very first checkpoint)** with
neighbouring ckpts also strong (50k +1.50, 75k +1.37) — so it's a real
plateau, not a single-point spike. Final.zip is +1.478 (also high). 17D
is a genuinely healthy run.

## 4. cyq / inst / fund verdict

| group | Phase 16a importance | Phase 17 retrain test | Verdict |
|---|---|---|---|
| cyq | −0.142 ± 0.044 (robust neg) | drop hurt OOS by −0.73 | **DO NOT drop**. Importance was conditional. |
| inst | −0.115 ± 0.030 (robust neg) | drop hurt OOS by −0.73 (joint) | **DO NOT drop**. |
| fund | −0.116 ± 0.053 (borderline) | not tested directly (17A failure removed motivation for C2) | Unknown; default keep. |

**No Phase 17 evidence supports dropping any factor group beyond mkt.**

## 5. gtja / alpha / mfp verdict (Phase 16 conclusions stand)

Phase 16 found gtja sharpe_drop +0.160 (load-bearing), alpha +0.071 (weak
pos), mfp +0.047 (weak pos). Phase 17 did NOT retrain anything that
dropped these, so the Phase 16 verdicts stand. **Keep all three.**

A subtle Phase 17A signal: when the policy was poorly trained (17A
failure), gtja's permutation importance flipped to **−1.382** —
massively anti-helpful. This is NOT a contradiction with Phase 16; it's
the conditional-importance phenomenon: a bad policy that nonetheless
relies heavily on gtja will show gtja-anti-helpful when gtja is permuted
because the permutation just adds noise to a wrong scoring signal.
**Treating that −1.382 as a real signal would be a mistake.** It's
artifact of the failed run.

## 6. Production recommendation

**Keep Phase 16a as production baseline.** No Phase 17 model exceeded its
adj Sharpe / vs random p50 adj numbers.

| Phase 16/17 model | adj S | vs p50 adj | recommendation |
|---|---:|---:|---|
| `phase16_16a_drop_mkt_best.zip` | +1.593 | +0.428 | KEEP as production. |
| `phase17_17d_drop_mkt_seed3_best.zip` | +1.514 | +0.348 | secondary; ensemble candidate. |
| `phase17_17b_drop_mkt_seed1_best.zip` | +1.446 | +0.280 | tertiary; ensemble candidate. |
| `phase17_17c_drop_mkt_seed2_best.zip` | +1.105 | −0.060 | DO NOT deploy. |
| `phase17_17a_drop_mkt_cyq_inst_seed42_best.zip` | +0.861 | −0.304 | DO NOT deploy; forensic only. |

If a 3-seed median ensemble is desired, candidates are 17B, 17D, and 16a
(all under the same drop_mkt regime, different seeds). Average their
predicted scores per stock per date before applying top-K. This was not
implemented in Phase 17 — it's a Phase 18 task.

## 7. Phase 18 recommendations

1. **3-seed median ensemble of drop_mkt models.** Cheapest production
   improvement available. Combines 16a + 17B + 17D's scores.
2. **More seeds (5-10) on drop_mkt** to put proper confidence intervals
   on the +0.249 mean lift vs p50 adj. The current 4-seed sample with
   one outlier (17C) suggests we should not estimate the mean from
   small samples.
3. **Per-feature drilldown of gtja / alpha / mfp / cyq / inst.** The
   per-prefix importance is too coarse — gtja's +0.16 load-bearing
   signal is averaged over 191 cols. Phase 17 has the IG saliency
   already from `eval_factor_importance.py`'s output; the Phase 17
   orchestrator wrote a per-feature ranking under
   `runs/phase17_7h/per_feature_drilldown__*.{md,json}`.
4. **Stop trying to drop more factor groups based on importance alone.**
   Phase 17 confirms the conditional-importance trap (17A's failure
   despite robust importance signals; 17A's gtja flip from +0.16 to
   −1.38). Future drops must be retrain-validated before ANY claim.
5. **Investigate why seed=2 failed.** Look at training_metrics for 17C
   vs 17D to see if there's a divergence point.

## 8. Final Phase 17 ranking

Orchestrator finished in 6:21:46 wall-clock (well within 7h budget).
Ranking by `vs random p50 adj` (the primary metric):

| # | run | adj S | vs p50 adj | non-overlap | IC | comment |
|---:|---|---:|---:|---:|---:|---|
| 1 | `phase17_17e_drop_mkt_seed42_450k` | +1.593 | **+0.428** | +1.112 | +0.0143 | bit-eq to 16a at 224928; 450k extension finds NO new peak |
| 2 | `phase17_17d_drop_mkt_seed3` | +1.514 | +0.348 | +1.690 | +0.0087 | early peak (24992); seed=3 healthy run |
| 3 | `phase17_17b_drop_mkt_seed1` | +1.446 | +0.280 | +0.759 | +0.0049 | seed=1; peak 174944 |
| 4 | `phase17_17c_drop_mkt_seed2` | +1.105 | −0.060 | +1.754 | −0.0025 | seed=2 FAILS to beat random; lone outlier |
| 5 | `phase17_17a_drop_mkt_cyq_inst_seed42` | +0.861 | −0.304 | +0.876 | +0.0038 | drop cyq+inst — failure case |

Phase 16 baseline reference: adj_S=+1.593, vs_p50_adj=+0.428, non_overlap=+1.112, IC=+0.0143.

### 17E extended trajectory (300k → 450k)

```
224928: adj +1.593  ← global peak (= 16a)
249920: adj +1.473
274912: adj +1.257
299904: adj +0.897
324896: adj +1.195
349888: adj +1.474
374880: adj +1.485   ← secondary plateau (+1.47-1.49 around 350-375k)
399872: adj +1.039
424864: adj +1.044
449856: adj +0.894
final:  adj +0.967
```

The 350k-375k secondary plateau is roughly seed=3-class (+1.47-1.49 vs
17D's +1.51-1.50) but never returns to +1.59. Conclusion: training
beyond 300k on the seed=42 trajectory provides no improvement.

## 9. Phase 17 winner importance (n_seeds=10)

The orchestrator re-ran factor importance on the 17E winner. Because
17E's 224928 ckpt is bit-identical to Phase 16a's 224928 (same seed,
same data, same fully-fixed pipeline), the importance values are
**bit-equivalent to Phase 16a's**. This serves as a reproducibility
check; both runs agree.

| group | n | ic_drop | sharpe_drop | ± std | std/abs(mean) | call |
|---|---:|---:|---:|---:|---:|---|
| cyq | 3 | -0.0001 | -0.142 | 0.044 | 0.31 | flagged anti-helpful — but **17A retrain test PROVED the signal does NOT transfer** |
| inst | 3 | -0.0005 | -0.115 | 0.030 | 0.26 | same — flagged but DOES NOT transfer |
| fund | 4 | -0.0002 | -0.116 | 0.053 | 0.46 | borderline; not retrained-tested in Phase 17 |
| ind | 2 | +0.0003 | -0.098 | 0.055 | 0.56 | weak negative; NOT robust |
| hm | 6 | +0.0005 | -0.087 | 0.054 | 0.62 | weak negative; NOT robust |
| mf | 14 | -0.0023 | -0.108 | 0.116 | 1.07 | NOISY — do not act |
| mg | 3 | +0.0006 | -0.061 | 0.058 | 0.95 | NOISY |
| senti | 3 | +0.0014 | -0.045 | 0.039 | 0.87 | NOISY |
| sh | 2 | +0.0002 | -0.010 | 0.017 | 1.70 | near zero |
| hk | 5 | +0.0002 | -0.002 | 0.031 | 15.5 | near zero |
| mfp | 12 | -0.0003 | +0.047 | 0.067 | 1.42 | weak positive (noisy) — KEEP |
| alpha | 105 | +0.0042 | +0.071 | 0.121 | 1.71 | weak positive (noisy) — KEEP |
| **gtja** | 191 | +0.0078 | +0.160 | 0.126 | 0.79 | **load-bearing** — KEEP |

**Key Phase 17 finding**: the conditional-importance trap is now
proven empirically. cyq + inst were the BEST candidates by Phase 16
importance (lowest std/|mean|, sharpe_drops well below the noise
floor) — and dropping them in 17A still made OOS WORSE. **Stop
chasing factor drops based on permutation importance alone.**

## 10. Per-feature IG saliency drilldown (Phase 17 winner)

Orchestrator skipped the drilldown step (remaining=38min < 45min
threshold), but the data was free to compute from existing IG output.
Manually invoked post-orchestrator. Output:

* `runs/phase17_7h/per_feature_drilldown__phase17_17e_drop_mkt_seed42_450k.md`
* `runs/phase17_7h/per_feature_drilldown__phase17_17e_drop_mkt_seed42_450k.json`

### Top per-feature contributors by IG saliency

**gtja_* top 5** (group total +0.160 sharpe_drop is concentrated here):
gtja_119, gtja_087, gtja_063, gtja_002, gtja_172.

**alpha_* top 5**: alpha_033, alpha_071, alpha_066, alpha_017, alpha_037.
Plus two custom alphas (alpha_custom_decaylinear_mom,
alpha_custom_argmax_recent) in the top 10.

**mfp_* top 3**: mfp_lg_buy_ratio_20d (≈ 2× the 2nd entry),
mfp_elg_buy_ratio_20d, mfp_main_net_cum_pct. The 12-feature mfp group's
small +0.047 sharpe_drop is dominated by these 3 columns; the other 9
are near-zero by saliency.

**cyq_* top 3** and **inst_* top 3** are documented in the JSON; they're
the candidates whose retrain transfer was PROVEN to fail in 17A, so the
per-feature ranking is informational only.

Phase 18 should consider:

* keeping the top-30 to top-50 features by IG saliency from {gtja, alpha,
  mfp} as a sanity-pruning test. Retrain with feature *whitelist*
  rather than feature drop. This addresses the conditional-importance
  trap directly: the question is whether the model can find the same
  policy with just the high-saliency features.
* not basing any further drops on importance signals alone.

## 9. What changed in code

NO production code was changed during Phase 17. The orchestrator
(`runs/phase17_7h/orchestrator.py`) is research code; no source under
`src/` or `scripts/` was modified during the 7h run. The only commits
expected from Phase 17 are research artifacts:

* `runs/phase17_7h/orchestrator.py`
* `runs/phase17_7h/decision_log.md`
* `handoffs/2026-05-03-phase17-7h/HANDOFF_2026-05-03_phase17.md` (this file)

Production models added (CANDIDATES, never auto-deployed):
`models/production/phase17_*.zip` for each Phase 17 run's best ckpt.

## 10. Artifacts

```
runs/phase17_7h/
  decision_log.md              (event log)
  orchestrator.py              (the runner)
  orchestrator_stdout.log
  phase17_17a_*.train.log / .eval.log
  phase17_17b_*.train.log / .eval.log
  phase17_17c_*.train.log / .eval.log
  phase17_17d_*.train.log / .eval.log
  phase17_17e_*.train.log / .eval.log
  phase17_*__importance__*/    (factor_importance staging dirs)
  per_feature_drilldown__*.{md,json}    (Phase 17 winner)

runs/phase17_17a_drop_mkt_cyq_inst_seed42/
  ppo_final.zip
  checkpoints/ppo_*_steps.zip
  oos_sweep.{md,json}
  factor_importance.json (in import dir)
  metadata.json
  training_summary.json

(same layout for 17b/17c/17d/17e)

models/production/
  phase17_17a_drop_mkt_cyq_inst_seed42_best.zip          (forensic)
  phase17_17b_drop_mkt_seed1_best.zip                    (ensemble candidate)
  phase17_17c_drop_mkt_seed2_best.zip                    (failure case, forensic)
  phase17_17d_drop_mkt_seed3_best.zip                    (ensemble candidate)
  phase17_17e_drop_mkt_seed42_450k_best.zip              (TBD)
```
