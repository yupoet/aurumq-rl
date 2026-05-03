# Phase 18 — Final Ranking

OOS window: 2025-07-01 → 2026-04-24, 199 dates, fp=10, top-K=30. Primary metric: `vs_random_p50_adjusted`.

## Single models (all 8 drop_mkt seeds)

| seed | run | best step | adj S | vs p50 adj | non-overlap | IC | gate |
|---:|---|---:|---:|---:|---:|---:|---|
| **4** | 18A | 124960 | **+1.917** | **+0.752** | +1.497 | +0.0169 | BIG WIN (Δ=+0.324) |
| **5** | 18B | 49984 | **+1.761** | **+0.596** | +1.368 | +0.0175 | BIG WIN (Δ=+0.168) |
| 6 | 18C | 299904 | +1.634 | +0.469 | +2.298 | +0.0111 | beat 16a (Δ=+0.041) |
| 42 | 16a | 224928 | +1.593 | +0.428 | +1.112 | +0.0143 | baseline |
| 3 | 17D | 24992 | +1.514 | +0.348 | +1.690 | +0.0087 | beat random |
| 1 | 17B | 174944 | +1.446 | +0.280 | +0.759 | +0.0049 | beat random |
| 7 | 18D | 174944 | +1.169 | +0.004 | +1.250 | -0.0002 | FAIL (≈ random) |
| 2 | 17C | 224928 | +1.105 | -0.060 | +1.754 | -0.0025 | FAIL (loses) |

### 8-seed distribution

| stat | value |
|---|---:|
| mean vs_p50_adj | +0.352 |
| median vs_p50_adj | +0.388 |
| std vs_p50_adj | 0.288 |
| min vs_p50_adj | -0.060 |
| max vs_p50_adj | +0.752 |
| win rate (vs_p50_adj > 0) | 6/8 = 75% |
| IC positive rate | 6/8 = 75% |

## Ensembles

| variant | members | adj S | vs p50 adj | non-overlap | IC | Δ vs 16a |
|---|---|---:|---:|---:|---:|---:|
| **ens_final_rankmean** | 16a + 17d + 17b + p18s4 + p18s5 + p18s6 | +1.877 | **+0.711** | +1.938 | +0.0278 | **+0.283** |
| ens_final_zmedian | (same 6) | +1.788 | +0.623 | +1.625 | +0.0244 | +0.195 |
| ens_final_zmean | (same 6) | +1.726 | +0.561 | +1.531 | +0.0278 | +0.133 |
| ens4_with_seed2_rankmean (Stage A) | 16a + 17d + 17b + 17c | +1.687 | +0.521 | +1.697 | +0.0117 | +0.093 |
| ens2_16a_17d_zmean (Stage A) | 16a + 17d | +1.635 | +0.470 | +1.102 | +0.0154 | +0.042 |
| ens3_zmean (Stage A) | 16a + 17d + 17b | +1.588 | +0.422 | +1.676 | +0.0160 | -0.006 |
| ens3_rankmean (Stage A) | 16a + 17d + 17b | +1.574 | +0.408 | +1.482 | +0.0159 | -0.020 |
| ens3_zmedian (Stage A) | 16a + 17d + 17b | +1.416 | +0.250 | +0.941 | +0.0145 | -0.178 |

## Overall winner (single-checkpoint or ensemble)

**single seed=4 (phase18_18a_drop_mkt_seed4_best.zip, step 124960)** is the highest-Sharpe single model with vs_p50_adj = +0.752 and IC +0.0169. **+0.324 above 16a.**

**ens_final_rankmean** is the best ensemble (5/6 eligible members; rank-mean) with vs_p50_adj = +0.711 and **IC +0.0278 (1.94× the 16a baseline)** and non-overlap +1.938 (1.74× 16a's). **+0.283 above 16a.**

The ensemble's marginally lower vs_p50_adj than single seed=4 comes with these advantages:
- IC is **64% higher** (better cross-section ranking, not just lucky top-30 picks)
- non-overlap Sharpe is **30% higher** (less window-dependent)
- pick-set is the consensus across 6 different policies — much less seed-risk

## Promotion rules (per Phase 18 §4)

| result | recommended action |
|---|---|
| ensemble beats Phase 16a by +0.283 (>+0.10) AND IC=+0.028 (≫+0.010) AND non-overlap=+1.938 healthy | **mark `ens_final_rankmean` as Phase 18 strong candidate** |
| single seed=4 beats Phase 16a by +0.324 alone | do NOT promote single seed; include in ensemble |
| seed=2 + seed=7 fail (75% of 8 seeds win) | exclude failing seeds from deployable ensemble |

## Recommendation

**Phase 18 strong candidate: `ens_final_rankmean` (rank-mean of 16a + 17B + 17D + 18A + 18B + 18C).**

But: per Phase 18 instruction §0.6, "single OOS optimum cannot be treated as production Sharpe". The +0.711 vs_p50_adj is on a single 199-day OOS window. Production deployment still requires:
- holdout period validation (e.g., 2026-05 onward)
- realistic transaction cost model
- T+1 / 涨跌停 / ST execution constraints
- live-style portfolio sizing

So the ensemble is the **best Phase 18 deliverable**, but the recommendation is to:

1. KEEP Phase 16a as production for now (no overwrite).
2. Add `ens_final_rankmean` as a parallel "Phase 18 candidate" track to test against a fresh post-2026-04 holdout.
3. Phase 19 should focus on holdout validation and realistic execution constraints, not on more single-seed RL training.
