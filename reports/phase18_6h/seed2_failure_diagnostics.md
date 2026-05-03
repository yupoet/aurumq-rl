# Phase 18 — seed=2 failure diagnostics

OOS window: 2025-07-01 → 2026-04-24 (199 dates, fp=10). Universe: 3014 stocks, factors: 353.

## 1. Training trajectory summary (last logged iter)

| run | n_rows | final timestep | approx_kl | value_loss | explained_var | clip_frac |
|---|---:|---:|---:|---:|---:|---:|
| 16a (seed=42) | 72 | 311296 | 0.01829 | 0.02925 | 0.9628 | 0.184 |
| 17C (seed=2 FAIL) | 72 | 311296 | 0.01845 | 0.01997 | 0.9748 | 0.181 |
| 17D (seed=3) | 72 | 311296 | 0.02001 | 0.02876 | 0.9636 | 0.208 |

## 2. Per-date score behavior

| run | mean daily std | median daily std | saturation days frac |
|---|---:|---:|---:|
| 16a (seed=42) | 0.00082 | 0.00082 | 0.000 |
| 17C (seed=2 FAIL) | 0.00115 | 0.00114 | 0.000 |
| 17D (seed=3) | 0.00550 | 0.00549 | 0.000 |

## 3. Daily top-30 pick overlap (Jaccard)

| pair | n_days | mean | min | max |
|---|---:|---:|---:|---:|
| 17C_vs_16a | 199 | 0.007 | 0.000 | 0.053 |
| 17C_vs_17D | 199 | 0.003 | 0.000 | 0.053 |
| 17D_vs_16a | 199 | 0.010 | 0.000 | 0.053 |

## 4. Per-month corrected metrics (adjusted Sharpe / mean top-K return / IC)

| month | 16a (seed=42) adj_S | 17C (seed=2 FAIL) adj_S | 17D (seed=3) adj_S | 16a (seed=42) mean_top_k | 17C (seed=2 FAIL) mean_top_k | 17D (seed=3) mean_top_k | 16a (seed=42) IC | 17C (seed=2 FAIL) IC | 17D (seed=3) IC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2025-07 | +11.428 | +6.331 | +7.835 | +3.09% | +1.66% | +2.45% | +0.0628 | +0.0118 | +0.0000 |
| 2025-08 | +3.032 | +2.883 | +2.435 | +1.53% | +1.19% | +1.10% | -0.0110 | -0.0139 | -0.0151 |
| 2025-09 | +1.124 | -0.820 | +2.725 | +0.55% | -0.37% | +1.20% | +0.0037 | -0.0150 | +0.0428 |
| 2025-10 | +4.115 | +7.958 | +6.561 | +1.50% | +2.23% | +1.87% | +0.0171 | +0.0425 | +0.0478 |
| 2025-11 | -2.229 | -3.602 | +0.233 | -1.06% | -1.36% | +0.15% | +0.0082 | +0.0062 | +0.0164 |
| 2025-12 | +4.798 | +2.496 | +3.664 | +2.25% | +1.12% | +2.32% | -0.0088 | -0.0627 | -0.0166 |
| 2026-01 | +3.615 | +3.594 | +1.144 | +1.76% | +1.26% | +0.45% | +0.0640 | +0.0506 | +0.0039 |
| 2026-02 | -1.325 | +1.096 | -0.702 | -0.58% | +0.45% | -0.38% | -0.0632 | -0.0351 | -0.0311 |
| 2026-03 | -2.620 | -2.003 | -2.715 | -1.98% | -1.27% | -2.33% | +0.0583 | +0.0474 | +0.0328 |
| 2026-04 | +5.132 | +1.336 | +8.039 | +2.98% | +0.47% | +4.09% | -0.0742 | -0.1547 | -0.0151 |

## 5. Classification of seed=2 failure

### Mechanical heuristic output

- inconclusive (no single signal dominant)

### Manual reading of the same evidence

The mechanical heuristic was conservative because:

1. **Training-loop level is healthy**: 17C's final approx_kl, value_loss, explained_var, and clip_fraction are all in the same range as 16a / 17D. Not an optimization failure.
2. **Score scale slightly compressed**: 17C daily std 0.00115 < 17D's 0.00550 but > 16a's 0.00082. 16a (the BEST single model) has the LOWEST std, so score scale is NOT the determining factor. Saturation = 0% on all three.
3. **Pick overlap is tiny across ALL pairs** (mean Jaccard 0.003–0.010, max 0.053). This is the single biggest finding, but it is NOT specific to 17C — every pair of seeds picks essentially disjoint top-30 sets. That's exactly why the ensemble wins so much: different seeds find different stocks; the ensemble votes among them.
4. **Per-month**: 17C is competitive in 7/10 months. Concentrated weakness in 3:
   - 2025-09 adj_S = -0.820 (vs 16a +1.124)
   - 2025-11 adj_S = -3.602 (vs 16a -2.229; both bad, 17C worse)
   - 2026-04 adj_S = +1.336 (vs 16a +5.132)
   The aggregate vs_p50_adj = -0.060 is the accumulation of those three; no single catastrophic month.

### Best-fit classification

**OOS date / month concentration problem** (one of §3.C's listed options).

Justification: seed=2 doesn't fail at training, doesn't saturate, doesn't pick a degenerate set. It just allocates worse than peers in 3 specific months out of 10. With a different OOS window the result could invert. With 8 trained seeds, getting 1–2 with this kind of bad-month luck is consistent with the overall 6/8 = 75% win rate against random.

This points to **ensembling** as the right mitigation rather than "fix seed=2's training". With 6 eligible ensemble members the bad-month draws of any single member get diluted — and the rank-mean ensemble's +0.711 vs_p50_adj (vs 16a's +0.428) is direct evidence of that.