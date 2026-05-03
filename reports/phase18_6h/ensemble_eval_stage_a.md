# Ensemble eval — stage_a

- val: 2025-07-01 → 2026-04-24 (199 dates, fp=10)
- top-K = 30; ranked by `vs_random_p50_adjusted` (Phase 16 corrected metric).
- Phase 16a baseline: adj_S=+1.593, vs_p50_adj=+0.428, non_overlap=+1.112, IC=+0.0143.

| variant | aggregation | members | adj S | vs p50 adj | non-overlap | IC | Δ vs 16a |
|---|---|---|---:|---:|---:|---:|---:|
| ens4_with_seed2_rankmean | rankmean | 16a+17d+17b+17c | +1.687 | +0.521 | +1.697 | +0.0117 | +0.093 |
| ens2_16a_17d_zmean | zmean | 16a+17d | +1.635 | +0.470 | +1.102 | +0.0154 | +0.042 |
| single_16a | passthrough | 16a | +1.593 | +0.428 | +1.112 | +0.0143 | -0.000 |
| ens3_zmean | zmean | 16a+17d+17b | +1.588 | +0.422 | +1.676 | +0.0160 | -0.006 |
| ens3_rankmean | rankmean | 16a+17d+17b | +1.574 | +0.408 | +1.482 | +0.0159 | -0.020 |
| single_17d | passthrough | 17d | +1.514 | +0.348 | +1.690 | +0.0087 | -0.080 |
| single_17b | passthrough | 17b | +1.446 | +0.280 | +0.759 | +0.0049 | -0.148 |
| ens3_zmedian | zmedian | 16a+17d+17b | +1.416 | +0.250 | +0.941 | +0.0145 | -0.178 |
| single_17c | passthrough | 17c | +1.105 | -0.060 | +1.754 | -0.0025 | -0.488 |