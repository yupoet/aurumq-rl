# Ensemble eval — final

- val: 2025-07-01 → 2026-04-24 (199 dates, fp=10)
- top-K = 30; ranked by `vs_random_p50_adjusted` (Phase 16 corrected metric).
- Phase 16a baseline: adj_S=+1.593, vs_p50_adj=+0.428, non_overlap=+1.112, IC=+0.0143.

| variant | aggregation | members | adj S | vs p50 adj | non-overlap | IC | Δ vs 16a |
|---|---|---|---:|---:|---:|---:|---:|
| single_p18s4 | passthrough | p18s4 | +1.917 | +0.752 | +1.497 | +0.0169 | +0.324 |
| ens_final_rankmean | rankmean | 16a+17d+17b+p18s4+p18s5+p18s6 | +1.877 | +0.711 | +1.938 | +0.0278 | +0.283 |
| ens_final_zmedian | zmedian | 16a+17d+17b+p18s4+p18s5+p18s6 | +1.788 | +0.623 | +1.625 | +0.0244 | +0.195 |
| single_p18s5 | passthrough | p18s5 | +1.761 | +0.596 | +1.368 | +0.0175 | +0.168 |
| ens_final_zmean | zmean | 16a+17d+17b+p18s4+p18s5+p18s6 | +1.726 | +0.561 | +1.531 | +0.0278 | +0.133 |
| single_p18s6 | passthrough | p18s6 | +1.634 | +0.469 | +2.298 | +0.0111 | +0.041 |
| single_16a | passthrough | 16a | +1.593 | +0.428 | +1.112 | +0.0143 | -0.000 |
| ens_final_top3_by_vsp50_zmean | zmean | 16a+17d+17b | +1.588 | +0.422 | +1.676 | +0.0160 | -0.006 |
| single_17d | passthrough | 17d | +1.514 | +0.348 | +1.690 | +0.0087 | -0.080 |
| single_17b | passthrough | 17b | +1.446 | +0.280 | +0.759 | +0.0049 | -0.148 |