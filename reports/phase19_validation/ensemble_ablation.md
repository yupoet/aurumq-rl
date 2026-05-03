# Phase 19 Stage D-1 — ensemble ablation

OOS panel: 2025-07-01 → 2026-04-24 (199 dates, fp=10). Aggregation: per-date percentile rank-mean.

| variant | members | adj S | vs p50 adj | non-overlap | IC | Δ vs base |
|---|---|---:|---:|---:|---:|---:|
| rankmean5_minus_16a | 17b+17d+p18s4+p18s5+p18s6 | +1.868 | +0.726 | +2.134 | +0.0255 | +0.045 |
| rankmean6 (baseline) | 16a+17b+17d+p18s4+p18s5+p18s6 | +1.823 | +0.681 | +1.884 | +0.0278 | +0.000 |
| rankmean5_minus_17b | 16a+17d+p18s4+p18s5+p18s6 | +1.781 | +0.639 | +2.113 | +0.0277 | -0.042 |
| rankmean3_top3_strong | p18s4+p18s5+p18s6 | +1.714 | +0.572 | +1.845 | +0.0261 | -0.109 |
| rankmean5_minus_17d | 16a+17b+p18s4+p18s5+p18s6 | +1.711 | +0.569 | +1.325 | +0.0276 | -0.112 |
| rankmean5_minus_p18s5 | 16a+17b+17d+p18s4+p18s6 | +1.682 | +0.540 | +1.420 | +0.0230 | -0.141 |
| rankmean_all_loaded | 16a+17b+17c+17d+p18s4+p18s5+p18s6 | +1.667 | +0.525 | +1.053 | +0.0245 | -0.156 |
| rankmean7_with_seed2_17c | 16a+17b+17d+p18s4+p18s5+p18s6+17c | +1.665 | +0.523 | +1.053 | +0.0245 | -0.158 |
| rankmean5_minus_p18s4 | 16a+17b+17d+p18s5+p18s6 | +1.654 | +0.512 | +1.593 | +0.0233 | -0.169 |
| rankmean5_minus_p18s6 | 16a+17b+17d+p18s4+p18s5 | +1.501 | +0.359 | +1.178 | +0.0273 | -0.322 |