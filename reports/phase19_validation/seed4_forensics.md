# Phase 19 Stage D-2 — seed=4 forensics

Question: how much of the rankmean6 ensemble lift over rankmean5-without-seed4 is attributable to seed=4? Is seed=4 a single-month/single-industry concentration?

## Daily top-30 Jaccard

- seed=4 vs ens6 baseline: mean Jaccard = 0.023
- ens5 (without seed=4) vs ens6 baseline: mean Jaccard = 0.393
- (seed=4 alone shares only 2.3% of ens6's daily picks; ens5-without-4 shares 39.3%.)

## Per-month lift from including seed=4 (ens6 − ens5)

| month | n_days | seed=4 alone adj_S | ens6 adj_S | ens5 (no seed4) adj_S | lift from seed=4 |
|---|---:|---:|---:|---:|---:|
| 2025-07 | 23 | +12.052 | +9.758 | +8.355 | +1.404 |
| 2025-08 | 21 | +3.577 | +2.505 | +2.208 | +0.297 |
| 2025-09 | 22 | +4.239 | +2.547 | +1.662 | +0.885 |
| 2025-10 | 17 | +7.389 | +5.618 | +7.316 | -1.698 |
| 2025-11 | 20 | -0.689 | -0.698 | -1.043 | +0.345 |
| 2025-12 | 23 | +4.397 | +4.001 | +3.605 | +0.396 |
| 2026-01 | 20 | +2.329 | +5.219 | +3.566 | +1.653 |
| 2026-02 | 14 | +0.455 | -0.788 | -0.481 | -0.307 |
| 2026-03 | 22 | -2.462 | -1.676 | -1.954 | +0.279 |
| 2026-04 | 7 | +5.864 | +4.285 | +5.895 | -1.610 |

Total lift from seed=4 (sum over months): +1.643
Max single-month lift: +1.653
Max single-month share of total lift: 100.6%

## Industry concentration of seed=4 vs ens6 daily picks

Top 10 industries by pick count (validation window):

| seed=4 industry | seed=4 picks | seed=4 pct | | ens6 industry | ens6 picks | ens6 pct |
|---|---:|---:|---|---|---:|---:|
| 汽车配件 | 346 | 6.10% | | 汽车配件 | 416 | 7.34% |
| 电气设备 | 314 | 5.54% | | 电气设备 | 319 | 5.63% |
| 化工原料 | 305 | 5.38% | | 化工原料 | 315 | 5.56% |
| 元器件 | 202 | 3.56% | | 专用机械 | 202 | 3.56% |
| 专用机械 | 178 | 3.14% | | 建筑工程 | 181 | 3.19% |
| 建筑工程 | 163 | 2.87% | | 元器件 | 164 | 2.89% |
| 家居用品 | 141 | 2.49% | | 食品 | 153 | 2.70% |
| 软件服务 | 137 | 2.42% | | 家居用品 | 152 | 2.68% |
| 机械基件 | 126 | 2.22% | | 化学制药 | 131 | 2.31% |
| 食品 | 119 | 2.10% | | 家用电器 | 125 | 2.20% |

## Verdict

⚠ seed=4's lift is concentrated: 100.6% from a single month (threshold = 35%). Phase 18 ensemble confidence should be downgraded.