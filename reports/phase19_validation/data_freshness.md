# Data freshness gate

## Result: INSUFFICIENT fresh holdout

| metric | value |
|---|---:|
| panel trade_date min | 2023-01-03 |
| panel trade_date max | 2026-04-24 |
| n unique trade_dates | 800 |
| post-2026-04-24 dates | 0 |
| forward_period | 10 |
| last evaluable date in panel | ~2026-04-10 |
| **available fresh holdout eval dates** | **0** (gate threshold: ≥40) |

## Locked conclusion

`fresh_holdout_status = INSUFFICIENT` (0 < 20 minimum, 0 < 40 formal).

Per Phase 19 instructions:
- "若 max_trade_date <= 2026-04-24 或 post-2026-04-24 可用评价日期 <20, fresh holdout 标记为 INSUFFICIENT, 不得给生产晋级结论"

This INSUFFICIENT verdict drives the final recommendation regardless of
Stage B/C/D outcomes.

## What this means for Phase 19

* Stages B (multi-window) / C (execution sim) / D (ablation + seed=4
  forensics) STILL RUN — they produce evidence that informs whether the
  Phase 18 ensemble *holds up* on the historical OOS under realistic
  constraints.
* The ensemble can at most be confirmed as a "candidate that survives
  realistic constraints on historical OOS". It CANNOT be promoted to
  production replacement.
* Phase 20 must wait for ≥40 fresh post-2026-04 eval dates (~2 calendar
  months of trading days) before the candidate is eligible for
  promotion.
