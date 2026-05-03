# Phase 19 Stage C — execution constraint simulation

OOS panel: 2025-07-01 → 2026-04-24 (199 dates, fp=10, top-K=30).

Conventions:
- T+1 entry: pick at signal day t, buy at close[t+1] (proxy for next-day open).
- 10-day hold: sell at close[t+11].
- Filters at entry: skip ST, suspended (vol=0), IPO < 60 days, day-1 limit-up (pct >= +0.099).
- Limit-down at exit: defer up to 5 days; if still locked, sell at last attempted close.
- Cost = round-trip total bps applied once per trade as a log-return subtraction.
- Aggregated as equal-weight log-return per rebalance, annualised by sqrt(252/hold_days) = sqrt(25.2).
- Random-pick baseline (60bps, 20 sims): p50 adj_S = +0.486

## Non-overlap (rebalance every 10 days)

| candidate | cost bps | n_rebal | adj S | total log ret | max DD | failed entry | deferred exit | turnover |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| single_16a | 30 | 19 | +1.188 | +0.111 | -0.076 | 0.000 | 0.000 | 0.974 |
| single_16a | 60 | 19 | +0.579 | +0.054 | -0.090 | 0.000 | 0.000 | 0.974 |
| single_16a | 100 | 19 | -0.233 | -0.022 | -0.130 | 0.000 | 0.000 | 0.974 |
| single_p18s4 | 30 | 19 | +1.183 | +0.136 | -0.094 | 0.000 | 0.004 | 0.972 |
| single_p18s4 | 60 | 19 | +0.689 | +0.079 | -0.102 | 0.000 | 0.004 | 0.972 |
| single_p18s4 | 100 | 19 | +0.029 | +0.003 | -0.126 | 0.000 | 0.004 | 0.972 |
| ens_rankmean6 | 30 | 19 | +1.496 | +0.163 | -0.093 | 0.000 | 0.007 | 0.976 |
| ens_rankmean6 | 60 | 19 | +0.971 | +0.106 | -0.103 | 0.000 | 0.007 | 0.976 |
| ens_rankmean6 | 100 | 19 | +0.272 | +0.030 | -0.118 | 0.000 | 0.007 | 0.976 |

## Daily 10-sleeve (10 parallel portfolios offset by 1 day)

| candidate | cost bps | n_sleeves | n_pooled | adj S | max DD |
|---|---:|---:|---:|---:|---:|
| single_16a | 30 | 10 | 188 | +1.245 | -0.103 |
| single_16a | 60 | 10 | 188 | +0.718 | -0.114 |
| single_16a | 100 | 10 | 188 | +0.017 | -0.191 |
| single_p18s4 | 30 | 10 | 188 | +1.357 | -0.144 |
| single_p18s4 | 60 | 10 | 188 | +0.918 | -0.154 |
| single_p18s4 | 100 | 10 | 188 | +0.334 | -0.168 |
| ens_rankmean6 | 30 | 10 | 188 | +1.427 | -0.112 |
| ens_rankmean6 | 60 | 10 | 188 | +0.950 | -0.117 |
| ens_rankmean6 | 100 | 10 | 188 | +0.313 | -0.141 |