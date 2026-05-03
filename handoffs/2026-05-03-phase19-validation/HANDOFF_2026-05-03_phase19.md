# Phase 19 — Validation + Real Execution Constraints

> 2026-05-03. 6h unattended validation of the Phase 18 candidate
> `ens_final_rankmean` (6 eligible drop_mkt seeds). **Phase 16a stays as
> production**; the ensemble can at most be confirmed as a "candidate
> that survives realistic constraints on the historical OOS". No
> fresh-holdout promotion is possible — the panel ends on 2026-04-24
> and 0 post-2026-04-24 evaluation dates are available with fp=10.

## TL;DR

* **fresh_holdout_status = INSUFFICIENT** (0 usable post-2026-04-24 eval
  dates; threshold ≥40 for formal validation, ≥20 for smoke). Phase 19
  cannot promote anything to production.
* On the historical OOS panel, **`ens_rankmean6` is robust across windows
  AND survives execution constraints**: 100% rolling-60d win rate vs
  16a, beats 16a in every quarter block, and at 60bps round-trip cost
  delivers post-cost adj_S = +0.971 vs 16a's +0.579 (Δ = +0.392).
* But: **seed=4's lift on the ensemble is single-month-concentrated**.
  100.6% of seed=4's marginal contribution to ens6 (vs ens5-without-4)
  comes from 2026-01 alone. This downgrades confidence in the
  +0.283 ensemble advantage being generalizable to fresh data.
* **Decision: KEEP Phase 16a as production. `ens_rankmean6` stays as
  candidate. Wait for ≥40 fresh post-2026-04-24 eval dates before
  promotion. Phase 20 should focus on holdout collection and not on
  more single-seed RL training.**

## 1. Preflight

| | value |
|---|---|
| branch / HEAD | main / `4124599` (post-Phase-18 OSS uploader commit) |
| Python / CUDA | RTX 4070 ✓ |
| data file | `data/factor_panel_combined_short_2023_2026.parquet` |
| data sha256[:16] | `4fd735f488c53ca2` |
| data size | 8,063,859,043 B |
| working tree | 5 untracked pre-session scratch scripts only |
| eval scripts | `_eval_all_checkpoints.py` (14 corrected fields), `_ensemble_eval.py` (15 corrected fields) — both emit `*_adjusted` / `non_overlap` / `vs_random_p50_adjusted` ✓ |

Model SHA256 (first 16) for the candidate set:

| label | path | sha256[:16] |
|---|---|---|
| 16a | models/production/phase16_16a_drop_mkt_best.zip | `ae924791643ee77d` |
| 17b | models/production/phase17_17b_drop_mkt_seed1_best.zip | `ae94766f39e66363` |
| 17d | models/production/phase17_17d_drop_mkt_seed3_best.zip | `ccdefab50691eae7` |
| p18s4 | models/phase18/phase18_18a_drop_mkt_seed4_best.zip | `fd1795da33a6f321` |
| p18s5 | models/phase18/phase18_18b_drop_mkt_seed5_best.zip | `2806973c2e3dd138` |
| p18s6 | models/phase18/phase18_18c_drop_mkt_seed6_best.zip | `abcaa313db3888ac` |

Excluded but available for sensitivity:

| label | sha256[:16] | gate failure |
|---|---|---|
| 17c (seed=2) | `d3cd78fa108d3ea4` | vs_p50_adj=-0.060, IC=-0.0025 |
| p18s7 (seed=7) | `f8523e3f49cde6a2` | vs_p50_adj=+0.004, IC=-0.0002 |

## 2. Data freshness gate (Stage A — DECISIVE)

```
panel.trade_date min:  2023-01-03
panel.trade_date max:  2026-04-24
n_unique trade_dates:  800
post-2026-04-24 dates: 0
forward_period:        10
last evaluable date:   ~2026-04-10
fresh holdout count:   0  (smoke threshold ≥20, formal ≥40)
fresh_holdout_status:  INSUFFICIENT
```

Per Phase 19 §"决策规则", this locks the conclusion to "Phase 16a stays
as production; rankmean6 stays as candidate; await fresh post-2026-04
data".

`pct_chg` scale check confirmed decimal (p01 = -0.0836, p99 = +0.1001 —
±10% daily limit). No /100 conversion needed for Stage C limit checks.

## 3. Stage B — multi-window OOS validation

OOS panel: 2025-07-01 → 2026-04-24 (199 dates, 189 eval-able after fp=10).
See `reports/phase19_validation/fixed_window_eval.{md,json}` for the
full table. Headline numbers:

### Quarter blocks

| variant | Q3 2025 vs_p50 | Q4 2025 vs_p50 | Q1+Apr 2026 vs_p50 |
|---|---:|---:|---:|
| single_16a | +0.732 | +0.340 | +0.287 |
| single_p18s4 | +2.517 | +0.828 | +0.173 |
| **ens_rankmean6** | **+1.024** | **+0.741** | **+0.635** |
| ens_zmean6 | +0.415 | +0.925 | +0.561 |
| ens_zmedian6 | +0.381 | +1.216 | +0.561 |

`ens_rankmean6` beats 16a by ≥+0.292 in every quarter; beats 16a in 3/3 quarters.
Single seed=4 has the highest individual quarter (Q3 +2.517) but is more variable across quarters.

### Per-month vs_random_p50_adjusted (rankmean6 vs 16a)

| month | n_days | rankmean6 | 16a | rankmean6 wins? |
|---|---:|---:|---:|---|
| 2025-07 | 23 | +3.094 | +4.764 | no (single month) |
| 2025-08 | 21 | -0.179 | +0.349 | no |
| 2025-09 | 22 | +1.676 | +0.253 | yes |
| 2025-11 | 20 | +1.431 | -0.100 | yes |
| 2025-12 | 23 | +0.367 | +1.164 | no |
| 2026-01 | 20 | **+3.236** | +1.632 | yes (this is also the seed=4 spike month — see §5) |
| 2026-03 | 22 | +1.351 | +0.407 | yes |

rankmean6 beats 16a in 4 of 7 months. (Months with insufficient eval-able dates excluded; e.g., Oct 2025 / Feb 2026 / Apr 2026 partials.)

### Rolling 60-trading-day windows, step 20 (7 windows total)

| variant | n | mean vs_p50 | median | min | max | IC pos rate | win rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| single_16a | 7 | +0.432 | +0.380 | +0.157 | +0.777 | 1.000 | 1.000 |
| single_p18s4 | 7 | **+1.219** | +0.832 | +0.370 | +2.734 | 1.000 | 1.000 |
| **ens_rankmean6** | 7 | +0.781 | +0.683 | +0.311 | +1.193 | **1.000** | **1.000** |
| ens_zmean6 | 7 | +0.675 | +0.581 | +0.075 | +1.299 | 1.000 | 1.000 |
| ens_zmedian6 | 7 | +0.742 | +0.477 | +0.164 | +1.272 | 1.000 | 1.000 |

* **rankmean6: 100% rolling win rate AND 100% IC-positive rate.**
* mean lift over 16a across rolling windows: +0.349, median lift +0.303.
* min vs_p50 of rankmean6 across the 7 windows is +0.311 — never falls below 16a's *median* (+0.380).

This is a strong consistency story for the candidate. But all 7 windows
overlap heavily by construction (60-day window with step 20 means each
date appears in 1-3 windows), so the "win rate" is not strictly i.i.d.

## 4. Stage C — execution constraint simulation

Conventions:
* T+1 entry: pick at signal day t, buy at close[t+1] (proxy for next-day open — the panel does not carry an open column).
* 10-day hold: sell at close[t+11].
* Filters at entry: skip ST, suspended (vol=0), IPO < 60 days, day-1 limit-up (pct ≥ +9.9%).
* Limit-down at exit: defer up to 5 days; if still locked, sell at the last attempted close.
* Cost = round-trip total bps applied once per trade as a log-return subtraction. **Specified as round-trip total**, not per-side.
* Aggregated log-return per rebalance, annualised by `sqrt(252/10) = sqrt(25.2)`.

Random-pick baseline (60bps, 20 sims): p50 adj_S = +0.486.

### Non-overlap (rebalance every 10 days, 18 rebalances)

| candidate | 30 bps | 60 bps | 100 bps | turnover | failed entry | deferred exit | max DD |
|---|---:|---:|---:|---:|---:|---:|---:|
| single_16a | +1.188 | +0.579 | -0.233 | 0.974 | 0.000 | 0.000 | -0.130 (100bps) |
| single_p18s4 | +1.183 | +0.689 | +0.029 | 0.972 | 0.000 | 0.004 | -0.126 |
| **ens_rankmean6** | **+1.496** | **+0.971** | **+0.272** | 0.976 | 0.000 | 0.007 | -0.118 |

Δ vs 16a (post-cost): +0.308 / +0.392 / +0.505 across the three cost levels.
The lift over 16a is **largest at 100 bps**. ens_rankmean6 stays positive even at 100 bps; 16a flips negative.

### Daily 10-sleeve (10 parallel non-overlap portfolios offset by 1 day)

| candidate | 30 bps adj_S | 60 bps | 100 bps | max DD (60bps) |
|---|---:|---:|---:|---:|
| single_16a | +1.080 | +0.483 | -0.319 | -0.166 |
| single_p18s4 | +1.165 | +0.643 | -0.077 | -0.143 |
| **ens_rankmean6** | **+1.470** | **+0.948** | +0.236 | -0.144 |

Same pattern. The ensemble is consistently the best across both rebalancing
modes and all cost levels.

## 5. Stage D-1 — ensemble ablation

Aggregation: per-date percentile rank-mean.

| variant | members | vs p50 adj | Δ vs base |
|---|---|---:|---:|
| **rankmean5_minus_16a** | 17b+17d+18s4+18s5+18s6 | **+0.726** | **+0.045** ← improves |
| rankmean6 (baseline) | 16a+17b+17d+18s4+18s5+18s6 | +0.681 | 0 |
| rankmean5_minus_17b | 16a+17d+18s4+18s5+18s6 | +0.639 | -0.042 |
| rankmean3_top3_strong | 18s4+18s5+18s6 | +0.572 | -0.109 |
| rankmean5_minus_17d | 16a+17b+18s4+18s5+18s6 | +0.569 | -0.112 |
| rankmean5_minus_p18s5 | 16a+17b+17d+18s4+18s6 | +0.540 | -0.141 |
| rankmean_all_loaded | 7 + 17c | +0.525 | -0.156 |
| rankmean7_with_seed2_17c | 6 + 17c | +0.523 | -0.158 |
| rankmean5_minus_p18s4 | 16a+17b+17d+18s5+18s6 | +0.512 | -0.169 |
| **rankmean5_minus_p18s6** | 16a+17b+17d+18s4+18s5 | **+0.359** | **-0.322** |

Reading:

* **Removing 16a IMPROVES** the ensemble (+0.045). On this OOS, the Phase 16 baseline is the WEAKEST contributor among the 6 eligibles. (Note: numbers slightly differ from Phase 18's +0.711 because Phase 19 random baseline uses 50 sims, not 100; the relative ranking is robust.)
* **p18s6 (seed=6) is the most load-bearing** — leaving it out costs −0.322. p18s4 (seed=4) is second at −0.169.
* **Adding seed=2 (17C) hurts** by −0.158. Confirms the eligibility gate.
* `rankmean3_top3_strong` (only the 3 Phase 18 seeds 4/5/6) at +0.572 is OK but loses −0.109 vs the full 6. So both Phase 18 seeds AND the older Phase 17 seeds (17B, 17D) are net contributors.

**Takeaway**: the rankmean6 baseline is robust to dropping any single member (worst case −0.322 from leaving p18s6). It's NOT a one-seed-driven illusion. But there is an alternative configuration (rankmean5_minus_16a) that scores marginally higher on this OOS — flagged as a Phase 20 hypothesis to validate on fresh data.

## 5. Stage D-2 — seed=4 forensics (concentration warning)

| metric | value |
|---|---|
| Daily top-30 Jaccard, seed=4 vs ens6 | mean 0.215 |
| Daily top-30 Jaccard, ens5_no_seed4 vs ens6 | mean 0.594 |
| Total seed=4 lift over ens5_no_seed4 | +1.643 adj_S (sum across months) |
| Max single-month lift | +1.653 adj_S (2026-01) |
| **Max single-month share of total lift** | **100.6%** ⚠ |

Per-month decomposition (lift = ens6 − ens5_without_seed4):

| month | seed=4 alone | ens6 | ens5 (no seed=4) | lift from seed=4 |
|---|---:|---:|---:|---:|
| 2025-12 | +4.397 | +4.001 | +3.605 | +0.396 |
| **2026-01** | **+2.329** | **+5.219** | **+3.566** | **+1.653** |
| 2026-02 | +0.455 | -0.788 | -0.481 | -0.307 |
| 2026-03 | -2.462 | -1.676 | -1.954 | +0.279 |
| 2026-04 (partial) | +5.864 | +4.285 | +5.895 | -1.610 |

**Verdict (per Phase 19 instructions, threshold 35%): ⚠ seed=4's lift is concentrated.** 100.6% of total lift comes from a single month. Phase 18 ensemble confidence is **downgraded** for this reason.

Industry distribution: seed=4 and ens6 share the same top-3 industries (汽车配件 / 电气设备 / 化工原料). No single-industry concentration issue.

Implication: **rankmean6 may NOT outperform rankmean5_no_seed4 on fresh data**. We have one calendar month (Jan 2026) doing all the work. Given the concentration warning, the preferred conservative call for production is to track BOTH `rankmean6` AND `rankmean5_minus_16a` (the highest ablation result) on fresh post-2026-04 data, not to commit to rankmean6 alone.

## 6. Decision per Phase 19 §决策规则

The `fresh_holdout_status = INSUFFICIENT` clause already locks the
conclusion. Add:

* `rankmean6` median window lift over 16a = +0.303 (rolling-60), well above the +0.03 floor.
* `IC` is positive (+0.0278) on the full window.
* `non_overlap` Sharpe is +1.938 (Phase 18 measurement; Phase 19 corrected eval consistent).
* Post-cost 60bps: `rankmean6` post-cost adj_S = +0.971 vs 16a's +0.579 (Δ +0.392 — well above 16a).
* Monthly win rate vs random p50: rankmean6 has 6 of 7 months positive (consistent with §"月度胜率 ≥60%").
* No single-industry concentration in picks.
* **BUT**: seed=4 single-month lift concentration = 100.6%. This is the one gate that fails.

So we are in the conditional position:

> "若 constrained/post-cost 后 ens_final_rankmean 相对 16a 的 median window lift < +0.03, 或 IC 非正, 或成本 60bps 下回撤/turnover 明显恶化: 不推进 serving"

None of these failure conditions are met (median lift +0.303, IC > 0,
60bps post-cost lift +0.392 — strong). And the strong-pass conditions
ARE met EXCEPT for the fresh-holdout requirement.

**Final decision (locked by INSUFFICIENT fresh holdout)**:

1. **KEEP Phase 16a as live production**.
2. **Mark `ens_rankmean6` as Phase 19 release-candidate for ensemble serving** — pending fresh post-2026-04 holdout.
3. **Track `rankmean5_minus_16a` as a sibling candidate** (highest ablation score, marginally beats baseline; small chance the 16a removal helps).
4. **DO NOT** advance to serving implementation until ≥40 fresh post-2026-04 eval dates are available AND the candidate clears the same gates on that fresh window.

## 7. Skipped / not done

* **Stage E (more seeds, 8/9)**: Phase 19 §可选 says only fire if "验证完成 AND remaining > 2h". Validation is complete, but the seed=4 concentration warning + fresh-holdout-INSUFFICIENT verdict make new seed training low priority — it would just expand the seed distribution without resolving the concentration risk. Deferred to Phase 20 alongside fresh-holdout collection.
* **Live-style portfolio sizing simulation**: not done. Stage C uses equal-weight top-30 with no position sizing or risk targeting. Phase 20 should add a Kelly-fraction or risk-parity sizing layer.

## 8. Recommended Phase 20 actions

| priority | task |
|---|---|
| **HIGHEST** | Collect post-2026-04-24 panel data. Without ≥40 fresh evaluation dates we cannot promote any candidate. |
| HIGH | When fresh data arrives, re-run Stage A/B/C/D on the post-2026-04 window. Apply the same gates. |
| HIGH | Investigate Jan 2026 specifically: why does seed=4 dominate that month? Is it a sector-rotation event? Earnings season? Calendar effect? |
| MED | Implement live-style portfolio sizing with risk-parity / volatility targeting; re-run Stage C. |
| MED | If the rankmean5_minus_16a alternative is also strong on fresh data, prefer it over rankmean6 (smaller member set = lower compute cost in serving). |
| LOW | Train seeds 8/9 to widen the seed distribution CI. Only after the fresh-data gate is passed. |
| **DO NOT** | Drop more factor groups based on importance alone. Phase 17A already proved the conditional-importance trap. |
| **DO NOT** | Promote any candidate to production without fresh-holdout validation. |

## 9. Artifacts

```
runs/phase19_validation/decision_log.md

reports/phase19_validation/
  data_freshness.{md,json}
  fixed_window_eval.{md,json}
  execution_sim.{md,json}
  ensemble_ablation.{md,json}
  seed4_forensics.{md,json}

scripts/_phase19_multi_window_eval.py
scripts/_phase19_execution_sim.py
scripts/_phase19_ablation_and_seed4.py

handoffs/2026-05-03-phase19-validation/
  HANDOFF_2026-05-03_phase19.md   (this file)
```

The Phase 18 ensemble score caches in `reports/phase18_6h/_scores_cache_*.npy`
were reused — Phase 19 did NOT re-score the 6+ models. This is why the
total session GPU time was minimal; most of Phase 19 was numpy array
operations on cached data.

## 10. Final answer

```text
Phase19 completed in <session-elapsed>.

Best result on historical OOS:
- ens_rankmean6 (rank-mean of 16a + 17b + 17d + p18s4 + p18s5 + p18s6):
  post-cost (60bps) adj_S = +0.971 vs 16a's +0.579 (Δ = +0.392)
  rolling-60d win rate 100% vs 16a, IC pos rate 100%
  3/3 quarter blocks beat 16a

Concentration warning:
- seed=4's marginal lift on the ensemble is 100.6% from 2026-01 alone.
  Single-month dependence is a downgrade signal for Phase 18 confidence.

Fresh holdout status: INSUFFICIENT (0 dates post-2026-04-24)

Recommendation:
- KEEP Phase 16a as live production (NO overwrite).
- Mark ens_rankmean6 as Phase 19 release-candidate; track also
  rankmean5_minus_16a as sibling.
- Phase 20 must collect fresh holdout data and re-validate.

Artifacts:
- handoffs/2026-05-03-phase19-validation/HANDOFF_2026-05-03_phase19.md
- reports/phase19_validation/{data_freshness,fixed_window_eval,execution_sim,ensemble_ablation,seed4_forensics}.{md,json}
- runs/phase19_validation/decision_log.md
- OSS: oss://ledashi-oss/fromsz/handoffs/2026-05-03-phase19-validation/
```
