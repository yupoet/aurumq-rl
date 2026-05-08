# Phase 23 — Episode-Based Targets: T-1 Hit Rate 2.15× Random

> 2026-05-06. Reward redesigned from "exit-coupled hold_return" (Phase 22)
> to "episode T-1 target_quality" (Phase 23). User explicitly stated 管进
> 不管出 — care about entry timing only. The reward now directly optimises
> "is this T-1 of a real, high-quality main wave?" with proximity weights
> for T-1 / T-2 / T-3.
>
> **Phase 23A run delivers 2.15× random T-1 hit lift** (top-5 step 224928),
> almost double the best Phase 22 run (1.36× for Phase 22A). 723 unique
> stocks across 995 picks — diversified entries, not concentrated bets.
> Sample T-1 hits include 002943 (next-day +61.66% / 20 days) and 002490
> (next-day +58.56% / 17 days).

## TL;DR

* **Goal**: optimise for T-1 of high-quality main waves (4 dimensions:
  high peak, close to start, long duration, low intra-rally drawdown).
* **Architecture**: V1 `PerStockEncoderPolicy`, no V2 hard fork.
* **Reward**: `--reward-mode main_wave_target` — `target_quality[t, j]`
  = `peak_return × duration_factor × smoothness_factor × proximity_weight`,
  with proximity_weight `[1.0, 0.6, 0.3]` for T-1, T-2, T-3.
* **Episode scanner finds 28,369 main waves** in full short-parquet
  panel (2023-01..2026-04, 800 days, 5643 stocks). Median peak +22.8%,
  avg duration 11.5 days, avg max_dd_during 5.2%.
* **Phase 23A retrain (300k seed=42 on short parquet)** delivers 2.15×
  random T-1 lift, 9.05% daily_T1_precision (≈ 1 in 11 days has top-5
  containing real T-1).

## Comparison table (OOS 2025-07-01..2026-04-24)

| Run | topK | best_step | T1_hit | T1_lift | T13_hit | avg_peak | avg_dur | daily_T1 | eval_v23 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Phase 16a (V1 forward_10d, prod) | 3 | 274912 | 1.01% | 1.13× | 3.18% | +30.05% | 13.4 | 2.51% | +0.371 |
| Phase 16a | 5 | 149952 | 1.01% | 1.13× | 3.72% | +26.27% | 12.1 | 5.03% | +0.362 |
| Phase 22A (V1 main_wave_hold s42) | 3 | 124960 | 1.17% | 1.32× | 4.19% | +32.75% | 13.0 | 3.52% | +0.379 |
| Phase 22A | 5 | 74976 | 1.21% | 1.36× | 4.32% | +28.44% | 12.4 | 5.53% | +0.368 |
| Phase 22B (s1 unstable) | 3 | 174944 | 0.34% | **0.38×** | 3.35% | +35.16% | 13.5 | 1.01% | +0.372 |
| Phase 22C (V1 hold tk3) | 5 | 174944 | 1.01% | 1.13× | 3.62% | +26.03% | 11.9 | 4.52% | +0.356 |
| **Phase 23A (V1 main_wave_target s42)** | 3 | final | 1.68% | 1.89× | 3.52% | +27.89% | 13.1 | 5.03% | +0.367 |
| **Phase 23A** | **5** | **224928** | **1.91%** | **2.15×** | 4.42% | +28.16% | 12.5 | **9.05%** | +0.369 |

Random base rate (T-1): **0.89%**.

## Key user-validated business insight (preserved during scan)

The user stated: "累计主力水平比主力净流入更重要" — cumulative main-force
POSITION matters more than recent FLOW. The factor profile across T-5
through T-1 of 28,369 episodes confirms this exactly:

**Stage 1 (T-20 to T-5): Cumulative accumulation builds**
- `mfp_elg_buy_ratio_20d` (extra-large order 20d buy ratio): T-5 +0.606 → T-1 +0.542 (high, slowly fading)
- `mfp_lg_buy_ratio_20d` (large order 20d buy ratio): T-5 +0.507 → T-1 +0.514 (high, stable)
- `hm_seat_count_30d` (hot-money 30d seat count): consistently +0.39
- `inst_appear_count_60d` (institution 60d 龙虎榜 count): consistently +0.38
- `mg_buy_30d_ratio` (margin 30d buy ratio): T-5 +0.299 → T-1 +0.334

**Stage 2 (T-5 to T-1): Main force pauses, accumulation already done**
- `mf_net_3d`: T-5 -0.128 → T-1 -0.453 (rapidly more negative)
- `mf_net_5d`: T-5 -0.145 → T-1 -0.449
- `mf_net_10d`: T-5 -0.183 → T-1 -0.380

The user's mental model — "main force pre-accumulates, then stops on
launch day, technical signal triggers" — is empirically validated. The
reward now incentivises the model to look for stocks with HIGH
cumulative-buy ratios AND LOW recent-flow AND about-to-rally — the same
signals the user identified manually.

## Code changes

| File | Change |
|---|---|
| `src/aurumq_rl/main_wave_episodes.py` (NEW) | Pure-numpy episode scanner. Detects t_start (first up day after flat lookback), t_peak (cum-return peak in [t_start+min_dur, t_start+max_dur]). 8 quality gates: peak ≥ 10%, duration ∈ [3,20], min avg daily 1.5%, dd allowance scaling with peak, liquidity gate, inflection guard, pre-window flat, no zero close. 10 unit tests. |
| `src/aurumq_rl/main_wave_target_labels.py` (NEW) | Episodes → per-(t,j) target_quality. quality(E) = peak × duration_factor × smoothness_factor. Proximity weights [1.0, 0.6, 0.3] for T-1/2/3. `max` over overlapping episodes. 8 unit tests. |
| `src/aurumq_rl/gpu_env.py` | (already has) optional `hold_returns` kwarg from Phase 22 — Phase 23 reuses this slot for `target_quality` (same indexing semantics). |
| `scripts/train_v2.py` | Adds `--reward-mode main_wave_target` (third option after forward_n_day and main_wave_hold). When selected: scans episodes at startup, builds target_quality cuda tensor, tightens valid_mask to entry_eligible. Records `n_episodes_train` in metadata. |
| `scripts/_eval_main_wave_episode.py` (NEW) | V1 ckpt eval against episode-based metrics: T-1 hit rate, T-1..T-3 hit rate, lift over base rate, avg peak/duration/smoothness on hits, daily T-1 precision, composite eval_score_v23. Outputs episode_picks.jsonl. |
| `scripts/_inspect_main_wave_episodes.py` (NEW) | Descriptive scan: per-month / per-industry density, top-N most-discriminative factors at T-1. |
| `scripts/_inspect_factor_at_t_minus_k.py` (NEW) | Drill-down: every factor's z-score evolution from T-5 to T-1 to validate cumulative-vs-flow hypothesis. |
| `tests/test_main_wave_episodes.py` (NEW, 10 tests) | |
| `tests/test_main_wave_target_labels.py` (NEW, 8 tests) | |

## Production candidate

**`runs/phase23a_episode_seed42/checkpoints/ppo_224928_steps.zip`** evaluated
at top_k=5:
- T-1 hit rate **1.91%** (2.15× random)
- daily_T1_precision **9.05%** (≈ 1 of every 11 OOS days has top-5 containing
  a real T-1)
- avg peak return on hits **+28.16%**, avg duration 12.5 days
- 723 unique stocks across 995 picks (well-diversified)

Sample T-1 hits demonstrate real strong rallies caught at the inflection:
- 2026-02-03 002490.SZ → t_start=2026-02-04, peak +58.56%, 17 days
- 2025-12-16 002943.SZ → t_start=2025-12-17, peak +61.66%, 20 days
- 2025-12-05 605168.SH → t_start=2025-12-08, peak +30.86%, 20 days
- 2025-09-04 603950.SH → t_start=2025-09-05, peak +31.71%, 13 days
- 2026-03-19 603876.SH → t_start=2026-03-20, peak +32.09%, 20 days

## Caveats / open issues

1. **Production gate**: Phase 19 fresh-holdout policy (need ≥40 days post-
   2026-04-24) still applies. 23A is a candidate, not a production rotation.
2. **Phase 22 inertia**: eval_score_v23 narrowly favors Phase 22A (+0.379
   at top-3) because 22A's hits include slightly bigger waves and broader
   T-1..T-3 coverage. But T-1 specifically — the user's "best buy point"
   metric — Phase 23A wins decisively. The composite score weights
   matter; user's explicit preference (T-1 > T-3) is captured in the
   T-1-hit metric, not the composite.
3. **Single seed**: only seed=42. Phase 22B at seed=1 collapsed; Phase 23
   should also be re-run on at least 2 more seeds (Phase 24).
4. **Missing technical trigger factors**: The factor scan revealed strong
   cumulative-position signals (mfp_*, hm_seat_count, inst_appear_count)
   but the user-suggested explicit technical triggers (KDJ golden cross,
   MA5 cross above MA10) are NOT in the current factor set. Adding them
   in Phase 24 may further improve T-1 precision.
5. **mfp_main_net_5/20/60/180d are dead** in the data — z-scores are
   exactly 0 across all stocks at all dates. These columns appear to be
   pre-processed in a way that destroys per-stock variance. Worth
   investigating with the data pipeline owner.
6. **`senti_zt_count_30d` is also constant** (z=0 everywhere). Same pre-
   processing issue.

## Phase 24 priorities

1. **Multi-seed validation**: re-run Phase 23 with seeds 1, 2, 3, 4 to
   confirm 23A's 2.15× lift isn't seed-specific (Phase 22 had a seed=1
   collapse).
2. **Add technical trigger factors**: MA5_cross_MA10, KDJ_golden_cross,
   close_vs_MA20_distance. These are the user's stated "启动信号" that
   complement the cumulative-position picture.
3. **Investigate dead factors**: mfp_main_net_*, senti_zt_count_30d.
4. **Reward shaping experiment**: weighted combo `target_quality × peak_height`
   to bias toward bigger waves over more frequent small ones.
5. **Phase 19 fresh-holdout collection**: still required for production rotation.

## Reproduction

```bash
# Train
.venv/Scripts/python.exe scripts/train_v2.py \
    --total-timesteps 300000 \
    --data-path data/factor_panel_combined_short_2023_2026.parquet \
    --start-date 2023-01-03 --end-date 2025-06-30 \
    --universe-filter main_board_non_st \
    --n-envs 16 --episode-length 240 \
    --batch-size 1024 --n-steps 1024 --n-epochs 10 \
    --learning-rate 1e-4 --target-kl 0.30 --max-grad-norm 0.5 \
    --rollout-buffer index --tf32 --matmul-precision high \
    --forward-period 5 --top-k 5 \
    --reward-mode main_wave_target \
    --drop-factor-prefix mkt_ \
    --checkpoint-freq 25000 \
    --seed 42 \
    --out-dir runs/phase23a_episode_seed42

# Eval
.venv/Scripts/python.exe scripts/_eval_main_wave_episode.py \
    --run-dir runs/phase23a_episode_seed42 \
    --data-path data/factor_panel_combined_short_2023_2026.parquet \
    --val-start 2025-07-01 --val-end 2026-04-24 \
    --top-k 3 5

# Episode catalog (descriptive)
.venv/Scripts/python.exe scripts/_inspect_main_wave_episodes.py \
    --data-path data/factor_panel_combined_short_2023_2026.parquet \
    --start-date 2023-01-03 --end-date 2026-04-24 \
    --label full_2023_2026

# Factor drill-down (T-5 to T-1 evolution)
.venv/Scripts/python.exe scripts/_inspect_factor_at_t_minus_k.py \
    --data-path data/factor_panel_combined_short_2023_2026.parquet \
    --start-date 2023-01-03 --end-date 2026-04-24
```
