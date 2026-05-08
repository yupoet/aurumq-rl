# Phase 23 — Episode-Based Targets for Main-Wave Entry

Status: locked 2026-05-06 (auto-mode, no review gate).

## 1. Goal

Replace Phase 22's exit-coupled `hold_return` reward with an entry-only target
that directly measures **"Is this T-1 of a high-quality main wave?"**, scored
along the four dimensions the user specified:

1. **Peak height** — final cumulative gain of the wave
2. **Proximity to start** — T-1 ideal, T-2/T-3 partial credit, T-4+ no credit
3. **Duration** — longer rallies score higher (within reason)
4. **Smoothness** — smaller drawdown during the rally scores higher

Phase 22's reward (`hold_return = price_at_5d_or_death_cross / entry - 1`)
caps at the 5-day window: it cannot distinguish T-1 of a 30% / 20-day rally
from T-1 of a 6% / 5-day rally because both look ~+5-8% in 5 days. The new
reward sees the full episode peak and rewards good entries proportionally.

## 2. Three layers

### 2.1 Episode scanner — `src/aurumq_rl/main_wave_episodes.py`

Pure-numpy module. Input: `close, vol, valid_basic` arrays of shape `(T, S)`.
Output: list of `MainWaveEpisode` records:

```python
@dataclass
class MainWaveEpisode:
    stock_idx: int
    t_start: int       # day price first qualifies as a wave start
    t_peak: int        # day cumulative return peaks within the wave
    peak_return: float # close[t_peak] / close[t_start] - 1
    duration: int      # t_peak - t_start (inclusive of t_start)
    max_dd_during: float  # max drawdown from running peak during [t_start, t_peak], abs value
    daily_returns: np.ndarray  # path of close ratios, len = duration + 1
```

Algorithm:

```
for each stock j:
    for each candidate t_start where:
        - amount_ma20[t_start, j] > 1e8 (liquidity gate)
        - valid_basic[t_start, j] is True
        - close[t_start - 1, j] / close[t_start - 2, j] - 1 < 0.02  (avoid "started yesterday")
    
    For each look-ahead L in [3, max_duration=20]:
        path = close[t_start : t_start+L+1, j] / close[t_start, j] - 1
        peak_offset = argmax(path)
        peak_ret = path[peak_offset]
        if peak_ret < min_peak_return (=0.10): continue
        running_peak = np.maximum.accumulate(path[:peak_offset+1])
        max_dd_during = max(running_peak - path[:peak_offset+1])
        if max_dd_during > max_dd_during_rally (=0.05 * peak_ret + 0.02): continue
        # this is a valid wave; record once at the FIRST L that triggers
        record(MainWaveEpisode(j, t_start, t_start + peak_offset, peak_ret, peak_offset, max_dd_during))
        break  # only one episode per t_start (the smallest valid window)
    
    # advance t_start past t_peak to avoid overlapping episodes from same starting region
```

Notes:
- Episodes do not overlap per stock (a new candidate must come after the previous t_peak).
- Liquidity gate is at t_start only — episodes that lose liquidity mid-rally are still recorded.
- "Started yesterday" guard prevents picking T+0 candidates that already gapped up.

### 2.2 Target labels — `src/aurumq_rl/main_wave_target_labels.py`

Pure numpy. Input: episodes list + (T, S). Output:

```python
@dataclass
class MainWaveTargets:
    target_quality: np.ndarray  # (T, S) float, the reward signal at decision day t
    is_pre_main_wave: np.ndarray  # (T, S, L) bool, where L=lookahead. layer k=1 means T-1
    episode_id: np.ndarray      # (T, S, L) int32, -1 if no episode
    proximity_at_decision: np.ndarray  # (T, S) int8, value in {1..L} = best (smallest k) hit, 0 = miss
```

`target_quality` formula (only nonzero when at least one episode has `T_start ∈ [t+1, t+L]`):

```
For each episode E, compute:
    quality(E) = peak_return(E)
                 * duration_factor(E.duration)        # = clip(duration / 10, 0.5, 1.5)
                 * smoothness_factor(E)                # = 1 - clip(max_dd / peak_return, 0, 1)
                 # range roughly [0, 0.10..0.50]
    
For each (t, j) decision day, find episodes E in stock j with T_start(E) in [t+1, t+L]:
    target_quality[t, j] = max over such E of:
        quality(E) * proximity_weight[T_start(E) - (t+1)]
    where proximity_weight = [1.0, 0.6, 0.3]  (T-1, T-2, T-3)
```

`max` (not `sum`) so target stays bounded; we want the best entry, not stacked credit.

### 2.3 Reward + eval wiring

- **Env**: `gpu_env.GPUStockPickingEnv` accepts `target_quality_t` cuda tensor, used as reward source when `--reward-mode main_wave_target` is selected. Same indexing convention (row t = reward for action at t).
- **Train**: new CLI flag in `train_v2.py`. Computes episodes + targets at startup, pushes target_quality to cuda, passes to env. Tightens valid_mask using episode-aware criteria (basic + liquidity).
- **Eval**: new script `_eval_main_wave_episode.py` (V1 path). Replaces `hold_return` metrics with episode-based ones (see §3).

## 3. New eval metrics

| Metric | Definition |
|---|---|
| `t_minus_1_hit_rate` | P(picked stock j on day t has an episode E with T_start(E) == t+1) |
| `t_minus_3_hit_rate` | Same with T_start in {t+1, t+2, t+3} |
| `avg_peak_return_of_hits` | Mean `peak_return(E)` over hits |
| `avg_duration_of_hits` | Mean `duration(E)` |
| `avg_smoothness_of_hits` | Mean `1 - max_dd / peak_return` |
| `proximity_distribution` | Counts of picks at T-1 / T-2 / T-3 / miss |
| `daily_t_minus_1_precision` | P(at least one of top_K is a real T-1 on a given day) |
| `eval_score_v23` | Composite (see §4) |

Removed: `basic_win_rate`, `avg_hold_return`, `payoff_ratio`, `avg_max_drawdown` (these are exit-dependent, no longer applicable; user explicitly said 管进不管出).

## 4. Composite eval_score_v23

```
eval_score_v23 = 
    0.35 * t_minus_1_hit_rate
  + 0.20 * t_minus_3_hit_rate
  + 0.20 * tanh(avg_peak_return_of_hits / 0.20)    # 20% peak satures the score
  + 0.10 * tanh((avg_duration_of_hits - 5) / 5)    # 5d baseline, 10d maxes out
  + 0.10 * avg_smoothness_of_hits
  + 0.05 * daily_t_minus_1_precision
```

Random pick should land at T-1 of a real episode at base rate `n_episodes / (n_label_valid_cells)`. We expect base rate ~0.5%-1% depending on threshold tuning. Hitting 2-3% is strong signal.

## 5. Descriptive analytics

`scripts/_inspect_main_wave_episodes.py` outputs:

1. **Episode catalog**: total count, distribution of peak_return / duration / max_dd
2. **Per-month**: how many episodes started each month
3. **Per-industry**: episode density and avg peak_return by industry
4. **Factor profile at T-1**: for each factor (mf_*, mfp_*, hk_*, inst_*, senti_*), compare its z-score distribution at T-1 of episodes vs at random (universe baseline). Output: top-20 most discriminative factors with effect size.
5. **Sample episodes**: 10 best by peak_return, 10 typical, with stock/date/duration/max_dd

This is the "(B) 找规律" step the user asked for, embedded in (A).

## 6. Implementation order

1. `main_wave_episodes.py` + tests (8 tests on hand-built panels)
2. `main_wave_target_labels.py` + tests
3. `_inspect_main_wave_episodes.py` (descriptive — runs FIRST so we see if scanner finds enough episodes before committing to retraining)
4. `gpu_env.py` extend `hold_returns` → generic `reward_per_stock` (rename or add alongside)
5. `train_v2.py` CLI flag `--reward-mode main_wave_target`
6. `_eval_main_wave_episode.py`
7. Phase 23A train (300k seed=42 top_k=5)
8. Eval comparison + commit + OSS

## 7. Out of scope

- Multi-seed sweep (Phase 24)
- Drawdown penalty (covered indirectly through smoothness factor in target_quality)
- Concept板块 ablation (data not available)
- Live execution simulation
