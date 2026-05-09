# Main-Wave (主升浪) Label Scanning — Math Reference

**P0 winner**: Method A `v2_excess_adaptive` at horizon `t3`, threshold `τ=1.2327`.
**Test PR_AUC**: 0.122 (lift 3.0× base 4.07%), ECE 0.010, daily_precision@5 0.20.
**Null tests**: both PASSED (label-shuffle 0.989×, date-shuffle 1.491×).

See `handoffs/2026-05-09-wave-label-ablation/{SPEC,RESULTS}.md` for the full design.

---

## Event primitive (all four methods)

```python
@dataclass(frozen=True)
class Event:
    ts_code: str
    event_start_idx: int   # first day of the wave (decision day = start - 1)
    event_peak_idx: int    # peak day
    event_quality: float   # method-specific continuous score
    event_method: str      # 'A' | 'B' | 'C' | 'D'
```

**Per-stock non-overlap dedupe**: any two accepted events satisfy
`e1.event_peak_idx < e2.event_start_idx` (or vice versa). On overlap, keep
higher `event_quality`.

**Decision-day labels**:
```
y_t1[t, j]  = 1 iff ∃ event e_j with event_start_idx == t+1
y_t3[t, j]  = 1 iff ∃ event e_j with event_start_idx ∈ {t+1, t+2, t+3}
y_e20[t, j] = 1 iff ∃ event e_j with event_start_idx ∈ {t+1, ..., t+20}
```

---

## Method A — `v2_excess_adaptive` (P0 winner)

User-original v2 with continuous score. Per-stock single-pass scan.

For each candidate decision day `t`:

```
1.  inflection:    today_gain ≥ 0.005   AND   prior_5d_cum ≤ 0.03
2.  liquidity:     amount_ma20 ≥ 1e8 元
3.  vol_t-1:       vol20 = ewm_std(pct_change, halflife=10)[t-1]
                   adaptive_thr = max(0.06, 1.8 · vol20)
4.  forward peak:  peak_offset = argmax_{k ∈ [3, 20]} (excess_return[t+k])
                   excess_return[k] = (close[t+k]/close[t] − 1) − (bench[t+k]/bench[t] − 1)
                   fwd_max_excess = excess_return[peak_offset]
5.  threshold:     fwd_max_excess ≥ adaptive_thr
6.  drawdown:      max_dd_during_rally ≤ 0.02 + 0.5 · fwd_max_excess
7.  pace:          fwd_max_excess / peak_offset ≥ 0.005

event_quality_A = fwd_max_excess / adaptive_thr     (continuous, ≥1 means above threshold)
event_start = t,  event_peak = t + peak_offset
```

**P0 lock**: `event_quality_A ≥ τ_A = 1.2327` (calibrated on train_eff
2023-01..2024-12-04 to land 0.80% positive rate).

---

## Method B — Trend-Scanning (López de Prado, 2020)

Vectorized OLS over forward windows.

```
For each L ∈ {5, 10, 15, 20}:
    Fit log(close[t : t+L]) ~ β₀ + β₁·k,  k = 0..L-1
    t_stat(L) = β₁ / SE(β₁)
    SE(β₁)    = sqrt(SSR/(L-2) / Σ(k - k̄)²)

best_t[t]  = argmax_L |t_stat(L)|
best_L[t]  = corresponding L

Event fires at t iff:
    universe[t] AND best_t[t] > 0 AND best_slope[t] > 0 AND amount_ma20[t] ≥ 1e8
event_start = t, event_peak = t + best_L
event_quality_B = best_t[t]   (signed t-stat, continuous)
```

τ_B (P0 calibration) = 4.31 t-stat; falls back to data-side ablation runner.

---

## Method C — Triple-Barrier (López de Prado, 2018)

```
For each candidate t:
    σ_t   = ewm_std(pct_change, halflife=10)[t-1]
    upper = close[t] · (1 + 2.0 · σ_t)
    lower = close[t] · (1 − 2.0 · σ_t)
    vert  = t + 20

    Walk forward k = 1..20:
        if close[t+k] ≥ upper, fire UP-event, break
        if close[t+k] ≤ lower, no event, break

event_start = t,  event_peak = t + first_upper_hit
event_quality_C = (close[event_peak] − close[t]) / (close[t] · σ_t)
                = "how many σ above entry the wave traveled"
```

τ_C (P0 calibration) = 2.90 σ.

---

## Method D — Directional Change (Glattfelder & Tsang, 2011)

Multi-θ state machine over `adj_close`.

```
For each θ ∈ {0.03, 0.05, 0.08}:
    state_machine over close per stock, tracking running extreme
    when |close[t] − last_low| / last_low ≥ θ:
        emit UP-event:
            event_start = last_low_idx
            event_peak  = t  (current extreme high)
            magnitude   = (close[t] − last_low) / last_low

After scanning all θ paths, dedupe overlapping events keeping highest quality.
event_quality_D = magnitude / θ_min     (multiples of base θ=3%)
```

τ_D (P0 calibration) = 2.72 (i.e. ~8.2% magnitude).

---

## Which to use for RL training

**P0 main label**: A_t3 with τ=1.2327. This is what the production LightGBM
predicts (`models/wave_label_ablation/A_t3/model.txt` on data side).

**For RL**:
- Reward shaping option 1 (dense): `r_t = event_quality_A[t,j]` (continuous, non-zero
  on ~all days; better than the previous binary `main_wave_target` which was
  0 99.2% of the time).
- Reward shaping option 2 (validated label): `r_t = y_t3[t,j]` (binary 0/1
  but at known 0.80% positive rate, with non-trivial 1.49× null-shuffle
  predictability ⇒ real signal).
- Multi-head training (P2): use t1 + t3 + e20 as 3 reward channels with
  decreasing weights {1.0, 0.6, 0.3}.

---

## Top-level convenience

```python
from aurumq_rl.labeling import scan_main_wave_p0, MarketPanel

panel = MarketPanel(
    trade_dates=...,        # list[date], length T
    ts_codes=...,           # list[str],  length S
    adj_close=...,          # (T, S) float64
    pct_change=...,         # (T, S) float64
    amount=...,             # (T, S) float64 (raw 元)
    universe=...,           # (T, S) bool
    benchmark_close=...,    # (T,) float64
)
events, label_df = scan_main_wave_p0(panel)
```

The data-side (AurumQ) provides PG-backed `MarketPanel` loader + universe mask
+ benchmark; here on the RL side, `MarketPanel` is just a numpy container —
caller produces the arrays from whatever bundle source.

---

## Test pass

```
21 / 21 passed
- tests/labeling/test_events_dedupe.py     8 (event primitive + horizon derivation)
- tests/labeling/test_methods_synthetic.py 9 (A/B/C/D detect known patterns)
- tests/labeling/test_thresholds.py        4 (target-pos-rate search)
```
