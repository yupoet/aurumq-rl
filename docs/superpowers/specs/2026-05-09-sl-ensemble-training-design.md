# SL Ensemble Training Plan — Earthquake-style Main-Wave Prediction

**Date**: 2026-05-09
**Author**: ledashi (training side)
**Predecessor**: `runs/p3_findings/RESULTS.md` (P3 PPO findings — drop residual PPO)
**Goal**: Build an SL ensemble on the P3 bundle that maximizes proximity-weighted excess return for top-K stock selection. Score should rise as a stock approaches a main wave (像预测地震).

---

## 1. Problem statement

Given features at trade_date `t` for each main-board non-ST stock, predict a score `s(t, j)` such that the top-K stocks by `s` have the **highest expected proximity-weighted excess return** over the next 3 trading days.

Three properties the score must encode:
1. **Hit**: stock has any positive excess return → score > 0
2. **Magnitude**: bigger rally → higher score
3. **Proximity**: rally on T+1 weighted more than T+3 → score peaks for "rally tomorrow" cases

This is the "earthquake-style" framing: just like seismologists give higher confidence to imminent quakes than distant ones, the model should give higher confidence to imminent rallies than 3-day-away ones.

## 2. Target formulation

```
y_t = (1.0 × max(0, excess_T+1)
     + 0.6 × max(0, excess_T+2)
     + 0.3 × max(0, excess_T+3)) / 1.9

excess_T+d = pct_chg_T+d - eq_weight_market_pct_T+d
```

Properties:
- Continuous regression target in approximately [0, 0.10] (clipped at 0 below)
- Hit + magnitude + proximity all baked in
- Independent of paris's binary label scanners (those become auxiliary check-only signals)
- Direct mapping: predicted ŷ = expected proximity-weighted excess return

Computation: T+1 returns come straight from `realized_returns.parquet`. T+2 / T+3 obtained by self-shifting the same table on `trade_date`. Market returns from `market_returns.parquet` analogously.

Why `max(0, ...)` rather than signed return: signed losses encode short-side info, but production decision is one-sided long-only. Equal-mag negative returns shouldn't be rewarded the same magnitude as gains. Asymmetric ReLU-style target keeps the regression aligned with the long-only decision.

Why weights 1.0/0.6/0.3: same proximity weights paris already used in 26F-v3's `main_wave_target` reward (validated empirically there). Defensible default; revisit if Path 1 results suggest alternative ratios.

## 3. Evaluation methodology

Every model produces predictions on H1 (2025-07..2025-09) and H2 (2025-10..2025-12). Eval reports five blocks:

### 3.1 Primary metric (decision-aligned)

**`mean_top50_proximity_excess`**: average actual `y_t` (computed from realized future returns) across the model's daily top-50 picks, over the eval window.

This is exactly the production decision: pick 50 stocks by score, hold for the proximity-weighted window, measure realized excess. Higher = the model's "high score" claim is real money.

### 3.2 Ranking quality

**`spearman(predicted_score, actual_y)`** across all (date, stock) pairs in eval window. Catches calibration-blind cases where top-K is fine but middle-of-distribution is noisy.

### 3.3 Hit-rate decomposition

For diagnostic only — not optimization target:
- `top50_T1_hit_rate`: fraction of top-50 picks that had any positive excess return on T+1
- `top50_T13_hit_rate`: fraction with any positive excess in T+1..T+3
- `top50_T1_avg_excess`: mean T+1 excess return over top-50 (unweighted)

These tell us where the model is paying its bet (T+1-heavy vs T+3-heavy vs hit-rate-heavy).

### 3.4 Cross-comparison with paris's labels

Sanity only:
- `PR_AUC vs labels_A_t3`: same metric paris reported (0.110 H1, 0.146 H2). New model should match or beat.
- `top1pct_lift vs labels_A_t3`

### 3.5 Calibration

- `ECE_10bin(ŷ vs actual_y)`: expected calibration error binned to 10 quantiles of predictions
- Reported per-window. Used downstream for isotonic calibration on H1, eval on H2.

---

## 4. Path roadmap

User-confirmed sequence: **1 → 4 → 2 → 3**.

```
[Path 1] LightGBM β-regression baseline
    │   today, 4-8h CPU
    ▼
[Path 4] Feature engineering (cross-sectional rank-z + outlier sanitize)
    │   tomorrow, 1-2 day
    ▼
[Path 2] Multi-model-class ensemble (CatBoost + XGBoost + stacking)
    │   day after, 2-3 day
    ▼
[Path 3] Tabular DL (TabNet / FT-Transformer) — conditional on prior paths
    │   if needed, ~1 day with 4070 GPU
```

---

### 4.1 Path 1 — LightGBM β-regression baseline (tonight)

**Build target**: compute `y_t` per (date, stock) using realized_returns self-shift. Output `data/p3_4070/target_y.parquet` (cached).

**Train**: LightGBM regression with `objective: regression_l2`, on 344 features × y_t. Window: TRAIN_EFF (2023-01-03 to 2024-12-04).

**Grid (~36 runs)**:
- Hyperparams: `num_leaves ∈ {31, 63, 127}`, `lr ∈ {0.03, 0.05}`, `min_data_in_leaf ∈ {50, 100}` = 12 configs
- Seeds: 3 (42, 43, 44)
- Total: 36 runs, expected ~3-5 min each on 36-core CPU

**Eval**: VAL_EFF for early stopping; H1 + H2 for reporting.

**Ensemble**: top-3 configs by VAL_EFF mean_top50_proximity_excess, seed-mean across them.

**Calibration**: isotonic on H1 → apply to H2 score, re-eval all metrics on calibrated score.

**Deliverables**:
- `runs/sl_path1/<config>_seed<s>/results.json` — per-run metrics
- `runs/sl_path1/ensemble.json` — ensemble eval on H1, H2
- `runs/sl_path1/predictions.parquet` — (trade_date, ts_code, score_raw, score_calibrated)

**Go/No-Go to Path 4**: ensemble's H1 `mean_top50_proximity_excess` > paris baseline (computed on same window).

### 4.2 Path 4 — Feature engineering (tomorrow)

**Cross-sectional rank-z per day** (the single highest-value feature engineering on Chinese alpha factors, missing in current bundle):

For each `trade_date` and each feature column, replace raw value with `(rank_within_day - 0.5) / n_in_universe_that_day`, then z-score per-stock across the time series. Eliminates per-day distribution shift and outlier dominance.

**Outlier audit + clip**:
- Audit all 344 features for `|x| > FP16_MAX` and document which factor formula is overflowing (gtja_017 = 3.4e38 is the worst, suggests divide-by-zero handling broken)
- Clip to per-feature [p1, p99] computed on TRAIN_EFF only (no leak)
- Rebuild `feature_panel_v3_344_clean.parquet`

**Rerun Path 1** on cleaned features. Compare lift directly.

**Deliverables**:
- `data/p3_4070/feature_panel_v3_344_clean.parquet` (gitignored)
- `data/p3_4070/feature_audit.md` — per-feature pre/post stats (for paris's record on which formulas need fixing upstream)
- `runs/sl_path4/` — same structure as Path 1, swapped panel

**Go/No-Go to Path 2**: clean features show ≥ +0.001 absolute lift on `mean_top50_proximity_excess`.

### 4.3 Path 2 — Multi-model-class ensemble (day after)

**Add CatBoost**: same target, same windows. CatBoost's gradient-bias handling differs from LightGBM, providing model-class diversity.

**Add XGBoost**: same target. XGBoost similar to LightGBM but enough algo difference (split selection, regularization) to add ~5-10% prediction diversity.

**Stacking**: train a small meta-learner (LightGBM with 16 leaves, 100 iters) on a held-out 2024-H2 OOF window using each base model's predictions as input features. Meta-learner outputs final score.

**Why stacking over rank-mean**: rank-mean assumes uniform contribution across base models; stacking can learn to lean on a base model that's strong specifically in market regime X.

**Deliverables**:
- `runs/sl_path2/<class>_<seed>/` per base model
- `runs/sl_path2/stacking_meta/results.json`
- `runs/sl_path2/predictions.parquet` — final calibrated score

**Go/No-Go to Path 3**: stacked ensemble beats Path 4 ensemble by ≥ +0.0005 absolute on H1 `mean_top50_proximity_excess` (small bar — model-class diversity often gives 5-10 bps improvement).

### 4.4 Path 3 — Tabular DL (conditional)

**Trigger**: only if Paths 1+4+2 plateau and there's appetite for higher risk-reward.

**Try TabNet** first (~12h on 4070): native tabular, attention-based, has structured inductive bias.

**Try FT-Transformer** if TabNet doesn't help (~12h): pure attention, more flexible but more variance.

**Stop criterion**: if first model class shows ≥ +0.0005 H1 improvement, ensemble it; else abandon Path 3.

**Deliverables**: `runs/sl_path3/`, including model ckpt for ONNX export.

---

## 5. Production deliverable

End state: a single `runs/sl_final/predictions.parquet` with columns `(trade_date, ts_code, score_calibrated)`. Plus:

- `model_bundle.zip`: best ensemble's component models (LightGBM .txt × N, CatBoost .cbm × N, etc.) + stacking meta + isotonic
- `INFER.md`: inference recipe (load → predict → calibrate → top-K)
- `RESULTS.md`: full eval report (all metrics in §3, per-path lift breakdown, recommendation)

These get uploaded to `oss://ledashi-oss/fromsz/handoffs/2026-05-XX-sl-ensemble-results/` for paris (= user) audit.

## 6. Out of scope (deferred)

- **Portfolio sizing / Kelly**: this plan produces SCORES, not position sizes. Sizing is a separate decision layer.
- **Industry / sector caps**: top-K with sector caps adds constraint but is decision-layer not scoring.
- **Backtesting with cost / impact**: need execution simulator (slippage, fee model). Out of scope; eval here uses gross returns.
- **Live data pipeline**: this is offline batch training; live inference plumbing is separate.
- **Real-time / intraday**: P3 panel is daily. Higher frequency requires different data pipeline.

## 7. Resource budget

- **CPU**: 36 logical cores idle. LightGBM/CatBoost/XGBoost all parallelize to use them.
- **GPU**: 4070 12.9 GB free. Used only by Path 3 (TabNet etc.).
- **RAM**: 68 GB total, ~32 GB usually free; bundle cache uses ~15 GB.
- **Disk**: 290 GB free; cache + intermediate predictions consume <5 GB.
- **Wall-clock**: Path 1 ~6h, Path 4 ~12h, Path 2 ~36h, Path 3 ~24h. Total ~3-4 days for full pipeline.

## 8. Decision log (so far)

- 2026-05-09 — Drop residual PPO. P3 architecture mismatched task (5517-dim action × scalar reward). Codex-style critique validated by experiment.
- 2026-05-09 — Adopt proximity-weighted continuous target instead of paris's binary labels. Better aligns with "earthquake" goal.
- 2026-05-09 — Primary eval metric = `mean_top50_proximity_excess` (decision-aligned), not PR_AUC.
- 2026-05-09 — Path order 1 → 4 → 2 → 3 confirmed by user.

## 9. Risks / open questions

- **R1 Target stability**: continuous regression on noisy daily returns may have higher variance than binary classification. Mitigation: large window TRAIN_EFF (~2 years), early stopping on VAL_EFF.
- **R2 Calibration breaks across regimes**: H1 isotonic may not generalize to H2 (different market regime). Mitigation: report both raw and calibrated scores; if calibrated H2 worse than raw, fall back.
- **R3 max(0, excess) loses sell-side info**: if a stock has -5% excess return today, that's signal. Currently we treat as zero. Acceptable for long-only top-K; reconsider for long-short variants.
- **R4 Feature audit might miss interaction effects**: cleaning per-feature doesn't address bad interaction terms. Mitigation: SHAP analysis on Path 4 model to surface unexpected feature importance ranks.

## 10. Versioning

This is `v1` of the SL plan. If Path 1 results suggest fundamental redesign (e.g., target weights need different ratios), update to `v2` with explicit changelog.
