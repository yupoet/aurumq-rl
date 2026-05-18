## §12 持续研究记录 (Research Log)

Comprehensive synthesis: see `docs/RANKINGS_COMPREHENSIVE_v18.md` for full 1,473-cell ranking, sanity checks and production routing.

### §12.1 Paradigm 分类 (持续科研协议)

- **Paradigm 1 — Predictive Cross-Sectional**: `features(t) → y(t) = f(forward_returns over [t+1, t+K])`
  - Sub-direction A: Proximity continuous regression — matrix v10 (wave_v1..v4), v10b (target_y)
  - Sub-direction B: Binary dense classification (P75 ~25% pos) — matrix v10c
  - Sub-direction C: Binary sparse paris-style (0.8% pos) — matrix v11 methods A/B/C/D × t1/t3/t5
  - Sub-direction D: Algorithm diversity (CatBoost / XGBoost) — matrix v10d, v10e
- **Paradigm 2 — Event-Anchored Pattern Recognition**: events → pre-event window as positive → classifier
  - Sub-direction A: Anchor α/β classification at T1/T3/T5 — matrix v12

### §12.2 研究进度 (Research progress)

| matrix | paradigm | cells | universe × panel grid | bootstrap CI | status |
|---|---|---|---|---|---|
| v10  | P1 proximity reg  | 174  | 7×6 + 6 ES eval-only  | partial (in v10h) | shipped |
| v10b | P1 proximity reg  | 42   | 7×6 (target_y)        | partial (in v10h) | shipped |
| v10c | P1 binary dense   | 168  | 7×6×4 labels          | partial (in v10h) | shipped |
| v10d | P1 CatBoost       | 48   | 2 panels × 6 univ × 4 labels | partial (in v10h) | shipped (gap: 5 panels missing) |
| v10e | P1 XGBoost        | 48   | 2 panels × 6 univ × 4 labels | partial (in v10h) | shipped (gap: 5 panels missing) |
| v10h | bootstrap CI      | 207×4 | top cells from v10/v10c/v10d/v10e | itself | shipped |
| v11  | P1 binary sparse  | 504  | 7×6 × 4 methods × 3 horizons | **missing** | shipped (gap: no CI) |
| v12  | P2 anchor α/β     | 252 (147 valid + 105 skipped) | 7×6 × 2 specs × 3 anchors | **missing** | shipped (gap: no CI; sparse univ thinned) |

### §12.3 实证结论 (Empirical findings)

**Master ranking — top-10 production-deployable cells** (composite = H2_IC × Sharpe_NET × max(Q1_IC,0)):

| # | cell_id | paradigm | univ | panel | H2 fwd20 IC | Q1 fwd20 IC | Sharpe_NET K10 fwd20 |
|---|---|---|---|---|---|---|---|
| 1 | `target_y_HARD_TECH_v2_null` | p1-proximity-reg | HARD_TECH | v2_null | +6.60% | +10.68% | 2.46 |
| 2 | `target_y_HARD_TECH_ledashi` | p1-proximity-reg | HARD_TECH | ledashi | +6.29% | +10.82% | 2.39 |
| 3 | `target_y_HARD_TECH_r2a` | p1-proximity-reg | HARD_TECH | r2a | +6.17% | +9.97% | 2.53 |
| 4 | `binary_v4_HARD_TECH_v3unified` | p1-binary-dense | HARD_TECH | v3unified | +5.84% | +5.87% | 4.25 |
| 5 | `target_y_HARD_TECH_r2b` | p1-proximity-reg | HARD_TECH | r2b | +6.63% | +8.69% | 2.32 |
| 6 | `target_y_HARD_TECH_v2_no_phase_c` | p1-proximity-reg | HARD_TECH | v2_no_phase_c | +6.19% | +9.44% | 2.24 |
| 7 | `target_y_HARD_TECH_v3unified` | p1-proximity-reg | HARD_TECH | v3unified | +5.88% | +9.20% | 2.42 |
| 8 | `target_y_HARD_TECH_tier4_v2_old` | p1-proximity-reg | HARD_TECH | tier4_v2_old | +6.00% | +9.38% | 1.97 |
| 9 | `alpha_T3_HARD_TECH_ledashi` | p2-anchor | HARD_TECH | ledashi | +6.14% | +5.33% | 2.85 |
| 10 | `binary_v3_HARD_TECH_v2_null` | p1-binary-dense | HARD_TECH | v2_null | +3.92% | +5.24% | 4.04 |

**Per-universe production recommendation** (best cell by avg IC for the chosen horizon):

| universe | short (fwd5) best | mid (fwd10) best | long (fwd20) best |
|---|---|---|---|
| MAIN_BOARD | `v2_MAIN_BOARD_r2b` | `v2_MAIN_BOARD_r2b` | `v4_MAIN_BOARD_ledashi` |
| CSI500 | `binary_v2_CSI500_v2_null` | `binary_v2_CSI500_v2_null` | `catboost_v2_CSI500_ledashi` |
| CSI1000 | `v2_CSI1000_tier4_v2_old` | `binary_v4_CSI1000_tier4_v2_old` | `binary_v2_CSI1000_tier4_v2_old` |
| NPF | `target_y_NPF_v3unified` | `v2_NPF_r2a` | `v2_NPF_r2a` |
| NPF_FULL | `v2_NPF_FULL_v3unified` | `binary_v3_NPF_FULL_v3unified` | `binary_v4_NPF_FULL_v2_no_phase_c` |
| HARD_TECH | `binary_v3_HARD_TECH_ledashi` | `binary_v3_HARD_TECH_ledashi` | `binary_v4_HARD_TECH_v3unified` |

**Sanity check status (10 items, see report §9 for detail):**

1. PASS — Baseline v3_MAIN_BOARD_ledashi H2 fwd20 IC == +4.143%
2. PASS — Cost model: mean - mean_net == 0.20% (0.002)
3. PASS — Gross Sharpe > Net Sharpe (cost increases drag) for positive-return cell
4. PASS — Train window (2022-2024) ≠ Eval window (H1_2025..Q2_2026) — no overlap
5. PASS — Deterministic random_state=42 fixed in lgb_params
6. PASS — CSI500/CSI1000 are PIT (per-date membership) per CLAUDE.md universe table
7. PASS — Bootstrap CI 2.5% > 0 (K=50 fwd20) for ≥ 30% cells (v10h)
8. PASS — Bootstrap CI 2.5% > 0 (K=10 fwd20) for ≥ 20% cells (v10h)

**Headline empirical findings:**

- The strongest single-cell deployable signal is **`target_y_HARD_TECH_v2_null`** (paradigm `p1-proximity-reg`, panel `v2_null`, universe `HARD_TECH`) with H2_2025 fwd20 IC = **+6.60%** and Sharpe_NET K10 fwd20 = **2.46**, beating the baseline `v3_MAIN_BOARD_ledashi` (+4.14% IC).
- Paradigm 1 (cross-sectional prediction) dominates Paradigm 2 (anchor) on H2 fwd20 IC by ~0.41pp — anchor labels useful as meta-feature, not standalone.
- Bootstrap CI (v10h K=50 fwd20): 207/207 cells (100%) have CI 2.5% > 0 — production should preferentially deploy K=50 sizing for tail-control.
- LGB binary dense (v10c) has the highest **mean** composite score; LGB proximity continuous (v10) has the highest **peak** composite score. Both retained for production diversification.
- CSI500/CSI1000 cells (PIT membership) are the safest universes; HARD_TECH and NPF cells need ≥ 1pp differential vs baseline to claim improvement (IC SE ≈ 0.018).
- **Gap**: v11/v12 lack bootstrap CI; v10d/v10e only cover 2 panels of 7. Production routing on those cells should be flagged as 'preliminary'.

**Visualisations** (saved to `docs/figures/`):

- `fig01_top20_overall_bar.png` — Top-20 cells overall
- `fig02_panel_universe_heatmap.png` — Panel × universe × paradigm IC heatmaps
- `fig03_horizon_scaling.png` — IC vs forward horizon, per paradigm
- `fig04_dyn_exit_ranking.png` — Top-5 cells per dyn-exit trigger
- `fig05_paradigm_compare_scatter.png` — H2 IC vs Q1 IC scatter, by paradigm
- `fig06_bootstrap_ci_distribution.png` — Bootstrap CI lower-bound histograms

**Papers in pipeline (from this evidence)**:

1. *Cross-sectional alpha decomposition by regime in A-share markets* — panel × regime interaction (v10/v10c × H1/H2/Q1).
2. *Regression vs binary classifier choice in proximity-weighted forecasting* — v10 vs v10c head-to-head.
3. *Adaptive exit triggers in factor-based portfolios* — 11 dyn-exit triggers × universe routing.
4. *Paradigm 1 vs Paradigm 2 in stock selection* — v10/v10c vs v12 anchor comparison.
5. *Bootstrap-validated portfolio sizing in A-share quant signals* — v10h K=10 vs K=50 sizing-Sharpe analysis.
