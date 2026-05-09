# Main-Wave Label Ablation — SPEC v1

**Owner**: Claude Code (paris)
**Date**: 2026-05-09
**Status**: SPEC LOCK — 仅文档，禁止此版本之外的代码 / 实验 / 模型训练
**Budget**: 9 h（SPEC 1.5h + Stage 0..4 7.5h）

---

## 0. 背景与目的

26F-v3 PPO 已确认：
- median 2.27× lift（v2 + 0.12），policy 实际不收敛（entropy 满，log_std 不动）
- `episode_reward_mean` 全程 0.0：Phase 23 的 `MainWaveEpisode` 二值标签太稀疏

**生产主线退出 PPO**，改成「**盘后 / 全主板 / 每只股票 T+? 主升浪概率**」的监督学习产品。
本 SPEC 只做这个产品的**第一步：选 P0 标签函数**，输出 1 个主标签 + 0–1 个辅助标签 + 锁死的实现。

不在本 SPEC 范围：
- ❌ `wave_scores_daily` 表 / Alembic / API → P1 另起 PR
- ❌ Walk-forward 月度重训 → P3
- ❌ Multi-task head（共享 booster） → P0 决出后再说
- ❌ RL / PPO 任何修改

---

## 1. 任务口径（消除原 audit #1 + 新约束 #1）

### 1.1 事件原语（event primitive）

每个 labeling 方法 **M ∈ {A, B, C, D}** 必须先生成事件，再派生标签。事件的 schema 统一：

```
Event(
    ts_code:        str,
    event_start:    date,    # 事件首日（决策日 = event_start - 1）
    event_peak:     date,    # 峰值日
    event_quality:  float,   # 方法内 score（用于阈值化 / 校准）
    event_method:   str,     # 'A' | 'B' | 'C' | 'D'
)
```

**事件去重**（新约束 #1）：同一只股票的事件**互不重叠**：
- 任意两个事件 `e1, e2` 满足 `e1.event_peak < e2.event_start` 或反之
- 若候选事件重叠，保留 `event_quality` 更高的，丢弃另一个
- 新事件起点必须 ≥ 上一事件 `event_peak + 1`（**min_gap = 0**，但通过 peak 强制隔开）

### 1.2 决策日 → 标签派生

决策日 `t` = 事件首日的前一交易日 `event_start - 1`。一个事件 e 在三个 horizon 下分别贡献：

```
y_t1[t, j]  = 1  iff  ∃ event e_j with event_start = t + 1                 # 严格 T-1
y_t3[t, j]  = 1  iff  ∃ event e_j with event_start ∈ {t+1, t+2, t+3}      # 宽松 T-1..T-3
y_e20[t, j] = 1  iff  ∃ event e_j with event_start ∈ {t+1, ..., t+20}     # 20 日内
```

**P0 主目标 = `y_t3`**（用户决策：纯 T-1 太稀，先做 t3，t1/e20 留辅助/对照）。
- t1 也跑出来评估，但**不作为 P0 选择标准**
- e20 跑出来作上限对照（"如果 horizon 任意 20 日，能多准"）

### 1.3 不直接从 (t, j) 扫前向窗口

**禁止**直接：
```python
# ❌ 错误，会把同一段上涨的多个 t 都标为正
y[t, j] = 1 if any_strong_trend(close[t+1:t+21]) else 0
```

**必须**：
```python
events = method_M.generate_events(close_full, ...)
events = deduplicate(events)
y_t1, y_t3, y_e20 = derive_labels_from_events(events)
```

---

## 2. 五个数学定义（A 你的 + B/C/D 我的；E 抽样）

### A — Excess + Vol-Adaptive（用户原 v2，连续化）

事件检测算法（per-stock 单趟扫描）：
```
for t in trading_days[stock_j]:
    if not universe_mask[t, j]: continue
    if last_event.event_peak >= t: continue   # 去重
    
    # inflection: t 必须是上行起点
    today_gain = adj_close[t] / adj_close[t-1] - 1
    if today_gain < 0.005: continue
    
    # 前 5 日累计涨幅 ≤ 3%（确保 t 是相变点）
    prior_gain = adj_close[t-1] / adj_close[t-6] - 1
    if prior_gain > 0.03: continue
    
    # 前向 [t, t+20] 找峰值
    peak_offset = argmax(adj_close[t:t+21])
    if peak_offset < 3 or peak_offset > 20: continue
    
    fwd_max_excess = (adj_close[t+peak_offset] / adj_close[t]) - 1
                     - (benchmark[t+peak_offset] / benchmark[t] - 1)
    
    vol20 = ewm_std(pct_chg[t-21:t-1], halflife=10)
    adaptive_thr = max(0.06, 1.8 * vol20)
    if fwd_max_excess < adaptive_thr: continue
    
    max_dd = max running drawdown in [t, t+peak_offset]
    if max_dd > 0.02 + 0.5 * fwd_max_excess: continue
    
    if amount_ma20[t, j] < 1e8: continue
    
    yield Event(
        ts_code=j,
        event_start=t,
        event_peak=t+peak_offset,
        event_quality=fwd_max_excess / adaptive_thr  # 连续 score, ≥1 是正事件
    )
```

`event_quality_A = fwd_max_excess / adaptive_thr` 是**连续值**：
- 阈值固定版 `mask_A_fixed`：所有 audit 通过的事件都计为正（即 `event_quality ≥ 1`）
- 阈值校准版 `mask_A_calib`：用 train 上的 `event_quality` 99% 分位作 τ_A，**只有 quality ≥ τ_A 的事件计为正**

两版**同时**报，决策池里只用 `mask_A_calib`（保证和 B/C/D 公平比较）；`mask_A_fixed` 写进 RESULTS 作为「用户原始定义」对照。

### B — Trend-Scanning（López de Prado, 2020）

事件检测：先求每个 (t, j) 的 best forward t-stat：
```
for L in {5, 10, 15, 20}:
    fit OLS:  log(adj_close[t : t+L]) ~ β₀ + β₁ · k,   k=0..L-1
    t_stat(L) = β₁ / SE(β₁)
t_stat_max[t, j], L_star[t, j] = argmax_L (|t_stat(L)|, t_stat(L))
```

事件起点 = 满足以下三条的最小 t：
1. `t_stat_max[t, j] ≥ τ_B`（阈值校准在 train 上）
2. `t > last_event.event_peak[j]`
3. `signed slope > 0`

事件 peak = `t + L_star[t, j]`。
`event_quality_B = t_stat_max[t, j]`（连续）。

### C — Triple-Barrier（López de Prado, 2018）

事件检测：每个 t 设三道屏障：
```
upper = adj_close[t] · (1 + 2.0 · σ_t)         # σ_t = ewm_std(pct_chg, hl=10) at t
lower = adj_close[t] · (1 - 2.0 · σ_t)
vert  = t + 20

label = which barrier hits first in adj_close[t+1 : t+21]
        +1 if upper, -1 if lower, 0 if vert
```

事件起点 = 满足以下条件的 t：
1. label = +1（先触发上轨）
2. `t > last_event.event_peak[j]`

事件 peak = first day k with `adj_close[t+k] ≥ upper`。
`event_quality_C = (adj_close[event_peak] - adj_close[t]) / (adj_close[t] · σ_t)`（实际超额触达倍数，连续）。

### D — Directional Change（Glattfelder & Tsang, 2011）

预先在每只股票的 `adj_close` 上跑多尺度 DC：
```
for θ in {0.03, 0.05, 0.08}:
    runs through close, marking up-events and down-events
    when |close[t] - last_extreme| / last_extreme >= θ:
        emit DC event with start = last_extreme_date
```

事件起点 = 最小 θ 触发的 up-DC 的 start_date。
事件 peak = 触达后的下一次反向 DC 的 start_date（即 `last_extreme` 在反转前的那天）。
`event_quality_D = magnitude / θ_min`，magnitude = peak_return at θ=0.03 path（连续）。

去重：同一只股票多 θ 路径下重叠事件 → 保留 quality 最高。

### E — L1 Trend Filter（Kim/Boyd, 2009）— **抽样**

```
solve  min ||close - x||² + λ · ||D² x||₁     # OSQP
λ via BIC selection
event = positive-slope segment with mean_slope >= τ_E  AND  duration >= 5d
```

**E 不进 P0 决策池**。在 Buffer 时间段（≤ 0.5h）跑 200 票随机抽样验证可行性，结果写 RESULTS.md「future work」节。

---

## 3. Universe 协议（消除原 audit #6 + 新约束 #4）

### 3.1 输入表

| 表 | 用途 |
|---|---|
| `daily_quotes` | OHLCV + adj_factor + amount + volume |
| `stock_info` | list_date / delist_date / 代码主板正则 |
| `stock_st` | **历史 ST 状态**（is_st=true 的当日记录） |
| `suspend_d` | 全日停牌（type='S' AND timing IS NULL） |

**禁止使用** `stock_info.is_st`（current state，会泄漏未来 ST/de-ST 转换）。
**禁止使用** `stock_info.is_suspended`（同理）。

### 3.2 逐日 universe SQL（锁定）

```sql
-- 单日 universe mask
SELECT
    q.stock_code,
    q.trade_date,
    -- core data presence
    (q.close IS NOT NULL AND q.amount IS NOT NULL
     AND q.volume IS NOT NULL AND q.volume > 0
     AND q.adj_factor IS NOT NULL) AS data_ok,
    -- main board
    (q.stock_code ~ '^60[0135][0-9]{3}\.SH'
     OR q.stock_code ~ '^00[0123][0-9]{3}\.SZ') AS main_board,
    -- listed >= 60d
    (si.list_date <= q.trade_date - 60) AS listed,
    -- not delisted
    (si.delist_date IS NULL OR si.delist_date > q.trade_date) AS not_delisted,
    -- not ST that day (historical, NOT current)
    NOT EXISTS (
        SELECT 1 FROM stock_st st
        WHERE st.stock_code = q.stock_code
          AND st.trade_date = q.trade_date
          AND st.is_st = true
    ) AS not_st,
    -- not all-day suspended that day
    NOT EXISTS (
        SELECT 1 FROM suspend_d sd
        WHERE sd.stock_code = q.stock_code
          AND sd.trade_date = q.trade_date
          AND sd.suspend_type = 'S'
          AND sd.suspend_timing IS NULL
    ) AS not_suspended
FROM daily_quotes q
JOIN stock_info si USING (stock_code)
WHERE q.trade_date BETWEEN :start AND :end
```

`universe_mask[t, j] = data_ok AND main_board AND listed AND not_delisted AND not_st AND not_suspended`

### 3.3 真实测试快照（Stage 0 必须通过）

| trade_date | 期望 universe 大小（±5%） |
|---|---|
| 2023-06-15 | 2966 ± 150 |
| 2024-06-14 | 3010 ± 150 |
| 2025-06-13 | 3027 ± 150 |

---

## 4. 复权 / 流动性 / 可交易性（新约束 #5）

| 用途 | 字段 | 计算 |
|---|---|---|
| **趋势 / 收益 / 回撤** | `adj_close` | `daily_quotes.close * daily_quotes.adj_factor` |
| **流动性 gate** | `amount_ma20` | rolling mean of **raw** `daily_quotes.amount`，window=20 |
| **可交易性** | `universe_mask`（§3.2） | raw daily_quotes + suspend_d + stock_st |
| **波动率** | `vol20` | ewm_std of `pct_change`（**raw** pct_change，已含除权除息冲击）halflife=10 |
| **基准** | `benchmark` | 主板等权 `adj_close` 日度（每日重平衡，universe ∩ stock 集合） |

`pct_change` 用 raw 值还是 adj 值？Tushare daily_quotes.pct_change 是**前复权后的真实涨跌**（已扣除分红除权），用 raw 即可。

---

## 5. 阈值校准（消除原 audit #3 + 新约束 #3）

### 5.1 切分（带 embargo，新约束 #2）

```
train_full:  2023-01-03 .. 2024-12-31
train_eff:   train_full minus last 20 trading days  →  2023-01-03 .. 2024-12-04
val_full:    2025-01-01 .. 2025-06-30
val_eff:     val_full minus last 20 trading days   →  2025-01-01 .. 2025-06-04
test:        2025-07-01 .. 2025-12-31
```

`train_eff` 上**算事件、校准阈值、训 LightGBM**。
`val_eff` 上做 isotonic 校准 + early stopping。
`test` 单次评估，不许回头调参。

### 5.2 阈值搜索（不用 quantile）

每个方法 M ∈ {A_calib, B, C, D} 在 train_eff 上：
```
for τ in linspace(τ_min(M), τ_max(M), 50):
    mask = events.event_quality >= τ
    pos_rate = mean(mask over train_eff decision cells)
    if 0.005 <= pos_rate <= 0.012:
        candidate τ
choose τ minimizing |pos_rate - 0.008|       # 目标 0.8%（介于 0.5%~1.2%）
```

τ_M 锁定后，对 val_eff 和 test **不再调整**。

A_calib 的 score = `event_quality_A`，τ_A 同样按上面方法搜。
A_fixed 用论文/用户原始版本，无 τ。

---

## 6. Feature panel 锁定（新约束 #6）

**P0 唯一允许的 feature 来源**：

```yaml
feature_panel:
  base:
    path: data/combined_panels/factor_panel_combined_short_2023_2026.parquet
    schema_check: 必须含 (ts_code, trade_date) + 23a_clean 列表的 336 列
  tech_events:
    path: data/duckdb/factor_eval/tech_event_panel/year=*.parquet
    schema_check: 8 列 tech_evt_*
  include_columns:
    file: handoffs/2026-05-08-full-data-audit/include_columns_23a_clean.txt
    expected_cols: 336
  total_factor_count: 344        # 336 + 8 tech events
  date_range: [2023-01-03, 2026-04-30]
  join_keys: [ts_code, trade_date]
  schema_hash: sha256(sorted(feature_names))[:12]   # 在 SPEC commit 时算并锁进 SPEC v1.1
```

LightGBM 训练前 **必须**：
1. Read `include_columns_23a_clean.txt` → assert len == 336
2. Read tech_event_panel → assert 8 cols 全在
3. Inner join base + tech_event_panel on (ts_code, trade_date)
4. Filter to universe_mask = True 的 (t, j) 对
5. Compute `feature_schema_hash = sha256(sorted(feature_cols))[:12]` 并写入 model metadata

如果数据真相和上面不符（如 base panel 缺列），**Stage 2 立即 abort，写报告解释，等用户决定**。

---

## 7. LightGBM 环境与 fallback（新约束 #7）

### 7.1 安装

```bash
source .venv/bin/activate
pip install lightgbm==4.5.0
```

Stage 2 启动前 first command 必须验证：
```python
import lightgbm
assert lightgbm.__version__.startswith('4.')
```

### 7.2 Fallback

如果 `pip install lightgbm` 失败（编译问题等），fallback 顺序：
1. `pip install lightgbm --prefer-binary`
2. `xgboost==2.1.0` + 同等参数
3. **不允许** sklearn `GradientBoostingClassifier`（10× 慢，结论不可比）

### 7.3 LGBM 配置（锁定）

```python
LGBM_PARAMS = dict(
    objective='binary',
    metric='average_precision',     # PR-AUC
    learning_rate=0.02,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=200,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    is_unbalance=True,
    n_estimators=2000,
    verbose=-1,
    n_jobs=3,                       # CLAUDE.md max=3
)
EARLY_STOPPING = 80
```

Calibration: `sklearn.isotonic.IsotonicRegression(out_of_bounds='clip')` on val_eff predictions.

---

## 8. 评估指标（消除原 audit #2 + #8 + 新约束 #7）

### 8.1 Stage 1 — 纯度（label 自身的质量）

每个 `(label_id, horizon, year)` 输出：

| 指标 | 公式 | 意义 |
|---|---|---|
| `positive_rate` | `mean(mask)` | 信号稀疏度 |
| `median_fwd20_excess` | 正 cell 上 `fwd_max_excess` 中位数 | 标签平均含金量 |
| `spearman_ic_train` | `spearmanr(score_train, fwd_max_excess_train)` | 仅 train 上算 |
| `top1pct_purity_lift` | top-1% score / global pos_rate | 高分段是否更准 |
| `industry_concentration` | `max_industry_rate / mean_industry_rate`（**新公式**） | 越小越均匀，门槛 ≤ 2.0 |
| `industry_cv` | `std_industry_rate / mean_industry_rate` | 门槛 ≤ 1.0 |
| `year_stability_cv` | `std(pos_rate over 2023/2024) / mean(...)` | 训练期年度方差 |
| `mean_holding_days` | 正 cell 上 `event_peak - event_start` 中位 | 直观 |

**通过 Stage 1 门槛**（任一不达即剔除）：
- `0.003 ≤ positive_rate ≤ 0.015`
- `industry_concentration ≤ 2.0`
- `industry_cv ≤ 1.0`
- `year_stability_cv ≤ 0.5`
- `top1pct_purity_lift ≥ 5×`

### 8.2 Stage 2 — 可预测性（factor → label 是否能学）

每个进 Stage 2 的标签训一个 LightGBM，评估在 **test (2025-07..2025-12)** 上：

| 指标 | 公式 | 门槛 |
|---|---|---|
| `PR_AUC` | `average_precision_score(y_test, p_test)` | `≥ 5 × y_test.mean()` |
| `Brier` | `brier_score_loss` | `≤ 0.95 × null_brier` |
| `ECE_10bin` | 10 bin reliability error | `≤ 0.02` |
| `top1pct_model_lift` | top 1% predicted / base | `≥ 8×` |
| `top5pct_model_lift` | top 5% predicted / base | `≥ 4×` |
| `daily_precision@5` | 每日按 score 排前 5 命中率 | `≥ 0.10` |

### 8.3 Null tests（新约束 #7 — 替换原 #2 的「平移 1 天」启发式）

每个 LGBM 同时跑两个 null：

1. **label-shuffle null**: `y_train_shuffled = np.random.permutation(y_train)`，重训。期望 `PR_AUC ≈ y_test.mean()`（即 0.005~0.012）。如果显著高于，说明 train/test 之间有时间结构泄漏。
2. **date-shuffle null**: `X_train` 按 date 内 shuffle ts_code（破坏截面结构），重训。期望 PR_AUC 大幅下降到 base rate 附近。

**未来信息哨兵（不进生产，仅诊断）**：
3. **future-feature upper bound**: 用 `X_train[t+1]` 训练对 `y_train[t]`。这是物理上不可能的，但作为"特征如果包含未来信息能多准"的天花板。如果生产模型 PR_AUC 已经接近这个上限，强烈怀疑标签或特征有泄漏。

---

## 9. Composite & 软组合（消除原 audit #5）

不跑 hard AND 进 P0 决策（原 audit #5 已证伪）。

Stage 3 只构造 **composite z-score**：
```
对 Stage 2 PR-AUC top-2 的两个方法 M1, M2：
z_M = (event_quality_M - mean_train) / std_train       # 在 train_eff 上的 z-score
score_composite_mean = 0.5 * z_M1 + 0.5 * z_M2
score_composite_min  = min(z_M1, z_M2)

τ_composite 同 §5.2 搜索到 pos_rate ≈ 0.8%
```

Composite 也跑一次 LightGBM，看 PR-AUC 是否 ≥ 单方法。

**hard AND 仅在 RESULTS.md 出现一次**：作为反例佐证「solo 0.8% AND solo 0.8% = 0.05% 不可训练」。

---

## 10. 决策矩阵（Stage 4）

P0 主标签从 `{A_calib, B, C, D, composite_mean, composite_min}` 中选，按以下加权得分：

```
score(M) = 0.45 × normalize(PR_AUC_test)
         + 0.20 × (1 - ECE_10bin)
         + 0.15 × normalize(top1pct_model_lift)
         + 0.10 × (1 - industry_cv)
         + 0.10 × (1 - year_stability_cv)
```

最高分胜出。如果差距 < 0.03，选 `industry_cv` 更小（更稳定）的。

辅助标签 = 第二名，**仅在它 horizon ≠ 主标签 horizon**（如主 t3 + 辅 t1）时保留；否则不要辅助。

---

## 11. 输出物清单

```
handoffs/2026-05-09-wave-label-ablation/
  SPEC.md                        # 本文件 v1
  RESULTS.md                     # Stage 1-4 结果与 P0 决策（Stage 4 完成后写）

src/aurumq/labeling/             # 全部新增
  __init__.py
  universe.py                    # Stage 0
  v2_excess_adaptive.py          # A
  trend_scanning.py              # B
  triple_barrier.py              # C
  directional_change.py          # D
  l1_trend_filter.py             # E（仅 Buffer 时段抽样）
  events.py                      # Event dataclass + dedupe + derive_labels
  benchmark.py                   # 主板等权
  thresholds.py                  # 阈值搜索
  p0_chosen.py                   # Stage 4 决出后的最终 wrapper

tests/labeling/
  test_universe.py
  test_v2_excess_adaptive.py
  test_trend_scanning.py
  test_triple_barrier.py
  test_directional_change.py
  test_events_dedupe.py
  test_thresholds.py

scripts/
  run_label_ablation.py          # 主驱动脚本

data/duckdb/labeling/            # 新增
  universe_mask_year=*.parquet
  events_{A,B,C,D}_year=*.parquet
  labels_{A,B,C,D}_{t1,t3,e20}_year=*.parquet
  benchmark_main_board_eq_weighted.parquet

results/                         # 新增
  purity_train_2023_24.csv
  purity_test_2025.csv
  learnability.csv
  composite.csv
  null_tests.csv
  plots/
    calibration_<label>.png
    industry_heatmap_<label>.png
    pr_curve_<label>.png

models/
  lgbm_label_<id>_t3/
    model.txt
    isotonic.pkl
    feature_schema.json
    metrics.json
```

---

## 12. 时间预算（9h）

| Stage | 任务 | 预算 |
|---|---|---|
| **SPEC** | 本文件 + commit | 1.5h ✓ |
| **0** | universe.py + benchmark.py + thresholds.py + tests + 跑一遍 mask | 1.5h |
| **1** | A-D 4 个方法实现 + 事件去重 + 派生标签 + 纯度评估 | 3.0h |
| **1.5** | LightGBM 安装 + 训 4 个 t3 model + null tests + 校准 | 2.5h |
| **3-4** | composite + RESULTS.md + p0_chosen.py | 0.5h |
| **Buffer** | E 抽样 / 漂移修复 | 0.5h |
| 合计 |  | 9.5h（10% 缓冲） |

**硬截断**：Stage 1 超 4h 仍未完成 → 砍 D（DC）方法，只对比 A/B/C；记入 RESULTS limitations。
**硬截断**：Stage 2 超 3h 仍未完成 → 只跑 A_calib + B + C 三个，砍 D + composite；记入 RESULTS。

---

## 13. Audit 闭环索引

| Audit 项 | SPEC 节 | 处理方式 |
|---|---|---|
| 原 #1 T-1 vs e20 不分 | §1 | event_start primitive + t1/t3/e20 派生 |
| 原 #2 仅纯度 | §8.2 | Stage 2 LightGBM ablation |
| 原 #3 阈值泄漏 | §5.1 + §5.2 | train_eff 校准 + 锁定 |
| 原 #4 A 固定 | §2.A | A_fixed + A_calib 双版本 |
| 原 #5 hard AND 太稀 | §9 | 仅 composite，不跑 hard AND |
| 原 #6 survivorship | §3 | 逐日 universe，无 current state |
| 原 #7 L1 太乐观 | §2.E + §12 | 抽样仅 200 票，进 Buffer |
| 原 #8 industry 公式 | §8.1 | concentration / cv |
| 新 #1 event 去重 | §1.1 | Event dataclass + non-overlap |
| 新 #2 split embargo | §5.1 | 20-day embargo |
| 新 #3 阈值按 pos_rate 搜 | §5.2 | linspace + 目标 0.8% |
| 新 #4 ST 用 stock_st | §3.1 + §3.2 | 禁用 stock_info.is_st |
| 新 #5 复权 | §4 | adj_close 算 trend，raw amount 算流动性 |
| 新 #6 panel 路径锁定 | §6 | 真实路径 + schema_hash |
| 新 #7 LGBM env + null | §7 + §8.3 | install protocol + label/date shuffle |

---

## 14. 第一个 commit 验收标准（用户原文）

SPEC.md 必须能回答：

1. ✅ event_start 如何定义和去重 → §1.1
2. ✅ t1/t3/e20 如何从 event_start 派生 → §1.2
3. ✅ universe 是否逐日、无 current 状态泄漏 → §3
4. ✅ 阈值是否只用 train 校准 → §5.1, §5.2
5. ✅ split 是否有 20 日 embargo → §5.1
6. ✅ feature panel 是否本地真实可复现路径 → §6
7. ✅ LightGBM 环境和 fallback 是否明确 → §7

---

## 15. 立刻执行（SPEC commit 后）

按 §12 时间表，无人值守 9h 推进 Stage 0 → 1 → 2 → 3 → 4。
任何阻塞（PG 表 schema 偏离、lightgbm 装不上、panel 列数对不上）→ 立即停在该 Stage，写 BLOCKER.md，等待用户介入。

每个 Stage 完成生成一个 git commit，commit 标题：
```
feat(label-ablation): Stage <N> <short summary>
```

最终 commit 标题：
```
feat(label-ablation): P0 main wave label decision — <chosen_method>
```
