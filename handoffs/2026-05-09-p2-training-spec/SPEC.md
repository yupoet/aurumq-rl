# P2 — Training Methodology Adjustment SPEC

**Owner**: Claude Code (paris) ↔ ledashi (RL side)
**Date**: 2026-05-09
**Status**: SPEC v1 — 仅文档，禁止此版本之外的代码 / 实验 / 模型部署
**Predecessor**: P1 (commit `bf67910` AurumQ + `7131a1c` AurumQ-RL)

---

## 0. Why P2 — what's wrong with P1's training

P1 落了完整管线但 **训练本身不够 production-grade**：

| 缺陷 | 现状 | 为什么不能上线 |
|---|---|---|
| **训练性质** | Stage 2 的 LGBM 是 **label ablation 训练**（4 method × t3 选 winner） | 不是为生产训练的，单 seed，no walk-forward |
| **CV** | 一刀切 train_eff / val_eff / test (固定 2023-24 / 25H1 / 25H2) | 无月度滚动，2026 年再训不知道用什么数据 |
| **多 seed** | 只跑了 1 seed | LGBM 单 seed 方差大，PR_AUC ±0.02 都正常 |
| **Multi-head** | 只有 t3，t1 / e20 / continue_5d / expected_peak 占位 NULL | RESULTS §6 说要补，未补 |
| **Composite** | labels 落盘但 LGBM 训练超时 kill | 是否真的能 +0.005~0.01 PR_AUC 未验证 |
| **特征 panel 日更** | `feature_panel_v3_344.parquet` 是 2026-05-09 一次性快照 | 18:45 cron 拉到的是过期数据，明天就跑空 |
| **标签泄漏哨兵** | null tests 通过但临界 (date-shuffle 1.49× / 阈值 1.5×) | 需要把 future-feature upper bound 也跑一遍当天花板 |
| **PPO 路线** | 已退到研究分支但未真正终止 | ledashi 的 4070 还在跑 26F-v3 PPO，资源浪费 |
| **生产模型版本管理** | `model.txt` 单文件，无 versioning | 重训覆盖会失去回滚能力 |

**P2 的核心目标**：把 ablation 模型升级为 **production model**，跑通 daily panel 日更 + 月度滚动重训 + 多 seed 集成，让 18:45 写出来的 `wave_scores_daily` 是真实可用的概率，不是 demo 行。

---

## 1. 三个训练方法选项 — 决策矩阵

### A. Supervised retrain (LightGBM/CatBoost ensemble)
**保留** P1 的 LightGBM 路线，做 production-grade 升级。

- 输入: 26F-v3 panel (348 cols) + Method A_t3 标签 (P0 winner)
- 训练: walk-forward 月度滚动 + 5 seed 集成
- 输出: 单一概率 `p_t3_start ∈ [0,1]`
- 优势: 已有 baseline 0.122 PR_AUC, 工程链条已通, 4-6 小时可落地
- 劣势: PR_AUC 上限可能在 0.13-0.15, alpha 信号不够强

### B. Multi-head extension
**扩展** A 路线为 4 头联合训练。

- 头 1: `p_t1_start` (严格 T-1, 最稀但最准)
- 头 2: `p_t3_start` (P0 主头)
- 头 3: `p_continue_5d` (启动后是否继续, 仓位 sizing)
- 头 4: `expected_peak_return` (regression head, 期望回报)
- 训练: 单 LGBM `objective='multiclass'` × 3 + 单独 regressor，或 4 个独立模型
- 优势: 给前端更丰富的信号; multi-task regularization 可能提升 PR_AUC
- 劣势: 代码复杂度 1.5×, 预算 +50%

### C. Hybrid — supervised baseline + PPO residual
**新增** RL 研究分支，PPO 学残差不学 raw policy。

- 阶段 1: A 路线产出 `p_t3_start_baseline`
- 阶段 2: PPO env 的 reward = `realized_excess_return`，state = (factors, baseline_p)
- 阶段 3: PPO 输出 `Δscore`，最终 score = `baseline_p + Δscore`
- 优势: 直接解 26F-v3 PPO 的稀疏 reward 问题; baseline 给 PPO 一个有信号的 anchor
- 劣势: 工程量 2.5×, 需要 4070, 训练时间 24-48h

### D. Composite finishing (跨方法融合)
**完成** P1 deferred 的 composite_mean(A, C)。

- 标签已落 (`labels_composite_mean_t3_year=composite.parquet`)
- 训 1 个 LGBM (n_estimators=500 上限避免长尾)
- 输出: `p_t3_composite`，作为 A 的同分位 fallback
- 估计 PR_AUC: 0.124-0.130 (RESULTS §6.2 hypothesis)
- 优势: 半小时跑完, 便宜; 验证 composite 假设
- 劣势: 即使 +0.01 PR_AUC, 不变 P0 winner

### 推荐执行序列

```
Stage 1 (必做, 1 周): A + 部分 B (训 A_t1, A_e20)
Stage 2 (必做, 0.5 周): D 的 LGBM 训练 (验证 hypothesis)
Stage 3 (可选, 2-3 周): C — RL 残差研究分支 (后台跑, 不阻塞生产)
```

**B 的 multi-task 单 booster 不上**（增加复杂度但 PR_AUC 改进存疑）。
独立训 4 个 LGBM (option B 的 simpler 版) 由 Stage 1 涵盖。

---

## 2. 数据契约 (P1.6 daily panel rebuild — 必做基建)

### 2.1 当前问题

`data/duckdb/labeling/feature_panel_v3_344.parquet` 是 2026-05-09 一次性 DuckDB 构建的快照, 覆盖 2023-01..2025-12 (3.78M 行 × 345 cols, 3.65 GB)。

生产 18:45 推理拉的是这个文件 → **明天 (2026-05-12 之后) 没有新数据**。

### 2.2 P1.6 daily panel rebuild 链路

新脚本 `scripts/rebuild_feature_panel_daily.py`:

```python
def rebuild_feature_panel_daily(target_date: date) -> dict:
    """
    每日 18:35 (在 phase20 rebuild_panels 18:30 之后, wave_scores 18:45 之前) 跑.
    
    1. Read combined_short panel for target_date (just one row per stock)
    2. Read tech_event_panel for target_date  
    3. Inner join on (ts_code, trade_date)
    4. TRY_CAST all features to FLOAT (避免 fp32 overflow)
    5. INSERT into feature_panel_v3_344.parquet 的当日 partition
       OR: 维护 separate per-day shards data/duckdb/labeling/feature_panel/year=YYYY/date=YYYY-MM-DD.parquet
    """
```

**推荐**: per-day shards 而不是单 parquet 增量写, 可读可回滚:
```
data/duckdb/labeling/feature_panel/
    year=2023/  (历史不变)
    year=2024/
    year=2025/
    year=2026/
        date=2026-05-12.parquet
        date=2026-05-13.parquet
        ...
```

`scripts/generate_wave_scores_daily.py` 改为读 `data/duckdb/labeling/feature_panel/year=*/date=*.parquet` glob。

### 2.3 一次性回填 + 永久日更

- 回填: 跑一次 `rebuild_feature_panel_daily.py --start 2023-01-01 --end 2026-05-09`，把单 parquet 拆成 per-day shards
- 日更: Celery beat 18:35 每日触发 `task_rebuild_feature_panel_daily`

### 2.4 Schema 校验

每日 rebuild 后必须 assert:
```python
assert set(written_cols) == set(P1_FEATURE_SCHEMA["feature_cols"])
assert sha256(sorted(written_cols))[:12] == "5e71e158e331"
```

如果 schema 漂移（新增 / 删除 / 重命名因子）→ FAIL fast，alert Telegram，**禁止继续推理**（feature_schema_hash mismatch 比预测错误更危险）。

---

## 3. Stage 1 — Supervised retrain spec (option A)

### 3.1 Walk-forward CV

**禁止** 一刀切 train/val/test 切分。改为月度滚动:

```
For each retrain anchor month M ∈ {2025-01, 2025-02, ..., 2026-04}:
    train_window: [M - 24mo, M - 1mo]    # 23 个月训练
    val_window:   M - 1mo (last 20 trading days minus 20-day embargo)
    test_window:  M (current month, 1 month forward)
    
    train LGBM, calibrate isotonic on val
    write model file: models/wave_t3_lgbm_v1/anchor=M/{model.txt, isotonic.pkl}
```

最近的 anchor 的 model 上线; 旧 anchor 留 6 个月做 A/B 对照 + 回滚。

### 3.2 多 seed 集成

每个 anchor 训 5 个 LGBM (seeds 42, 43, 44, 45, 46):
```python
p_ensemble = np.mean([
    iso_seed.transform(model_seed.predict(X))
    for model_seed in 5_models
], axis=0)
```

**减方差**: 单 seed PR_AUC = 0.122 ± 0.02; 5-seed mean 应该减到 ± 0.005。

### 3.3 LGBM 配置

P0 已锁:
```python
LGBM_PARAMS = dict(
    objective="binary",
    metric="average_precision",
    learning_rate=0.02,
    num_leaves=63, max_depth=-1, min_child_samples=200,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
    is_unbalance=True,
    n_estimators=2000, early_stopping=80,
    n_jobs=3,
)
```

**改动**: 加 `bagging_seed = seed_i` 让多 seed 实际不同。

### 3.4 评估门槛 (production 准入)

每个 anchor 必须通过 (在自己的 test_window 上):

| 指标 | 门槛 | 备注 |
|---|---|---|
| `PR_AUC` | ≥ 5 × test_pos_rate | P0 是 3.0×, 抬到 5× 因 ensemble |
| `Brier_ratio` | ≤ 0.95 × null Brier | P0 是 0.972, 必须降到 ≤ 0.95 |
| `ECE_10bin` | ≤ 0.02 | P0 是 0.010 |
| `top1pct_lift` | ≥ 5× | P0 是 3.65×, 抬到 5× |
| `daily_precision@5` | ≥ 0.15 | P0 是 0.20 |
| `null_PR_AUC` | label-shuffle ≤ 1.2 × pos_rate, date-shuffle ≤ 1.5× | P0 临界 |

**任一 anchor 不达标 → STOP, 写 BLOCKER.md, 等用户介入**。不允许「降标准上线」。

### 3.5 模型版本管理

```
models/
  wave_t3_lgbm_v1/
    anchor=2025-01/
      seed=42/{model.txt, isotonic.pkl, metrics.json}
      seed=43/...
      ...
    anchor=2025-02/...
    ...
    anchor=2026-04/   # 当前生产
      seed=42..46/
      ensemble.json   # 5 seed metric 汇总 + 选择策略
```

数据库 `wave_scores_daily.model_version` 改成 `'wave_t3_lgbm_v1.anchor=2026-04.ensemble'`。

### 3.6 Stage 1 时间预算

- A 模型 4 anchor × 5 seed × ~3 min/seed = **5 hours wall-clock**
- A_t1 单 anchor × 5 seed = **1.5 hours**  
- A_e20 单 anchor × 5 seed = **1.5 hours**
- 评估 + 报告 = **1 hour**
- 部署 + Alembic migration = **1 hour** (新增 `model_version` enum 字段, 老数据 backfill)
- **合计: 10 小时**

---

## 4. Stage 2 — Composite finishing (option D)

承接 P1 deferred 的 `labels_composite_mean_t3_year=composite.parquet`:

```bash
python scripts/run_label_composite.py --resume --n-estimators 500 --early-stop 50
```

跑通后:
- 如果 PR_AUC ≥ 0.13 (vs A 的 0.122) → 加为 ensemble member
- 如果 PR_AUC < 0.123 → 关 hypothesis, 不进 production

**预算: 0.5 小时** (比 P1 中途 kill 的那次更紧的训练上限)。

---

## 5. Stage 3 — Hybrid PPO residual (option C, 研究分支)

**仅在 4070 闲置时跑, 不阻塞生产**。

### 5.1 Reward shaping (替换 26F-v3 PPO 的 main_wave_target)

旧 reward (稀疏, ep_rew_mean 全程 0):
```python
y_t3_start = 1 if event_start in {t+1, t+2, t+3} else 0
```

新 reward (option α: 用 baseline 概率作信号):
```python
r_t = p_t3_start_baseline[t, j] - p_market_baseline_mean
```

新 reward (option β: realized excess):
```python
r_t = pct_chg_actual[t+1, j] - pct_chg_actual[t+1, market]   # density 1.0
```

新 reward (option γ: residual + actual):
```python
r_t = (predicted_top_k - baseline_top_k) · (realized_excess_at_t+1)
```

### 5.2 PPO config (vs 26F-v3 失败配置)

| 项 | 26F-v3 (失败) | 26H 建议 |
|---|---|---|
| reward | binary main_wave_target | dense option β 或 γ |
| log_std init | free, 学不动 | fixed 0.3 前 100k step → linear→0.1 |
| ent_coef | default 0 | 0.01 → 0 schedule |
| target_kl | 无 | 0.03 (限制后期 clip_fraction) |
| LR | 1e-4 constant | 1e-4 → 1e-5 linear decay |
| ckpt | 每 50k | 每 25k |
| ckpt selection | manual eyeball | 在 2025-07..2025-12 H1 上 PR_AUC, H2 报告 |

### 5.3 验收门槛

PPO `Δscore` 模型上线门槛:
```
PR_AUC(baseline + Δscore on test) >= PR_AUC(baseline alone) + 0.005
AND ECE(baseline + Δscore) <= 0.025
AND PPO ep_rew_mean 单调 >0 (不再是 0)
```

不达标 → 永久放弃 PPO 路线, ledashi 转到 Stage 1 supervised 全力做。

### 5.4 时间预算

- 3 reward 选项 × 100k step PPO @4070 ≈ 6h × 3 = **18 hours wall-clock**
- 评估 + 决策 = **2 hours**
- **合计: 20 小时**, 整个一周后台跑

---

## 6. 部署与监控

### 6.1 切换协议

P1 → P2 切换:
1. P2 训练完, ensemble metrics.json 满足 §3.4 门槛
2. P2 模型 dry-run 7 个交易日，写到 `wave_scores_daily.model_version='wave_t3_lgbm_v2.anchor=...'`
3. 同时保留 P1 model_version, 双写 (前端可 A/B)
4. 7 天后审 P2 daily_precision@5 ≥ P1 → 切默认 model_version 到 v2
5. 老 v1 保留 30 天可回滚

### 6.2 漂移监控 (新)

每日 19:00 后跑 `scripts/wave_drift_check.py`:
```python
- feature PSI (Population Stability Index): vs 训练分布, 阈值 0.2
- prediction distribution: median p_t3 vs 历史 7 日均值, |Δ| > 0.005 alert
- daily_precision@5 trailing 30 天: < 0.10 持续 5 天 alert
- ECE rolling 30 天: > 0.04 alert
```

任一 alert → Telegram + 自动暂停 next day inference (`wave_scores_daily.status='paused'`，前端 fallback 到上一天)。

### 6.3 模型卡片 (新)

每个 anchor 生成 `MODEL_CARD.md`:
- 训练数据 范围 + 行数 + pos_rate
- 5 seed 指标 (mean ± std)
- ensemble 指标
- 通过 / 不通过门槛
- 已知 limitations
- 建议监控指标

---

## 7. 时间表

```
Week 1 (2026-05-12..05-18):
  Mon  Stage 0 — P1.6 daily panel rebuild 链路 + 一次性回填  (paris, 4h)
  Tue  Stage 1 — Walk-forward A_t3 训练 (4 anchor × 5 seed)  (paris, 5h)
  Wed  Stage 1 续 — A_t1 + A_e20  (paris, 3h)
  Thu  Stage 2 — Composite finishing  (paris, 1h)
  Fri  Alembic migration + 部署 + 7-day dry-run start  (paris, 2h)

Week 2 (2026-05-19..05-25):
  全周 dry-run, 每日观察 wave_drift_check + manual spot-check
  Fri  审 7 天结果, 决定 v1 → v2 切换

Week 3-4 (2026-05-26..06-08):
  ledashi 后台跑 Stage 3 RL 残差 (option C), 4070
  paris 监控 v2 production

Week 5 (2026-06-09..06-15):
  Stage 3 RL 评估, 决定上线 / 放弃
  写 P3 plan: walk-forward 自动化 + 多市场扩展 (港股 / 美股的对应版本)
```

---

## 8. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| Daily panel rebuild 失败 (panel 缺列 / schema 变) | 中 | 推理直接挂 | schema_hash assert + Telegram alert + 自动暂停 |
| Walk-forward 4 anchor 中 1-2 个不达标 | 中 | 需要诊断分布漂移 | 每个 anchor 跑前先看训练分布 cv, > 0.5 警示 |
| 5-seed mean PR_AUC 仍 < 5× base | 低 | A 路线天花板 | 考虑增加因子 (MFP / 行业内 z-score) 而非加 seed |
| Composite 验证失败 (PR_AUC < 0.123) | 中 | 不影响 P0 | 关 hypothesis, 删 composite_min 死代码 |
| RL 残差 PPO 仍不收敛 (Stage 3) | 高 | 浪费 ledashi 一周 | 设硬截断: 100k step ep_rew_mean 仍 0 → kill |
| 4070 算力不够支撑 multi-seed | 低 | Stage 3 拖延 | LGBM ensemble 全部走 ECS CPU, 不依赖 4070 |
| 生产前端用错 model_version | 低 | 数据展示混乱 | API 强制带 `?model_version=` 默认指向最新 ensemble |

---

## 9. Audit 闭环 (vs P1)

| P1 deferred | P2 处理 |
|---|---|
| Null tests (label/date shuffle) — 已 PASS | ✓ 继续要求每个 anchor 跑 |
| Stage 3 composite — labels 已建, LGBM 训练超时 | Stage 2 收尾 |
| Method E (L1 trend filter) | 暂不做, 留 P3 |
| 3 cyq_* cols 缺失 | 验证是否 P2 重训能补回 (查 short panel build script) |
| industry_concentration 门槛 2.0 偏严 | 改 ≤ 2.5 (RESULTS §2.3 已 documented) |
| feature_panel daily rebuild | Stage 0 必做 |
| Production model versioning | §3.5 落地 |
| Multi-head A_t1 + A_e20 | Stage 1 涵盖 |
| Walk-forward CV | Stage 1 月度滚动 |
| Multi-seed ensemble | Stage 1 ×5 |
| 漂移监控 | §6.2 落地 |

---

## 10. 第一个 commit 验收标准

P2 SPEC.md 必须能回答:

1. ✅ 当前训练为什么不能上线生产 (§0)
2. ✅ 三个候选方法之间如何选 (§1)
3. ✅ 选定的方法 (Stage 1 + 2) 的具体 CV 协议 (§3.1)
4. ✅ 多 seed 集成的具体配置 (§3.2)
5. ✅ 生产准入门槛 (§3.4) — 不是 ablation 门槛
6. ✅ daily panel rebuild 链路 (§2.2) — P1.6 必做
7. ✅ 模型版本 + 切换 + 回滚协议 (§3.5 + §6.1)
8. ✅ 漂移监控 (§6.2)
9. ✅ 时间表 + 责任人 (§7)
10. ✅ 风险表 (§8) — 含每条缓解

---

## 11. 输出物 (P2 完成后)

```
handoffs/2026-05-09-p2-training-spec/
  SPEC.md              # 本文件
  RESULTS.md           # P2 收尾后写, 类似 P1 RESULTS.md

src/aurumq/labeling/
  walk_forward.py      # 新, walk-forward CV 工具
  ensemble.py          # 新, 多 seed 集成 + 校准

scripts/
  rebuild_feature_panel_daily.py    # 新, P1.6
  train_walk_forward.py             # 新, Stage 1 主驱动
  resume_composite.py               # 新, Stage 2 (n_estimators=500 上限)
  wave_drift_check.py               # 新, §6.2

models/wave_t3_lgbm_v2/
  anchor=*/seed=*/{model.txt, isotonic.pkl, metrics.json}
  anchor=*/ensemble.json
  anchor=*/MODEL_CARD.md

data/duckdb/labeling/feature_panel/
  year=*/date=*.parquet           # 新, per-day shards

alembic/versions/
  052_wave_scores_v2_columns.py   # 新, model_version enum + status field

src/aurumq/tasks/
  celery_jobs.py                  # 加 task_rebuild_feature_panel_daily, task_wave_drift_check
  celery_beat.py                  # schedule 18:35 (rebuild) + 19:00 (drift check)
```

`AurumQ-RL` 同步:
- `src/aurumq_rl/labeling/` 已有 (P1 已 push), 无变化
- 如走 Stage 3 hybrid: 新增 `aurumq_rl/reward_functions/main_wave_residual.py`

---

## 12. 立即执行 (SPEC commit 后)

按 §7 时间表, 周一开始 Stage 0 (P1.6 daily panel rebuild 链路).

⚠️ **强制 gate**: P2 任何代码改动前, 必须先 commit 此 SPEC, 由 paris (or ledashi) 审一遍。
任何 SPEC 偏离 → 写 BLOCKER.md 暂停, 不允许「先做了再说」。
