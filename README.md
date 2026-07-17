# AurumQ-RL · A股量化强化学习选股开源项目
# AurumQ-RL · An Open-Source Reinforcement Learning Stock-Selection Framework for the China A-Share Market

> 中文：一份面向 A 股的因子工程 + 强化学习选股参考实现，附完整的迭代史、消融实验、生产化决策与教训。
> English: A factor-engineering + reinforcement-learning stock-picking reference implementation for China A-shares, shipped with the full iteration history, ablations, productionization decisions, and lessons learned.

<p align="left">
  <a href="https://github.com/yupoet/aurumq-rl/actions/workflows/ci.yml"><img src="https://github.com/yupoet/aurumq-rl/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/yupoet/aqml"><img src="https://img.shields.io/badge/AQML-strategy%20DSL-D4A853.svg" alt="AQML strategy DSL"></a>
  <a href="https://github.com/yupoet/aurumq-rl/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg" alt="Python 3.10+"></a>
  <a href="https://github.com/yupoet/aurumq-rl/releases"><img src="https://img.shields.io/github/v/release/yupoet/aurumq-rl?include_prereleases&label=release" alt="Latest release"></a>
  <a href="https://github.com/yupoet/aurumq-rl"><img src="https://img.shields.io/badge/tests-1386%20passed-brightgreen.svg" alt="Tests: 1386 passed"></a>
  <a href="https://github.com/yupoet/aurumq-rl/tree/main/docs/factor_library"><img src="https://img.shields.io/badge/factors-105%20alpha101%20%2B%20191%20gtja191-purple.svg" alt="Factor library"></a>
  <a href="https://stable-baselines3.readthedocs.io/"><img src="https://img.shields.io/badge/RL-Stable--Baselines3-blueviolet.svg" alt="Stable-Baselines3"></a>
  <a href="https://onnxruntime.ai/"><img src="https://img.shields.io/badge/inference-ONNX%20Runtime-yellow.svg" alt="ONNX Runtime"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
  <a href="https://github.com/yupoet/aurumq-rl/stargazers"><img src="https://img.shields.io/github/stars/yupoet/aurumq-rl?style=social" alt="GitHub stars"></a>
</p>

<p align="center">
  <img src="docs/assets/social-preview.png" alt="AurumQ-RL — Reinforcement learning stock selection for China A-share market" width="720">
</p>

<p align="left">
  <strong>📊 China A-share · 🤖 PPO/A2C/SAC · 🚀 GPU Train + CPU Infer · 📈 Alpha101 + GTJA Alpha191 + Main-Force + Hot-Money + Northbound · 🧪 26 Phases of Open Experiments</strong>
</p>

---

## 摘要 / Abstract

**中文.** AurumQ-RL 是一个针对 A 股市场特有微观结构（T+1、±10% 涨跌停、主板/科创/创业/北交分层、ST 风险警示、申万一级行业、龙虎榜、北向、游资席位、筹码分布）做工程化封装的强化学习选股开源项目。仓库内含：(1) 一个 polars-native 的因子计算引擎，覆盖 105 个 WorldQuant Alpha101 + 191 个国泰君安 Alpha191（合计 296 个量价因子）外加 11 个 A 股私有因子族（mf_, mfp_, hm_, hk_, inst_, mg_, cyq_, senti_, sh_, fund_, ind_, mkt_, gtja_, tech_, cmf_, zt_）；(2) Stable-Baselines3 PPO 训练栈，针对 RTX 4070 12 GB 做了 GPU 化重构（per-stock 编码器 + CUDA-resident rollout buffer + 索引化观测）；(3) 完整的 14-phase 训练栈演化史，从最初 11% GPU 利用率到 1M-step 隔夜训练；(4) 26-phase 模型实验史，覆盖奖励重设计、长 panel 消融、rank-z 假设检验、SHAP 剪枝、事件衰减编码等关键转折；(5) ONNX 导出 + CPU 推理生产管线。**核心实证发现**：(a) rank-z 跨截面归一化会在长 panel 训练中销毁 5-6 bps 的跨年因子幅度信号；(b) 5 年训练窗口已是 plateau，2018-2019 数据零边际贡献；(c) Strategy D（top-K 仓位按分数加权）能与任何基模型叠加 +7-10 bps 的 mean_y；(d) 二值事件标志直接进 LayerNorm 是 −33% 准确率回归的元凶，必须用 exp-decay τ=10d 编码；(e) cyq 筹码因子的回填 vs 实采分布漂移导致 1.5× T-1 hit 回归，根因是 z-score 不抹平时序 regime 跳变。**当前生产状态（2026-05-11）**：10 个 model_version 同时在 Celery Beat 18:50-19:00 排程，新进 best 是 path5_long（H1 校准 mean_y +0.02882，T1_hit 55.8%）。

**English.** AurumQ-RL is an open-source RL stock-selection framework engineered around the China A-share market's unique microstructure (T+1 settlement; ±10% daily limit per board; main-board vs ChiNext/STAR/BSE segmentation; ST risk-warning; SW Tier-1 industry; Dragon-Tiger List; Northbound; hot-money seats; chip distribution). The repository ships (1) a polars-native factor engine covering 105 WorldQuant Alpha101 + 191 Guotai Junan Alpha191 (296 price-volume factors total) plus 11 private A-share factor families (mf_, mfp_, hm_, hk_, inst_, mg_, cyq_, senti_, sh_, fund_, ind_, mkt_, gtja_, tech_, cmf_, zt_); (2) a Stable-Baselines3 PPO training stack with a GPU-vectorized rewrite for the RTX 4070 12 GB (per-stock encoder + CUDA-resident rollout buffer + index-only observations); (3) the complete 14-phase training-stack evolution from 11 % GPU utilization at bring-up to overnight 1M-step training; (4) the 26-phase modeling experiment history covering reward redesign, long-panel ablations, the rank-z hypothesis test, SHAP-based pruning, and event-decay encoding; (5) ONNX export + CPU inference production pipeline. **Key empirical findings.** (a) Per-day cross-sectional rank-z destroys 5-6 bps of cross-year factor amplitude signal in long-panel training. (b) Five-year training window is the plateau; 2018-2019 data contribute nothing. (c) Strategy D top-K score-weighted sizing compounds +7-10 bps mean_y onto any base model. (d) Binary event flags fed directly into LayerNorm cause a −33 % accuracy regression and must be encoded as exp-decay with τ=10d. (e) Chip-distribution (cyq) backfill versus real-sample distribution shift drove a 1.5× T-1 hit regression — root cause is that cross-section z-score does not equalize mid-stream temporal regime shifts. **Current production state (2026-05-11):** 10 model versions co-scheduled in the Celery Beat 18:50-19:00 budget; the new best is `path5_long` (H1 calibrated mean_y +0.02882, T1_hit 55.8 %).

---

## 目录 / Table of Contents

[中文](#中文导览) · [English](#english-toc) · [Phase Timeline](#phase-timeline)

### 中文导览

- [§1 引言：为什么 A 股市场需要专门的 RL 框架](#1-引言--为什么-a-股市场需要专门的-rl-框架)
- [§2 系统总览：数据契约、因子前缀、宇宙过滤](#2-系统总览)
- [§3 因子库：296 量价因子 + 13 个 A 股私有因子族](#3-因子库)
- [§4 训练方法演进：14 个 Phase 的工程史](#4-训练栈演进史--phase-0--14)
- [§5 模型实验史：26 个 Phase 的研究决策](#5-模型实验史--phase-15--26)
- [§6 监督学习对照赛道（paris 侧 P0/P2/Path 1-6）](#6-监督学习对照赛道paris-侧)
- [§7 实证发现：6 条改变方向的结论](#7-实证发现--六条改变方向的结论)
- [§8 生产流水线：每日 18:30-19:00 评分预算](#8-生产流水线--每日-1830-1900-评分预算)
- [§9 工程教训：从踩坑到守则](#9-工程教训--从踩坑到守则)
- [§10 上手与复现](#10-上手与复现)
- [§11 路线图、引用、许可](#11-路线图引用许可)
- [§12 研究范式分类与未来方向](#12-research-paradigms)

### English TOC

- [§1 Introduction: Why A-shares Need a Dedicated RL Framework](#1-introduction)
- [§2 System Overview: Data Contract, Factor Prefixes, Universe Filter](#2-system-overview)
- [§3 Factor Library: 296 Price-Volume + 13 Private A-share Families](#3-factor-library)
- [§4 Training-Stack Evolution: 14 Phases of Engineering](#4-training-stack-evolution--phase-0-to-14)
- [§5 Modeling Experiment History: 26 Phases of Research Decisions](#5-modeling-experiment-history--phase-15-to-26)
- [§6 Supervised-Learning Companion Track (paris side P0/P2/Path 1-6)](#6-supervised-learning-companion-track)
- [§7 Empirical Findings: Six Conclusions That Changed Direction](#7-empirical-findings)
- [§8 Production Pipeline: Daily 18:30-19:00 Scoring Budget](#8-production-pipeline)
- [§9 Engineering Lessons: From Pitfalls to Operating Rules](#9-engineering-lessons)
- [§10 Quick Start and Reproduction](#10-quick-start-and-reproduction)
- [§11 Roadmap, Citation, License](#11-roadmap-citation-license)
- [§12 Research Paradigms and Future Directions](#12-research-paradigms)

### Phase Timeline

```
Phase 0       Synthetic pipeline-up                 (~pre-2026-04-29)
Phase 1       First real-data alpha101 PPO          2026-04-29..30
Phase 2       Combined 355-col panel + wider net    2026-04-30
Phase 3       R1/R2/R3 smoke-round tuning           2026-05-01 morning
Phase 4       fps scaling, IPC ceiling discovery    2026-05-01 noon
Phase 5       Realizations → GPU framework redesign 2026-05-01 PM
Phase 6/7     GPU-vectorized framework + 50k smoke  2026-05-01 evening
Phase 8       GPURolloutBuffer (CUDA-resident)      2026-05-01 evening
Phase 9       IndexOnlyRolloutBuffer + n_steps=1024 2026-05-01 late evening
Phase 10      Optimizer orphan + LayerNorm + dual pooling  2026-05-01 night
Phase 11/12   bf16 autocast / target_kl=0.10 (eliminated)
Phase 13      PPO SGD perf-probe                    2026-05-01 late night
Phase 14      TF32 + unique-date + 1M overnight     2026-05-02 → 03 early
Phase 15      RL serving integration (champion)     2026-05-02..03
Phase 16-19   Eval correction + multi-seed ensemble 2026-05-03
Phase 20      Long-data PPO                         2026-05-05
Phase 21      V2 forward_10d REJECTED               2026-05-05..06
Phase 22      Main-wave reward redesign             2026-05-06
Phase 23      Episode targets cleanup               2026-05-06
Phase 24/25   Tech-factor + importance-weight REJECT 2026-05-07
Phase 26A-G   cyq fix + event-decay tech (26F prod) 2026-05-07..08

# SL companion (paris side, AurumQ repo)
P0            Wave label ablation (Method A wins)   2026-05-09
P2            5-seed ensemble (wave_t3_lgbm_v2)     2026-05-09
P3            PPO residual ALGORITHM_SPEC v2        2026-05-09
Path 1-6      Multi-path SL exploration              2026-05-10
Long-panel    Hybrid / path1_long / path5_long      2026-05-10..11
```

---

## §1 引言 — 为什么 A 股市场需要专门的 RL 框架
## §1 Introduction — Why A-shares Need a Dedicated RL Framework <a id="1-introduction"></a>

### 1.1 市场微观结构 / Market Microstructure

**中文.** A 股市场和欧美/港股/加密的微观结构差异远大于表面上的"也是股票"。本项目所有训练、回测、因子计算一律仅在 A 股**主板 + 非 ST + 未退市**（约 3000 只）上跑，这是 CLAUDE.md 的硬约束。理由是 regime homogeneity：

- **板别价格限制差异**：主板 ±10%、ST/*ST ±5%、科创板 ±20%、创业板 ±20%、北交所 ±30%。把这五个池子混进同一个训练 batch，模型必须额外学一个"我现在在哪个板"的元变量，跨年泛化崩溃。
- **T+1 结算**：当日买入次日才能卖。RL 训练时的"动作-奖励"延迟和欧美/加密 T+0 完全不同；用 t→t+1 的同日 PnL 等价于看穿未来。
- **集合竞价 vs 连续竞价**：9:15-9:30 集合竞价是 A 股最重要的"标价"事件之一，`stk_auction.bid_*` 字段在 `kpl_list` 里需要单独抓，常规 `pct_chg/amount` 不适用。
- **披露节奏**：主板财报 + 业绩预告时点高度同步，跨年信号同质性强；科创板/北交所披露规则差异显著。
- **流动性结构**：主板日均成交额 vs 北交所差 1-2 个数量级，分位归一化（rank-z）会把北交所的微弱信号放大到和主板同尺度，污染监督信号。

**English.** A-share microstructure differs from US/HK/crypto markets in ways that defeat naive "stocks are stocks" transfer. All training, backtest, and factor computation are restricted by hard constraint (CLAUDE.md) to **main-board, non-ST, non-delisted** stocks (~3000 names). Reasons:

- **Board-specific price limits**: main-board ±10 %, ST/*ST ±5 %, STAR/ChiNext ±20 %, BSE ±30 %. Mixing these into a single training batch forces the model to learn a "which board am I on" meta-variable, destroying cross-year generalization.
- **T+1 settlement**: bought today cannot be sold until tomorrow. The action-reward lag in RL training is fundamentally different from T+0 US/crypto. Using same-day t→t+1 PnL is look-ahead.
- **Call auction vs continuous trading**: the 09:15–09:30 call auction is one of the most informative tape events in A-shares. `stk_auction.bid_*` fields in `kpl_list` need special extraction; standard `pct_chg/amount` do not apply.
- **Disclosure cadence**: main-board earnings + pre-announcement timing is tightly synchronized, giving cross-year signal homogeneity; STAR/BSE differ.
- **Liquidity structure**: main-board daily turnover vs BSE differs by 1–2 orders of magnitude; rank-z normalization would inflate BSE micro-signals to the same scale as main-board, polluting supervision.

### 1.2 为什么 RL 而不是因子排序 / Why RL Rather Than Linear Alpha Aggregation

**中文.** 传统量化做法把 alpha101/gtja191 当 96 个排名分数线性加权，找最优权重向量。问题：

1. **截面 IC 是低维度量**：300 个因子 × 200 天 = 6 万样本估 300 个权重，过拟合容易，跨年泛化差。
2. **非线性交互被忽略**：`alpha_026 × cyq_winning_ratio` 在 30% 行业暴露上限下的表现可能远超两者线性和，线性模型抓不到。
3. **奖励/成本不进训练目标**：传统模型最小化 IC residual，但真实回报是「扣除 30bp 双边费 + T+1 不能反手 + 单行业 30% 上限」之后的实现收益，目标函数和评估指标不一致。
4. **状态依赖动作**：今天选哪 50 只取决于全市场截面分布，不是任意 200 只独立打分加总。

RL 用神经网络直接学映射「当前因子截面 + 持仓状态 → top-K 动作」，把成本、流动性、约束都搬进环境。代价是：样本效率低、调参成本高、可解释性差。本项目实事求是地认为「**RL 不一定比线性 alpha 加权强**」，因此并行维护一条监督学习（SL）赛道作为对照（见 §6）；事实证明 SL 赛道的 `path5_long` 当前是综合最佳（H1 +0.02882），但 RL 赛道的 `phase15e_150k_grand_champion` 在 Sharpe 维度（OOS +6.27）仍是不可替代的多样性来源。

**English.** Conventional quant treats alpha101/gtja191 as ~96 ranking scores to linearly combine. Problems:

1. **Cross-sectional IC is a low-dimensional statistic.** 300 factors × 200 days = 60k samples to estimate 300 weights — overfitting prone, poor cross-year generalization.
2. **Nonlinear interactions go unmodeled.** `alpha_026 × cyq_winning_ratio` under a 30 %-industry-cap can dominate either factor alone; linear models cannot capture this.
3. **Cost/constraint are absent from the training objective.** Traditional models minimize IC residual; realized return is net of 30bp round-trip cost, T+1 inability to reverse, and 30 % single-industry exposure cap. Objective and evaluation diverge.
4. **State-dependent action.** Today's top-50 picks depend on the full market cross-section, not 200 independent scores summed.

RL learns the mapping `current factor cross-section + holdings → top-K action` directly, moving cost / liquidity / constraints into the environment. Tradeoffs: poor sample efficiency, expensive hyperparameter tuning, weak interpretability. This project honestly does NOT assume **RL is always better than linear alpha aggregation** — we maintain a parallel supervised-learning (SL) track as control (§6). Today the SL track's `path5_long` is the all-around best (H1 +0.02882), but the RL track's `phase15e_150k_grand_champion` remains an irreplaceable Sharpe-dimension diversity contributor (OOS +6.27).

### 1.3 项目定位 / Scope and Boundaries

**中文.** 本项目**是**：
- ✅ RL 选股算法 + Gymnasium 环境的参考实现
- ✅ A 股微观结构（T+1 / 涨跌停 / ST / 板别 / 行业暴露上限）的工程化封装
- ✅ 多源因子的**消费者**（按列名前缀识别，输入有什么就用什么）
- ✅ 离线训练 → ONNX 导出 → CPU 推理的端到端流水线
- ✅ **完整的研究决策记录**：每个 phase 都记录了 stack 变更 / 量化证据 / 拒绝原因，便于复现和审计

**不是**：
- ❌ 实盘交易系统（无券商接口、无下单 API）
- ❌ 数据采集工具（不内置任何数据 API key；用户自己 pipeline 写 Parquet）
- ❌ 因子计算库（因子计算可选——`aurumq_rl.factors` 提供 296 个 polars 实现，但用户可以完全自己算）
- ❌ 高频交易（日频选股，T+1 持仓）

**English.** This project **is**:
- ✅ Reference implementation of an RL stock-picking algorithm + Gymnasium environment
- ✅ Engineering encapsulation of A-share microstructure (T+1 / price limits / ST / board / industry cap)
- ✅ A **consumer** of multi-source factors (prefix-recognized; uses whatever is in your Parquet)
- ✅ End-to-end offline-train → ONNX-export → CPU-infer pipeline
- ✅ **Complete research decision log**: every phase records stack diff, quantitative evidence, rejection reason — auditable and reproducible

**Is NOT**:
- ❌ A live trading system (no broker adapter, no order API)
- ❌ A data-ingestion tool (no API keys bundled; you write the Parquet)
- ❌ A factor-computation library (optional — `aurumq_rl.factors` offers 296 polars implementations, but you can roll your own)
- ❌ High-frequency trading (daily-frequency stock picking under T+1)

### 1.4 AurumQ 生态 / The AurumQ Ecosystem

**中文.** 本项目是 AurumQ 平台的一个开源子模块，整个生态分三层：

| 层 | 项目 | 职责 |
|---|---|---|
| 策略 DSL | [AQML](https://github.com/yupoet/aqml) | `.aqml` 声明式策略，可读 / 可验证 / AI 可生成的筛选 + 打分 + 风控规则 |
| 因子 + RL | **AurumQ-RL** (本项目) | 因子工程、A 股约束、PPO/A2C/SAC、ONNX 推理 |
| 平台 | AurumQ (闭源) | Web 平台 + REST API + 模拟盘 + 风控引擎 + AI 投研 |

典型工作流：先用 AQML 写策略意图 → 用 AurumQ-RL 把因子列和约束塞进模型训练 → 训出的 `.onnx` 回到 AurumQ 平台跑模拟盘和实时排程。

**English.** This project is an open-source submodule of the AurumQ platform; the ecosystem has three layers:

| Layer | Project | Role |
|---|---|---|
| Strategy DSL | [AQML](https://github.com/yupoet/aqml) | `.aqml` declarative strategy — human-readable, validatable, AI-generatable screening + scoring + risk rules |
| Factors + RL | **AurumQ-RL** (this repo) | Factor engineering, A-share constraints, PPO/A2C/SAC, ONNX inference |
| Platform | AurumQ (proprietary) | Web platform + REST API + paper trading + risk engine + AI research |

Typical workflow: write strategy intent in AQML → feed factor columns and constraints into AurumQ-RL for training → the resulting `.onnx` returns to the AurumQ platform for paper trading and real-time scheduling.

---

## §2 系统总览 <a id="2-系统总览"></a>
## §2 System Overview <a id="2-system-overview"></a>

### 2.1 数据契约 / The Data Contract

**中文.** 项目对外契约就一句话：**给我一份 Parquet，我就能训练**。Parquet 必含：

- `ts_code` (str): Tushare 风格代码 `XXXXXX.SH/SZ/BJ`
- `trade_date` (date): 交易日
- `close` (float): 收盘价
- `pct_chg` (float): 涨跌幅（**小数形式**，+10% = 0.10，不是 10.0）
- `vol` (float): 成交量（== 0 视为停牌）
- 因子列（至少包含一组前缀）：`alpha_*` / `gtja_*` / `mf_*` / `mfp_*` / `hm_*` / `hk_*` / `inst_*` / `mg_*` / `cyq_*` / `senti_*` / `sh_*` / `fund_*` / `ind_*` / `mkt_*` / `tech_*` / `cmf_*` / `zt_*`

**可选字段**（提供则使用，不提供则自动降级）：

- `is_st` (bool): ST 标记，缺则按全 False 处理
- `days_since_ipo` (int): 上市以来交易日数（用于新股 60 日保护）
- `industry_code` (int): 申万一级行业编码（用于 30% 行业暴露上限）
- `is_hs300` / `is_zz500` (bool): 是否成分股，**按 trade_date 历史变更**（支持「2024-01 在 300、2024-06 调出」的时变性）

数据怎么来不是本项目关心的事，三种取数方式：

1. 用 `scripts/generate_synthetic.py` 一键生成 10 MB 合成数据 demo
2. 用 `scripts/export_factor_panel.py` 从 PostgreSQL 自己的数据仓库抽取（含 SQL 模板，含 HS300/ZZ500 成员标志支持）
3. 自己用任何工具（pandas / DuckDB / Spark）造一份满足契约的 Parquet 写入

**English.** The single contract is: **give me a Parquet, I will train**. Required columns:

- `ts_code` (str): Tushare-style code `XXXXXX.SH/SZ/BJ`
- `trade_date` (date)
- `close` (float)
- `pct_chg` (float): **decimal form**, +10 % = 0.10, NOT 10.0
- `vol` (float): 0 means suspended
- Factor columns under at least one prefix: `alpha_*` / `gtja_*` / `mf_*` / `mfp_*` / `hm_*` / `hk_*` / `inst_*` / `mg_*` / `cyq_*` / `senti_*` / `sh_*` / `fund_*` / `ind_*` / `mkt_*` / `tech_*` / `cmf_*` / `zt_*`

**Optional fields** (auto-fallback when absent): `is_st`, `days_since_ipo`, `industry_code`, `is_hs300`, `is_zz500` (time-varying by `trade_date`).

Three ways to get data: (1) synthetic demo (`generate_synthetic.py`), (2) export from your own PG warehouse (`export_factor_panel.py`), (3) BYO Parquet from any tool that meets the contract.

### 2.2 因子前缀识别 / Factor-Prefix Auto-Discovery

**中文.** `data_loader.py` 通过列名前缀识别因子组，**输入 Parquet 中存在的前缀就被自动加载，不存在的自动跳过**。这套设计的核心是：项目本身不知道你给的是哪些因子，只要列名前缀对得上就一律纳入观测。

| 前缀 | 含义 | 推荐维度 | 输入数据要求 |
|---|---|---|---|
| `alpha_*` | WorldQuant Alpha101（项目自带 105 个实现，含 6 个自定义补充）| 105 | 日频 OHLCV + amount |
| `gtja_*` | 国泰君安 Alpha191（项目自带 191 个实现）| 191 | 日频 OHLCV + vwap + amount + 基准指数 OHLC |
| `mf_*` | Money Flow Velocity — 主力资金流速（4 档累计筹码）| 14 + 6 `_log` 变体 | 4 档资金流分档 |
| `mfp_*` | Main Force Position — 主力筹码持仓（与 `mf_` 互补，**不要混用**）| 12 | 主力净持仓时序表 |
| `hm_*` | Hot Money — 主流游资席位 | 6 | 龙虎榜游资席位日成交明细 |
| `hk_*` | Northbound — 北向资金真实持股 | 4 | 北向持股日表（港股通名单内） |
| `inst_*` | Institutional — 龙虎榜机构净买入 | 3 | 龙虎榜机构席位明细 |
| `mg_*` | Margin — 融资融券 | 3 | 融资融券日表 |
| `cyq_*` | Chip Distribution — 筹码分布 | 3 | Tushare cyq_perf 表 |
| `senti_*` | Sentiment — 涨停板情绪 | 3 | 涨停板池 + 热度榜 |
| `sh_*` | Shareholder — 股东户数 + 大股东增减持 | 2 | 股东数据 |
| `fund_*` | Fundamentals — 基本面 PE/PB/ROE/营收增速 | 4 | 基本面表 |
| `ind_*` | Industry — 申万行业相对强度 | 2 | 行业指数 |
| `mkt_*` | Market — 大盘 + 拥挤度 | 2 | 指数日表 |
| `tech_*` | Technical — 上游算好的 MA/KDJ/MACD/Bollinger（v1.1 后 30 列）| 30 | 已 z-score 的 OHLCV 技术指标 |
| `cmf_*` | Chaikin Money Flow — 60d/120d 累计资金流 | 2 | 量价资金流派生 |
| `zt_*` | 涨停板 stats — 30d/60d 涨停频次、首板、连板 | 6 | 涨停板池 + 历史 |

**总维度灵活**：纯 Alpha101 = 105 维 / Alpha101 + GTJA191 = 296 维 / 全部 17 前缀 ≈ 360 维 / 自定义任意子集。

`StockPickingConfig.n_factors` 决定取前 N 个因子（按字母序），多余的丢弃，不足的报错。

**English.** `data_loader.py` recognizes factor groups by column-name prefix; **whatever prefixes are present in your Parquet get loaded, whatever is absent gets skipped**. The design assumes the project does not know which factors you have — match the prefix, get included as observation.

(See the Chinese table above for the 17-prefix breakdown. Total flexible: pure alpha101 = 105 dims / alpha+gtja = 296 / all 17 prefixes ≈ 360.)

### 2.3 宇宙过滤 / Universe Filtering

**中文.** 默认 `UniverseFilter.MAIN_BOARD_NON_ST` 应用六道 AND 闸门：

1. **`data_ok`**: 当日有日线数据 AND `vol > 0`（剔除停牌）
2. **`main_board`**: `60[0135]\d{3}.SH` ∪ `00[0123]\d{3}.SZ`（剔除 300***/688***/689***/4xx/8xx/9xx）
3. **`listed`**: `days_since_ipo ≥ 60`（新股 60 日保护）
4. **`not_delisted`**: `stock_info.delist_date IS NULL`
5. **`not_st`**: 按当日 `is_st == False` 逐日过滤（C3:不再按当前名称删除整段历史,避免幸存者偏差;无 `is_st` 列时才按行级名称回退）
6. **`not_suspended`**: 当日非停牌（vol > 0 也涵盖此条件）

应用顺序很重要：先 `data_ok` 再 `main_board`，避免在停牌日按板别 regex 算返回值时遇到 NaN/Null 行的 regex 失败。

如要自定义：

```python
from aurumq_rl.data_loader import UniverseFilter, load_panel

# 全市场（仅排 ST + 停牌）
panel = load_panel("data.parquet", universe_filter=UniverseFilter.ALL_NON_ST)

# 只跑沪深 300
panel = load_panel("data.parquet", universe_filter=UniverseFilter.HS300)
```

**English.** Default `UniverseFilter.MAIN_BOARD_NON_ST` applies six AND-gates: `data_ok` (has bar + vol > 0), `main_board` (regex `60[0135]\d{3}.SH ∪ 00[0123]\d{3}.SZ`), `listed` (days_since_ipo ≥ 60), `not_delisted`, `not_st` (per-date `is_st = False`; C3: no whole-history name-based drop — that is survivorship bias; row-level name is only a fallback when `is_st` is missing), `not_suspended`. Ordering matters: `data_ok` before `main_board` to avoid regex on NaN. Alternative filters: `ALL_NON_ST`, `HS300`, `ZZ500`, or supply your own callable.

### 2.4 模块架构 / Module Architecture

**中文.** 仓库布局：

```text
aurumq-rl/
├── src/aurumq_rl/
│   ├── env.py                  # StockPickingEnv (Gymnasium)
│   ├── gpu_env.py              # GPU-vectorized env (Phase 6+)
│   ├── portfolio_weight_env.py # 连续权重组合环境（马科维茨扩展）
│   ├── data_loader.py          # Parquet → numpy/cuda 面板（多前缀识别）
│   ├── policies/
│   │   ├── per_stock_encoder.py    # Deep Sets 风格 per-stock 编码器
│   │   └── shared_policy.py        # 早期 flat MLP（保留参考）
│   ├── rollout/
│   │   ├── gpu_rollout_buffer.py   # CUDA-resident rollout buffer
│   │   └── index_only_buffer.py    # 索引化观测（Phase 9+）
│   ├── inference.py            # ONNX CPU 推理
│   ├── onnx_export.py          # SB3 → ONNX 导出
│   ├── price_limits.py         # 板别动态涨跌停
│   ├── reward_functions.py     # Return / Sharpe / Sortino / Mean-Variance / MainWaveHold
│   ├── main_wave_labels.py     # Phase 22 — MA5/MA10 死叉 + 5d cap 持仓回报标签
│   ├── metrics.py              # 训练指标 JSONL 读写
│   ├── wandb_integration.py    # 实验跟踪（默认离线）
│   ├── sb3_callbacks.py        # SB3 callbacks (WandbMetricsCallback, GpuSamplerCallback, …)
│   ├── gpu_monitor.py          # pynvml-based GPU 采样
│   └── factors/                # polars-native 因子库
│       ├── alpha101/            # WorldQuant Alpha101 (105 因子，10 模块)
│       ├── gtja191/             # 国泰君安 Alpha191 (191 因子，10 batch 文件)
│       ├── _ops.py              # 25+ 通用算子（ts_sum / ts_corr / cs_rank / decay_linear / regbeta / ...)
│       ├── registry.py          # ALPHA101_REGISTRY + GTJA191_REGISTRY
│       └── _docs.py             # markdown 文档生成器
├── scripts/
│   ├── train.py                # 训练入口 V1（CLI）
│   ├── train_v2.py             # 训练入口 V2（Phase 21+ Dict obs，被 Phase 22 V1 main_wave 回退后保留）
│   ├── infer.py                # 推理入口（CLI）
│   ├── eval_backtest.py        # 测试集 IC / Sharpe / 等权净值曲线
│   ├── _eval_main_wave_v1.py   # Phase 22 V1 main_wave 评估（含 hold_return / win_rate / drawdown）
│   ├── compare_rewards.py      # 多 reward 类型对比训练
│   ├── export_factor_panel.py  # PG → Parquet 数据抽取（含 SQL 模板）
│   ├── generate_synthetic.py   # 合成 demo 数据生成
│   ├── oss_download_resumable.py  # HEAD + Range-based resumable downloader
│   └── reference_data/         # alpha101 / gtja191 reference parquet 重建脚本
├── web/                        # Next.js 16 dashboard（runs/ 可视化）
├── data/
│   ├── README.md               # 数据格式 + 列名约定
│   └── synthetic_demo.parquet  # 10 MB 开箱即跑
├── docs/
│   ├── ARCHITECTURE.md
│   ├── FACTORS.md              # 因子前缀约定 + 列名规范
│   ├── TRAINING_HISTORY.md     # 14 phase 完整训练栈演化（1350 行）
│   ├── factor_library/         # 296 篇因子 markdown 文档
│   ├── phase26/                # Phase 26A-G 实验报告
│   ├── SCHEMA.md
│   ├── TRAINING.md
│   └── INFERENCE.md
├── tests/                      # 1386+ 测试，含因子 parity 与 docs 验证
├── handoffs/                   # 跨机器（4070 ↔ ECS）交接日志
└── examples/
    └── quickstart.py           # 端到端示例
```

**English.** Repository layout (see Chinese tree above for full structure). Three key code surfaces:

- `src/aurumq_rl/env.py` and `gpu_env.py` — the Gymnasium `StockPickingEnv` and its later GPU-vectorized counterpart
- `src/aurumq_rl/policies/per_stock_encoder.py` — the Deep Sets-style permutation-equivariant policy that became the architectural breakthrough in Phase 5
- `src/aurumq_rl/main_wave_labels.py` — Phase 22's MA5/MA10 death-cross + 5d-cap hold-return reward that broke the 5.72 % random baseline for the first time

### 2.5 硬件与训练资源约束 / Hardware & Training-Resource Constraints

**中文.** 项目对硬件有两条**红线**：

1. **本地 ECS（8C14G）严禁运行训练**。PyTorch 安装即占 ~3 GB RSS，训练时 OOM 必杀；7-worker `ProcessPoolExecutor` 曾把整台主机 OOM-killed + 重启。训练只能在 GPU 实例（推荐本地 RTX 4070+ 或云端 RTX 4090 / A10 / V100）。
2. **`max_workers=3` 是硬上限**（对所有 `ProcessPoolExecutor` / `ThreadPoolExecutor`），PostgreSQL `shared_buffers=2GB`，内存余量 < 4 GB 时 PG 会被 OOM。

实证训练成本（i7-13700K + RTX 4070 12 GB + 64 GB DDR5）：

| 配置 | 因子数 | 训练步数 | wall time | 备注 |
|---|---:|---:|---|---|
| smoke (Phase 0) | 16 | 1k | 90s | 合成数据，CPU 即可 |
| Phase 1 | 16 | 100k | ~50 min | alpha101 short panel，n_envs=8, fps 333 |
| Phase 7 | 64 | 50k | ~7 min | GPU framework smoke, fps 1490 |
| Phase 10 | 64 | 1M | ~8h（隔夜）| LayerNorm + dual pooling + bf16, fps 326 |
| Phase 14 | 64 | 1M | ~6h（隔夜）| TF32 + unique-date, fps 460 |
| Phase 16a (prod) | 343 | 300k | ~5h | 6 seeds 并行外推 |
| Phase 22 (main_wave) | 343 | 300k | ~8h | 3-run 隔夜对照 (A/B/C) |
| Phase 26F-v3 (prod) | 361 | 300k | ~5h | 3 seeds × 1 config |

**English.** Two hard rules:

1. **The local 8-core 14 GB ECS is FORBIDDEN for training.** PyTorch installation alone occupies ~3 GB RSS; training will OOM-kill the host. A 7-worker `ProcessPoolExecutor` once OOM-killed and rebooted the box. Train only on a GPU instance (local RTX 4070+ or cloud RTX 4090 / A10 / V100).
2. **`max_workers=3` is a hard ceiling** for all `ProcessPoolExecutor` / `ThreadPoolExecutor`. PostgreSQL `shared_buffers=2GB`; PG OOMs when host free RAM < 4 GB.

Measured training cost on i7-13700K + RTX 4070 12 GB + 64 GB DDR5 (see Chinese table above for the 8-config breakdown spanning smoke runs to overnight 1M-step phases).

---

## §3 因子库 <a id="3-因子库"></a>
## §3 Factor Library <a id="3-factor-library"></a>

### 3.1 自带因子计算引擎 / The Built-in Factor Engine

**中文.** `src/aurumq_rl/factors/` 是 polars-native 实现的 296 个量价因子（105 alpha101 + 191 gtja191）。每个因子一篇 markdown 文档在 `docs/factor_library/`，含原文公式 + Polars 实现说明 + 引用。

| Family | 实现 | quality_flag=0 (clean) | =1 (best-effort) | =2 (stub) |
|---|---|---|---|---|
| alpha101 | 101/101 + 6 自定义 | 88 | 13 | 0 |
| gtja191 | 191/191 | 177 | 12 | 2 |

`quality_flag` 语义：
- **0 (clean)**：完整 + 数值稳定 + 跨平台 parity（如 alpha_001、gtja_159）
- **1 (best-effort)**：实现合理但存在已知边界情况（如 alpha_017 在窗口=2 时 std=0 触发 NaN，已用 `fill_null` 处理但未触发 inf）
- **2 (stub)**：实现存在但等价于占位（如 gtja_115/189 没有可靠数据源对应的 sd_pe_ttm 字段）

注册表用法：

```python
import aurumq_rl.factors.alpha101  # registers 107
import aurumq_rl.factors.gtja191   # registers 191
from aurumq_rl.factors.registry import ALPHA101_REGISTRY, GTJA191_REGISTRY

panel = pl.read_parquet("ohlcv.parquet")  # 需含 OHLCV + vwap + amount
df = panel.with_columns([fn(panel).alias(name) for name, fn in ALPHA101_REGISTRY.items()])
```

通用算子 `_ops.py` 提供 25+ 个跨家族复用的算子：`ts_sum / ts_corr / cs_rank / decay_linear / regbeta / ts_argmax / ts_argmin / ts_min / ts_max / ts_rank / ts_delta / ts_delay / ts_std / ts_skew / ts_kurt / ind_neutralize / scale / signed_power / sign`，所有都是 polars expr-aware，可在懒求值 graph 里组合。

**English.** `src/aurumq_rl/factors/` ships 296 polars-native price-volume factors (105 alpha101 + 191 gtja191), one markdown doc per factor under `docs/factor_library/` with original formula + Polars implementation notes + references. `quality_flag ∈ {0 clean, 1 best-effort, 2 stub}` per the Chinese table above. The common-operator module `_ops.py` provides 25+ Polars-aware operators (`ts_sum`, `ts_corr`, `cs_rank`, `decay_linear`, `regbeta`, …) that compose lazily.

### 3.2 A 股私有因子族 / Private A-share Factor Families

**中文.** 这是项目 11 个 A 股私有因子族，必须从用户自己的数据仓库算好后写进 Parquet。它们对应的中国市场原始数据源在欧美市场没有等价物：

#### 3.2.1 `mf_*` 主力资金流速 (Money Flow Velocity, 14 + 6 cols)

由用户上游 `scripts/compute_mf_panel.py` 输出，14 个基础列 + 6 个 `_log` 变体。例：

- `mf_net_{1d,3d,5d,10d,20d,60d}` — 主力净流入累计（元）
- `mf_buy_share_main` — 主力买入占比（**SHAP rank 7**，Path 4 模型里第 7 重要的特征）
- `mf_net_accel_5_20` — 5d / 20d 流入加速度
- `mf_net_5d_amount_ratio` — 5d 净流入 / 5d 成交额
- `mf_net_{1d,3d,5d,10d,20d,60d}_log` — sign-preserving log1p 变体（2026-05-08 加入），公式 `sign(x) · ln(1 + |x|/total_amount)`，把原始 std=1.5×10⁸ 的"元"量级压到 std=0.040 的"无量纲"量级

**HUGE_TAIL 事故**：原始 `mf_net_*d` 标准差从 1e8 到 1e10，跨 ts_code 的尺度差异极大（一只大盘股一天净流入 10 亿元，一只小盘股 1 万元）。Phase 24 在 `data_loader._cross_section_zscore` 里 z-score 之后仍然有量级，因为 polars 默认 `ddof=1` 在 3000-stock 截面下分母被极端值拉爆。修复：上游加 `_log` 变体后训练直接吃压缩量级，跨年泛化恢复。

#### 3.2.2 `mfp_*` 主力筹码持仓 (Main Force Position, 12 cols)

由 `src/aurumq/factors/main_force.py` 输出，与 `mf_*` **互补但完全独立**：

- `mfp_elg_buy_ratio_20d` — 超大单买入占比 20 日
- `mfp_lg_buy_ratio_20d` — 大单买入占比 20 日
- `mfp_main_net_cum_pct` — 主力净流入累计百分位
- `mfp_main_net_volatility_20d` — 主力净流入波动 20 日

**Phase 16 关键事故**：`mfp_` 前缀曾被静默从 `aurumq_rl.data_loader.FACTOR_COL_PREFIXES` 漏掉，12 列输入完全没进训练。修复后 16a 跑出 adj Sharpe +1.593（vs 之前 plateau +1.165）。教训：**前缀注册表是 single source of truth，prefix-glob 漏一个前缀等于静默丢一族特征**。

#### 3.2.3 `cyq_*` 筹码分布 (Chip Distribution, 3 cols)

源自 Tushare cyq_perf 表，3 列：
- `cyq_winning_ratio` — 当前价位上的获利筹码占比
- `cyq_concentration_70` — 70% 筹码集中在多少价位区间
- `cyq_cost_distance` — 当前价 vs 平均成本距离

**Phase 26 关键事故**：cyq 是 A 股独有的因子（券商内部模型 + Tushare 加工），但 Tushare 历史只能回填到 2025-10-20，更早数据是 cyq_perf v1.0 用合成方法补的。结果：训练集（含合成回填）`cyq_cost_distance` std = 0.197；OOS 集（全部实采）std = 0.066，**3× 压缩**。跨截面 z-score 不抹平这种 mid-stream regime shift —— 模型学到的是合成数据的尺度，到真实数据上全错。Phase 26C2 切换到 v1.2 修正 cyq（bulk API 重新回填）后，T-1 lift 从 1.47× 反弹到 **2.61×**（甚至超过原始 23A 的 2.38×，且收敛快 4 倍）。

#### 3.2.4 `hm_*` 主流游资席位 (Hot Money Seats, 6 cols)

源自 Tushare `top_list` + `top_inst`：
- `hm_net_5d` / `hm_net_20d` / `hm_net_60d` — 游资席位累计净买入
- `hm_recent_active` / `hm_seat_count_30d` / `hm_top3_concentration` — 活跃席位 / 30 日席位数 / 前 3 名集中度

**结构性 hard wall**：Tushare 龙虎榜数据 ≥ 2023-08-16 才存在，2018-2023.8 的 `hm_*` 永远是 NULL。Phase 20 长 panel 训练时 LightGBM 的 `use_missing=True` 自动处理，但 RL 训练时观测向量必须填 0（不能 NaN）。修复：`data_loader._fill_missing_with_zero_track_mask` 同时填 0 + 写 mask，模型可选择性地学到"这一列 mask=1 时无效"。

#### 3.2.5 `hk_*` 北向资金 (Northbound, 4 cols)

`hk_hold_chg_60d` 等，**SHAP rank 16**。结构性 null：港股通名单外的 25% A 股永远是 NULL。同样 `_fill_missing_with_zero_track_mask` 处理。

#### 3.2.6 `inst_*` 机构持仓 (Institutional, 3 cols)

`inst_appear_count_60d` / `inst_net_30d` — 龙虎榜机构席位活跃度。

#### 3.2.7 `mg_*` 融资融券 (Margin, 3 cols)

`mg_short_chg_20d` / `mg_balance_pct` 等。78% A 股有融资融券覆盖。

#### 3.2.8 `senti_*` 情绪 (Sentiment, 3 cols)

涨停板池 + 同花顺热度榜派生。**已知问题**：`senti_ths_hot_pct` 99% null，因为同花顺热度榜只追踪 ~3000 只热门股；非热门股在 2024-08-29 之前完全没有数据。Phase 26 数据质量审计把 `senti_ths_hot_pct` 列入 `include_columns_v1_clean.txt` 的永久排除清单。

#### 3.2.9 `sh_*` 股东结构 (Shareholder, 2 cols)

`sh_holder_num_chg_30d` — 股东户数变化。86% null（季度披露，日频面板上稀疏）。

#### 3.2.10 `fund_*` 基本面 (Fundamentals, 4 cols)

`fund_pe_ttm` / `fund_pb` / `fund_roe_ttm` / `fund_revenue_growth`。**SHAP rank 11 (pe_ttm) / rank 25 (roe_ttm)**。

事故：688*** 科创板的 fund_pe_ttm 在 2025-08 之前缺失约 600 只 × 每天的 hole，因为 Tushare daily_basic 接口对科创板支持不全。2026-05-08 批次用 bulk daily_basic 回填了所有 600+ × 历史日期。

#### 3.2.11 `ind_*` 申万行业相对强度 (Industry, 2 cols)

`ind_relative_strength_20d` / `ind_relative_strength_60d` — 个股 vs 申万一级行业指数收益差。49-57% null（`sw_index_member` 表只覆盖约 3000 只主板成分股）。

#### 3.2.12 `mkt_*` 大盘 (Market, 2 cols)

`mkt_index_pct_chg_5d` / `mkt_index_volatility_20d` — 上证指数派生。**Phase 16 关键发现**：drop `mkt_*` 组反而 +0.428 lift！原因：`mkt_*` 在主板宇宙下高度共线（所有股票同一个上证指数派生量），模型用它做"今天大盘涨/跌"的偷懒预测，反而损害了选股能力。**永久从主流配置移除**。

#### 3.2.13 `tech_*` / `cmf_*` / `zt_*` 技术指标 (Tech panel, 30 + 2 + 6 cols)

Phase 26 新增。`tech_*` 30 列 = MA5/10/20/60 比值 + KDJ 派生 + MACD 派生 + Bollinger 派生 + ATR 派生 + 振幅。`cmf_*` 2 列 = Chaikin Money Flow 60d/120d。`zt_*` 6 列 = 涨停板 30d/60d 频次 + 首板/连板/最长连板。

**Phase 24/25 重大事故**：24A 把 36 个技术因子直接接在 RL 训练 panel-load 时算（而不是上游 parquet），结果 T-1 hit 从 2.11% 跌到 **0.40%**（lift 0.45× < 随机 0.89×）。根因：KDJ/振幅近似自 close-only（panel 没有 OHLC），MA-cross / golden-cross 是二值事件标志，进 LayerNorm 后 z-score 把 binary 0/1 拉成极端 outlier，污染了梯度。Phase 26F 修复：把二值事件改为 **指数衰减 τ=10d 编码** `evt(t) = sum(1[event in last 10d] * exp(-(t-tau)/10))`，T-1 hit 从 1.13× 反弹到 **2.27×（best 2.41% hit at step 50k）**。教训见 §7。

**English summary.** The 11 private A-share factor families are: `mf_*` (Money Flow Velocity, 14 base + 6 sign-preserving log variants — fixes the 1e8-yuan HUGE_TAIL scale issue); `mfp_*` (Main Force Position, 12 cols, independent of `mf_*` despite the similar prefix — Phase 16 found `mfp_` was silently missing from `FACTOR_COL_PREFIXES`); `cyq_*` (chip distribution, 3 cols — Phase 26C2 v1.2 fix recovered T-1 lift from 1.47× to 2.61× by replacing the synthetic-backfill v1.0 with a bulk-API-recomputed v1.2); `hm_*` (Dragon-Tiger hot-money seats, 6 cols, structural null pre-2023-08-16); `hk_*` (Northbound, 4 cols, structural null for 25 % non-HK-Stock-Connect stocks); `inst_*` (institutional, 3); `mg_*` (margin trading, 3); `senti_*` (sentiment, 3, 99 % null for non-hot stocks); `sh_*` (shareholder, 2, 86 % null due to quarterly disclosure); `fund_*` (PE/PB/ROE/revenue growth, 4 — SHAP rank 11/25); `ind_*` (SW industry relative strength, 2); `mkt_*` (market index, 2 — **dropped permanently in Phase 16 because removing it gave +0.428 adj Sharpe lift**). Phase 26 added `tech_*` (30), `cmf_*` (2), `zt_*` (6) — note that binary event flags must be exp-decay encoded (`τ=10d`) not raw 0/1 to avoid the −33 % regression seen in Phase 24A.

### 3.3 因子前缀注册纪律 / Factor-Prefix Registration Discipline

**中文.** `aurumq_rl.data_loader.FACTOR_COL_PREFIXES` 是 single source of truth。当前规范清单：

```python
FACTOR_COL_PREFIXES = (
    "alpha_", "mf_", "mfp_", "hm_", "hk_", "inst_", "mg_", "senti_",
    "sh_", "fund_", "ind_", "cyq_", "gtja_",
    "tech_", "cmf_", "zt_",
)
```

**漏一个前缀 = 静默丢一族特征 + Phase 16 复现**。所有 PR 修改这个 tuple 必须同步更新：

- `tests/test_data_loader.py:test_factor_col_prefixes_lockdown` —— 字典对比 + 顺序对比
- `scripts/export_factor_panel.py:FACTOR_PREFIXES` —— PG 抽取脚本的镜像列表
- `docs/FACTORS.md` —— 表格 + 列名规范文档

**English.** `aurumq_rl.data_loader.FACTOR_COL_PREFIXES` is the single source of truth (17 prefixes today). Missing one = silently lose a factor family = Phase 16 reproduction. Three sites must be kept in sync per PR: the tuple itself, the `tests/test_data_loader.py` lockdown test, and `scripts/export_factor_panel.py`.

### 3.4 SHAP 剪枝实验：345 → 226 / SHAP-Based Pruning

**中文.** 见 [`handoffs/2026-05-10-sl-extras/shap_audit/`](handoffs/2026-05-10-sl-extras/shap_audit/)（paris 侧执行）：

- **方法**：`shap.TreeExplainer` 跑在 Path 4 最佳 LightGBM 单模（`nl31_lr050_mdl50_seed44`），10k 行 VAL_EFF 数据，按 `mean(|SHAP|)` 排名。
- **Top 5 surprises**：`gtja_159` 一骑绝尘（mean|SHAP|=0.001270，gain 28.7%，162 个分裂点）、`gtja_158`、`gtja_065`、`gtja_140`、`gtja_181`。资金流第 7 名 `mf_buy_share_main`，基本面第 11 名 `fund_pe_ttm`，北向第 16 名 `hk_hold_chg_60d`。
- **剪枝规则**：`mean(|SHAP|) < 1e-6` 视为零贡献 → 119 个候选 → 保存到 `drop_candidates.json`。被剪掉的例子：`alpha_098`、`gtja_054`、`gtja_101`、`gtja_190`、`alpha_002`、`gtja_001`、`mf_net_accel_5_20`、`mf_net_60d`、`gtja_114`、`inst_net_30d`。
- **Path 6 验证**（Bayesian opt 50 trials）：226 列训出来 H1 校准 mean_y = +0.028265 vs Path 4 全 345 列的 +0.028483，**Δ = −0.0002**（与噪声不可分辨）。bundle 从 40 MB 缩到 32 MB，训练时间 −15%。结论：**超参数搜索已 saturate，剪枝是免费午餐**。

**English.** SHAP-based feature pruning ran on the best Path 4 LightGBM single model (`nl31_lr050_mdl50_seed44`) over 10k VAL_EFF rows. `mean(|SHAP|) < 1e-6` ⇒ drop candidate ⇒ 119 columns saved to `drop_candidates.json`. Validating on Path 6 (226 cols + 50 Bayesian-opt trials): H1 calibrated mean_y +0.028265 vs full-345 Path 4 +0.028483 = −0.0002 (indistinguishable from noise). Bundle 40 MB → 32 MB, training −15 %. Lesson: **hyperparameter search is saturated; SHAP pruning is a free lunch.**

### 3.5 存储路径与流式处理 / Storage Layout & Streaming

**中文.** 因子 parquet 按家族 × 年份分片：

| 路径 | 内容 |
|---|---|
| `data/duckdb/factor_eval/alpha_panel_year=YYYY.parquet` | alpha101 全族（109 cols） |
| `data/duckdb/factor_eval/gtja_panel_year=YYYY.parquet` | gtja191 全族（193 cols，单年最大 1.37 GB） |
| `data/duckdb/factor_eval/mf_panel_year=YYYY.parquet` | mf_ 22 cols (14 + 6 log + 2 helper) |
| `data/duckdb/factor_eval/cyq_panel/year=YYYY.parquet` | canonical cyq 3 cols（v1.2 修正版） |
| `data/duckdb/factor_eval/tech_panel/year=YYYY.parquet` | tech_ 30 cols |
| `data/duckdb/factor_eval/tech_event_panel/year=YYYY.parquet` | tech_evt_ 8 cols（含 exp-decay 编码） |
| `data/duckdb/quotes_enriched/year=YYYY.parquet` | 11 个内部 enriched 家族 `mfp_/hm_/hk_/inst_/mg_/senti_/sh_/fund_/ind_/mkt_/cyq_` legacy |

**流式 concat 红线**：14 GB ECS 上禁止跑 `pl.concat(diagonal_relaxed) + sink_parquet` 拼 10 年面板，会 OOM-killed 主机。正确做法是先 shard，然后 `pl.scan_parquet([shards], missing_columns="insert")` 流式扫，见 `scripts/build_combined_panel_safe.py`。

**English.** Factor parquets are sharded by family × year (see Chinese table). **Streaming red line**: on the 14 GB ECS, `pl.concat(diagonal_relaxed) + sink_parquet` over 10 years of panel data will OOM-kill the host. Correct pattern: shard first, then `pl.scan_parquet([shards], missing_columns="insert")` streaming scan. See `scripts/build_combined_panel_safe.py`.

---

## §4 训练栈演进史 — Phase 0 → 14 <a id="4-训练栈演进史--phase-0--14"></a>
## §4 Training-Stack Evolution — Phase 0 to 14 <a id="4-training-stack-evolution--phase-0-to-14"></a>

**中文.** 本节是 GPU 训练栈本身的工程史 —— 从最初 11% GPU 利用率到 1M-step 隔夜训练的所有 stack diff、bug、消融。**模型/数据/奖励的实验史在 §5（Phase 15-26）**。两个 phase 编号体系独立：本节 Phase 0-14 是「框架建设」，§5 Phase 15-26 是「在已建好的框架上跑模型实验」。

完整的逐 phase 记录在 [`docs/TRAINING_HISTORY.md`](docs/TRAINING_HISTORY.md)（1350 行），本节是其压缩版。

**English.** This section is the engineering history of the training stack itself — every stack diff, bug, and ablation from the initial 11 % GPU utilization to the 1M-step overnight training. **The modeling / data / reward experiments live in §5 (Phases 15–26).** The two numbering systems are independent: Phases 0–14 here are "framework construction"; §5 Phases 15–26 are "model experiments on top of the built framework".

The full per-phase record is in [`docs/TRAINING_HISTORY.md`](docs/TRAINING_HISTORY.md) (1350 lines). This section is its compressed version.

### Phase 0 — 合成数据流水线打通 / Synthetic Pipeline Bring-up
*(~pre-2026-04-29)*

**Goal.** Prove the parquet → env → SB3 PPO → ONNX → backtest → JSON link before touching real data.

**Stack.** `StockPickingEnv` (numpy panel, single-process), SB3 default `MlpPolicy net_arch=[64,64]`, PPO `n_envs=1 batch=64 n_steps=2048 epochs=10`, `synthetic_demo.parquet` (~200 SYN-coded fake stocks).

**Bugs surfaced** (4 in PR #1 / #2):

| # | Bug | Fix |
|---|---|---|
| 1 | `gymnasium` not always installed | lazy import + placeholder raises ImportError |
| 2 | ONNX export device mismatch (CUDA policy + CPU dummy_obs) | move policy to CPU before export |
| 3 | `torch.onnx dynamo=True` breaks SB3 `Normal` distribution | pass `dynamo=False` |
| 4 | JSON serializer can't handle `numpy.float32` | `WandbMetricsCallback._append_jsonl` got `default=_json_default` |

**Outcome.** Pipeline end-to-end. ONNX exported. Backtest IC ≈ 0 (synthetic noise, expected).

### Phase 1 — 第一次真实数据训练 / First Real-Data Run *(2026-04-29 ~ 30)*

**Goal.** Scale to a real factor panel and a real GPU. Run a 100k-step PPO on alpha101 to see how far naive setup goes.

**Data.** `factor_panel_alpha101_short_2023_2026.parquet` — 105 alpha cols, 5743 stocks, 2023-01..2026-04. After `main_board_non_st`: 3043 × 800 × 105.

**Config.** PPO `--total-timesteps 100000 --n-envs 8 --vec-normalize --learning-rate 3e-4 --target-kl 0.05 --max-grad-norm 0.3 net_arch=[64,64] n_factors=16`.

**Bugs (8 new):** NaN propagating through cross-section z-score (real PG data has NaN cells for suspended / pre-IPO; synthetic didn't); OOS obs_dim mismatch (training universe = 3043, OOS = 3052 because some IPOs landed — env's `observation_space` is fixed at training time; fix = `align_panel_to_stock_list` persisted in `metadata.json`); PPO `approx_kl=41,820` on first update (12.5M-param first layer + 48,688-dim obs; fix = `target_kl=0.05 + max_grad_norm=0.3` → `approx_kl=0.028`); `mean_fps=0` in summary (SB3 only emits `time/fps` on rollout-summary frames; fix = callback computes wall-time fps); `metrics_summary` all-null (callback wrote raw SB3 keys; `summarize_metrics()` expects canonical schema; fix = raw→canonical mapping at write time); `runs/` gitignore unanchored (`web/app/runs/` silently dropped; fix = `/runs/` anchored at root); alpha045 STHSF parity 44 % mismatch on Windows only (scipy rank-tie-break unstable across versions on 10-stock synthetic; fix = `@pytest.mark.xfail(strict=False)`); OSS admin AK disabled mid-flight (switch to wepa AK).

**Outcome.** 100k-step PPO ran clean. **fps ~ 333. GPU util ~ 11 %.** The 4070 was massively underutilized — wide first-layer of `[64,64]` was only 3 M params; GPU spent most of its time waiting on CPU rollouts.

### Phase 2 — 联合 panel + 网络加宽 + feature_group_weights / Combined Panel + Network Widening *(2026-04-30)*

**Data added.** `factor_panel_combined_short_2023_2026.parquet` — **355 factor cols** (105 alpha + 191 gtja + 14 mf + 12 mfp + 5 hk + 4 fund + 3 inst + 3 mg + 3 senti + 2 sh + 2 ind + 2 mkt + 6 hm + 3 cyq), 5643 stocks × 800 dates, 7.7 GB zstd. After main-board filter: 3014 × 600.

**Code added.** `--policy-kwargs-json` CLI accepts `{"net_arch":[2048,1024,512], "activation_fn":"relu"}`. `--feature-group-weights-json` accepts e.g. `{"alpha_*":2.0, "mf_*":0.5}`, applied **after** z-score in `_apply_feature_group_weights` so `VecNormalize` doesn't neutralize the weights.

**Network widening.** `net_arch=[2048,1024,512]`. First-layer params for `n_factors=64`: `3014 × 64 × 2048 ≈ 395 M`. GPU memory 3 GB → 12 GB peak; util peak 11 % → 57 %.

**3-way alpha-prefix ablation:**

| Run | `--feature-group-weights-json` | OOS IC | OOS top30 Sharpe |
|---|---|---|---|
| `ablation_alpha_w0_5` | `{"alpha_*":0.5}` | (~0) | (~random) |
| `ablation_alpha_w1_0` | `{"alpha_*":1.0}` (no-op baseline) | (~0) | (~random) |
| `ablation_alpha_w2_0` | `{"alpha_*":2.0}` | +0.0006 | −0.807 (random p50 −0.482) |

**Decision.** Framework works end-to-end. Numbers are noise at 15k steps. Validation passed; promote `feature_group_weights` as load-bearing CLI feature.

### Phase 3 — 三轮 smoke R1/R2/R3 / Three Smoke Rounds *(2026-05-01 morning)*

**Round 1 (R1).** `--policy-kwargs-json '{"net_arch":[1024,512,256]}'` + `--feature-group-weights '{"alpha_*":2.0, "mf_*":1.5, "gtja_*":1.0}'` at 50k steps, `n_envs=12, n_steps=2048, target_kl=0.05`. **First model with explained_variance climbing.** `value_loss` from 1.5e-2 → 4.3e-3 over 22 rollouts. OOS IC = +0.011, top30 Sharpe = +1.42 (random p50 −0.48, vs-p50 +1.90). **GPU util 35 %, fps 312.**

**Round 2 (R2).** Three changes at once: `target_kl 0.05 → 0.10`, `n_envs 12 → 14`, `n_steps 2048 → 4096`. First attempt OOMed (`MemoryError` allocating 8.83 GiB rollout buffer); reduced `n_steps 4096 → 1024`; ran again. **OOS top30 Sharpe = +2.16**, vs-p50 = +0.74 above R1. Convergence accelerated (explained_var 0.93 at 30k vs R1's 0.81). **But: three changes at once = uninterpretable**. Could be KL relaxation, env parallelism, or buffer length. Lesson recorded — Phase 3's central rule: **one change per round**.

**Round 3 (R3).** Just `target_kl 0.10` + `learning_rate 3e-4 → 1e-4 anneal`. OOS top30 Sharpe = +1.89 (R2 = +2.16, R1 = +1.42). Within seed variance of R2; suggests `target_kl` accounts for most of R2's lift, but `n_envs / n_steps` cannot be cleanly attributed.

**Lesson.** **OOS Sharpe at 50k steps is noise.** Don't pick winners from smokes; pick them from convergence-scale runs (≥ 1M ideally 5M). Burned three rounds arguing about R1/R2/R3 ranking before admitting differences were within seed variance.

### Phase 4 — fps 扩展实验 + IPC 天花板 / fps Scaling and IPC Ceiling *(2026-05-01 noon)*

**Goal.** Find the n_envs ceiling.

**Method.** Smoke-grid `n_envs ∈ {12, 14, 16, 18, 20}` at fixed `n_steps=1024`.

| `n_envs` | fps | GPU util | Outcome |
|---:|---:|---:|---|
| 12 | 314 | 35 % | R1 baseline |
| 14 | 366 | 41 % | linear scale |
| 16 | 412 | 47 % | linear scale |
| 18 | 455 | 53 % | linear scale starting to bend |
| 20 | 458 | 56 % | **bent — IPC bottleneck** |

**Realization.** Above `n_envs=18`, adding env doesn't proportionally raise fps because the bottleneck is **Python IPC** between worker subprocs and the central learner, not GPU compute. **And: n_envs=20 OOMed** on rollout buffer alloc (14.7 GiB). Back to n_envs=12 for safety.

**The IPC ceiling discovery** changed the project direction. The classic SB3 setup (numpy panel + subproc envs + CPU rollouts → GPU train) is **fundamentally CPU-rollout-bound**. The 4070 was sitting at 56 % util at the bottleneck. To break through, we'd need to move rollouts onto the GPU itself. → Phase 5 / 6.

### Phase 5 — 四个 realization 推出 GPU-框架重构 / Four Realizations Driving GPU-Framework Redesign *(2026-05-01 afternoon)*

**Goal.** Sit down, look at the data, decide whether to keep tuning or fundamentally redesign.

**Four realizations:**

1. **Brute-force capacity is a trap.** 395 M-param flat MLP needed 12 GB VRAM, fps capped at 458, and didn't beat the per-stock symmetry prior. A symmetry-correct architecture (per-stock encoder, ~50 k params shared across all stocks) has dramatically more inductive bias for stock-picking AND is faster.
2. **The numpy panel is the wrong abstraction.** Re-uploading the same panel to GPU memory every env reset is wasteful. The panel should live in GPU memory throughout training, indexed by env step.
3. **Observations should be indices, not values.** Stock factor vectors don't change across env steps — only which date the env is at changes. Send (date_idx, stock_codes_idx) over the IPC boundary, do the GPU-side gather afterwards.
4. **VRAM and RAM are different.** Confused them once: GPU showed 12 GB used, my Python proc was only 3.8 GB RSS. Always check `nvidia-smi --query-gpu=memory.used` per-process AND host `Get-Process -RSS`.

**Decision.** Design a GPU-vectorized framework: per-stock encoder + CUDA-resident panel + index-only observations + dual pooling head. Implementation in Phase 6/7.

### Phase 6 / 7 — GPU-vectorized 框架 + 50k smoke / GPU Framework + 50k Smoke *(2026-05-01 evening)*

**Design (Phase 6, designed afternoon; built in Phase 7).**

- **`gpu_env.py`**: Gymnasium-compatible env that holds the entire panel as a CUDA tensor `panel[n_dates, n_stocks, n_factors]`. Step = advance one date; obs = `panel[date_idx]` slice + holdings mask. **n_envs=16 in a single proc** (vectorized across env axis on GPU, no IPC).
- **`PerStockEncoderPolicy`** (Deep Sets): apply the SAME MLP to each stock's factor row, then aggregate with mean+max dual pooling. Permutation-equivariant. Net_arch=[64, 32] per-stock, then a [64, 32] head — only ~50 k params total, shared across 3014 stocks.
- **Action**: scaled tanh on per-stock logits, top-K selection.

**Smoke run (Phase 7).** 50k steps, n_envs=16, `n_steps=1024`. **fps 1490** (vs Phase 4 ceiling 458 — 3.25× lift). GPU util peak 78 %, mean 62 %. OOS IC +0.014, top30 Sharpe +1.78 — better than Phase 3 R3 +1.89 was within noise of, and **convergence reached at 30k vs R3's 50k**.

**Result.** The GPU framework eclipses the previous best at one-third the wall time. The redesign was worth it.

### Phase 8 — GPURolloutBuffer (CUDA-resident) *(2026-05-01 evening)*

**Bottleneck.** Even with GPU env, rollout buffer was numpy on host. Every PPO update did host→device copies for each minibatch.

**Fix.** Wrote `gpu_rollout_buffer.py`: holds `obs / actions / values / log_probs / rewards / advantages / returns` as CUDA tensors. PPO update reads directly from device memory — zero copies.

**Outcome.** fps 1490 → 1820. GPU util mean 62 % → 71 %. VRAM +1.2 GB (acceptable).

### Phase 9 — IndexOnlyRolloutBuffer + n_steps=1024 / Index-Only Observations *(2026-05-01 late evening)*

**Bottleneck identified.** Even with cuda-resident buffer, each entry stored `obs = (n_stocks × n_factors)` float32 = 3014 × 64 × 4 = 770 KB per step. At `n_envs=16, n_steps=1024`: 16 × 1024 × 770 KB = **12.6 GiB** rollout buffer. We were paying for storing the entire factor cross-section in memory N_envs × N_steps times.

**Insight.** All observations are slices of the same panel. Store **only the date index** (4 bytes) and gather on-the-fly during minibatch.

**Fix.** `IndexOnlyRolloutBuffer`: stores `date_idx[n_envs, n_steps] int32` (~64 KB) + `holdings_mask[...] bool` (~250 KB). Gather `panel[date_idx[batch]]` at minibatch read time. Effective batch size keeps the same numerical behavior; memory is **200× smaller**.

**Outcome.** Rollout buffer 12.6 GiB → 0.06 GiB. **Freed VRAM lets us raise `n_envs=16 → 32` and `n_steps=1024 → 2048`** within the same 12 GB budget. fps 1820 → 2050. GPU util peak 88 %.

### Phase 10 — Optimizer-Orphan Bug + LayerNorm + Dual Pooling *(2026-05-01 night)*

**Bug.** `value_loss` plateaued around 4e-3, never broke below. Suspected vanishing gradients in the value head. Inspected: **value head's parameters were not in the optimizer**. SB3 default uses one optimizer for both policy and value when they share an extractor; my custom policy split them and only registered policy params.

**Fix.** Explicit `optim.AdamW([{params: policy.parameters()}, {params: value_net.parameters(), lr: 1e-3}], lr=3e-4)`. Value head learning unlocked. `explained_variance` climbed 0.78 → **0.99** within 50k steps.

**Two more additions in this phase:**

- **LayerNorm after each per-stock MLP layer.** Real-data cross-section z-scores still have outliers (after `nan_to_num`, a single inf cell at the cell level can still pull mean). LayerNorm gives stable gradients. ~Verified ablation: removing LayerNorm gave value_loss instability across seeds.
- **Dual pooling head**: aggregate per-stock representations by `concat(mean, max)` instead of pure mean. Mean captures average market state; max captures the most-extreme stock signal. Worth +0.06 explained_var on the 50k smoke.

**Outcome.** fps 2050 → 1980 (the LayerNorm + dual pool cost ~3 %, but `explained_var=0.99` was worth it). bf16 was attempted but eliminated in Phase 11.

### Phase 11 / 12 — bf16 / adaptive target_kl (eliminated) *(2026-05-01 night)*

**Phase 11.** Tried `torch.autocast(dtype=bfloat16)` for matmul+linear. Memory −20 %, fps +12 %. **But**: `approx_kl` became unstable, occasionally spiking to 0.3 (vs nominal 0.02). Inspected: tail of policy logits at bf16 dynamic range had quantization noise that compounded in KL computation. **Rolled back to fp32**.

**Phase 12.** Tried `target_kl=0.10` adaptive (raise to 0.15 if violated 3x in a row). PPO update frequency dropped from every rollout to ~70 %. Total update count similar, value loss similar, IC similar — **no measurable benefit**. **Removed for code simplicity.**

### Phase 13 — PPO SGD perf-probe / Profiling *(2026-05-01 late night)*

**Goal.** Profile a single PPO update with torch.profiler to find any remaining low-hanging fruit.

**Findings.**
- 53 % of SGD time was in advantage computation (`compute_returns_and_advantage`).
- 22 % in policy log-prob recomputation.
- 11 % in value head forward.
- 14 % in `optimizer.step()`.

**Fix.** Wrote `compute_returns_and_advantage_vectorized()` that uses prefix-sum on CUDA tensors instead of Python for-loop over time steps. **53 % → 7 %**. SGD wall time per update −40 %.

**fps 1980 → 2580.** GPU util mean 78 %.

### Phase 14 — TF32 + unique-date + 1M overnight / TF32 + Unique-Date + 1M Overnight *(2026-05-02 → 2026-05-03 early)*

**Goal.** Capacity build. Run 1M steps overnight on the post-Phase 13 stack to test stability and final convergence.

**Two micro-optimizations** going in:

- **TF32**: `torch.backends.cuda.matmul.allow_tf32 = True; cudnn.allow_tf32 = True`. Free ~12 % matmul speedup on RTX 4070 (Ampere) with no measurable accuracy loss.
- **Unique-date dedup**: same date often appears multiple times in a 16-env × 2048-step rollout because envs reset asynchronously. Detected and gathered only unique `date_idx` once per minibatch, then duplicated rows. Saves ~30 % gather time.

**1M-step overnight run** (alpha101+gtja191 296-col on short panel; n_envs=16, n_steps=2048, target_kl=0.05 fixed, lr=3e-4 → 1e-5 linear, ent_coef 0.01 → 0):

- Wall time: **5h 47min**
- fps mean: **460** (started 326 from cold start, reached 540 at ~200k)
- GPU util mean: 73 %
- Peak VRAM: 9.8 GiB / 12 GiB
- approx_kl trajectory: stable 0.025-0.045, no spikes
- explained_variance: 0.99 from ~150k onwards
- Final OOS top30 Sharpe: **+5.83** (vs random p50 +0.62, vs-p50 +5.21)

**Outcome.** Stack hardened. Phase 14 is the "ready for real experiments" milestone. All Phase 15+ modeling experiments run on this exact stack.

**Section recap.** From Phase 0's 1k-step smoke (fps 50ish) to Phase 14's 1M-step overnight (fps 460), the framework went through **15 stack diffs, 24 documented bugs, and 5 mid-flight algorithmic redesigns**. The biggest single jump was Phase 5/6/7 — moving rollouts onto the GPU **3.25×'d fps**, and the symmetry-correct per-stock encoder simultaneously **reduced parameter count by 8000×** (395 M flat MLP → 50 k Deep Sets) while improving OOS Sharpe.

---

## §5 模型实验史 — Phase 15 → 26 <a id="5-模型实验史--phase-15--26"></a>
## §5 Modeling Experiment History — Phase 15 to 26 <a id="5-modeling-experiment-history--phase-15-to-26"></a>

**中文.** Phase 14 之后框架不再变了。Phase 15-26 是**在固定 stack 上跑模型/数据/奖励实验**。每个 phase 都有明确假设、单变量改动、量化决策。

**English.** After Phase 14 the framework stopped changing. Phases 15–26 are **model / data / reward experiments on a frozen stack**, each with a clear hypothesis, single-variable change, and quantitative decision.

### Phase 15 — RL serving 集成 / RL Serving Integration *(2026-05-02 ~ 03)*

**Goal.** Take the best 1M-step model from Phase 14 and integrate it into the AurumQ platform for live serving.

**Three SB3 PPO models registered:**

| Agent ID | OOS Sharpe | Note |
|---|---:|---|
| `phase15e_150k_grand_champion` | **+6.277** | active production model |
| `phase15e_100k_alt_peak` | +5.94 | alternative early-peak ckpt |
| `phase15e2_225k_continuation_peak` | +5.83 | continuation of Phase 15e but with extended steps |

**Bundle layout** `models/rl/<id>/`:
- `policy.zip` — SB3 native (kept for `PPO.load` path)
- `factor_schema.json` — factor name list, ordering, hash
- `metadata.json` — train_start_date, train_end_date, stock_codes, feature_group_weights, factor_count, policy_class
- `golden_inference.json` + `golden_obs.npy` + `golden_scores.npy` — sanity-check pair persisted with the bundle
- `checksums.json` — sha256 of all artifacts

**5 admin-only endpoints** under `/api/v1/rl/agents/`:
- `POST /import-bundle` — upload, validate schema, persist
- `POST /{id}/validate` — replay `golden_obs` through policy, compare to `golden_scores`
- `POST /{id}/archive` — soft delete
- `POST /{id}/inference` — async job (returns job_id)
- `GET /api/v1/rl/inference-jobs/{job_id}` — poll

**Tech-debt addressed.** SB3 `PPO.load(device='cpu')` + LRU(3) policy cache + single-flight lock to prevent concurrent reloads under high traffic.

**Frontend.** `ModelHashBadge`, `MissingDataAlert`, `RlAgentsView`, `RlStockPicksView`, `useInferenceJob` composable.

### Phase 16 — 修复 eval bug 重新基线 / Eval Bug Fixes Force New Baseline *(2026-05-03, 4h)*

**Goal.** Re-validate Phase 15's "drop `mkt_*` group helps" finding under three independent bug fixes.

**Three bugs:**

1. **Reward double-shift fix.** `FactorPanelLoader` was already encoding the fp-day forward return, but the env AND the importance-permutation pass were re-indexing `t+fp` ⇒ rewards were `fp` days too late.
2. **Sharpe over-annualization.** Overlapping fp-day forward returns must be annualized by `√(252/fp)`, not `√252`. For fp=10, that's an inflation factor of ~3.16×. Phase 15's legacy Sharpe +6.277 includes this inflation; the **bug-corrected** Sharpe is ~+1.99 on the same data.
3. **`mfp_` prefix silently missing.** 12 mfp cols had been dropped from `FACTOR_COL_PREFIXES`. Training input was 343 cols, not 355.

**Models retrained.** 16a (drop `mkt_` only), 16b (drop `mkt_+gtja_`), 16c (extend 16b to 450k).

**Key findings (under bug-fixed eval):**

| Run | adj Sharpe | vs random p50 | IC | Note |
|---|---:|---:|---:|---|
| 15e legacy (uncorrected) | +6.27 | — | — | annualization artifact |
| **16a (drop mkt_)** | **+1.593** | **+0.428** | +0.0143 | new prod candidate |
| 16b (drop mkt_+gtja_) | +1.32 | −0.27 | +0.0109 | gtja_ is load-bearing, contrary to Phase 15 belief |
| 16c (16b @ 450k) | +1.36 | −0.23 | +0.0112 | extension didn't help |

**Two robustly anti-helpful groups emerged** in permutation importance: cyq (−0.142 ± 0.044) and inst (−0.115 ± 0.030). mfp turned out weakly positive (+0.047 ± 0.067), gtja_ load-bearing (+0.160 ± 0.126).

**Decision.** 16a → production. Phase 15 legacy peaks retired as annualization artifacts. Phase 17 scoped to (a) seed-sweep 16a (b) test "drop cyq+inst" hypothesis.

### Phase 17 — 种子鲁棒性 + 条件重要性陷阱 / Seed Robustness + The Conditional-Importance Trap *(2026-05-03, 7h)*

**Goal.** Test whether Phase 16's "robust anti-helpful cyq+inst" signal transfers under retrain; measure seed dispersion.

**Method.** 17A: train drop_mkt+cyq+inst at seed=42, 300k. 17B/C/D: re-run 16a at seeds 1/2/3. 17E: extend 16a (seed=42) to 450k.

**Key findings.**

- **17A failed catastrophically.** adj Sharpe = +0.861, vs-p50 = **−0.304** (16a was +0.428). The cyq+inst drop hypothesis is **FALSE**. The "robust negative permutation importance" signal turned out to be **conditional on the trained policy, not causal**.
- **Seed sweep.** 3/4 seeds beat random; mean lift +0.249; range [−0.060, +0.428]. seed=42 sits at the upper edge of the noise band; seed=2 is a lone failure.
- **17E (450k) produced no new peak.** Phase 16a's +1.593 at step 224k is confirmed as the seed=42 global maximum.

**Critical lesson.** **Stop chasing factor drops based on permutation importance alone.** Permutation importance reflects **what the trained policy uses**, not **what is causally helpful for prediction**. To test causality you must retrain after the drop.

### Phase 18 — 多种子集成 / Multi-Seed Ensemble *(2026-05-03, 6h)*

**Goal.** Convert the seed-sensitivity finding into a deployable mitigation.

**Method.** Add seeds 4-7 (18A-D) at the unchanged 16a config. Build rank-mean / z-mean / z-median ensembles.

**Key findings.**

| Run | adj Sharpe | vs random p50 |
|---|---:|---:|
| seed=42 (16a) | +1.593 | +0.428 |
| seed=1 (17B) | +0.97 | +0.115 |
| seed=2 (17C) | +0.40 | −0.060 — failure |
| seed=3 (17D) | +1.42 | +0.408 |
| **seed=4 (18A)** | **+1.917** | **+0.752** — single-seed big win |
| seed=5 (18B) | +1.84 | +0.596 |
| seed=6 (18C) | +0.92 | +0.080 |
| seed=7 (18D) | +0.27 | −0.140 — failure |

Across 8 seeds: mean vs-p50 +0.352, median +0.388, **win rate 6/8**. **Top-K Jaccard between seeds was 0.003–0.010** — seeds chose almost completely disjoint baskets. This is exactly why ensembling lifts.

**The 6-member rank-mean ensemble** (excluding seeds 2 & 7): vs-p50 **+0.711** (Δ vs seed=42 alone = +0.283), IC = **+0.0278** (1.94× 16a), non-overlap Sharpe +1.938.

**Decision.** Ensemble passed strong-candidate gate. But: single-OOS-window optimum ≠ production Sharpe. Keep 16a live; advance ensemble as candidate pending fresh post-2026-04 holdout.

### Phase 19 — 执行约束 + 多窗口验证 / Execution Constraints + Multi-Window Validation *(2026-05-03, 6h)*

**Goal.** Stress-test `ens_rankmean6` against realistic costs, T+1 / limit-down filters, multi-window stability, and seed=4's contribution distribution.

**Method.** Quarter blocks, rolling 60-day windows (step 20), execution simulation at 30/60/100 bps round-trip with limit-down deferral.

**Key findings.**

- **3/3 quarters won. 100 % rolling-60d win rate. 7/7 windows IC-positive.**
- **Post-cost at 60 bps**: ensemble adj_S = **+0.971** vs 16a **+0.579** (Δ +0.392).
- **At 100 bps**: gap **widened** to +0.272 vs −0.233 — **ensemble stayed positive when 16a flipped negative**.
- **Seed=4 forensics warning**: **100.6 % of seed=4's marginal lift came from a single month (2026-01)**. Removing 16a from the ensemble actually improved score slightly (+0.045) — 16a was the weakest of the six.
- **Fresh holdout check**: 0 dates past 2026-04-24 with fp=10, threshold ≥ 40 — **INSUFFICIENT**.

**Decision.** Conclusion locked by data freshness. Phase 16a stays as live production; ensemble remains release-candidate. **No factor drops based on importance alone.** Phase 20 priority: collect fresh holdout.

### Phase 20 — 长 panel 训练 / Long-Panel PPO *(2026-05-05)*

**Goal.** Retrain 16a config on the long panel (2018-01 → 2025-06 train, 2025-07 → 2026-04 OOS) and check whether more history improves.

**Two seeds:** 20A seed=42, 20B seed=4. Each 300k steps on the new 7y panel.

**Key findings.**

| Run | adj Sharpe | OOS hit_rate@5 | OOS win_rate |
|---|---:|---:|---:|
| Phase 16a (2y train) | +1.593 | 4.88 % | 36.9 % |
| **Phase 20A (7y, seed=42)** | **+1.78** | 4.94 % | 37.4 % |
| Phase 20B (7y, seed=4) | +1.42 | 4.85 % | 37.1 % |

**Combined-evidence.** 2-seed average vs 16a: +0.012 adj Sharpe (within noise). The long panel is **at parity** with the short panel for the RL track on this metric.

**20C cross-data ensemble: BLOCKED.** Ensembling the 7y-trained policy with the 2y-trained policy required obs_dim alignment, but the 7y policy was trained with 2018-2019 data that includes ~600 stocks that delisted before 2025; padding caused obs_dim mismatch. Decision: defer to Phase 21.

**Decision.** Long panel doesn't help RL track (will revisit in Phase 22 after reward redesign). Phase 19's 16a stays live. The long-panel data **does** later become foundational for the SL track (see §6: P0 / Path 1 long / path5_long).

### Phase 21 — V2 forward_10d 拒绝 / V2 forward_10d REJECTED *(2026-05-05 ~ 06)*

**Goal.** Try a brand-new V2 architecture (Dict observation space + larger transformer-style head) on the existing forward_10d reward.

**Result.**

| Run | adj Sharpe | hit_rate@5 | win_rate | avg_hold |
|---|---:|---:|---:|---:|
| Phase 16a (V1) @ top_k=3 | +1.59 | 4.88 % | 36.9 % | +0.20 % |
| **Phase 21A (V2 forward_10d)** | **−0.60** | **3.70 %** | **34.5 %** | **−0.16 %** |

**Catastrophic regression.** V2 Dict obs scheme broke something about the policy's gradient flow; or maybe the Transformer attention head on 3014 stocks is too parameter-heavy for the 300k-step budget. Three retries with seed sweep — same result.

**Decision.** **V2 rejected.** V1 `PerStockEncoderPolicy` stays canonical. Phase 22 reverts to V1 architecture but changes the reward function (see below).

### Phase 22 — 主升浪奖励重设计 / Main-Wave Reward Redesign *(2026-05-06)*

**Goal.** The forward_10d reward target is "average 10-day forward log-return". But what we actually care about is **realized return until a sensible exit signal fires**. Implement an MA5/MA10 death-cross exit with a 5-day hard cap.

**Method.** New module `src/aurumq_rl/main_wave_labels.py` computes per-stock per-day `hold_return[t, j]` under signal-exit (`min(5d, MA5<MA10 death cross)`). The env's reward function reads from this tensor instead of computing forward returns on-the-fly. The `valid_mask` is tightened to `entry_eligible & label_valid` so training reward and OOS evaluation filter on the same criterion.

**Three runs (8h each on RTX 4070, short panel 2023-01..2025-06 train / 2025-07..2026-04 OOS).**

- **22A**: seed=42, top_k=5, 300k steps
- **22B**: seed=1, top_k=5, 300k steps (robustness check)
- **22C**: seed=42, top_k=3 train, 200k steps (concentration check)

**Key findings.**

| Run | top_k | best step | hit_rate | win_rate | avg_hold | avg_dd | eval_score |
|---|---:|---:|---:|---:|---:|---:|---:|
| Phase 16a (V1 forward_10d, prod) | 3 | 224928 | 4.88 % | 36.9 % | +0.20 % | 3.00 % | +0.0490 |
| Phase 21A (V2 forward_10d, REJECTED) | 3 | 149952 | 3.70 % | 34.5 % | −0.16 % | 4.71 % | −0.0596 |
| **Phase 22A (V1 main_wave seed=42)** | 3 | 299904 | **5.89 %** | 41.1 % | +0.44 % | 3.68 % | +0.0419 |
| Phase 22B (V1 main_wave seed=1) | 3 | 24992 | 6.06 %* | 40.2 % | +0.18 % | 3.79 % | +0.0168 |
| **Phase 22C (V1 main_wave train_topk=3 → eval@5)** | 5 | 174944 | **6.16 %** | **44.0 %** | **+0.62 %** | 3.84 % | **+0.0505** |

*22B's "best" is at step 24992 (≈ 1.5 PPO iterations); subsequent regress.

**Three deltas vs Phase 16a baseline.**

1. **Hit rate**: random base is 5.72 %. Forward_10d models all below random (−0.84 to −0.87 pp). main_wave models seed=42 series consistently **above random** (+0.14 to +0.44 pp).
2. **Win rate**: +4 to +7 pp uniform lift across all three runs / all top_k variants.
3. **Avg hold return**: 3× lift (16a +0.20 % → 22C +0.62 %).
4. **Drawdown trade-off**: slightly higher `avg_max_drawdown` (3.65–3.95 % vs 16a's 3.00 %). The hold_return reward doesn't directly penalize drawdown — model is willing to ride larger in-hold drawdowns to capture larger gains. Net positive on eval_score, worth a Phase 23 fix.

**Decision.** Phase 22C → production candidate. First time a model **clears the 5.72 % random baseline** on the main_wave criterion.

### Phase 23 — Episode-target 清理 / Episode-Target Cleanup *(2026-05-06)*

**Goal.** Clean up valid_mask edge cases and tighten the eval pipeline. Specifically:
- last-week-of-data label_valid sometimes True because label window extends past panel end → forward fill gives +0 % return.
- T+0 entry/exit on same day (entry day = exit day, hold_return = 0, treated as "valid loss") was double-counted.

**Method.** Three patches in `main_wave_labels.py`: (a) strict label window check; (b) drop t==exit_t entries; (c) recompute `entry_eligible` to exclude T+0 (only buy after 9:30 of day t, exit at or after day t+1 close).

**Result.**

| Run | hit_rate | win_rate | avg_hold | eval_score |
|---|---:|---:|---:|---:|
| Phase 22C | 6.16 % | 44.0 % | +0.62 % | +0.0505 |
| **Phase 23A (same config + cleanup)** | **6.42 %** | **44.6 %** | **+0.71 %** | **+0.0617** |

**Production candidate locked.** Phase 23A is the first model that combines: (a) clears random base rate, (b) >44 % win rate, (c) >0.7 % avg hold return, (d) tight label semantics. Becomes "23A baseline" in subsequent phases.

### Phase 24 / 25 — 技术因子改训 + 重要性权重 — 全部拒绝 / Tech-Factor Detour + Importance-Weighting REJECTED *(2026-05-07)*

**Goal.** Add ~36 technical-analysis factors (MA / KDJ / MACD / Bollinger / amplitude / limit-up counts) on top of the 353-col baseline. Secondarily test per-factor importance-derived input weights.

**Method.**
- **Phase 24A**: compute tech factors at panel-load time inside the RL repo (KDJ/MACD computed from close-only because parquet had no OHLC). Train 300k seed=42 + `--add-technical-factors`.
- **Phase 25A**: add IG-saliency × |T-1 z-score| sigmoid weights on top of 24A.
- **Phase 25D**: weighting on the 353-col base WITHOUT tech, to isolate the weighting paradigm itself.

**Results.**

| Run | top-5 T-1 hit | lift vs random | Note |
|---|---:|---:|---|
| 23A baseline | 2.11 % | **2.38×** | the reference |
| **Phase 24A (tech, 353+36=389 cols)** | **0.40 %** | **0.45×** | **below random 0.89 %** |
| Phase 25A (24A + weighting) | 0.50 % | 0.56 % | VRAM-thrashed (96 % on RTX 4070), fps 172 → 4, killed at 60 % |
| Phase 25D (weighting only, no tech) | 1.41 % | 1.59× | **−33 % vs 23A** |

**Root causes** (post-mortem):

1. **KDJ / amplitude approximated from close-only.** The parquet didn't carry OHLC at the time. `kdj_k = 100 * (close - min(close,9)) / (max(close,9) - min(close,9))` is a degenerate variant of true KDJ. Z-scoring this approximation produces strong artifacts on quiet stocks.
2. **Binary event flags pollute LayerNorm gradients.** `ma_cross_5_10`, `golden_cross` are 0/1 indicators. After z-score, "1" days become ~3-5σ outliers; LayerNorm scales them down, but back-propagated gradient hits these few outlier samples and produces large updates that destabilize the policy.
3. **Weighted top-30 factors saturate encoder capacity.** The per-stock encoder is 64-dim. Forcing 30 high-weight factors through a 64-dim bottleneck kills fine-grained ensemble structure that previously distributed signal across the full 353 cols.

**Decision.** **Both directions rejected.** Architecture rule re-affirmed: **factor computation belongs in the upstream parquet pipeline, never at panel-load time inside the RL repo**. The team wrote `TECH_FACTOR_SPEC.md` (203 lines) and handed it back to the data team for proper OHLC-based tech factors. 23A remained production. Importance-weighting paradigm permanently dropped.

### Phase 26A → G — cyq 修复 + 事件衰减编码 / cyq Fix + Event-Decay Encoding *(2026-05-07 ~ 08)*

**Phase 26 is a 7-step recovery and breakthrough chain.** Track it carefully — this is where the project's current production model came from.

#### Phase 26A: 加入 v1.0 cyq + 30 tech 列 / Add v1.0 cyq + 30 tech cols

**Result.** 373-col training (343 base + 30 upstream tech_/cmf_/zt_) with new "canonical" cyq replacing legacy 88%-NaN cyq from `quotes_enriched`. **Regressed**: T-1 lift = 1.36× (vs 23A 2.38×).

#### Phase 26B-baseline: 移除 30 tech 列, 保留 v1.0 cyq / Remove tech, keep cyq

**Result.** 343-col, no tech. Still regressed: T-1 lift = 1.47×. **The regression is NOT from tech factors.**

#### Phase 26 root-cause analysis: cyq backfill regime shift

Traced to **`cyq_perf` backfill regime split**:
- v1.0 cyq table only had **real Tushare data ≥ 2025-10-20**; everything before was backfilled with a different methodology.
- Backfill null rate **0.61 %** vs real **26.53 %**.
- `cyq_cost_distance` std: **0.197** (backfill, in training window) vs **0.066** (real, in OOS) — a **3× compression**.
- Training was **100 % backfill**; OOS was **~63 % real**.
- **Cross-section z-score does NOT equalize a mid-stream regime shift.** The model learned the synthetic-backfill scale; on real data the dispersion is 3× smaller, the model's "this stock has high cyq" signal collapses.

**23A's accidental advantage.** 23A used the legacy 88%-NaN cyq, so the model had effectively learned to ignore cyq entirely. By contrast 26A/B's cleaner-but-distribution-shifted cyq actively misled the model.

#### Phase 26C: 343-col + v1.2 cyq, wrong train window

**v1.2 cyq fix from data team**: re-fetch via bulk Tushare API, all dates have consistent methodology. But: train end date was 2024-12-31 (vs 23A's 2025-06-30 — 6-month gap before OOS start).

**Result.** T-1 lift 1.47× (same as 26B). Train-window mismatch swamps any cyq improvement.

#### Phase 26C2: 353-col 23A-exact config + v1.2 cyq + correct train window ⭐

**Result.** T-1 lift **2.61×** (2.31 % hit rate, best ckpt at **step 50k**). **+9.7 % over 23A's 2.38 %, AND converges 4× faster** (50k vs 200k). v1.2 cyq fix is validated. **Production candidate.**

#### Phase 26D: 26C2 + 30 tech cols (clean panel)

**Result.** T-1 lift **1.13×**. Adding 30 tech factors on the clean panel still hurts by −57 %. Confirms Phase 24's diagnosis: at 128→64→32 per-stock encoder capacity, the 30 raw tech cols dilute attention from strong alpha/gtja/mfp signals.

#### Phase 26E/F/G: 事件衰减编码 / Event-Decay Encoding

**Three new variants** at fresh 3-seed baseline:

- **26E**: 26C2 (353 cols) + 2 continuous tech cols (`tech_boll_percent`, `cmf_120d_pct_amt`) = 355 cols
- **26F**: 26E + 6 event-decay cols (`tech_evt_*` with τ=10d exp decay) = 361 cols
- **26G**: 26F at bigger encoder (256→128, 256k params per-stock)

3-seed median (seeds 42/43/44):

| Run | factors | T-1 lift median | T-1 hit median | best ckpt |
|---|---:|---:|---:|---|
| 26C2 (clean panel sanity) | 353 | 1.70× | 1.50 % | step 50k |
| 26E | 355 | 1.59× | 1.41 % | step 50k |
| **26F (event-decay)** | **361** | **2.15×** | **1.90 %** | **step 50k** |
| 26G (bigger encoder) | 361 | (abandoned) | — | — |

**Best 26F seed**: seed=44 hit **2.72×** lift at step 50k (T-1 hit 2.41 %).

**26G abandoned**: 256→128 encoder on RTX 4070 12 GB thrashed fps from 326 down to 4–55 with VRAM stranded by zombie contexts. 3 hours of attempts, no clean result. The capacity question is deferred to Linux / 16 GB-class hardware.

#### Phase 26F-v3 / G-v3: clean-panel re-verification *(2026-05-08)*

Re-run the comparison on the panel-v3 build (cyq v1.2 fully shipped, alpha/gtja sanitizer integrated upstream, mf_ `_log` variants emitted).

| Run | T-1 lift median | range |
|---|---:|---:|
| 26C2-v3 sanity | 1.70× | 1.36–2.15× |
| **26F-v3 ⭐ (PRODUCTION)** | **2.27×** | **2.04–2.38×** |
| 26G-v3 | 1.82× | 1.59–2.05× |

**Decision.** **26F-v3 = final RL production model.** 361 cols (353 base + 2 continuous tech + 6 event-decay) at 128→64 encoder. Bigger-encoder hypothesis closed (disproven on clean panel too — not a hardware artifact).

**Phase 26 lesson summary.**

1. The win came from **continuous event-decay encoding of binary signals** (the explicit fix to Phase 24's binary-flag failure), not from continuous TA on close prices.
2. Mid-stream regime shifts (cyq v1.0 backfill vs real) defeat cross-section z-score; only the data team can fix it (v1.2 bulk-API recompute).
3. Train-window matters as much as architecture. A 6-month gap between train-end and OOS-start is enough to wipe out any cyq-quality improvement.

---

## §6 监督学习对照赛道（paris 侧）<a id="6-监督学习对照赛道paris-侧"></a>
## §6 Supervised-Learning Companion Track <a id="6-supervised-learning-companion-track"></a>

**中文.** 与 RL 赛道并行，paris 侧（AurumQ 主仓）维护了一条 LightGBM/CatBoost/XGBoost 监督学习赛道作为对照。这条赛道**已经超越了 RL 赛道的实证表现**，但 RL 仍然作为多样性来源继续生产。下文是 SL 赛道关键节点的精简记录。

**English.** In parallel with the RL track, paris (AurumQ main repo) maintains a LightGBM/CatBoost/XGBoost supervised-learning track as control. The SL track **has empirically outperformed the RL track**, but RL remains in production as a diversity source. Below is a compressed record of the SL track's key milestones.

### 6.1 P0 — Wave Label 消融 / Wave Label Ablation *(2026-05-09)*

**Goal.** Among four labeling methods (A v2_excess_adaptive, B trend-scanning, C triple-barrier, D directional-change), pick the production main-wave label.

**Method.** Calibrate every method to train pos_rate ≈ 0.80 %, train LGBM (3 horizons each — t1, t3, e20) on the 26F-v3 348-col panel, score on test 2025-07..2025-12. Decision weights: 0.45·PR_AUC + 0.20·(1−ECE) + 0.15·top1 %_lift + 0.10·(1−ind_cv) + 0.10·(1−year_cv).

**Stage 2 test PR-AUC at t3:**

| Method | best_iter | PR_AUC | lift | Brier_ratio | ECE | top1 % | top5 % | daily_prec@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **A** | 191 | **0.1217** | **3.0×** | 0.972 | **0.010** | 3.65× | 4.21× | **0.200** |
| B | 196 | 0.1052 | 2.4× | 0.977 | 0.013 | 4.16× | 3.35× | 0.156 |
| C | 162 | 0.1195 | 2.5× | 0.968 | 0.015 | 4.25× | 3.70× | 0.203 |
| D | 8 | 0.0961 | 2.5× | 0.971 | 0.005 | 2.31× | 3.25× | 0.114 |

Composite-decision score: A = 0.836, C = 0.815 (within tiebreak band 0.03). C wins industry_cv tiebreak (0.40 vs 0.51); A wins on operational clarity + 15d vs 5d median event duration.

**Null tests** both **PASS**: label-shuffle PR_AUC = 0.04021 (0.989× base), date-shuffle 0.06060 (1.491× — borderline, threshold 1.5×). Real model lift 3.0× / shuffle lift 1.49× → **2.0× fresh-signal ratio**.

**Decision.** P0 = **Method A at horizon t3**, threshold τ_A = 1.2327, locked in `src/aurumq/labeling/p0_chosen.py`. C kept as drop-in fallback. P0 cleared for P1 production deployment.

### 6.2 P2 — Production Hardening *(2026-05-09)*

**Goal.** Close P1's training gaps so the LGBM is production-grade, not an ablation prototype.

**Pieces.**

- **Stage 0**: Daily panel rebuild. Split monolithic `feature_panel_v3_344.parquet` (3.65 GB) into 804 per-day shards (`year=YYYY/date=YYYY-MM-DD.parquet`). Celery beat @18:35 between phase20 rebuild (18:30) and wave_scores (18:45). Schema-hash assert against P0 lock `5e71e158e331`.
- **Stage 1**: Walk-forward on single anchor 2025-07, **5 seeds × 3 horizons** (t1, t3, e20) = 15 LGBM trainings. LGBM params locked from P0 (lr=0.02, num_leaves=63, n_est=1500, early_stop=80), per-seed isotonic, mean ensemble.
- **Stage 2**: composite_mean(A, C) finishing — **REJECTED** (PR_AUC = 0.0433 vs A_t3's 0.1217, Δ = −0.0785). Composite labels mix A∪C event positions that don't overlap, raising noise.
- **Stage 3**: PPO residual — SKIPPED, deferred to P3 4070 work.
- **Stage 4**: Alembic 052, v1/v2 dual-write, `drift_check @19:00`.

| Horizon | test_pos_rate | PR_AUC | lift | ECE | top1 % | daily@5 | per-seed std |
|---|---:|---:|---:|---:|---:|---:|---:|
| **t1** | 0.0135 | **0.0721** | **5.34×** | 0.0024 | **9.28×** | 0.103 | 0.002 |
| **t3** | 0.0407 | 0.1224 | 3.01× | 0.0100 | 3.33× | **0.209** | 0.001 |
| **e20** | 0.2650 | 0.4136 | 1.56× | 0.0477 | 1.96× | 0.548 | 0.002 |

**Decision.** Promote `wave_t3_lgbm_v1` (P0 single seed) → `wave_t3_lgbm_v2.anchor=2025-07.ensemble` (5 seeds) under tiered "realistic gate": t3 + t1 ship, e20 reference-only.

### 6.3 Multi-Path SL Exploration: Path 1 / 2 / 4 / 5 / 6 *(2026-05-10)*

**Diversity exploration.** 4 new bundles + 1 meta-stack on the same H1/H2 windows.

| Path | Input panel | Preprocessing | n_features | Model |
|---|---|---|---:|---|
| Path 1 | feature_panel_v3_344 | **NO rank-z** (raw) | 345 | LightGBM β-regression |
| Path 2 | feature_panel_clean | rank-z (same as Path 4) | 345 | **CatBoost + XGBoost mix** |
| Path 4 | feature_panel_clean | rank-z | 345 | LightGBM (prod) |
| Path 6 | feature_panel_clean_pruned | rank-z + drop 119 SHAP-zero | 226 | LightGBM (Bayesian opt) |
| Path 5 | meta-LGB over (Path1, Path4, Path2) | + 11 regime + 9 interactions | 23 meta | LightGBM meta |

**Short-panel scoreboard** (H1 = 2025-07..09, H2 = 2025-10..12):

| Path | H1 cal primary | H2 cal primary | T1_hit H1 |
|---|---:|---:|---:|
| Path 1 | +0.028030 | +0.030497 | 54.4 % |
| **Path 4** | **+0.028483** | **+0.030577** | 54.5 % |
| Path 2 | +0.027932 | +0.030658 | 54.0 % |
| Path 6 | +0.028265 | +0.030739 | 54.5 % |
| **Path 5 (meta)** | +0.028372 | +0.030245 | **55.8 %** |
| Path D (long, REJECTED as standalone) | +0.028358 | +0.030738 | — |
| Path 3 (TabNet, **REJECTED**) | +0.019417 | +0.020 | — |

**Path 3 TabNet's rejection.** H1 +0.019 vs Path 1 +0.028 (−30-40 %); 47 min train vs 50 s for LightGBM (~55× slower); killed 7/8 grid combos. Also drove T1_hit DOWN by ~1 pp when added to meta — excluded from Path 5 stacking. **TabNet is not the answer for this problem under this data scale.**

**Cross-path ensemble**: rank-mean across Path 1+4+2 gives H1 +0.028407 vs best single +0.028483 — Δ = −0.0001. **Three GBDT-family paths are highly correlated; ensemble diversity is exhausted.**

### 6.4 Long-Panel Sweep: The Climax *(2026-05-10 ~ 11)*

**Goal.** Use the 7-year long panel (2018-01-02..2024-12-04 train, same H1/H2 as short) to retrain Path 1/2/4/5. Test the rank-z hypothesis.

**Combined headline:**

| Path | short H1 | long H1 | Δ H1 | short H2 | long H2 | Δ H2 |
|---|---:|---:|---:|---:|---:|---:|
| Path 1 (raw LGB) | +0.028030 | +0.028626 | **+5.97 bps** | +0.030497 | +0.031089 | +5.92 bps |
| Path 4 (rank-z LGB) | +0.028483 | +0.028358 | **−1.25 bps** | +0.030577 | +0.030738 | +1.61 bps |
| Path 2 (CB+XGB) | +0.027932 | +0.028343 | +4.11 bps | +0.030658 | +0.030857 | +2.00 bps |
| Path 5 (regime stack) | +0.028372 | +0.028817 | +4.46 bps | +0.030245 | +0.031131 | **+8.86 bps** |

**Finding 1 — rank-z kills long-panel info.** Experiment B isolates the variable: Path 4 hyperparams + the same raw long panel = **numerically identical to Path 1 long** (H1 = 0.028626). The per-day cross-sectional rank operation erases cross-year factor amplitude information — fine for 2y where amplitude variation is small, catastrophic for 7y where most of the new info lives in cross-year amplitude regimes.

**Finding 2 — 5-year is the plateau** (Experiment A):

| Train window | n years | H1 mean | H2 mean |
|---|---:|---:|---:|
| 2023-2024 (2y) | 2 | +0.027928 | +0.030285 |
| 2022-2024 (3y) | 3 | +0.028182 | +0.030880 |
| **2020-2024 (5y)** | 5 | **+0.028529** | **+0.030953** |
| 2018-2024 (7y) | 7 | +0.028533 | +0.030983 (plateau) |

Strictly monotonic up to 5y then flat. **2018-2019 contribute nothing.** Implications: if retraining, use 2020-2024, save 30 % compute.

**Finding 3 — Strategy D compounds.** Top-50 score-weighted sizing:

| Path | + Strategy D H1 mean_y | + Strategy D H2 mean_y |
|---|---:|---:|
| Path 4 short + Strategy D (current prod v2) | +0.0308 | +0.0333 |
| **Path 1 long + Strategy D** | **+0.0315** | **+0.0343** |
| Δ | **+7 bps** | **+10 bps** |

Strategy D's +8 % concentration effect **stacks additively** on a stronger base — bigger absolute gains, not just same percentage of smaller pie.

**Three production candidates:**

| Candidate | H1 vs prod | H2 vs prod | + Strategy D vs prod v2 | Ops cost |
|---|---:|---:|---:|---|
| **C. Hybrid** (Path 1 long + Path 4 short 50/50) | +2.7 bps | +4.5 bps | +5-8 bps | **simplest** — 2 inferences + average, no new model |
| **A. Path 1 long** (raw, 2018-2024) | +1.4 bps | +5.1 bps | **+7-10 bps** | medium — 1 new bundle |
| **B. Path 5 long stacking** | +3.3 bps | +5.5 bps | strongest | high — 3 base + meta + 11 regime + 9 interactions |

**All three were shipped to production on 2026-05-11.** Hybrid + path1_long went live first (commit `78d71ce`); path5_long followed 30 minutes later after ledashi shipped the missing path4_long + path2_long base bundles (commit `0ab6a55`). **Today path5_long is the H1-leading model** (+0.02882) and is one signal-A/B-window from being promoted to `is_recommended=True` over Path 4 short.

---

## §7 实证发现 — 六条改变方向的结论 <a id="7-实证发现--六条改变方向的结论"></a>
## §7 Empirical Findings — Six Conclusions That Changed Direction <a id="7-empirical-findings"></a>

### Finding 1 — Rank-z 会销毁长 panel 跨年幅度信号 / Rank-z Destroys Long-Panel Cross-Year Amplitude

**中文.** 截面 z-score（rank-z）在每个 trade_date 内把当天因子重排到 [-σ, +σ] 区间。短 panel（2 年）训练时这没问题：所有因子的截面分布相似。长 panel（7 年）训练时灾难：2020 年和 2024 年的市场结构完全不同（科创板规模、北向资金占比、机构持仓比例都翻了几倍），而 rank-z 把跨年的「因子绝对幅度」信息全部抹掉，模型只能学到「相对排名」，反而损失 5-6 bps 的 H1 mean_y。

**English.** Cross-section z-score (rank-z) re-ranks every factor's per-day distribution to [-σ, +σ]. Fine for short panels (2y) where cross-section distributions are similar. Catastrophic for long panels (7y) where market structure shifts dramatically year-over-year (STAR board size, Northbound share, institutional holdings all multi'd over 7y). Rank-z erases the cross-year factor amplitude information; the model only learns relative ranks, losing 5-6 bps H1 mean_y.

**Action.** For long-panel training, **use raw features**. Verified: Path 1 long (raw, 7y) > Path 4 long (rank-z, 7y) by 5.97 bps H1 with identical hyperparams.

### Finding 2 — 5 年训练窗口是 plateau / Five Years is the Plateau

**中文.** 2018-2019 数据零边际贡献。原因：A 股市场在 2019 年下半年到 2020 年初经历了科创板开板、注册制改革、外资准入扩大 —— 实质上是个 regime change。把 2018-2019 数据塞进训练集相当于让模型同时学两个市场结构，跨年泛化变差。

**English.** 2018-2019 contributes nothing. Reason: A-shares went through a regime change in late 2019 / early 2020 (STAR opening, registration-based IPO reform, foreign-access expansion). Training on 2018-2019 forces the model to learn two market structures simultaneously, hurting cross-year generalization.

**Action.** Default train window is **2020-2024 (5y)**. Saves 30 % compute at same precision.

### Finding 3 — Strategy D 与任何 base 都正向叠加 / Strategy D Compounds Additively

**中文.** Strategy D = top-K 仓位按校准分数加权（`weight = max(score, 0) / sum`），而不是等权。在 Path 4 short 上 +8% mean_y；在 Path 1 long 上 +10% mean_y。**新指标提升的绝对值 ≈ 旧 base 提升的绝对值**，不是百分比。换 base 越强，Strategy D 累加越多。

**English.** Strategy D = top-K score-weighted position sizing (`weight = max(score, 0) / sum`), not equal-weight. Adds +8 % mean_y on Path 4 short, +10 % on Path 1 long. **Absolute improvement is the same whether base is weak or strong** — stronger base means larger combined gain. Always apply Strategy D.

### Finding 4 — 二值事件标志必须 exp-decay 编码 / Binary Event Flags Must Be Exp-Decay Encoded

**中文.** 二值事件（MA 金叉、KDJ 死叉、突破 3σ）原始 0/1 进 LayerNorm 是 −33% 准确率的元凶（Phase 24A）。正确做法：把二值序列做 `exp-decay τ=10d` 转换，`evt_decay(t) = Σ_{tau ≤ 10d} 1[event in last 10d] · exp(-(t-tau)/10)`，得到 0..1 连续衰减值。Phase 26F 用这个修复把 T-1 lift 从 1.13× 拉回 **2.27×**。

**English.** Raw binary event flags (MA cross, KDJ death-cross, 3σ breakout) directly into LayerNorm cause a −33 % accuracy regression (Phase 24A). Correct encoding: exp-decay `τ=10d`. Phase 26F applied this fix and brought T-1 lift back from 1.13× to **2.27×**.

### Finding 5 — Permutation importance 是 conditional, 不是 causal / Permutation Importance Is Conditional, Not Causal

**中文.** Phase 16 跑出 cyq 和 inst 都是「robust 负向」（permutation importance 显著负）。Phase 17A 把它们 drop 掉重训 → adj Sharpe 暴跌 −0.732。原因：permutation importance 度量的是**当前训好的策略有多依赖某个特征**，不是**这个特征是否对预测有因果贡献**。要测因果必须**重训**而不是**重 permutation**。

**English.** Phase 16 found cyq + inst both had "robust negative" permutation importance. Phase 17A dropped them and retrained → adj Sharpe collapsed −0.732. Reason: permutation importance measures **how much the trained policy depends on a feature**, not **whether the feature is causally helpful for prediction**. To test causality you must **retrain** after the drop, not just re-permute.

### Finding 6 — 大多数 inf 是上游数据创伤, 不是公式 bug / Most Infs Are Upstream Data Wounds, Not Formula Bugs

**中文.** Phase 26 跑 inf-root-cause 审计时，发现 19 个 inf-producing 因子的根因不是公式问题：

1. **adj_factor 损坏**：5 个日期 + 2026 全年 16.5 万 NULL → `compute_alpha101.adj.fill_null(1.0)` 把不复权价拼到复权价上，制造假 regime shift。
2. **一字板 high=low=close**：vwap-close、high-low 都 = 0，公式 div-zero。
3. **rolling 相关性 window=2**：std=0 触发 NaN/inf。
4. **rank^delta 当 rank=0 且 delta 极端**：`0^-large = inf`，gtja_017 实测 max 达 1.4×10³⁰⁸（fp64 上限）。

**英文 lesson**: sanitizer（inf→NaN + clip ±1e6）只是兜底，真正的修复在上游数据管线。

**English.** A Phase 26 inf-root-cause audit traced 19 inf-producing factors not to formula bugs but to four upstream data wounds: (1) adj_factor corruption (5 dates + all of 2026 = 165k NULL → `compute_alpha101.adj.fill_null(1.0)` stitched unadjusted prices to adjusted prices, creating fake regime shifts); (2) one-字板 days (high=low=close → vwap-close and high-low both zero, divide-by-zero); (3) rolling correlation window=2 producing std=0; (4) `rank^delta` with rank=0 and extreme delta blowing past fp64 (gtja_017 hit max 1.4×10³⁰⁸). **Lesson**: sanitizer (inf→NaN + clip ±1e6) is only a backstop; the real fix is upstream data.

---

## §8 生产流水线 — 每日 18:30-19:00 评分预算 <a id="8-生产流水线--每日-1830-1900-评分预算"></a>
## §8 Production Pipeline — Daily 18:30-19:00 Scoring Budget <a id="8-production-pipeline"></a>

**中文.** 当前生产环境（AurumQ 主仓侧）的 Celery Beat 调度（工作日）：

```
18:30  phase20_rebuild_panels_daily      重建 short combined panel + 上传 OSS
18:35  rebuild_feature_panel_daily       当日 shard 写入
18:45  generate_wave_scores_daily        v1/v2 集成（legacy 兼容）
18:50  generate_wave_scores_path4_daily  Path 4 + Strategy D（当前推荐）
18:51  path1 评分
18:52  path2 评分
18:53  path6 评分
18:54  path1_long 评分                   ★ 长 panel raw
18:55  path4_long 评分                   ★ path5_long base
18:56  path5 评分                        meta on path1+path4+path2
18:57  path2_long 评分                   ★ path5_long base
18:58  hybrid 评分                       ★ path1_long + path4 50/50
18:59  path5_long 评分                   ★ NEW BEST regime stack on long bases
19:00  wave_drift_check                  漂移监控
```

**关键点**：
- **10 个 model_version 并存**。任何模型上线，路径：`runner.py:PATH_CONFIG` 加一条 + `celery_beat.py:BEAT_SCHEDULE` 加一行 + `wave_scores.py:_PATH_DISPLAY_INFO` 加 display 元数据 → 自动出现在 `/api/v1/wave/model_versions` 接口。
- **预测就绪时间：每工作日 19:00 BST**。
- **生产推荐**：当前 `is_recommended=True` 仍在 Path 4 short；A/B 几周后切到 path5_long。

**English.** Current production Celery Beat schedule (weekdays) — see Chinese block above for the 14-slot 18:30→19:00 timeline. **Predictions are ready by 19:00 BST every weekday.** Adding a new model is a 3-line patch: add a `PathConfig` in `runner.py`, a beat entry in `celery_beat.py`, and a display row in `wave_scores.py`. It then auto-surfaces on `/api/v1/wave/model_versions`. **Current `is_recommended=True` is Path 4 short**; will A/B-promote `path5_long` after a few weeks of shadow.

---

## §9 工程教训 — 从踩坑到守则 <a id="9-工程教训--从踩坑到守则"></a>
## §9 Engineering Lessons — From Pitfalls to Operating Rules <a id="9-engineering-lessons"></a>

### 9.1 元守则 / Meta-Rules

**中文.**

1. **每轮只改一个变量**。复合实验不可解释。Phase 3 R2 同时改 `target_kl + n_envs + n_steps`，分不出哪个贡献了 +0.74 OOS Sharpe。
2. **永远先跑 10-30k step 的 micro-smoke**。Phase 3 R2 第一次尝试 + Phase 4 n_envs=20 实验都因为没先 micro-smoke 就跑 `MemoryError`。
3. **50k 步的 OOS Sharpe 是噪声**。从 smoke 排座次不靠谱；要从 ≥1M 步 convergence-scale 跑里挑赢家。
4. **不基于 permutation importance 做特征剔除**。Phase 17 已经栽过一次。要剔除必须 retrain ablation。
5. **重新审视基线 > 增量调参**。Phase 5 重构（per-stock encoder + GPU env）一次给了 10× fps + 5× 因子上限。所有 Phase 1-4 hyperparam 调整加起来都没这一次重构贡献大。
6. **VRAM ≠ RAM**。Phase 5 realization #2 教训。
7. **对称性正确的架构 > 蛮力容量**。50 K 参数的 per-stock encoder 比 800 M 参数的 flat MLP **更准 + 更快 + 更省**。
8. **数据收敛速度必须 > 网络容量增长速度**。Phase 2 把网络从 [64, 64] 加到 [2048, 1024, 512]（12 倍），但训练步数还是 50k，于是误判「网络容量不够」。
9. **管线对 ≠ 模型好**。R1 的「跑通端到端」和 R3 的「explained_var=0.99」是同等重要的里程碑。

**English.**

1. **One change per round.** Compound experiments are uninterpretable. Phase 3 R2 changed three things at once and couldn't attribute the +0.74 OOS Sharpe jump.
2. **Always micro-smoke at 10-30k steps first.** Phase 3 R2 first attempt and Phase 4 `n_envs=20` both died with `MemoryError` because nobody micro-smoked first.
3. **OOS Sharpe at 50k steps is noise.** Don't pick winners from smoke; pick from convergence-scale runs (≥1M).
4. **Don't drop features based on permutation importance alone.** Phase 17 ate this one. Always retrain ablation to test causality.
5. **Re-examining baselines > incremental tuning.** Phase 5 redesign (per-stock encoder + GPU env) gave 10× fps + 5× factor capacity at once. All Phase 1-4 hyperparam tuning combined didn't match it.
6. **VRAM ≠ RAM.** Phase 5 realization #2.
7. **Symmetry-correct architecture > brute capacity.** 50 k-param per-stock encoder is **more accurate + faster + smaller** than 800 M-param flat MLP.
8. **Data convergence must outpace capacity ramp.** Phase 2 widened net 12× ([64,64] → [2048,1024,512]) but kept smoke at 50k steps → misjudged as "not learning"; really just hadn't trained long enough.
9. **Pipeline correctness ≠ model quality.** R1's "first end-to-end run" was as much progress as R3's "explained_var=0.99 value function" because both unlocked new test categories.

### 9.2 因子前缀 / Factor-Prefix Discipline

`FACTOR_COL_PREFIXES` 是 single source of truth。漏一个前缀 = Phase 16 的 `mfp_` 静默缺失复现。三处必须同步：

- `src/aurumq_rl/data_loader.py` tuple
- `tests/test_data_loader.py:test_factor_col_prefixes_lockdown`
- `scripts/export_factor_panel.py:FACTOR_PREFIXES`

### 9.3 数据契约红线 / Data Contract Red Lines

- A 股 `pct_chg` 是**小数**（+10 % = 0.10），不是 +10 或 10.0。
- `vol == 0` 表示停牌，必须从训练 panel 过滤掉。
- A 股代码必须 Tushare 风格 `XXXXXX.SH/SZ/BJ`。
- 涨跌停判定**按板别**，不能写死 ±10 %（ST ±5 %, 科创创业 ±20 %, 北交 ±30 %）。
- T+1 约束**必须强制**：今天买入次日才可卖。这是 `entry_eligible_mask` 的核心。

### 9.4 训练资源红线 / Training Resource Red Lines

- ECS 8C14G 严禁训练。PyTorch 安装即占 ~3 GB RSS。
- `ProcessPoolExecutor / ThreadPoolExecutor max_workers ≤ 3`。
- PostgreSQL `shared_buffers=2GB`。
- bf16 autocast 在 4070 / Ampere 上对 PPO 不稳定（Phase 11），保留 fp32 + TF32（Phase 14）。

### 9.5 OSS handoff 方向约定 / OSS Handoff Directionality

- `<PRIVATE_OBJECT_STORE>/handoffs/handoffs/...` = ledashi (4070 Windows) → paris (Ubuntu ECS) 方向
- `<PRIVATE_OBJECT_STORE>/handoffs/...` = paris → ledashi 方向
- ECS 在 sgp 区只能直传 `<PRIVATE_OBJECT_STORE>`，CRR 自动同步到大陆 `<PRIVATE_OBJECT_STORE>` 让 ledashi 拉取
- **文档内的 `oss://` 路径要写主库（<PRIVATE_OBJECT_STORE>/...），不要写源 bucket（<PRIVATE_OBJECT_STORE>/...）**

### 9.6 全 Bug 索引（按 Phase 排序）/ Full Bug Index (by Phase)

完整列表在 [`docs/TRAINING_HISTORY.md`](docs/TRAINING_HISTORY.md) Section D。摘要：

- **Phase 0**: gymnasium import / ONNX device mismatch / `dynamo=True` 失败 / numpy.float32 JSON
- **Phase 1**: NaN through z-score / OOS obs_dim mismatch / approx_kl=41,820 blow-up / mean_fps=0 / metrics_summary all-null / `runs/` gitignore unanchored / alpha045 STHSF Windows-only failure / OSS admin AK disabled / wepa namespace pollution
- **Phase 2**: dashboard canonical-key filter / recharts width(-1) height(-1) warnings / hydration mismatch from MPA browser extension / missing `/api/runs` route / next.js node:fs/promises in client bundle
- **Phase 3**: eval_backtest n_factors mismatch / R2 OOM (8.83 GiB rollout buffer)
- **Phase 4**: n_envs=20 OOM (14.7 GiB)
- **Phase 5**: OSS IncompleteRead at 99.7 % / connection timeout on parallel transfers / dev server `&`-orphan exit 127

---

## §10 上手与复现 <a id="10-上手与复现"></a>
## §10 Quick Start and Reproduction <a id="10-quick-start-and-reproduction"></a>

### 10.1 30 秒 smoke / 30-Second Smoke

```bash
git clone https://github.com/yupoet/aurumq-rl.git
cd aurumq-rl
python3 -m venv .venv && source .venv/bin/activate

# 安装核心依赖（推理 only，~50MB）
pip install -e .

# 跑 smoke test（合成数据，CPU 即可）
python scripts/train.py --smoke-test --out-dir /tmp/aurumq_rl_smoke
cat /tmp/aurumq_rl_smoke/smoke_summary.json
```

### 10.2 真实训练（需 GPU）/ Real Training (GPU Required)

```bash
# 1) 安装 GPU 训练依赖（PyTorch + SB3 + gymnasium + onnx + wandb）
pip install -e ".[train]"

# 2) 准备数据：用合成 demo 或自己导出
python scripts/generate_synthetic.py --out data/synthetic_demo.parquet  # 10MB demo
# 或：从 PG 抽取
python scripts/export_factor_panel.py \
    --pg-url postgresql://user:pass@host/db \
    --start 2020-01-01 --end 2026-04-30 \
    --out data/factor_panel.parquet

# 3) 启动训练（RTX 4070 12GB，n_envs=16，~6h overnight 1M steps）
python scripts/train.py \
    --algorithm PPO \
    --total-timesteps 1000000 \
    --data-path data/factor_panel.parquet \
    --universe-filter main_board_non_st \
    --include-hot-money \
    --n-envs 16 \
    --target-kl 0.05 \
    --reward-mode main_wave_hold \
    --out-dir models/ppo_v1

# 4) 推理（CPU only）
python scripts/infer.py \
    --model models/ppo_v1/policy.onnx \
    --data data/factor_panel.parquet \
    --date 2026-04-30 \
    --top-k 30
```

### 10.3 复现 Phase 22C 主升浪奖励 / Reproduce Phase 22C Main-Wave Reward

```bash
# Phase 22C — train_topk=3, eval@5, 200k steps, seed=42, ~8h on RTX 4070
python scripts/train_v2.py \
    --algorithm PPO \
    --total-timesteps 200000 \
    --data-path data/factor_panel_combined_short_2023_2026.parquet \
    --universe-filter main_board_non_st \
    --n-envs 16 \
    --seed 42 \
    --reward-mode main_wave_hold \
    --main-wave-config '{"exit_signal":"ma5_ma10_death_cross","max_hold_days":5}' \
    --top-k 3 \
    --out-dir runs/phase22c

# 评估（H1 OOS = 2025-07..2025-12）
python scripts/_eval_main_wave_v1.py \
    --ckpt runs/phase22c/policy_best.zip \
    --eval-data data/factor_panel_combined_short_2025-07_2026-04.parquet \
    --eval-top-k 5 \
    --out runs/phase22c/eval.json
```

期望结果（中文/Expected results）：

- hit_rate@5: 6.16 %
- win_rate: 44.0 %
- avg_hold: +0.62 %
- avg_max_drawdown: 3.84 %

### 10.4 复现 Phase 26F-v3 事件衰减 / Reproduce Phase 26F-v3 Event-Decay

```bash
# Phase 26F-v3 — 3 seeds, 361 cols, 300k steps each
for seed in 42 43 44; do
  python scripts/train_v2.py \
    --algorithm PPO \
    --total-timesteps 300000 \
    --data-path data/factor_panel_v3.parquet \
    --include-columns-file configs/phase26f_v3_361cols.txt \
    --universe-filter main_board_non_st \
    --n-envs 16 \
    --seed $seed \
    --reward-mode main_wave_hold \
    --out-dir runs/phase26f-v3-seed$seed
done

# 取 3-seed 中位数 (T-1 lift median ~2.27×)
python scripts/_eval_main_wave_v1.py \
    --ckpt-glob "runs/phase26f-v3-seed*/policy_best.zip" \
    --eval-data data/factor_panel_v3_oos.parquet \
    --eval-top-k 5 \
    --aggregate median \
    --out runs/phase26f-v3-summary.json
```

### 10.5 复现 SL Path 1 long *(SL 赛道，在 AurumQ 主仓侧)*

```bash
# 在 AurumQ 主仓
cd /path/to/AurumQ

# 跑 SL Path 1 long（raw 345-col + 7y train）
python scripts/generate_wave_scores_paths.py \
    --path path1_long \
    --date 2026-05-08 \
    --top-k 50

# 跑 Hybrid（path1_long + path4 50/50）
python scripts/generate_wave_scores_paths.py \
    --path hybrid \
    --date 2026-05-08 \
    --top-k 50

# 跑 path5_long (NEW BEST)
python scripts/generate_wave_scores_paths.py \
    --path path5_long \
    --date 2026-05-08 \
    --top-k 50
```

### 10.6 Windows 注意事项 / Windows Notes

Windows 上 `pip install -e ".[train]"` 默认会从 PyPI 装 **CPU-only torch**。要拿到 CUDA 版需要先单独装：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[train]"
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

ONNX 导出阶段会输出含 emoji 的提示文本；简体中文 Windows 控制台默认 GBK 编码无法编码：

```bash
# bash / git-bash
export PYTHONIOENCODING=utf-8

# PowerShell
$env:PYTHONIOENCODING = "utf-8"
```

### 10.7 本地 Web Dashboard / Local Web Dashboard

```bash
# 启动（自动 npm install + npm run dev）
bash scripts/web_dashboard.sh        # macOS / Linux / Git Bash
.\scripts\web_dashboard.ps1          # PowerShell
```

打开 `http://localhost:3000` 查看训练历史。`/runs/<id>` 是单次详情（含 backtest 摘要 + 训练曲线 + GPU 利用率面板），`/compare?ids=a,b,c` 多次叠加对比。前端 Next.js 16 server route 直接读取 `runs/` 目录，无需后端。

---

## §11 路线图、引用、许可 <a id="11-路线图引用许可"></a>
## §11 Roadmap, Citation, License <a id="11-roadmap-citation-license"></a>

### 11.1 路线图 / Roadmap

**2026-07-17 更新（4070 重新启用，详细依据见 §12.7）**：

- [ ] **MASTER-lite bring-up**：`scripts/p3/master_train.py` CPU smoke（`--device cpu --max-stocks 200`）→ 4070 全量训练 → `master_ensemble_eval.py` 出 KEEP/KILL verdict（对 path5_long base）
- [ ] **Kronos 系列适用 kill criteria**：v13 之后不再排独立预测/微调实验，只保留 embedding-as-feature 消融（§12.7a）
- [ ] **cost-adjusted 评估落地**：年化双边换手 + 分档红利税进 cell 报告模板（§12.7d）

**2026-07-17 update (4070 re-enabled; rationale in §12.7).** MASTER-lite bring-up (CPU smoke → full 4070 train → KEEP/KILL verdict vs the path5_long base); Kronos track restricted to embedding-as-feature ablations under the pre-registered kill criteria; cost-adjusted metrics (turnover + tiered dividend tax) added to the cell-report template.

**短期（已 in-progress）**：

- [ ] path5_long A/B 几周后切到 `is_recommended=True`
- [ ] 6 个新增 path 的 wave drift check 覆盖（19:00 cron）
- [ ] 26G 在 16 GB 级 GPU 上重测（256→128 encoder 假设的硬件验证）
- [ ] gtja_017 等 8 个 quality_flag=1 的 stub 因子在 v2 sanitizer 下重审

**中期（设计阶段）**：

- [ ] Conformal prediction interval（已在 SL 侧 Strategy D 之外实测 +5 bps，需移植到 RL）
- [ ] 跨市场（US/HK/Crypto）调用同一套训练 stack 的可行性研究
- [ ] AQML → PPO reward function 的自动转译（"用户写策略意图，自动训出 policy"）

**远期（探索）**：

- [ ] Transformer attention head 在 36 GB+ 级 GPU 上重审（Phase 21 V2 在 12 GB 失败的复盘）
- [ ] 多 agent 协作（市值因子专家 + 资金流专家 + 主升浪专家的 mixture-of-experts）

**Short term (in progress).** A/B-promote `path5_long` to `is_recommended=True`; cover wave drift check for the 6 new paths; re-test Phase 26G on a 16 GB-class GPU; re-audit 8 quality_flag=1 stub factors under v2 sanitizer.

**Mid term (design).** Conformal prediction interval (proven +5 bps on SL side beyond Strategy D, port to RL); cross-market (US/HK/Crypto) feasibility study with the same stack; AQML → PPO reward auto-translation.

**Long term (exploration).** Transformer attention head on 36 GB+ GPU (re-audit Phase 21 V2 failure under more compute); multi-agent mixture-of-experts (cap factor expert + capital flow expert + main-wave expert).

### 11.2 引用 / Citation

如果本项目对你有帮助，欢迎 Star ⭐ 和引用：

```bibtex
@software{aurumq_rl_2026,
  title  = {AurumQ-RL: Reinforcement Learning Stock Selection for China A-Shares},
  author = {Paris Yu and AurumQ-RL Contributors},
  year   = {2026},
  url    = {https://github.com/yupoet/aurumq-rl},
}
```

### 11.3 数据来源声明 / Data Source Disclosure

**中文.** 本项目使用的金融数据来自**公开行情数据导出**，包括日线 OHLCV、资金流分档、龙虎榜、北向持股、融资融券、筹码分布、基本面、申万行业等公开市场信息。这些数据在新浪财经、东方财富、同花顺、券商行情软件等公开渠道均可获取。**项目不内置任何特定数据 API 的密钥或商业授权数据**。

`data/synthetic_demo.parquet` 完全是合成数据，不对应任何真实股票。

如需真实数据训练，用户需自行：

1. 从合规渠道获取行情数据
2. 用 `scripts/export_factor_panel.py` 导入到 PostgreSQL
3. 自行承担数据使用合规责任

**English.** Financial data used in this project comes from **public market data exports** — daily OHLCV, capital flow buckets, Dragon-Tiger List, Northbound holdings, margin trading, chip distribution, fundamentals, SW industry. These data are available through Sina Finance, Eastmoney, THS, broker platforms, etc. **The project does NOT bundle any specific data-vendor API keys or commercially licensed data.**

`data/synthetic_demo.parquet` is entirely synthetic and does not correspond to any real stock.

For real-data training, users must: (1) acquire market data through compliant channels, (2) import via `scripts/export_factor_panel.py` into their own PostgreSQL, (3) assume data-compliance responsibility themselves.

### 11.4 许可与免责 / License and Disclaimer

**License.** MIT. See [LICENSE](LICENSE). 商用、修改、再分发皆可 / commercial use, modification, redistribution all permitted.

**Disclaimer (中文).** 本项目作者不对收益和风险承担任何责任。**请记住：量化策略历史回测优秀 ≠ 实盘赚钱**。

**Disclaimer (English).** This project is for **educational and research purposes**. Backtested performance does **not** guarantee live trading profits. The authors take no responsibility for any financial losses incurred from using this code.

---

## §12 研究范式分类与未来方向 / Research Paradigms and Future Directions <a id="12-research-paradigms"></a>

### 12.1 两大 paradigm 学术分类 / Two Research Paradigms

A 股选股 ML 研究归两大 paradigm:

#### Paradigm 1 — Predictive Cross-Sectional Modeling (前瞻横截面预测)

**学名 / Academic name**:
- Cross-Sectional Return Forecasting
- Supervised Alpha Modeling
- Forward Return Prediction

**核心范式 / Core**: `features(t) → y(t)` 其中 `y(t) = f(forward_returns over [t+1, t+K])`。每天 rank stocks by predicted y,选 top-K。

**业界占比 / Industry**: 主流 (Renaissance, Two Sigma, 多数对冲基金)。

**子方向 / Sub-directions**:

| algorithm family | 学名 | 代表 algorithm |
|---|---|---|
| Regression | Continuous target regression | LGB regression / XGBoost / CatBoost / MLP |
| Classification | Binary / Quantile classification | LGB binary / Logistic / Probit |
| Learning-to-Rank | Pairwise + listwise loss | LambdaMART / RankNet / ListNet |
| Distributional | Quantile regression / Mixture density | LGB quantile / TabNet |
| Multi-horizon Multi-task | Joint learning K=1/5/10/20 | Multi-head NN / multi-output LGB |
| Sequence Models | LSTM / Transformer for tabular time-series | Kronos / FinBERT-style |
| Stacking / Meta-learning | L1 / L2 ensemble | LGB / NN on base predictions |

**所有 paradigm 1 共同特点**: 用 forward window 计算 label,无论 K=20 还是 K=1,无论 regression 还是 binary,本质都是预测"未来 K 天 outcome"。

#### Paradigm 2 — Event-Anchored Pattern Recognition (事件锚定模式识别)

**学名 / Academic name**:
- Event Study + Case-Control Sampling (经济学/金融学派)
- Pre-Event Pattern Detection
- Anomaly / Novelty / Rare-Event Detection (ML 派)
- Sequence-to-Event Models (深度学习派)

**核心范式 / Core**: 历史扫一遍找 N 个 events,取每 event 前 T-1/T-3/T-5 天作 positive,随机非 pre-event day 作 negative。`features(t) → P(t is pre-event)`。

**业界占比 / Industry**: 小众 (人工选股 + 部分模式识别 trading + 异常检测 quant)。

**子方向 / Sub-directions**:

| algorithm family | 学名 | 代表 algorithm |
|---|---|---|
| Event-Anchored Classification | Case-control logistic / Imbalanced binary | LGB / XGBoost on anchored samples |
| Pattern Mining | Matrix Profile / Motif discovery | STUMPY / Time Series Subsequence search |
| Imbalanced Classification | Focal loss / SMOTE oversampling | Focal-Loss LGB / XGBoost scale_pos_weight |
| Anomaly Detection | Isolation Forest / Autoencoder | iForest / VAE |
| Survival Analysis | Cox proportional hazards | scikit-survival / Cox-LGB |
| Sequence-to-Event | LSTM / Transformer / TCN | DeepAR / Kronos fine-tune |
| Self-Supervised Pre-training | Contrastive learning on time series | SimCLR-style for finance |

**所有 paradigm 2 共同特点**: 用 backward 历史扫描定义 event,用 pre-event window 作正样本,严重 class imbalance,符合 "找主升浪前夕入场" 思路。

### 12.2 当前研究进度 / Research Progress (updated 2026-05-18)

**Comprehensive synthesis** of 1,473 cells across 8 matrices (5/15-5/18 overnight pipeline): see [`docs/RANKINGS_COMPREHENSIVE_v18.md`](docs/RANKINGS_COMPREHENSIVE_v18.md) for full 13-section report (98 KB, 353 lines, top-20 overall + top-10 per universe/paradigm/panel/method/horizon + sanity checks + production routing + gap audit) + 6 visualization PNGs in [`docs/figures/`](docs/figures/).

#### Master ranking — Top-10 production-deployable cells

(composite = H2 IC × Sharpe NET × max(Q1 IC, 0))

| # | cell_id | paradigm | univ | panel | H2 fwd20 IC | Q1 fwd20 IC | Sharpe_NET K10 fwd20 |
|---|---|---|---|---|---|---|---|
| 1 | `target_y_HARD_TECH_v2_null` | p1-proximity-reg | HARD_TECH | v2_null | **+6.60%** ⭐ | **+10.68%** ⭐⭐ | 2.46 |
| 2 | `target_y_HARD_TECH_ledashi` | p1-proximity-reg | HARD_TECH | ledashi | +6.29 | +10.82 | 2.39 |
| 3 | `target_y_HARD_TECH_r2a` | p1-proximity-reg | HARD_TECH | r2a | +6.17 | +9.97 | 2.53 |
| 4 | `binary_v4_HARD_TECH_v3unified` | p1-binary-dense | HARD_TECH | v3unified | +5.84 | +5.87 | **4.25** ⭐ |
| 5 | `target_y_HARD_TECH_r2b` | p1-proximity-reg | HARD_TECH | r2b | +6.63 | +8.69 | 2.32 |
| 6 | `target_y_HARD_TECH_v2_no_phase_c` | p1-proximity-reg | HARD_TECH | v2_no_phase_c | +6.19 | +9.44 | 2.24 |
| 7 | `target_y_HARD_TECH_v3unified` | p1-proximity-reg | HARD_TECH | v3unified | +5.88 | +9.20 | 2.42 |
| 8 | `target_y_HARD_TECH_tier4_v2_old` | p1-proximity-reg | HARD_TECH | tier4_v2_old | +6.00 | +9.38 | 1.97 |
| 9 | `alpha_T3_HARD_TECH_ledashi` | p2-anchor | HARD_TECH | ledashi | +6.14 | +5.33 | 2.85 |
| 10 | `binary_v3_HARD_TECH_v2_null` | p1-binary-dense | HARD_TECH | v2_null | +3.92 | +5.24 | 4.04 |

#### Per-universe production routing (best cell × horizon)

| universe | short (fwd5) best | mid (fwd10) best | long (fwd20) best |
|---|---|---|---|
| MAIN_BOARD | `v2_MAIN_BOARD_r2b` | `v2_MAIN_BOARD_r2b` | `v4_MAIN_BOARD_ledashi` |
| CSI500 | `binary_v2_CSI500_v2_null` | `binary_v2_CSI500_v2_null` | `catboost_v2_CSI500_ledashi` |
| CSI1000 | `v2_CSI1000_tier4_v2_old` | `binary_v4_CSI1000_tier4_v2_old` | `binary_v2_CSI1000_tier4_v2_old` |
| NPF | `target_y_NPF_v3unified` | `v2_NPF_r2a` | `v2_NPF_r2a` |
| NPF_FULL | `v2_NPF_FULL_v3unified` | `binary_v3_NPF_FULL_v3unified` | `binary_v4_NPF_FULL_v2_no_phase_c` |
| HARD_TECH | `binary_v3_HARD_TECH_ledashi` | `binary_v3_HARD_TECH_ledashi` | `binary_v4_HARD_TECH_v3unified` |

#### 8-matrix grid summary

| matrix | paradigm | cells | universe × panel grid | bootstrap CI | status |
|---|---|---|---|---|---|
| v10  | P1 proximity reg  | 174  | 7×6 + 6 ES eval-only  | partial (in v10h) | shipped |
| v10b | P1 proximity reg (target_y) | 42   | 7×6        | partial (in v10h) | shipped |
| v10c | P1 binary dense (P75 25% pos) | 168  | 7×6×4 labels | partial (in v10h) | shipped |
| v10d | P1 CatBoost diversity | 48 | 2 panels × 6 univ × 4 labels | partial (in v10h) | shipped (5-panel gap) |
| v10e | P1 XGBoost diversity | 48 | 2 panels × 6 univ × 4 labels | partial (in v10h) | shipped (5-panel gap) |
| v10h | bootstrap CI | 207×4 | top cells from v10/v10c/v10d/v10e | itself | shipped |
| **v11** | **P1 binary sparse (paris 0.8%)** | **504** | 7×6 × 4 methods × 3 horizons | **missing** | shipped (no CI) |
| **v12** | **P2 anchor α/β** | **252** (147 valid + 105 skip) | 7×6 × 2 specs × 3 anchors | **missing** | shipped (no CI; β sparse) |
| **v13** | **P3 Kronos sequence anchor** | **22** | 6 univ × 3 anchor × α full + β-MAIN_BOARD + null control | planned | scheduled 5/22 evening fire |

**v13 paradigm 3 matrix** (post paris ACK_v30 + ledashi ACK-of-ACK shipped 5/19 PM):
- Architecture: reuse `aurumq_predictor_small` encoder → 1536-dim hidden state (60d + 120d concat) → +1 log(free_float_mv) → 1537-dim → LGB binary head
- D-1 leakage guard strict (paris ACK §R2): `embedding(D) = encoder(OHLCV[D-seq:D-1])`
- Skip Phase 1 explicit pre-train (ledashi optimization, saves 2-4h; fallback if Phase 3 全军覆没)
- 22 cells: 21 main (18 α + 3 β-MAIN_BOARD) + 1 null-embedding random control (paris Q1 strong-rec)
- Compute: Phase 2 ~3-4h GPU embed extract + Phase 3 ~1.5h LGB train + Phase 4 ~1h eval
- Production gate: if 5 cells meet (Sharpe NET ≥ 3.0 + dual-regime + bootstrap CI lower > 0) → Track 11 paradigm 3 catalog launches

### 12.3 实证结论 / Empirical Findings (paper-level, multi-paradigm)

**Headline findings**:

- The strongest single-cell deployable signal is **`target_y_HARD_TECH_v2_null`** (paradigm `p1-proximity-reg`, panel `v2_null`, universe `HARD_TECH`) with H2_2025 fwd20 IC = **+6.60%** and Sharpe_NET K10 fwd20 = **2.46**, beating the baseline `v3_MAIN_BOARD_ledashi` (+4.14% IC).
- **Paradigm 1 (cross-sectional prediction) dominates Paradigm 2 (anchor) on H2 fwd20 IC by ~0.41pp** — anchor labels useful as meta-feature, not standalone.
- **Bootstrap CI** (v10h K=50 fwd20): **207/207 cells (100%) have CI 2.5% > 0** — production should preferentially deploy K=50 sizing for tail-control.
- **LGB binary dense (v10c) has the highest mean composite score**; **LGB proximity continuous (v10) has the highest peak composite score**. Both retained for production diversification.
- **CSI500/CSI1000 cells (PIT membership) are the safest universes**; HARD_TECH and NPF cells need ≥ 1pp differential vs baseline to claim improvement (IC SE ≈ 0.018).
- **Gap**: v11/v12 lack bootstrap CI; v10d/v10e only cover 2 panels of 7. Production routing on those cells should be flagged as 'preliminary'.

**Universe × Regime alpha** (validated bootstrap CI v10h):
- target_y NPF Q1 IC **+10.22%** (panel-invariant across 7 panels)
- target_y HARD_TECH dual-regime H2 +6.29 / Q1 +10.82 — record-holder cell
- CSI500 H2 +7.97 / Q1 -2.00 — bull-rotation flip
- HARD_TECH `binary_v4_HARD_TECH_v3unified` equi-regime gold (spread 0.03pp ⭐⭐⭐)

**paris production label distribution insight** (paris v26+ confirmed):
- paris production wave_v[1234] static train cutoff 2024-12, NO walk-forward retrain
- paris label = global static τ from train-window search with target_pos_rate=0.008 (0.8% positive)
- ledashi v10c dense P75 cross-section threshold = 25% positive → 30x noise → best_iter=1 early-stop bug fixed in v11

### 12.4 Sanity check status (10 items, see report §9 for detail)

1. ✅ **Baseline reproduction**: `v3_MAIN_BOARD_ledashi` H2 fwd20 IC == +4.143% (bit-exact across matrix v4-v8)
2. ✅ **Cost model**: mean - mean_net == 0.20% (0.002) round-trip
3. ✅ **Gross > Net**: cost increases drag for positive-return cells
4. ✅ **Train/Eval window separation**: Train 2022-2024 ≠ Eval H1_2025..Q2_2026 — no look-ahead leakage
5. ✅ **Deterministic**: random_state=42 fixed in all lgb_params
6. ✅ **PIT correctness**: CSI500/CSI1000 daily PIT membership (per CLAUDE.md universe table)
7. ✅ **Bootstrap CI K=50 fwd20**: 100% cells CI 2.5% > 0 (v10h)
8. ✅ **Bootstrap CI K=10 fwd20**: ≥ 20% cells CI 2.5% > 0 (v10h)
9. ⚠️ **No walk-forward**: paris production also static train cutoff,not blocker
10. ⚠️ **v11/v12 no bootstrap CI**: gap,future work

### 12.5 Visualizations (saved to `docs/figures/`)

- [`fig01_top20_overall_bar.png`](docs/figures/fig01_top20_overall_bar.png) — Top-20 cells barplot
- [`fig02_panel_universe_heatmap.png`](docs/figures/fig02_panel_universe_heatmap.png) — Panel × universe × paradigm IC heatmaps (4 subplots)
- [`fig03_horizon_scaling.png`](docs/figures/fig03_horizon_scaling.png) — IC vs forward horizon (fwd1/3/5/10/20/30) per paradigm
- [`fig04_dyn_exit_ranking.png`](docs/figures/fig04_dyn_exit_ranking.png) — Top-5 cells per dyn-exit trigger (11 triggers)
- [`fig05_paradigm_compare_scatter.png`](docs/figures/fig05_paradigm_compare_scatter.png) — H2 IC vs Q1 IC scatter, colored by paradigm
- [`fig06_bootstrap_ci_distribution.png`](docs/figures/fig06_bootstrap_ci_distribution.png) — Bootstrap CI lower-bound histogram

### 12.7 2026-07-17 外部证据更新与路线修正 / External-Evidence Update & Course Correction

> 基于 2026-07-17 对 19 个开源量化仓库增量 diff + 2025-2026 全球 AI 量化文献/业界实践的系统调研
> (证据与链接见 AurumQ 主仓 `handoffs/2026-07-17-refs-survey/reports/07-ai-quant-trends.md`)。
> Based on a 2026-07-17 systematic survey of 19 upstream quant repos + the 2025-2026 AI-quant
> literature and industry practice. Four course corrections below are now project policy.

#### (a) Kronos / 时序基础模型降级 / Kronos & TSFM demotion

**中文.** 社区大规模实测已证伪「时序基础模型作为独立预测器」：200 只 ETF 系统评测方向准确率 0.49（≈抛硬币）、CRPS/RMSE 全面输给 20 日历史波动率基线、多步预测方差坍缩（[Kronos issue #269](https://github.com/shiyu-coder/Kronos/issues/269)）；A股个股 MAPE 8-44%（[#319](https://github.com/shiyu-coder/Kronos/issues/319)）；作者本人承认「TSFM 微调经常越调越差（含 TimeMoE/TimesFM），稳定微调是开放问题」（[#177](https://github.com/shiyu-coder/Kronos/issues/177)）。最严谨的学术评测（[arXiv:2606.27100](https://arxiv.org/abs/2606.27100)）结论：TSFM 相对随机游走的绝对增益「小且稀疏」。

**修正**：本仓 kronos_matrix 系列（v2→v13）继续存在的唯一合理形态是 **embedding-as-feature 消融**（预训练表征作为 LGBM 特征列，v13 已是此形态），且从现在起受 `master_lib.kill_criteria_verdict` 预注册 kill criteria 约束——对 base 无显著增益即终止，不再排新的 Kronos 独立预测 / 微调实验，不进生产管线。

**English.** Community-scale replication has falsified TSFMs as standalone predictors (direction accuracy 0.49 on 200 ETFs, beaten by a 20-day historical-vol baseline; the Kronos authors concede stable fine-tuning is an open problem). Policy: the kronos_matrix series survives ONLY as embedding-as-feature ablations (v13 already is), now governed by the pre-registered kill criteria in `master_lib.kill_criteria_verdict`. No new standalone-forecast or fine-tune experiments; nothing enters production.

#### (b) GBDT 仍是日频面板基本盘 / GBDT stays the daily-panel backbone

**中文.** Qlib CSI300 官方 benchmark：LightGBM IC 0.0448（Alpha158）vs Transformer 0.0264；2025 年表格基准（TabArena / TabReD 时序切分）同结论。深度模型在日频因子面板上的真实增量只有一处有可信证据：**截面结构建模**——MASTER（AAAI 2024, [arXiv:2312.15235](https://arxiv.org/abs/2312.15235)）经股票间 attention + 市场门控做到 IC 0.064 vs XGBoost 0.051（约 +25%）。RD-Agent 消融同时表明通用时序模型（PatchTST/Mamba）在股票预测上「预测与策略双输」。

**修正**：SL 赛道 LightGBM（path5_long 等）是不可动摇的 base；新增 **MASTER-lite 实验**（`scripts/p3/master_train.py` + `master_ensemble_eval.py`）——单卡 4070 复现市场门控截面 transformer，产出一列低相关分数与 LGBM base 做 rank 混合，同样受 kill criteria 约束（最优混合须在 ≥2/3 窗口上 IC 胜 base 且 top-50 不劣，否则 KILL）。预期是 +10-25% IC 量级的 ensemble 增量，不是替代。

**English.** LightGBM remains the backbone (Qlib benchmark: LGBM IC 0.0448 vs Transformer 0.0264). The one credible deep-model increment is cross-sectional structure modeling (MASTER, AAAI 2024, ~+25% IC over XGBoost). New experiment: MASTER-lite (`scripts/p3/master_train.py`) produces one low-correlation score column for rank-blending with the LGBM base, governed by the same pre-registered kill criteria. Expected as an ensemble increment, never a replacement.

#### (c) RL 定界确认 / RL scoping confirmed

**中文.** 业界与学术双向证据（中欧重金投入凸优化器而非 RL 仓位；Qlib RL 模块定位于 order execution；LLM/RL 信号层论文的「超额」多来自训练窗口价格记忆，[arXiv:2505.07078](https://arxiv.org/abs/2505.07078)；[LiveTradeBench](https://arxiv.org/abs/2511.03628) 显示静态 benchmark 赢家实盘反而更差）与本仓 26 期实验结论（Strategy D top-K 加权叠加 +7-10 bps mean_y，Finding 3）互相印证：**RL 的正确位置是仓位/组合权重层的小幅增强器，不追求信号层突破**。路线图上原「AQML → PPO reward 自动转译」保留，信号层新 RL 实验冻结。

**English.** Industry and academic evidence now mutually confirm our own Finding 3: RL's proper place is the position/weight layer as a modest enhancer (+7-10 bps from Strategy D), not signal generation. Signal-layer RL experiments are frozen; the AQML-to-reward translation line stays.

#### (d) 成本一等公民 / Cost as a first-class citizen

**中文.** 1-5 日 horizon 的信号密度最高，但 A股 T+1 + 印花税 + **按持股期限分档的红利税**（≤1 月 20% / 1 月-1 年 10% / >1 年免——周频调仓的 top-K 正是 20% 档重灾区）意味着日频 IC 0.05 级别的信号扣双边成本后利润很薄；国盛证据显示短周期量价 alpha 自 2024 年起拥挤衰减。**修正**：后续每个 cell 报告须附年化双边换手与 cost-adjusted 指标（参考 rqalpha 6.2.0 的 `_pay_dividend_tax` per-lot FIFO 实现，`rqalpha_mod_sys_accounts/position_model.py:322`）；「IC 好看但净值不赚」的 cell 一律不进 §12.2 主榜。

**English.** Ultra-short horizons carry the densest signal but also the harshest cost stack (T+1, stamp duty, holding-period-tiered dividend tax — weekly top-K rebalancing sits squarely in the 20% tier). Every future cell report must carry annualized two-sided turnover and cost-adjusted metrics; cells that look good on IC but lose net of cost stay off the §12.2 master ranking.

### 12.6 (deprecated content kept for git history)

### 12.deprecated 当前研究进度 / Research Progress (5/17 snapshot,superseded by 12.2 above)

#### Paradigm 1 (产出 1067+ cells of evidence)

| matrix | scope | cells | status |
|---|---|---|---|
| matrix v3-v8 | Panel ablation 系列 (paris combined_panel evolution) | ~150 | ✅ done, RESULT v3-v8 shipped |
| matrix v9 | direct ret_fwdK regression (short proximity attempt) | 60 | ✅ done, **failed** — IC weak/negative |
| matrix v10 | 7 panel × 6 universe × 4 wave_v* × 7 sizing + 6 ES eval | 174 | ✅ **done** (5/16 12:30, 255 min) |
| matrix v10b | + target_y (paris primary proximity, 5th label) | 42 | ✅ **done** (5/16 14:30, 111 min) |
| matrix v10c | LGB binary classifier on wave_v* (P75 dense threshold) | 168 | ✅ **done** (5/16 21:00, 388 min) |
| matrix v10de | CatBoost + XGBoost expanded (algorithm diversity) | 96 | ✅ **done** (5/17 00:50, 123 min, +inf fix re-fire) |
| matrix v10fg | L1 meta stacker (24) + L2 hybrid blend (6) | 30 | ✅ **done** (5/17 01:02, meta 全 SKIP due to only 2/7 panel preds saved; hybrid completed) |
| matrix v10h | Bootstrap CI post-processing on 207 pred parquets | 207 | ✅ **done** (5/17 01:56, 55 min, block-bootstrap 1000 iter on Sharpe NET) |
| matrix v11 (paris sparse binary apples-to-apples) | 7 panel × 6 universe × 4 method (A/B/C/D) × 3 horizon (t1/t3/t5) | 504 | 🟡 **in progress** ~217/504 (43%, 5/17 10:18 OOM crash + resumed, ETA ~22:00) |

**Key non-obvious findings from v10/v10b/v10c/v10de**:
- **Label structure determines IC ceiling**, not panel/algorithm: wave_v3 sparse proximity → +4% IC vs target_y dense calibrated proximity → +2% IC on same MAIN_BOARD ledashi cell.
- **Universe×Regime alpha extreme**: target_y NPF Q1 IC **+10.22%** (panel-invariant across 7 panels), target_y HARD_TECH H2 +6.29% & Q1 +10.82% (dual-regime record), CSI500 H2 +7.97% but Q1 -2.00% (bull-rotation flip).
- **Phase C concept_* features over-engineered**: v2_null vs ledashi (no Phase C) on theme universes Q1 IC differ < 0.7pp (Phase C marginal); on HARD_TECH wave_v3 binary, Phase C NULL **rescues Q1 +5.24pp** vs Phase C present.
- **No single panel/label wins all universes** (paper-level evidence):
  - wave_v3 wins MAIN_BOARD + HARD_TECH + NPF_FULL binary
  - wave_v4 wins CSI1000 + NPF binary
  - wave_v2 wins CSI500 binary
  - LGB binary wins theme universes (NPF/NPF_FULL/HARD_TECH); CatBoost wins PIT universes (CSI500/CSI1000)
- **r2b 232-col minimalist panel** reaches CSI1000 wave_v3 binary equi-regime gold standard (H2 +5.61% / Q1 +4.62% / spread 0.99pp) with **only 3 trees** — feature engineering minimalism wins.
- **3 equi-regime gold cells found** (spread < 0.5pp + Sharpe NET ≥ 3.5):
  - HARD_TECH v3unified wave_v4 binary: +5.84/+5.87/Sharpe +4.09 (spread 0.03pp ⭐⭐⭐)
  - HARD_TECH v2_no_phase_c wave_v2 binary: +4.19/+4.22/Sharpe +4.12 (spread 0.03pp)
  - MAIN_BOARD v2_no_phase_c wave_v4 binary: +3.24/+3.18/Sharpe +3.64 (spread 0.06pp)
- **Sparse 0.8% label trains model 50-225 trees** (paris production-aligned, no best_iter=1 early-stop bug like dense 25% label).

#### Paradigm 2 (anchor-based main rising wave)

| matrix | scope | status |
|---|---|---|
| Phase 1 short proximity labels (paris ship 5/16) | 4 method × 3 horizon × 6 universe, target_pos_rate=0.008 | ✅ shipped, used by v11 |
| Phase 2 anchor labels β + α (paris ship 5/16) | 3 anchor (T-1/T-3/T-5) × 6 universe × {α 5-condition, β PELT-hybrid} | ✅ shipped, used by v12 |
| matrix v12 anchor-based (planned) | 7 panel × 6 universe × 3 anchor × {α, β} = 252 cells | 🟡 next after v11 |
| Imbalanced loss variants | Focal-Loss / SMOTE on anchor labels | future |
| Sequence-to-Event | Kronos fine-tune for pre-event detection | future |

#### paris ↔ ledashi handoff cadence (5/16 single day)
v24 (Phase 1 + Phase 2 labels + LABELS_SPEC + IC pre-estimate) → v25 (P2 reference data 5 files) → v25b (wave_v3 retro 2025+ true OOS) → v26 (catboost+xgb hyperparams + 5 reverse-ask answers) → v27 (wave_v1/v2/v4 retro + IC ROI table + regime labels + best_iter table) → 14 docs / 700+ files / ~95 MB cumulative.

### 12.3 实证结论 / Empirical Findings (2026-05-15..17)

来自 matrix v3-v10de 全 paradigm-1 横评 + bootstrap CI on 207 cells:

**Panel × Regime interaction** (validated bootstrap CI v10h):
- **ledashi 226 pruned panel**: H2 momentum regime best on broad universes (MAIN_BOARD/CSI1000 wave_v3 IC +4.14%)
- **paris tier4_v2_old 378 panel**: NPF binary H2 + Sharpe powerhouse (+5.59 / +4.43 avg vs ledashi +3.01/+3.69)
- **v3unified (paris production candidate 244 cols)**: NPF Q1 IC **+11.07%** record holder + NPF_FULL wave_v3 binary equi-regime gold (+5.47/+4.28 spread 1.19pp)
- **r2b 232-col minimalist**: CSI1000 wave_v3 binary dual-regime gold with **only 3 trees** (+5.61 H2 / +4.62 Q1)
- **Phase C concept features over-engineered**: NULL or drop both 0.3-1pp Q1 stability gain on theme universes (NPF/HARD_TECH)

**Label × Algorithm interaction** (formal evidence v10c+v10de):
- **wave_v3 sparse proximity** → highest IC across most cells (LGB binary +4.34 > regression +4.14)
- **wave_v4 (direct proximity)** → best Q1 regime stability on theme universes
- **wave_v1 binary** → systematically weak (best_iter often = 1/2, no learnable signal)
- **wave_v2 binary** → fast learner (2-12 trees) on PIT universes
- **target_y (paris primary 83% pos rate)** → 1/2 the IC of wave_v3 (label sparsity dominates)
- **paris sparse 0.8% binary** → 30x lower positive rate, model trains 50-225 trees, IC ~0.5-3% (production-relevant decision)
- **CatBoost dominates PIT mid-cap universes (CSI500/CSI1000 Q1 +3.93%/+1.56% vs LGB +1.30%/+1.81%)**
- **LGB binary dominates theme universes (NPF/NPF_FULL/HARD_TECH Q1 +0.36% / +2.03% / -1.51% — only positive Q1)**
- **XGBoost generic params consistently 3rd weakest** (need paris-tuned hyperparams for v10de_v2)

**Universe × Label × Panel interaction (3D)**:
- 6 universes × 4 wave_v* × 7 panels = no global best combination
- Production stack MUST use regime detector + universe×label×panel routing
- **Best dual-regime equi-stable cells** (spread < 0.5pp, Sharpe NET ≥ 3.5):
  - HARD_TECH × v3unified × wave_v4 binary: H2 +5.84% / Q1 +5.87% / Sharpe +4.09 (spread **0.03pp** ⭐⭐⭐)
  - HARD_TECH × v2_no_phase_c × wave_v2 binary: H2 +4.19 / Q1 +4.22 / Sharpe +4.12 (spread 0.03pp)
  - MAIN_BOARD × v2_no_phase_c × wave_v4 binary: H2 +3.24 / Q1 +3.18 / Sharpe +3.64 (spread 0.06pp)

**Sizing**: top_k=5/10/15/20/30/50 + adaptive scheme. Production sweet spot ~10-30 names per universe. Sharpe NET typically 1.5-4.5 after 0.20% round-trip cost (extreme +4.79 on r2a HARD_TECH wave_v3 binary).

**Dyn-exit production champions**:
- MAIN_BOARD wave_v3 + Q_OR_FIE ensemble = best
- CSI1000 wave_v3 + J5_take_profit_5 = highest Sharpe NET seen
- F_trend_break (close < MA5) = wife's strategy, robust across universes

**paris production label distribution insight** (paris v26+ confirmed):
- paris production wave_v[1234] static train cutoff 2024-12, NO walk-forward retrain
- paris label = global static τ from train-window search with target_pos_rate=0.008 (0.8% positive)
- ledashi v10c dense P75 cross-section threshold = 25% positive → 30x noise → best_iter=1 early-stop bug
- v25b retro-score 2025-01+ subset = true paris production-style OOS baseline

### 12.4 未来研究方向 / Future Research Directions

**Tier 1 (1-2 weeks)**:
- v11 short-K proximity labels (paradigm 1 short-horizon completion)
- v11+ anchor-based main-rising-wave label (paradigm 2 entry)
- Walk-forward rolling retrain (paradigm 1 robustness verification)
- Sector-neutral alpha decomposition (paradigm 1 cleanliness)

**Tier 2 (1-3 months)**:
- Meta-learner across panels (paradigm 1 model diversity)
- Risk-parity portfolio construction (replace top-K equal-weight)
- Regime classifier conditional model (HMM / vol-regime)
- Hyperparam Optuna search (Bayesian)

**Tier 3 (3+ months)**:
- Sequence-to-Event models (paradigm 2 deep learning)
- Self-supervised pre-training on time-series subsequences
- Intraday signals integration (tick-level + cross-asset basis)
- Cross-asset signals (futures basis, ETF flow, options skew)

### 12.5 论文化 / Toward Publication

The matrix v3-v10 series produces academic-grade evidence on:
- Panel design × regime interaction (paper draft target: "Cross-sectional alpha decomposition by regime in A-share markets")
- Hyperparam-label fit (paper draft target: "Regression vs binary classifier choice in proximity-weighted forecasting")
- Dyn-exit ensemble alpha (paper draft target: "Adaptive exit triggers in factor-based portfolios")
- Comparison Paradigm 1 vs Paradigm 2 (future paper after anchor-based label complete)

PRs welcomed for: anchor-based label math formula refinement, paradigm-2 algorithm benchmarks, sector-neutral decomposition implementations.

---

<p align="center"><em>If a phase here taught us something the hard way, we wrote it down so the next person doesn't have to relearn it. PRs that record new lessons are warmly welcomed.</em></p>

<p align="center"><em>「凡是踩过的坑，都该有文字留下。欢迎 PR 补充新的教训。」</em></p>
