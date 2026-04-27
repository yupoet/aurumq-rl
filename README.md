# AurumQ-RL · A股量化强化学习选股开源项目

> An open-source reinforcement learning stock selection framework for the China A-share market, with multi-source factor inputs (alpha101 + main force flow + hot money seats + northbound capital + institutional + fundamentals).

<p align="left">
  <a href="https://github.com/yupoet/aurumq-rl/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg" alt="Python 3.10+"></a>
  <a href="https://github.com/yupoet/aurumq-rl"><img src="https://img.shields.io/badge/status-beta-orange.svg" alt="Status: Beta"></a>
  <a href="https://github.com/yupoet/aurumq-rl"><img src="https://img.shields.io/badge/version-0.1.0-green.svg" alt="Version 0.1.0"></a>
  <a href="https://github.com/yupoet/aurumq-rl"><img src="https://img.shields.io/badge/tests-115%20passed-brightgreen.svg" alt="Tests: 115 passed"></a>
  <a href="https://stable-baselines3.readthedocs.io/"><img src="https://img.shields.io/badge/RL-Stable--Baselines3-blueviolet.svg" alt="Stable-Baselines3"></a>
  <a href="https://onnxruntime.ai/"><img src="https://img.shields.io/badge/inference-ONNX%20Runtime-yellow.svg" alt="ONNX Runtime"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
  <a href="https://github.com/yupoet/aurumq-rl/stargazers"><img src="https://img.shields.io/github/stars/yupoet/aurumq-rl?style=social" alt="GitHub stars"></a>
</p>

<p align="left">
  <strong>📊 China A-share · 🤖 PPO/A2C/SAC · 🚀 GPU Train + CPU Infer · 📈 alpha101 + Main-Force + Hot-Money + Northbound</strong>
</p>

[English](#english) · [中文](#中文)

---

## 中文

### 这是什么

`AurumQ-RL` 是一个为 **A 股市场**定制的强化学习选股框架。和把 alpha101 当成排名分数加权的传统做法不同，本项目用神经网络（PPO / A2C / SAC）直接学习「**看到哪种因子组合 → 选这些股票能在未来 N 天扣除成本后赚钱**」的策略。

**核心特色**：

1. **多源因子开箱即用**：data_loader 自动识别 12 种因子前缀（`alpha_*` / `mf_*` / `hm_*` / `hk_*` / `inst_*` / `mg_*` / `cyq_*` / `senti_*` / `sh_*` / `fund_*` / `ind_*` / `mkt_*`），输入 Parquet 含哪些就用哪些
2. **A 股专属约束完整支持**：T+1、板别动态涨跌停（沪深主板 ±10% / 科创创业 ±20% / 北交 ±30% / ST ±5%）、ST 剔除、停牌剔除、新股 60 日保护、单行业 30% 暴露上限
3. **股票池可配置过滤**：默认沪深主板非 ST（约 3500 只），剔除科创板 / 创业板 / 北交所 / ST
4. **离线训练 + CPU 推理分离**：训练在 GPU（推荐本地 RTX 4070+）做，推理走 ONNX + onnxruntime CPU only，~150ms/5000 股
5. **板别识别 + 业绩归因 + 风险分析**：完整生产可用工具链

**项目边界（重要）**：

⚠️ 本项目**只做 RL**，不做因子计算。输入 Parquet 必须已含计算好的因子列。

* 简单跑通：用 `scripts/generate_synthetic.py` 生成合成数据
* 真实训练：自己计算因子（参考 [docs/FACTORS.md](docs/FACTORS.md) 列名约定），或用 `scripts/export_factor_panel.py` 从已有 PG 数据仓库抽取

### 快速上手（30 秒）

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

### 真实训练（需 GPU）

```bash
# 1) 安装 GPU 训练依赖（PyTorch + SB3 + gymnasium + onnx + wandb）
pip install -e ".[train]"

# 2) 准备数据：用合成 demo 数据 OR 自己导出
python scripts/generate_synthetic.py --out data/synthetic_demo.parquet  # 10MB demo
# 或
python scripts/export_factor_panel.py \
    --pg-url postgresql://user:pass@host/db \
    --start 2023-01-01 --end 2025-12-31 \
    --out data/factor_panel.parquet

# 3) 启动训练（RTX 4070 12GB，n_envs=6，~5-6 小时跑 1M steps 全 A 股）
python scripts/train.py \
    --algorithm PPO \
    --total-timesteps 1000000 \
    --data-path data/factor_panel.parquet \
    --universe-filter main_board_non_st \
    --include-hot-money \
    --n-envs 6 \
    --out-dir models/ppo_v1

# 4) 推理（CPU only）
python scripts/infer.py \
    --model models/ppo_v1/policy.onnx \
    --data data/factor_panel.parquet \
    --date 2025-12-30 \
    --top-k 30
```

### 因子前缀约定

`data_loader` 通过列名前缀识别因子组。**输入 Parquet 中存在的前缀就被自动加载，不存在的自动跳过**。

| 前缀 | 含义 | 推荐维度 | 输入数据要求 |
|---|---|---|---|
| `alpha_*` | alpha101 量价因子 | 64 | 日频 OHLCV |
| `mf_*` | 主力资金（大单+超大单累计筹码） | 12 | 4 档资金流分档 |
| `hm_*` | 主流游资席位 | 6 | 游资席位日成交明细 |
| `hk_*` | 北向资金真实持股 | 4 | 北向持股日表 |
| `inst_*` | 龙虎榜机构净买入 | 3 | 龙虎榜机构席位明细 |
| `mg_*` | 融资融券 | 3 | 融资融券日表 |
| `cyq_*` | 筹码分布 | 3 | 筹码分布表 |
| `senti_*` | 涨停板情绪 | 3 | 涨停板池 + 热度榜 |
| `sh_*` | 股东户数 + 大股东增减持 | 2 | 股东数据 |
| `fund_*` | 基本面 PE/PB/ROE/营收增速 | 4 | 基本面表 |
| `ind_*` | 申万行业相对强度 | 2 | 行业指数 |
| `mkt_*` | 大盘 + 拥挤度 | 2 | 指数日表 |

总维度可灵活：纯 alpha101 = 64 维 / 全部因子 ≈ 108 维 / 自定义任意组合。

`StockPickingConfig.n_factors` 决定取前 N 个因子（按字母序），多余的丢弃，不足的报错。

### 训练区间建议

| 配置 | 训练区间 | 测试区间 | 因子数 |
|---|---|---|---|
| 全因子（含游资） | 2023-01-01 ~ 2025-06-30 | 2025-07-01 ~ 2025-12-31 | 108 |
| 长历史（不含游资） | 2017-08-01 ~ 2025-06-30 | 2025-07-01 ~ 2025-12-31 | 102 |
| 经典（仅 alpha101） | 2010-01-01 ~ 2025-06-30 | 2025-07-01 ~ 2025-12-31 | 64 |

### 项目结构

```
aurumq-rl/
├── src/aurumq_rl/
│   ├── env.py                  # StockPickingEnv (Gymnasium)
│   ├── portfolio_weight_env.py # 连续权重组合环境（马科维茨扩展）
│   ├── data_loader.py          # Parquet → numpy 面板（多前缀识别）
│   ├── inference.py            # ONNX CPU 推理
│   ├── onnx_export.py          # SB3 → ONNX 导出
│   ├── price_limits.py         # 板别动态涨跌停
│   ├── reward_functions.py     # Return/Sharpe/Sortino/Mean-Variance
│   ├── metrics.py              # 训练指标 JSONL 读写
│   ├── wandb_integration.py    # 实验跟踪（默认离线）
│   └── sb3_callbacks.py        # SB3 callbacks
├── scripts/
│   ├── train.py                # 训练入口（CLI）
│   ├── infer.py                # 推理入口（CLI）
│   ├── export_factor_panel.py  # PG → Parquet 数据抽取（含 SQL 模板）
│   └── generate_synthetic.py   # 合成 demo 数据生成
├── data/
│   ├── README.md               # 数据格式 + 列名约定
│   └── synthetic_demo.parquet  # 10MB 开箱即跑
├── docs/
│   ├── ARCHITECTURE.md
│   ├── FACTORS.md              # 因子前缀约定 + 列名规范
│   ├── TRAINING.md
│   └── INFERENCE.md
├── tests/                      # pytest 单元测试
└── examples/
    └── quickstart.py           # 端到端示例
```

**因子计算不在本项目**：本项目仅做 RL。因子怎么算请参考你自己的数据 pipeline，或参考 [docs/FACTORS.md](docs/FACTORS.md) 中的字段定义自行实现。

### License

MIT — 商用、修改、再分发皆可，但本项目作者不对收益和风险承担任何责任。**请记住：量化策略历史回测优秀 ≠ 实盘赚钱**。

### 数据来源声明

本项目使用的金融数据来自**公开行情数据导出**，包括日线 OHLCV、资金流分档、龙虎榜、北向持股、融资融券、筹码分布、基本面、申万行业等公开市场信息。这些数据在新浪财经、东方财富、同花顺、券商行情软件等公开渠道均可获取。**项目不内置任何特定数据 API 的密钥或商业授权数据**。

`data/synthetic_demo.parquet` 完全是合成数据，不对应任何真实股票。

如需真实数据训练，用户需自行：

1. 从合规渠道获取行情数据
2. 用 `scripts/export_factor_panel.py` 导入到 PostgreSQL
3. 自行承担数据使用合规责任

### 引用

如果本项目对你有帮助，欢迎 Star ⭐ 和引用：

```
@software{aurumq_rl_2026,
  title  = {AurumQ-RL: Reinforcement Learning Stock Selection for China A-Shares},
  author = {Paris Yu and AurumQ-RL Contributors},
  year   = {2026},
  url    = {https://github.com/yupoet/aurumq-rl},
  author = {Paris Yu}
}
```

---

## English

### What is this

`AurumQ-RL` is a reinforcement learning framework for stock selection on the **China A-share market**. Unlike conventional approaches that rank-weight alpha factors linearly, this project uses neural networks (PPO / A2C / SAC) to directly learn the policy: *"given this combination of factors, select these stocks to maximize next-N-day risk-adjusted returns after costs"*.

### Key features

- **Multi-source factor fusion** (~108 dims): alpha101 + main-force capital flow + speculator seat flow + northbound capital + institutional flow + margin trading + chip distribution + limit-up sentiment + shareholder dynamics + fundamentals + industry strength + market regime
- **Full A-share constraint support**: T+1, board-aware price limits (±10%/±20%/±30%/±5%), ST exclusion, suspension exclusion, new-stock protection, 30% industry exposure cap
- **Configurable universe filter**: by default main-board non-ST (~3500 stocks), excluding STAR / ChiNext / BSE / ST
- **Offline GPU training + CPU ONNX inference**: train on consumer GPU (RTX 4070+), infer in production with onnxruntime, ~150ms for 5000 stocks
- **Production toolchain**: dynamic price limits, Brinson attribution, risk analytics, online fine-tuning

### Quick start

```bash
git clone https://github.com/yupoet/aurumq-rl.git
cd aurumq-rl && pip install -e .
python scripts/train.py --smoke-test --out-dir /tmp/smoke
```

For real training, install `[train]` extra and follow `docs/TRAINING.md`.

### License

MIT. See [LICENSE](LICENSE).

### Disclaimer

This project is for **educational and research purposes**. Backtested performance does **not** guarantee live trading profits. The authors take no responsibility for any financial losses incurred from using this code.
