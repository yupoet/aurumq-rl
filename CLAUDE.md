# CLAUDE.md — AurumQ-RL AI 协作约定

> 给 Claude Code / Cursor / Copilot 等 AI 助手看的项目协作规范。

## 项目定位

`AurumQ-RL` 是 A 股量化强化学习选股的**开源参考实现**。

**它不是什么**：
- ❌ 不是实盘交易系统（无券商对接、无下单 API）
- ❌ 不是数据采集工具（不内置任何数据源 API key）
- ❌ 不是因子计算库（**因子计算不在本项目**，由用户自己 pipeline 提供）
- ❌ 不是高频交易（日频选股，T+1 约束下持仓）

**它是什么**：
- ✅ RL 选股算法 + Gymnasium 环境的参考实现
- ✅ A 股市场约束（T+1 / 涨跌停 / ST / 板别）的工程化封装
- ✅ 多源因子的**消费者**（按列名前缀识别，输入有什么就用什么）
- ✅ 离线训练 → ONNX 导出 → CPU 推理的端到端流水线

## 输入数据契约（核心约定）

本项目对外契约：**给我一份 Parquet，我就能训练**。

Parquet 必须含字段：
- `ts_code` (str): Tushare 风格代码 `XXXXXX.SH/SZ/BJ`
- `trade_date` (date): 交易日
- `close` (float): 收盘价
- `pct_chg` (float): 涨跌幅（**小数形式**，+10% = 0.10）
- `vol` (float): 成交量（== 0 视为停牌）
- 因子列（**至少一组前缀**）：`alpha_*` / `mf_*` / `hm_*` / `hk_*` / `inst_*` / `mg_*` / `cyq_*` / `senti_*` / `sh_*` / `fund_*` / `ind_*` / `mkt_*`

可选字段（提供则使用，不提供则降级）：
- `is_st` (bool): ST 标记
- `days_since_ipo` (int): 上市以来交易日数
- `industry_code` (int): 申万一级行业编码

**数据怎么来不是本项目的关心范围**。用户可以：
1. 用 `scripts/export_factor_panel.py` 从自己的 PG 数据仓库抽取（含 SQL 模板）
2. 用 `scripts/generate_synthetic.py` 生成合成数据 demo
3. 自己用任何工具（pandas / polars / DuckDB / Spark）造一份满足契约的 Parquet

## 技术红线（不可违反）

### 1. 训练资源限制

- **本地 ECS（8C14G）严禁运行训练**。PyTorch 安装即占 ~3GB RSS，训练时 OOM 必杀。
- 训练只能在 **GPU 实例**（本地 RTX 4070+ 或云端 RTX 4090 / A10 / V100）。
- 推理在 CPU（onnxruntime），~150ms / 5000 股。

### 2. 数据合规红线

- **绝不在 README / 注释 / 错误信息中提及任何特定商业数据源（如 Tushare）的 API endpoint 或 token 格式**。
- 数据来源描述统一用「公开行情数据导出」「市场公开数据」等中性表述。
- `data/synthetic_demo.parquet` 永远是合成数据，不对应真实股票代码。

### 3. 代码组织红线

- 包名 `aurumq_rl`（带下划线，Python import 用）— 仓库名 `aurumq-rl`（带连字符，URL/PyPI 用）。
- **本项目不包含因子计算逻辑**。所有因子由输入数据 pipeline 提供，本项目仅按列名前缀识别和消费。
- `src/aurumq_rl/env.py` 和 `inference.py` 必须支持 gymnasium 缺失时优雅降级（占位类抛 ImportError，但 import 模块不报错）。

### 4. Universe 过滤红线

默认 universe 过滤（`UniverseFilter.MAIN_BOARD_NON_ST`）必须排除：
- ❌ 北交所（`.BJ` 后缀，或代码 8/4 开头）
- ❌ 科创板（688 开头）
- ❌ 创业板（300/301 开头）
- ❌ ST / *ST / 退市整理

保留：
- ✅ 沪主板：60[01356] 开头
- ✅ 深主板：00[0123] 开头

## 因子识别策略

`data_loader.py` 通过列名前缀**自动识别**因子，没有开关也没有强制：

- 输入 Parquet 中前缀为 `alpha_` / `mf_` / `hm_` 等的所有数值列均被视为因子
- `StockPickingConfig.n_factors` 决定取前 N 个（按字母序），多余丢弃
- 用户可在 export 阶段决定包含哪些前缀（例如不算游资就不导出 `hm_*` 列）

**这意味着 RL 项目不需要知道任何因子的语义、来源或时效约束**。如果训练数据缺某些列，RL 项目不报错，模型自己学到「这些位置是常量 0」。

## 列命名前缀约定

`data_loader.py` 通过列名前缀识别因子组，**不可修改**：

| 前缀 | 来源 |
|---|---|
| `alpha_*` | alpha101 计算 |
| `mf_*` | main force（主力资金） |
| `hm_*` | hot money（游资） |
| `hk_*` | northbound（北向） |
| `inst_*` | institution（机构龙虎榜） |
| `mg_*` | margin（融资融券） |
| `cyq_*` | chips（筹码分布） |
| `senti_*` | sentiment（情绪） |
| `sh_*` | shareholders（股东） |
| `fund_*` | fundamentals（基本面） |
| `ind_*` | industry（行业） |
| `mkt_*` | market（大盘） |

## 测试约定

- **80% 覆盖率**为最低要求。
- 每个因子模块都要有对应的 `tests/test_factors_<name>.py`。
- E2E 测试用 `tests/test_smoke.py`，跑 `train.py --smoke-test`，验证 IO 路径和 ONNX 导出。

## 风险与免责声明

每个用户面向的入口（`scripts/train.py` / `scripts/infer.py` / README）**必须包含明确免责声明**：

> 本项目仅供教育和研究目的。回测结果不代表未来收益。作者不对任何因使用本代码导致的财务损失负责。

## Git 工作流

- 主分支 `main`，feature 用 `feat/xxx` 分支。
- Commit message 用 conventional commits：`feat:`, `fix:`, `docs:`, `test:`, `refactor:`。
- 不强制 commit 签名，不要求 PR 模板。

## 性能基准

修改任何 hot path 代码（`env.step`, `data_loader.load_panel`, `inference.predict`）后必须跑：

```bash
pytest tests/test_perf.py --benchmark-only
```

基线（在 RTX 4070 + i7-13700K 上）：
- `env.step`: < 2ms
- `load_panel(all_a, 5y)`: < 30s
- `inference.predict(5000 stocks)`: < 200ms

退化超过 20% 视为回归，必须修复或写明原因。

## 不要做的事

- ❌ 不要把真实 A 股数据 commit 到仓库（即使脱敏过）
- ❌ 不要在 demo 数据中使用真实股票代码（如 600519.SH）—— 用合成代码（如 SYN001 ~ SYN200）
- ❌ 不要硬编码任何 API key、token、密码、内部 URL
- ❌ 不要 import `aurumq.*`（原项目）—— 这是独立项目，所有依赖必须在 `aurumq_rl` 包内
- ❌ 不要写「这段代码来自 AurumQ 内部项目」之类的注释 —— 项目就是它自己

## 应该做的事

- ✅ 每个公开函数都要有 docstring + type hints
- ✅ 错误信息要给出可操作的修复建议
- ✅ 因子计算返回 NaN 时要有明确策略（z-score 时跳过 NaN，训练时填 0）
- ✅ Universe 过滤要在数据加载阶段完成，不要在训练循环里过滤
- ✅ 所有时间戳用 UTC 或显式标注时区，**A 股交易时间用 `Asia/Shanghai`**
