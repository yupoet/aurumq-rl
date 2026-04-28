# Real-data extraction & transfer guide / 真实数据抽取与传输指南

> Step-by-step recipe for extracting a production factor Parquet on the Ubuntu PG warehouse and shipping it to the Windows / GPU box for training. Pairs with `scripts/export_factor_panel.py` and `docs/example_query_with_index.sql`.
>
> Ubuntu PG 数据仓库导出生产级因子 Parquet，并传到 Windows / GPU 机器训练的完整指引。配合 `scripts/export_factor_panel.py` 和 `docs/example_query_with_index.sql` 使用。

[English](#english) · [中文](#中文)

---

## English

### Topology / 拓扑

```
┌────────────────────┐    extract     ┌─────────────────┐
│ Ubuntu (PG host)   │ ─────────────> │ Aliyun OSS       │
│ scripts/           │                │ ledashi-oss/     │
│   export_factor_   │                │   aurumq-rl/     │
│   panel.py         │                │     panels/      │
└────────────────────┘                └─────────────────┘
                                              │
                                              │ pull
                                              ▼
                                       ┌─────────────────┐
                                       │ Windows + 4070  │
                                       │ scripts/train.py│
                                       └─────────────────┘
```

OSS is the **single source of truth** for real Parquet. Both machines pull from / push to the same `ledashi-oss/aurumq-rl/` prefix. **Never** commit real Parquet to git.

### Prerequisites

- Ubuntu machine has read access to your PostgreSQL warehouse (containing `daily_quotes`, `stock_basic`, `hs300_constituents`, `zz500_constituents`, factor tables).
- Aliyun OSS credentials (provided privately — see "OSS credentials" below).
- Python 3.10+ on both Ubuntu and Windows.
- `ossutil` v2 on both machines: <https://help.aliyun.com/zh/oss/developer-reference/install-ossutil>.

### Step 1 — Sync repo on Ubuntu

```bash
# Either clone fresh or pull
git clone git@github.com:yupoet/aurumq-rl.git ~/dev/aurumq-rl
# or:  cd ~/dev/aurumq-rl && git pull --ff-only origin main

cd ~/dev/aurumq-rl
python -m venv .venv && source .venv/bin/activate
pip install -e ".[factors]"
```

### Step 2 — Configure environment

Copy `.env.example` to `.env` and fill in **PG_URL** and **OSS_*** values:

```bash
cp .env.example .env
$EDITOR .env
```

Key fields:

```bash
PG_URL=postgresql://user:password@pg-host:5432/your_db

# OSS — bucket reuse mode: ledashi-oss + aurumq-rl/ prefix
OSS_ACCESS_KEY_ID=LTAI5...                # (ask the project owner)
OSS_ACCESS_KEY_SECRET=...                 # (ask the project owner)
OSS_ENDPOINT=oss-cn-shenzhen.aliyuncs.com
OSS_BUCKET_NAME=ledashi-oss
OSS_PREFIX=aurumq-rl/
```

Then load it for ossutil:

```bash
set -a; source .env; set +a
ossutil config -e "$OSS_ENDPOINT" -i "$OSS_ACCESS_KEY_ID" -k "$OSS_ACCESS_KEY_SECRET"
ossutil ls "oss://$OSS_BUCKET_NAME/$OSS_PREFIX"   # should succeed (empty list is OK)
```

### Step 3 — Adapt SQL to your warehouse

The reference template assumes table names like `daily_quotes`, `stock_basic`, `hs300_constituents`, `zz500_constituents`, `alpha_factors`, `moneyflow_features`, etc. Almost certainly your real schema is different — fork the template and adjust:

```bash
mkdir -p queries
cp docs/example_query_with_index.sql queries/full_with_index.sql
$EDITOR queries/full_with_index.sql
```

What you **must** keep (for AurumQ-RL data contract):

- `ts_code`, `trade_date`, `close`, `pct_chg`, `vol` (required)
- At least one factor prefix column (`alpha_*`, `mf_*`, `hm_*`, etc.)
- Placeholders `:start_date` / `:end_date`

What you **may** add to make `--universe-filter hs300` and `--universe-filter zz500` work end-to-end:

- `is_hs300` (boolean, per-row from `hs300_constituents` LEFT JOIN)
- `is_zz500` (boolean, per-row from `zz500_constituents` LEFT JOIN)

If your warehouse only has the *current* index snapshot rather than per-date history, document that limitation locally — `data_loader.filter_universe` will still consume the column but the curriculum hook (#9 in the optimization plan) will be approximate.

### Step 4 — Dry-run

```bash
python scripts/export_factor_panel.py \
    --pg-url "$PG_URL" \
    --start-date 2023-01-01 --end-date 2025-06-30 \
    --universe-filter main_board_non_st \
    --sql-file queries/full_with_index.sql \
    --out data/factor_panel_2023_2025.parquet \
    --dry-run
```

Reads the SQL, wraps it with the universe filter, prints the resolved query — does **not** connect to PG. Sanity-check the SQL.

### Step 5 — Small validation extract (2023-2025, ~600 MB-1 GB)

```bash
python scripts/export_factor_panel.py \
    --pg-url "$PG_URL" \
    --start-date 2023-01-01 --end-date 2025-06-30 \
    --universe-filter main_board_non_st \
    --sql-file queries/full_with_index.sql \
    --out data/factor_panel_2023_2025.parquet
```

Streaming through a server-side cursor in 100k-row chunks. 30 min – 2 h depending on your warehouse.

### Step 6 — Verify the Parquet

```bash
python - <<'PY'
import polars as pl
df = pl.read_parquet("data/factor_panel_2023_2025.parquet")
print("rows           :", len(df))
print("dates          :", df["trade_date"].min(), "->", df["trade_date"].max(),
      f"({df['trade_date'].n_unique()} days)")
print("stocks         :", df["ts_code"].n_unique())
print("columns        :", len(df.columns))
factor_cols = [c for c in df.columns if c.startswith(
    ("alpha_", "mf_", "hm_", "hk_", "inst_", "mg_",
     "cyq_", "senti_", "sh_", "fund_", "ind_", "mkt_"))]
print("factor cols    :", len(factor_cols))
if "is_hs300" in df.columns:
    print("hs300 rate     :", round(df["is_hs300"].mean(), 4))
if "is_zz500" in df.columns:
    print("zz500 rate     :", round(df["is_zz500"].mean(), 4))
PY
```

Expected for `main_board_non_st` × 2023-2025: ~3500 stocks × ~600 trade dates, hs300 rate ≈ 0.08, zz500 rate ≈ 0.13.

### Step 7 — Upload to OSS

```bash
ossutil cp \
    data/factor_panel_2023_2025.parquet \
    "oss://$OSS_BUCKET_NAME/$OSS_PREFIX""panels/factor_panel_2023_2025.parquet" \
    --include "*.parquet" \
    --update
```

`--update` skips identical files (re-runs are cheap). For large uploads (>100MB) ossutil automatically does multipart with resume.

### Step 8 — Pull on Windows

On the GPU box:

```bash
cd D:/dev/aurumq-rl
# .env already has the same OSS_* values (copy from Ubuntu's .env once)
ossutil cp \
    "oss://$OSS_BUCKET_NAME/$OSS_PREFIX""panels/factor_panel_2023_2025.parquet" \
    data/factor_panel_2023_2025.parquet
```

### Step 9 — Production extract (2017-2025, ~3-5 GB)

After the small extract validates the SQL and pipeline:

```bash
python scripts/export_factor_panel.py \
    --pg-url "$PG_URL" \
    --start-date 2017-01-01 --end-date 2025-12-31 \
    --universe-filter main_board_non_st \
    --sql-file queries/full_with_index.sql \
    --out data/factor_panel_2017_2025.parquet \
    --chunk-size 200000

ossutil cp \
    data/factor_panel_2017_2025.parquet \
    "oss://$OSS_BUCKET_NAME/$OSS_PREFIX""panels/factor_panel_2017_2025.parquet" \
    --update
```

### OSS layout convention

```
ledashi-oss/aurumq-rl/
├── panels/
│   ├── factor_panel_2023_2025.parquet      # small validation
│   ├── factor_panel_2017_2025.parquet      # production
│   └── factor_panel_<window>.parquet       # ad-hoc experiments
├── models/                                  # optional: trained ONNX archive
└── README.md                                # this file's outline (mirror)
```

### OSS credentials

Reused from a sibling project (`wepa`). Ask the project owner for the values. **Never commit** them to this repo. The shared bucket is `ledashi-oss` in `oss-cn-shenzhen`; this project lives under the `aurumq-rl/` prefix.

If the OSS access keys ever rotate:

1. Update `.env` on every machine that talks to the bucket.
2. `ossutil config -e ... -i ... -k ...` to refresh the local ossutil profile.

---

## 中文

### 拓扑

```
┌────────────────────┐    抽取        ┌─────────────────┐
│ Ubuntu（PG 主机）  │ ─────────────> │ 阿里云 OSS      │
│ scripts/           │                │ ledashi-oss/    │
│   export_factor_   │                │   aurumq-rl/    │
│   panel.py         │                │     panels/     │
└────────────────────┘                └─────────────────┘
                                              │
                                              │ 拉取
                                              ▼
                                       ┌─────────────────┐
                                       │ Windows + 4070  │
                                       │ scripts/train.py│
                                       └─────────────────┘
```

OSS 是真实 Parquet 的**唯一权威源**。两台机器都通过 `ledashi-oss/aurumq-rl/` 前缀读写，**绝不**把真实 Parquet commit 到 git。

### 前置条件

- Ubuntu 能访问你的 PostgreSQL 仓库（含 `daily_quotes` / `stock_basic` / `hs300_constituents` / `zz500_constituents` / 因子表等）
- 阿里云 OSS 凭证（私下传递，见下文「OSS 凭证」）
- Ubuntu 和 Windows 都装 Python 3.10+
- 两机都装 `ossutil` v2：<https://help.aliyun.com/zh/oss/developer-reference/install-ossutil>

### 第 1 步 — Ubuntu 同步 repo

```bash
git clone git@github.com:yupoet/aurumq-rl.git ~/dev/aurumq-rl
# 或：cd ~/dev/aurumq-rl && git pull --ff-only origin main

cd ~/dev/aurumq-rl
python -m venv .venv && source .venv/bin/activate
pip install -e ".[factors]"
```

### 第 2 步 — 配置 .env

```bash
cp .env.example .env
$EDITOR .env
```

关键字段：

```bash
PG_URL=postgresql://user:password@pg-host:5432/your_db

# OSS 复用模式：ledashi-oss 桶 + aurumq-rl/ 前缀
OSS_ACCESS_KEY_ID=LTAI5...                # 找项目 owner 拿
OSS_ACCESS_KEY_SECRET=...                 # 找项目 owner 拿
OSS_ENDPOINT=oss-cn-shenzhen.aliyuncs.com
OSS_BUCKET_NAME=ledashi-oss
OSS_PREFIX=aurumq-rl/
```

加载到 shell + 配 ossutil：

```bash
set -a; source .env; set +a
ossutil config -e "$OSS_ENDPOINT" -i "$OSS_ACCESS_KEY_ID" -k "$OSS_ACCESS_KEY_SECRET"
ossutil ls "oss://$OSS_BUCKET_NAME/$OSS_PREFIX"   # 应能列出（空也行）
```

### 第 3 步 — 适配你 PG 的 schema

参考模板里的表名（`daily_quotes` / `stock_basic` / `hs300_constituents` 等）大概率和你 PG 的实际表名不一样。复制一份再改：

```bash
mkdir -p queries
cp docs/example_query_with_index.sql queries/full_with_index.sql
$EDITOR queries/full_with_index.sql
```

**必保留**（数据契约要求）：

- `ts_code` / `trade_date` / `close` / `pct_chg` / `vol`
- 至少一组因子前缀列（`alpha_*` / `mf_*` / `hm_*` 等）
- 占位符 `:start_date` / `:end_date`

**建议加**（让 `--universe-filter hs300` / `zz500` 真正生效，curriculum 学习的 hook）：

- `is_hs300`：bool，从 `hs300_constituents` LEFT JOIN 拿
- `is_zz500`：bool，从 `zz500_constituents` LEFT JOIN 拿

如果你 PG 只有当前快照而没有历史变更，可以接受降级（`data_loader.filter_universe` 仍能用列过滤，只是 curriculum 会有一定误差）。

### 第 4 步 — Dry-run

```bash
python scripts/export_factor_panel.py \
    --pg-url "$PG_URL" \
    --start-date 2023-01-01 --end-date 2025-06-30 \
    --universe-filter main_board_non_st \
    --sql-file queries/full_with_index.sql \
    --out data/factor_panel_2023_2025.parquet \
    --dry-run
```

打印拼好的 SQL，不连 PG。检查没问题再正式抽。

### 第 5 步 — 小窗口验证抽取（2023-2025，~600MB-1GB）

```bash
python scripts/export_factor_panel.py \
    --pg-url "$PG_URL" \
    --start-date 2023-01-01 --end-date 2025-06-30 \
    --universe-filter main_board_non_st \
    --sql-file queries/full_with_index.sql \
    --out data/factor_panel_2023_2025.parquet
```

服务端游标分块拉，每 10 万行一个 round-trip。耗时 30min - 2h，看仓库性能。

### 第 6 步 — 验证 Parquet

```bash
python - <<'PY'
import polars as pl
df = pl.read_parquet("data/factor_panel_2023_2025.parquet")
print("rows           :", len(df))
print("dates          :", df["trade_date"].min(), "->", df["trade_date"].max(),
      f"({df['trade_date'].n_unique()} days)")
print("stocks         :", df["ts_code"].n_unique())
print("columns        :", len(df.columns))
factor_cols = [c for c in df.columns if c.startswith(
    ("alpha_", "mf_", "hm_", "hk_", "inst_", "mg_",
     "cyq_", "senti_", "sh_", "fund_", "ind_", "mkt_"))]
print("factor cols    :", len(factor_cols))
if "is_hs300" in df.columns:
    print("hs300 rate     :", round(df["is_hs300"].mean(), 4))
if "is_zz500" in df.columns:
    print("zz500 rate     :", round(df["is_zz500"].mean(), 4))
PY
```

预期 `main_board_non_st` × 2023-2025：~3500 只股 × ~600 个交易日，is_hs300 比例 ≈ 0.08，is_zz500 ≈ 0.13。

### 第 7 步 — 上传到 OSS

```bash
ossutil cp \
    data/factor_panel_2023_2025.parquet \
    "oss://$OSS_BUCKET_NAME/$OSS_PREFIX""panels/factor_panel_2023_2025.parquet" \
    --include "*.parquet" \
    --update
```

`--update` 跳过相同文件（重跑零成本）。文件 >100MB ossutil 自动分片上传 + 断点续传。

### 第 8 步 — Windows 端拉取

在 GPU 机器上：

```bash
cd D:/dev/aurumq-rl
# .env 里同样填好 OSS_*（从 Ubuntu 的 .env 拷一次）
ossutil cp \
    "oss://$OSS_BUCKET_NAME/$OSS_PREFIX""panels/factor_panel_2023_2025.parquet" \
    data/factor_panel_2023_2025.parquet
```

### 第 9 步 — 全量抽取（2017-2025，~3-5GB）

小窗口验证流水线 OK 后再跑全量：

```bash
python scripts/export_factor_panel.py \
    --pg-url "$PG_URL" \
    --start-date 2017-01-01 --end-date 2025-12-31 \
    --universe-filter main_board_non_st \
    --sql-file queries/full_with_index.sql \
    --out data/factor_panel_2017_2025.parquet \
    --chunk-size 200000

ossutil cp \
    data/factor_panel_2017_2025.parquet \
    "oss://$OSS_BUCKET_NAME/$OSS_PREFIX""panels/factor_panel_2017_2025.parquet" \
    --update
```

### OSS 目录约定

```
ledashi-oss/aurumq-rl/
├── panels/
│   ├── factor_panel_2023_2025.parquet      # 小窗口验证
│   ├── factor_panel_2017_2025.parquet      # 全量生产
│   └── factor_panel_<window>.parquet       # 临时实验
├── models/                                  # （可选）导出的 ONNX 归档
└── README.md                                # 本文档目录大纲（镜像）
```

### OSS 凭证

复用同账号下 `wepa` 项目的凭证（同一桶 `ledashi-oss`，`aurumq-rl/` 前缀独立）。**绝不** commit 到本 repo。

如果 OSS access key 轮换：

1. 同步更新所有机器的 `.env`
2. 各机器 `ossutil config -e ... -i ... -k ...` 重置 ossutil profile
