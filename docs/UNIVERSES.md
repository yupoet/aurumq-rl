# Universes — AurumQ-RL 统一选股池定义

> **Lock date**: 2026-05-14 by paris v15.2
> **Source of truth**: `<PRIVATE_OBJECT_STORE>/handoffs/2026-05-14-5-universe-lock/`
> **Local cache**: `data/universes/<NAME>_membership.parquet` (gitignored)

## TL;DR

9 个 universe 锁死, 分 3 大类:

| 类 | Universes | 用途 |
|---|---|---|
| **指数 PIT** | CSI300 / CSI500 / CSI1000 | 大中小盘对照, 指数增强, point-in-time 季度调仓 |
| **板块 static** | MAIN_BOARD / GROWTH_BOARDS | 主板 vs 创科板分离建模 (涨跌停规则不同) |
| **概念 static** | NPF / NPF_FULL / NPF_CROSS_BOARD / HARD_TECH | 新质生产力 (新经济) 分层选股池 |

所有 universe 由 paris 端 production pipeline 算并 ship, **ledashi 不自己 derive** (避免 calendar / suspended-stock / 行业分类边角差导致 production vs 实验漂移)。

---

## 9 个 universe 详细定义

### 指数 PIT (point-in-time, 按 trade_date forward-fill)

#### 1. `CSI300` — 沪深 300

| 属性 | 值 |
|---|---|
| 大小 | ~300 stocks/day (paris ship: 405 唯一 stocks 跨 4 年, 体现季度调仓 churn) |
| Type | PIT (季度调仓, 按 trade_date forward-fill) |
| Schema | `(stock_code, trade_date)` 31,500 rows |
| Source | paris `index_weight` table, `index_code='000300.SH'` |
| Time range | 2022-01-04 ~ 2026-05-06 |
| 用途 | 大盘 / 蓝筹核心, **Q1 低波段 regime 的 alpha 主战场** (matrix v2 验证: wave_v2 Q1 fwd20 IC +13.89) |

#### 2. `CSI500` — 中证 500

| 属性 | 值 |
|---|---|
| 大小 | ~500/day (854 unique) |
| Type | PIT |
| Source | `index_weight`, `index_code='000905.SH'` |
| Time range | 2022-01-28 ~ 2026-04-30 |
| 用途 | 中盘 / 行业散布 |

#### 3. `CSI1000` — 中证 1000 (新加)

| 属性 | 值 |
|---|---|
| 大小 | ~1000/day (1733 unique) |
| Type | PIT |
| Source | `index_weight`, `index_code='000852.SH'` (2022 H1 走 Tushare backfill + DB merge = 34001 rows) |
| Time range | 2022-01-28 ~ 2026-04-30 |
| 用途 | 小盘 benchmark. **注意**: paris wave label MAIN_BOARD-only, CSI1000 含很多小盘股可能不在 main board (科创/创业), 训练时 label 覆盖率低, matrix v2 实测 wave 训练样本 < 10K → IC NaN. 用作 alpha 池仍 OK, 但需配套 train 一个 CSI1000-specific label. |

---

### 板块 static

#### 4. `MAIN_BOARD` — A 股主板

| 属性 | 值 |
|---|---|
| 大小 | 3,003 stocks |
| Type | static (trade_date=NULL) |
| Regex | `60[0135]\d{3}\.SH` 或 `00[0123]\d{3}\.SZ` |
| 过滤 | 排除 ST/退市 (`is_st=False AND delist_date IS NULL`) |
| 用途 | A 股**主板基础 universe**, paris 所有 wave_v* / path1_long / path4 production model 默认训练 universe. 跟 NPF v2.1 兼容 (NPF 全在主板内). |

#### 5. `GROWTH_BOARDS` — 创业板 + 科创板 + 北交所

| 属性 | 值 |
|---|---|
| 大小 | 2,253 stocks |
| Type | static |
| Regex | `30\d{4}\.SZ` (创业板) 或 `68[89]\d{3}\.SH` (科创板) 或 `[849]\d{5}\.BJ` (北交所) |
| 过滤 | 排除 ST/退市 |
| 涨跌停 | 创业板 ±20% / 科创板 ±20% / 北交所 ±30% (**跟主板 ±10% 不同, 不要跟 MAIN_BOARD 混训**) |
| 用途 | 创科北单独 universe, **paris 当前 wave label 未覆盖**, 训练前需要单独 build growth-board wave label. paris 5/14 标记为 paper-trade backup, 现阶段 matrix v2 skip. |

---

### NPF 系 (新质生产力, 4 个分层)

> **NPF 定义**: 申万 L1 ∈ {电子, 通信, 计算机, 电力设备, 有色金属, 国防军工, 汽车} ∖ ST ∖ 退市
> 覆盖: 半导体 / 5G / CPO / AI / 新能源 / 光伏 / 储能 / 商业航天 / 新能源车 / 稀土 / 小金属

> **历史版本** (paris 2026-05-14 lockdown 前):
> - **v1 (deprecated)**: 916 stocks, 跨板块 (含创科北). 涨跌停规则混. 已被覆盖.
> - **v2 (deprecated)**: 779 stocks, 跨板块 v2 design. 已重命名为 `NPF_CROSS_BOARD`.
> - **v2.1 (current default)** ⭐: 401 stocks, **主板限定**, 涨跌停 ±10% 一致.
> - **NPF_FULL**: 618 stocks, NPF v2.1 + Layer 2/3 (主板内).
> - **HARD_TECH**: 193 stocks, NPF v2.1 Layer 1A core.

#### 6. `NPF` ⭐ — 新质生产力 v2.1 默认 (主板限定)

| 属性 | 值 |
|---|---|
| 大小 | **401 stocks** |
| Type | static |
| 定义 | NPF Layer 1A + 1B 主板限定. SW L1 ∈ 7 行业 ∩ MAIN_BOARD. |
| 涨跌停 | ±10% 一致 (主板) |
| 用途 | **Production default NPF universe**. 干净小池子, H1 fwd20 IC 强 (matrix v2 wave_v3 H1 fwd20 +7.24). 2022+ window 上 Q1 regime 偶尔 mismatch (regime-aware label v3/v4 救场). |

#### 7. `NPF_FULL` — NPF 全集 (主板)

| 属性 | 值 |
|---|---|
| 大小 | 618 stocks |
| Type | static |
| 定义 | NPF v2.1 (Layer 1A + 1B) + Layer 2 hot_cross + Layer 3 dc_concept, 主板限定 |
| 用途 | **大池子 + 干净涨跌停**, 适合 paris 大 panel (377 cols) 的 LGB 训练 (paris ablation v3 显示 NPF_FULL > NPF on IC). 我侧 228 cols panel 上 NPF 反超 NPF_FULL (feature 不够多 时小池子赢). |

#### 8. `NPF_CROSS_BOARD` — NPF 跨板块 (含创科北)

| 属性 | 值 |
|---|---|
| 大小 | 779 stocks |
| Type | static |
| 定义 | NPF Layer 1A + 1B **不限主板** (含 + 跨板 ~378 stocks 创业/科创/北交所内 NPF 公司) |
| 涨跌停 | 混 (主板 ±10% + 创业/科创 ±20% + 北交 ±30%) |
| 用途 | **仅 exploration**, paris 主推 NPF v2.1 (401). 跨板块涨跌停 label 分布尾部不一致, **不推荐 production**. matrix v2 实测: 跟 NPF (401) IC 几乎相同 (因为 paris wave label MAIN_BOARD-only, 378 cross-board 是 shadow 无 label). |

#### 9. `HARD_TECH` — 硬科技核心 (NPF Layer 1A)

| 属性 | 值 |
|---|---|
| 大小 | **193 stocks** |
| Type | static |
| 定义 | NPF v2.1 Layer 1A L2 core: **半导体 + 通信 + 计算机 + 电池 + 军电** (5 个 L2 行业核心) |
| 用途 | **最干净小池子**, H1 短线 alpha 巨强 (matrix v2 wave_v2 H1 fwd20 IC **+10.18**, fwd5 +7.69). Q1 低波段反而最脆 (193 太少 regime shock 放大, Q1 fwd20 -4.13). 适合短线 + 强 regime 状态. |

---

## 类比关系 (NPF 系 hierarchy)

```
NPF_CROSS_BOARD (779) — 跨板块全集
   │
   ├── NPF_FULL (618) ── 主板限定 + Layer 2/3
   │   │
   │   └── NPF (401) ⭐  ── 主板限定 + Layer 1A+1B (paris production default)
   │       │
   │       └── HARD_TECH (193) ── NPF Layer 1A L2 core (硬科技教科书定义)
   │
   └── (跨板 ~378 + Layer 2/3 跨板) — 不单独 ship
```

成员关系 (集合 ⊆ 表示子集):
```
HARD_TECH ⊆ NPF ⊆ NPF_FULL ⊆ (NPF_CROSS_BOARD 主板部分)
HARD_TECH (193) ⊆ NPF (401) ⊆ NPF_FULL (618) ⊆ NPF_CROSS_BOARD ∩ MAIN_BOARD
```

---

## 使用 (`UniverseFilter` enum)

```python
from aurumq_rl.data_loader import UniverseFilter, filter_universe

# Static universes — frozenset[str] lookup
out = filter_universe(df, UniverseFilter.NPF)              # 401 stocks
out = filter_universe(df, UniverseFilter.NPF_FULL)         # 618 stocks
out = filter_universe(df, UniverseFilter.NPF_CROSS_BOARD)  # 779 stocks (exploration only)
out = filter_universe(df, UniverseFilter.HARD_TECH)        # 193 stocks
out = filter_universe(df, UniverseFilter.MAIN_BOARD)       # 3003 stocks
out = filter_universe(df, UniverseFilter.GROWTH_BOARDS)    # 2253 stocks

# PIT universes — inner-join on (ts_code, trade_date), requires trade_date col in input df
out = filter_universe(df, UniverseFilter.CSI300)           # ~300/day
out = filter_universe(df, UniverseFilter.CSI500)           # ~500/day
out = filter_universe(df, UniverseFilter.CSI1000)          # ~1000/day

# Legacy aliases (向后兼容)
UniverseFilter.MAIN_BOARD_NON_ST  # → MAIN_BOARD
UniverseFilter.HS300              # → CSI300
UniverseFilter.ZZ500              # → CSI500
```

---

## 何时用哪个 universe (per matrix v2 数据指导)

| 场景 | 推荐 Universe | 配套 |
|---|---|---|
| **PROXIMITY 短线 (fwd5)** | **HARD_TECH** (193) | wave_v4 + J_take_profit_5 dyn exit. 期望 H1 fwd5 IC +8.20, Sharpe gross +5.4. **Q1 regime 时不出击** (Q1 fwd5 -0.6 ~ +0.7 弱). |
| **WAVE 中线 (fwd20)** | **NPF** (401) | wave_v3 + I_kdj_death dyn exit. H1 fwd20 +7.24, Q1 fwd20 +2.78 (v3 regime-aware bonus 救 Q1). |
| **Q1 低波段防守** | **CSI300** (大盘 ~300/day) | wave_v2 + held >= 20d. Q1 fwd20 IC **+13.89** (flight to quality alpha). |
| **大 panel + 大 universe (paris 风格)** | NPF_FULL (618) | paris 377-col combined_panel + LGB num_leaves=127. 大 panel 需要更多 samples → NPF_FULL > NPF. |
| **指数增强 / benchmark** | CSI300 / CSI500 / CSI1000 | 套对应指数 PIT 选股, 计算超额收益 |
| **市场全集 baseline** | MAIN_BOARD (3003) | A 股主板基础 universe, 跟 paris 所有 production model 训练 universe 一致 |
| **创科板独立建模** | GROWTH_BOARDS (2253) | 等 paris ship 创科板 wave label 后启用. 现阶段 skip. |
| **NPF 跨板块 exploration** | NPF_CROSS_BOARD (779) | 仅 research, 不 production (涨跌停规则混) |

---

## paris 端 OSS 落盘

```
<PRIVATE_OBJECT_STORE>/handoffs/2026-05-14-5-universe-lock/
├── CSI300_membership.parquet              31,500 rows PIT
├── CSI500_membership.parquet              26,000 PIT
├── CSI1000_membership.parquet             34,001 PIT (含 Tushare 2022 H1 backfill)
├── MAIN_BOARD_membership.parquet           3,003 static
├── NPF_membership.parquet                    401 static (v2.1)
├── NPF_FULL_membership.parquet               618 static
├── NPF_CROSS_BOARD_membership.parquet        779 static
├── HARD_TECH_membership.parquet              193 static
├── GROWTH_BOARDS_membership.parquet        2,253 static
├── MANIFEST.json                          per-file schema/sha256/rows/date_range
├── DEPRECATED_916.md                      NPF 老 916 deprecation warning
└── README.md
```

**Canonical schema** (所有 universe 统一 2 列):
```
stock_code: String  (Tushare 格式 XXXXXX.SH/SZ/BJ)
trade_date: Date    (PIT: 真实日期; static: NULL)
```

元数据 (schema_version / git_commit / generated_at) 在 MANIFEST.json, **不在 parquet 内**。

---

## NPF v2.x 路线图 (paris 跟进)

| Version | 状态 | 大小 | 说明 |
|---|---|---|---|
| v1 | deprecated | 916 | SW L1 粗筛跨板块, 涨跌停混 |
| v2 | deprecated | 779 | 跨板块 3-layer (Layer 1A + 1B 不限主板) |
| **v2.1** | **current** ⭐ | **401** | 主板限定 Layer 1A + 1B, paris production default |
| v2.2 (规划) | TBD | TBD | paris CLAUDE.md changelog 跟进新 L2 行业 (e.g. 商业航天专门 sub-tag) |

**接口稳定保证**: paris 端 NPF 定义如有大改动, 在 `aurumq-rl/handoffs/<date>-npf-vX/README.md` 用 `git_commit` + `schema_version` 标记, ledashi 端**只读 paris ship 的 parquet, 不自己 derive**, 避免漂移。

---

## 历史 reference

- 2026-05-14 v1 → v1.1: paris ship 9 universe parquets, 加 4 个 universe class + REGISTRY 5→9, CLAUDE.md HARD CONSTRAINT 同步, 前端 dropdown 6→10
- 2026-05-14 PM v1.1 → v15.2: 修 schema 统一 / CSI1000 backfill / HARD_TECH static / router has_* / holdings 拆 retro vs paper_trade / NPF 916→401 覆盖 / 加 MANIFEST.json
