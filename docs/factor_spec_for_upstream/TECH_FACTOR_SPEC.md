# Factor Spec for Upstream Parquet Pipeline

> 给数据 pipeline 维护者: 这些技术因子需要在 parquet 生成阶段计算并作为列加入,
> 命名前缀 `tech_*` / `cmf_*` / `zt_*`。RL 项目按 prefix 自动消费,**RL 项目自身不计算因子**。
>
> **Phase 24/25 的失败教训**: RL 项目曾尝试在 panel-load 时用 close-only 近似计算这些因子,
> 但缺 high/low/open 导致 KDJ/振幅严重失真,加上二元因子 z-score 噪声,模型反而退步。
> 必须由上游 pipeline 用完整 OHLC 数据计算。

## 数据范围与命名规范

| 维度 | 说明 |
|---|---|
| 频率 | 日频(交易日) |
| 时点对齐 | 因子计算的截止时点 = `trade_date` 当日收盘后 |
| 缺失值 | 前 N-1 天(N=window 长度)用 NULL 或者 forward-fill 标记 |
| 数据类型 | float32 或 float64,`*_event` 类用 int8 (0/1) |
| 命名 | snake_case,统一小写 |

## 推荐 36 个技术因子(分 7 组)

### Group A: MA 系列(prefix `tech_`)

```sql
-- 输入: close (日收盘价)
ma5  = AVG(close) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS 4 PRECEDING)
ma10 = AVG(close) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS 9 PRECEDING)
ma20 = AVG(close) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS 19 PRECEDING)
ma60 = AVG(close) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS 59 PRECEDING)

tech_close_vs_ma5  = close / ma5  - 1.0
tech_close_vs_ma10 = close / ma10 - 1.0
tech_close_vs_ma20 = close / ma20 - 1.0
tech_close_vs_ma60 = close / ma60 - 1.0

-- 状态标志(连续比距离更稳定)— 推荐用距离差代替二元
tech_ma5_minus_ma10_pct  = (ma5 - ma10) / ma10
tech_ma10_minus_ma20_pct = (ma10 - ma20) / ma20
tech_ma20_minus_ma60_pct = (ma20 - ma60) / ma60

-- MA60 ±4% 带状(可保留二元,但建议改成距离 abs)
tech_dist_to_ma60_abs = ABS(close / ma60 - 1.0)
```

**避免二元事件**(我之前用了,事实证明 z-score 后噪声大):
- ❌ ~~tech_ma5_above_ma10 (二元)~~ → 用 `tech_ma5_minus_ma10_pct` 连续值
- ❌ ~~tech_ma5_cross_ma10 (二元事件)~~ → 用 `LAG(tech_ma5_minus_ma10_pct, 1)` 看趋势

### Group B: KDJ(prefix `tech_`,**必须用真 OHLC**)

```sql
-- 输入: high, low, close (日内极值)
high_9d = MAX(high) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS 8 PRECEDING)
low_9d  = MIN(low)  OVER (PARTITION BY ts_code ORDER BY trade_date ROWS 8 PRECEDING)
rsv = (close - low_9d) / NULLIF(high_9d - low_9d, 0) * 100

-- 递推平滑(SQL 难表达,用 Python/Polars 计算)
-- K[t] = (2/3) * K[t-1] + (1/3) * RSV[t],初始 K[0] = 50
-- D[t] = (2/3) * D[t-1] + (1/3) * K[t],初始 D[0] = 50
-- J[t] = 3*K[t] - 2*D[t]

tech_kdj_k = K (clip to [0, 100])
tech_kdj_d = D (clip to [0, 100])
tech_kdj_j = J (clip to [-50, 150])

-- 推荐用 K - D 连续值代替金叉二元
tech_kdj_k_minus_d = K - D
```

**为什么强调真 OHLC**: 我之前用 close 近似 RSV,信号比真 OHLC 弱很多。日内是否触及 9 日新高/新低**只有 high/low 能告诉你**。

### Group C: MACD(prefix `tech_`,close-only 标准做法)

```python
# Polars / pandas 风格
ema12 = close.ewm(span=12, adjust=False).mean()
ema26 = close.ewm(span=26, adjust=False).mean()
dif = ema12 - ema26
dea = dif.ewm(span=9, adjust=False).mean()
hist = 2.0 * (dif - dea)

# 归一化:除以 close 让跨股可比(否则高价股 MACD 大,低价股 MACD 小)
tech_macd_dif_norm = dif / close
tech_macd_dea_norm = dea / close
tech_macd_hist_norm = hist / close
tech_macd_dif_minus_dea_norm = (dif - dea) / close   # 替代金叉二元
```

### Group D: Bollinger(prefix `tech_`)

```sql
mid20 = AVG(close) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS 19 PRECEDING)
std20 = STDDEV(close) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS 19 PRECEDING)
upper = mid20 + 2.0 * std20
lower = mid20 - 2.0 * std20

tech_boll_pct_b      = (close - lower) / NULLIF(upper - lower, 0)   -- [0=底, 1=顶]
tech_boll_band_width = (upper - lower) / mid20                       -- 相对宽度

-- Squeeze:当前 band_width 在过去 60 日的相对低位 (用 percentile rank 比 hard threshold 好)
tech_boll_band_width_pct_60d = PERCENT_RANK(tech_boll_band_width)
                               OVER (PARTITION BY ts_code ORDER BY trade_date ROWS 59 PRECEDING)
```

### Group E: 量能 / 振幅(prefix `tech_`,**用真 high/low**)

```sql
vol_ma20 = AVG(vol) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS 19 PRECEDING)
amount = close * vol     -- 或上游已有 amount 列直接用
amount_ma20 = AVG(amount) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS 19 PRECEDING)

tech_vol_ratio        = vol / NULLIF(vol_ma20, 0)         -- 今日量比
tech_amount_ratio     = amount / NULLIF(amount_ma20, 0)
tech_vol_decay_5d     = AVG(vol, 5) / NULLIF(vol_ma20, 0)

-- 振幅(必须用 high/low!不能用 close 近似)
tech_amplitude_today = (high - low) / NULLIF(prev_close, 0)
tech_amplitude_5d    = MAX(high, 5) / NULLIF(MIN(low, 5), 0) - 1
tech_amplitude_20d   = MAX(high, 20) / NULLIF(MIN(low, 20), 0) - 1
```

### Group F: 累计主力(prefix `cmf_`,从已有 `mf_net_1d` 计算)

```sql
-- mf_net_1d 是已有列(原 mf_* 因子组),元单位
cmf_60d  = SUM(mf_net_1d) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS 59 PRECEDING)
cmf_120d = SUM(mf_net_1d) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS 119 PRECEDING)

-- 占成交额比例(让跨股可比)
cmf_60d_pct  = cmf_60d  / NULLIF(SUM(amount, 60), 0)
cmf_120d_pct = cmf_120d / NULLIF(SUM(amount, 60), 0)

-- 累计净流入天数(60 日)
cmf_pos_days_60d = SUM(CASE WHEN mf_net_1d > 0 THEN 1 ELSE 0 END) OVER (...)
```

### Group G: 涨停统计(prefix `zt_`,**必须用真 limit_up_flag**)

`senti_zt_count_30d` 在当前 parquet 里 4.26M 行只有 59 个非零,**严重损坏**,需要修复或新增。

```sql
-- 真涨停判定(交易所规则,不能简单 pct_chg >= 0.099)
-- 主板 ±10%, 创业板/科创板 ±20%, ST ±5%, 北交所 ±30%, 上市首日不限
-- 上游应该有 limit_up_flag / limit_down_flag 列
is_zt = (limit_up_flag = 1)   -- 不是 pct_chg threshold
is_dt = (limit_down_flag = 1)

zt_count_30d = SUM(is_zt::int) OVER (...30 ROWS...)
zt_count_60d = SUM(is_zt::int) OVER (...60 ROWS...)
dt_count_60d = SUM(is_dt::int) OVER (...60 ROWS...)
zt_dt_imbalance_60d = zt_count_60d - dt_count_60d

-- 连板数(连续涨停天数)— 这个最有用但难写,需要专门函数
zt_consecutive_days = ...   -- 至 trade_date 为止的连续涨停天数(0 = 今日不涨停)
zt_max_step_60d = MAX(zt_consecutive_days) OVER (...60d...)
```

### Group H: 缺失字段建议补齐

如果 parquet 还没这些,**强烈建议加上**:

| 字段 | 为什么需要 |
|---|---|
| `open` | 真实买入价(回测假设次日开盘买入) |
| `high`, `low` | KDJ / 振幅必需 |
| `amount` | 已有 close*vol 替代但精度差(集合竞价/收盘集合竞价的成交价不同) |
| `limit_up_flag`, `limit_down_flag` | 真涨停判定 |
| `is_halt`(停牌标记) | 比 `vol == 0` 更准确 |
| `pre_close`(前收盘) | 计算开盘跳空,必需 |

## 验收标准(给数据 pipeline 维护者)

新 parquet 应满足:

- [ ] 36 个新列全部存在,prefix 正确
- [ ] 无 NaN(用 NULL 或 forward-fill,RL 项目接受 NULL)
- [ ] 数值范围合理:
  - `tech_kdj_k/d` ∈ [0, 100]
  - `tech_kdj_j` ∈ [-50, 150]
  - `tech_close_vs_ma*` ∈ [-0.5, 0.5] 99%分位
  - `tech_macd_*_norm` ∈ [-0.1, 0.1] 99%分位
  - `cmf_*_pct` ∈ [-2.0, 2.0] 99%分位
  - `zt_count_*` ∈ [0, 30 or 60]
- [ ] 跨股 cross-section 检查:某天的 30 个新因子,std > 1e-6(否则 z-score 后全变 0)
- [ ] **不要发布二元事件因子**,RL 项目要连续值

## RL 项目侧验收

收到新 parquet 后:

1. 在 `STOCK_FACTOR_PREFIXES` 加入 `tech_`、`cmf_`、`zt_`
2. 删除 `src/aurumq_rl/technical_factors.py`(不再需要)
3. 删除 `data_loader.add_technical_factors=True` 路径
4. 删除 `--add-technical-factors` CLI flag
5. 运行 `_inspect_factor_at_t_minus_k.py --prefixes tech_ cmf_ zt_` 看真实因子的 T-1 信号
6. 训 Phase 26A 用 `--reward-mode main_wave_target` + 全因子(包括正确算的 tech)
7. 对比 Phase 23A,看真实 tech 因子是否提升 T-1 命中率

## 下一步:**等数据团队消化此文档,出新 parquet**

期间 Phase 25D(353 base + weights only)继续跑(避免被错误 tech 因子污染),用来纯净验证"importance 加权范式"是否本身有效。

Phase 26 等真 tech 因子来了再启动。
