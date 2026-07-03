# 全仓库代码审查报告(2026-07-02)

6 个并行审查域:核心 RL、数据管线/标签、alpha101/gtja191 算子库、alpha101 分类因子文件、p3/Kronos 管线、训练/评估脚本。所有发现均经 agent 读取实际代码验证(部分标注"需数据验证"的除外)。

---

## 一、Critical(直接使结果不可信)

### C1. 前向收益用未复权 close 计算 — `src/aurumq_rl/data_loader.py:751-753`
`return_array[t] = log(close[t+fp]/close[t])` 用原始 `close`。仓库 schema 里有 `adj_factor`,p3 脚本都正确构建了 `adj_close`,但 `export_factor_panel.py:104-115` 导出的是原始 `q.close`,`data_loader` 从未见过 `adj_factor`。10 送 3 除权日 close 跌 ~23% → 面板记录假 -26% 十日对数收益。污染范围:RL 奖励、回测、所有 main_wave 标签(`train_v2.py:423`、`_eval_main_wave_v1.py:209` pivot 同一原始 close,除息缺口造成假死叉/假回撤)。注意 `labeling/` 子包用了 `MarketPanel.adj_close` 是对的——仓库目前混用两种价格口径。
**修复**:导出 `adj_factor`,前向收益和 wave 标签全部用复权价;原始 close/pct_chg 只用于涨跌停和成交额逻辑。

### C2. 缺失 (date, stock) 单元格默认"可交易、因子=0、未停牌、IPO age=120" — `data_loader.py:711-748`
数组零初始化,只写入 parquet 中存在的行。退市/停牌无行/未上市的格子:`is_suspended=False`、`close=0` → 前向收益恰好 0.0、原始因子 0.0 **作为真实观测进入横截面 z-score**(偏移当日所有股票的 mean/std,自身获得系统性非零 z)。下跌日幽灵 0 收益选股虚增业绩。
**修复**:`factor_array`/`close` 初始化为 NaN,缺失格子 `is_suspended=True`(`align_panel_to_stock_list` 对整只缺失股票已经这么做),`_safe_log_return` 对无效价格返回 NaN。

### C3. 静态股票池(2026-05-14 锁定)套全历史 = 幸存者偏差 — `data_loader.py:293-361`
`MAIN_BOARD`/`NPF*` 等是无日期维度的 `frozenset`;第 355 行还按**当前名称**含 ST/退 删除整段历史。2024 后退市加速的 A 股,这类偏差通常每年虚增数个百分点。p3 的 `MAIN_BOARD`/`NPF`/`HARD_TECH` 冻结集合同样存在(2022→2026 全程套用)。
**修复**:point-in-time 成分(CSI300/500 已有先例),按当日 `is_st` 过滤而非当前名称。

### C4. GPU 训练路径完全不检查涨跌停 — `scripts/train_v2.py:376-381` + `src/aurumq_rl/gpu_env.py:131-141`
`valid_mask` 只有 ST/停牌/IPO 三项;CPU env 的 `_apply_trading_mask` 有板别涨跌停逻辑但 V2 路径不用。涨停连板收益强正 → 策略学会追一字板(生产中买不进)。回测端同样无过滤(见 M6),训练/评估双双虚高。
**修复**:把 `pct_change_array` 传入 GPU env,用 `price_limits` 预计算的每股涨跌幅带向量化 AND 进 `valid_mask`。

### C5. 奖励函数除零爆炸 — `src/aurumq_rl/reward_functions.py:82-83, 113-116`
rolling Sharpe/Sortino 除以 `max(std, 1e-8)`:episode 第 0 步 σ=0 → 奖励 ≈ ±1e6~1e7;Sortino 在近 20 步全非负时(反弹行情常见)同样爆炸。每 episode 一个尖峰就摧毁 value target 和 GAE 尺度。
**修复**:窗口样本 <5 时返回 0,σ 下限设 1e-4;更好的方案是 Differential Sharpe Ratio(见算法建议 A1)。

### C6. ONNX 导出的是随机采样而非确定性动作 — `src/aurumq_rl/onnx_export.py:190-205`
`torch.onnx.export(policy, ...)` trace 了 `forward(deterministic=False)`,图内含高斯 `sample()`。生产推理输出 `mean + 0.5·noise`,与 SB3 zip 评估路径系统性不同;`inference.py` 的 `deterministic=True` 参数是无效摆设。无 SB3→ONNX parity 测试所以从未被发现。
**修复**:导出 wrapper(`features_extractor` + `action_net` 或 `policy._predict(deterministic=True)`),并加 `np.allclose` parity 测试。

### C7. `infer.py` 分数映射到错误股票 — `scripts/infer.py:100-153`
忽略 `metadata.factor_names`/`stock_codes`,按前缀发现因子 + 今日股票池扁平化,再按**尾部**截断/填充到模型 obs 维度。obs 是 stock-major,股票池任何差异都会平移所有股票的因子块;分数位置属于训练股票池、名字却来自今日股票池。生产选股在股票池不一致时(必然)基本是乱的。Phase 16 在 `eval_backtest.py:88-152` 修过完全相同的 bug。
**修复**:复用 `align_panel_to_stock_list` + metadata 因子名对齐,与 `eval_backtest.py` 一致。

### C8. `--vec-normalize` 训练的模型评估/导出不带归一化统计 — `scripts/train.py:637-773`
训练用 `VecNormalize(norm_obs=True)` 并保存 `vec_normalize.pkl`,但 ONNX 导出裸策略,`eval_backtest.py`/`infer.py` 从不加载该 pkl。`compare_rewards.py:69` 无条件传 `--vec-normalize`,其产出的所有对比数字都经过错配观测,结论无意义。
**修复**:把 `obs_rms.mean/var` + clip 折进导出图或 metadata,评估端应用。

### C9. v13 `*_BASE` 对照组训练/打分模型不一致 — `scripts/p3/kronos_matrix_v13.py:504-569`
Phase 3 训练用 `base_embs`,eval 打分却无条件用微调模型的 `eval_embs`(phase 2 `--base-model --eval-window` 写出的 `..._base_eval.parquet` 从未被加载)。base-vs-finetuned 三方对照(paris ACK §3)无效,可能假性显示"微调无增益"。
**修复**:`is_base` 格子加载并使用 `embeddings_..._base_eval.parquet`,训练前断言其存在。

### C10. polars 版本下限声明错误 — `pyproject.toml:42`
`polars>=0.20` 太松:`Expr.rolling_rank` 在 1.34.0 不存在(alpha101 ~106 处 + GTJA TSRANK 全灭,`AttributeError`);alpha026 的嵌套窗口在 1.34 抛 `InvalidOperationError`、1.42.1 正常(双向实测验证)。按声明下限安装得到的是坏库。
**修复**:下限提到 1.42(已验证可用)。

---

## 二、Major

### 训练/评估一致性与统计有效性
- **M1. 信号在计算它的同一收盘价成交(1 天前视)** — `data_loader.py:751`:因子含 t 日收盘信息,成交价也是 t 日收盘。`main_wave_labels` 用 `close[t+1]` 入场是对的,核心面板不是。修复:收益改为 `close[t+1]→close[t+1+fp]`。
- **M2. 重叠 10 日收益按日复利,P&L 约 10 倍重复计入** — `env.py:330-347`、`gpu_env.py:139-141`(训练奖励)+ `backtest.py:195-208, 399-405`(cumret/equity 曲线,还把对数收益当单利乘)。Phase 16 只修了 Sharpe 没修 cumret。修复:按 `forward_period` 步长复利或持仓式记账。
- **M3. checkpoint 选择/晋级在报告用测试窗口上 argmax(winner's curse)** — `_eval_all_checkpoints.py:207`、`phase26ef_scoreboard.py:32-168`、`_ensemble_eval.py`(成员本身就是该窗口选出的 `*_best.zip`)。"2.61× T-1 lift" 类基线上偏。修复:选择窗/终测窗分离或 walk-forward + Deflated Sharpe/PBO(算法建议 A4)。
- **M4. v13 early-stopping 验证集未按日期排序切分、无 embargo** — `kronos_matrix_v13.py:530-538`、`kronos_matrix_v12.py:169-171`:`train.tail(10%)` 切自未排序 frame;anchor 标签含未来 ~20 日 wave 事件,边界样本与训练共享事件窗。修复:按日期排序切分 + ≥30 交易日 embargo。
- **M5. 回测在不可得价格成交** — `backtest.py:132-144` 只按 `isfinite` 过滤;涨停/停牌/ST 股自由入选 top-K,而 GPU 训练却屏蔽 ST/停牌——训练评估方向性不一致。`_eval_main_wave_v1.py:126-143` 已有正确做法。修复:传 eligibility mask,不可交易置 -inf。
- **M6. eval 默认股票池与训练默认矛盾** — `eval_backtest.py:34` 默认 `all_a`,训练默认 `main_board_non_st`;z-score 在 `align` 之前按加载股票池计算,横截面统计整体偏移。修复:默认读 metadata 中的 universe,不一致时告警。

### 数据/标签
- **M7. price_limits 多处 A 股规则错误** — `price_limits.py:126-133` ST 判断先于板别:创业板/科创板/北交所 ST 应保持 ±20%/±30% 而非 ±5%;`:121-124` 注册制新股前 5 个交易日无涨跌幅(现只豁免上市首日);`:85-96` 689xxx(科创板 CDR)落入主板 ±10%;`:141-159` ε=1e-3 漏检低价股真实涨停(限价按 0.01 元取整)。
- **M8. 入场资格只查决策日 t,不查实际入场日 t+1** — `main_wave_labels.py:229-237`:模型要找的正是 t+1 一字涨停/复牌概率最高的股票,标签把不可成交的入场全额记为命中,目标类命中率系统性乐观。修复:AND `~is_suspended[t+1]` 和 t+1 非一字板。
- **M9. 面板尾部截断持有窗仍标 `label_valid`** — `main_wave_labels.py:304-315`:`T−entry<H` 时对 1-4 天路径测 5 天阈值,eval 窗口末尾标签系统性更难命中。`labeling/events.py:151` 的处理是对的。修复:`label_valid &= (T−entry>=H)`。
- **M10. `fill_null(0.0)` 的 close 毒化 MA/vol** — `train_v2.py:412`、`_eval_main_wave_v1.py:152` → `main_wave_labels.py:164-227`:中途缺行的 0 价造成假死叉、流动性 MA 稀释、路径中 `max_adverse_excursion=-1` 否决好波段。修复:pivot 处 ffill 或 NaN 传播,路径含 NaN 判 label-invalid。
- **M11. directional_change 事件在 θ 确认日而非波段高点发出** — `labeling/directional_change.py:73-77`:overshoot(信息量所在)被丢弃,`event_quality≈θ/θ_min` 近乎确定值 → `dedupe_events` 机械保留 θ=0.08 副本,`search_threshold` 失去意义。修复:在向下反转时用记录的 `extreme_high` 发事件。

### 因子库
- **M12. alpha080/089 硬编码 `adv15`,公式要求 `adv10`** — `industry_neutral.py:861, 998`:生产面板明明有 `adv10`(alpha081 直接在用),这两个因子永远算的是另一个因子。
- **M13. 裸 `ts_corr` 的 NaN 进入 `ts_rank` 会成为窗口最高秩** — `volume_price.py`、`industry_neutral.py`、`adv_extended.py`、`momentum.py:362`、`technical.py:85` 未用 `ts_corr_safe`;polars 浮点全序 NaN 最大,一字板/停牌股在恰好与未来收益相关的日子产生虚假强信号;registry 消毒器只清 ±inf 不清 NaN(`registry.py:63-85`)。
- **M14. `pl.when(null)→otherwise` 在 warm-up 期捏造恒定信号** — alpha007 前 ~19 天恒为 -1(`momentum.py:77`)、alpha027 恒 +1、alpha021 恒 -1、alpha023/065/068/074/081/099 同类;新股连续上市使当日横截面 z-score 被系统污染。修复:`pl.when(cond.is_null()).then(None)` 前置。
- **M15. `cs_scale` 无 NaN/零分母防护,一个 NaN 毁掉全天横截面** — `alpha101/_ops.py:457-465`(实测验证);alpha028 路径:一只常数窗口股票 → 当日所有股票 NaN;波及 alpha028/029/031/032/060/100。
- **M16. GTJA `sma` 用 `adjust=True`,论文递推是 `adjust=False`** — `gtja191/_ops.py:183-201`(数值实测最大偏差 0.26,几何衰减),52 处调用;docstring 声明与实现不符。
- **M17. alpha101 `ts_rank` 归一化 `(rank-1)/(w-1)` 与 pandas pct rank、STHSF、gtja191 三者都不一致** — `alpha101/_ops.py:232-248`;喂相关性的场合无害(仿射不变),但 `rank ^ Ts_Rank`、alpha095 类比较处结果偏移。
- **M18. `industry`/`cap` 列 point-in-time 属性无从验证** — 若上游按最新申万分类/当前市值静态 join,35 个行业中性因子 + alpha056 全部泄漏。需要向面板构建方确认 as-of join,并在 registry 边界加 PIT 契约断言。

### p3 / 环境
- **M19. smoke 与正式跑共用 checkpoint/预测文件** — `kronos_matrix_v13.py:481, 552`:先 `--smoke` 后正式跑,所有格子被静默跳过,phase 4 评估的是 1/10 采样训练的 smoke 预测;`skipped` 条目永久跳过。修复:路径加 `_smoke` 后缀,`skipped` 不算完成。
- **M20. "D-1 泄漏防护单测"是空测试** — `kronos_matrix_v13.py:314-326`:手造两个错位窗口断言不同(恒真),从不触碰真实抽取切片;第 370 行写成 `idx+1` 它也通过。(实际窗口算术验证是对的,但测试不设防。)
- **M21. 停牌股奖励 hack 可被 PPO 利用** — `src/aurumq_rl/p3/env.py:132-134`、`gpu_env.py:192-198`:NaN 次日收益置 0 后仍进 top-k 均值,下跌行情中选中"预测将停牌"的股票获得正超额——不可实现策略。GPU/CPU env 在 `n_uni<top_k` 时还有 parity 分歧(`gpu_env.py:186` vs `env.py:124`)。
- **M22. 高斯动作裁剪产生大规模并列,top-30 按下标选** — `policy.py:91-94` + `gpu_env.py:93-98`:σ=0.5、均值近 0,SB3 将 Box 动作裁到 [0,1] → ~一半动作恰为 0、约 69 只恰为 1.0(> top_k=30),执行组合与策略排序脱钩。修复:action space 改无界(自定义 VecEnv 无需 [0,1])或 squashed 分布。
- **M23. 固定长度 episode 截断未标 truncation** — `gpu_env.py:157-167`(及 `env.py:354`、p3 `gpu_env.py:216-226`):SB3 不做 value bootstrap,240 步截断处 value target 归零,且 obs 无时间特征 → 值函数混叠。修复:`infos["TimeLimit.truncated"]=True` + `terminal_observation`。
- **M24. 交易成本与换手无关** — `env.py:345`、`portfolio_weight_env.py:330`、`gpu_env.py:141`:任何换手都收满额 30bps(gpu_env 甚至零换手也扣),策略面临零边际换手成本。修复:`cost = cost_bps/1e4 × turnover/2`。
- **M25. `log_size` 在 D 日 join 而 embedding 止于 D-1** — `kronos_matrix_v13.py:514-516`:同日信息进特征,与"严格 D-1"表述矛盾(收盘执行口径下可辩护,但两个口径必须统一)。

---

## 三、性能优化(按预期收益排序)

- **P1. `data_loader._df_to_panel` 纯 Python 行循环** — `data_loader.py:723-748`:~2.4M 行 × 345 列 ≈ 8×10⁸ 次 Python 级操作,每个训练/评估脚本都付这笔启动成本。向量化(date/stock 索引数组 + `df.select(...).to_numpy()` 一次写入)可提速 100-1000×。p3 的 `data.py:249-259`(文档自述 8 分钟首载)同理,`searchsorted` 化后 ~10 秒。**三个 agent 独立命中此项。**
- **P2. GPU 路径每步 ~100MB PCIe 往返** — `gpu_env.py:229-239` obs GPU→CPU→numpy,SB3 立刻转回 cuda;`--rollout-buffer gpu` 模式再复制一次 host→device。覆写 `collect_rollouts` 直读 cuda tensor 可消除。这与 4070 只有 ~11% 利用率直接相关(另一半原因:train.py 默认 batch_size=64 + SubprocVecEnv spawn 把整个面板 pickle 进每个 worker)。
- **P3. v13 phase 3 每格重复 merge/decode ~5M 行 eval embedding(~20GB pandas)** — `kronos_matrix_v13.py:459-467, 554-561`:merge/fill/coerce 提出循环、按 universe(6 而非 23)预过滤、float16 memmap,可省 20+GB 峰值内存、数倍时长。配套:`iterrows` 建任务表(`:344-352`)、逐行 `np.frombuffer`(`:435-437`)、BATCH=64 fp32 双次前向(`:330,381`,autocast+大 batch 可砍一半以上 GPU 时间)。
- **P4. `ts_decay_linear`/`wma` 每个 shift 一次 `.over` 分区扫描** — `alpha101/_ops.py:271-290`、`gtja191/_ops.py:204-221`:polars `rolling_sum(weights=…)` 原生支持(1.42 实测),单核融合 kernel,~w 倍减少分配;`_window_arg_extreme`/`ts_rank_int` 的 O(n·w) when-链同理(bottleneck `move_argmax`/`move_rank` 是 O(n) C kernel)。
- **P5. 292 个因子各自重算共享子表达式** — `cs_rank(volume)` 在十几个因子里重建;LazyFrame + CSE 或预 enrichment 一次性算 ~10 个最常用列,全量面板构建多倍提速。
- **P6. `export_factor_panel._execute_paged` 全结果集 Python list 物化** — `export_factor_panel.py:357-446`:改 pyarrow `ParquetWriter` 增量写,消除面板构建最大内存尖峰。`scan_parquet` 全列 collect(`data_loader.py:633`)加投影下推,26GB 峰值按未用列比例直降。
- **P7. 其他**:`train_v2.py:393` 等处读全 345 列只用 3 列(改 `scan_parquet().select()`);`compute_realized_and_exits` 每次 v10/v12/v13-p4 运行重算(持久化 parquet);`reward_functions.py` O(T²) 历史重算 + 每步 3000×3000 协方差(deque + 组合收益方差);随机基线每 variant 重算;LGB 1024 维连续特征建议 `max_bin=63/127` 或 GPU。

---

## 四、更好的算法

- **A1. 奖励:Differential Sharpe Ratio(Moody & Saffell)** 替代 rolling-Sharpe-as-reward——精确的每步 Sharpe 增量、有界、O(1),顺带消灭 C5 和 O(T²) 历史问题。
- **A2. 策略头:Plackett-Luce / perturbed top-k** 替代 3014 维对角高斯——`log_prob` 累加 ~3000 项使 PPO ratio 立即饱和 clip(所以才需要 target_kl=0.2);env 只消费排序。至少换 gSDE + 小 log_std 初始化。
- **A3. 验证协议:purged + embargoed walk-forward CV**(López de Prado)用于 early stopping 和一切超参选择——anchor/triple-barrier 标签都是前向构造,任何未 purge 的切分构造性泄漏。
- **A4. 评估统计:Deflated Sharpe Ratio + PBO(CSCV)**,N = checkpoints × seeds × tiers;Spearman rank IC 替代 Pearson(概率/分数单调任意、收益厚尾);重叠收益用 Newey-West/HAC 或 block bootstrap 而非 `sqrt(252/fp)`;回测加成本/换手调整(训练扣 30bps 回测记零成本,两边度量的不是同一个量)。v13 phase 4 的 22 格 × 7 horizon × 7 sizing 选优是多重比较,选择用 H2_2025、确认用 Q1_2026,并接上已有的 bootstrap CI 脚本。
- **A5. p3 head:线性/logistic probe + 小 MLP** 替代(或对照)LightGBM——GBDT 对稠密分布式 embedding 是弱归纳偏置,线性 probe 才是 embedding 质量的诚实度量,也是更干净的 null-vs-base-vs-finetuned 对照;或 LoRA 端到端微调;若保留树,拼接 344 因子面板隔离 embedding 增量价值。
- **A6. 标签:CUSUM 事件采样 + 样本并发加权**(LdP ch.2.5/4)进 triple-barrier;trend_scanning 的 t-stat 需截断/tanh 压缩(否则低波微漂移主导阈值搜索);`v2_excess_adaptive` 的 `1.8·日波动` 与最长 20 日累计收益比较缺 √horizon 缩放,6% 地板几乎恒 binding,"自适应"名存实亡(P0 锁定配置,改动需重跑消融)。
- **A7. 因子工程**:行业中性化改回归式(行业哑变量 + log 市值,A 股行业×市值强纠缠)+ `min_group_size` 防单股子行业恒零;波动率估计换 Yang-Zhang(隔夜缺口大的 A 股增益明显);z-score 前加每日 MAD winsorize(±5 MAD)——现在唯一的离群控制是 ±1e6 clip,一个 1e6 值仍能压垮全天 std。
- **A8. 探索多样性**:CPU env 每次 reset 到 t=0,n_envs 个 worker 回放同一轨迹;ResidualGPUEnv 16 envs 同起点锁步。随机化每 env 起始日期,零成本去相关。
- **A9. 特征提取器**:value head 只见 per-stock embedding 的 (mean, max) 池化,actor 独立打分——加一层 ISAB 交叉注意力或 std/top-k-mean 池化,让模型表达广度/离散度 regime。
- **A10. 增量因子计算协议**:当前 `impl(df)→Series` 契约下日更需全历史重算;最深回看有界(252d),per-stock tail-buffer 的 `impl_incremental` 使日更为 O(stocks × max_window)。

---

## 五、Minor(节选,完整见各 agent 原始输出)

- `rank_z.py:49` `method="ordinal"` 平局按行序,训练/推理行序不同则不确定(改 `"average"`)。
- `gtja191/_ops.py:393-403` `sequence()` 构造即坏(长度不广播)、已死代码但在 `__all__` 导出。
- registry 消毒器 NaN 直通(`registry.py:63-85`,加 `.fill_nan(None)`)。
- 无排序防护:全部 TS 算子假设 `[stock_code, trade_date]` 升序,倒序面板 = 全库前视。registry 边界加断言。
- `volume_price` 23/31 因子在 `_KNOWN_PARITY_DIVERGENT` xfail 集合,~75% 模块数值回归不可执行。
- `sb3_callbacks.py:63,164` `num_timesteps % freq` 与 n_envs 相位问题,实际保存/日志频率错位。
- `_ensemble_eval.py:244-250` 分数缓存只验 shape,重训后静默复用旧模型分数。
- `feature_extractor.py:84-103` `unique_date` 按 stock-0 因子行浮点哈希去重,全零行跨日碰撞(改用 t 索引)。
- `kronos_matrix_v13.py`:768/1536 维文档 vs 512/1024 实现;`"cells": 22` vs CELL_SPEC 23;`SEQ_LEN_LONG=120` 超微调 90d lookback(OOD 风险);`idx-1<SEQ_LEN_LONG` off-by-one 丢每股首个合格锚点日。
- `train.py` universe choices 缺 CLAUDE.md 指定的 `npf` ⭐ 等;训练窗口默认值与 CLAUDE.md 锁定值矛盾;smoke test 忽略 `--seed`。
- `pct_chg` 无小数/百分比形式断言(百分比形式仓库会静默禁用涨跌停掩码)。
- `backtest.py:288` `forward_period==1` 留一行全零尾行;回撤用加法差近似(多处)。
- p3 `data.py:44-48` MANIFEST 缺失时 cache key 为 `""`,过期缓存可被接受。
- v10 自适应 gating 分位数取自 in-sample 训练窗预测,OOS 端过度降仓。

---

## 交叉验证一致的主题(多 agent 独立命中,置信度最高)

1. 未复权价 + 缺失行填 0 + 静态股票池 —— 数据层三大偏差,方向全部虚增回测。
2. 涨跌停/停牌在训练(V2)与回测双双缺失,而标签层(main_wave t+1 入场)有意识地做对了一半。
3. 重叠 forward return 的日频复利,训练奖励与回测 cumret 同病。
4. 在测试窗口上做模型选择(checkpoints、ensemble、matrix cells)。
5. NaN/warm-up 被静默转成信号(z-score 填 0、when-otherwise、ts_rank-NaN、cs_scale)。
6. `_df_to_panel` Python 循环是全仓库最大单点性能损失。
