# P3 4070 Training — Handoff to ledashi

**Date**: 2026-05-09
**From**: paris (AurumQ data side)
**To**: ledashi (4070 training side)
**Bundle**: `oss://ledashi-oss-sgp/aurumq-rl/handoffs/2026-05-09-p3-4070-training/`
**Total**: 3.72 GB

---

## 0. TL;DR — 一行启动

```bash
# On 4070 box (after env setup)
cd /d/dev/aurumq-rl && git pull origin main
ossutil cp -r oss://ledashi-oss-sgp/aurumq-rl/handoffs/2026-05-09-p3-4070-training/ ./data_p3/
.venv/bin/python scripts/reward_sanity_check.py --bundle ./data_p3   # 必须先过
.venv/bin/python scripts/train_residual_ppo.py --bundle ./data_p3 --smoke 100000
# 100k smoke 通过后再:
.venv/bin/python scripts/train_residual_ppo.py --bundle ./data_p3 --resume --total 300000
```

---

## 1. Pre-flight checklist (必须按顺序过)

### 1.1 Bundle 完整性
```bash
cd ./data_p3
sha256sum -c MANIFEST.sha256.txt    # 由 manifest 衍生
python -c "import json; m = json.load(open('MANIFEST.json')); print(m['total_gb'], 'GB,', sum(f.get('row_count', 0) for f in m['files'].values()), 'rows total')"
# Expect: 3.72 GB, ~6M rows
```

### 1.2 Reward sanity (强制, ALGORITHM_SPEC §7)
```bash
python scripts/reward_sanity_check.py --bundle ./data_p3
```
必须打印:
```
mean(r) ≈ 0.00065   ✓
std(r)  ≈ 0.0266    ✓
nonzero fraction ≈ 0.966 ✓
```
任一不符 → 数据/对齐错, **不许开训**。

### 1.3 Δ 头零初始化自检 (强制, ALGORITHM_SPEC §3 修正 #6)
```bash
python -c "
from aurumq_rl.policy.residual_actor import ResidualActorHead
import torch
head = ResidualActorHead(encoder_out_dim=32)
x = torch.randn(100, 32)
delta = head(x)
assert delta.abs().max() < 1e-3, f'Δ should be ~0 at init, got max={delta.abs().max()}'
print('Δ init OK:', delta.abs().max().item())
"
```

### 1.4 GPU 可用
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expect: True NVIDIA GeForce RTX 4070
```

---

## 2. Bundle 结构

```
data_p3/
├── ALGORITHM_SPEC.md                       # 算法 + config (v2, 9 fixes)
├── HANDOFF.md                              # 本文件
├── MANIFEST.json                           # row count + sha256
│
├── feature_panel_v3_344.parquet            # 3.65 GB, 3.78M rows
│   schema: ts_code, trade_date, + 345 features (fp32)
│   schema_hash: 5e71e158e331
│   覆盖: 2023-01-03 .. 2025-12-31
│
├── baseline_predictions.parquet            # 13 MB, 2.15M rows
│   schema: trade_date, ts_code, p_t3_baseline (fp32), rank_pct_baseline (fp32)
│   产出: P2 v2 5-seed ensemble 推理 (PR_AUC 0.122 lift 3.0×)
│
├── realized_returns.parquet                # 23 MB, 2.18M rows
│   schema: trade_date, ts_code, close_t, close_t_plus_1, trade_date_t_plus_1, pct_chg_t_plus_1
│   t+1 是下一交易日 close-to-close adj_close 对齐
│
├── market_returns.parquet                  # 12 KB, 727 rows
│   schema: trade_date, eq_weight_pct_chg_t_plus_1, n_stocks
│   in-universe 主板等权
│
├── benchmark_main_board_eq_weighted.parquet # 18 KB, 803 rows
│   (历史口径, 用于 P0 标签 excess return; 现 PPO reward 用 market_returns)
│
├── labels/                                 # 27 MB, 50 files
│   labels_A_{t1,t3,e20}_year={2023,2024,2025,2026}.parquet
│   schema: trade_date, ts_code, y (int8)
│   生产用 t3, 其他备用
│
└── universe_mask/                          # 3 MB, 4 files
    year={2023,2024,2025,2026}.parquet
    schema: trade_date, ts_code, in_universe (bool), 6 个分项 flag
```

---

## 3. 训练数据切分 (复用 P2 ablation, 修正 #4 加 H1/H2)

```python
TRAIN_EFF = (date(2023, 1, 3),  date(2024, 12, 4))   # 23 个月 + 20 日 embargo
VAL_EFF   = (date(2025, 1, 1),  date(2025, 6, 4))    # PPO val (train 监控)
H1        = (date(2025, 7, 1),  date(2025, 9, 30))   # ckpt 选择 + isotonic 校准
H2        = (date(2025, 10, 1), date(2025, 12, 31))  # 单次报告, 不许回头
```

---

## 4. PPO env contract — `aurumq_rl.env.residual_ppo_env.ResidualPPOEnv`

```python
class ResidualPPOEnv(gym.Env):
    """
    Step semantics:
        obs[t]  = concat(panel_features[t,:], p_baseline[t,:], rank_pct_baseline[t,:])
        action  = Δscore[j] for j in current universe, ∈ [-0.2, +0.2] (post tanh*0.2)
        reward  = sum_{j in top_k} (pct_chg_t_plus_1[t, j] - eq_weight_pct_chg_t_plus_1[t])
                  / k    (mean excess return of selected k stocks)
        done    = t == last_train_day - 1

    PPO 输出 Δ, env 内部用 score = logit(p_baseline) + λ·Δ 重新排序选 top_k.
    生产 inference: score → isotonic_h1 校准恢复为概率.
    """
    
    def reset(self):
        self.t = self.start_idx
        return self._build_obs(self.t)

    def step(self, action_delta):
        # 1. compute logit-space score
        logit_p = np.log(self.p_baseline[self.t] / (1 - self.p_baseline[self.t] + 1e-9))
        score   = logit_p + self.lambda_logit * action_delta   # action_delta ∈ [-0.2, 0.2]
        
        # 2. select top-k
        topk_idx = np.argsort(-score)[: self.top_k]
        
        # 3. realized excess for the top-k
        excess = self.realized_pct[self.t, topk_idx] - self.market_pct[self.t]
        reward = float(excess.mean())
        
        self.t += 1
        return self._build_obs(self.t), reward, self.t == self.end_idx, {
            "delta_abs_mean": float(np.abs(action_delta).mean()),
            "delta_abs_p95": float(np.percentile(np.abs(action_delta), 95)),
            "saturation_fraction": float((np.abs(action_delta) >= 0.18).mean()),
        }
```

`top_k` 默认 = 50. `lambda_logit` = 1.0.

**重要**: `action_delta` 是 PPO actor 头 `tanh(linear(x)) * 0.2` 之后的值. PPO sampling 在 logit 前.

---

## 5. PPO actor head 实现

`aurumq_rl.policy.residual_actor.ResidualActorHead`:

```python
import torch
import torch.nn as nn

class ResidualActorHead(nn.Module):
    def __init__(self, encoder_out_dim: int, action_range: float = 0.2):
        super().__init__()
        self.head = nn.Linear(encoder_out_dim, 1, bias=False)
        nn.init.zeros_(self.head.weight)             # ALGORITHM_SPEC §3 修正 #6
        self.action_range = action_range

    def forward(self, x):                             # x: (B, encoder_out_dim)
        delta_raw = self.head(x).squeeze(-1)          # (B,)
        return torch.tanh(delta_raw) * self.action_range
```

---

## 6. log_std schedule (修正 #2)

SB3 中 log_std 是 `ln(std)`. 不要传 `log_std_init=0.3`.

```python
import math
LOG_STD_INIT      = math.log(0.3)     # = -1.20397
LOG_STD_FINAL     = math.log(0.1)     # = -2.30259
FREEZE_STEPS      = 100_000
ANNEAL_STEPS      = 200_000
```

实现 (callback):

```python
class LogStdAnnealCallback(BaseCallback):
    def _on_step(self):
        step = self.num_timesteps
        if step < FREEZE_STEPS:
            target = LOG_STD_INIT
        else:
            progress = min((step - FREEZE_STEPS) / ANNEAL_STEPS, 1.0)
            target = LOG_STD_INIT + (LOG_STD_FINAL - LOG_STD_INIT) * progress
        with torch.no_grad():
            self.model.policy.log_std.fill_(target)
        return True
```

---

## 7. Smoke test gate (100k step, ALGORITHM_SPEC §6 现实版)

100k step 后 ledashi 跑 `scripts/eval_smoke.py --ckpt smoke_100k.zip` 检查:

| Check | Pass | Notes |
|---|---|---|
| reward sanity (再次确认) | std ∈ [0.02, 0.04] mean ∈ [-0.001, 0.001] | 防止训练中数据 corruption |
| KL 平均 | ≤ 0.05 (target_kl=0.03 + tolerance) | |
| clip_fraction | < 0.4 全程 | |
| Δ saturation | < 0.10 | 修正 #5 |
| log_std actual | 0.27-0.33 (前 100k frozen) | log_std exp |
| H1 PR_AUC (不校准) | ≥ baseline_H1_PR_AUC - 0.005 | smoke 不要求显著优 |
| **ep_rew_mean exactly 0 over 50k** | **kill 协议** | 实现 bug, 永久放弃 |

任一打勾失败 → ledashi 停掉, 反馈 paris 调 SPEC, 不许直接 300k.

100k smoke 全过 → resume 300k.

---

## 8. 300k 完成后

`scripts/eval_full.py --ckpt-dir runs/p3_residual` 输出:

```
H1 fit isotonic on (logit(p_baseline) + λ·Δ_mean_over_seeds)
H2 final metrics:
  - PR_AUC(p_final), ECE_10bin(p_final), top1pct_lift(p_final)
  - vs baseline_H2: ΔPR_AUC, ΔECE, Δtop1pct_lift
  - delta_saturation_fraction (训练全程平均)
  - log_std final value
```

通过 production gates (ALGORITHM_SPEC §6) → 上线 v3.

上传到 OSS:
```
oss://ledashi-oss-sgp/aurumq-rl/handoffs/2026-05-09-p3-4070-training/results/
  ppo_final.zip
  policy_state_dict.pt
  isotonic_h1.pkl
  ckpt_metrics.json
  training_log.jsonl
  RESULTS.md
```

paris 拉下来跑离线 inference 写 `wave_scores_daily` model_version='wave_t3_lgbm_v3.ppo_residual'，进 v2/v3 双写 + 7 天 dry-run。

---

## 9. AurumQ 主仓 (refs)

paris 已 push: https://github.com/yupoet/AurumQ commit `b94d9de`
- `src/aurumq/labeling/` (Method A_t3 实现 + universe + benchmark)
- `models/wave_lgbm_v2/anchor=2025-07/` (P2 ensemble, 5 seed)
- `handoffs/2026-05-09-{wave-label-ablation,p2-training-spec,p3-4070-training}/`

aurumq-rl 已 push: https://github.com/yupoet/aurumq-rl
- `src/aurumq_rl/labeling/` (P0 数学化主升浪 scanning, 已合)
- 还需要 paris push: `aurumq_rl/env/residual_ppo_env.py`, `aurumq_rl/policy/residual_actor.py`, `scripts/train_residual_ppo.py`, `scripts/reward_sanity_check.py` (本 PR)

---

## 10. 联系

paris 在 ECS 守着. 训练中遇到任何配置疑问, 直接看 ALGORITHM_SPEC §5 / §6, 或者 commit 一个 RFC 到 aurumq-rl.

100k smoke 完跑通就喊一声.
