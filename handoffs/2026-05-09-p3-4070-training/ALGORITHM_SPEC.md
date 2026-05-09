# P3 — 4070 PPO Residual Training Algorithm Spec (v2)

**Date**: 2026-05-09 (revised)
**For**: ledashi (4070)
**Source SPEC**: `handoffs/2026-05-09-p2-training-spec/SPEC.md` §5
**Audit (paris)**: 9 corrections incorporated below.

---

## 0. v2 修正清单 (vs v1)

| # | v1 错误 | v2 修正 |
|---|---|---|
| 1 | 估计 target_kl 让 timesteps 减 50k+ | 撤销, target_kl 只 early-stop epoch, 不减 step. 锁 5-7h for 300k |
| 2 | log_std_init=0.3 | 改 ln(std), 即 `log_std_init=-1.204` (std=0.3); anneal 到 `-2.303` (std=0.1) |
| 3 | ep_rew_mean 单调 > 0 | 改为 reward sanity + training sanity + H1 PR_AUC; 严格 kill = exactly 0 over 50k (bug) |
| 4 | p+Δ 当概率 | 改 logit 空间累加: `score = logit(p_baseline) + λ·Δ`, H1 isotonic 校准, λ=1.0 默认 |
| 5 | Δ ±0.5 | 砍到 ±0.2, 加 saturation 监控 (>5% 警示) |
| 6 | Δ 初始未约束 | actor head zero-init bias=False + 最后 Linear weight zero-init; deterministic eval @ iter 0 必须 \|Δ\|<1e-3 |
| 7 | realized return 契约空白 | OSS 加 realized_returns + market_returns 两 parquet, t+1 close-to-close 对齐 |
| 8 | OSS 体积 ~4G 估计 | 实测 7.3G, 上传 15-25min, 必须有 manifest |
| 9 | AurumQ 主仓 11 commits 没 push | 已 push (b94d9de) |

---

## 1. Architecture — Hybrid PPO Residual

```
score_raw[t, j] = logit(p_baseline[t, j]) + λ · Δscore_PPO[t, j]    (默认 λ=1.0)
p_final[t, j]   = isotonic_H1(score_raw[t, j])                       (校准恢复为概率)
```

PPO 输出对 baseline **logit** 的 residual Δ ∈ [-0.2, +0.2]. 
最终概率通过 H1 holdout fit 的 isotonic 校准，保证 ECE 可解释。

**为什么 logit 空间**: 原始概率 p ∈ [0, 1] 不可加 (会出 [-0.2, 1.2]); logit 空间 ∈ ℝ 自然支持加性 residual。

**为什么 isotonic on H1**: PPO 的 raw score 不天然校准, 需要后处理。H1 (2025-07..2025-09) 用作校准, H2 (2025-10..2025-12) 单次报告。

---

## 2. Reward — option β (realized excess return)

```python
r_t[j] = pct_chg_t_plus_1[t, j] - eq_weight_pct_chg_t_plus_1[t]
```

数据契约 (随 OSS bundle):

```
realized_returns.parquet  (2,180,517 行, in-universe only):
  trade_date, ts_code, close_t, close_t_plus_1, pct_chg_t_plus_1
  - close-to-close 对齐 (非 open-to-close)
  - 所有 in-universe (主板可交易非 ST) cells

market_returns.parquet  (727 dates):
  trade_date, eq_weight_pct_chg_t_plus_1, n_stocks
  - 当日 in-universe 平均 pct_chg_t_plus_1
```

**Sanity (实测)**:
- 个股 r: mean=0.00065 std=**0.02663** range=[-0.103, 0.106] zero_frac=0.034
- 市场 r: mean=0.00065 std=0.01361 range=[-0.089, 0.085] avg_n=2999

**符合修正 #3 sanity 标准** (std≈0.02-0.03 ✓, 非零率 96.6% ✓).

---

## 3. Action space

```python
delta_dim   = n_stocks_t          # 当日 universe 大小, ~3000
delta_range = (-0.2, 0.2)         # logit 空间, 非概率空间

# Saturation 监控 (训练全程):
saturation_fraction = (|delta| >= 0.18).mean()
# 预警: > 0.05  →  考虑放宽到 ±0.3
# 阻塞: > 0.20  →  Δ 头被打满, residual 退化, 算法失败
```

**Δ 初始 ≈ 0 强约束** (修正 #6):
```python
class ResidualActorHead(nn.Module):
    def __init__(self, encoder_out_dim):
        self.head = nn.Linear(encoder_out_dim, 1, bias=False)
        nn.init.zeros_(self.head.weight)            # zero-init weight
        # bias=False 已经避免 bias 漂移
    
    def forward(self, x):
        delta_raw = self.head(x).squeeze(-1)
        return torch.tanh(delta_raw) * 0.2          # ∈ [-0.2, 0.2]

# 训练前 self-test (强制):
with torch.no_grad():
    delta_test = policy.actor.head(sample_features)
    assert delta_test.abs().max() < 1e-3, "Δ head not zero-initialized!"
```

---

## 4. State / observation

每个 (t, j):

```python
obs = concat([
    panel_features[t, j, :],        # 345 features (schema_hash 5e71e158e331)
    p_baseline[t, j],               # P2 v2 ensemble 输出 ∈ [0, 1], 1 维
    rank_pct_baseline[t, j],        # baseline 当日截面 rank ∈ [0, 1], 1 维
])
# obs_dim = 347
```

policy 共享 encoder over stocks (PerStockEncoderPolicy, 同 26F-v3),
hidden=[128, 64] → encoder_out_dim=32 → ResidualActorHead.

---

## 5. PPO config (v2 锁定数值)

```python
PPO_CONFIG = {
    # network
    "policy_class": "PerStockEncoderPolicy",
    "encoder_hidden": [128, 64],
    "encoder_out_dim": 32,
    "actor_head": "ResidualActorHead",   # zero-init, tanh(...) * 0.2

    # optimization
    "learning_rate": {
        "schedule": "linear_decay",
        "init": 1e-4,
        "final": 1e-5,
        "total_timesteps": 300_000,
    },
    "n_epochs": 10,
    "batch_size": 4096,
    "n_envs": 16,
    "n_steps": 2048,
    "gamma": 0.95,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "target_kl": 0.03,             # SB3 early-stops 当前 PPO update epoch (不减 timesteps)
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,

    # entropy / log_std (修正 #2)
    "ent_coef": {
        "schedule": "linear_decay",
        "init": 0.01,
        "final": 0.0,
        "anneal_steps": 200_000,
    },
    "log_std_init": -1.204,        # ln(0.3) — 实际 std=0.3
    "log_std_schedule": {
        "freeze_until_step": 100_000,
        "anneal_to": -2.303,       # ln(0.1) — 实际 std=0.1
        "anneal_total_steps": 200_000,
    },

    # checkpoints
    "checkpoint_every_steps": 25_000,

    # evaluation (修正 #4)
    "ckpt_selection_window": ("2025-07-01", "2025-09-30"),  # H1 选 ckpt + fit isotonic
    "ckpt_report_window":    ("2025-10-01", "2025-12-31"),  # H2 单次报告
    "calibration": "isotonic_on_H1",
    "lambda_logit": 1.0,           # score = logit(p_baseline) + λ·Δ
}
```

**修正 #1 锁定**: 完整 300k timesteps, 不假设 target_kl 提前结束。

---

## 6. Validation gates (production 准入, 修正 #3)

PPO v3 上线 (写入 `wave_scores_daily` 作为 v3) 必须 ALL pass on H2:

| Gate | 要求 | 备注 |
|---|---|---|
| `PR_AUC(p_final on H2)` | ≥ `PR_AUC(p_baseline on H2)` + **0.005** | 至少 +0.5% improvement, **修正 #4 后 p_final 是校准概率** |
| `ECE_10bin(p_final)` | ≤ 0.025 | 校准不能崩 |
| `top1pct_lift(p_final)` | ≥ `top1pct_lift(p_baseline) × 1.05` | top-K 也得改进 |
| `delta_saturation_fraction` | ≤ 0.05 (训练全程) | Δ 不应该被 ±0.2 打满 |
| `train/std (after anneal)` | 应在 0.1 ± 0.05 范围 | log_std anneal 真实生效 |

**Smoke test gate (100k step)** — 比 production gate 宽:
- reward sanity passes (std≈0.02-0.03, mean ≈0)
- KL ≤ 0.05 (target 0.03 + tolerance)
- Δ saturation < 0.10
- H1 PR_AUC NOT significantly worse than baseline (绝对值 ≥ baseline - 0.005)
- 至少有一段 ep_rew_mean > 0 (不要求单调)

**Hard fail (kill 协议)**:
- `ep_rew_mean` exactly 0 over 50k steps (数据/实现 bug, 不是收敛慢)
- CUDA OOM 重试后仍崩
- crash 导致 ckpt 无法 resume

---

## 7. Pre-training reward sanity (强制, 修正 #3 + #7)

ledashi 训练前必须先运行 `scripts/reward_sanity_check.py`:

```python
# 1. Load realized_returns.parquet, market_returns.parquet
# 2. Compute r[t,j] = pct_chg_t_plus_1[t,j] - eq_weight_pct_chg_t_plus_1[t]
# 3. Assert:
assert abs(r.mean()) < 0.001, "r should be ~zero mean by construction"
assert 0.020 < r.std() < 0.030, "r std should be ~2-3% (A-share daily vol)"
assert (r != 0).mean() > 0.95, "non-zero reward fraction must > 95%"
assert r.min() > -0.15 and r.max() < 0.15, "outliers > 15% suspect"
```

**实测已通过** (此处生成数据时):
- mean=0.00065 ✓
- std=0.02663 ✓
- nonzero fraction=0.966 ✓
- range=[-0.103, 0.106] ✓

---

## 8. Resource budget (修正 #1 校准)

- **GPU**: 4070 12GB
- **300k step wall-clock**: **5-7 hours** (锁定, 不假设 target_kl 减时)
- **100k smoke**: 1.5-2.5 hours
- **n_envs=16 × n_steps=2048 = 32768 step/rollout**, target ~10 rollouts/min
- **磁盘**: ~5 GB panel + 12 ckpt × 100 MB = 1.2 GB
- **OOM 应对**: 减 n_envs to 8 or batch_size to 2048

---

## 9. Failure protocols (修正 #3)

```
Trigger                                | Action
---------------------------------------|--------------------------------------------------
ep_rew_mean exactly 0 over 50k step    | Kill. SPEC §5.3, 永久放弃 PPO. 数据/实现 bug.
H1 PR_AUC < baseline - 0.005           | Save 但不上线. RESULTS 解释.
Δ saturation > 0.20                    | Save 但不上线. Δ 头退化为 raw policy.
log_std 卡死不 anneal (>0.4 in late)   | Save 但不上线. 配置错误.
CUDA OOM                               | Drop n_envs to 8, retry from latest ckpt.
Crash mid-training                     | Resume from latest ckpt + continue.
```

---

## 10. Output spec

训练完成后, ledashi 上传到 OSS:

```
oss://ledashi-oss-sgp/aurumq-rl/handoffs/2026-05-09-p3-4070-training/results/
  ppo_final.zip                    # SB3 model
  policy_state_dict.pt             # 仅 actor weights
  isotonic_h1.pkl                  # H1 校准 (修正 #4)
  ckpt_metrics.json                # 每 25k 的 H1 PR_AUC + Δ saturation + log_std
  training_log.jsonl               # ep_rew_mean, KL, std, clip_fraction per iter
  RESULTS.md                       # H2 final + verdict
```

paris 拉下来跑离线 inference 写 wave_scores_daily v3, 进 v2/v3 双写 + 7 天 dry-run。
