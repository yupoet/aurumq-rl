# Phase 10 架构修复指令（审计修复版）

> Phase 9 在当前仓库已用于 `IndexOnlyRolloutBuffer + n_steps=1024`，本架构修复文档改名为 **Phase 10**，避免和既有训练史冲突。
> 目标：不改 environment / reward / 因子集合 / PPO 超参，修复 `PerStockEncoderPolicy` 的结构问题，并先验证训练稳定性，再谈 OOS 提升。

## 审计结论

原文方向基本成立，但有四个必须修正的问题：

1. **当前 `policy._build()` 可能存在 optimizer 漏参风险**

   `super()._build()` 会创建 optimizer；当前代码随后替换 `action_net / value_net / log_std`。必须在替换后重建 optimizer，或完全接管 `_build()`。

2. **P1 原代码有数学 bug**

   `encoded = encoded - encoded.mean(dim=1)` 之后再做 `encoded.mean(dim=1)`，结果恒接近 0。Value head 的 mean 分支应使用 **center 前的 market mean**。

3. **不能上全量 cross-stock self-attention**

   当前真实股票数约 `3014`，不是文档示例里的 `300`。全量 MHA 是 `O(S^2)`，训练 batch 下极易 OOM。可选 P2 应改成 induced / low-rank attention。

4. **P3 不应直接上未初始化的 state-dependent log_std**

   若做 state-dependent std，必须零初始化 weight、bias 初始化到 `-0.69`，否则初始探索尺度会漂移。更低风险方案是先把固定 `(n_stocks,)` log_std 改成 shared scalar，恢复随机策略的置换等变性。

## 优先级

| 优先级 | 改动 | 文件 | 必要性 |
|---|---|---|---|
| P0 | 修复 custom head optimizer 注册 | `policy.py` + tests | 必做 |
| P1 | LayerNorm + cross-section centering + 正确 dual pooling | `feature_extractor.py` + `policy.py` + tests | 必做 |
| P2 | induced cross-stock attention，禁止 full MHA | `feature_extractor.py` | 可选 |
| P3 | log_std 改造：先 shared scalar，再 state-dependent | `policy.py` | 可选 |

## P0：修复 optimizer 漏参风险

当前模式是：

```python
super()._build(lr_schedule)
self.action_net = ...
self.value_net = ...
self.log_std = ...
```

必须在替换完 custom heads 后重建 optimizer：

```python
def _build(self, lr_schedule) -> None:
    super()._build(lr_schedule)

    n_stocks = self.action_space.shape[0]
    self.action_net = nn.Linear(self._encoder_out_dim, 1)

    pooled_dim = 2 * self._encoder_out_dim
    layers: list[nn.Module] = []
    prev = pooled_dim
    for h in self._value_hidden:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        prev = h
    layers.append(nn.Linear(prev, 1))
    self.value_net = nn.Sequential(*layers)

    self.action_dist = DiagGaussianDistribution(n_stocks)
    self.log_std = nn.Parameter(torch.full((n_stocks,), -0.69, dtype=torch.float32))

    if self.ortho_init:
        self.action_net.apply(partial(self.init_weights, gain=0.01))
        self.value_net.apply(partial(self.init_weights, gain=1.0))

    self.optimizer = self.optimizer_class(
        self.parameters(),
        lr=lr_schedule(1),
        **self.optimizer_kwargs,
    )
```

测试必须新增：

```python
def test_policy_optimizer_tracks_all_trainable_params():
    policy = _make_policy()
    opt_ids = {id(p) for g in policy.optimizer.param_groups for p in g["params"]}
    missing = [
        name for name, p in policy.named_parameters()
        if p.requires_grad and id(p) not in opt_ids
    ]
    assert missing == []
```

## P1：LayerNorm + 中心化 + 正确 dual pooling

`feature_extractor.py` 应保留两个版本的 embedding：

- `normed`：LayerNorm 后、center 前，用于 value 的 market baseline。
- `centered`：cross-section centered 后，用于 actor ranking。

```python
class PerStockExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        hidden: tuple[int, ...] = (128, 64),
        out_dim: int = 32,
    ):
        n_stocks, n_factors = observation_space.shape
        super().__init__(observation_space, features_dim=out_dim)

        self.n_stocks = n_stocks
        self.n_factors = n_factors
        self.out_dim = out_dim
        self.pooled_dim = 2 * out_dim

        layers: list[nn.Module] = []
        prev = n_factors
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, out_dim))

        self.mlp = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        b, s, f = obs.shape
        flat = obs.reshape(b * s, f)

        normed = self.norm(self.mlp(flat).reshape(b, s, self.out_dim))

        market_mean = normed.mean(dim=1)
        centered = normed - market_mean.unsqueeze(1)

        opportunity_max = centered.max(dim=1).values
        pooled = torch.cat([market_mean, opportunity_max], dim=-1)

        return {"per_stock": centered, "pooled": pooled}
```

`policy.py` 的 value head 输入必须是 `2 * encoder_out_dim`：

```python
pooled_dim = 2 * self._encoder_out_dim
```

不要使用 center 后的 mean 作为 value 输入，它恒接近 0。

## P1 验收

```python
out = extractor(obs)

assert out["per_stock"].shape == (batch_size, n_stocks, 32)
assert out["pooled"].shape == (batch_size, 64)
assert out["per_stock"].mean(dim=1).abs().max() < 1e-5
assert torch.isfinite(out["per_stock"]).all()
assert torch.isfinite(out["pooled"]).all()
```

策略测试：

```python
assert policy.value_net[0].in_features == 2 * policy._encoder_out_dim
```

命令：

```bash
pytest tests/test_policy.py -v
```

## P2：可选 cross-stock attention，但禁止 full MHA

不要这样做：

```python
nn.MultiheadAttention(embed_dim=out_dim, num_heads=4, batch_first=True)
```

对 `S≈3014` 的全量股票序列，`O(S^2)` attention 在 PPO batch 下不可控。

若 P1 后仍需要更强 cross-stock 归纳偏置，使用 induced attention：

```python
self.inducing = nn.Parameter(torch.randn(1, n_inducing, out_dim) * 0.02)
self.pool_attn = nn.MultiheadAttention(out_dim, n_attn_heads, batch_first=True)
self.broadcast_attn = nn.MultiheadAttention(out_dim, n_attn_heads, batch_first=True)
self.attn_norm = nn.LayerNorm(out_dim)
```

forward 顺序：

```python
normed = self.norm(self.mlp(flat).reshape(b, s, self.out_dim))

if self.use_attention:
    seeds = self.inducing.expand(b, -1, -1)
    prototypes, _ = self.pool_attn(seeds, normed, normed, need_weights=False)
    ctx, _ = self.broadcast_attn(normed, prototypes, prototypes, need_weights=False)
    normed = self.attn_norm(normed + ctx)

market_mean = normed.mean(dim=1)
centered = normed - market_mean.unsqueeze(1)
opportunity_max = centered.max(dim=1).values
pooled = torch.cat([market_mean, opportunity_max], dim=-1)
```

建议默认：

```python
n_inducing = 16
n_attn_heads = 4
use_attention = False
```

只有 P1 的 200k 结果接近但未过线时再打开。

## P3：log_std 改造

最低风险版本：把 `(n_stocks,)` 固定参数改成 shared scalar：

```python
self.log_std = nn.Parameter(torch.tensor(-0.69, dtype=torch.float32))
```

使用时：

```python
log_std = self.log_std.expand_as(scores)
distribution = self.action_dist.proba_distribution(scores, log_std)
```

如果要做 state-dependent log_std，必须初始化为接近旧行为：

```python
self.log_std_net = nn.Linear(self._encoder_out_dim, 1)
nn.init.zeros_(self.log_std_net.weight)
nn.init.constant_(self.log_std_net.bias, -0.69)
```

forward / evaluate_actions / _predict 中统一：

```python
log_std = self.log_std_net(feats["per_stock"]).squeeze(-1)
log_std = log_std.clamp(-2.0, 0.5)
distribution = self.action_dist.proba_distribution(scores, log_std)
```

验收：

- 初始 `log_std.mean()` 接近 `-0.69`
- 训练中 `log_std` 不贴 `-2.0` 或 `0.5` 边界
- stochastic policy 的 permutation equivariance 需要新增测试

## 训练验证

使用当前 Phase 9 的 index buffer 配置作为基线，不再改超参：

```bash
python scripts/train_v2.py \
  --total-timesteps 200000 \
  --data-path data/factor_panel_combined_short_2023_2026.parquet \
  --start-date 2023-01-03 --end-date 2025-06-30 \
  --universe-filter main_board_non_st \
  --rollout-buffer index \
  --n-envs 16 --n-steps 1024 --batch-size 1024 \
  --n-epochs 10 --learning-rate 1e-4 \
  --target-kl 0.30 --max-grad-norm 0.5 \
  --out-dir runs/phase10_p1_200k
```

判定：

| 200k OOS top30 Sharpe | 处理 |
|---|---|
| `>= 0` | 先跑 1M，再考虑 5M |
| `-0.5 ~ 0` | 方向有效，可继续 1M 或试 P2 |
| `< -0.7` | 停止架构叠加，排查数据、reward、backtest、action clipping |

50k 结果只能看稳定性，不用于判断最终收益。

## 不要做

1. 不要只做 center，不修 value pooling。
2. 不要把 center 后的 mean 接给 value head。
3. 不要上 full cross-stock self-attention。
4. 不要同时改 P1、P2、P3。
5. 不要在这个阶段改 environment、reward、factor set、PPO 超参。
6. 不要忽略 optimizer 参数注册测试。

## Commit 顺序

```text
fix(policy): rebuild optimizer after custom per-stock heads
feat(policy): add encoder norm and critic dual pooling
feat(policy): add induced cross-stock attention
feat(policy): make exploration std permutation-equivariant
```

## 审计依据

- 当前实现：`src/aurumq_rl/policy.py`、`src/aurumq_rl/feature_extractor.py`
- 当前 Phase 9 记录：`docs/TRAINING_HISTORY.md`
- SB3 `ActorCriticPolicy._build()` 会在构建网络后创建 optimizer：<https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/policies.html>
- SB3 `DiagGaussianDistribution.proba_distribution()` 对 `log_std` 做 broadcast：<https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/distributions.html>
