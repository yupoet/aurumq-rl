# Phase 26F/G v3 — 方案 (codex round-1 fixes applied)

> 2026-05-08 (revised after codex review). 26F-v2 实测 PASS
> (median 2.15× vs 26C2 1.70×, +0.45×) 之后, 这是下一轮在 v3-final-patched
> panel 上验证 + 扩 encoder 的方案。

## Codex round-2 反馈处理 (2026-05-08 10:30)

| # | finding | status |
|---|---|---|
| 1 [HIGH] | verify_panels ST 检查静默降级, 默认 fail-open | ✅ 改 fail-closed: `fetch_st_or_delisted_codes()` 返回 `(codes, ok, err_msg)`, 默认 abort + rc=2; 仅 `--skip-st-check` 显式跳过. help 文案明确不再说 "name regex 兜底". 已 smoke test PG 不可达场景 → 真退 rc=2 |
| 2 [MED] | panel_dtype 没写进 metadata/training_summary | ✅ fp16 patch 扩到 27 行 (从 25 行), `metadata.json["panel_dtype"]` + `training_summary.json["panel_dtype"]` 都加; 复盘只看 metadata 就能知道 fp16/fp32 |
| 3 [MED] | runner failures > 0 时仍生成 scoreboard, 可能误推荐 | ✅ runner 末尾改: `failures > 0 → INCOMPLETE 提示 + exit failures, 不调 scoreboard`. 必须全部 9 个 cycle 都过才有推荐 |
| 4 [LOW] | `if ! cmd; then echo rc=$?` 打印的 rc 是 0 (是 `! cmd` 的 rc) | ✅ 改为先 `cmd; rc=$?; if [ "$rc" -ne 0 ]; then`. train/eval/IG 三处全改 |
| 5 [LOW] | build_combined_panel docstring 写 `--tier 26F` (实际 `26F-v3`) | ✅ docstring 改回 `26F-v3` + 加显式说明; 同时 build 脚本输出前加 universe assertion (anchored regex 自包含, 不依赖 _universe import) |
| 6 [LOW] | bundle 内有 __pycache__ 缓存 | ✅ 已清, OSS 重传时 manifest 8 文件干净 |

## Codex round-1 反馈处理 (2026-05-08 09:48)

| # | finding | status |
|---|---|---|
| 1 | `patch_train_v2_fp16.diff` 第二个 hunk 行号错乱, `git apply --check` corrupt | ✅ 已用真实 `git diff` 重新生成, `--check rc=0` + `apply rc=0` + 立即 revert 验证 |
| 2 | runner 用 ` \|\| { ... }` 削弱 `set -e`, 训练失败被吞 | ✅ 改为显式 `if ! cmd; then return 1; fi`, 加每步 sanity check + 退出码透传 |
| 3 | runbook 写 `--tier 26C2/26F`, builder 只接受 `26C2-v3/26F-v3/26G-v3` | ✅ runbook + runner + scoreboard 全部统一 `-v3` 后缀, runner 加未知 tier 早退检查 |
| 4 | universe 散在多处, ST 检查只是名字 regex | ✅ 抽 `scripts/_universe.py` 单源 (anchored regex + ST 名 + assert helper), `filter_panel_main_board` + `verify_panels` 都 import; verify 加 PG `stock_info` JOIN 真做 ST 排除, 不可达时 `--skip-st-check` 显式降级 |
| 5 | `PHASE26EF_RESULTS.md` 26F seed 归属错: seed43=2.722 (实际是 seed44 写错), seed44=2.154 写成 1.587 | ✅ 已知错误, **不影响 median/max 决策**。次方案的 baseline 数仍用 median 2.15× / max 2.72×. 真正影响是 IG 必须以 seed=42 或当前 best=seed43 重跑 |
| 6 | 26G abandoned 与 orchestrator log 三 seed DONE 自相矛盾, 包内无 26G run dir | ⚠️ 不基于此包做 26G 容量结论。**本提案中的 26G-v3 是从零跑**, 不依赖任何旧 26G 数据 |

新增的 known-good guardrails (写在每个 component 注释里):
- `metadata.json` / `training_summary.json` 必须记录 `panel_dtype` (fp32/fp16) 字段, 方便复盘
- 任何新 panel-build / verify / RL-side script 都从 `_universe` import, 禁止 inline `code.startswith("60")`

## 主决策

1. **不再做 26E (curated continuous tech only)**: 已确认 neutral, 浪费算力。
2. **不再做 path B (26F seeds 45/46/47 on v2 panel)**: panel 已升级到 v3-final-patched, 在旧 panel 上扩 seed 没有上行。
3. **跑 3 tier × 3 seed = 9 runs** on v3-final-patched main-board panel:
   - **26C2-v3**: 353 cols, encoder 128→64→32 (新 baseline replication)
   - **26F-v3**: 361 cols (= 23A 353 + 2 curated tech + 6 events_decay10), encoder 128→64→32
   - **26G-v3**: 361 cols same as 26F-v3, encoder **192→96→48** (mid encoder bump, 4070-friendly)
4. **fp16 panel-on-CUDA** for VRAM headroom: ~1.5 GB saved, unblocks 26G on 4070.

## Universe — 硬性约束 (写入 CLAUDE.md)

**仅主板 A 股**:
- ✅ `60[0135]\d{3}.SH` + `00[0123]\d{3}.SZ`
- ❌ 30* 创业板 / 688* 689* 科创板 / *.BJ 北交所
- ❌ ST / *ST / 退 / delisted

`UniverseFilter.MAIN_BOARD_NON_ST` (RL 侧 default) 已经是这套规则。
panel 在数据侧也已预过滤 (`scripts/filter_panel_main_board.py`).

理由: 涨跌停规则一致 (±10%)、T+1 一致、披露频率一致、流动性结构同源。
其他板块差异 (30*/688* ±20%, BJ ±30%) 引入 regime confounding, 必须分开训练。

**Codex review 点**: 这条约束是否需要在更多入口强制 (build_combined_panel?
phase20_holdout?)。当前每个 caller 各自传 universe_filter 参数, 无中央 enforcement.

## 实验矩阵

| tier | factors | encoder | encoder_out_dim | n_envs | n_steps | batch | LR | timesteps | panel_dtype |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| 26C2-v3 | 353 | 128,64 | 32 | 16 | 128 | 512 | 1e-4 const | 300k | fp32 |
| 26F-v3 | 361 | 128,64 | 32 | 16 | 128 | 512 | 1e-4 const | 300k | fp32 |
| **26G-v3** | 361 | **192,96** | **48** | 16 | 128 | 512 | 1e-4 const | 300k | **fp16** |

3 seeds (42/43/44) per tier = 9 train+eval cycles. ~3-4 GPU-h on RTX 4070.

(可选) 收紧置信区间: 加 seeds 45/46/47 → 6 seeds = 6-8 GPU-h.

## 26G — encoder 192→96→48 数学合理性

- 128→64 (current): 128*64 + 64*32 = ~10.4k weights/biases (per-stock encoder)
- **192→96 (new)**: 192*96 + 96*48 = ~23.2k = **~2.2× params**
- 256→128 (ledashi failed): 256*128 + 128*64 = ~41k = **~4× params** (4070 OOM)

192→96 是**保守 bump**, 理论 fps 比 256→128 快 ~2×, VRAM ~50%. 加 fp16 panel
释放的 1.5 GB, 4070 12 GB 应该有 3-4 GB 余量。

`encoder_out_dim=48` (vs 32) 让 actor/value head 看到更宽 feature space,
pooled_dim 也从 64 (=2*32) → 96 (=2*48), 价值估计可塑性更强。

## fp16 Panel-on-CUDA 设计 (Codex 重点 review)

### 现状

`scripts/train_v2.py:307`:
```python
panel_t = torch.from_numpy(panel.factor_array).to("cuda")  # fp32 by default
```

`gpu_env.py:23`: `panel: torch.Tensor (T, S, F) fp32 cuda`

主板 panel 形状 (~720 dates × ~3000 stocks × 361 factors):
- fp32: **3.12 GB** on cuda
- fp16: **1.56 GB** on cuda → 节省 1.56 GB

### 改动

**`scripts/train_v2.py`** (添加 1 个 CLI flag + 1 行 cast):
```python
parser.add_argument(
    "--panel-dtype",
    choices=["fp32", "fp16"],
    default="fp32",
    help="cuda dtype for the (T,S,F) panel tensor. fp16 saves ~50% VRAM.",
)
...
panel_t = torch.from_numpy(panel.factor_array).to("cuda")
if args.panel_dtype == "fp16":
    panel_t = panel_t.half()
```

**`src/aurumq_rl/feature_extractor.py`** (1 行 — cast obs back to fp32 before MLP):
```python
def forward(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
    # Phase 26G: panel may be stored as fp16 to save VRAM. MLP / LayerNorm
    # weights are fp32, so cast obs back to fp32 here. Numerical impact:
    # ~3-4 mantissa bits lost on factor values, equivalent to z-score noise
    # at the 5th decimal — negligible vs cross-section z-score's own noise.
    obs = obs.float()
    b, s, f = obs.shape
    ...
```

(原 forward 后续不变。)

### Codex 审查点

1. **是否影响 numerical convergence?** PPO gradient flow stays fp32 (model
   weights, optimizer state, returns/advantages 都是 fp32). 只有 panel 存储
   被 fp16 量化。fp16 mantissa 10 位, 对 [-3, +3] z-score 范围的因子值精度
   ~1e-3 — 远小于 cross-section z-score 自身随机性。
2. **gpu_env.py 需要改吗?** 不用 — `self.panel[self.t]` slice 出来的 obs
   保持 fp16, traverse VecEnv 后到 policy.forward, 在 feature_extractor 里
   被 cast 回 fp32。VecEnv 内部 cast 一次, observation_space dtype 仍 fp32
   (我们告诉 sb3 obs 是 fp32, 实际传 fp16 SB3 不检查 dtype)。
3. **训练曲线对照**: 建议 26G 同时跑一组 fp32 (panel_dtype=fp32) 看 curve
   有无 divergence。如果 26G fp16 跟 26F fp32 训练曲线 (training_summary
   里的 ep_rew_mean) 在前 50k steps 差异 > 5%, 说明 fp16 有问题。
4. **回退路径**: `--panel-dtype fp32` (default) 完全等价于现状, 一键回退。

## Pass/fail rubric (rebaselined)

26F-v2 已经把"baseline"定到 26C2-v2 median 1.70× (3 seeds)。这次:

| tier | criterion |
|---|---|
| **26C2-v3 sanity** | best lift ≥ 1.85× — 必须复现 v2 baseline |
| **26F-v3** | median > 26C2-v3 + 0.20× **AND** median > 26F-v2 (2.15×) — 数据修复 + events 双增益 |
| **26G-v3** | median > 26F-v3 + 0.10× — encoder bump 解锁更多容量 |

**任一 v3 tier median 持平或退化 → 留 v2 panel 作 production**, formula
patches 仅作"防御性零信号损失"; v3 数据本身可能在某些 corner 引入新 noise.

## 风险 & mitigations

| 风险 | mitigation |
|---|---|
| 26G fp16 数值不稳 | 同步跑一个 fp32 对照 seed=42, 用 training_metrics.jsonl 对比 |
| 26G OOM (192→96) | 先用 batch=384 试一次, 不行降到 256 |
| v3 panel 和 v2 panel 行集差异 | verify_panels.py 严格对比 sha256 + ts_code 集合 |
| 6 seeds 跑不完 | 优先级: 26C2/26F 各 3 seed → 26G 至少 1 seed → 余下 seeds |
| sanitizer 找回失败 | 26C2-v3 best < 1.85×, abort, 保留 v2 |

## 文件清单

```
oss://ledashi-oss-sgp/aurumq-rl/handoffs/2026-05-08-phase26fg-v3/
├── PROPOSAL_PHASE26FG_V3.md            (this file — for codex review)
├── run_phase26fg_v3_overnight.sh       — 12 train + eval cycles
├── configs/
│   └── extras_26f_v3_8cols.txt         — 2 curated tech + 6 events_decay10
├── tools/
│   ├── verify_panels.py                — main_board hard assertion
│   └── generate_include_files.py       — 26C2/F/G include lists from 23A 353
├── scripts/
│   ├── build_combined_panel_phase26fg_v3.py
│   ├── phase26fg_v3_scoreboard.py
│   └── patch_train_v2_fp16.diff        — codex审完后由 ledashi apply
└── (panels NOT in this bundle — pull from 2026-05-08-panels-main-board-v3)
```

## 提交后续

- [x] codex round-1 review → 6 反馈全部已修 (见上方表)
- [ ] codex round-2 review (重发的 patch + runner + universe helper)
- [ ] ledashi review fp16 patch diff (~25 行 +) → apply 或拒绝
- [ ] paris 已上传 26F-v3-overnight bundle (round-1) → 重发 round-2
- [ ] ledashi 拉取 + verify_panels + 跑 9 个 cycle → 上传 results

## Round-3 文件 SHA / 验证

```
patch_train_v2_fp16.diff:      git apply --check rc=0 (real diff against AurumQ-RL @ 5e92594)
                                +metadata.json + training_summary.json now record panel_dtype
run_phase26fg_v3_overnight.sh: bash -n OK; unknown-tier exit 2; rc capture honest;
                                failures>0 → no scoreboard, exit failures
scripts/_universe.py:          16/16 unit case PASS (incl. 60[02], 30*, 688*, BJ rejections)
tools/verify_panels.py:        PG stock_info JOIN fail-closed; --skip-st-check explicit;
                                smoke-tested PG-unreachable → rc=2 + clear message
build_combined_panel_phase26fg_v3.py: pre-write universe assertion (anchored regex)
bundle:                        no __pycache__ / *.pyc; 8 hand-written files
```
