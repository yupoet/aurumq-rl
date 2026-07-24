# AGENTS.md — AurumQ-RL

Lean project rules for AI agents. **Deep policy, universe lock, research protocol → [`CLAUDE.md`](CLAUDE.md).** Architecture narrative → [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## What this is / isn't

**Is:** Open-source RL stock-selection reference for China A-shares (day frequency, T+1). Gymnasium envs + SB3 training + ONNX CPU inference. Multi-source factor **consumer** via Parquet column prefixes.

**Isn't:** Live trading / broker APIs, data-vendor SDKs or API keys, high-frequency trading.

**Factor modules:** `src/aurumq_rl/factors/` ships Alpha101 + GTJA191 as optional library code. The RL training/inference path still treats factors as **input columns** on a Parquet panel — do not hard-wire factor semantics into env/policy.

## Layout

| Path | Role |
|---|---|
| `src/aurumq_rl/` | Core package: `data_loader`, `env`, `policy`, `inference`, `onnx_export`, rewards, limits |
| `src/aurumq_rl/factors/` | Alpha101 + GTJA191 implementations + registry |
| `src/aurumq_rl/labeling/` | Event / barrier / trend labels |
| `src/aurumq_rl/p3/` | P3 SL glue (Kronos heads, rank-z, residual policy) |
| `scripts/` | Production CLIs: `train.py`, `infer.py`, `export_factor_panel.py`, … |
| `scripts/p3/` | Research matrix / ensemble (lint-**excluded**) |
| `scripts/_*.py` | One-shot diagnostics (lint-excluded) |
| `tests/` | pytest suite |
| `docs/` | SCHEMA, TRAINING, INFERENCE, UNIVERSES, factor library |
| `web/` | Next.js run dashboard — see `web/AGENTS.md` |
| `data/synthetic_demo.parquet` | Synthetic demo only (never real tickers) |

Package import name: **`aurumq_rl`**. Repo / PyPI: **`aurumq-rl`**.

## Commands

```bash
# install (CPU-friendly core + dev tools)
pip install -e ".[dev]"
# optional extras
pip install -e ".[train]"     # torch, SB3, gymnasium — GPU host only for real training
pip install -e ".[factors]"   # PG/pandas for export scripts

pytest tests/ -v --tb=short
pytest tests/ -v -k smoke
ruff check src/ tests/ scripts/
ruff format --check src/ tests/ scripts/
python scripts/train.py --smoke-test --out-dir /tmp/smoke

# after touching hot paths (env.step / load_panel / inference.predict):
pytest tests/test_perf.py --benchmark-only
```

CI (`.github/workflows/ci.yml`): ruff + pytest with cov on 3.10/3.11/3.12, then wheel build.

## Hard red lines

1. **No training on small ECS (e.g. 8C14G).** Train only on GPU hosts. Infer on CPU via onnxruntime.
2. **No commercial data-vendor endpoints, tokens, or key formats** in README, comments, or error strings. Neutral wording only ("public market data export").
3. **No real A-share codes** in demos/tests/fixtures — use `SYN001`… style names. Never commit real market panels.
4. **No secrets**, passwords, or private internal URLs.
5. **Never `import aurumq.*`** — this repo is standalone; all deps live under `aurumq_rl`.
6. **`env.py` / `inference.py` must import without gymnasium/torch** (graceful placeholders that raise on use is OK).
7. **Universe defaults:** prefer membership parquet when present; fallback main-board regex. Locked universes and train window rules → `CLAUDE.md`.
8. **User-facing entrypoints** (`scripts/train.py`, `scripts/infer.py`, README) must keep the education/research disclaimer.

## Coding conventions

- Python **3.10+**. Public APIs: type hints + docstrings. Actionable error messages.
- **ruff** line-length 100 (`pyproject.toml`). Conventional commits: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`.
- Branches: `feat/…`, `fix/…` off `main`.
- **Git remote / push (locked 2026-07-25):** origin = **SSH** `git@github.com:yupoet/aurumq-rl.git` (same `~/.ssh/id_ed25519` as sibling repos). **Do not use `gh` or HTTPS+gh credential helper for push.** If remote is still `https://github.com/...`, run `git remote set-url origin git@github.com:yupoet/aurumq-rl.git` then `git push`. Details → `CLAUDE.md` §Git 工作流.
- New production code: aim **≥80%** coverage; pair factor modules with tests under `tests/factors/`.
- Core package stays inference-friendly: do not pull torch/SB3 into default import path.
- NaN strategy: skip in z-score; fill 0 for training tensors unless a module documents otherwise.
- A-share session times: **`Asia/Shanghai`**. Prefer UTC elsewhere with explicit tz.
- Research scripts in `scripts/p3/` / `scripts/_*.py` are not production — promote into `src/` only with tests and stable APIs.

## Data contract (summary)

Parquet in → train. Required columns:

- `ts_code` (str, `XXXXXX.SH/SZ/BJ` style)
- `trade_date` (date)
- `close`, `pct_chg` (**decimal**: +10% = `0.10`), `vol` (`0` = suspended)
- ≥1 factor prefix group: `alpha_`, `mf_`, `hm_`, `hk_`, `inst_`, `mg_`, `cyq_`, `senti_`, `sh_`, `fund_`, `ind_`, `mkt_`, `gtja_`, …

Optional: `is_st`, `days_since_ipo`, `industry_code`, `is_hs300`, `is_zz500`.

Prefix discovery is automatic in `data_loader.py` — do not rename prefixes. Full tables → `CLAUDE.md`, `docs/SCHEMA.md`, `docs/FACTORS.md`. Universes → `docs/UNIVERSES.md`.

## Research stream (pointer)

Two paradigms (label **before** designing a matrix cell):

1. Predictive cross-sectional (`features(t) → forward return labels`)
2. Event-anchored pattern recognition

After matrix runs: update README §12 progress/conclusions per `CLAUDE.md`. Strong vs weak statistical claims must be distinguished (sample size / IC SE). Public writeups: relative deltas only — no internal panel names, OSS paths, or vendor details.

## Current work context

Open issue-style feature branches (local): `#2` DSR reward, `#5` purged CV, `#6` eval metrics, `#7` p3 heads, `#8` labeling, `#9` factor engineering, `#10` incremental factors. Confirm with `git branch` before assuming merge state.

## When unsure

1. `CLAUDE.md` for red lines and research protocol  
2. `docs/ARCHITECTURE.md` for data flow  
3. Matching tests under `tests/` for intended behavior  
4. Do not invent data sources, real tickers, or broker integrations  
