# Repository Guidelines

## Project Structure & Module Organization

AurumQ-RL is a Python 3.10+ package with source code in `src/aurumq_rl/`. Core modules cover data loading, Gymnasium-style environments, reward functions, backtesting, metrics, ONNX inference, and factor registries. CLI entry points live in `scripts/`, examples in `examples/`, and documentation in `docs/`. Tests are under `tests/`, with factor-specific coverage in `tests/factors/`. Demo data and schema notes are in `data/`. The dashboard is a separate Next.js app in `web/`; follow `web/AGENTS.md` before editing that subtree.

## Build, Test, and Development Commands

- `python3 -m venv .venv && source .venv/bin/activate`: create a local virtual environment.
- `pip install -e ".[dev]"`: install core package plus test, lint, and type-check tools.
- `pip install -e ".[dev,train]"`: add training dependencies for GPU-capable environments.
- `pytest tests/ -v --tb=short`: run the full Python test suite.
- `pytest tests/ -v -k smoke`: run the CPU-safe smoke path.
- `ruff check src tests scripts` and `ruff format src tests scripts`: lint and format Python code.
- `bash scripts/web_dashboard.sh`: install and run the local dashboard at `http://localhost:3000`.

## Coding Style & Naming Conventions

Use 4-space indentation, type hints on public APIs, and docstrings for public functions/classes. Ruff is configured in `pyproject.toml` with a 100-character line length and Python 3.10 target. Keep the import package name `aurumq_rl`; do not introduce `aurumq.*` imports. Factor columns are discovered by prefixes such as `alpha_*`, `mf_*`, `hm_*`, and `gtja_*`; preserve these contracts when changing loaders or docs.

## Testing Guidelines

Tests use `pytest` with files named `test_*.py` and functions named `test_*`. Use markers already defined in `pyproject.toml`, including `smoke`, `factors`, `env`, and `slow`. Add focused tests for new behavior and aim for at least 80% coverage on new code: `pytest tests/ -v --cov=src/aurumq_rl`.

## Commit & Pull Request Guidelines

Recent history uses conventional commits, for example `feat(data): ...`, `fix(backtest): ...`, and `chore: ...`. Prefer branches like `feat/your-change`. PRs should describe intent, list tests run, link relevant issues, and include screenshots for dashboard UI changes.

## Security & Configuration Tips

Do not commit API keys, tokens, internal URLs, commercial data-source details, or real A-share datasets. `data/synthetic_demo.parquet` must remain synthetic. Keep core inference dependencies lightweight; PyTorch, Stable-Baselines3, Gymnasium, and W&B belong behind optional training extras.
