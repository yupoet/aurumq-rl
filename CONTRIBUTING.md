# Contributing to AurumQ-RL

Thanks for your interest in contributing! This document outlines how to get involved.

## Development setup

```bash
git clone https://github.com/yupoet/aurumq-rl.git
cd aurumq-rl
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,train]"
pre-commit install      # optional
```

## Workflow

1. **Open an issue first** for non-trivial changes — saves rework.
2. Fork the repo and create a feature branch: `git checkout -b feat/your-thing`.
3. Make changes, add tests, ensure CI is green locally.
4. Open a PR against `main`.

## Code style

- **Python 3.10+**, type hints required on public APIs.
- **`ruff`** for lint + format. Config in `pyproject.toml`.
- **`pytest`** for tests. Aim for ≥80% line coverage on new code.
- No PyTorch / SB3 imports outside the `[train]` optional dependency boundary —
  the core package must remain inference-only.

## Running tests

```bash
pytest tests/ -v --tb=short                    # full suite
pytest tests/ -v -k smoke                      # smoke only
pytest tests/ -v --cov=src/aurumq_rl           # with coverage
```

## What to contribute

**Easy first issues**:
- Documentation improvements
- Test coverage for edge cases (NaN handling, boundary dates)
- Additional reward function variants (e.g. CVaR, Calmar)
- Bug reports with reproducers

**More involved**:
- Additional RL algorithms (TD3, DQN variants)
- Multi-asset extension (HK / US markets)
- Onnx export for non-MLP policies (LSTM, Transformer)
- Visualization tools for training curves and factor attribution

## Out of scope

- **Factor computation** — by design, this project consumes pre-computed factor
  columns. Factor pipelines belong in your own data warehouse.
- **Live trading / brokerage integration** — this is a research / backtest
  framework. Live execution is your responsibility and risk.
- **Specific commercial data APIs** — `aurumq-rl` is data-source agnostic.
  Don't add hard dependencies on any specific vendor.

## Code of conduct

Be kind. Disagree on ideas, not on people. No harassment, no spam, no political
flame wars. Project maintainers reserve the right to remove disruptive content.

## License

By contributing, you agree your contributions are MIT-licensed.

## Questions?

Open a [GitHub Discussion](https://github.com/yupoet/aurumq-rl/discussions) or
ping the maintainer.
