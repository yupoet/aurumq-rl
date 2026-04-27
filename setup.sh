#!/usr/bin/env bash
# AurumQ-RL Setup Script
#
# Quick installer for development environment.
# Usage:
#     bash setup.sh             # Core only (CPU inference)
#     bash setup.sh --train     # + GPU training deps
#     bash setup.sh --all       # + factor computation + dev tools

set -euo pipefail

PYTHON="${PYTHON:-python3}"
EXTRAS=""

for arg in "$@"; do
    case "$arg" in
        --train)
            EXTRAS="${EXTRAS},train"
            ;;
        --factors)
            EXTRAS="${EXTRAS},factors"
            ;;
        --dev)
            EXTRAS="${EXTRAS},dev"
            ;;
        --all)
            EXTRAS="${EXTRAS},train,factors,dev"
            ;;
        -h|--help)
            cat <<EOF
AurumQ-RL Setup

Usage: bash setup.sh [options]

Options:
  (no args)      Install core deps only (CPU inference, ~50MB)
  --train        Add GPU training deps (PyTorch + SB3, ~3GB)
  --factors      Add factor computation deps (PG / pandas / scipy)
  --dev          Add dev tools (pytest / ruff / mypy)
  --all          All of the above
  -h, --help     Show this help

Examples:
  bash setup.sh                    # Smoke test only
  bash setup.sh --train            # Train on GPU
  bash setup.sh --factors          # Export factor panel from your DB
  bash setup.sh --all              # Full development setup
EOF
            exit 0
            ;;
    esac
done

EXTRAS="${EXTRAS#,}"  # strip leading comma

echo "==> Checking Python version..."
PY_VER=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "    Found Python $PY_VER"
if ! "$PYTHON" -c "import sys; assert sys.version_info >= (3, 10)" 2>/dev/null; then
    echo "ERROR: Python 3.10+ required, got $PY_VER"
    exit 1
fi

if [[ ! -d ".venv" ]]; then
    echo "==> Creating virtual environment..."
    "$PYTHON" -m venv .venv
fi

echo "==> Activating venv..."
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Upgrading pip / wheel..."
pip install -q --upgrade pip wheel

if [[ -z "$EXTRAS" ]]; then
    echo "==> Installing core dependencies..."
    pip install -e .
else
    echo "==> Installing with extras: [$EXTRAS]"
    pip install -e ".[$EXTRAS]"
fi

if [[ ! -f ".env" ]]; then
    echo "==> Creating .env from .env.example..."
    cp .env.example .env
    echo "    Edit .env to set your DB URL and other config"
fi

echo ""
echo "==> Setup complete!"
echo ""
echo "Next steps:"
echo "  source .venv/bin/activate"
echo "  python scripts/train.py --smoke-test --out-dir /tmp/smoke"
echo ""
