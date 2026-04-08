#!/usr/bin/env bash
# run.sh — Alpha Research 2026 startup script (Ubuntu/Linux)
# Usage: ./run.sh [command] [options]
#
# Commands:
#   backtest          Run full signal backtest (local, via Bazel)
#   monitor           Run signal decay monitor (local, via Bazel)
#   notebook          Launch JupyterLab (local)
#   docker-build      Build Docker image
#   docker-notebook   Launch JupyterLab in Docker
#   docker-backtest   Run backtest in Docker
#   docker-monitor    Run monitor in Docker
#   docker-down       Stop Docker stack
#   test              Run all tests (C++ + Python via Bazel)
#   build             Build all C++ targets via Bazel
#
# Environment:
#   N_DAYS=2520  N_CURRENCIES=8  SLACK_WEBHOOK=<url>  LOG_LEVEL=INFO
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BAZEL="${BAZEL_PATH:-bazel}"
PYTHON="${PYTHON_PATH:-python3.13}"
CMD="${1:-help}"
shift || true

# ── Ensure Bazel is available ─────────────────────────────────────────────────
_check_bazel() {
  if ! command -v "$BAZEL" &>/dev/null; then
    echo "Bazel not found. Installing bazelisk..."
    curl -fsSL https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64 \
      -o /usr/local/bin/bazel
    chmod +x /usr/local/bin/bazel
    BAZEL="bazel"
  fi
}

# ── Ensure Python package is installed ───────────────────────────────────────
_check_python() {
  if ! "$PYTHON" -c "import alpha_research" &>/dev/null 2>&1; then
    echo "Installing alpha_research Python package..."
    pip install -e ".[dev]" --quiet
  fi
}

case "$CMD" in
  backtest)
    _check_bazel
    echo "▶ Running full backtest via Bazel..."
    export N_DAYS="${N_DAYS:-2520}"
    export N_CURRENCIES="${N_CURRENCIES:-8}"
    "$BAZEL" run //:backtest "$@"
    ;;

  monitor)
    _check_bazel
    echo "▶ Running signal decay monitor via Bazel..."
    "$BAZEL" run //:monitor "$@"
    ;;

  notebook)
    _check_python
    echo "▶ Launching JupyterLab (local)..."
    echo "   Open: http://localhost:8888"
    jupyter lab --notebook-dir=notebooks --ip=0.0.0.0 --no-browser "$@"
    ;;

  docker-build)
    echo "▶ Building Docker image..."
    "$BAZEL" run //:docker_build
    ;;

  docker-notebook)
    echo "▶ Launching JupyterLab in Docker..."
    echo "   Open: http://localhost:8888"
    docker compose up jupyter
    ;;

  docker-backtest)
    echo "▶ Running backtest in Docker..."
    docker compose run --rm backtest
    ;;

  docker-monitor)
    echo "▶ Running monitor in Docker..."
    docker compose run --rm monitor
    ;;

  docker-down)
    "$BAZEL" run //:docker_down
    ;;

  test)
    _check_bazel
    echo "▶ Running all tests (C++ + Python) via Bazel..."
    "$BAZEL" test //:test_all --test_output=short
    ;;

  build)
    _check_bazel
    echo "▶ Building all targets via Bazel..."
    "$BAZEL" build //...
    ;;

  help|*)
    cat << 'HELP'
Alpha Research 2026 — run.sh

Usage: ./run.sh <command>

Commands:
  backtest         Full signal backtest (local, Bazel)
  monitor          Signal decay monitor (local, Bazel)
  notebook         JupyterLab notebook (local)
  docker-build     Build Docker image
  docker-notebook  JupyterLab in Docker (http://localhost:8888)
  docker-backtest  Backtest in Docker
  docker-monitor   Monitor in Docker
  docker-down      Stop Docker stack
  test             All C++ + Python tests (Bazel)
  build            Build all targets (Bazel)

Environment variables:
  N_DAYS          Synthetic data days (default: 2520)
  N_CURRENCIES    G10 currencies (default: 8)
  SLACK_WEBHOOK   Slack alert URL
  LOG_LEVEL       INFO | WARNING | DEBUG

Examples:
  ./run.sh backtest
  N_DAYS=500 ./run.sh monitor
  ./run.sh docker-notebook
  ./run.sh test
HELP
    ;;
esac
