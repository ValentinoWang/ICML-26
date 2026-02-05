#!/usr/bin/env bash
set -euo pipefail

cmd="${1:-demo}"
shift || true

case "$cmd" in
  demo)
    exec bash scripts/run_regression_demo.sh "$@"
    ;;
  *)
    echo "Usage: bash run.sh [demo] [args...]" >&2
    echo "  demo: run the small CPU regression demo (default)" >&2
    exit 2
    ;;
esac

