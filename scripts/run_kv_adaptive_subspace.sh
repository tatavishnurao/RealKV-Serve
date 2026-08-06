#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! [ -f scripts/build_kv_subspace.py ]; then
  echo "ERROR: scripts/build_kv_subspace.py not found"
  exit 1
fi

if ! [ -f scripts/kv_subspace_update.py ]; then
  echo "ERROR: scripts/kv_subspace_update.py not found"
  exit 1
fi

if ! [ -f scripts/run_kv_adaptive_subspace.py ]; then
  echo "ERROR: scripts/run_kv_adaptive_subspace.py not found"
  exit 1
fi

if command -v uv >/dev/null 2>&1; then
  uv run python scripts/build_kv_subspace.py
  uv run python scripts/run_kv_adaptive_subspace.py
else
  python scripts/build_kv_subspace.py
  python scripts/run_kv_adaptive_subspace.py
fi

echo "KV_ADAPTIVE_SUBSPACE_OK=1"
