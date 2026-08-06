#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if command -v uv >/dev/null 2>&1; then
  uv run python scripts/build_kv_subspace.py
  uv run python scripts/run_kv_subspace_decode.py
else
  python scripts/build_kv_subspace.py
  python scripts/run_kv_subspace_decode.py
fi

echo "KV_SUBSPACE_DECODE_OK=1"
