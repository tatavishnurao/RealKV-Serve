#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if command -v uv >/dev/null 2>&1; then
  uv run python scripts/run_kv_structured_compare.py
else
  python scripts/run_kv_structured_compare.py
fi

echo "KV_STRUCTURED_COMPARE_OK=1"
