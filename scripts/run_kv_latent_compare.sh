#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if command -v uv >/dev/null 2>&1; then
  uv run python scripts/run_kv_latent_compare.py
else
  python scripts/run_kv_latent_compare.py
fi

echo "KV_LATENT_COMPARE_OK=1"
