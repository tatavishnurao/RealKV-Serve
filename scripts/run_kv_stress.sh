#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python scripts/run_kv_stress.py

echo "KV_STRESS_RUN_OK=1"
