#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${REALKV_RUNTIME_IMAGE:?Set REALKV_RUNTIME_IMAGE to a digest-pinned supported image}"
[[ "${REALKV_ALLOW_MODEL_DOWNLOAD:-}" == "1" ]] || { echo "Set REALKV_ALLOW_MODEL_DOWNLOAD=1 before downloading Qwen/Qwen3-8B into ${REALKV_MODEL_CACHE:-$ROOT/.cache/models}" >&2; exit 2; }
MODEL_CACHE="${REALKV_MODEL_CACHE:-$ROOT/.cache/models}"
mkdir -p "$MODEL_CACHE"
docker run --rm --runtime=nvidia --network=host -e HF_HOME=/models -v "$ROOT:/workspace" -v "$MODEL_CACHE:/models" "$REALKV_RUNTIME_IMAGE" \
  bash -lc 'cd /workspace && python -m realkv.generation --output reports/milestone1/baseline_generation.json' \
  2>&1 | tee "$ROOT/reports/milestone1/baseline_stdout.txt"
