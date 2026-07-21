#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${REALKV_RUNTIME_IMAGE:?Set REALKV_RUNTIME_IMAGE to a digest-pinned supported image}"
[[ "${REALKV_ALLOW_MODEL_DOWNLOAD:-}" == "1" ]] || { echo "Set REALKV_ALLOW_MODEL_DOWNLOAD=1 before downloading Qwen/Qwen3-8B into ${REALKV_MODEL_CACHE:-$ROOT/.cache/models}" >&2; exit 2; }
MODEL_CACHE="${REALKV_MODEL_CACHE:-$ROOT/.cache/models}"; mkdir -p "$MODEL_CACHE"
echo "Tracing requires the pinned source-verified observer harness; no trace is emitted by this placeholder until runtime method signatures are resolved." | tee "$ROOT/reports/milestone1/traced_stdout.txt"
exit 2
