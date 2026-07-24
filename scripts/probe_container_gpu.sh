#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/reports/milestone1/raw/container_gpu_probe.txt"
mkdir -p "$(dirname "$OUT")"

set +e
docker run --rm --gpus all ubuntu:24.04 nvidia-smi >"$OUT" 2>&1
status=$?
set -e
printf 'command=docker run --rm --gpus all ubuntu:24.04 nvidia-smi\nexit_code=%s\n' "$status" >>"$OUT"
if (( status != 0 )); then
  echo "CONTAINER_GPU_ACCESS_CONFIRMED=FAILED" >&2
  exit "$status"
fi
echo "CONTAINER_GPU_ACCESS_CONFIRMED=1"
