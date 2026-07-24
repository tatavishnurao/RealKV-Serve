#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_OUT="$ROOT/reports/milestone1/raw/container_gpu_probe.txt"
PUBLISHED_OUT="$ROOT/reports/milestone1/published/environment_summary.json"
mkdir -p "$(dirname "$RAW_OUT")" "$(dirname "$PUBLISHED_OUT")"

set +e
docker run --rm --gpus all ubuntu:24.04 nvidia-smi >"$RAW_OUT" 2>&1
status=$?
set -e

if [[ "$status" -eq 0 ]]; then
  probe_status="CONFIRMED"
  failure_reason=""
else
  probe_status="FAILED"
  failure_reason="$(sed -n 's/^docker: Error response from daemon: //p' "$RAW_OUT" | tail -n 1)"
  if [[ -z "$failure_reason" ]]; then
    failure_reason="container GPU probe exited with code $status"
  fi
fi

PYTHONPATH="$ROOT" python3 -c 'from realkv.environment import write; write("'"$PUBLISHED_OUT"'", container_gpu_probe_status="'"$probe_status"'", container_gpu_probe_failure_reason="'"$failure_reason"'")'

cat "$RAW_OUT"
echo "CONTAINER_GPU_PROBE_STATUS=$probe_status"
exit "$status"
