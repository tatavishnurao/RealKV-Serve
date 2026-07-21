#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/reports/milestone1/host_environment.txt"
mkdir -p "$(dirname "$OUT")"
{
  date -u +%FT%TZ; uname -a; uname -m; cat /etc/os-release; nvidia-smi; nvcc --version || true
  docker --version; docker info; nvidia-container-cli info || true; free -h; df -h "$HOME"; git --version; python3 --version
  echo '--- container GPU probe ---'
  docker run --rm --runtime=nvidia ubuntu:24.04 nvidia-smi
} 2>&1 | tee "$OUT"
PYTHONPATH="$ROOT" python3 -c 'from realkv.environment import write; write("'"$ROOT"'/reports/milestone1/environment.json")'
