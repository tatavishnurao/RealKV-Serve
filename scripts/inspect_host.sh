#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/reports/milestone1/raw/host_environment.txt"
mkdir -p "$(dirname "$OUT")"
mandatory_failed=0

probe() {
  local required="$1" label="$2"; shift 2
  printf '\n--- %s ---\n' "$label"
  if "$@" 2>&1; then
    echo "PROBE_STATUS label=$label required=$required exit_code=0"
  else
    local code=$?
    echo "PROBE_STATUS label=$label required=$required exit_code=$code"
    if [[ "$required" == 1 ]]; then
      mandatory_failed=1
    fi
  fi
}

exec > >(tee "$OUT") 2>&1
{
  date -u +%FT%TZ
  probe 1 architecture uname -m
  probe 1 os_release cat /etc/os-release
  probe 1 gpu nvidia-smi
  probe 0 cuda nvcc --version
  probe 1 docker_version docker --version
  probe 0 docker_info docker info
  probe 0 nvidia_container_cli nvidia-container-cli info
  probe 1 system_memory free -h
  probe 0 disk_space df -h "$HOME"
  probe 1 git git --version
  probe 1 python python3 --version
  probe 1 container_gpu docker run --rm --runtime=nvidia ubuntu:24.04 nvidia-smi
}

PYTHONPATH="$ROOT" python3 -c 'from realkv.environment import write; write("'"$ROOT"'/reports/milestone1/raw/environment.json")'
if (( mandatory_failed )); then
  echo "mandatory DGX host probes failed" >&2
  exit 1
fi
