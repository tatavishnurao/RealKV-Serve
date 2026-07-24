#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_OUT="$ROOT/reports/milestone1/raw/host_environment.txt"
PUBLISHED_OUT="$ROOT/reports/milestone1/published/environment_summary.json"
mkdir -p "$(dirname "$RAW_OUT")" "$(dirname "$PUBLISHED_OUT")"
mandatory_failed=0

probe() {
  local required="$1" label="$2"
  shift 2
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

exec > >(tee "$RAW_OUT") 2>&1
date -u +%FT%TZ
probe 1 uname_all uname -a
probe 1 architecture uname -m
probe 1 os_release cat /etc/os-release
probe 1 nvidia_smi nvidia-smi
probe 1 nvidia_smi_query nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv
probe 0 nvcc nvcc --version
probe 1 docker_version docker --version
probe 1 docker_info docker info
probe 0 nvidia_container_cli nvidia-container-cli info
probe 1 system_memory free -h
probe 1 disk_space df -h "$HOME"
probe 1 python python3 --version
probe 1 git git --version

PYTHONPATH="$ROOT" python3 -c 'from realkv.environment import write; write("'"$PUBLISHED_OUT"'")'
if (( mandatory_failed )); then
  echo "mandatory RTX 4060 workstation probes failed" >&2
  exit 1
fi
