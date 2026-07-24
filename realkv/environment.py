"""Sanitized host-environment metadata collection."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from pathlib import Path


def _cmd(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _os_release() -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key in {"ID", "VERSION_ID", "PRETTY_NAME"}:
                values[key.lower()] = value.strip('"')
    except OSError:
        pass
    return values


def collect(
    *,
    container_gpu_probe_status: str = "NOT_RUN",
    container_gpu_probe_failure_reason: str | None = None,
) -> dict:
    memory = None
    try:
        memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError):
        pass

    docker_version = _cmd(["docker", "--version"])
    docker_info = _cmd(["docker", "info"])
    nvidia_smi = _cmd(["nvidia-smi"])
    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "architecture": platform.machine(),
        "kernel": platform.release(),
        "os_release": _os_release(),
        "gpu_query": _cmd(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,compute_cap",
                "--format=csv,noheader",
            ]
        ),
        "nvidia_smi_available": nvidia_smi is not None,
        "docker_version": docker_version,
        "docker_info_available": docker_info is not None,
        "nvidia_container_cli_available": _cmd(["nvidia-container-cli", "info"]) is not None,
        "system_memory_bytes": memory,
        "python_version": platform.python_version(),
        "git_version": _cmd(["git", "--version"]),
        "container_gpu_probe_status": container_gpu_probe_status,
        "container_gpu_probe_failure_reason": container_gpu_probe_failure_reason,
        "memory_note": "Record CUDA allocation, reservation, process RSS, and system memory separately.",
    }


def write(
    path: str,
    *,
    container_gpu_probe_status: str = "NOT_RUN",
    container_gpu_probe_failure_reason: str | None = None,
) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            collect(
                container_gpu_probe_status=container_gpu_probe_status,
                container_gpu_probe_failure_reason=container_gpu_probe_failure_reason,
            ),
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
