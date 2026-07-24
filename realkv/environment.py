"""Runtime metadata collection that does not alter the host."""

from __future__ import annotations
import json
import os
import platform
import subprocess
import time


def _cmd(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def collect() -> dict:
    memory = None
    try:
        memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError):
        pass
    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "architecture": platform.machine(),
        "os": platform.platform(),
        "kernel": platform.release(),
        "driver": _cmd(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
        "gpu_query": _cmd(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,compute_cap",
                "--format=csv,noheader",
            ]
        ),
        "reported_cuda_version": _cmd(
            ["nvidia-smi", "--query-gpu=cuda_version", "--format=csv,noheader"]
        ),
        "system_memory_bytes": memory,
        "container_runtime": _cmd(["docker", "--version"]),
        "git_commit": _cmd(["git", "rev-parse", "HEAD"]),
        "memory_note": "Record CUDA allocation, reservation, process RSS, and system memory separately.",
    }


def write(path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(collect(), handle, indent=2, sort_keys=True)
        handle.write("\n")
