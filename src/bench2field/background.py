"""What else the machine was doing when a run started.

A `bench-idle` label is a claim. This records the evidence for it in the
report: the containers that were up, the busiest processes, the load
average, and whatever the operator says they stopped beforehand.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any

TOP_N = 5


def _docker_containers() -> list[dict[str, str]] | None:
    """Running containers, or None when docker is not usable here."""
    if shutil.which("docker") is None:
        return None
    try:
        out = subprocess.run(["docker", "ps", "--format", "{{.Names}}\t{{.Image}}\t{{.Status}}"],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    rows = []
    for line in out.stdout.splitlines():
        name, image, status = (line.split("\t") + ["", ""])[:3]
        if name:
            rows.append({"name": name, "image": image, "status": status})
    return rows


def _top_processes(n: int = TOP_N) -> list[dict[str, Any]]:
    """The n busiest processes by CPU, excluding this one."""
    try:
        out = subprocess.run(["ps", "-eo", "pid,pcpu,pmem,args", "--sort=-pcpu", "--no-headers"],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return []
    me = os.getpid()
    rows = []
    for line in out.stdout.splitlines():
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        pid, pcpu, pmem, cmd = parts
        if int(pid) == me:
            continue
        rows.append({"pid": int(pid), "cpu_pct": float(pcpu), "mem_pct": float(pmem), "cmd": cmd[:120]})
        if len(rows) == n:
            break
    return rows


def snapshot(stopped: list[str] | None = None) -> dict[str, Any]:
    """Record for the report: `stopped` is what the operator shut down for
    this run (free text, e.g. "docker container urban-edge-vllm"); the rest
    is observed."""
    try:
        load_1m = os.getloadavg()[0]
    except OSError:
        load_1m = None
    return {
        "stopped_for_this_run": list(stopped or []),
        "containers_running": _docker_containers(),
        "top_processes": _top_processes(),
        "load_avg_1m": load_1m,
    }
