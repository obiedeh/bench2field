"""Jetson telemetry via tegrastats (Orin, Thor).

tegrastats prints one line per interval. Field names differ between Jetson
generations and JetPack releases, so the parser is pattern-based rather than
positional: anything that looks like a power rail, a temperature, a RAM
reading, a GPU load or a memory-controller (EMC) load is captured.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import threading
from typing import Any

from .base import TelemetrySampler

_RAIL = re.compile(r"\b([A-Z][A-Z0-9_]+)\s+(\d+)mW/(\d+)mW")
_TEMP = re.compile(r"\b([A-Za-z][A-Za-z0-9_]*)@(-?\d+(?:\.\d+)?)C\b")
_RAM = re.compile(r"\bRAM\s+(\d+)/(\d+)MB")
_GR3D = re.compile(r"\bGR3D_FREQ\s+(\d+)%")
_EMC = re.compile(r"\bEMC_FREQ\s+(\d+)%")
_CPU = re.compile(r"\bCPU\s+\[([^\]]*)\]")


def parse_tegrastats_line(line: str) -> dict[str, float]:
    s: dict[str, float] = {}
    for name, inst, _avg in _RAIL.findall(line):
        s[f"power_{name.lower()}_w"] = int(inst) / 1000.0
    for name, val in _TEMP.findall(line):
        s[f"temp_{name.lower()}_c"] = float(val)
    if m := _RAM.search(line):
        s["ram_used_mb"] = float(m.group(1))
        s["ram_total_mb"] = float(m.group(2))
    if m := _GR3D.search(line):
        s["gpu_util_pct"] = float(m.group(1))
    if m := _EMC.search(line):
        s["emc_util_pct"] = float(m.group(1))
    if m := _CPU.search(line):
        loads = [float(x.split("%")[0]) for x in m.group(1).split(",") if "%" in x]
        if loads:
            s["cpu_util_mean_pct"] = sum(loads) / len(loads)
            s["cpu_util_max_pct"] = max(loads)
            s["cpu_cores_online"] = float(len(loads))
    return s


class TegrastatsSampler(TelemetrySampler):
    """Streams `tegrastats --interval` and keeps the latest parsed line."""

    name = "tegrastats"

    @staticmethod
    def available() -> bool:
        return shutil.which("tegrastats") is not None

    def __init__(self, interval_s: float = 0.5) -> None:
        super().__init__(interval_s)
        self._latest: dict[str, float] = {}
        self._lock = threading.Lock()
        self._proc = subprocess.Popen(
            ["tegrastats", "--interval", str(max(100, int(interval_s * 1000)))],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self) -> None:
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            parsed = parse_tegrastats_line(line)
            if parsed:
                with self._lock:
                    self._latest = parsed

    def read_once(self) -> dict[str, float]:
        with self._lock:
            return dict(self._latest)

    def describe(self) -> dict[str, Any]:
        info: dict[str, Any] = {"telemetry": self.name}
        try:
            out = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True, timeout=5)
            info["nvpmodel"] = out.stdout.strip().splitlines()[0] if out.stdout else None
        except Exception:
            info["nvpmodel"] = None
        return info

    def close(self) -> None:
        self._proc.terminate()
