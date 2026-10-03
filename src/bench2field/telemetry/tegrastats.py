"""Jetson telemetry via tegrastats (Orin, Thor).

tegrastats prints one line per interval. Field names differ between Jetson
generations and JetPack releases, so the parser is pattern-based rather than
positional: anything that looks like a power rail, a temperature, a RAM
reading, a GPU load or a memory-controller (EMC) load is captured.

What a board prints, from real captures under bringup/:

* AGX Thor, L4T R38.4: RAM, per-core CPU, five temperatures, and the rails
  VDD_GPU, VDD_CPU_SOC_MSS, VIN_SYS_5V0 and VIN. No GR3D_FREQ and no EMC_FREQ,
  idle or under GPU load, with or without --readall, so `gpu_util_pct` and
  `emc_util_pct` do not exist there and replay cannot steer the GPU or
  memory-bandwidth stressors on a Thor.
* Orin NX, L4T R36.4: RAM, SWAP, per-core CPU (six online in MAXN_SUPER),
  GR3D_FREQ, nine temperatures, and the rails VDD_IN, VDD_CPU_GPU_CV and
  VDD_SOC. No EMC_FREQ, with or without --readall.

Total board power goes by a different rail name on each board, so it is also
reported under one neutral name, `power_board_w`. It is the only power
channel that may be compared across Jetsons, and it is never the same thing
as NVML's `power_gpu_w`.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import threading
from typing import Any

from .base import TelemetrySampler

# Rails that carry the whole board's input power, in order of preference.
# Only rails confirmed against a real capture belong here.
#   VIN:    AGX Thor (bringup/thor/tegrastats_*.txt); it is the largest rail
#           and exceeds the sum of the other three.
#   VDD_IN: Orin NX (bringup/orin/tegrastats_*.txt); likewise the largest rail,
#           above VDD_CPU_GPU_CV + VDD_SOC.
BOARD_POWER_RAILS = ("VIN", "VDD_IN")

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
    for rail in BOARD_POWER_RAILS:
        if (key := f"power_{rail.lower()}_w") in s:
            s["power_board_w"] = s[key]
            break
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


def read_nvpmodel() -> str | None:
    """The active power mode as `nvpmodel -q` states it, e.g. "NV Power Mode:
    120W". Read-only; None if nvpmodel is missing or fails."""
    try:
        out = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True, timeout=5)
    except Exception:
        return None
    lines = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
    for ln in lines:
        if "Power Mode" in ln:
            return ln
    return lines[0] if lines else None


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
        return {"telemetry": self.name, "nvpmodel": read_nvpmodel()}

    def close(self) -> None:
        self._proc.terminate()
