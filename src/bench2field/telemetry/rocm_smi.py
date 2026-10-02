"""AMD GPU telemetry via rocm-smi (e.g. MI300X on AMD Developer Cloud).

rocm-smi's JSON keys change between ROCm releases, so values are matched by
key text rather than exact names. Not yet validated on hardware: confirm the
parsed channels against `rocm-smi` output on first use and pin them here.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from typing import Any

from .base import TelemetrySampler

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def parse_rocm_smi_json(payload: dict[str, Any], card: str = "card0") -> dict[str, float]:
    data = payload.get(card) or next(iter(payload.values()), {})
    s: dict[str, float] = {}
    for key, raw in data.items():
        k = key.lower()
        m = _NUM.search(str(raw))
        if not m:
            continue
        v = float(m.group())
        if "power" in k and "(w)" in k and "cap" not in k:
            s["power_gpu_w"] = v
        elif "temperature" in k and "(c)" in k:
            label = "junction" if "junction" in k else "edge" if "edge" in k else "memory" if "memory" in k else "gpu"
            s[f"temp_{label}_c"] = v
        elif "gpu use" in k:
            s["gpu_util_pct"] = v
        elif "memory" in k and "use" in k and "%" in k:
            s["gpu_mem_util_pct"] = v
    return s


class RocmSmiSampler(TelemetrySampler):
    name = "rocm-smi"

    @staticmethod
    def available() -> bool:
        return shutil.which("rocm-smi") is not None

    def read_once(self) -> dict[str, float]:
        out = subprocess.run(
            ["rocm-smi", "--showpower", "--showtemp", "--showuse", "--showmemuse", "--json"],
            capture_output=True, text=True, timeout=10,
        )
        return parse_rocm_smi_json(json.loads(out.stdout))
