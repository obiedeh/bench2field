"""CPU frequency and governor, read from Linux cpufreq sysfs.

Host-side stages (decode, preprocess, postprocess) run on the CPU, and a
governor such as schedutil moves each core's clock with load. This records
which governor was active and what frequency each core actually ran at, so a
difference in host-stage time between two runs can be checked against it.
Read-only: nothing here changes a governor or a clock.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

SYSFS_CPU = Path("/sys/devices/system/cpu")


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def cpu_dirs(root: Path = SYSFS_CPU) -> list[Path]:
    return sorted((d for d in root.glob("cpu[0-9]*") if (d / "cpufreq").is_dir()),
                  key=lambda d: int(d.name[3:]))


def read_cpufreq(root: Path = SYSFS_CPU) -> dict[str, int]:
    """Current frequency of each core in kHz, e.g. {"cpu0": 1984000}."""
    out: dict[str, int] = {}
    for d in cpu_dirs(root):
        v = _read(d / "cpufreq" / "scaling_cur_freq")
        if v and v.isdigit():
            out[d.name] = int(v)
    return out


def describe_cpufreq(root: Path = SYSFS_CPU) -> dict[str, Any]:
    """Governor and frequency limits per core; empty if cpufreq is absent."""
    info: dict[str, Any] = {}
    for d in cpu_dirs(root):
        f = d / "cpufreq"
        info[d.name] = {
            "governor": _read(f / "scaling_governor"),
            "min_khz": int(v) if (v := _read(f / "scaling_min_freq")) and v.isdigit() else None,
            "max_khz": int(v) if (v := _read(f / "scaling_max_freq")) and v.isdigit() else None,
        }
    return info


class CpuFreqSampler:
    """Samples every core's frequency in a background thread."""

    def __init__(self, interval_s: float = 0.05, root: Path = SYSFS_CPU) -> None:
        self.interval_s, self.root = interval_s, root
        self.samples: list[tuple[float, dict[str, int]]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            r = read_cpufreq(self.root)
            if r:
                self.samples.append((time.perf_counter(), r))
            self._stop.wait(self.interval_s)

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        return self.summary()

    def summary(self) -> dict[str, Any]:
        """Per-core p50/min/max in kHz, the mean across cores over time, the
        governors, and the time series of the cross-core mean (seconds on the
        perf_counter clock, kHz) for matching against per-frame timings."""
        cores = sorted({c for _, r in self.samples for c in r}, key=lambda c: int(c[3:]))
        per_core = {}
        for c in cores:
            v = np.array([r[c] for _, r in self.samples if c in r], dtype=float)
            per_core[c] = {"p50_khz": float(np.median(v)), "min_khz": float(v.min()), "max_khz": float(v.max())}
        series = [(t, float(np.mean(list(r.values())))) for t, r in self.samples]
        limits = describe_cpufreq(self.root)
        means = np.array([m for _, m in series]) if series else np.array([])
        return {
            "n_samples": len(self.samples), "interval_s": self.interval_s,
            "cores": limits,
            "per_core": per_core,
            "mean_khz": {"p50": float(np.median(means)), "min": float(means.min()), "max": float(means.max()),
                         "mean": float(means.mean())} if means.size else None,
            # Share of core-samples that were at the core's maximum frequency.
            "share_at_max": float(np.mean([r[c] >= (limits.get(c, {}).get("max_khz") or float("inf"))
                                           for _, r in self.samples for c in r])) if self.samples else None,
            "series": [[round(t, 4), round(m, 1)] for t, m in series],
        }
