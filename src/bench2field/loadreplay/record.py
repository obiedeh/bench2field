"""Record a field-load profile on the robot.

Run this on the robot with everything that normally runs (SLAM, sensor
drivers, planners) but WITHOUT the model under test. The result is the
background load the model will have to share the machine with, saved as a
reusable JSON artifact that `replay` can reproduce on the bench.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..schema import describe_platform
from ..telemetry.base import TelemetrySampler

# Channels the replayer knows how to reproduce.
TARGET_CHANNELS = ("cpu_util_mean_pct", "gpu_util_pct", "emc_util_pct")


@dataclass
class LoadProfile:
    name: str
    duration_s: float
    interval_s: float
    platform: dict[str, Any]
    samples: list[dict[str, float]]
    targets: dict[str, float] = field(default_factory=dict)  # p50 of each target channel
    soak_temp_c: float | None = None                         # p50 of the hottest sensor
    recorded_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    notes: str = ""

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> LoadProfile:
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def profile_from_samples(name: str, samples: list[dict[str, float]], interval_s: float,
                         platform: dict[str, Any] | None = None, notes: str = "") -> LoadProfile:
    targets: dict[str, float] = {}
    for ch in TARGET_CHANNELS:
        vals = [s[ch] for s in samples if ch in s]
        if vals:
            targets[ch] = float(np.median(vals))
    hottest = [max(v for k, v in s.items() if k.startswith("temp_") and k.endswith("_c"))
               for s in samples if any(k.startswith("temp_") for k in s)]
    duration = (samples[-1]["t"] - samples[0]["t"]) if len(samples) > 1 and "t" in samples[0] else 0.0
    return LoadProfile(
        name=name, duration_s=duration, interval_s=interval_s,
        platform=platform or {}, samples=samples, targets=targets,
        soak_temp_c=float(np.median(hottest)) if hottest else None, notes=notes,
    )


def record(name: str, sampler: TelemetrySampler, duration_s: float, notes: str = "") -> LoadProfile:
    sampler.start()
    time.sleep(duration_s)
    sampler.stop()
    return profile_from_samples(name, sampler.samples, sampler.interval_s,
                                describe_platform() | sampler.describe(), notes)
