"""Report contract.

Every Bench2Field run writes one RunReport as JSON. Everything else in the
project (retention, attribution, verdicts, dashboards) reads only this file,
so the schema is the contract: add fields, never rename or repurpose them.
"""

from __future__ import annotations

import json
import platform
import socket
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"

# Where the run happened. Field retention compares the same variant across
# environments, so this label is mandatory and must be one of these forms.
ENV_BENCH_IDLE = "bench-idle"
ENV_FIELD = "field"
ENV_REPLAY_PREFIX = "bench-replay:"  # followed by the load-profile name


@dataclass
class LatencyStats:
    n: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    mean_ms: float


@dataclass
class TierResult:
    target_hz: float
    actual_hz: float
    duration_s: float
    deadline_ms: float
    deadline_misses: int
    latency: LatencyStats
    # Summaries from the platform telemetry sampler (power rails, temps,
    # GPU/EMC utilisation). Keys are sampler-specific; see telemetry/.
    telemetry: dict[str, Any] = field(default_factory=dict)

    @property
    def miss_rate(self) -> float:
        total = max(1, self.latency.n)
        return self.deadline_misses / total


@dataclass
class Variant:
    model: str            # e.g. "yolo-detector-s"
    backend: str          # e.g. "onnxruntime"
    provider: str         # e.g. "TensorrtExecutionProvider"
    precision: str        # fp32 | fp16 | int8 | fp8 | int4 | ...
    technique: str = "baseline"  # baseline | ptq | prune-2:4 | distill | kernel:<name> ...
    notes: str = ""

    @property
    def key(self) -> str:
        return f"{self.model}/{self.backend}/{self.provider}/{self.precision}/{self.technique}"


@dataclass
class RunReport:
    variant: Variant
    environment: str
    tiers: list[TierResult]
    platform: dict[str, Any] = field(default_factory=dict)
    accuracy: dict[str, float] = field(default_factory=dict)
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    schema_version: str = SCHEMA_VERSION

    def tier(self, target_hz: float) -> TierResult:
        for t in self.tiers:
            if abs(t.target_hz - target_hz) < 1e-9:
                return t
        raise KeyError(f"no tier at {target_hz} Hz in run {self.run_id}")

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunReport:
        major = str(d.get("schema_version", "0")).split(".")[0]
        if major != SCHEMA_VERSION.split(".")[0]:
            raise ValueError(f"unsupported schema_version {d.get('schema_version')}")
        tiers = [
            TierResult(
                target_hz=t["target_hz"],
                actual_hz=t["actual_hz"],
                duration_s=t["duration_s"],
                deadline_ms=t["deadline_ms"],
                deadline_misses=t["deadline_misses"],
                latency=LatencyStats(**t["latency"]),
                telemetry=t.get("telemetry", {}),
            )
            for t in d["tiers"]
        ]
        return cls(
            variant=Variant(**d["variant"]),
            environment=d["environment"],
            tiers=tiers,
            platform=d.get("platform", {}),
            accuracy=d.get("accuracy", {}),
            run_id=d["run_id"],
            created_at=d["created_at"],
            schema_version=d["schema_version"],
        )

    @classmethod
    def load(cls, path: str | Path) -> RunReport:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def describe_platform() -> dict[str, Any]:
    """Best-effort host description; telemetry samplers add device detail."""
    info: dict[str, Any] = {
        "host": socket.gethostname(),
        "machine": platform.machine(),
        "system": platform.system(),
        "kernel": platform.release(),
        "python": platform.python_version(),
    }
    nv_tegra = Path("/etc/nv_tegra_release")
    if nv_tegra.exists():
        info["l4t_release"] = nv_tegra.read_text(errors="replace").splitlines()[0]
        info["is_jetson"] = True
    return info


def validate_environment(env: str) -> str:
    if env in (ENV_BENCH_IDLE, ENV_FIELD) or (
        env.startswith(ENV_REPLAY_PREFIX) and len(env) > len(ENV_REPLAY_PREFIX)
    ):
        return env
    raise ValueError(
        f"environment must be '{ENV_BENCH_IDLE}', '{ENV_FIELD}' or "
        f"'{ENV_REPLAY_PREFIX}<profile>', got {env!r}"
    )
