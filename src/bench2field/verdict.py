"""Field-readiness verdict: does a run fit the deployment budget?

A budget is what the robot needs, written down before testing: the control
loop deadline, how often it may be missed, the power envelope, the thermal
ceiling, the accuracy floor. A run passes only if every gate it can be
checked against passes; a gate with no data is reported, never assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .schema import RunReport


@dataclass
class Budget:
    name: str
    target_hz: float
    p99_ms: float | None = None
    max_miss_rate: float | None = None
    max_power_w: float | None = None
    power_channel: str | None = None          # e.g. power_vin_w on Jetson
    max_temp_c: float | None = None
    min_accuracy: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Budget:
        import yaml

        return cls(**yaml.safe_load(Path(path).read_text(encoding="utf-8")))


@dataclass
class Gate:
    name: str
    limit: str
    measured: str
    status: str  # pass | fail | no-data


@dataclass
class Verdict:
    budget: str
    run_id: str
    environment: str
    gates: list[Gate]

    @property
    def passed(self) -> bool:
        return all(g.status == "pass" for g in self.gates)

    @property
    def status(self) -> str:
        if any(g.status == "fail" for g in self.gates):
            return "NO-GO"
        if any(g.status == "no-data" for g in self.gates):
            return "INCOMPLETE"
        return "GO"

    def table(self) -> str:
        rows = [f"Verdict: {self.status}  (budget {self.budget}, run {self.run_id}, {self.environment})",
                f"{'gate':<16}{'limit':<16}{'measured':<16}status"]
        rows += [f"{g.name:<16}{g.limit:<16}{g.measured:<16}{g.status}" for g in self.gates]
        return "\n".join(rows)


def _hottest(tel: dict[str, Any]) -> float | None:
    peaks = [v["peak"] for k, v in tel.items()
             if k.startswith("temp_") and k.endswith("_c") and isinstance(v, dict)]
    return max(peaks) if peaks else None


def evaluate(report: RunReport, budget: Budget) -> Verdict:
    tier = report.tier(budget.target_hz)
    tel = tier.telemetry
    gates: list[Gate] = []

    def gate(name: str, limit: float, measured: float | None, fmt: str, ok) -> None:
        if measured is None:
            gates.append(Gate(name, fmt.format(limit), "-", "no-data"))
        else:
            gates.append(Gate(name, fmt.format(limit), fmt.format(measured),
                              "pass" if ok(measured, limit) else "fail"))

    if budget.p99_ms is not None:
        gate("p99 latency", budget.p99_ms, tier.latency.p99_ms, "{:.2f} ms", lambda m, l: m <= l)
    if budget.max_miss_rate is not None:
        gate("deadline miss", budget.max_miss_rate, tier.miss_rate, "{:.3%}", lambda m, l: m <= l)
    if budget.max_power_w is not None:
        ch = tel.get(budget.power_channel or "", {})
        gate("power p50", budget.max_power_w, ch.get("p50") if isinstance(ch, dict) else None,
             "{:.1f} W", lambda m, l: m <= l)
    if budget.max_temp_c is not None:
        gate("peak temp", budget.max_temp_c, _hottest(tel), "{:.1f} C", lambda m, l: m <= l)
    for metric, floor in budget.min_accuracy.items():
        gate(metric, floor, report.accuracy.get(metric), "{:.3f}", lambda m, l: m >= l)

    return Verdict(budget.name, report.run_id, report.environment, gates)
