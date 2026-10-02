"""The metrics that make Bench2Field different.

* field_retention: how much of an optimization's bench speedup survives on
  the robot.
* attribute_gap: how the bench-to-field slowdown splits across causes, using
  controlled replay runs that add one stressor at a time.
* replay_validity: whether a replayed load profile reproduces the field well
  enough to stand in for it.

All three take latency statistics from RunReports, so they work the same on
any backend, model or vendor.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from .schema import LatencyStats, RunReport

# Service-time stats, and the same stats on response time (scheduled arrival
# to completion). Comparisons default to response p95: it is what the robot's
# control loop experiences, and it includes queueing behind slow frames.
SERVICE_STATS = ("p50_ms", "p95_ms", "p99_ms", "max_ms", "mean_ms")
RESPONSE_PREFIX = "response_"
STATS = tuple(RESPONSE_PREFIX + s for s in SERVICE_STATS) + SERVICE_STATS
DEFAULT_STAT = "response_p95_ms"


def latency_stats(samples_ms: Sequence[float]) -> LatencyStats:
    if len(samples_ms) == 0:
        raise ValueError("no latency samples")
    a = np.asarray(samples_ms, dtype=float)
    p50, p95, p99 = np.percentile(a, [50, 95, 99])
    return LatencyStats(
        n=int(a.size),
        p50_ms=float(p50),
        p95_ms=float(p95),
        p99_ms=float(p99),
        max_ms=float(a.max()),
        mean_ms=float(a.mean()),
    )


def _stat(report: RunReport, target_hz: float, stat: str) -> float:
    if stat not in STATS:
        raise ValueError(f"stat must be one of {STATS}")
    tier = report.tier(target_hz)
    if not stat.startswith(RESPONSE_PREFIX):
        return float(getattr(tier.latency, stat))
    if tier.response is None:
        raise ValueError(
            f"run {report.run_id} (schema {report.schema_version}) has no response-time "
            f"stats; re-run it, or compare service time with --stat {stat[len(RESPONSE_PREFIX):]}"
        )
    return float(getattr(tier.response, stat[len(RESPONSE_PREFIX):]))


def speedup(baseline: RunReport, optimized: RunReport, target_hz: float, stat: str = DEFAULT_STAT) -> float:
    """Latency speedup of `optimized` over `baseline` at one load tier (>1 is faster)."""
    return _stat(baseline, target_hz, stat) / _stat(optimized, target_hz, stat)


@dataclass
class Retention:
    stat: str
    target_hz: float
    bench_speedup: float
    field_speedup: float
    retention: float | None  # share of the bench gain kept in the field; None if no bench gain
    note: str

    def summary(self) -> str:
        r = "n/a" if self.retention is None else f"{self.retention:.0%}"
        return (
            f"{self.stat} @ {self.target_hz:g} Hz: bench {self.bench_speedup:.2f}x, "
            f"field {self.field_speedup:.2f}x, retention {r}. {self.note}"
        )


def field_retention(
    bench_baseline: RunReport,
    bench_optimized: RunReport,
    field_baseline: RunReport,
    field_optimized: RunReport,
    target_hz: float,
    stat: str = DEFAULT_STAT,
) -> Retention:
    """Share of an optimization's bench gain that survives in the field.

        retention = (S_field - 1) / (S_bench - 1)

    where S is the latency speedup of optimized over baseline. 100% means the
    whole gain survived; 0% means none did; negative means the optimization
    made things worse on the robot. Defined only when the bench shows a gain.
    """
    _check_pair(bench_baseline, bench_optimized, field_baseline, field_optimized)
    s_bench = speedup(bench_baseline, bench_optimized, target_hz, stat)
    s_field = speedup(field_baseline, field_optimized, target_hz, stat)
    if s_bench <= 1.0:
        return Retention(stat, target_hz, s_bench, s_field, None,
                         "No bench gain, so retention is undefined.")
    r = (s_field - 1.0) / (s_bench - 1.0)
    if r < 0:
        note = "Optimization is slower than baseline in the field."
    elif r < 0.5:
        note = "Most of the bench gain is lost in the field."
    elif r <= 1.05:
        note = "Gain largely survives in the field."
    else:
        note = "Field gain exceeds bench gain; check for contention the baseline suffered more from."
    return Retention(stat, target_hz, s_bench, s_field, r, note)


def _check_pair(bb: RunReport, bo: RunReport, fb: RunReport, fo: RunReport) -> None:
    if bb.variant.key != fb.variant.key or bo.variant.key != fo.variant.key:
        raise ValueError("bench and field runs must use the same variants")
    if bb.environment == fb.environment:
        raise ValueError("baseline bench and field runs share an environment label")


@dataclass
class Attribution:
    target_hz: float
    stat: str
    idle_ms: float
    field_ms: float
    total_gap_ms: float
    components_ms: dict[str, float]  # stressor name -> added latency vs idle
    interaction_ms: float            # field gap not explained by single stressors

    def shares(self) -> dict[str, float]:
        if self.total_gap_ms <= 0:
            return {}
        out = {k: v / self.total_gap_ms for k, v in self.components_ms.items()}
        out["interaction/unexplained"] = self.interaction_ms / self.total_gap_ms
        return out


def attribute_gap(
    idle: RunReport,
    field: RunReport,
    single_stressor_runs: Iterable[tuple[str, RunReport]],
    target_hz: float,
    stat: str = DEFAULT_STAT,
) -> Attribution:
    """Split the idle-bench to field slowdown across causes.

    Each single-stressor run is the same variant on the bench with exactly one
    replayed stressor active (e.g. "thermal", "cpu", "membw", "gpu"). A
    stressor's component is its latency added over idle; whatever the sum of
    components does not explain is reported as interaction/unexplained rather
    than forced into a cause.
    """
    idle_v = _stat(idle, target_hz, stat)
    field_v = _stat(field, target_hz, stat)
    comps: dict[str, float] = {}
    for name, rep in single_stressor_runs:
        if rep.variant.key != idle.variant.key:
            raise ValueError(f"stressor run {name!r} uses a different variant")
        comps[name] = _stat(rep, target_hz, stat) - idle_v
    gap = field_v - idle_v
    return Attribution(target_hz, stat, idle_v, field_v, gap, comps, gap - sum(comps.values()))


@dataclass
class ReplayValidity:
    stat: str
    target_hz: float
    replay_ms: float
    field_ms: float
    rel_error: float
    tolerance: float

    @property
    def valid(self) -> bool:
        return self.rel_error <= self.tolerance


def replay_validity(
    replay: RunReport, field: RunReport, target_hz: float,
    stat: str = DEFAULT_STAT, tolerance: float = 0.10,
) -> ReplayValidity:
    """A load profile may stand in for the field only if replaying it reproduces
    field latency within `tolerance` (relative). Check this before trusting any
    replay-only result."""
    if replay.variant.key != field.variant.key:
        raise ValueError("replay and field runs must use the same variant")
    r = _stat(replay, target_hz, stat)
    f = _stat(field, target_hz, stat)
    return ReplayValidity(stat, target_hz, r, f, abs(r - f) / f, tolerance)
