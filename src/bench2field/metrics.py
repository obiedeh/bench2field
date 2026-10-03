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
from dataclasses import dataclass, field

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


MIN_REPEATS = 3  # docs/METHODOLOGY.md, section 4

@dataclass
class RepeatStat:
    """One latency statistic across the repeats of one variant in one environment."""

    n: int
    median: float
    lo: float
    hi: float

    @property
    def spread(self) -> float:
        return self.hi - self.lo


def repeat_stat(reports: Sequence[RunReport], target_hz: float, stat: str = DEFAULT_STAT) -> RepeatStat:
    vals = [_stat(r, target_hz, stat) for r in reports]
    return RepeatStat(len(vals), float(np.median(vals)), min(vals), max(vals))


def _gain_is_finding(base: RepeatStat, opt: RepeatStat) -> bool | None:
    """A difference smaller than the spread between repeats is not a finding.
    None when there are too few repeats to know the spread."""
    if min(base.n, opt.n) < 2:
        return None
    return abs(base.median - opt.median) > max(base.spread, opt.spread)


@dataclass
class Retention:
    stat: str
    target_hz: float
    bench_speedup: float
    field_speedup: float
    retention: float | None  # share of the bench gain kept in the field; None if no bench gain
    note: str
    # Per group ("bench_baseline", "bench_optimized", "field_baseline",
    # "field_optimized"): the statistic across repeats. Speedups use medians.
    groups: dict[str, RepeatStat] = field(default_factory=dict)
    # Whether each gain is larger than the spread between repeats (None: unknown).
    bench_gain_is_finding: bool | None = None
    field_gain_is_finding: bool | None = None
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        r = "n/a" if self.retention is None else f"{self.retention:.0%}"
        lines = [
            f"{self.stat} @ {self.target_hz:g} Hz: bench {self.bench_speedup:.2f}x, "
            f"field {self.field_speedup:.2f}x, retention {r}. {self.note}"
        ]
        for name, g in self.groups.items():
            lines.append(f"  {name:<16} median {g.median:.3f} ms  range {g.lo:.3f} to {g.hi:.3f} ms  "
                         f"({g.n} repeat{'s' if g.n != 1 else ''})")
        lines += [f"  WARNING: {w}" for w in self.warnings]
        return "\n".join(lines)


def _as_group(runs: RunReport | Sequence[RunReport]) -> list[RunReport]:
    group = [runs] if isinstance(runs, RunReport) else list(runs)
    if not group:
        raise ValueError("a comparison group has no runs")
    return group


def field_retention(
    bench_baseline: RunReport | Sequence[RunReport],
    bench_optimized: RunReport | Sequence[RunReport],
    field_baseline: RunReport | Sequence[RunReport],
    field_optimized: RunReport | Sequence[RunReport],
    target_hz: float,
    stat: str = DEFAULT_STAT,
) -> Retention:
    """Share of an optimization's bench gain that survives in the field.

        retention = (S_field - 1) / (S_bench - 1)

    where S is the latency speedup of optimized over baseline. 100% means the
    whole gain survived; 0% means none did; negative means the optimization
    made things worse on the robot. Defined only when the bench shows a gain.

    Each argument is one run or the repeats of that run. With repeats, the
    speedups use the median across repeats, and a gain no larger than the
    spread between repeats is flagged as not a finding. Runs that may not be
    compared (different variant, deadline, drop policy or power mode) are
    refused.
    """
    groups = {"bench_baseline": _as_group(bench_baseline), "bench_optimized": _as_group(bench_optimized),
              "field_baseline": _as_group(field_baseline), "field_optimized": _as_group(field_optimized)}
    warnings = _check_comparable(groups, target_hz)
    st = {name: repeat_stat(g, target_hz, stat) for name, g in groups.items()}
    s_bench = st["bench_baseline"].median / st["bench_optimized"].median
    s_field = st["field_baseline"].median / st["field_optimized"].median
    bench_finding = _gain_is_finding(st["bench_baseline"], st["bench_optimized"])
    field_finding = _gain_is_finding(st["field_baseline"], st["field_optimized"])

    short = {name: s.n for name, s in st.items() if s.n < MIN_REPEATS}
    if short:
        warnings.append(f"fewer than {MIN_REPEATS} repeats ({', '.join(f'{k}: {v}' for k, v in short.items())}); "
                        "the spread is not established, so this is not yet a reportable result")
    if bench_finding is False:
        warnings.append("the bench gain is no larger than the spread between repeats: not a finding")
    if field_finding is False:
        warnings.append("the field difference is no larger than the spread between repeats: not a finding")

    def result(retention: float | None, note: str) -> Retention:
        return Retention(stat, target_hz, s_bench, s_field, retention, note, st,
                         bench_finding, field_finding, warnings)

    if s_bench <= 1.0:
        return result(None, "No bench gain, so retention is undefined.")
    r = (s_field - 1.0) / (s_bench - 1.0)
    if r < 0:
        note = "Optimization is slower than baseline in the field."
    elif r < 0.5:
        note = "Most of the bench gain is lost in the field."
    elif r <= 1.05:
        note = "Gain largely survives in the field."
    else:
        note = "Field gain exceeds bench gain; check for contention the baseline suffered more from."
    return result(r, note)


def _one(values: set, what: str, where: str):
    if len(values) > 1:
        raise ValueError(f"{where} differ in {what}: {sorted(map(str, values))}")
    return next(iter(values))


def _check_comparable(groups: dict[str, list[RunReport]], target_hz: float) -> list[str]:
    """Refuse comparisons docs/METHODOLOGY.md does not allow; return warnings
    for the ones it allows but a reader should know about."""
    key, env, mode, ort, deadline, drop = {}, {}, {}, {}, {}, {}
    for name, runs in groups.items():
        where = f"the repeats of {name}"
        key[name] = _one({r.variant.key for r in runs}, "variant", where)
        env[name] = _one({r.environment for r in runs}, "environment", where)
        mode[name] = _one({r.platform.get("nvpmodel") for r in runs}, "power mode", where)
        ort[name] = _one({r.platform.get("onnxruntime") for r in runs}, "onnxruntime version", where)
        deadline[name] = _one({r.tier(target_hz).deadline_ms for r in runs}, "deadline", where)
        drop[name] = _one({r.tier(target_hz).drop_late for r in runs}, "drop-late policy", where)

    if key["bench_baseline"] != key["field_baseline"] or key["bench_optimized"] != key["field_optimized"]:
        raise ValueError("bench and field runs must use the same variants")
    if env["bench_baseline"] == env["field_baseline"]:
        raise ValueError("baseline bench and field runs share an environment label")
    for side in ("bench", "field"):
        if env[f"{side}_baseline"] != env[f"{side}_optimized"]:
            raise ValueError(f"{side} baseline and optimized runs are from different environments")
        if mode[f"{side}_baseline"] != mode[f"{side}_optimized"]:
            raise ValueError(
                f"{side} baseline and optimized runs used different power modes: "
                f"{mode[f'{side}_baseline']!r} vs {mode[f'{side}_optimized']!r}")
        if ort[f"{side}_baseline"] != ort[f"{side}_optimized"]:
            raise ValueError(
                f"{side} baseline and optimized runs used different onnxruntime versions: "
                f"{ort[f'{side}_baseline']!r} vs {ort[f'{side}_optimized']!r}")
    _one(set(deadline.values()), "deadline (ms)", "the runs being compared")
    _one(set(drop.values()), "drop-late policy", "the runs being compared")

    warnings = []
    if mode["bench_baseline"] != mode["field_baseline"]:
        warnings.append(f"bench and field used different power modes ({mode['bench_baseline']!r} vs "
                        f"{mode['field_baseline']!r}); expected if they are different boards, "
                        "a mistake if they are the same one")
    if ort["bench_baseline"] != ort["field_baseline"]:
        warnings.append(f"bench and field used different onnxruntime versions ({ort['bench_baseline']!r} vs "
                        f"{ort['field_baseline']!r}); the runtime, not only the machine, differs")
    return warnings


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
