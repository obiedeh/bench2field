"""Core runner: open-loop load tiers with deadline accounting.

Requests are scheduled at a fixed rate (open loop), the way sensor frames
arrive on a robot. A request misses its deadline when it finishes later than
its scheduled arrival plus the deadline, so queueing behind a slow frame counts
against you, as it would in a real control loop.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .metrics import latency_stats
from .schema import RunReport, TierResult, Variant, describe_platform, validate_environment
from .telemetry.base import NullSampler, TelemetrySampler

InferFn = Callable[[Any], Any]
InputFn = Callable[[int], Any]


@dataclass
class RunConfig:
    tiers_hz: Sequence[float] = (10.0, 30.0, 100.0)
    duration_s: float = 60.0
    warmup_s: float = 5.0
    deadline_ms: float = 33.3
    # Cooldown gating: before each tier, wait until the sampler reports the
    # hottest sensor at or below this temperature (skipped if None or the
    # sampler has no temperature).
    cooldown_max_c: float | None = None
    cooldown_timeout_s: float = 300.0
    extra: dict[str, Any] = field(default_factory=dict)


def _wait_for_cooldown(sampler: TelemetrySampler, max_c: float, timeout_s: float) -> float | None:
    deadline = time.monotonic() + timeout_s
    temp = sampler.read_max_temp_c()
    while temp is not None and temp > max_c and time.monotonic() < deadline:
        time.sleep(2.0)
        temp = sampler.read_max_temp_c()
    return temp


def run_tier(
    infer: InferFn,
    make_input: InputFn,
    target_hz: float,
    duration_s: float,
    deadline_ms: float,
    warmup_s: float = 0.0,
    sampler: TelemetrySampler | None = None,
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
) -> TierResult:
    if target_hz <= 0 or duration_s <= 0:
        raise ValueError("target_hz and duration_s must be positive")
    sampler = sampler or NullSampler()
    period = 1.0 / target_hz

    warm_end = clock() + warmup_s
    i = 0
    while clock() < warm_end:
        infer(make_input(i))
        i += 1

    n_requests = max(1, int(round(duration_s * target_hz)))
    latencies: list[float] = []
    misses = 0
    sampler.start()
    t0 = clock()
    for k in range(n_requests):
        scheduled = t0 + k * period
        now = clock()
        if now < scheduled:
            sleep(scheduled - now)
        x = make_input(k)
        start = clock()
        infer(x)
        end = clock()
        latencies.append((end - start) * 1000.0)
        if (end - scheduled) * 1000.0 > deadline_ms:
            misses += 1
    elapsed = clock() - t0
    telemetry = sampler.stop()

    return TierResult(
        target_hz=target_hz,
        actual_hz=n_requests / elapsed if elapsed > 0 else float("nan"),
        duration_s=elapsed,
        deadline_ms=deadline_ms,
        deadline_misses=misses,
        latency=latency_stats(latencies),
        telemetry=telemetry,
    )


def run(
    variant: Variant,
    environment: str,
    infer: InferFn,
    make_input: InputFn,
    config: RunConfig,
    sampler: TelemetrySampler | None = None,
    platform_extra: dict[str, Any] | None = None,
) -> RunReport:
    validate_environment(environment)
    sampler = sampler or NullSampler()
    plat = describe_platform() | sampler.describe() | (platform_extra or {})
    tiers: list[TierResult] = []
    for hz in config.tiers_hz:
        if config.cooldown_max_c is not None:
            plat.setdefault("cooldown_start_c", {})[str(hz)] = _wait_for_cooldown(
                sampler, config.cooldown_max_c, config.cooldown_timeout_s
            )
        tiers.append(
            run_tier(infer, make_input, hz, config.duration_s, config.deadline_ms,
                     config.warmup_s, sampler)
        )
    return RunReport(variant=variant, environment=environment, tiers=tiers, platform=plat)
