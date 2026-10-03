"""Response time (scheduled arrival to completion), the drop-late policy, and
schema 1.0 compatibility."""

import json

import pytest

from bench2field.metrics import DEFAULT_STAT, field_retention, replay_validity, speedup
from bench2field.runner import RunConfig, run, run_tier
from bench2field.schema import SCHEMA_VERSION, RunReport, Variant
from bench2field.verdict import Budget, evaluate
from test_core import BASE, OPT, report


class FakeTime:
    """A clock the test controls: sleeping and inferring both advance it."""

    def __init__(self, infer_s):
        self.t, self.infer_s = 0.0, infer_s

    def clock(self):
        return self.t

    def sleep(self, d):
        self.t += d

    def infer(self, _):
        self.t += self.infer_s


def overloaded_tier(**kw):
    # 25 ms inference against frames arriving every 10 ms: frame k starts at
    # 25k ms, so it has already waited 15k ms and responds after 25 + 15k ms.
    ft = FakeTime(0.025)
    return run_tier(ft.infer, lambda i: None, target_hz=100, duration_s=0.1,
                    clock=ft.clock, sleep=ft.sleep, **kw)


def test_response_time_includes_queue_wait_service_time_does_not():
    t = overloaded_tier(deadline_ms=32.0)
    assert t.latency.n == t.response.n == 10 and t.dropped == 0
    assert t.latency.max_ms == pytest.approx(25.0)
    assert t.response.p50_ms == pytest.approx(25 + 15 * 4.5)
    assert t.response.max_ms == pytest.approx(25 + 15 * 9)
    assert t.deadline_misses == 9  # every frame but the first


def test_response_equals_service_time_when_keeping_up():
    ft = FakeTime(0.004)
    t = run_tier(ft.infer, lambda i: None, target_hz=100, duration_s=0.1, deadline_ms=10.0,
                 clock=ft.clock, sleep=ft.sleep)
    assert t.response.p95_ms == pytest.approx(t.latency.p95_ms) == pytest.approx(4.0)
    assert t.deadline_misses == 0 and t.dropped == 0


def test_drop_late_skips_stale_frames_and_counts_them_separately():
    t = overloaded_tier(deadline_ms=32.0, drop_late=True)
    # Frames 0, 1, 2, 5, 7 run; 3, 4, 6, 8, 9 are over 32 ms stale at their turn.
    assert t.drop_late and t.dropped == 5 and t.latency.n == 5
    assert t.deadline_misses == 4  # ran, but finished late: 1, 2, 5, 7
    assert t.n_scheduled == 10
    assert t.miss_rate == pytest.approx(0.4) and t.drop_rate == pytest.approx(0.5)
    # Dropping bounds the backlog, so the worst response is far below the 160 ms without it.
    assert t.response.max_ms == pytest.approx(55.0)


def test_drop_late_is_off_by_default_and_reaches_the_report(tmp_path):
    import time

    rep = run(BASE, "bench-idle", lambda _: time.sleep(0.025), lambda i: None,
              RunConfig(tiers_hz=[100.0], duration_s=0.1, warmup_s=0.0, deadline_ms=32.0,
                        drop_late=True))
    assert RunConfig().drop_late is False
    loaded = RunReport.load(rep.save(tmp_path / "r.json"))
    t = loaded.tier(100.0)
    assert loaded.schema_version == SCHEMA_VERSION == "1.1"
    assert t.drop_late and t.dropped > 0 and t.response.n == t.latency.n


def test_schema_1_0_report_loads_but_has_no_response_stats(tmp_path):
    d = json.loads(report(BASE, "field", 10.0).to_json())
    d["schema_version"] = "1.0"
    for tier in d["tiers"]:
        for added in ("response", "dropped", "drop_late"):
            del tier[added]
    old = RunReport.from_dict(d)
    t = old.tier(30.0)
    assert t.response is None and t.dropped == 0 and t.drop_late is False
    new = report(BASE, "bench-replay:rover", 10.0)
    with pytest.raises(ValueError, match="no response-time stats"):
        replay_validity(new, old, 30.0)
    assert replay_validity(new, old, 30.0, stat="p95_ms").valid  # service time still works


def test_comparisons_default_to_response_p95():
    assert DEFAULT_STAT == "response_p95_ms"
    # Same service time on bench and in the field; only queueing differs.
    bb, bo = report(BASE, "bench-idle", 9.0), report(OPT, "bench-idle", 3.0)
    fb = report(BASE, "field", 30.0, service_p95=9.0)
    fo = report(OPT, "field", 15.0, service_p95=3.0)
    r = field_retention(bb, bo, fb, fo, 30.0)
    assert r.stat == "response_p95_ms" and r.retention == pytest.approx(0.5, rel=1e-3)
    assert field_retention(bb, bo, fb, fo, 30.0, stat="p95_ms").retention == pytest.approx(1.0)
    assert speedup(fb, fo, 30.0) == pytest.approx(2.0, rel=1e-3)


def test_verdict_gates_response_time_and_counts_drops():
    rep = report(BASE, "field", 30.0, service_p95=5.0)  # fast inference, slow response
    assert evaluate(rep, Budget("b", 30.0, p95_ms=25.0)).status == "NO-GO"
    assert evaluate(rep, Budget("b", 30.0, p99_ms=35.0)).status == "GO"

    rep.tiers[0].response = None  # a schema 1.0 report
    assert evaluate(rep, Budget("b", 30.0, p99_ms=35.0)).status == "INCOMPLETE"

    dropped = overloaded_tier(deadline_ms=1000.0, drop_late=True)  # nothing late, nothing dropped
    assert dropped.deadline_misses == 0 and dropped.dropped == 0
    late = overloaded_tier(deadline_ms=32.0, drop_late=True)
    rep2 = RunReport(Variant("m", "b", "p", "fp32"), "field", [late])
    v = evaluate(rep2, Budget("b", 100.0, max_miss_rate=0.45))
    assert v.status == "NO-GO"  # 40% late alone would pass; 40% late + 50% dropped does not
