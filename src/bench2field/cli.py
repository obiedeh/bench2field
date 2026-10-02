"""Command line: b2f <command>.

  b2f run          benchmark an ONNX model (optionally under a replayed field load)
  b2f record-load  record a field-load profile on the robot
  b2f retention    field retention of an optimization (4 reports)
  b2f attribute    split the bench-to-field gap across stressors
  b2f validity     check a replay against the real field run
  b2f verdict      go/no-go against a deployment budget
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

from .schema import ENV_BENCH_IDLE, ENV_REPLAY_PREFIX, RunReport, Variant


def _cmd_run(a: argparse.Namespace) -> int:
    from .backends.onnxruntime import OnnxRuntimeBackend, OrtOptions
    from .loadreplay import LoadProfile, Replay
    from .runner import RunConfig, run
    from .telemetry import NullSampler, auto_sampler

    opts = OrtOptions(provider=a.provider, precision=a.precision,
                      intra_op_threads=a.intra_threads, inter_op_threads=a.inter_threads,
                      allow_spinning=not a.no_spin)
    be = OnnxRuntimeBackend(a.model, opts)
    feeds = be.synthetic_input(batch=a.batch)
    sampler = NullSampler() if a.no_telemetry else auto_sampler()

    env = a.environment
    replay_ctx: contextlib.AbstractContextManager = contextlib.nullcontext()
    extra = be.describe()
    if a.replay:
        prof = LoadProfile.load(a.replay)
        only = a.only.split(",") if a.only else None
        replay_ctx = Replay(prof, sampler, only=only)
        suffix = "" if only is None else "+" + "+".join(sorted(only))
        env = f"{ENV_REPLAY_PREFIX}{prof.name}{suffix}"

    variant = Variant(model=a.name or Path(a.model).stem, backend=be.name,
                      provider=be.provider(), precision=a.precision, technique=a.technique)
    cfg = RunConfig(tiers_hz=[float(x) for x in a.tiers.split(",")], duration_s=a.duration,
                    warmup_s=a.warmup, deadline_ms=a.deadline_ms, cooldown_max_c=a.cooldown_c)
    with replay_ctx as rc:
        if rc is not None:
            extra |= rc.describe()
        report = run(variant, env, be.infer, lambda i: feeds, cfg, sampler, extra)
    out = report.save(a.out or f"reports/{report.run_id}.json")
    for t in report.tiers:
        print(f"{t.target_hz:>7g} Hz  p50 {t.latency.p50_ms:.3f}  p95 {t.latency.p95_ms:.3f}  "
              f"p99 {t.latency.p99_ms:.3f} ms  misses {t.deadline_misses}")
    print(f"wrote {out}")
    return 0


def _cmd_record(a: argparse.Namespace) -> int:
    from .loadreplay import record
    from .telemetry import auto_sampler

    prof = record(a.name, auto_sampler(a.interval), a.duration, a.notes)
    print(json.dumps(prof.targets | {"soak_temp_c": prof.soak_temp_c}, indent=2))
    print(f"wrote {prof.save(a.out or f'profiles/{a.name}.json')}")
    return 0


def _cmd_retention(a: argparse.Namespace) -> int:
    from .metrics import field_retention

    r = field_retention(*(RunReport.load(p) for p in
                          (a.bench_base, a.bench_opt, a.field_base, a.field_opt)),
                        target_hz=a.hz, stat=a.stat)
    print(r.summary())
    return 0


def _cmd_attribute(a: argparse.Namespace) -> int:
    from .metrics import attribute_gap

    runs = []
    for item in a.stressor:
        name, path = item.split("=", 1)
        runs.append((name, RunReport.load(path)))
    at = attribute_gap(RunReport.load(a.idle), RunReport.load(a.field), runs, a.hz, a.stat)
    print(f"{a.stat} @ {a.hz:g} Hz: idle {at.idle_ms:.3f} ms -> field {at.field_ms:.3f} ms "
          f"(gap {at.total_gap_ms:+.3f} ms)")
    for k, v in at.shares().items():
        print(f"  {k:<26}{v:>7.0%}")
    return 0


def _cmd_validity(a: argparse.Namespace) -> int:
    from .metrics import replay_validity

    v = replay_validity(RunReport.load(a.replay), RunReport.load(a.field), a.hz, a.stat, a.tolerance)
    print(f"replay {v.replay_ms:.3f} ms vs field {v.field_ms:.3f} ms: error {v.rel_error:.1%} "
          f"(tolerance {v.tolerance:.0%}) -> {'VALID' if v.valid else 'NOT VALID'}")
    return 0 if v.valid else 1


def _cmd_verdict(a: argparse.Namespace) -> int:
    from .verdict import Budget, evaluate

    v = evaluate(RunReport.load(a.report), Budget.from_yaml(a.budget))
    print(v.table())
    return 0 if v.passed else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="b2f", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="benchmark an ONNX model")
    r.add_argument("model")
    r.add_argument("--name")
    r.add_argument("--provider", default="cpu", help="cpu|cuda|tensorrt|migraphx|rocm")
    r.add_argument("--precision", default="fp32")
    r.add_argument("--technique", default="baseline")
    r.add_argument("--tiers", default="10,30,100", help="comma-separated Hz")
    r.add_argument("--duration", type=float, default=60.0)
    r.add_argument("--warmup", type=float, default=5.0)
    r.add_argument("--deadline-ms", type=float, default=33.3)
    r.add_argument("--batch", type=int, default=1)
    r.add_argument("--intra-threads", type=int, default=0)
    r.add_argument("--inter-threads", type=int, default=0)
    r.add_argument("--no-spin", action="store_true")
    r.add_argument("--cooldown-c", type=float)
    r.add_argument("--environment", default=ENV_BENCH_IDLE, help="bench-idle or field")
    r.add_argument("--replay", help="load profile JSON to replay during the run")
    r.add_argument("--only", help="replay only these stressors: thermal,cpu,membw,gpu")
    r.add_argument("--no-telemetry", action="store_true")
    r.add_argument("--out")
    r.set_defaults(fn=_cmd_run)

    rec = sub.add_parser("record-load", help="record a field-load profile on the robot")
    rec.add_argument("name")
    rec.add_argument("--duration", type=float, default=300.0)
    rec.add_argument("--interval", type=float, default=0.5)
    rec.add_argument("--notes", default="")
    rec.add_argument("--out")
    rec.set_defaults(fn=_cmd_record)

    for cmd, fn, helptext in (("retention", _cmd_retention, "field retention of an optimization"),
                              ("attribute", _cmd_attribute, "split the bench-to-field gap"),
                              ("validity", _cmd_validity, "check a replay against the field")):
        s = sub.add_parser(cmd, help=helptext)
        s.add_argument("--hz", type=float, required=True)
        s.add_argument("--stat", default="p95_ms")
        s.set_defaults(fn=fn)
        if cmd == "retention":
            for x in ("bench_base", "bench_opt", "field_base", "field_opt"):
                s.add_argument(x)
        elif cmd == "attribute":
            s.add_argument("idle")
            s.add_argument("field")
            s.add_argument("--stressor", action="append", default=[], help="name=report.json")
        else:
            s.add_argument("replay")
            s.add_argument("field")
            s.add_argument("--tolerance", type=float, default=0.10)

    v = sub.add_parser("verdict", help="go/no-go against a deployment budget")
    v.add_argument("report")
    v.add_argument("budget")
    v.set_defaults(fn=_cmd_verdict)

    a = p.parse_args(argv)
    try:
        return a.fn(a)
    except (KeyError, ValueError, RuntimeError) as exc:
        print(f"b2f {a.cmd}: {exc.args[0] if exc.args else exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
