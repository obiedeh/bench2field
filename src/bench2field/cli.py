"""Command line: b2f <command>.

  b2f run          benchmark an ONNX model (optionally under a replayed field load)
  b2f sweep        repeats of several variants, run alternately (A, B, A, B, ...)
  b2f record-load  record a field-load profile on the robot
  b2f retention    field retention of an optimization (4 reports)
  b2f attribute    split the bench-to-field gap across stressors
  b2f validity     check a replay against the real field run
  b2f verdict      go/no-go against a deployment budget
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import json
import sys
from pathlib import Path

from . import background
from .metrics import DEFAULT_STAT
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

    if a.sweep:
        extra["sweep"] = {"name": a.sweep, "label": a.sweep_label, "repeat": a.repeat, "order": a.order}
    extra["background"] = background.snapshot(a.stopped)

    variant = Variant(model=a.name or Path(a.model).stem, backend=be.name,
                      provider=be.provider(), precision=a.precision, technique=a.technique)
    cfg = RunConfig(tiers_hz=[float(x) for x in a.tiers.split(",")], duration_s=a.duration,
                    warmup_s=a.warmup, deadline_ms=a.deadline_ms, cooldown_max_c=a.cooldown_c,
                    drop_late=a.drop_late)
    try:
        with replay_ctx as rc:
            if rc is not None:
                extra |= rc.describe()
            report = run(variant, env, be.infer, lambda i: feeds, cfg, sampler, extra)
    finally:
        sampler.close()
    out = report.save(a.out or f"reports/{report.run_id}.json")
    for t in report.tiers:
        print(f"{t.target_hz:>7g} Hz  response p50 {t.response.p50_ms:.3f}  p95 {t.response.p95_ms:.3f}  "
              f"p99 {t.response.p99_ms:.3f} ms  (service p95 {t.latency.p95_ms:.3f} ms)  "
              f"misses {t.deadline_misses}  dropped {t.dropped}")
    print(f"wrote {out}")
    return 0


def _cmd_sweep(a: argparse.Namespace) -> int:
    from .sweep import SweepConfig, run_sweep, summarize

    cfg = SweepConfig.from_yaml(a.config)
    cfg.stopped = [*cfg.stopped, *a.stopped]
    manifest = run_sweep(cfg, a.out_dir, resume=a.resume, config_file=a.config)
    print(summarize(cfg, a.out_dir, a.stat))
    print(f"wrote {len(manifest['runs'])} runs and sweep_{cfg.name}.json to {a.out_dir}")
    return 0


def _cmd_record(a: argparse.Namespace) -> int:
    from .loadreplay import record
    from .telemetry import auto_sampler

    sampler = auto_sampler(a.interval)
    try:
        prof = record(a.name, sampler, a.duration, a.notes)
    finally:
        sampler.close()
    print(json.dumps(prof.targets | {"soak_temp_c": prof.soak_temp_c}, indent=2))
    print(f"wrote {prof.save(a.out or f'profiles/{a.name}.json')}")
    return 0


def _cmd_retention(a: argparse.Namespace) -> int:
    from .metrics import field_retention

    r = field_retention(*(_load_group(spec) for spec in
                          (a.bench_base, a.bench_opt, a.field_base, a.field_opt)),
                        target_hz=a.hz, stat=a.stat)
    print(r.summary())
    return 0


def _load_group(spec: str) -> list[RunReport]:
    """Reports named by one argument: a file, a glob, or a comma-separated
    list of either. More than one report means repeats of the same run."""
    paths: list[str] = []
    for part in spec.split(","):
        part = part.strip()
        paths += sorted(glob.glob(part)) if glob.has_magic(part) else [part]
    if not paths:
        raise ValueError(f"no reports match {spec!r}")
    return [RunReport.load(p) for p in paths]


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


def build_parser() -> argparse.ArgumentParser:
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
    r.add_argument("--drop-late", action="store_true",
                   help="skip a frame already past its deadline when it would start "
                        "(camera-style); drops are counted separately from misses")
    r.add_argument("--batch", type=int, default=1)
    r.add_argument("--intra-threads", type=int, default=0)
    r.add_argument("--inter-threads", type=int, default=0)
    r.add_argument("--no-spin", action="store_true")
    r.add_argument("--cooldown-c", type=float)
    r.add_argument("--environment", default=ENV_BENCH_IDLE, help="bench-idle or field")
    r.add_argument("--replay", help="load profile JSON to replay during the run")
    r.add_argument("--only", help="replay only these stressors: thermal,cpu,membw,gpu")
    r.add_argument("--no-telemetry", action="store_true")
    r.add_argument("--stopped", action="append", default=[], metavar="WHAT",
                   help="something you shut down for this run, recorded in the report "
                        "(repeatable), e.g. 'docker container urban-edge-vllm'")
    r.add_argument("--out")
    r.add_argument("--sweep", help="set by b2f sweep: the sweep this run belongs to")
    r.add_argument("--sweep-label")
    r.add_argument("--repeat", type=int)
    r.add_argument("--order", type=int, help="set by b2f sweep: position in the sweep's run order")
    r.set_defaults(fn=_cmd_run)

    sw = sub.add_parser("sweep", help="repeats of several variants, run alternately")
    sw.add_argument("config", help="sweep YAML: name, variants, repeats, tiers (see configs/sweeps/)")
    sw.add_argument("--out-dir", required=True, help="one report per run plus the sweep manifest")
    sw.add_argument("--resume", action="store_true", help="keep reports already there, run the rest")
    sw.add_argument("--stopped", action="append", default=[], metavar="WHAT",
                   help="something you shut down on this machine for the sweep; recorded in every "
                        "report and in the manifest (repeatable)")
    sw.add_argument("--stat", default=DEFAULT_STAT, help="statistic for the summary table")
    sw.set_defaults(fn=_cmd_sweep)

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
        s.add_argument("--stat", default=DEFAULT_STAT,
                       help="response_{p50,p95,p99,max,mean}_ms (arrival to completion) "
                            "or {p50,p95,p99,max,mean}_ms (service time)")
        s.set_defaults(fn=fn)
        if cmd == "retention":
            for x in ("bench_base", "bench_opt", "field_base", "field_opt"):
                s.add_argument(x, help="report file, glob, or comma-separated list (repeats)")
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

    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        return a.fn(a)
    except (KeyError, ValueError, RuntimeError) as exc:
        print(f"b2f {a.cmd}: {exc.args[0] if exc.args else exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
