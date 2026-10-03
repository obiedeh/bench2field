"""Sweeps: the repeats of several variants, run alternately.

docs/METHODOLOGY.md asks for at least three repeats of every comparison,
alternating variants (A, B, A, B, A, B) so that slow drift, such as the
device warming up or a background job starting, does not favour one of them.
A sweep does that from one small YAML file and leaves one report per run plus
a manifest that records the order they ran in.

Every run is a separate `b2f run` process, so no state (GPU memory arenas,
thread pools, a warm cache) carries over from one variant to the next.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .metrics import DEFAULT_STAT, repeat_stat
from .provenance import git_state
from .schema import ENV_BENCH_IDLE, RunReport, describe_platform, validate_environment

Runner = Callable[[list[str]], int]
_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass
class SweepVariant:
    label: str                 # short name used in file names, e.g. "fp32"
    model: str                 # path to the .onnx file
    provider: str = "cpu"
    precision: str = "fp32"
    technique: str = "baseline"
    name: str | None = None    # model name for the report; defaults to the file stem


@dataclass
class SweepConfig:
    name: str
    variants: list[SweepVariant]
    repeats: int = 3
    environment: str = ENV_BENCH_IDLE
    tiers_hz: list[float] = field(default_factory=lambda: [10.0, 30.0, 100.0])
    duration_s: float = 60.0
    warmup_s: float = 5.0
    deadline_ms: float = 33.3
    drop_late: bool = False
    cooldown_c: float | None = None
    replay: str | None = None  # load profile to replay during every run
    only: str | None = None    # restrict replay to these stressors
    stopped: list[str] = field(default_factory=list)  # what was shut down for the sweep, for the reports
    no_spin: bool = False      # onnxruntime threads sleep between runs instead of spin-waiting

    def __post_init__(self) -> None:
        if not _LABEL.match(self.name):
            raise ValueError(f"sweep name {self.name!r} must be letters, digits, '_', '.' or '-'")
        if self.repeats < 1:
            raise ValueError("repeats must be at least 1")
        if not self.variants:
            raise ValueError("a sweep needs at least one variant")
        labels = [v.label for v in self.variants]
        for label in labels:
            if not _LABEL.match(label):
                raise ValueError(f"variant label {label!r} must be letters, digits, '_', '.' or '-'")
        if len(set(labels)) != len(labels):
            raise ValueError(f"variant labels must be unique, got {labels}")
        validate_environment(self.environment)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> SweepConfig:
        d = dict(d)
        try:
            d["variants"] = [SweepVariant(**v) for v in d.get("variants") or []]
            return cls(**d)
        except TypeError as exc:  # unknown or missing key
            raise ValueError(f"bad sweep config: {exc}") from exc

    @classmethod
    def from_yaml(cls, path: str | Path) -> SweepConfig:
        import yaml

        return cls.from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def plan(cfg: SweepConfig) -> list[tuple[int, SweepVariant]]:
    """(repeat, variant) in run order: every variant once per repeat, so two
    variants run A, B, A, B, A, B rather than A, A, A, B, B, B."""
    return [(r, v) for r in range(1, cfg.repeats + 1) for v in cfg.variants]


def report_path(out_dir: str | Path, variant: SweepVariant, repeat: int) -> Path:
    return Path(out_dir) / f"{variant.label}_r{repeat}.json"


def run_argv(cfg: SweepConfig, variant: SweepVariant, repeat: int, order: int, out: Path) -> list[str]:
    """Arguments for the `b2f run` that makes one run of the sweep."""
    argv = [
        "run", variant.model, "--provider", variant.provider, "--precision", variant.precision,
        "--technique", variant.technique, "--tiers", ",".join(f"{h:g}" for h in cfg.tiers_hz),
        "--duration", f"{cfg.duration_s:g}", "--warmup", f"{cfg.warmup_s:g}",
        "--deadline-ms", f"{cfg.deadline_ms:g}", "--environment", cfg.environment,
        "--sweep", cfg.name, "--sweep-label", variant.label,
        "--repeat", str(repeat), "--order", str(order), "--out", str(out),
    ]
    if variant.name:
        argv += ["--name", variant.name]
    if cfg.drop_late:
        argv.append("--drop-late")
    if cfg.no_spin:
        argv.append("--no-spin")
    if cfg.cooldown_c is not None:
        argv += ["--cooldown-c", f"{cfg.cooldown_c:g}"]
    for what in cfg.stopped:
        argv += ["--stopped", what]
    if cfg.replay:
        argv += ["--replay", cfg.replay]
        if cfg.only:
            argv += ["--only", cfg.only]
    return argv


def subprocess_runner(argv: list[str], env: Mapping[str, str] | None = None) -> int:
    return subprocess.run([sys.executable, "-m", "bench2field.cli", *argv],
                          env=None if env is None else dict(env)).returncode


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_sweep(cfg: SweepConfig, out_dir: str | Path, runner: Runner | None = None,
              resume: bool = False, config_file: str | None = None) -> dict[str, Any]:
    """Run the sweep and return its manifest, also written to
    <out_dir>/sweep_<name>.json after every run so a sweep that stops half
    way still says what it did.

    Existing reports are never overwritten: without `resume` their presence
    is an error, with it those runs are skipped.
    """
    runner = runner or subprocess_runner
    out_dir = Path(out_dir)
    runs = plan(cfg)
    existing = [p for r, v in runs if (p := report_path(out_dir, v, r)).exists()]
    if existing and not resume:
        raise ValueError(f"{existing[0]} already exists; use a new --out-dir, or --resume to keep "
                         "finished runs and do the rest")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"sweep_{cfg.name}.json"
    manifest: dict[str, Any] = {
        "sweep": cfg.name, "config_file": config_file, "config": asdict(cfg),
        "host": describe_platform().get("host"), "git": git_state(), "started_at": _now(), "finished_at": None,
        "complete": False, "runs": [],
    }

    def save() -> None:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    for order, (repeat, variant) in enumerate(runs, start=1):
        out = report_path(out_dir, variant, repeat)
        entry: dict[str, Any] = {"order": order, "label": variant.label, "repeat": repeat,
                                 "file": out.name, "started_at": _now()}
        if out.exists():
            entry["skipped"] = "report already existed (resume)"
        else:
            print(f"[{order}/{len(runs)}] {variant.label} repeat {repeat}", flush=True)
            code = runner(run_argv(cfg, variant, repeat, order, out))
            if code != 0 or not out.exists():
                entry["failed"] = f"b2f run exited {code}"
                manifest["runs"].append(entry)
                save()
                raise RuntimeError(f"sweep {cfg.name}: {variant.label} repeat {repeat} failed "
                                   f"(b2f run exited {code}); {order - 1} of {len(runs)} runs done")
        entry["run_id"] = RunReport.load(out).run_id
        entry["finished_at"] = _now()
        manifest["runs"].append(entry)
        save()

    manifest["finished_at"], manifest["complete"] = _now(), True
    save()
    return manifest


def summarize(cfg: SweepConfig, out_dir: str | Path, stat: str = DEFAULT_STAT) -> str:
    """Median and range of `stat` across repeats, per variant and tier."""
    rows = [f"{cfg.name}: {stat} across {cfg.repeats} repeat{'s' if cfg.repeats != 1 else ''}",
            f"{'variant':<16}{'Hz':>6}{'median':>10}{'lo':>10}{'hi':>10}  ms"]
    for v in cfg.variants:
        reports = [RunReport.load(report_path(out_dir, v, r)) for r in range(1, cfg.repeats + 1)]
        for hz in cfg.tiers_hz:
            s = repeat_stat(reports, hz, stat)
            rows.append(f"{v.label:<16}{hz:>6g}{s.median:>10.3f}{s.lo:>10.3f}{s.hi:>10.3f}")
    return "\n".join(rows)
