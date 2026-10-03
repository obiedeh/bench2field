"""Load everything `b2f report` draws from: the case study's report.yaml,
its sweeps (manifests and run reports), its pipeline profiles, the budget and
the camera-rate capture. Pure reading and arithmetic; no HTML here.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..metrics import repeat_stat
from ..schema import RunReport
from ..verdict import Budget

STAGES = ("decode", "preprocess", "h2d", "inference", "d2h", "postprocess")


@dataclass
class Sweep:
    name: str                       # directory name under runs/
    manifest: dict[str, Any]
    reports: dict[str, list[RunReport]]  # variant label -> repeats in order
    status: str                     # "baseline" | "reference" | "stopped" ...
    note: str = ""

    @property
    def tiers_hz(self) -> list[float]:
        return [float(h) for h in self.manifest["config"]["tiers_hz"]]

    @property
    def deadline_ms(self) -> float:
        return float(self.manifest["config"]["deadline_ms"])

    @property
    def platform(self) -> dict[str, Any]:
        first = next(iter(self.reports.values()))[0]
        return first.platform

    def tier_stat(self, variant: str, hz: float, stat: str = "response_p95_ms"):
        return repeat_stat(self.reports[variant], hz, stat)

    def tier_telemetry(self, variant: str, hz: float, key: str) -> float | None:
        vals = [r.tier(hz).telemetry.get(key, {}).get("p50") for r in self.reports[variant]]
        vals = [v for v in vals if v is not None]
        return float(np.median(vals)) if vals else None

    def misses(self, variant: str, hz: float) -> tuple[int, int]:
        reps = self.reports[variant]
        return (sum(r.tier(hz).deadline_misses for r in reps), sum(r.tier(hz).n_scheduled for r in reps))

    def saturated(self, variant: str, hz: float) -> bool:
        """The tier could not be served: the achieved rate fell short of the
        target and most frames were late."""
        reps = self.reports[variant]
        late, total = self.misses(variant, hz)
        actual = np.median([r.tier(hz).actual_hz for r in reps])
        return actual < 0.95 * hz and late > 0.5 * total


@dataclass
class Board:
    key: str
    label: str
    role: str
    baseline: Sweep | None
    references: list[Sweep]
    power_key: str = "power_board_w"


@dataclass
class Profile:
    name: str
    board: str
    label: str
    data: dict[str, Any]

    def stage(self, s: str, q: str = "p50_ms") -> float:
        return float(self.data["stages"][s][q])

    @property
    def total(self) -> dict[str, float]:
        return self.data["total_per_frame"]

    @property
    def host_ms(self) -> float:
        return sum(self.stage(s) for s in ("decode", "preprocess", "postprocess"))

    @property
    def copies_ms(self) -> float:
        return self.stage("h2d") + self.stage("d2h")


@dataclass
class CaseStudy:
    root: Path
    config: dict[str, Any]
    boards: dict[str, Board]
    sweeps: dict[str, Sweep]
    profiles: dict[str, Profile]
    budget: Budget
    camera_rate_hz: float | None     # mean of the capture's window averages
    camera_rate_samples: list[float]
    rate_hz: float                   # the swept tier closest to the camera rate
    deadline_ms: float
    warnings: list[str] = field(default_factory=list)

    @property
    def title(self) -> str:
        return self.config.get("title", self.root.name)


def parse_topic_hz(text: str) -> list[float]:
    """The 'average rate' values from a `ros2 topic hz` capture, first block
    only (a second `#`-headed block, e.g. for /scan, is ignored)."""
    lines = text.splitlines()
    i = 0
    while i < len(lines) and lines[i].startswith("#"):  # the capture's own header
        i += 1
    block = []
    for ln in lines[i:]:
        if ln.startswith("#"):  # a second captured topic begins
            break
        block.append(ln)
    return [float(x) for x in re.findall(r"average rate: ([\d.]+)", "\n".join(block))]


def load_sweep(runs_dir: Path, name: str, status: str, note: str = "") -> Sweep:
    d = runs_dir / name
    manifests = sorted(d.glob("sweep_*.json"))
    if not manifests:
        raise FileNotFoundError(f"no sweep manifest in {d}")
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    reports: dict[str, list[RunReport]] = {}
    for entry in manifest["runs"]:
        if "failed" in entry:
            continue
        reports.setdefault(entry["label"], []).append(RunReport.load(d / entry["file"]))
    for label in reports:
        reports[label].sort(key=lambda r: r.platform.get("sweep", {}).get("repeat", 0))
    return Sweep(name, manifest, reports, status, note)


def load_case_study(root: str | Path) -> CaseStudy:
    root = Path(root).resolve()
    cfg_path = root / "report.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"{cfg_path} not found; b2f report needs it to know which runs are baselines")
    import yaml

    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    runs = root / "runs"
    warnings: list[str] = []

    sweeps: dict[str, Sweep] = {}
    boards: dict[str, Board] = {}
    for key, b in config["boards"].items():
        baseline = None
        if b.get("baseline"):
            baseline = load_sweep(runs, b["baseline"], "baseline")
            sweeps[baseline.name] = baseline
        refs = []
        for name, note in (b.get("references") or {}).items():
            try:
                sw = load_sweep(runs, name, "reference", note)
            except FileNotFoundError as exc:
                warnings.append(str(exc))
                continue
            if not sw.manifest.get("complete", True):
                sw.status = "stopped"
            sweeps[sw.name] = sw
            refs.append(sw)
        power_key = "power_board_w"
        if baseline:
            tel0 = next(iter(baseline.reports.values()))[0].tiers[0].telemetry
            if "power_board_w" not in tel0 and "power_gpu_w" in tel0:
                power_key = "power_gpu_w"
        boards[key] = Board(key, b["label"], b["role"], baseline, refs, power_key)

    profiles: dict[str, Profile] = {}
    for name, p in (config.get("profiles") or {}).items():
        path = runs / f"{name}.json"
        if not path.exists():
            warnings.append(f"profile {path} not found")
            continue
        profiles[name] = Profile(name, p["board"], p["label"], json.loads(path.read_text(encoding="utf-8")))

    budget = Budget.from_yaml((root / config["budget"]).resolve())
    samples: list[float] = []
    cam_path = (root / config["camera_rate_capture"]).resolve() if config.get("camera_rate_capture") else None
    if cam_path and cam_path.exists():
        samples = parse_topic_hz(cam_path.read_text(encoding="utf-8"))
    elif cam_path:
        warnings.append(f"camera-rate capture {cam_path} not found")
    camera_rate = float(np.mean(samples)) if samples else None

    all_tiers = sorted({hz for sw in sweeps.values() for hz in sw.tiers_hz})
    if camera_rate is not None and all_tiers:
        rate_hz = min(all_tiers, key=lambda h: abs(h - camera_rate))
    else:
        rate_hz = float(budget.target_hz)
        warnings.append("no camera-rate capture; using the budget's target_hz as the headline tier")
    deadlines = {sw.deadline_ms for sw in sweeps.values()}
    deadline = deadlines.pop() if len(deadlines) == 1 else 1000.0 / budget.target_hz
    if len(deadlines) > 0:
        warnings.append("sweeps disagree on the deadline; using the budget's frame period")

    return CaseStudy(root, config, boards, sweeps, profiles, budget, camera_rate, samples, rate_hz, deadline, warnings)
