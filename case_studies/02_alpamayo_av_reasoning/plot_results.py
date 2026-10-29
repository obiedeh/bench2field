"""Plots and the results table for case study 02, from the committed run files only.

Reads every ``runs/decisions_<board>_<variant>.json`` and
``runs/profile_<board>_<variant>.json`` that exists and writes SVG figures to
``report/`` plus a Markdown results table to stdout. Nothing is typed in by
hand: if a number is in the README it came through this script from a run
file, and tests/test_alpamayo_results.py checks that the README table matches
what this script prints.

Figures (each written only when the runs it needs exist):

    report/latency_per_clip.svg     median latency per clip, one bar per board
                                    and variant, every repeat drawn as a dot
    report/stage_split.svg          where a median decision's time goes:
                                    time to first token (vision encoding and
                                    prefill), decode, trajectory sampling, glue
    report/tokens_vs_latency.svg    generated reasoning tokens against latency,
                                    one point per decision
    report/memory.svg               GPU memory after load and at the peak of a
                                    decision, per board and variant
    report/profile_timeline.svg     the profiled decision's NVML power and
                                    utilisation over time, decode tokens marked
    report/power_per_decision.svg   mean board or GPU power during each decision
                                    (Thor: VIN via tegrastats; 5090: NVML GPU
                                    power), where the run recorded it

Colour: one fixed hue per board and variant (5090 sdpa blue, 5090 flash aqua,
Thor sdpa orange), the validated three-slot categorical palette; stage
segments reuse the same three plus a neutral grey for glue. Text is never
drawn in a series colour. One axis per panel, never two scales.

    env -u PYTHONPATH .venv/bin/python plot_results.py            # figures + table
    env -u PYTHONPATH .venv/bin/python plot_results.py --table    # table only
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
REPORT = HERE / "report"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"
SERIES = {  # fixed slot per (board, variant); never cycled
    ("bench-5090", "sdpa"): ("#2a78d6", "RTX 5090, sdpa"),
    ("bench-5090", "flash"): ("#1baf7a", "RTX 5090, flash-attn"),
    ("bench-thor", "sdpa"): ("#eb6834", "Jetson AGX Thor, sdpa"),
}
STAGES = [("ttft", "#2a78d6", "first token (vision + prefill)"), ("decode", "#1baf7a", "decode"),
          ("trajectory", "#eb6834", "trajectory sampling"), ("glue", "#b5b4ae", "glue")]


def load_runs(prefix: str) -> list[dict[str, Any]]:
    out = []
    for p in sorted(RUNS.glob(f"{prefix}_*.json")):
        rec = json.loads(p.read_text())
        if rec.get("input") == "dataset":
            rec["_path"] = p
            out.append(rec)
    return out


def style(ax: Any, ylabel: str = "", xlabel: str = "") -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)


def fig(w: float = 7.2, h: float = 3.6, n: int = 1, sharex: bool = False) -> tuple[Any, Any]:
    f, axes = plt.subplots(n, 1, figsize=(w, h), facecolor=SURFACE, sharex=sharex)
    return f, axes


def save(f: Any, name: str) -> None:
    REPORT.mkdir(exist_ok=True)
    f.tight_layout()
    f.savefig(REPORT / name, facecolor=SURFACE)
    f.savefig((REPORT / name).with_suffix(".png"), facecolor=SURFACE, dpi=160)  # the README embeds the SVG; the PNG is for viewers without SVG
    plt.close(f)
    print(f"wrote report/{name} (+ .png)", file=sys.stderr)


def stage_medians(rec: dict[str, Any]) -> dict[str, float]:
    ds = rec["decisions"]
    ttft = statistics.median(d["ttft_s"] for d in ds if d["ttft_s"])
    rollout = statistics.median(d["rollout_s"] for d in ds)
    traj = statistics.median(d["trajectory_s"] for d in ds)
    total = statistics.median(d["latency_total_s"] for d in ds)
    return {"ttft": ttft, "decode": max(rollout - ttft, 0.0), "trajectory": traj, "glue": max(total - rollout - traj, 0.0)}


def plot_latency_per_clip(runs: list[dict[str, Any]]) -> None:
    clips = runs[0]["protocol"]["clips"]
    f, ax = fig(8.4, 3.8)
    n = len(runs)
    width = 0.8 / n
    for i, rec in enumerate(runs):
        color, name = SERIES[(rec["board"], rec["variant"])]
        xs = [c + (i - (n - 1) / 2) * width for c in range(len(clips))]
        meds, dots = [], []
        for clip in clips:
            vals = [d["latency_total_s"] for d in rec["decisions"] if d["clip_id"] == clip]
            meds.append(statistics.median(vals) if vals else 0)
            dots.append(vals)
        ax.bar(xs, meds, width=width * 0.92, color=color, label=name, linewidth=0, zorder=2)
        for x, vals in zip(xs, dots):
            ax.scatter([x] * len(vals), vals, s=14, color=SURFACE, edgecolor=INK, linewidth=0.8, zorder=3)
    ax.set_xticks(range(len(clips)))
    ax.set_xticklabels([c[:8] for c in clips], fontsize=8)
    style(ax, "seconds per decision", "clip (first 8 characters of the id)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    ax.set_title("Latency per decision: median bar, every repeat as a dot", color=INK, fontsize=10, loc="left")
    save(f, "latency_per_clip.svg")


def plot_stage_split(runs: list[dict[str, Any]]) -> None:
    f, ax = fig(7.2, 1.2 + 0.7 * len(runs))
    for i, rec in enumerate(runs):
        meds = stage_medians(rec)
        left = 0.0
        for key, color, label in STAGES:
            v = meds[key]
            ax.barh(i, v, left=left, color=color, height=0.55, linewidth=0, label=label if i == 0 else None, zorder=2)
            if v > 0.12 * sum(meds.values()):
                ax.text(left + v / 2, i, f"{v:.2f} s", ha="center", va="center", fontsize=8, color=INK)
            left += v + 0.004 * sum(meds.values())
    ax.set_yticks(range(len(runs)))
    ax.set_yticklabels([SERIES[(r["board"], r["variant"])][1] for r in runs], fontsize=9)
    ax.invert_yaxis()
    style(ax, "", "seconds (medians over all decisions)")
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.yaxis.grid(False)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.35), labelcolor=INK2)
    ax.set_title("Where a decision's time goes", color=INK, fontsize=10, loc="left")
    save(f, "stage_split.svg")


def plot_tokens_vs_latency(runs: list[dict[str, Any]]) -> None:
    f, ax = fig(6.4, 3.6)
    for rec in runs:
        color, name = SERIES[(rec["board"], rec["variant"])]
        ax.scatter([d["generated_tokens"] for d in rec["decisions"]], [d["latency_total_s"] for d in rec["decisions"]],
                   s=26, color=color, edgecolor=SURFACE, linewidth=0.8, label=name, zorder=2)
    style(ax, "seconds per decision", "generated reasoning tokens")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    ax.set_title("Latency follows the length of the reasoning trace", color=INK, fontsize=10, loc="left")
    save(f, "tokens_vs_latency.svg")


def plot_memory(runs: list[dict[str, Any]]) -> None:
    f, ax = fig(6.4, 3.2)
    names = [SERIES[(r["board"], r["variant"])][1] for r in runs]
    after = [r["load"]["memory_after_load"]["torch_allocated_gib"] for r in runs]
    peak = [r["summary"]["peak_torch_allocated_gib_max"] for r in runs]
    xs = range(len(runs))
    ax.bar([x - 0.2 for x in xs], after, width=0.38, color="#9ec5f4", label="after load", linewidth=0, zorder=2)
    ax.bar([x + 0.2 for x in xs], peak, width=0.38, color="#2a78d6", label="peak during a decision", linewidth=0, zorder=2)
    for x, a, p in zip(xs, after, peak):
        ax.text(x - 0.2, a, f"{a:.1f}", ha="center", va="bottom", fontsize=8, color=INK)
        ax.text(x + 0.2, p, f"{p:.1f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(names, fontsize=9)
    style(ax, "GiB allocated by torch")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    ax.set_title("GPU memory, bf16 weights", color=INK, fontsize=10, loc="left")
    save(f, "memory.svg")


def plot_profile_timeline(prof: dict[str, Any]) -> None:
    samples = prof["telemetry"]["samples"]
    if not samples:
        return
    w0, w1 = prof["window"]["start"], prof["window"]["end"]
    t = [s["t"] - w0 for s in samples]
    color, name = SERIES[(prof["board"], prof["variant"])]
    f, (a1, a2) = fig(7.2, 4.2, n=2, sharex=True)
    a1.plot(t, [s.get("power_gpu_w") for s in samples], color=color, linewidth=2)
    a2.plot(t, [s.get("gpu_util_pct") for s in samples], color=color, linewidth=2)
    for a in (a1, a2):
        a.axvspan(0, w1 - w0, color="#e6e5e1", alpha=0.6, zorder=0)
    style(a1, "GPU power (W)")
    style(a2, "GPU utilisation (%)", "seconds (shaded: the profiled decision)")
    a1.set_title(f"{name}: one decision under the profiler, {prof['generated_tokens']} tokens", color=INK, fontsize=10, loc="left")
    save(f, "profile_timeline.svg")


def plot_power(runs: list[dict[str, Any]]) -> None:
    rows = [(r, [d["power"]["mean_w"] for d in r["decisions"] if d.get("power")]) for r in runs]
    rows = [(r, v) for r, v in rows if v]
    if not rows:
        return
    f, ax = fig(6.4, 3.2)
    for i, (r, vals) in enumerate(rows):
        color, name = SERIES[(r["board"], r["variant"])]
        ax.bar(i, statistics.median(vals), width=0.55, color=color, linewidth=0, zorder=2)
        ax.scatter([i] * len(vals), vals, s=14, color=SURFACE, edgecolor=INK, linewidth=0.8, zorder=3)
        ax.text(i, statistics.median(vals), f"{statistics.median(vals):.0f} W", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([SERIES[(r["board"], r["variant"])][1] for r, _ in rows], fontsize=9)
    style(ax, "mean watts during a decision")
    ax.set_title("Power per decision: median bar, every decision as a dot (5090: GPU rail via NVML; Thor: VIN via tegrastats)", color=INK, fontsize=9, loc="left")
    save(f, "power_per_decision.svg")


def results_table(runs: list[dict[str, Any]]) -> str:
    cols = [SERIES[(r["board"], r["variant"])][1] for r in runs]
    head = "| Metric | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols) + "\n"
    def row(name: str, fn: Any) -> str:
        return f"| {name} | " + " | ".join(fn(r) for r in runs) + " |\n"
    def power(r: dict[str, Any]) -> str:
        vals = [d["power"]["mean_w"] for d in r["decisions"] if d.get("power")]
        return f"{statistics.median(vals):.0f} W" if vals else "n/a"
    def energy(r: dict[str, Any]) -> str:
        vals = [d["power"]["mean_w"] * d["latency_total_s"] for d in r["decisions"] if d.get("power")]
        return f"{statistics.median(vals):.0f} J" if vals else "n/a"
    s = head
    s += row("Model load time", lambda r: f"{r['load']['load_total_s']:.1f} s")
    s += row("Latency per decision, median", lambda r: f"{r['summary']['latency_median_s']:.2f} s")
    s += row("Latency per decision, p95", lambda r: f"{r['summary']['latency_p95_s']:.2f} s")
    s += row("Time to first token, median", lambda r: f"{r['summary']['ttft_median_s']:.2f} s")
    s += row("Decode rate, median", lambda r: f"{r['summary']['decode_tokens_per_s_median']:.1f} tok/s")
    s += row("Reasoning tokens, median", lambda r: f"{r['summary']['generated_tokens_median']:.0f}")
    s += row("Decisions per minute", lambda r: f"{r['summary']['decisions_per_minute']:.1f}")
    s += row("GPU memory after load", lambda r: f"{r['load']['memory_after_load']['torch_allocated_gib']:.1f} GiB")
    s += row("Peak GPU memory during a decision", lambda r: f"{r['summary']['peak_torch_allocated_gib_max']:.1f} GiB")
    s += row("Power during a decision, median", power)
    s += row("Energy per decision, median", energy)
    s += row("Decisions", lambda r: f"{r['summary']['n_decisions']} ({len(r['protocol']['clips'])} clips x {r['protocol']['repeats']})")
    return s


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--table", action="store_true", help="print the Markdown results table only")
    ap.add_argument("--runs", type=Path, default=None, help="alternative runs directory (for smoke output)")
    args = ap.parse_args(argv)
    global RUNS, REPORT
    if args.runs:
        RUNS = args.runs
        REPORT = args.runs / "report"
    runs = load_runs("decisions")
    if args.runs:  # smoke output is labelled synthetic; allow it only when pointed at explicitly
        runs = [dict(json.loads(p.read_text()), _path=p) for p in sorted(RUNS.glob("decisions_*.json"))]
    runs = [r for r in runs if (r["board"], r["variant"]) in SERIES]
    if not runs:
        print("no decision runs found", file=sys.stderr)
        return 1
    print(results_table(runs))
    if args.table:
        return 0
    plot_latency_per_clip(runs)
    plot_stage_split(runs)
    plot_tokens_vs_latency(runs)
    plot_memory(runs)
    plot_power(runs)
    profs = [dict(json.loads(p.read_text())) for p in sorted(RUNS.glob("profile_*.json"))]
    for prof in profs:
        if (prof["board"], prof["variant"]) in SERIES:
            plot_profile_timeline(prof)
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
