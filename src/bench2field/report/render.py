"""Render a CaseStudy as one self-contained HTML page: inline CSS and JS,
charts as inline SVG drawn here, no network, light and dark themes, readable
on a phone. Every figure on the page is computed from the loaded files.
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .. import __version__
from ..provenance import git_state
from .data import STAGES, Board, CaseStudy, Profile, Sweep

# Colour tokens are CSS variables so the same SVG reads in both themes.
SERIES = ["var(--c1)", "var(--c2)", "var(--c3)", "var(--c4)", "var(--c5)", "var(--c6)"]
STAGE_COLOURS = dict(zip(STAGES, SERIES))
STAGE_LABELS = {"decode": "decode", "preprocess": "preprocess", "h2d": "copy to GPU",
                "inference": "inference", "d2h": "copy back", "postprocess": "postprocess"}


def esc(x: Any) -> str:
    return html.escape(str(x))


def fmt(x: float | None, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    if x >= 1000:
        return f"{x:,.0f}"
    return f"{x:.{digits}f}"


# --- SVG helpers -------------------------------------------------------------

@dataclass
class Frame:
    """A chart area inside an SVG: maps data to pixels."""
    w: int
    h: int
    left: int = 56
    right: int = 16
    top: int = 16
    bottom: int = 36

    @property
    def x0(self) -> int:
        return self.left

    @property
    def x1(self) -> int:
        return self.w - self.right

    @property
    def y0(self) -> int:
        return self.h - self.bottom

    @property
    def y1(self) -> int:
        return self.top


def svg_open(fr: Frame, title: str) -> str:
    return (f'<svg class="chart" viewBox="0 0 {fr.w} {fr.h}" role="img" aria-label="{esc(title)}">'
            f"<title>{esc(title)}</title>")


def text(x: float, y: float, s: str, cls: str = "", anchor: str = "start", dy: str = "0") -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}" dy="{dy}">{esc(s)}</text>'


def line(x1: float, y1: float, x2: float, y2: float, cls: str = "", extra: str = "") -> str:
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="{cls}" {extra}/>'


def rect(x: float, y: float, w: float, h: float, fill: str, extra: str = "") -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" height="{max(h, 0):.1f}" fill="{fill}" {extra}/>'


def log_axis_ticks(lo: float, hi: float) -> list[float]:
    ticks = []
    e = math.floor(math.log10(lo))
    while 10 ** e <= hi * 1.001:
        for m in (1, 2, 5):
            v = m * 10 ** e
            if lo <= v <= hi:
                ticks.append(v)
        e += 1
    return ticks


def nice_ticks(hi: float, n: int = 4) -> list[float]:
    if hi <= 0:
        return [0]
    raw = hi / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min(s for s in (1, 2, 2.5, 5, 10) if s * mag >= raw) * mag
    return [i * step for i in range(int(hi / step) + 2) if i * step <= hi * 1.15]


# --- panels ------------------------------------------------------------------

def headline_bars(cs: CaseStudy) -> tuple[str, list[dict[str, Any]]]:
    """Model alone (baseline sweep at the camera rate) vs the full frame (same
    frames profile) per board, against the deadline."""
    rows = []
    for key, b in cs.boards.items():
        if not b.baseline or cs.config["headline_variant"] not in b.baseline.reports:
            continue
        st = b.baseline.tier_stat(cs.config["headline_variant"], cs.rate_hz)
        prof = next((p for n, p in cs.profiles.items() if n in cs.config.get("same_frames", []) and p.board == key), None)
        rows.append({"board": b.label, "key": key, "model_p95": st.median, "model_lo": st.lo, "model_hi": st.hi,
                     "frame_p95": prof.total["p95_ms"] if prof else None, "frame_p50": prof.total["p50_ms"] if prof else None,
                     "profile": prof.name if prof else None, "sweep": b.baseline.name})
    fr = Frame(720, 60 + 58 * len(rows), left=150, right=70, top=10, bottom=28)
    hi = max([r["frame_p95"] or 0 for r in rows] + [r["model_hi"] for r in rows] + [cs.deadline_ms]) * 1.08
    sx = lambda v: fr.x0 + (fr.x1 - fr.x0) * v / hi
    out = [svg_open(fr, "Model alone versus the full frame, per board")]
    for t in nice_ticks(hi):
        out.append(line(sx(t), fr.y1, sx(t), fr.y0, "grid"))
        out.append(text(sx(t), fr.y0 + 14, f"{t:g}", "tick", "middle"))
    out.append(text((fr.x0 + fr.x1) / 2, fr.h - 4, "milliseconds, p95", "axis", "middle"))
    for i, r in enumerate(rows):
        y = fr.y1 + 8 + i * 58
        out.append(text(fr.x0 - 8, y + 22, r["board"], "label", "end"))
        out.append(rect(sx(0), y, sx(r["model_p95"]) - sx(0), 16, "var(--c1)", 'rx="2"'))
        out.append(text(sx(r["model_p95"]) + 6, y + 12, f"model alone {fmt(r['model_p95'], 1)} ms", "value"))
        if r["frame_p95"] is not None:
            out.append(rect(sx(0), y + 20, sx(r["frame_p95"]) - sx(0), 16, "var(--c3)", 'rx="2"'))
            out.append(text(sx(r["frame_p95"]) + 6, y + 32, f"full frame {fmt(r['frame_p95'], 1)} ms", "value"))
    out.append(line(sx(cs.deadline_ms), fr.y1, sx(cs.deadline_ms), fr.y0, "deadline"))
    out.append(text(sx(cs.deadline_ms) + 4, fr.y1 + 10, f"deadline {cs.deadline_ms:g} ms", "deadline-label"))
    out.append("</svg>")
    return "".join(out), rows


def camera_rate_panel(cs: CaseStudy) -> str:
    s = cs.camera_rate_samples
    if not s:
        return "<p>No camera-rate capture.</p>"
    fr = Frame(720, 180, left=48, right=16, top=14, bottom=30)
    lo, hi = min(min(s), cs.budget.target_hz) - 2, max(max(s), cs.budget.target_hz) + 2
    sx = lambda i: fr.x0 + (fr.x1 - fr.x0) * i / max(1, len(s) - 1)
    sy = lambda v: fr.y0 - (fr.y0 - fr.y1) * (v - lo) / (hi - lo)
    out = [svg_open(fr, "Delivered camera rate over the capture")]
    for t in range(int(math.ceil(lo)), int(hi) + 1, 2):
        out.append(line(fr.x0, sy(t), fr.x1, sy(t), "grid"))
        out.append(text(fr.x0 - 6, sy(t), f"{t}", "tick", "end", "0.35em"))
    out.append(line(fr.x0, sy(cs.budget.target_hz), fr.x1, sy(cs.budget.target_hz), "deadline"))
    out.append(text(fr.x1, sy(cs.budget.target_hz) - 4, f"configured {cs.budget.target_hz:g} Hz", "deadline-label", "end"))
    pts = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(s))
    out.append(f'<polyline points="{pts}" class="series" style="stroke:var(--c1)"/>')
    out.append(line(fr.x0, sy(cs.camera_rate_hz), fr.x1, sy(cs.camera_rate_hz), "mean"))
    out.append(text(fr.x0 + 4, sy(cs.camera_rate_hz) - 4, f"delivered, mean {cs.camera_rate_hz:.1f} Hz", "value"))
    out.append(text((fr.x0 + fr.x1) / 2, fr.h - 6, f"{len(s)} window averages over the 60 s capture", "axis", "middle"))
    out.append("</svg>")
    return "".join(out)


def latency_vs_rate(cs: CaseStudy) -> str:
    """Response p95 against request rate for every board's baseline, log y,
    deadline line, saturated tiers marked."""
    fr = Frame(720, 340, left=58, right=16, top=20, bottom=40)
    series = []
    for i, (key, b) in enumerate(cs.boards.items()):
        if not b.baseline:
            continue
        for j, variant in enumerate((cs.config["headline_variant"], cs.config.get("reference_variant"))):
            if not variant or variant not in b.baseline.reports:
                continue
            pts = []
            for hz in b.baseline.tiers_hz:
                st = b.baseline.tier_stat(variant, hz)
                pts.append((hz, st.median, st.lo, st.hi, b.baseline.saturated(variant, hz)))
            series.append((f"{b.label}, {variant}", SERIES[i], j == 0, pts))
    ys = [v for _, _, _, pts in series for (_, m, lo, hi, _) in pts for v in (m, lo, hi)] + [cs.deadline_ms]
    ylo, yhi = 10 ** math.floor(math.log10(min(ys))), 10 ** math.ceil(math.log10(max(ys)))
    xs = sorted({hz for _, _, _, pts in series for (hz, *_r) in pts})
    xlo, xhi = min(xs) / 1.3, max(xs) * 1.3
    sx = lambda v: fr.x0 + (fr.x1 - fr.x0) * (math.log10(v) - math.log10(xlo)) / (math.log10(xhi) - math.log10(xlo))
    sy = lambda v: fr.y0 - (fr.y0 - fr.y1) * (math.log10(v) - math.log10(ylo)) / (math.log10(yhi) - math.log10(ylo))
    out = [svg_open(fr, "Response p95 against request rate")]
    for t in log_axis_ticks(ylo, yhi):
        out.append(line(fr.x0, sy(t), fr.x1, sy(t), "grid"))
        out.append(text(fr.x0 - 6, sy(t), f"{t:g}", "tick", "end", "0.35em"))
    for hz in xs:
        out.append(line(sx(hz), fr.y0, sx(hz), fr.y1, "grid"))
        out.append(text(sx(hz), fr.y0 + 14, f"{hz:g}", "tick", "middle"))
    out.append(text((fr.x0 + fr.x1) / 2, fr.h - 6, "request rate, Hz (log)", "axis", "middle"))
    out.append(text(12, (fr.y0 + fr.y1) / 2, "response p95, ms (log)", "axis", "middle",
                    ) .replace("<text ", f'<text transform="rotate(-90 12 {(fr.y0 + fr.y1) / 2:.1f})" ', 1))
    out.append(line(fr.x0, sy(cs.deadline_ms), fr.x1, sy(cs.deadline_ms), "deadline"))
    out.append(text(fr.x1, sy(cs.deadline_ms) - 4, f"deadline {cs.deadline_ms:g} ms", "deadline-label", "end"))
    out.append(line(sx(cs.rate_hz), fr.y0, sx(cs.rate_hz), fr.y1, "rate"))
    out.append(text(sx(cs.rate_hz) + 4, fr.y1 + 10, f"camera {cs.rate_hz:g} Hz", "deadline-label"))
    for name, colour, solid, pts in series:
        dash = "" if solid else 'stroke-dasharray="5 4"'
        path = " ".join(f"{sx(hz):.1f},{sy(m):.1f}" for hz, m, *_r in pts)
        out.append(f'<polyline points="{path}" class="series" style="stroke:{colour}" {dash}/>')
        for hz, m, lo, hi, sat in pts:
            out.append(line(sx(hz), sy(lo), sx(hz), sy(hi), "range", f'style="stroke:{colour}"'))
            if sat:
                out.append(f'<text x="{sx(hz):.1f}" y="{sy(m):.1f}" class="sat" text-anchor="middle" dy="0.35em" '
                           f'style="fill:{colour}">✕</text>')
            else:
                out.append(f'<circle cx="{sx(hz):.1f}" cy="{sy(m):.1f}" r="{4 if solid else 3}" fill="{colour}"/>')
    out.append("</svg>")
    legend = "".join(
        f'<span class="key"><i style="background:{c};{"" if solid else "opacity:.55"}"></i>{esc(n)}</span>'
        for n, c, solid, _ in series)
    return "".join(out) + f'<div class="legend">{legend}<span class="key">✕ saturated: the tier could not be served</span></div>'


def power_panels(cs: CaseStudy) -> str:
    """Per board: power per tier, plus the GPU rail or SM clock where the
    sampler reports one."""
    variant = cs.config["headline_variant"]
    cards = []
    for i, (key, b) in enumerate(cs.boards.items()):
        if not b.baseline or variant not in b.baseline.reports:
            continue
        tel0 = b.baseline.reports[variant][0].tiers[0].telemetry
        channels = [(b.power_key, "board power, W" if b.power_key == "power_board_w" else "GPU power, W")]
        if "power_vdd_gpu_w" in tel0:
            channels.append(("power_vdd_gpu_w", "GPU rail, W"))
        if "sm_clock_mhz" in tel0:
            channels.append(("sm_clock_mhz", "SM clock, MHz"))
        rows = []
        for hz in b.baseline.tiers_hz:
            st = b.baseline.tier_stat(variant, hz)
            rows.append((hz, st.median, [b.baseline.tier_telemetry(variant, hz, ch) for ch, _ in channels]))
        head = "".join(f"<th>{esc(l)}</th>" for _, l in channels)
        body = "".join(
            f"<tr><td>{hz:g} Hz</td><td>{fmt(m)}</td>" + "".join(f"<td>{fmt(v, 1 if 'clock' not in ch else 0)}</td>" for v, (ch, _) in zip(vals, channels)) + "</tr>"
            for hz, m, vals in rows)
        cards.append(f'<div class="card"><h4>{esc(b.label)}</h4><table><thead><tr><th>tier</th><th>p95, ms</th>{head}</tr></thead>'
                     f"<tbody>{body}</tbody></table></div>")
    return '<div class="cards">' + "".join(cards) + "</div>"


def thermal_repeats(cs: CaseStudy) -> str:
    field = next((b for b in cs.boards.values() if b.role == "field" and b.baseline), None)
    if not field:
        return "<p>No field board baseline.</p>"
    variant = cs.config["headline_variant"]
    reps = field.baseline.reports[variant]
    rows = []
    for r in reps:
        t = r.tier(cs.rate_hz)
        tel = t.telemetry
        rows.append((r.platform.get("sweep", {}).get("repeat", "?"), t.response.p95_ms, t.response.p50_ms,
                     tel.get("temp_tj_c", {}).get("p50"), tel.get("temp_tj_c", {}).get("peak"),
                     tel.get("gpu_util_pct", {}).get("p50"), r.platform.get("sweep", {}).get("order", "?")))
    body = "".join(f"<tr><td>{rep}</td><td>{order}</td><td>{fmt(p95)}</td><td>{fmt(p50)}</td><td>{fmt(tj, 1)}</td>"
                   f"<td>{fmt(tjp, 1)}</td><td>{fmt(g, 0)}</td></tr>" for rep, p95, p50, tj, tjp, g, order in rows)
    p95s = [r[1] for r in rows]
    swing = (max(p95s) - min(p95s)) / min(p95s) if p95s else 0
    return (f"<p>{esc(field.label)}, {esc(variant)} at {cs.rate_hz:g} Hz, one row per repeat of the baseline sweep. "
            f"Spread between repeats: <b>{swing:.0%}</b> of the fastest. Temperature and GPU load are shown for each "
            f"repeat as recorded; a difference between repeats is not attributed to either here, and the findings "
            f"say what is and is not established about its cause.</p>"
            f"<table><thead><tr><th>repeat</th><th>position in sweep</th><th>response p95, ms</th><th>p50</th>"
            f"<th>junction °C p50</th><th>peak</th><th>GPU load %</th></tr></thead><tbody>{body}</tbody></table>")


def stacked_stages(profiles: list[tuple[str, Profile]], deadline: float | None, title: str) -> str:
    fr = Frame(720, 50 + 44 * len(profiles), left=170, right=90, top=10, bottom=30)
    hi = max([p.total["p95_ms"] for _, p in profiles] + [deadline or 0]) * 1.08
    sx = lambda v: fr.x0 + (fr.x1 - fr.x0) * v / hi
    out = [svg_open(fr, title)]
    for t in nice_ticks(hi):
        out.append(line(sx(t), fr.y1, sx(t), fr.y0, "grid"))
        out.append(text(sx(t), fr.y0 + 14, f"{t:g}", "tick", "middle"))
    out.append(text((fr.x0 + fr.x1) / 2, fr.h - 4, "milliseconds per frame, p50 by stage; p95 total at right", "axis", "middle"))
    for i, (label, p) in enumerate(profiles):
        y = fr.y1 + 6 + i * 44
        out.append(text(fr.x0 - 8, y + 13, label, "label", "end"))
        x = 0.0
        for s in STAGES:
            w = p.stage(s)
            out.append(rect(sx(x), y, sx(x + w) - sx(x), 20, STAGE_COLOURS[s]))
            x += w
        out.append(text(sx(p.total["p95_ms"]) + 6, y + 14, f"p95 {fmt(p.total['p95_ms'], 1)}", "value"))
        out.append(line(sx(p.total["p95_ms"]), y - 2, sx(p.total["p95_ms"]), y + 22, "marker"))
    if deadline:
        out.append(line(sx(deadline), fr.y1, sx(deadline), fr.y0, "deadline"))
        out.append(text(sx(deadline) + 4, fr.y1 + 10, f"deadline {deadline:g} ms", "deadline-label"))
    out.append("</svg>")
    legend = "".join(f'<span class="key"><i style="background:{STAGE_COLOURS[s]}"></i>{STAGE_LABELS[s]}</span>' for s in STAGES)
    return "".join(out) + f'<div class="legend">{legend}</div>'


def stage_table(profiles: list[tuple[str, Profile]]) -> str:
    head = "".join(f"<th>{esc(l)}</th>" for l, _ in profiles)
    rows = []
    for s in STAGES:
        rows.append(f"<tr><td>{STAGE_LABELS[s]}</td>" + "".join(
            f"<td>{fmt(p.stage(s), 3)} / {fmt(p.stage(s, 'p95_ms'), 3)}</td>" for _, p in profiles) + "</tr>")
    rows.append("<tr class=\"total\"><td>total</td>" + "".join(
        f"<td>{fmt(p.total['p50_ms'], 3)} / {fmt(p.total['p95_ms'], 3)}</td>" for _, p in profiles) + "</tr>")
    rows.append("<tr><td>host stages (decode + preprocess + postprocess)</td>" + "".join(
        f"<td>{fmt(p.host_ms, 1)} ms, {p.host_ms / p.total['p50_ms']:.0%}</td>" for _, p in profiles) + "</tr>")
    rows.append("<tr><td>inference inside the frame / back to back</td>" + "".join(
        f"<td>{fmt(p.stage('inference'), 1)} / {fmt(p.data['session_run_numpy']['p50_ms'], 1)}</td>" for _, p in profiles) + "</tr>")
    return (f"<table><thead><tr><th>stage, p50 / p95 ms</th>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>")


def provenance_table(cs: CaseStudy) -> str:
    rows = []
    for key, b in cs.boards.items():
        for sw in ([b.baseline] if b.baseline else []) + b.references:
            p = sw.platform
            g = sw.manifest.get("git") or p.get("git") or {}
            commit = (g.get("commit") or "")[:8] or "not recorded"
            if g.get("recorded_afterwards"):
                commit += " (recorded afterwards)"
            elif g.get("dirty"):
                commit += " (dirty)"
            first = next(iter(sw.reports.values()))[0]
            bg = first.platform.get("background", {})
            containers = ", ".join(c["name"] for c in (bg.get("containers_running") or [])) or "none"
            stopped = "; ".join(bg.get("stopped_for_this_run") or []) or "nothing"
            tiers = ", ".join(f"{h:g}" for h in sw.tiers_hz)
            n_rep = sw.manifest["config"]["repeats"]
            status = {"baseline": "baseline", "reference": "reference", "stopped": "stopped, not a baseline"}[sw.status]
            rows.append(
                f"<tr class=\"{sw.status}\"><td>{esc(b.label)}</td><td><code>runs/{esc(sw.name)}/</code></td>"
                f"<td><b>{status}</b>{(' – ' + esc(sw.note)) if sw.note else ''}</td>"
                f"<td>{tiers} Hz × {n_rep}{'' if sw.manifest.get('complete', True) else ' (incomplete)'}</td>"
                f"<td>{'off' if p.get('allow_spinning') is False else 'on'}</td>"
                f"<td>ORT {esc(p.get('onnxruntime'))}, TRT {esc(p.get('tensorrt'))}, cuDNN {esc(p.get('cudnn'))}, CUDA {esc(p.get('cuda_runtime'))}</td>"
                f"<td>{esc(p.get('nvpmodel') or '–')}</td><td><code>{esc(commit)}</code></td>"
                f"<td>{esc(containers)}</td><td>{esc(stopped)}</td></tr>")
    for name, p in cs.profiles.items():
        d = p.data; pl = d["platform"]; g = pl.get("git") or {}
        commit = (g.get("commit") or "")[:8] or ("not recorded" if not g.get("recorded_afterwards") else "noted afterwards")
        bg = d.get("background", {})
        containers = ", ".join(c["name"] for c in (bg.get("containers_running") or [])) or "none"
        rows.append(
            f"<tr class=\"profile\"><td>{esc(cs.boards[p.board].label)}</td><td><code>runs/{esc(name)}.json</code></td>"
            f"<td>pipeline profile – {esc(p.label)}</td><td>{d['frames']['n']} frames, {esc(d['frames']['dir'])}</td>"
            f"<td>{'off' if not d.get('settings', {}).get('ort_allow_spinning', True) else 'on'}</td>"
            f"<td>ORT {esc(pl.get('onnxruntime'))}, TRT {esc(pl.get('tensorrt'))}, cuDNN {esc(pl.get('cudnn'))}, CUDA {esc(pl.get('cuda_runtime'))}</td>"
            f"<td>{esc(pl.get('nvpmodel') or '–')}</td><td><code>{esc(commit)}</code></td><td>{esc(containers)}</td><td>–</td></tr>")
    return ("<table class=\"prov\"><thead><tr><th>board</th><th>files</th><th>status</th><th>tiers × repeats</th>"
            "<th>ORT spinning</th><th>runtime</th><th>power mode</th><th>commit</th><th>containers up</th>"
            "<th>stopped for the run</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


# --- page --------------------------------------------------------------------

def title_html(title: str) -> str:
    """Page title in the lab's two accents: the model under test in blue, the field platform in green.

    "Case study 01: YOLOX-s on the ROSMASTER rover" -> blue "YOLOX-s", green "ROSMASTER rover".
    Titles without that shape are returned escaped and unaccented.
    """
    head, sep, rest = title.partition(": ")
    model, on, platform = rest.partition(" on the ")
    if not (sep and on and model and platform):
        return esc(title)
    return (f'{esc(head)}: <span class="ee-b">{esc(model)}</span> on the '
            f'<span class="ee-g">{esc(platform)}</span>')


CSS = """
:root{--bg:#202224;--fg:#eef1e8;--muted:#a3aa9c;--dim:#7c8378;--line:#393d3f;--line-strong:#4a4f52;--card:#181b1d;
--field:#b7f34a;--sim:#68b7ff;--signal:#ff9c59;--accent:#b7f34a;
--c1:#68b7ff;--c2:#ff9c59;--c3:#ff7a6b;--c4:#b7f34a;--c5:#c3a6ff;--c6:#8d948a;--deadline:#ff7a6b;--rate:#b7f34a;
--sans:"Helvetica Neue",Helvetica,Arial,sans-serif;--mono:ui-monospace,"SFMono-Regular",Consolas,"Liberation Mono",Menlo,monospace;color-scheme:dark}
:root[data-theme="light"]{--bg:#f6f7f3;--fg:#1a1c1e;--muted:#5d6459;--dim:#7c8378;--line:#d8dbd2;--line-strong:#b9bdb3;--card:#ffffff;
--field:#4d7c0f;--sim:#1f6fd1;--signal:#c2560c;--accent:#4d7c0f;
--c1:#1f6fd1;--c2:#c2560c;--c3:#c62828;--c4:#4d7c0f;--c5:#6d4bc4;--c6:#7c8378;--deadline:#c62828;--rate:#4d7c0f;color-scheme:light}
@media print{:root{--bg:#ffffff;--fg:#1a1c1e;--muted:#5d6459;--line:#d8dbd2;--line-strong:#b9bdb3;--card:#ffffff;--field:#4d7c0f;--sim:#1f6fd1;--signal:#c2560c;--accent:#4d7c0f;
--c1:#1f6fd1;--c2:#c2560c;--c3:#c62828;--c4:#4d7c0f;--c5:#6d4bc4;--c6:#7c8378;--deadline:#c62828;--rate:#4d7c0f;color-scheme:light}button.theme{display:none}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 var(--sans)}
a{color:var(--sim);text-decoration:none}a:hover{text-decoration:underline}
.ee-band{border-bottom:1px solid var(--line);background:var(--card)}
.ee-band .in{max-width:960px;margin:0 auto;padding:10px 16px;display:flex;flex-wrap:wrap;gap:8px 18px;align-items:center;justify-content:space-between;font-size:.85rem}
.ee-mark{display:inline-flex;align-items:center;gap:8px;color:var(--fg);font-weight:600;letter-spacing:-.01em}.ee-mark:hover{text-decoration:none}
.ee-mark i{display:inline-block;width:14px;height:14px;background:linear-gradient(135deg,var(--field) 0 50%,var(--sim) 50% 100%);border-radius:2px}
.ee-mark span{color:var(--muted);font-weight:400}.ee-band nav{display:flex;gap:16px}.ee-band nav a{color:var(--muted)}.ee-band nav a:hover{color:var(--fg)}
main{max-width:960px;margin:0 auto;padding:24px 16px 48px}
.eyebrow{color:var(--field);font:600 12px/1.4 var(--mono);letter-spacing:.08em;text-transform:uppercase;margin:0 0 8px}
header{display:flex;flex-wrap:wrap;gap:8px 16px;align-items:flex-end;justify-content:space-between;border-bottom:1px solid var(--line);padding-bottom:16px;margin-bottom:8px}
h1{font-size:clamp(44px,4.6vw,66px);line-height:1;font-weight:500;letter-spacing:-.06em;margin:0}h1 .ee-g{color:var(--field)}h1 .ee-b{color:var(--sim)}
h2{font-size:clamp(28px,3.2vw,42px);line-height:1.08;font-weight:500;letter-spacing:-.04em;margin:40px 0 8px;padding-top:16px;border-top:1px solid var(--line)}h4{margin:0 0 6px;font-size:.95rem;font-weight:600}
.sub{color:var(--muted);margin-top:6px}button.theme{background:transparent;color:var(--fg);border:1px solid var(--line-strong);border-radius:0;padding:6px 12px;cursor:pointer;font:600 .8rem var(--sans)}
.big{display:flex;flex-wrap:wrap;gap:12px;margin:16px 0}.big .n{flex:1 1 200px;background:var(--card);border:1px solid var(--line);border-radius:6px;padding:14px 16px}
.big .n b{display:block;font-size:2rem;line-height:1.1;font-weight:500;letter-spacing:-.03em}.big .n.bad b{color:var(--signal)}.big .n.ok b{color:var(--field)}.big .n span{color:var(--muted);font-size:.9rem}
.chartbox{overflow-x:auto}svg.chart{width:100%;min-width:600px;height:auto;display:block;margin:8px 0;font:13px var(--sans)}
svg .grid{stroke:var(--line);stroke-width:1}svg .tick,svg .axis{fill:var(--muted)}svg .label{fill:var(--fg);font-size:13px}svg .value{fill:var(--fg)}
svg .deadline{stroke:var(--deadline);stroke-width:1.5;stroke-dasharray:6 4}svg .deadline-label{fill:var(--deadline);font-size:11px}
svg .rate{stroke:var(--rate);stroke-width:1.5;stroke-dasharray:2 4}svg .mean{stroke:var(--fg);stroke-width:1;stroke-dasharray:3 3}
svg .series{fill:none;stroke-width:2}svg .range{stroke-width:1.5;opacity:.6}svg .marker{stroke:var(--fg);stroke-width:1}svg .sat{font-size:16px;font-weight:700}
.legend{display:flex;flex-wrap:wrap;gap:6px 14px;font-size:.85rem;color:var(--muted);margin:4px 0 12px}.key i{display:inline-block;width:12px;height:12px;border-radius:2px;margin-right:5px;vertical-align:-1px}
table{border-collapse:collapse;width:100%;font-size:.9rem;margin:8px 0}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}th{color:var(--muted);font-weight:600;font-size:.78rem;letter-spacing:.06em;text-transform:uppercase}
tr.total td{font-weight:700}tr.baseline td:nth-child(3){color:var(--field)}tr.stopped td:nth-child(3){color:var(--signal)}
.scroll{overflow-x:auto}table.prov{width:auto;min-width:100%;font-size:.82rem}.prov td,.prov th,.cards td,.cards th{white-space:nowrap}.prov td:nth-child(3){white-space:normal;min-width:180px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}.card{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:12px 14px;overflow-x:auto}.card th{text-transform:none;letter-spacing:0;font-size:.82rem}
.note{border-left:3px solid var(--signal);padding:4px 12px;color:var(--muted);margin:8px 0}code{font:.88em var(--mono);color:var(--fg)}
footer{color:var(--muted);font-size:.85rem;margin-top:48px;border-top:1px solid var(--line);padding-top:12px}footer .lab{display:block;margin-top:6px;color:var(--dim)}
"""

LAB_URL = "https://embodiededge.ai"
REPO_URL = "https://github.com/obiedeh/bench2field"
PROFILE_URL = "https://github.com/obiedeh"

JS = """
(function(){var r=document.documentElement,b=document.getElementById('theme');
function cur(){return r.getAttribute('data-theme')||'dark'}
function label(){b.textContent=cur()==='dark'?'Light theme':'Dark theme'}
b.addEventListener('click',function(){r.setAttribute('data-theme',cur()==='dark'?'light':'dark');label()});label()})();
"""


def render(cs: CaseStudy) -> str:
    variant = cs.config["headline_variant"]
    head_svg, head_rows = headline_bars(cs)
    field_row = next((r for r in head_rows if cs.boards[r["key"]].role == "field"), None)
    fits = field_row and field_row["frame_p95"] is not None and field_row["frame_p95"] <= cs.deadline_ms
    big = []
    if field_row:
        big.append(f'<div class="n {"ok" if fits else "bad"}"><b>{fmt(field_row["frame_p95"], 1)} ms</b>'
                   f'<span>full frame on the {esc(field_row["board"])}, p95, against the {cs.deadline_ms:g} ms deadline: '
                   f'{"fits" if fits else "does not fit"}</span></div>')
        big.append(f'<div class="n"><b>{fmt(field_row["model_p95"], 1)} ms</b><span>the model alone on the same board at '
                   f'{cs.rate_hz:g} Hz (inference plus copies, median of repeats)</span></div>')
    if cs.camera_rate_hz is not None:
        big.append(f'<div class="n"><b>{cs.camera_rate_hz:.1f} Hz</b><span>delivered by the rover camera, configured {cs.budget.target_hz:g} Hz; '
                   f'the headline tier is {cs.rate_hz:g} Hz</span></div>')

    same = [(cs.boards[p.board].label, p) for n in cs.config.get("same_frames", []) if (p := cs.profiles.get(n))]
    spin = [(p.label, p) for n in cs.config.get("spin_pair", []) if (p := cs.profiles.get(n))]
    frames_desc = ""
    if same:
        d = same[0][1].data["frames"]
        frames_desc = f"{d['n']} frames of {d['decoded_shape'][1]}×{d['decoded_shape'][0]}, set <code>{esc(d['set_sha256'][:12])}…</code>"

    gen = git_state()
    warn = "".join(f"<li>{esc(w)}</li>" for w in cs.warnings)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(cs.title)} – Bench2Field report</title><style>{CSS}</style></head><body>
<div class="ee-band"><div class="in"><a class="ee-mark" href="{LAB_URL}"><i></i>EmbodiedEdge <span>Labs · Evidence</span></a>
<nav><a href="{REPO_URL}">Repository</a><a href="{LAB_URL}">Lab</a><a href="{PROFILE_URL}">Obinna Edeh</a></nav></div></div><main>
<header><div><p class="eyebrow">Bench2Field · Case study</p><h1>{title_html(cs.title)}</h1><div class="sub">{esc(cs.config.get('model', ''))} · Bench2Field {esc(__version__)} report, generated from <code>{esc(cs.root.name)}/runs/</code></div></div>
<button class="theme" id="theme">Light theme</button></header>

<div class="big">{''.join(big)}</div>
<h2>Model alone versus the full frame</h2>
<p>The model-alone bar is what <code>b2f run</code> times: inference plus the input and output copies, response p95 at {cs.rate_hz:g} Hz, median of the baseline sweep's repeats. The full-frame bar is the pipeline profile on the same {frames_desc or 'frames'}: decode, preprocess, copies, inference and postprocess, one frame at a time.</p>
<div class="chartbox">{head_svg}</div>

<h2>Camera rate</h2>
<p>Configured {cs.budget.target_hz:g} Hz; delivered as measured with <code>ros2 topic hz</code> on the rover with its stack running. Every comparison on this page is made at the swept tier nearest the delivered rate, {cs.rate_hz:g} Hz.</p>
<div class="chartbox">{camera_rate_panel(cs)}</div>

<h2>Latency against request rate</h2>
<p>Response p95 (median of repeats, bar = range) for the baseline sweeps. Solid lines are <code>{esc(variant)}</code>, dashed <code>{esc(cs.config.get('reference_variant', ''))}</code>. The same model gets slower as frames get sparser, because the GPU drops into lower power states between them.</p>
<div class="chartbox">{latency_vs_rate(cs)}</div>

<h2>Power state by tier</h2>
<p><code>{esc(variant)}</code>, p50 over each tier. Where the board reports a GPU rail or an SM clock, it is shown; the Jetsons report board power, the 5090 GPU power only, and the two are never compared.</p>
{power_panels(cs)}

<h2>Repeat-to-repeat spread on the field board</h2>
{thermal_repeats(cs)}

<h2>The same frames on every board</h2>
<p>Stage cost per frame, p50, with the p95 total marked. {frames_desc}.</p>
<div class="chartbox">{stacked_stages(same, cs.deadline_ms, 'Stage cost per frame on each board') if same else '<p>No same-frames profiles.</p>'}</div>
<div class="scroll">{stage_table(same) if same else ''}</div>

<h2>Stage breakdown on the 5090 and the spin effect</h2>
<p>onnxruntime's intra-op threads spin-wait between runs by default. In an inference-only sweep that costs almost nothing; in a frame with host-side stages they compete for the CPU.</p>
<div class="chartbox">{stacked_stages(spin, None, 'Stage cost per frame on the 5090, spinning on and off') if spin else '<p>No spin pair.</p>'}</div>
<div class="scroll">{stage_table(spin) if spin else ''}</div>

<h2>Provenance</h2>
<p>Every run set this report draws on. Baselines are the sweeps marked as such in <code>report.yaml</code>; references are kept as evidence and shown in the tables above only where named.</p>
<div class="scroll">{provenance_table(cs)}</div>
{('<div class="note"><b>Loader warnings</b><ul>' + warn + '</ul></div>') if warn else ''}

<footer>Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by <code>b2f report</code> (Bench2Field {esc(__version__)}, commit {esc((gen.get('commit') or 'unknown')[:8])}{' dirty' if gen.get('dirty') else ''}). Budget: <code>{esc(cs.budget.name)}</code>. No number on this page was typed in.<span class="lab">An <a href="{LAB_URL}">EmbodiedEdge Labs</a> project. Measured on real hardware, published as found.</span></footer>
</main><script>{JS}</script></body></html>"""
    return page


def headline_svg(cs: CaseStudy) -> str:
    """The headline chart as a standalone SVG (for the README), with the
    colour tokens resolved to fixed light-theme values."""
    svg, _ = headline_bars(cs)
    fixed = {"var(--c1)": "#1f6fd1", "var(--c3)": "#c62828", "var(--line)": "#d8dbd2", "var(--muted)": "#5d6459",
             "var(--fg)": "#1a1c1e", "var(--deadline)": "#c62828"}
    style = ("<style>text{font:12px 'Helvetica Neue',Helvetica,Arial,sans-serif}.grid{stroke:#d8dbd2}.tick,.axis{fill:#5d6459}.label{fill:#1a1c1e;font-size:13px}"
             ".value{fill:#1a1c1e}.deadline{stroke:#c62828;stroke-width:1.5;stroke-dasharray:6 4}.deadline-label{fill:#c62828;font-size:11px}</style>")
    for k, v in fixed.items():
        svg = svg.replace(k, v)
    svg = svg.replace('<svg class="chart"', '<svg xmlns="http://www.w3.org/2000/svg" style="background:#f6f7f3"', 1)
    return svg.replace("</title>", "</title>" + style, 1)
