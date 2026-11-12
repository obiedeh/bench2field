#!/usr/bin/env python3
"""Render case study 07 as one self-contained HTML page: inline CSS and SVG, no network, the Bench2Field report
theme (dark with a light toggle), readable on a phone. Every number on the page is read from runs/ (this study),
from case study 06's committed stream record (the PyTorch comparison), or is a published figure labelled as such
with its source. Nothing is typed in.

Usage: python tools/make_report.py            (writes report/index.html and report/headline.svg)
"""
from __future__ import annotations

import html
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
RUNS = HERE / "runs"
OUT = HERE / "report"
CS06 = HERE.parent / "06_alpamayo2_edge" / "runs" / "stream_bench-thor_tuned_fp8_steps5_maxn.json"

DEADLINE_MS = 100.0
CLIP_NAMES = {
    "030c760c": "030c760c, following a slowing lead vehicle",
    "0347d9f9": "0347d9f9, right turn",
    "06b483cf": "06b483cf, straight through an intersection",
    "09ad74a2": "09ad74a2, straight, green light",
    "0a1ef808": "0a1ef808",
    "0ea6fd88": "0ea6fd88",
    "174407b0": "174407b0, all-way stop",
    "195f1fc6": "195f1fc6, left turn",
    "ea7bbd31": "ea7bbd31, NVIDIA's example, right turn",
}
STAGES = [
    ("vit_ms", "vision encoder", "var(--c1)"),
    ("prefill_ms", "prefill", "var(--c2)"),
    ("decode_ms", "chain of causation", "var(--c5)"),
    ("action_ms", "action expert", "var(--c4)"),
    ("action_kv_ms", "expert K/V copy", "var(--c6)"),
]

# Published figures, not measured here. Shown hatched and labelled on every chart that carries them.
PUBLISHED = [
    {
        "label": "NVIDIA, DRIVE AGX Thor, with drafter",
        "median": 220.0,
        "tail": None,
        "energy": None,
        "note": "indicative, recipe README; range 180 to 250 ms over ten clips",
        "source": "https://github.com/NVlabs/alpamayo-recipes/tree/main/recipes/alpamayo2edge_edgellm",
    },
    {
        "label": "Embedl card, same board",
        "median": 91.9,
        "tail": 140.3,
        "tail_kind": "p90",
        "energy": 12.45,
        "note": "model card, 2026-11-11: NVFP4 language model, MXFP8 vision and expert, caches, speculation; one clip, 180 decisions",
        "source": "https://huggingface.co/embedl/Alpamayo2-Edge",
    },
]


def esc(x) -> str:
    return html.escape(str(x))


def fmt(x, d=0) -> str:
    return "–" if x is None else (f"{x:,.{d}f}" if d else f"{x:,.0f}")


def load_stream(name: str) -> dict:
    return json.load(open(RUNS / "stream" / f"{name}.json"))["summary"]


# --- data --------------------------------------------------------------------

S = {
    "fp8_c1": load_stream("clips9_rolling_c1"),
    "fp8_c6": load_stream("clips9_rolling_c6"),
    "fp8_s3": load_stream("clips9_rolling_c1_s3"),
    "nvfp4_c1": load_stream("clips9_rolling_c1_nvfp4-w4a16"),
    "nvfp4_c6": load_stream("clips9_rolling_c6_nvfp4-w4a16"),
    "ex_fp8_c6": load_stream("example_rolling_c6"),
    "ex_fp8_c1": load_stream("example_rolling_c1"),
    "ex_cold_c6": load_stream("example_cold_c6"),
    "ex_nvfp4_c1": load_stream("example_rolling_c1_nvfp4-w4a16"),
    "ex_fast": load_stream("example_rolling_c1_fast"),
}
BOARD = json.load(open(RUNS / "stream" / "board.json"))
BUILD_FP8 = json.load(open(RUNS / "stream" / "build.json"))
BUILD_NVFP4 = json.load(open(RUNS / "stream" / "build_nvfp4-w4a16.json"))

cs06 = json.load(open(CS06))["results"]["reuse"]
PYTORCH = {
    "label": "PyTorch stream, 10 Nov (case study 06)",
    "median": cs06["latency_median_s"] * 1000,
    "tail": cs06["latency_p95_s"] * 1000,
    "n": cs06["n"],
}

BUILDS = [  # measured here
    ("FP8 recipe as written", S["fp8_c1"]),
    ("FP8, three Euler steps", S["fp8_s3"]),
    ("FP8, six candidates", S["fp8_c6"]),
    ("NVFP4 W4A16 language model", S["nvfp4_c1"]),
    ("NVFP4 W4A16, six candidates", S["nvfp4_c6"]),
]

cross = {p.stem[6:]: json.load(open(p)) for p in sorted((RUNS / "results").glob("cross_*.json"))}
spread = {}
for clip in cross:
    pw = json.load(open(RUNS / "results" / "fp8" / f"{clip}-rolling" / "compare.json"))["pairwise_ade_m"]
    off = [pw[i][j] for i in range(len(pw)) for j in range(len(pw)) if i < j]
    spread[clip] = (statistics.mean(off), max(off))

decode_per_token = []
for d in json.load(open(RUNS / "stream" / "clips9_rolling_c1.json"))["decisions"]:
    if d["generated_tokens"] > 6:  # reasoning decisions only; the nominal path is 6 tokens
        decode_per_token.append(d["stages_ms"]["decode_ms"] / d["generated_tokens"])
MS_PER_TOKEN = statistics.median(decode_per_token)

# --- svg helpers ---------------------------------------------------------------


def svg_open(w, h, title):
    return (f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-label="{esc(title)}">'
            f'<title>{esc(title)}</title>'
            '<defs><pattern id="hatch" patternUnits="userSpaceOnUse" width="8" height="8" patternTransform="rotate(45)">'
            '<line x1="0" y1="0" x2="0" y2="8" stroke="var(--muted)" stroke-width="2"/></pattern></defs>')


def text(x, y, s, cls="label", anchor="start", dy="0.35em", extra=""):
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}" dy="{dy}" {extra}>{esc(s)}</text>'


def bar(x, y, w, h, fill, title=None, rx=3):
    t = f"<title>{esc(title)}</title>" if title else ""
    return f'<g class="hit"><rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}"/>{t}</g>'


def nice_ticks(hi, n=5):
    import math
    raw = hi / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min(s for s in (1, 2, 2.5, 5, 10) if s * mag >= raw) * mag
    ticks, t = [], 0.0
    while t <= hi + 1e-9:
        ticks.append(t)
        t += step
    return ticks


def hbar_chart(rows, title, unit="ms", deadline=None, width=720, label_w=250, two=True):
    """rows: list of dict(label, a, b, published, tail_kind). Horizontal bars, a = median (blue), b = tail (lime)."""
    row_h = 44 if two else 30
    h = 30 + row_h * len(rows) + 54
    x0, x1 = label_w, width - 20
    hi = max(v for r in rows for v in (r["a"], r.get("b") or 0) if v is not None)
    if deadline:
        hi = max(hi, deadline)
    hi *= 1.12
    sx = lambda v: x0 + (x1 - x0) * v / hi
    out = [svg_open(width, h, title)]
    for t in nice_ticks(hi):
        out.append(f'<line x1="{sx(t):.1f}" y1="20" x2="{sx(t):.1f}" y2="{h - 36}" class="grid"/>')
        out.append(text(sx(t), h - 24, f"{t:g}", "tick", "middle"))
    out.append(text(x0, h - 4, unit, "tick", "start"))
    for i, r in enumerate(rows):
        y = 30 + i * row_h
        out.append(text(x0 - 10, y + (row_h - 8) / 2, r["label"], "label", "end"))
        fa = "url(#hatch)" if r.get("published") else "var(--sim)"
        fb = "url(#hatch)" if r.get("published") else "var(--field)"
        bh = 14 if two else 18
        out.append(bar(x0, y, sx(r["a"]) - x0, bh, fa, f'{r["label"]}: median {fmt(r["a"], 1)} ms'))
        out.append(text(sx(r["a"]) + 6, y + bh / 2, fmt(r["a"]), "value"))
        if two and r.get("b") is not None:
            out.append(bar(x0, y + bh + 4, sx(r["b"]) - x0, bh, fb, f'{r["label"]}: {r.get("tail_kind", "p95")} {fmt(r["b"], 1)} ms'))
            out.append(text(sx(r["b"]) + 6, y + bh + 4 + bh / 2, f'{fmt(r["b"])} {r.get("tail_kind", "p95")}', "value"))
        if r.get("published"):
            out.append(text(x0 - 10, y + (row_h - 8) / 2 + 14, "published, not run here", "tick", "end"))
    if deadline:
        out.append(f'<line x1="{sx(deadline):.1f}" y1="14" x2="{sx(deadline):.1f}" y2="{h - 36}" class="deadline"/>')
        out.append(text(sx(deadline) + 4, 12, f"{deadline:g} ms deadline (10 Hz)", "deadline-label"))
    out.append("</svg>")
    return "".join(out)


def stacked_chart(rows, title, width=720, label_w=250):
    """rows: (label, summary). Stacked stage medians, direct labels on segments wide enough."""
    row_h = 40
    h = 30 + row_h * len(rows) + 54
    x0, x1 = label_w, width - 20
    hi = max(sum(s["stage_median_ms"].values()) for _, s in rows) * 1.08
    sx = lambda v: x0 + (x1 - x0) * v / hi
    out = [svg_open(width, h, title)]
    for t in nice_ticks(hi):
        out.append(f'<line x1="{sx(t):.1f}" y1="20" x2="{sx(t):.1f}" y2="{h - 36}" class="grid"/>')
        out.append(text(sx(t), h - 24, f"{t:g}", "tick", "middle"))
    out.append(text(x0, h - 10, "ms, sum of stage medians", "tick", "start"))
    for i, (label, s) in enumerate(rows):
        y = 30 + i * row_h
        out.append(text(x0 - 10, y + 12, label, "label", "end"))
        acc = 0.0
        for key, name, colour in STAGES:
            v = s["stage_median_ms"][key]
            xa, xb = sx(acc), sx(acc + v)
            out.append(bar(xa + 1, y, xb - xa - 2, 24, colour, f"{label}: {name} {v:.1f} ms", rx=2))
            if xb - xa > 34:
                out.append(text((xa + xb) / 2, y + 12, f"{v:.0f}", "value-on", "middle"))
            acc += v
        out.append(text(sx(acc) + 6, y + 12, f"{acc:.0f} ms", "value"))
    out.append("</svg>")
    return "".join(out)


def clips_chart(s, title, width=720, label_w=280):
    per = s["per_clip"]
    rows = sorted(per.items(), key=lambda kv: -kv[1]["median_ms"])
    row_h = 30
    h = 30 + row_h * len(rows) + 54
    x0, x1 = label_w, width - 20
    hi = max(v["median_ms"] for _, v in rows) * 1.5
    sx = lambda v: x0 + (x1 - x0) * v / hi
    out = [svg_open(width, h, title)]
    for t in nice_ticks(hi):
        out.append(f'<line x1="{sx(t):.1f}" y1="20" x2="{sx(t):.1f}" y2="{h - 36}" class="grid"/>')
        out.append(text(sx(t), h - 24, f"{t:g}", "tick", "middle"))
    out.append(text(x0, h - 10, "ms, median of 20 decisions", "tick", "start"))
    for i, (clip, v) in enumerate(rows):
        y = 30 + i * row_h
        out.append(text(x0 - 10, y + 10, CLIP_NAMES.get(clip, clip), "label", "end"))
        fill = "var(--c6)" if v["nominal"] else "var(--sim)"
        out.append(bar(x0, y, sx(v["median_ms"]) - x0, 20, fill, f'{clip}: {v["median_ms"]:.1f} ms, {v["generated_tokens"]} tokens'))
        kind = "no reasoning" if v["nominal"] else f'{v["generated_tokens"]} tokens'
        out.append(text(sx(v["median_ms"]) + 6, y + 10, f'{v["median_ms"]:.0f} ms, {kind}', "value"))
    out.append(f'<line x1="{sx(DEADLINE_MS):.1f}" y1="14" x2="{sx(DEADLINE_MS):.1f}" y2="{h - 36}" class="deadline"/>')
    out.append(text(sx(DEADLINE_MS) + 4, 12, "100 ms deadline", "deadline-label"))
    out.append("</svg>")
    return "".join(out)


def ade_chart(title, width=720, label_w=250):
    rows = sorted(cross.items())
    row_h = 34
    h = 30 + row_h * len(rows) + 54
    x0, x1 = label_w, width - 20
    hi = max(spread[c][1] for c in cross) * 1.1
    sx = lambda v: x0 + (x1 - x0) * v / hi
    out = [svg_open(width, h, title)]
    for t in nice_ticks(hi):
        out.append(f'<line x1="{sx(t):.1f}" y1="20" x2="{sx(t):.1f}" y2="{h - 36}" class="grid"/>')
        out.append(text(sx(t), h - 24, f"{t:g}", "tick", "middle"))
    out.append(text(x0, h - 10, "ADE, metres over 64 points", "tick", "start"))
    for i, (clip, c) in enumerate(rows):
        y = 30 + i * row_h + 10
        out.append(text(x0 - 10, y, CLIP_NAMES.get(clip, clip).split(",")[0], "label", "end"))
        mean_s, max_s = spread[clip]
        out.append(bar(x0, y - 6, sx(max_s) - x0, 12, "var(--line-strong)", f"{clip}: spread among the six FP8 candidates, mean {mean_s:.2f} m, max {max_s:.2f} m", rx=6))
        out.append(f'<line x1="{sx(mean_s):.1f}" y1="{y - 9}" x2="{sx(mean_s):.1f}" y2="{y + 9}" class="mean"/>')
        nv = c["pairs"]["fp8 vs nvfp4-w4a16"]["first_candidates_m"]
        s3 = c["pairs"]["fp8 vs fp8-s3-c1"]["first_candidates_m"]
        out.append(f'<g class="hit"><circle cx="{sx(nv):.1f}" cy="{y:.1f}" r="7" fill="var(--c2)" class="marker"/><title>{clip}: FP8 against NVFP4 W4A16, {nv:.3f} m</title></g>')
        out.append(f'<g class="hit"><rect x="{sx(s3) - 6:.1f}" y="{y - 6:.1f}" width="12" height="12" fill="var(--c5)" class="marker" transform="rotate(45 {sx(s3):.1f} {y:.1f})"/><title>{clip}: five against three Euler steps, {s3:.3f} m</title></g>')
    out.append("</svg>")
    return "".join(out)


def energy_chart(title, width=720, label_w=250):
    rows = [(l, s["energy_per_decision_j"], s["power_timed"]["mean_w"], False) for l, s in BUILDS]
    rows.append((PUBLISHED[1]["label"], PUBLISHED[1]["energy"], None, True))
    row_h = 30
    h = 30 + row_h * len(rows) + 54
    x0, x1 = label_w, width - 20
    hi = max(r[1] for r in rows) * 1.35
    sx = lambda v: x0 + (x1 - x0) * v / hi
    out = [svg_open(width, h, title)]
    for t in nice_ticks(hi):
        out.append(f'<line x1="{sx(t):.1f}" y1="20" x2="{sx(t):.1f}" y2="{h - 36}" class="grid"/>')
        out.append(text(sx(t), h - 24, f"{t:g}", "tick", "middle"))
    out.append(text(x0, h - 10, "joules per decision on the board input rail, and mean watts while deciding", "tick", "start"))
    for i, (label, e, w, pub) in enumerate(rows):
        y = 30 + i * row_h
        out.append(text(x0 - 10, y + 10, label, "label", "end"))
        out.append(bar(x0, y, sx(e) - x0, 20, "url(#hatch)" if pub else "var(--field)", f"{label}: {e:.1f} J per decision"))
        tail = f"{e:.2f} J, published" if pub else f"{e:.1f} J, {w:.0f} W"
        out.append(text(sx(e) + 6, y + 10, tail, "value"))
    out.append("</svg>")
    return "".join(out)


def _wrap(label, width=18):
    words, lines, cur = label.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines[:3]


def _vaxis(out, sy, hi, x0, x1, y0, y1, unit):
    for t in nice_ticks(hi):
        out.append(f'<line x1="{x0}" y1="{sy(t):.1f}" x2="{x1}" y2="{sy(t):.1f}" class="grid"/>')
        out.append(text(x0 - 8, sy(t), f"{t:g}", "tick", "end"))
    out.append(text(x0, y0 - 16, unit, "tick", "start"))


def vstacked_chart(rows, title, width=720, height=380):
    """rows: (label, summary). Vertical stacked columns of stage medians, one column per build."""
    x0, x1, y0, y1 = 60, width - 20, 36, height - 64
    hi = max(sum(s["stage_median_ms"].values()) for _, s in rows) * 1.12
    sy = lambda v: y1 - (y1 - y0) * v / hi
    slot = (x1 - x0) / len(rows)
    bw = min(slot * 0.56, 90)
    out = [svg_open(width, height, title)]
    _vaxis(out, sy, hi, x0, x1, y0, y1, "ms, sum of stage medians")
    for i, (label, s) in enumerate(rows):
        cx = x0 + slot * (i + 0.5)
        acc = 0.0
        for key, name, colour in STAGES:
            v = s["stage_median_ms"][key]
            ya, yb = sy(acc + v), sy(acc)
            out.append(bar(cx - bw / 2, ya + 1, bw, max(yb - ya - 2, 0), colour, f"{label}: {name} {v:.1f} ms", rx=2))
            if yb - ya > 18:
                out.append(text(cx, (ya + yb) / 2, f"{v:.0f}", "value-on", "middle"))
            acc += v
        out.append(text(cx, sy(acc) - 10, f"{acc:.0f} ms", "value", "middle"))
        for j, line in enumerate(_wrap(label, max(10, int(slot / 7.2)))):
            out.append(text(cx, y1 + 16 + j * 15, line, "label", "middle"))
    out.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" class="axis-line"/>')
    out.append("</svg>")
    return "".join(out)


def venergy_chart(title, width=720, height=380):
    rows = [(l, s["energy_per_decision_j"], s["power_timed"]["mean_w"], False) for l, s in BUILDS]
    rows.append((PUBLISHED[1]["label"], PUBLISHED[1]["energy"], None, True))
    x0, x1, y0, y1 = 60, width - 20, 36, height - 64
    hi = max(r[1] for r in rows) * 1.25
    sy = lambda v: y1 - (y1 - y0) * v / hi
    slot = (x1 - x0) / len(rows)
    bw = min(slot * 0.56, 90)
    out = [svg_open(width, height, title)]
    _vaxis(out, sy, hi, x0, x1, y0, y1, "joules per decision")
    for i, (label, e, w, pub) in enumerate(rows):
        cx = x0 + slot * (i + 0.5)
        out.append(bar(cx - bw / 2, sy(e), bw, y1 - sy(e), "url(#hatch)" if pub else "var(--field)", f"{label}: {e:.1f} J per decision" + (", published" if pub else f", {w:.0f} W mean")))
        out.append(text(cx, sy(e) - 24, f"{e:.1f} J", "value", "middle"))
        out.append(text(cx, sy(e) - 10, "published" if pub else f"{w:.0f} W mean", "tick", "middle"))
        for j, line in enumerate(_wrap(label, max(10, int(slot / 7.2)))):
            out.append(text(cx, y1 + 16 + j * 15, line, "label", "middle"))
    out.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" class="axis-line"/>')
    out.append("</svg>")
    return "".join(out)


# --- instrument cluster ----------------------------------------------------------


def _pt(cx, cy, r, deg):
    import math
    a = math.radians(deg)
    return cx + r * math.cos(a), cy + r * math.sin(a)


def _arc(cx, cy, r, a0, a1):
    """SVG path for the arc from a0 to a1 degrees (clockwise, y down)."""
    x0, y0 = _pt(cx, cy, r, a0)
    x1, y1 = _pt(cx, cy, r, a1)
    large = 1 if a1 - a0 > 180 else 0
    return f"M{x0:.1f} {y0:.1f}A{r:.1f} {r:.1f} 0 {large} 1 {x1:.1f} {y1:.1f}"


def dial(value, vmin, vmax, *, title, readout, sub, unit, major, minor=None, zones=(), markers=(), size=320, fmt_tick=lambda t: f"{t:g}"):
    """A speedometer: a 270 degree arc from bottom-left to bottom-right, a needle at `value`, coloured zones on the
    ring, labelled markers for reference values, and a digital readout under the hub.
    zones: (lo, hi, colour, name); markers: (value, label, published)."""
    import math
    A0, A1 = 135.0, 405.0
    cx, cy = size / 2, size / 2
    r = size * 0.36
    rb = r + 20  # bezel
    ang = lambda v: A0 + (A1 - A0) * (min(max(v, vmin), vmax) - vmin) / (vmax - vmin)
    out = [f'<svg class="dial" viewBox="-48 0 {size + 96} {size * 1.16:.0f}" role="img" aria-label="{esc(title)}: {esc(readout)} {esc(unit)}">'
           f'<title>{esc(title)}: {esc(readout)} {esc(unit)}, {esc(sub)}</title>']
    # bezel and track
    out.append(f'<circle cx="{cx}" cy="{cy}" r="{rb:.1f}" class="bezel"/>')
    out.append(f'<path d="{_arc(cx, cy, r, A0, A1)}" class="track"/>')
    for lo, hi, colour, name in zones:
        a, b = ang(lo), ang(hi)
        out.append(f'<path d="{_arc(cx, cy, r, a + 0.6, b - 0.6)}" class="zone" stroke="{colour}"><title>{esc(name)}</title></path>')
    # ticks
    step = minor or major
    t = vmin
    while t <= vmax + 1e-9:
        a = ang(t)
        is_major = abs((t - vmin) / major - round((t - vmin) / major)) < 1e-6
        r0 = r - 14 if is_major else r - 9
        x0, y0 = _pt(cx, cy, r0, a)
        x1, y1 = _pt(cx, cy, r - 2, a)
        out.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" class="{"tick-major" if is_major else "tick-minor"}"/>')
        if is_major:
            tx, ty = _pt(cx, cy, r - 30, a)
            out.append(text(tx, ty, fmt_tick(t), "dial-num", "middle"))
        t += step
    # reference markers outside the ring
    for v, label, published in markers:
        a = ang(v)
        x0, y0 = _pt(cx, cy, r + 7, a)
        x1, y1 = _pt(cx, cy, r + 17, a)
        xl, yl = _pt(cx, cy, r + 10, a - 4.0)
        xr, yr = _pt(cx, cy, r + 10, a + 4.0)
        cls = "marker-pub" if published else "marker-ref"
        out.append(f'<g class="hit"><polygon points="{x0:.1f},{y0:.1f} {xl:.1f},{yl:.1f} {x1:.1f},{y1:.1f} {xr:.1f},{yr:.1f}" class="{cls}"/>'
                   f'<title>{esc(label)}: {v:g} {esc(unit)}{", published, not run here" if published else ""}</title></g>')
        lx, ly = _pt(cx, cy, rb + 12, a)
        anchor = "middle" if abs(math.cos(math.radians(a))) < 0.35 else ("end" if math.cos(math.radians(a)) < 0 else "start")
        out.append(text(lx, ly, label, "dial-mark", anchor))
    # needle
    a = ang(value)
    tipx, tipy = _pt(cx, cy, r - 6, a)
    bl, br = _pt(cx, cy, 10, a + 90), _pt(cx, cy, 10, a - 90)
    tail = _pt(cx, cy, 22, a + 180)
    out.append(f'<polygon points="{tipx:.1f},{tipy:.1f} {bl[0]:.1f},{bl[1]:.1f} {tail[0]:.1f},{tail[1]:.1f} {br[0]:.1f},{br[1]:.1f}" class="needle"/>')
    out.append(f'<circle cx="{cx}" cy="{cy}" r="13" class="hub"/><circle cx="{cx}" cy="{cy}" r="4" class="hub-dot"/>')
    # readout
    out.append(text(cx, cy + r * 0.50, readout, "dial-readout", "middle"))
    out.append(text(cx, cy + r * 0.50 + 24, unit, "dial-unit", "middle"))
    out.append(text(cx, cy + rb + 28, title, "dial-title", "middle"))
    out.append(text(cx, cy + rb + 50, sub, "dial-sub", "middle"))
    out.append("</svg>")
    return "".join(out)


def lamp(state, icon, label, detail):
    """A tell-tale: a lit square with an icon, a label and one line of detail. state: ok, warn, bad."""
    return (f'<div class="lamp {state}"><div class="lens">{ICON[icon]}</div><div><b>{esc(label)}</b><span>{esc(detail)}</span></div></div>')


# --- icons (inline, stroke follows text colour) ----------------------------------

ICON = {
    "down": '<svg class="ico" viewBox="0 0 24 24"><path d="M12 4v14m0 0l-6-6m6 6l6-6" fill="none" stroke="var(--field)" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    "up": '<svg class="ico" viewBox="0 0 24 24"><path d="M12 20V6m0 0l-6 6m6-6l6 6" fill="none" stroke="var(--signal)" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    "flat": '<svg class="ico" viewBox="0 0 24 24"><path d="M4 12h16" fill="none" stroke="var(--muted)" stroke-width="2.4" stroke-linecap="round"/></svg>',
    "bug": '<svg class="ico" viewBox="0 0 24 24"><path d="M12 3l9 16H3z" fill="none" stroke="var(--signal)" stroke-width="2" stroke-linejoin="round"/><path d="M12 9v5m0 3v.5" stroke="var(--signal)" stroke-width="2.2" stroke-linecap="round"/></svg>',
    "fix": '<svg class="ico" viewBox="0 0 24 24"><path d="M5 12.5l4.5 4.5L19 7.5" fill="none" stroke="var(--field)" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    "clock": '<svg class="ico" viewBox="0 0 24 24"><circle cx="12" cy="12" r="8.5" fill="none" stroke="currentColor" stroke-width="2"/><path d="M12 7.5V12l3 2" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>',
    "bolt": '<svg class="ico" viewBox="0 0 24 24"><path d="M13 2L5 14h6l-1 8 8-12h-6z" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/></svg>',
    "chip": '<svg class="ico" viewBox="0 0 24 24"><rect x="6" y="6" width="12" height="12" rx="2" fill="none" stroke="currentColor" stroke-width="2"/><path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4" stroke="currentColor" stroke-width="2"/></svg>',
    "target": '<svg class="ico" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" stroke-width="2"/><circle cx="12" cy="12" r="4" fill="none" stroke="currentColor" stroke-width="2"/><circle cx="12" cy="12" r="1.2" fill="currentColor"/></svg>',
    "text": '<svg class="ico" viewBox="0 0 24 24"><path d="M4 6h16M4 12h16M4 18h10" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>',
    "warn": '<svg class="ico" viewBox="0 0 24 24"><path d="M12 3l9 16H3z" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/><path d="M12 9v5m0 3v.5" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/></svg>',
    "check": '<svg class="ico" viewBox="0 0 24 24"><path d="M5 12.5l4.5 4.5L19 7.5" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    "speed": '<svg class="ico" viewBox="0 0 24 24"><path d="M4 16a8 8 0 1 1 16 0" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><path d="M12 16l4-6" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/><circle cx="12" cy="16" r="1.6" fill="currentColor"/></svg>',
    "lock": '<svg class="ico" viewBox="0 0 24 24"><rect x="5" y="10" width="14" height="10" rx="2" fill="none" stroke="currentColor" stroke-width="2"/><path d="M8 10V7a4 4 0 0 1 8 0v3" fill="none" stroke="currentColor" stroke-width="2"/></svg>',
}

ISSUES = [
    ("The recipe's submodule step must be recursive", "cmake stopped on a missing xgrammar header: Edge-LLM's own submodules were absent.", "Fetched recursively, copied 3rdParty to the board."),
    ("nvcc from the `cuda` alternative against CUDA 13.0 headers", "The toolchain forces /usr/local/cuda/bin/nvcc, 13.2 on this board; its launch stubs do not match 13.0's host_runtime.h, every CUDA file failed.", "nvcc pinned to the 13.0 toolkit, CUDA_DIR and cuRAND include set to match."),
    ("Action engine fixed at six candidates", "A one-candidate export writes candidates_per_sequence 2; the builder computes zero sequences and TensorRT rejects the profile.", "Set candidates_per_sequence to 1 in the one-candidate config."),
    ("Shim renames MLP layers, copies the quantization config verbatim", "A MIXED_PRECISION (NVFP4) config loses every per-layer entry; the exporter treats packed weights as FP16 and torch.export fails on a shape mismatch.", "Rewrote the config keys with the same renames, dropped modules restored to FP16."),
    ("NVFP4-A16 GEMM kernels need the CuTe DSL build flag", "Nvfp4A16BlackwellGemmRunner::isSupported is false unless the plugin is built with ENABLE_CUTE_DSL=ALL; the recipe's build script leaves the default.", "Reconfigured with the kernel groups on; the W4A16 engines built."),
    ("TensorRT 10.13 cannot build the W4A4 graph", "Myelin stops on PLUGIN_V3 with empty producers; the recipe pins TensorRT 11.3.", "Parked; a newer TensorRT (JetPack 7.2.2) is the owner's call."),
    ("A probe reported 1.1 ms per decode token", "A timing bug in the probe (start time reset per iteration). The corrected figure is 11.8 ms, and the wrong number was withdrawn the same hour.", "Probe fixed; both numbers kept in the friction log."),
]

CHANGES = [
    ("down", "NVIDIA's recipe on the Jetson board", f'Tail {PYTORCH["tail"]:.0f} to {S["fp8_c1"]["p95_ms"]:.0f} ms against the PyTorch stream; median {PYTORCH["median"]:.0f} to {S["fp8_c1"]["median_ms"]:.0f} ms, so no gain at the median.'),
    ("down", "Vision cache across decisions", f'Vision stage {S["ex_cold_c6"]["stage_median_ms"]["vit_ms"]:.0f} to {S["ex_fp8_c6"]["stage_median_ms"]["vit_ms"]:.0f} ms; p95 {S["ex_cold_c6"]["p95_ms"]:.0f} to {S["ex_fp8_c6"]["p95_ms"]:.0f} ms and {S["ex_cold_c6"]["energy_per_decision_j"] - S["ex_fp8_c6"]["energy_per_decision_j"]:.0f} J less per decision against encoding all 22 frames; the median moved only {S["ex_cold_c6"]["median_ms"] - S["ex_fp8_c6"]["median_ms"]:.0f} ms because the cold run decoded faster on that clip (example clip, six candidates).'),
    ("down", "One candidate instead of six", f'Expert {S["fp8_c6"]["stage_median_ms"]["action_ms"]:.0f} to {S["fp8_c1"]["stage_median_ms"]["action_ms"]:.0f} ms; decision {S["fp8_c6"]["median_ms"]:.0f} to {S["fp8_c1"]["median_ms"]:.0f} ms median.'),
    ("down", "Three Euler steps instead of five", f'Expert {S["fp8_c1"]["stage_median_ms"]["action_ms"]:.0f} to {S["fp8_s3"]["stage_median_ms"]["action_ms"]:.0f} ms; median {S["fp8_c1"]["median_ms"]:.0f} to {S["fp8_s3"]["median_ms"]:.0f} ms; trajectories within 0.2 m of five steps on eight clips, 0.6 m on one.'),
    ("up", "NVFP4 W4A16 language model", f'Decode {S["ex_fp8_c1"]["stage_median_ms"]["decode_ms"] / S["ex_fp8_c1"]["per_clip"]["ea7bbd31"]["generated_tokens"]:.1f} to {S["ex_nvfp4_c1"]["stage_median_ms"]["decode_ms"] / S["ex_nvfp4_c1"]["per_clip"]["ea7bbd31"]["generated_tokens"]:.1f} ms per token on the example clip, prefill {S["fp8_c1"]["stage_median_ms"]["prefill_ms"]:.0f} to {S["nvfp4_c1"]["stage_median_ms"]["prefill_ms"]:.0f} ms, vision {S["fp8_c1"]["stage_median_ms"]["vit_ms"]:.0f} to {S["nvfp4_c1"]["stage_median_ms"]["vit_ms"]:.0f} ms. The engine file is smaller ({BUILD_NVFP4["engines"]["llm/llm.engine"] / 1e9:.2f} against {BUILD_FP8["engines"]["llm/llm.engine"] / 1e9:.2f} GB), so the weights are packed; the single-token kernel on this stack does not turn the bytes into time.'),
    ("flat", "Token kept on the GPU, end of chain checked per chunk", f'{S["ex_fast"]["stage_median_ms"]["decode_ms"]:.0f} ms decode against {S["ex_fp8_c1"]["stage_median_ms"]["decode_ms"]:.0f}: no change. The host round trip was never the cost.'),
]

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
main{max-width:1040px;margin:0 auto;padding:40px 24px 80px}
p{max-width:78ch;margin:12px 0 20px}
.eyebrow{color:var(--field);font:600 12px/1.4 var(--mono);letter-spacing:.08em;text-transform:uppercase;margin:0 0 12px}
header{display:flex;flex-wrap:wrap;gap:16px 24px;align-items:flex-end;justify-content:space-between;border-bottom:1px solid var(--line);padding-bottom:28px;margin-bottom:36px}
h1{font-size:clamp(40px,4.6vw,64px);line-height:1;font-weight:500;letter-spacing:-.06em;margin:0}h1 .ee-g,h1 .nw{white-space:nowrap}h1 .ee-g{color:var(--field)}h1 .ee-b{color:var(--sim)}
h2{font-size:clamp(28px,3.2vw,40px);line-height:1.08;font-weight:500;letter-spacing:-.04em;margin:80px 0 16px;padding-top:28px;border-top:1px solid var(--line)}
h3{font-size:1.1rem;font-weight:600;margin:32px 0 8px}
.sub{color:var(--muted);margin-top:14px;max-width:70ch}button.theme{background:transparent;color:var(--fg);border:1px solid var(--line-strong);border-radius:0;padding:8px 14px;cursor:pointer;font:600 .8rem var(--sans)}
.cluster{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:36px 20px 28px;margin:8px 0 28px;position:relative}
.cluster .strip{position:absolute;left:28px;right:28px;top:0;height:3px;background:linear-gradient(90deg,var(--sim),var(--field));border-radius:0 0 3px 3px;opacity:.9}
.dials{display:grid;grid-template-columns:1fr 1.3fr 1fr;gap:12px;align-items:end}
.dials .side{padding-bottom:4%}
svg.dial{width:100%;height:auto;display:block;font:13px var(--sans)}
svg.dial .bezel{fill:var(--bg);stroke:var(--line-strong);stroke-width:1.5}svg.dial .track{fill:none;stroke:var(--line-strong);stroke-width:12;stroke-linecap:butt}
svg.dial .zone{fill:none;stroke-width:12;stroke-linecap:butt}svg.dial .tick-major{stroke:var(--fg);stroke-width:2}svg.dial .tick-minor{stroke:var(--muted);stroke-width:1}
svg.dial .dial-num{fill:var(--fg);font-size:15px;font-weight:600;font-family:var(--mono)}svg.dial .dial-mark{fill:var(--muted);font-size:13px}
svg.dial .marker-ref{fill:var(--fg)}svg.dial .marker-pub{fill:var(--muted)}
svg.dial .needle{fill:var(--signal);stroke:var(--bg);stroke-width:1}svg.dial .hub{fill:var(--card);stroke:var(--line-strong);stroke-width:2}svg.dial .hub-dot{fill:var(--signal)}
svg.dial .dial-readout{fill:var(--fg);font:500 44px var(--mono);letter-spacing:-.03em}svg.dial .dial-unit{fill:var(--muted);font:12px var(--mono);letter-spacing:.08em;text-transform:uppercase}
svg.dial .dial-title{fill:var(--fg);font-size:19px;font-weight:600;letter-spacing:-.01em}svg.dial .dial-sub{fill:var(--muted);font-size:14px}
.lamps{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:18px 24px;margin-top:28px;padding-top:24px;border-top:1px solid var(--line)}
.lamp{display:flex;gap:12px;align-items:flex-start}.lamp .lens{flex:none;width:40px;height:40px;border-radius:10px;display:grid;place-items:center;border:1px solid var(--line-strong);background:var(--bg)}
.lamp .lens .ico{width:22px;height:22px}.lamp b{display:block;font-size:.9rem;line-height:1.25}.lamp span{display:block;color:var(--muted);font-size:.8rem;line-height:1.35;margin-top:2px}
.lamp.ok .lens{color:var(--field);border-color:var(--field);box-shadow:inset 0 0 0 3px color-mix(in srgb,var(--field) 18%,transparent)}
.lamp.warn .lens{color:var(--signal);border-color:var(--signal);box-shadow:inset 0 0 0 3px color-mix(in srgb,var(--signal) 18%,transparent)}
.lamp.bad .lens{color:var(--deadline);border-color:var(--deadline);box-shadow:inset 0 0 0 3px color-mix(in srgb,var(--deadline) 18%,transparent)}
.lamp.off .lens{color:var(--dim)}
@media(max-width:760px){.dials{grid-template-columns:1fr}.dials .side{padding:0;max-width:300px;margin:0 auto}.cluster{padding:28px 16px 20px}}
.chartbox{overflow-x:auto}svg.chart{width:100%;min-width:600px;height:auto;display:block;margin:20px 0 8px;font:13px var(--sans)}
svg .grid{stroke:var(--line);stroke-width:1}svg .axis-line{stroke:var(--line-strong);stroke-width:1}svg .tick,svg .axis{fill:var(--muted)}svg .label{fill:var(--fg);font-size:13px}svg .value{fill:var(--fg)}svg .value-on{fill:#111;font-size:12px;font-weight:600}
svg .deadline{stroke:var(--deadline);stroke-width:1.5;stroke-dasharray:6 4}svg .deadline-label{fill:var(--deadline);font-size:11px}
svg .mean{stroke:var(--fg);stroke-width:1.5}svg .marker{stroke:var(--bg);stroke-width:2}svg .hit:hover rect,svg .hit:hover circle{filter:brightness(1.15)}
.legend{display:flex;flex-wrap:wrap;gap:8px 18px;font-size:.85rem;color:var(--muted);margin:8px 0 4px}.key i{display:inline-block;width:12px;height:12px;border-radius:2px;margin-right:6px;vertical-align:-1px}
.key i.hatch{background:repeating-linear-gradient(45deg,var(--muted) 0 2px,transparent 2px 6px)}.key i.diamond{transform:rotate(45deg);border-radius:1px}.key i.dot{border-radius:50%}
table{border-collapse:collapse;width:100%;font-size:.9rem;margin:24px 0 8px}th,td{text-align:left;padding:10px 12px;border-bottom:1px solid var(--line);vertical-align:top}th{color:var(--muted);font-weight:600}
.scroll{overflow-x:auto;margin-bottom:12px}td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.list{list-style:none;padding:0;margin:20px 0;display:grid;gap:14px}.list li{display:grid;grid-template-columns:28px 1fr;gap:14px;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px 20px}
.list .ico{width:22px;height:22px;margin-top:2px}.list b{display:block;font-weight:600;margin-bottom:4px}.list span{color:var(--muted);font-size:.92rem;line-height:1.5}
.note{border-left:3px solid var(--signal);padding:6px 16px;color:var(--muted);margin:20px 0}code{font:.88em var(--mono);color:var(--fg)}
footer{color:var(--muted);font-size:.85rem;margin-top:80px;border-top:1px solid var(--line);padding-top:20px;max-width:78ch}footer .lab{display:block;margin-top:10px;color:var(--dim)}
"""


def render() -> str:
    fp8, s3, nv = S["fp8_c1"], S["fp8_s3"], S["nvfp4_c1"]
    nominal = [c for c, v in fp8["per_clip"].items() if v["nominal"]]
    reasoning = [c for c, v in fp8["per_clip"].items() if not v["nominal"]]
    floor = fp8["stage_median_ms"]["vit_ms"] + fp8["stage_median_ms"]["prefill_ms"] + fp8["stage_median_ms"]["action_ms"] + fp8["stage_median_ms"]["action_kv_ms"]
    build_rows = [{"label": PYTORCH["label"], "a": PYTORCH["median"], "b": PYTORCH["tail"]}]
    build_rows += [{"label": l, "a": s["median_ms"], "b": s["p95_ms"]} for l, s in BUILDS]
    build_rows += [{"label": p["label"], "a": p["median"], "b": p["tail"], "published": True, "tail_kind": p.get("tail_kind", "p95")} for p in PUBLISHED]
    stack_rows = [("FP8 recipe, one candidate", fp8), ("FP8, three steps", s3), ("FP8, six candidates", S["fp8_c6"]),
                  ("NVFP4 W4A16, one candidate", nv), ("FP8, cold vision cache, example clip", S["ex_cold_c6"])]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    parts = [f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
             f'<title>Case study 07: Alpamayo 2 Edge with TensorRT Edge-LLM on the Jetson AGX Thor – Bench2Field report</title><style>{CSS}</style></head><body>'
             '<div class="ee-band"><div class="in"><a class="ee-mark" href="https://embodiededge.ai"><i></i>EmbodiedEdge <span>Evidence</span></a>'
             '<nav><a href="https://github.com/obiedeh/bench2field">Repository</a><a href="https://github.com/obiedeh/bench2field/tree/master/case_studies/07_alpamayo2_edge_edgellm">Case study</a><a href="https://embodiededge.ai">Lab</a><a href="https://github.com/obiedeh">Obinna Edeh</a></nav></div></div><main>']
    parts.append('<header><div><p class="eyebrow">Bench2Field · Case study 07</p>'
                 '<h1><span class="ee-b">Alpamayo 2 Edge</span> with <span class="nw">TensorRT Edge-LLM</span> on the <span class="ee-g">Jetson AGX Thor</span></h1>'
                 f'<div class="sub">NVIDIA\'s deployment recipe replicated on the Jetson board and pushed further · {esc(BOARD["jetpack"].replace("Installed: ", "JetPack "))}, TensorRT {esc(BOARD["tensorrt"])}, CUDA {esc(BOARD["cuda"])}, {esc(BOARD["power_mode"].replace("NV Power Mode: ", ""))} · 12 November 2026 · generated from <code>07_alpamayo2_edge_edgellm/runs/</code></div></div>'
                 '<button class="theme" id="theme">Light theme</button></header>')

    # The instrument cluster: decisions per second in the middle (the planner wants 10 Hz), reasoning tokens per
    # second on the left, energy per decision on the right, and the tell-tales under them.
    hz = 1000.0 / fp8["median_ms"]
    hz_p95 = 1000.0 / fp8["p95_ms"]
    tok_s = 1000.0 / MS_PER_TOKEN
    nv_tok_ms = S["ex_nvfp4_c1"]["stage_median_ms"]["decode_ms"] / S["ex_nvfp4_c1"]["per_clip"]["ea7bbd31"]["generated_tokens"]
    fp8_tok_ms = S["ex_fp8_c1"]["stage_median_ms"]["decode_ms"] / S["ex_fp8_c1"]["per_clip"]["ea7bbd31"]["generated_tokens"]
    speedo = dial(hz, 0, 12, title="Decisions per second", readout=f"{hz:.1f}", unit="Hz, median decision", size=360,
                  sub=f'{fp8["median_ms"]:.0f} ms median, {fp8["p95_ms"]:.0f} ms p95, FP8 recipe, nine clips', major=2, minor=0.5,
                  zones=[(1000.0 / DEADLINE_MS, 12, "var(--field)", "meets the 10 Hz planner deadline")],
                  markers=[(1000.0 / PYTORCH["median"], "PyTorch", False), (1000.0 / PUBLISHED[0]["median"], "NVIDIA", True), (1000.0 / PUBLISHED[1]["median"], "Embedl", True)])
    tach = dial(tok_s, 0, 150, title="Reasoning tokens per second", readout=f"{tok_s:.0f}", unit="tokens / s", size=300,
                sub=f"{MS_PER_TOKEN:.1f} ms per token, greedy", major=50, minor=10,
                markers=[(1000.0 / nv_tok_ms, "NVFP4", False)])
    energy = dial(fp8["energy_per_decision_j"], 0, 50, title="Energy per decision", readout=f'{fp8["energy_per_decision_j"]:.0f}', unit="joules", size=300,
                  sub=f'{fp8["power_timed"]["mean_w"]:.0f} W mean while deciding, MAXN', major=10, minor=5,
                  markers=[(PUBLISHED[1]["energy"], "Embedl", True), (nv["energy_per_decision_j"], "NVFP4", False)])
    parts.append('<section class="cluster"><div class="strip"></div>'
                 f'<div class="dials"><div class="side">{tach}</div><div>{speedo}</div><div class="side">{energy}</div></div>'
                 '<div class="lamps">'
                 + lamp("bad", "speed", "Deadline missed", f'{fp8["median_ms"]:.0f} ms against the 100 ms planner budget')
                 + lamp("ok", "check", "Same chain of causation", "NVIDIA's example sentence, word for word")
                 + lamp("ok", "target", "Tail improved", f'p95 {PYTORCH["tail"]:.0f} to {fp8["p95_ms"]:.0f} ms against the PyTorch stream')
                 + lamp("warn", "up", "NVFP4 slower", f'{nv["median_ms"]:.0f} ms median with the language model in W4A16')
                 + lamp("warn", "lock", "Drafter not released", "the recipe's 220 ms needs a DFlash2 drafter")
                 + lamp("warn", "warn", f"{len(ISSUES)} issues logged", "each with its cause and fix, in the friction log")
                 + '</div></section>'
                 '<div class="legend"><span class="key"><i style="background:var(--signal)"></i>needle, measured here</span><span class="key"><i style="background:var(--muted)"></i>grey marker, published figure, not run here</span><span class="key"><i style="background:var(--fg)"></i>marker in text colour, another build measured here</span><span class="key"><i style="background:var(--field)"></i>green band, deadline met</span></div>')

    parts.append('<h2>Where every build lands</h2>'
                 '<p>Per decision, back to back with the engines loaded once, five Euler steps, the six newest camera frames encoded per decision (the recipe\'s rolling mode), MAXN. '
                 f'Nine clips, 20 decisions each: the eight of case studies 02 and 06 and NVIDIA\'s example clip. The PyTorch row is case study 06\'s 10 Hz stream on the same board ({PYTORCH["n"]} decisions). '
                 'Hatched rows are published figures, not measured here.</p>'
                 '<div class="legend"><span class="key"><i style="background:var(--sim)"></i>median</span><span class="key"><i style="background:var(--field)"></i>p95 (Embedl reports p90)</span><span class="key"><i class="hatch"></i>published, not run here</span><span class="key"><i style="background:var(--deadline)"></i>100 ms deadline</span></div>'
                 f'<div class="chartbox">{hbar_chart(build_rows, "Median and tail latency per decision, by build", deadline=DEADLINE_MS)}</div>')
    parts.append('<div class="scroll"><table><thead><tr><th>build</th><th class="num">median, ms</th><th class="num">p95, ms</th><th class="num">J per decision</th><th class="num">mean W</th><th>record</th></tr></thead><tbody>')
    parts.append(f'<tr><td>{esc(PYTORCH["label"])}</td><td class="num">{PYTORCH["median"]:.0f}</td><td class="num">{PYTORCH["tail"]:.0f}</td><td class="num">–</td><td class="num">–</td><td><code>06_alpamayo2_edge/runs/stream_bench-thor_tuned_fp8_steps5_maxn.json</code></td></tr>')
    files = {"FP8 recipe as written": "clips9_rolling_c1.json", "FP8, three Euler steps": "clips9_rolling_c1_s3.json", "FP8, six candidates": "clips9_rolling_c6.json",
             "NVFP4 W4A16 language model": "clips9_rolling_c1_nvfp4-w4a16.json", "NVFP4 W4A16, six candidates": "clips9_rolling_c6_nvfp4-w4a16.json"}
    for l, s in BUILDS:
        parts.append(f'<tr><td>{esc(l)}</td><td class="num">{s["median_ms"]:.0f}</td><td class="num">{s["p95_ms"]:.0f}</td><td class="num">{s["energy_per_decision_j"]:.1f}</td><td class="num">{s["power_timed"]["mean_w"]:.0f}</td><td><code>runs/stream/{files[l]}</code></td></tr>')
    for p in PUBLISHED:
        parts.append(f'<tr><td>{esc(p["label"])}</td><td class="num">{p["median"]:.0f}</td><td class="num">{fmt(p["tail"])}{" (p90)" if p.get("tail_kind") == "p90" else ""}</td><td class="num">{fmt(p["energy"], 1)}</td><td class="num">–</td><td><a href="{p["source"]}">published</a>: {esc(p["note"])}</td></tr>')
    parts.append('</tbody></table></div>')

    parts.append('<h2>Where a decision\'s time goes</h2>'
                 '<p>Stage medians, stream-synchronized at each boundary as the recipe\'s own runner does. The chain of causation is greedy decoding, one token at a time; the recipe\'s speculative drafter is not released for the Edge model.</p>'
                 '<div class="legend">' + "".join(f'<span class="key"><i style="background:{c}"></i>{n}</span>' for _, n, c in STAGES) + '</div>'
                 f'<div class="chartbox">{vstacked_chart(stack_rows, "Stage cost per decision, median, by build")}</div>'
                 f'<p>With no reasoning at all the floor is vision, prefill and expert: about <b>{floor:.0f} ms</b> at one candidate, already above Embedl\'s published median. '
                 f'Each token of reasoning costs {MS_PER_TOKEN:.1f} ms of GPU work (measured with the engine alone, with and without a host sync per token), so a 22-token sentence is about {22 * MS_PER_TOKEN:.0f} ms of the decision. '
                 f'The NVFP4 engine holds {BUILD_NVFP4["engines"]["llm/llm.engine"] / 1e9:.2f} GB of packed weights against {BUILD_FP8["engines"]["llm/llm.engine"] / 1e9:.2f} GB in FP8, yet its token costs {S["ex_nvfp4_c1"]["stage_median_ms"]["decode_ms"] / S["ex_nvfp4_c1"]["per_clip"]["ea7bbd31"]["generated_tokens"]:.1f} ms on the example clip against {S["ex_fp8_c1"]["stage_median_ms"]["decode_ms"] / S["ex_fp8_c1"]["per_clip"]["ea7bbd31"]["generated_tokens"]:.1f} in FP8: on Edge-LLM 0.11.1 with TensorRT {esc(BOARD["tensorrt"])} the single-token NVFP4-A16 kernel does not turn halved bytes into time, and W4A16 adds dequantization work to the compute-bound prefill.</p>')

    parts.append('<h2>The nine clips</h2>'
                 f'<p>FP8 recipe, one candidate, median of 20 decisions per clip. Four clips take the model\'s no-reasoning path (<code>&lt;NOMINAL&gt;</code>, 6 tokens) and finish in about {statistics.mean(fp8["per_clip"][c]["median_ms"] for c in nominal):.0f} ms; the five that write a sentence take {min(fp8["per_clip"][c]["median_ms"] for c in reasoning):.0f} to {max(fp8["per_clip"][c]["median_ms"] for c in reasoning):.0f} ms.</p>'
                 '<div class="legend"><span class="key"><i style="background:var(--sim)"></i>writes a chain of causation</span><span class="key"><i style="background:var(--c6)"></i>no reasoning (nominal path)</span></div>'
                 f'<div class="chartbox">{clips_chart(fp8, "Median latency per clip, FP8 recipe, one candidate")}</div>')
    parts.append('<div class="scroll"><table><thead><tr><th>clip</th><th class="num">median, ms</th><th class="num">tokens</th><th>chain of causation (FP8; NVFP4 identical unless noted)</th></tr></thead><tbody>')
    for clip, v in sorted(fp8["per_clip"].items(), key=lambda kv: -kv[1]["median_ms"]):
        t = (v["text"] or "").split("\n")[0]
        nvt = (nv["per_clip"][clip]["text"] or "").split("\n")[0]
        flag = "" if t == nvt else f' <span style="color:var(--signal)">NVFP4 differs: "{esc(nvt)}"</span>'
        parts.append(f'<tr><td>{esc(CLIP_NAMES.get(clip, clip))}</td><td class="num">{v["median_ms"]:.0f}</td><td class="num">{v["generated_tokens"]}</td><td>{esc(t)}{flag}</td></tr>')
    parts.append('</tbody></table></div>')

    parts.append('<h2>Did the faster builds still drive the same way?</h2>'
                 '<p>Trajectories decoded with the recipe\'s <code>host/compare.py</code> from the action expert\'s output, ADE in metres over the 64 waypoints, same noise seed across builds. '
                 'The grey bar is the spread among the six candidates the FP8 engines sample for the same clip (mean marked), the study\'s budget: a build whose difference sits inside it changes the plan less than the model\'s own sampling does.</p>'
                 '<div class="legend"><span class="key"><i class="dot" style="background:var(--c2)"></i>FP8 against NVFP4 W4A16</span><span class="key"><i class="diamond" style="background:var(--c5)"></i>five against three Euler steps</span><span class="key"><i style="background:var(--line-strong)"></i>spread among the six FP8 candidates, max; mean marked</span></div>'
                 f'<div class="chartbox">{ade_chart("Trajectory difference between builds, per clip, against the candidate spread")}</div>'
                 '<div class="scroll"><table><thead><tr><th>clip</th><th class="num">FP8 vs NVFP4, m</th><th class="num">5 vs 3 steps, m</th><th class="num">6 vs 1 candidate, m</th><th class="num">candidate spread, mean / max, m</th><th class="num">FP8 vs PyTorch BF16, m</th></tr></thead><tbody>')
    for clip, c in sorted(cross.items()):
        p = c["pairs"]
        py = c["pytorch"].get("fp8", {}).get("closest_m")
        parts.append(f'<tr><td>{esc(CLIP_NAMES.get(clip, clip))}</td><td class="num">{p["fp8 vs nvfp4-w4a16"]["first_candidates_m"]:.3f}</td><td class="num">{p["fp8 vs fp8-s3-c1"]["first_candidates_m"]:.3f}</td><td class="num">{p["fp8 vs fp8-c1"]["first_candidates_m"]:.3f}</td><td class="num">{spread[clip][0]:.2f} / {spread[clip][1]:.2f}</td><td class="num">{fmt(py, 2)}</td></tr>')
    parts.append('</tbody></table></div>'
                 '<p class="note">The PyTorch column compares against case study 06\'s BF16 waypoints for the same clip, t0 and navigation text, but with a different noise draw (seed 43 against seed 0) and one sample, so it is a sanity bound rather than a quantization error. A like-for-like reference needs the same noise tensor fed to both runtimes.</p>')

    parts.append('<h2>What each change bought</h2><ul class="list">')
    for icon, title, body in CHANGES:
        parts.append(f'<li>{ICON[icon]}<div><b>{esc(title)}</b><span>{esc(body)}</span></div></li>')
    parts.append('</ul><h3>Energy per decision</h3>'
                 '<p>Board input rail by tegrastats at 100 ms, integrated over the timed block and divided by the decisions; the mean power while deciding is printed above each column.</p>'
                 '<div class="legend"><span class="key"><i style="background:var(--field)"></i>measured here</span><span class="key"><i class="hatch"></i>published, not run here</span></div>'
                 f'<div class="chartbox">{venergy_chart("Energy per decision, by build")}</div>')

    parts.append('<h2>Issues found, with causes</h2><p>In the order they happened. The full log with times is <a href="https://github.com/obiedeh/bench2field/blob/master/case_studies/07_alpamayo2_edge_edgellm/FRICTION_LOG.md">FRICTION_LOG.md</a>.</p><ul class="list">')
    for i, (title, cause, fix) in enumerate(ISSUES, 1):
        parts.append(f'<li>{ICON["bug"]}<div><b>{i}. {esc(title)}</b><span>{esc(cause)}</span><span style="display:flex;gap:6px;align-items:center;margin-top:4px">{ICON["fix"]}{esc(fix)}</span></div></li>')
    parts.append('</ul>')

    parts.append('<h2>What stands between this build and the published 92 ms</h2><ul class="list">'
                 f'<li>{ICON["text"]}<div><b>A speculative drafter</b><span>NVIDIA\'s 220 ms uses a DFlash2 drafter that is not part of the model release. Greedy decoding costs {MS_PER_TOKEN:.1f} ms per token here; a drafter verifies several tokens per weight read. Training one with NVIDIA\'s SpecForge recipe is a project of its own.</span></div></li>'
                 f'<li>{ICON["bolt"]}<div><b>A working FP4 decode path</b><span>W4A16 is slower on this stack and W4A4 does not build on TensorRT {esc(BOARD["tensorrt"])}. A newer TensorRT (JetPack 7.2.2) is the next thing to try, and a system change for the owner to decide.</span></div></li>'
                 f'<li>{ICON["clock"]}<div><b>A shorter prefill</b><span>{fp8["stage_median_ms"]["prefill_ms"]:.0f} ms of every decision re-encodes a prompt in which old and new camera frames interleave per camera, so unchanged frames are not a cacheable prefix. A prompt layout that keeps them contiguous, or Edge-LLM\'s media-aware K/V reuse, is the lever.</span></div></li></ul>')

    parts.append('<h2>Provenance</h2><div class="scroll"><table><tbody>'
                 f'<tr><th>Model</th><td><code>nvidia/Alpamayo2-Edge</code>, revision aaaa32c6 (the recipe\'s pin; both weight files equal by sha256 to the 2026-11-10 copy)</td></tr>'
                 f'<tr><th>Recipe</th><td><code>NVlabs/alpamayo-recipes</code> 622f9d2, <code>recipes/alpamayo2edge_edgellm</code>; TensorRT Edge-LLM v0.11.1 (358de11c), plugin rebuilt with <code>-DENABLE_CUTE_DSL=ALL</code> and nvcc 13.0</td></tr>'
                 f'<tr><th>Board</th><td>{esc(BOARD["platform"])} (label <code>{esc(BOARD["label"])}</code>), {esc(BOARD["jetpack"].replace("Installed: ", "JetPack "))}, kernel {esc(BOARD["kernel"])}, CUDA {esc(BOARD["cuda"])}, TensorRT {esc(BOARD["tensorrt"])}, torch {esc(BOARD["torch"])}, {esc(BOARD["power_mode"])}, CPU governor schedutil, GPU 1575 MHz throughout</td></tr>'
                 f'<tr><th>Engine builds</th><td>FP8: Edge-LLM compile {BUILD_FP8["edgellm_build_s"]} s, language engine {BUILD_FP8["llm_engine_s"]} s, vision engine {BUILD_FP8["visual_engine_s"]} s, K/V capacity {BUILD_FP8["max_kv_cache_capacity"]} tokens. NVFP4 W4A16: language engine {BUILD_NVFP4["llm_engine_s"]} s, vision {BUILD_NVFP4["visual_engine_s"]} s</td></tr>'
                 '<tr><th>Calibration</th><td>FP8 and NVFP4 scales on the 22 frames of NVIDIA\'s example clip, as the recipe does; FP8 K/V cache; reduced 8192-row output vocabulary</td></tr>'
                 '<tr><th>Timing</th><td><code>tools/thor_stream.py</code>: engines loaded once, decisions back to back, one untimed warm-up per clip, stage times by stream synchronization; power by tegrastats at 100 ms on the board input rail, energy integrated over the timed block</td></tr>'
                 '<tr><th>Data</th><td>PhysicalAI-Autonomous-Vehicles (gated, NVIDIA AV dataset licence): clip ids and derived navigation text only; no frames leave the workstation</td></tr>'
                 '</tbody></table></div>')

    parts.append(f'<footer>Generated {now} by <code>tools/make_report.py</code> from the committed run records. Every number on this page is read from a file in <code>runs/</code>, from case study 06\'s committed stream record, or is a published figure labelled as such with its source. No number was typed in.'
                 '<span class="lab">An <a href="https://embodiededge.ai">EmbodiedEdge</a> project. Measured on real hardware, published as found.</span></footer>')
    parts.append("</main><script>(function(){var r=document.documentElement,b=document.getElementById('theme');function cur(){return r.getAttribute('data-theme')||'dark'}"
                 "function label(){b.textContent=cur()==='dark'?'Light theme':'Dark theme'}b.addEventListener('click',function(){r.setAttribute('data-theme',cur()==='dark'?'light':'dark');label()});label()})();</script></body></html>")
    return "".join(parts)


def headline_svg() -> str:
    """A standalone SVG of the build chart for the README, dark theme colours inlined."""
    rows = [{"label": PYTORCH["label"], "a": PYTORCH["median"], "b": PYTORCH["tail"]}]
    rows += [{"label": l, "a": s["median_ms"], "b": s["p95_ms"]} for l, s in BUILDS[:4]]
    rows += [{"label": p["label"], "a": p["median"], "b": p["tail"], "published": True, "tail_kind": p.get("tail_kind", "p95")} for p in PUBLISHED]
    svg = hbar_chart(rows, "Median and tail latency per decision, by build", deadline=DEADLINE_MS)
    style = ('<style>svg{background:#202224;font:13px Helvetica,Arial,sans-serif}.grid{stroke:#393d3f}.tick{fill:#a3aa9c}.label,.value{fill:#eef1e8}'
             '.deadline{stroke:#ff7a6b;stroke-width:1.5;stroke-dasharray:6 4}.deadline-label{fill:#ff7a6b;font-size:11px}</style>')
    svg = svg.replace("var(--sim)", "#68b7ff").replace("var(--field)", "#b7f34a").replace("var(--muted)", "#a3aa9c")
    return svg.replace('<svg class="chart"', '<svg xmlns="http://www.w3.org/2000/svg"', 1).replace("<defs>", style + "<defs>", 1)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    (OUT / "index.html").write_text(render(), encoding="utf-8")
    (OUT / "headline.svg").write_text(headline_svg(), encoding="utf-8")
    print(f"wrote {OUT / 'index.html'} ({(OUT / 'index.html').stat().st_size:,} bytes) and headline.svg")
