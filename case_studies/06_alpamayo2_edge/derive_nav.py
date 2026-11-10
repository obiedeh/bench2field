"""Derive the navigation instruction for each clip of case study 06 by a fixed rule.

Alpamayo 2 Edge needs a route instruction per sample ("Turn right in 30m"), and the
public PhysicalAI-AV dataset does not carry one. A car's route planner knows the
route, so this script derives the instruction from the clip's logged future path,
by a rule fixed before any timed run (PLAN.md, Protocol):

- Take the 64 logged ego poses from t0 + 0.1 s to t0 + 6.4 s, in the ego frame at t0.
- Heading change = yaw of the last pose relative to t0. If its magnitude is at least
  30 degrees, the instruction is "Turn left in N m" (positive yaw, x forward and
  y left) or "Turn right in N m"; N is the path length from t0 to the first pose
  whose heading differs by more than 10 degrees, rounded to the nearest 10 m with
  a minimum of 10 m.
- Otherwise the instruction is "Continue straight".

It also writes the logged future (64 x 3, ego frame) to a separate file outside git,
so the runner can report minADE against it, the same sanity value case study 02 recorded for the teacher.
The instruction uses logged data; the write-up says so.

Usage (inside the Alpamayo 2 Edge environment, which has physical_ai_av):
    python derive_nav.py --clips ../02_alpamayo_av_reasoning/clips.yaml --out nav_clips.json --futures /mnt/backup/<run folder>/nav_futures.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml

TURN_DEG = 30.0
START_DEG = 10.0


def future_in_ego_frame(avdi, clip_id: str, t0_us: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the 64 future positions (m) and yaws (rad) in the ego frame at t0."""
    ego = avdi.get_clip_feature(clip_id, avdi.features.LABELS.EGOMOTION, maybe_stream=True)
    ts = t0_us + np.arange(0, 6_400_001, 100_000, dtype=np.int64)  # t0 and 64 future steps
    poses = ego(ts)
    rot = poses.pose.rotation
    xyz = np.asarray(poses.pose.translation)
    inv = rot[0].inv()
    local = inv.apply(xyz - xyz[0])[1:]
    yaw = (inv * rot).as_euler("zyx")[1:, 0]
    return local, np.unwrap(yaw)


def nav_rule(local: np.ndarray, yaw: np.ndarray) -> tuple[str, dict]:
    """Apply the fixed rule; return the instruction and the numbers behind it."""
    final_deg = math.degrees(float(yaw[-1]))
    steps = np.linalg.norm(np.diff(np.vstack([[0, 0, 0], local]), axis=0), axis=1)
    path_len = np.cumsum(steps)
    detail = {"final_heading_deg": round(final_deg, 1), "path_length_m": round(float(path_len[-1]), 1)}
    if abs(final_deg) < TURN_DEG:
        return "Continue straight", detail
    start = int(np.argmax(np.abs(np.degrees(yaw)) > START_DEG))
    dist = max(10, int(round(float(path_len[start]) / 10.0)) * 10)
    side = "left" if final_deg > 0 else "right"
    detail["turn_starts_at_m"] = round(float(path_len[start]), 1)
    return f"Turn {side} in {dist}m", detail


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clips", required=True, help="clips.yaml of case study 02")
    ap.add_argument("--out", required=True, help="output JSON with the nav text (committed)")
    ap.add_argument("--futures", required=True, help="output JSON that also holds the logged futures (kept outside git: gated dataset)")
    args = ap.parse_args()

    import physical_ai_av

    cfg = yaml.safe_load(Path(args.clips).read_text())
    t0 = int(cfg["t0_us"])
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    rows = []
    for clip in cfg["clips"]:
        local, yaw = future_in_ego_frame(avdi, clip, t0)
        nav, detail = nav_rule(local, yaw)
        rows.append({"clip_id": clip, "t0_us": t0, "nav_text": nav, **detail,
                     "future_xyz_ego": np.round(local, 4).tolist()})
        print(f"{clip}  {nav:<22} {detail}")
    out = {"rule": {"turn_deg": TURN_DEG, "start_deg": START_DEG, "horizon_s": 6.4,
                    "source": "logged egomotion, PhysicalAI-AV"},
           "dataset_revision": getattr(avdi, "revision", None), "samples": rows}
    Path(args.futures).write_text(json.dumps(out))
    for r in out["samples"]:
        r.pop("future_xyz_ego")
    out["futures"] = "logged future poses kept outside git (gated dataset); regenerate with derive_nav.py --futures"
    Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
