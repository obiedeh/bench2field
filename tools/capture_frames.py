"""Grab JPEG frames from a UVC camera as the camera encodes them.

Used for the rover frames the case studies profile and evaluate on. The
camera's own MJPEG bytes are kept (no decode and re-encode), so a pipeline
fed these frames decodes exactly what it would decode live. Frames are
written as <out>/frame_000123.jpg plus a manifest.json with the capture
settings and a SHA-256 per frame; the frames themselves stay out of git.

    python tools/capture_frames.py --device /dev/video0 --size 1280x720 --frames 300 --out data/rover_frames_720p
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


def capture(device: str, width: int, height: int, n: int, warm: int, out: Path) -> dict:
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {device}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)  # hand over the JPEG bytes, do not decode
    got_w, got_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out.mkdir(parents=True, exist_ok=True)

    for _ in range(warm):  # let exposure and white balance settle
        cap.read()
    frames, t0 = [], time.perf_counter()
    passthrough = True
    while len(frames) < n:
        ok, buf = cap.read()
        if not ok:
            raise SystemExit(f"camera stopped after {len(frames)} frames")
        data = np.asarray(buf).reshape(-1)
        if data.size >= 2 and data[0] == 0xFF and data[1] == 0xD8:  # JPEG SOI: raw MJPEG as hoped
            jpeg = data.tobytes()
        else:  # driver decoded it after all; encode once so the set is still JPEG
            passthrough = False
            ok, enc = cv2.imencode(".jpg", buf, [cv2.IMWRITE_JPEG_QUALITY, 90])
            jpeg = enc.tobytes()
        frames.append(jpeg)
    elapsed = time.perf_counter() - t0
    cap.release()

    manifest_frames = []
    for i, jpeg in enumerate(frames):
        name = f"frame_{i:06d}.jpg"
        (out / name).write_bytes(jpeg)
        manifest_frames.append({"file": name, "bytes": len(jpeg), "sha256": hashlib.sha256(jpeg).hexdigest()})
    sample = cv2.imdecode(np.frombuffer(frames[0], np.uint8), cv2.IMREAD_COLOR)
    manifest = {
        "device": device, "requested": [width, height], "reported": [got_w, got_h],
        "decoded_shape": list(sample.shape), "reported_fps": fps,
        "frames": len(frames), "capture_seconds": round(elapsed, 3), "achieved_fps": round(len(frames) / elapsed, 2),
        "mjpeg_passthrough": passthrough, "warmup_frames_discarded": warm,
        "host": platform.node(), "opencv": cv2.__version__,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "set_sha256": hashlib.sha256(b"".join(f["sha256"].encode() for f in manifest_frames)).hexdigest(),
        "files": manifest_frames,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="/dev/video0")
    p.add_argument("--size", default="1280x720", help="WIDTHxHEIGHT the camera offers")
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--warm", type=int, default=30)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    w, h = (int(x) for x in a.size.lower().split("x"))
    m = capture(a.device, w, h, a.frames, a.warm, Path(a.out))
    print(f"{m['frames']} frames {m['reported'][0]}x{m['reported'][1]} at {m['achieved_fps']} fps, "
          f"mjpeg passthrough={m['mjpeg_passthrough']}, set sha256 {m['set_sha256'][:12]}..., wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
