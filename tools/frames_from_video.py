"""Turn a video into a JPEG frame set with the same manifest layout as
tools/capture_frames.py, for profiling on a busy scene when the camera at
hand is looking at a wall.

    python tools/frames_from_video.py video.mp4 --frames 300 --quality 90 --out data/scene_frames
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video")
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--skip", type=int, default=0, help="frames to skip at the start")
    p.add_argument("--quality", type=int, default=90, help="JPEG quality")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {a.video}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files, shape = [], None
    for _ in range(a.skip):
        cap.read()
    while len(files) < a.frames:
        ok, frame = cap.read()
        if not ok:
            raise SystemExit(f"video ended after {len(files)} frames")
        shape = list(frame.shape)
        ok, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, a.quality])
        jpeg = enc.tobytes()
        name = f"frame_{len(files):06d}.jpg"
        (out / name).write_bytes(jpeg)
        files.append({"file": name, "bytes": len(jpeg), "sha256": hashlib.sha256(jpeg).hexdigest()})
    manifest = {
        "source_video": a.video, "source_sha256": hashlib.sha256(Path(a.video).read_bytes()).hexdigest(),
        "skipped": a.skip, "jpeg_quality": a.quality, "decoded_shape": shape, "frames": len(files),
        "host": platform.node(), "opencv": cv2.__version__, "captured_at": datetime.now(timezone.utc).isoformat(),
        "set_sha256": hashlib.sha256(b"".join(f["sha256"].encode() for f in files)).hexdigest(),
        "files": files,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"{len(files)} frames {shape[1]}x{shape[0]} from {a.video}, set sha256 {manifest['set_sha256'][:12]}..., wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
