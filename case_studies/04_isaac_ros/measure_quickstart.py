#!/usr/bin/env python3
"""Measure the Isaac ROS YOLOv8 quickstart pipeline from inside the container.

Subscribes to the pipeline's input image topic and its detections output for
a fixed window and writes one JSON record: message counts and rates for both,
detections per frame, the class ids seen, and the pipeline's end-to-end
latency measured as the gap between an image header stamp and the detections
message that carries the same stamp (Isaac ROS keeps the stamp through the
graph). Runs with the system ROS 2 in the container; rclpy only.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray


class Meter(Node):
    def __init__(self, image_topic: str, det_topic: str) -> None:
        super().__init__("quickstart_meter")
        # The bag publishes images reliably; a best-effort subscriber on the
        # loopback drops most 1.2 MB frames (UDP fragments), so the image side
        # must be reliable or the input count is wrong. Detections are small.
        self.create_subscription(Image, image_topic, self.on_image,
                                 QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE))
        self.create_subscription(Detection2DArray, det_topic, self.on_det,
                                 QoSProfile(depth=50, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.images = 0
        self.dets = 0
        self.first_image = None
        self.first_det = None
        self.last_image = None
        self.last_det = None
        self.per_frame: list[int] = []
        self.classes: Counter = Counter()
        self.scores: list[float] = []
        self.stamps: dict[tuple[int, int], float] = {}
        self.latency_ms: list[float] = []
        self.image_encoding = None
        self.image_shape = None

    def on_image(self, msg: Image) -> None:
        now = time.time()
        self.images += 1
        self.first_image = self.first_image or now
        self.last_image = now
        self.image_encoding = msg.encoding
        self.image_shape = [msg.height, msg.width]
        key = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        self.stamps[key] = now
        if len(self.stamps) > 500:
            self.stamps.pop(next(iter(self.stamps)))

    def on_det(self, msg: Detection2DArray) -> None:
        now = time.time()
        self.dets += 1
        self.first_det = self.first_det or now
        self.last_det = now
        self.per_frame.append(len(msg.detections))
        for d in msg.detections:
            for h in d.results:
                self.classes[h.hypothesis.class_id] += 1
                self.scores.append(float(h.hypothesis.score))
        key = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        t_in = self.stamps.get(key)
        if t_in is not None:
            self.latency_ms.append(1000.0 * (now - t_in))


def pct(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return round(s[min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1))))], 2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-topic", default="/image_rect")
    ap.add_argument("--det-topic", default="/detections_output")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    rclpy.init()
    node = Meter(args.image_topic, args.det_topic)
    t0 = time.time()
    while time.time() - t0 < args.seconds:
        rclpy.spin_once(node, timeout_sec=0.1)
    window = time.time() - t0
    rec = {
        "schema": "bench2field.isaac_ros.quickstart/1",
        "label": args.label,
        "window_s": round(window, 1),
        "input": {"topic": args.image_topic, "messages": node.images, "hz": round(node.images / window, 2),
                  "encoding": node.image_encoding, "shape_hw": node.image_shape},
        "output": {"topic": args.det_topic, "messages": node.dets, "hz": round(node.dets / window, 2),
                   "detections_per_frame_mean": round(statistics.mean(node.per_frame), 2) if node.per_frame else None,
                   "detections_per_frame_min_max": [min(node.per_frame), max(node.per_frame)] if node.per_frame else None,
                   "frames_with_no_detection": sum(1 for n in node.per_frame if n == 0),
                   "class_ids": dict(node.classes.most_common(10)),
                   "score_median": round(statistics.median(node.scores), 3) if node.scores else None},
        "latency_ms_image_stamp_to_detections": {"n": len(node.latency_ms), "p50": pct(node.latency_ms, 50),
                                                  "p95": pct(node.latency_ms, 95), "max": pct(node.latency_ms, 100),
                                                  "note": "wall time between the meter receiving the image and receiving the detections with the same header stamp; includes the meter's own subscription path"},
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with open(args.out, "w") as f:
        json.dump(rec, f, indent=1)
        f.write("\n")
    print(json.dumps(rec, indent=1))
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
