"""Check the standalone C++ driver against the PyTorch extension.

Runs after ``preprocess_standalone`` has written an output file and a tables
prefix. Loads the same raw frames, runs the extension with the same variant,
and reports the maximum absolute difference of the outputs (expected 0: same
kernel, same tables, same device) and of the C++ tables against the Python
tables (expected 0 for the start indices and at most float rounding for the
weights). Exits non-zero if either exceeds its gate.

Usage (from the case study directory, with the case study venv):
    python kernels/standalone/compare_standalone.py <frames.u8> <N> <variant> <out.f32> <tables_prefix>
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import preprocess_ext as pe  # noqa: E402


def main(argv: list[str]) -> int:
    frames_path, n, variant, out_path, prefix = argv[0], int(argv[1]), argv[2], argv[3], argv[4]
    frames = torch.from_numpy(np.fromfile(frames_path, dtype=np.uint8).reshape(n, 3, pe.IN_H, pe.IN_W)).cuda()
    tabs = pe.tables(frames.device)
    ref = pe.preprocess(frames, tabs, variant=variant).cpu().numpy().ravel()
    got = np.fromfile(out_path, dtype=np.float32)
    if got.size != ref.size:
        print(f"output size {got.size} != {ref.size}")
        return 2
    out_err = float(np.abs(got - ref).max())
    ys = np.fromfile(f"{prefix}_y_start.i32", dtype=np.int32)
    xs = np.fromfile(f"{prefix}_x_start.i32", dtype=np.int32)
    yw = np.fromfile(f"{prefix}_y_weight.f32", dtype=np.float32).reshape(pe.OUT_H, pe.MAX_TAPS)
    xw = np.fromfile(f"{prefix}_x_weight.f32", dtype=np.float32).reshape(pe.OUT_W, pe.MAX_TAPS)
    start_err = int(max(np.abs(ys - tabs["y_start"].cpu().numpy()).max(), np.abs(xs - tabs["x_start"].cpu().numpy()).max()))
    weight_err = float(max(np.abs(yw - tabs["y_weight"].cpu().numpy()).max(), np.abs(xw - tabs["x_weight"].cpu().numpy()).max()))
    print(f"[compare] variant {variant}: output max abs diff {out_err:.3e} (gate 0), "
          f"table start max diff {start_err} (gate 0), table weight max diff {weight_err:.3e} (gate 1e-6)")
    return 0 if out_err == 0.0 and start_err == 0 and weight_err <= 1e-6 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
