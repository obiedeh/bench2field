"""Write a small conv net to ONNX for hardware bring-up runs.

Not a real model: four 3x3 conv+ReLU layers and a global pool on a
1x3x224x224 input, with fixed random weights. It is just big enough to put
measurable load on a GPU, and small enough to build a TensorRT engine in
seconds. Needs `onnx`.

    python tools/make_tiny_model.py models/tiny_conv.onnx
"""

from __future__ import annotations

import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def build(channels: tuple[int, ...] = (32, 64, 64, 32), size: int = 224) -> onnx.ModelProto:
    rng = np.random.default_rng(0)
    nodes, inits, prev, c_in = [], [], "x", 3
    for i, c_out in enumerate(channels):
        w = (rng.standard_normal((c_out, c_in, 3, 3)) * 0.05).astype(np.float32)
        inits += [numpy_helper.from_array(w, f"w{i}"),
                  numpy_helper.from_array(np.zeros(c_out, np.float32), f"b{i}")]
        nodes += [helper.make_node("Conv", [prev, f"w{i}", f"b{i}"], [f"c{i}"], pads=[1, 1, 1, 1]),
                  helper.make_node("Relu", [f"c{i}"], [f"r{i}"])]
        prev, c_in = f"r{i}", c_out
    nodes.append(helper.make_node("GlobalAveragePool", [prev], ["y"]))
    graph = helper.make_graph(
        nodes, "tiny_conv",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, size, size])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, c_in, 1, 1])], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "models/tiny_conv.onnx"
    onnx.save(build(), out)
    print(f"wrote {out}")
