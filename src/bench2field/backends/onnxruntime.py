"""ONNX Runtime backend: one artifact, many execution providers.

The same .onnx file runs on CPU, CUDA, TensorRT (NVIDIA), MIGraphX or ROCm
(AMD), which is what makes cross-vendor comparisons apples to apples.
Session threading is exposed on purpose: on Jetson Thor, ORT's default
spin-waiting thread pool cost ~29 W of board power at idle-rate loads
(see the jetson-edge-ai-security case study), so it is a tuning knob here,
not a hidden default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

PROVIDER_ALIASES = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "tensorrt": "TensorrtExecutionProvider",
    "migraphx": "MIGraphXExecutionProvider",
    "rocm": "ROCMExecutionProvider",
}


@dataclass
class OrtOptions:
    provider: str = "cpu"
    precision: str = "fp32"          # fp32 | fp16 | int8 (TensorRT/MIGraphX only)
    intra_op_threads: int = 0        # 0 = runtime default
    inter_op_threads: int = 0
    allow_spinning: bool = True
    trt_cache_dir: str = ".trt_cache"
    int8_calibration_table: str | None = None
    extra_provider_options: dict[str, Any] = field(default_factory=dict)


class OnnxRuntimeBackend:
    name = "onnxruntime"

    def __init__(self, model_path: str, options: OrtOptions | None = None) -> None:
        import onnxruntime as ort

        self.options = opts = options or OrtOptions()
        self._ort = ort
        prov = PROVIDER_ALIASES.get(opts.provider, opts.provider)
        available = ort.get_available_providers()
        if prov not in available:
            raise RuntimeError(f"{prov} not available; this build has {available}")

        so = ort.SessionOptions()
        if opts.intra_op_threads:
            so.intra_op_num_threads = opts.intra_op_threads
        if opts.inter_op_threads:
            so.inter_op_num_threads = opts.inter_op_threads
        so.add_session_config_entry(
            "session.intra_op.allow_spinning", "1" if opts.allow_spinning else "0"
        )

        prov_opts: dict[str, Any] = dict(opts.extra_provider_options)
        if prov == "TensorrtExecutionProvider":
            prov_opts.setdefault("trt_engine_cache_enable", True)
            prov_opts.setdefault("trt_engine_cache_path", opts.trt_cache_dir)
            if opts.precision == "fp16":
                prov_opts["trt_fp16_enable"] = True
            elif opts.precision == "int8":
                prov_opts["trt_int8_enable"] = True
                if opts.int8_calibration_table:
                    prov_opts["trt_int8_calibration_table_name"] = opts.int8_calibration_table
        elif prov == "MIGraphXExecutionProvider":
            if opts.precision == "fp16":
                prov_opts["migraphx_fp16_enable"] = True
            elif opts.precision == "int8":
                prov_opts["migraphx_int8_enable"] = True
        # CPU, CUDA and ROCm providers run whatever precision the graph was
        # exported in, so for them `precision` is a label for the report.

        providers: list[Any] = [(prov, prov_opts)] if prov_opts else [prov]
        if prov != "CPUExecutionProvider":
            providers.append("CPUExecutionProvider")  # fallback for unsupported ops
        self.session = ort.InferenceSession(model_path, sess_options=so, providers=providers)
        self.model_path = model_path
        self._inputs = self.session.get_inputs()

    def provider(self) -> str:
        return self.session.get_providers()[0]

    def input_specs(self) -> list[tuple[str, list[Any], str]]:
        return [(i.name, list(i.shape), i.type) for i in self._inputs]

    def synthetic_input(self, batch: int = 1, seed: int = 0) -> dict[str, np.ndarray]:
        """Random tensors matching the model's inputs (dynamic dims -> batch or 1)."""
        rng = np.random.default_rng(seed)
        feeds: dict[str, np.ndarray] = {}
        for name, shape, typ in self.input_specs():
            dims = [batch if i == 0 and not isinstance(d, int) else (d if isinstance(d, int) else 1)
                    for i, d in enumerate(shape)]
            dtype = np.float16 if "float16" in typ else np.float32
            feeds[name] = rng.standard_normal(dims).astype(dtype)
        return feeds

    def infer(self, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        return self.session.run(None, inputs)

    def describe(self) -> dict[str, Any]:
        o = self.options
        return {
            "backend": self.name,
            "onnxruntime": self._ort.__version__,
            "provider_active": self.provider(),
            "precision": o.precision,
            "intra_op_threads": o.intra_op_threads or "default",
            "inter_op_threads": o.inter_op_threads or "default",
            "allow_spinning": o.allow_spinning,
        }
