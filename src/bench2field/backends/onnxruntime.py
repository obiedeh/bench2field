"""ONNX Runtime backend: one artifact, many execution providers.

The same .onnx file runs on CPU, CUDA, TensorRT (NVIDIA), MIGraphX or ROCm
(AMD), which is what makes cross-vendor comparisons apples to apples.
Session threading is exposed on purpose: on Jetson Thor, ORT's default
spin-waiting thread pool cost ~29 W of board power at idle-rate loads
(see the jetson-edge-ai-security case study), so it is a tuning knob here,
not a hidden default.
"""

from __future__ import annotations

import ctypes
import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

PROVIDER_ALIASES = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "tensorrt": "TensorrtExecutionProvider",
    "migraphx": "MIGraphXExecutionProvider",
    "rocm": "ROCMExecutionProvider",
}


NVIDIA_PROVIDERS = ("CUDAExecutionProvider", "TensorrtExecutionProvider")


def preload_gpu_libraries(ort: Any, provider: str) -> bool:
    """Load CUDA and cuDNN before an NVIDIA session is created.

    `pip install onnxruntime-gpu[cuda,cudnn]` puts those libraries under
    site-packages/nvidia/, which is not on the loader path, so without this the
    CUDA provider loads and then fails on its first cuDNN call ("dlopen failed
    for libcudnn.so"). onnxruntime.preload_dlls() (1.21+) finds them there and
    falls back to the system copies. Returns whether a preload was attempted.
    """
    if provider not in NVIDIA_PROVIDERS or not hasattr(ort, "preload_dlls"):
        return False
    ort.preload_dlls()
    return True


# Load order matters: the parser and plugin libraries depend on libnvinfer.
TENSORRT_LIBRARIES = ("libnvinfer.so.10", "libnvinfer_plugin.so.10", "libnvonnxparser.so.10")


def preload_tensorrt_libraries() -> list[str]:
    """Load pip-installed TensorRT so onnxruntime's TensorRT provider finds it.

    `pip install tensorrt-cu13` puts the libraries in site-packages/tensorrt_libs,
    off the loader path, and onnxruntime.preload_dlls() does not cover
    TensorRT. Loading them here by full path satisfies the provider's
    dependency on their sonames. Returns the paths loaded: empty when the pip
    package is absent, as on a Jetson, where TensorRT is a system library.
    """
    spec = importlib.util.find_spec("tensorrt_libs")
    if spec is None or not spec.submodule_search_locations:
        return []
    libdir = Path(next(iter(spec.submodule_search_locations)))
    loaded: list[str] = []
    for name in TENSORRT_LIBRARIES:
        path = libdir / name
        if not path.exists():
            continue
        try:
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        except OSError as exc:
            raise RuntimeError(f"could not load TensorRT library {path}: {exc}") from exc
        loaded.append(str(path))
    return loaded


def _decode_nv_version(v: int) -> str:
    """TensorRT, cuDNN 9 and the CUDA runtime all encode major*10000 + minor*100 + patch
    (TensorRT 10.16.1 -> 101601; cuDNN 9.19.0 -> 91900; CUDA 13.0 -> 13000)."""
    return f"{v // 10000}.{v // 100 % 100}.{v % 100}"


def loaded_nvidia_library_versions() -> dict[str, str | None]:
    """Versions of the TensorRT, cuDNN and CUDA runtime libraries this process
    has loaded, asked of the libraries themselves, so a report says which
    copies actually ran. None for a library that is not loaded (there is no
    TensorRT in a CUDA-provider run) or that cannot be asked.

    dlopen by soname returns a library that is already loaded rather than
    searching for another, so this reports the copies the provider is using.
    """
    out: dict[str, str | None] = {}
    queries = (
        ("tensorrt", "libnvinfer.so.10", "getInferLibVersion", ctypes.c_int32),
        ("cudnn", "libcudnn.so.9", "cudnnGetVersion", ctypes.c_size_t),
    )
    for key, soname, symbol, restype in queries:
        try:
            fn = getattr(ctypes.CDLL(soname), symbol)
            fn.restype = restype
            out[key] = _decode_nv_version(int(fn()))
        except (OSError, AttributeError):
            out[key] = None
    out["cuda_runtime"] = None
    for soname in ("libcudart.so.13", "libcudart.so.12"):
        try:
            lib = ctypes.CDLL(soname)
        except OSError:
            continue
        v = ctypes.c_int()
        if lib.cudaRuntimeGetVersion(ctypes.byref(v)) == 0:
            out["cuda_runtime"] = f"{v.value // 1000}.{v.value % 1000 // 10}"
        break
    return out


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

        preload_gpu_libraries(ort, prov)
        if prov == "TensorrtExecutionProvider":
            preload_tensorrt_libraries()

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
        # If a provider's libraries are missing, onnxruntime prints an error,
        # retries on CPU and carries on. A benchmark must not: a "TensorRT"
        # run that quietly measured the CPU is worse than no run.
        active = self.session.get_providers()[0]
        if active != prov:
            raise RuntimeError(
                f"asked for {prov} but onnxruntime fell back to {active}; "
                "its libraries failed to load (see the onnxruntime error above)"
            )
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
        libs = (loaded_nvidia_library_versions() if self.provider() in NVIDIA_PROVIDERS
                else {"tensorrt": None, "cudnn": None, "cuda_runtime": None})
        return {
            "backend": self.name,
            "onnxruntime": self._ort.__version__,
            **libs,
            "provider_active": self.provider(),
            "precision": o.precision,
            "intra_op_threads": o.intra_op_threads or "default",
            "inter_op_threads": o.inter_op_threads or "default",
            "allow_spinning": o.allow_spinning,
        }
