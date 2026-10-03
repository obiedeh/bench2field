import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

import numpy as np  # noqa: E402
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from bench2field.backends.onnxruntime import (  # noqa: E402
    OnnxRuntimeBackend,
    OrtOptions,
    _decode_nv_version,
    loaded_nvidia_library_versions,
    preload_gpu_libraries,
    preload_tensorrt_libraries,
    synthetic_tensor,
)


@pytest.fixture
def tiny_model(tmp_path):
    w = numpy_helper.from_array(np.eye(8, dtype=np.float32), "W")
    g = helper.make_graph([helper.make_node("MatMul", ["x", "W"], ["y"])], "tiny",
                          [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 8])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", 8])], [w])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    path = tmp_path / "tiny.onnx"
    onnx.save(m, path)
    return str(path)


def test_cpu_backend_runs_with_thread_options(tiny_model):
    be = OnnxRuntimeBackend(tiny_model, OrtOptions(intra_op_threads=1, inter_op_threads=1,
                                                   allow_spinning=False))
    feeds = be.synthetic_input(batch=2)
    (y,) = be.infer(feeds)
    assert y.shape == (2, 8)
    np.testing.assert_allclose(y, feeds["x"], rtol=1e-6)
    d = be.describe()
    assert d["provider_active"] == "CPUExecutionProvider" and d["allow_spinning"] is False
    assert (d["tensorrt"], d["cudnn"], d["cuda_runtime"]) == (None, None, None)  # not an NVIDIA run


def test_missing_provider_is_a_clear_error(tiny_model):
    with pytest.raises(RuntimeError, match="not available"):
        OnnxRuntimeBackend(tiny_model, OrtOptions(provider="migraphx"))


def test_b2f_run_closes_the_sampler_even_when_the_run_fails(tiny_model, tmp_path, monkeypatch):
    from bench2field import cli
    from bench2field.telemetry import NullSampler

    closed = []

    class Sampler(NullSampler):
        def close(self):
            closed.append(True)

    monkeypatch.setattr("bench2field.telemetry.auto_sampler", lambda interval_s=0.5: Sampler())
    args = ["run", tiny_model, "--tiers", "50", "--duration", "0.1", "--warmup", "0"]
    assert cli.main([*args, "--out", str(tmp_path / "r.json"), "--stopped", "docker container x"]) == 0
    from bench2field.schema import RunReport

    bg = RunReport.load(tmp_path / "r.json").platform["background"]
    assert bg["stopped_for_this_run"] == ["docker container x"] and "top_processes" in bg
    assert cli.main([*args, "--environment", "lab"]) == 2  # rejected label
    assert closed == [True, True]


def test_gpu_libraries_are_preloaded_for_nvidia_providers_only():
    import types

    calls = []
    ort = types.SimpleNamespace(preload_dlls=lambda: calls.append(1))
    assert preload_gpu_libraries(ort, "CUDAExecutionProvider")
    assert preload_gpu_libraries(ort, "TensorrtExecutionProvider")
    assert not preload_gpu_libraries(ort, "CPUExecutionProvider")
    assert not preload_gpu_libraries(ort, "MIGraphXExecutionProvider")
    assert len(calls) == 2
    old_ort = types.SimpleNamespace()  # onnxruntime < 1.21 has no preload_dlls
    assert not preload_gpu_libraries(old_ort, "CUDAExecutionProvider")


@pytest.fixture
def conv_model(tmp_path):
    w = numpy_helper.from_array(np.ones((4, 3, 3, 3), dtype=np.float32), "w")
    g = helper.make_graph([helper.make_node("Conv", ["x", "w"], ["y"], pads=[1, 1, 1, 1])], "conv",
                          [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 16, 16])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 16, 16])], [w])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.save(m, tmp_path / "conv.onnx")
    return str(tmp_path / "conv.onnx")


def test_cuda_provider_runs_a_conv(conv_model):
    """Real hardware only. A Conv needs cuDNN, which is what a pip-installed
    onnxruntime-gpu fails to find unless its libraries are preloaded."""
    import onnxruntime as ort

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        pytest.skip("this onnxruntime build has no CUDA provider")
    try:
        be = OnnxRuntimeBackend(conv_model, OrtOptions(provider="cuda"))
    except Exception as exc:  # GPU build installed on a machine with no usable GPU
        pytest.skip(f"CUDA session could not be created: {exc}")
    if be.provider() != "CUDAExecutionProvider":
        pytest.skip("onnxruntime fell back to CPU: no usable GPU")
    feeds = be.synthetic_input()
    (y,) = be.infer(feeds)
    assert y.shape == (1, 4, 16, 16)
    # All-ones 3x3 kernels: an interior output is the sum of its 3x3x3 input patch.
    assert y[0, 0, 5, 5] == pytest.approx(feeds["x"][0, :, 4:7, 4:7].sum(), rel=1e-4, abs=1e-4)


def test_silent_fallback_to_cpu_is_an_error(tiny_model, monkeypatch):
    """Seen on the RTX 5090 with no TensorRT libraries installed: the
    provider is listed as available, session creation logs an error and
    comes back on CPU, and the run would have gone ahead."""
    import onnxruntime as ort

    real_session = ort.InferenceSession
    monkeypatch.setattr(ort, "get_available_providers",
                        lambda: ["TensorrtExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(ort, "preload_dlls", lambda: None, raising=False)
    monkeypatch.setattr(ort, "InferenceSession", lambda path, sess_options=None, providers=None:
                        real_session(path, sess_options=sess_options, providers=["CPUExecutionProvider"]))
    with pytest.raises(RuntimeError, match="asked for TensorrtExecutionProvider but .* fell back to CPU"):
        OnnxRuntimeBackend(tiny_model, OrtOptions(provider="tensorrt", precision="fp16"))


def test_tensorrt_preload_is_a_no_op_without_the_pip_package(monkeypatch):
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert preload_tensorrt_libraries() == []


def test_tensorrt_preload_loads_libnvinfer_first_and_globally(tmp_path, monkeypatch):
    import ctypes
    import importlib.util
    import types

    for name in ("libnvonnxparser.so.10", "libnvinfer.so.10", "libnvinfer_builder_resource_sm120.so.10.16.1"):
        (tmp_path / name).write_bytes(b"")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: types.SimpleNamespace(
        submodule_search_locations=[str(tmp_path)]))
    calls = []
    monkeypatch.setattr(ctypes, "CDLL", lambda path, mode=0: calls.append((path, mode)))
    loaded = preload_tensorrt_libraries()
    # The plugin library is absent here and skipped; builder resources are left to libnvinfer.
    assert loaded == [str(tmp_path / "libnvinfer.so.10"), str(tmp_path / "libnvonnxparser.so.10")]
    assert calls == [(p, ctypes.RTLD_GLOBAL) for p in loaded]


def test_tensorrt_library_that_will_not_load_is_a_clear_error(tmp_path, monkeypatch):
    import importlib.util
    import types

    (tmp_path / "libnvinfer.so.10").write_bytes(b"not an ELF file")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: types.SimpleNamespace(
        submodule_search_locations=[str(tmp_path)]))
    with pytest.raises(RuntimeError, match="could not load TensorRT library"):
        preload_tensorrt_libraries()


def test_tensorrt_provider_runs_a_conv_in_fp16(conv_model, tmp_path):
    """Real hardware only: needs a TensorRT-enabled onnxruntime and TensorRT 10."""
    import ctypes
    import importlib.util

    import onnxruntime as ort

    if "TensorrtExecutionProvider" not in ort.get_available_providers():
        pytest.skip("this onnxruntime build has no TensorRT provider")
    if importlib.util.find_spec("tensorrt_libs") is None:
        try:
            ctypes.CDLL("libnvinfer.so.10")
        except OSError:
            pytest.skip("TensorRT 10 is not installed")
    be = OnnxRuntimeBackend(conv_model, OrtOptions(provider="tensorrt", precision="fp16",
                                                   trt_cache_dir=str(tmp_path / "trt")))
    assert be.provider() == "TensorrtExecutionProvider"
    feeds = be.synthetic_input()
    (y,) = be.infer(feeds)
    assert y.shape == (1, 4, 16, 16)
    assert y[0, 0, 5, 5] == pytest.approx(feeds["x"][0, :, 4:7, 4:7].sum(), rel=2e-2, abs=2e-2)
    d = be.describe()  # the libraries that ran are named in the report
    assert d["tensorrt"].startswith("10.") and d["cudnn"].startswith("9.") and d["cuda_runtime"]


def test_nvidia_version_encoding():
    # Values read on real machines: TensorRT 10.16.1 and cuDNN 9.19.0 on the 5090,
    # TensorRT 10.13.3 and cuDNN 9.12.0 on the Thor.
    assert _decode_nv_version(101601) == "10.16.1" and _decode_nv_version(91900) == "9.19.0"
    assert _decode_nv_version(101303) == "10.13.3" and _decode_nv_version(91200) == "9.12.0"


def test_library_versions_are_none_when_nothing_is_loaded(monkeypatch):
    import ctypes

    def no_such_library(name, mode=0):
        raise OSError(f"{name}: cannot open shared object file")

    monkeypatch.setattr(ctypes, "CDLL", no_such_library)
    assert loaded_nvidia_library_versions() == {"tensorrt": None, "cudnn": None, "cuda_runtime": None}


def test_synthetic_inputs_follow_each_input_dtype(tmp_path):
    """Two inputs like RT-DETR's: a float image and int64 sizes."""
    sizes = helper.make_tensor_value_info("orig_target_sizes", TensorProto.INT64, ["N", 2])
    img = helper.make_tensor_value_info("images", TensorProto.FLOAT, ["N", 3, 4, 4])
    g = helper.make_graph([helper.make_node("Cast", ["orig_target_sizes"], ["y"], to=TensorProto.FLOAT),
                           helper.make_node("Identity", ["images"], ["z"])], "two",
                          [img, sizes],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", 2]),
                           helper.make_tensor_value_info("z", TensorProto.FLOAT, ["N", 3, 4, 4])])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.save(m, tmp_path / "two.onnx")
    be = OnnxRuntimeBackend(str(tmp_path / "two.onnx"))
    feeds = be.synthetic_input(batch=2)
    assert feeds["images"].dtype == np.float32 and feeds["images"].shape == (2, 3, 4, 4)
    assert feeds["orig_target_sizes"].dtype == np.int64 and feeds["orig_target_sizes"].tolist() == [[1, 1], [1, 1]]
    y, z = be.infer(feeds)
    assert y.shape == (2, 2) and z.shape == (2, 3, 4, 4)


def test_synthetic_tensor_dtypes():
    rng = np.random.default_rng(0)
    assert synthetic_tensor([2], "tensor(float16)", rng).dtype == np.float16
    assert synthetic_tensor([2], "tensor(double)", rng).dtype == np.float64
    assert synthetic_tensor([3], "tensor(int32)", rng).tolist() == [1, 1, 1]
    assert synthetic_tensor([2], "tensor(bool)", rng).tolist() == [True, True]
    u8 = synthetic_tensor([1000], "tensor(uint8)", rng)
    assert u8.dtype == np.uint8 and u8.min() >= 0 and u8.max() <= 255 and u8.std() > 0
    with pytest.raises(ValueError, match="no synthetic input"):
        synthetic_tensor([1], "tensor(string)", rng)
