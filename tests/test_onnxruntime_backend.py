import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

import numpy as np  # noqa: E402
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from bench2field.backends.onnxruntime import (  # noqa: E402
    OnnxRuntimeBackend,
    OrtOptions,
    preload_gpu_libraries,
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
    assert cli.main([*args, "--out", str(tmp_path / "r.json")]) == 0
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


def test_cuda_provider_runs_a_conv(tmp_path):
    """Real hardware only. A Conv needs cuDNN, which is what a pip-installed
    onnxruntime-gpu fails to find unless its libraries are preloaded."""
    import onnxruntime as ort

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        pytest.skip("this onnxruntime build has no CUDA provider")
    w = numpy_helper.from_array(np.ones((4, 3, 3, 3), dtype=np.float32), "w")
    g = helper.make_graph([helper.make_node("Conv", ["x", "w"], ["y"], pads=[1, 1, 1, 1])], "conv",
                          [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 16, 16])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 16, 16])], [w])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.save(m, tmp_path / "conv.onnx")
    try:
        be = OnnxRuntimeBackend(str(tmp_path / "conv.onnx"), OrtOptions(provider="cuda"))
    except Exception as exc:  # GPU build installed on a machine with no usable GPU
        pytest.skip(f"CUDA session could not be created: {exc}")
    if be.provider() != "CUDAExecutionProvider":
        pytest.skip("onnxruntime fell back to CPU: no usable GPU")
    (y,) = be.infer(be.synthetic_input())
    assert y.shape == (1, 4, 16, 16)
