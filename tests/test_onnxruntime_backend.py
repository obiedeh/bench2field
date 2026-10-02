import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

import numpy as np  # noqa: E402
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from bench2field.backends.onnxruntime import OnnxRuntimeBackend, OrtOptions  # noqa: E402


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
