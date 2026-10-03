"""Every report and manifest names the commit it came from; sweeps can refuse
to run from the wrong checkout."""

import json
import subprocess

import pytest

from bench2field import cli, provenance
from bench2field.provenance import check_expected_commit, git_state


def test_git_state_of_this_checkout():
    s = git_state()
    if s["commit"] is None:
        pytest.skip("not running from a git checkout")
    assert len(s["commit"]) == 40 and isinstance(s["dirty"], bool) and s["root"]


def test_git_state_outside_a_repo(tmp_path):
    assert git_state(tmp_path) == {"commit": None, "dirty": None, "root": None}


def test_git_state_when_git_is_missing(monkeypatch):
    def missing(*a, **k):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", missing)
    assert git_state() == {"commit": None, "dirty": None, "root": None}


def test_expected_commit_check():
    clean = {"commit": "abcdef0123456789" * 2 + "abcdef01", "dirty": False, "root": "/r"}
    assert check_expected_commit("abcdef01", clean) is clean
    assert check_expected_commit("ABCDEF0123", clean) is clean
    with pytest.raises(RuntimeError, match="expected commit 999999, but the checkout is at abcdef012345"):
        check_expected_commit("999999", clean)
    with pytest.raises(RuntimeError, match="has uncommitted changes"):
        check_expected_commit("abcdef01", {**clean, "dirty": True})
    with pytest.raises(RuntimeError, match="not running from a git checkout"):
        check_expected_commit("abcdef01", {"commit": None, "dirty": None, "root": None})


def test_sweep_refuses_the_wrong_checkout(tmp_path, monkeypatch, capsys):
    cfg = tmp_path / "s.yaml"
    cfg.write_text("name: s\nrepeats: 1\ntiers_hz: [30]\nvariants:\n  - {label: a, model: m.onnx}\n")
    monkeypatch.setattr(provenance, "git_state",
                        lambda path=None: {"commit": "f" * 40, "dirty": False, "root": "/r"})
    assert cli.main(["sweep", str(cfg), "--out-dir", str(tmp_path / "o"), "--expect-commit", "abc123"]) == 2
    assert "expected commit abc123, but the checkout is at ffffffffffff" in capsys.readouterr().err
    assert not (tmp_path / "o").exists()  # nothing was started


def test_reports_and_manifests_record_the_commit(tmp_path, monkeypatch):
    from bench2field import sweep as sweep_mod
    from test_sweep import FakeRunner

    state = {"commit": "a" * 40, "dirty": True, "root": "/r"}
    monkeypatch.setattr(sweep_mod, "git_state", lambda path=None: state)
    cfg = tmp_path / "s.yaml"
    cfg.write_text("name: s\nrepeats: 1\ntiers_hz: [30]\nvariants:\n  - {label: a, model: m.onnx}\n")
    monkeypatch.setattr(sweep_mod, "subprocess_runner", FakeRunner())
    assert cli.main(["sweep", str(cfg), "--out-dir", str(tmp_path / "o")]) == 0
    assert json.loads((tmp_path / "o" / "sweep_s.json").read_text())["git"] == state


def test_b2f_run_records_the_commit(tmp_path, monkeypatch):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    from bench2field.schema import RunReport

    w = numpy_helper.from_array(np.eye(4, dtype=np.float32), "W")
    g = helper.make_graph([helper.make_node("MatMul", ["x", "W"], ["y"])], "t",
                          [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])], [w])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.save(m, tmp_path / "t.onnx")
    monkeypatch.setattr(provenance, "git_state", lambda path=None: {"commit": "b" * 40, "dirty": False, "root": "/r"})
    assert cli.main(["run", str(tmp_path / "t.onnx"), "--tiers", "50", "--duration", "0.1", "--warmup", "0",
                     "--no-telemetry", "--out", str(tmp_path / "r.json")]) == 0
    assert RunReport.load(tmp_path / "r.json").platform["git"] == {"commit": "b" * 40, "dirty": False, "root": "/r"}
