"""The report records what else the machine was doing."""

import os
import subprocess
import types

import pytest

from bench2field import background
from bench2field.background import snapshot


def fake_run(docker_out="", docker_rc=0, ps_out=""):
    def run(argv, **kw):
        if argv[0] == "docker":
            return types.SimpleNamespace(returncode=docker_rc, stdout=docker_out)
        return types.SimpleNamespace(returncode=0, stdout=ps_out)
    return run


def test_snapshot_records_containers_processes_and_what_was_stopped(monkeypatch):
    monkeypatch.setattr(background.shutil, "which", lambda name: "/usr/bin/docker")
    me = os.getpid()
    ps = (f"{me} 99.0 1.0 python -m pytest\n"
          "4321 87.5 2.1 /opt/venv/bin/python -m vllm.entrypoints.openai.api_server --model x\n"
          "2222 12.0 0.3 openclaw-gateway\n"
          "  10  0.0 0.0 [kworker/0:1]\n")
    monkeypatch.setattr(subprocess, "run", fake_run("physical-ai-vllm\tghcr.io/x/vllm:0.14\tUp 11 hours\n", 0, ps))
    s = snapshot(["docker container urban-edge-vllm"])
    assert s["stopped_for_this_run"] == ["docker container urban-edge-vllm"]
    assert s["containers_running"] == [{"name": "physical-ai-vllm", "image": "ghcr.io/x/vllm:0.14", "status": "Up 11 hours"}]
    assert [p["pid"] for p in s["top_processes"]] == [4321, 2222, 10]  # the test runner itself is left out
    assert s["top_processes"][0]["cpu_pct"] == 87.5 and "vllm" in s["top_processes"][0]["cmd"]
    assert isinstance(s["load_avg_1m"], float)


def test_no_docker_is_recorded_as_unknown_not_empty(monkeypatch):
    monkeypatch.setattr(background.shutil, "which", lambda name: None)
    monkeypatch.setattr(subprocess, "run", fake_run(ps_out="1 0.0 0.0 init\n"))
    assert snapshot()["containers_running"] is None
    monkeypatch.setattr(background.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(subprocess, "run", fake_run("permission denied", docker_rc=1, ps_out="1 0.0 0.0 init\n"))
    assert snapshot()["containers_running"] is None  # docker there, but we could not ask it


def test_snapshot_survives_a_missing_ps(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("ps")

    monkeypatch.setattr(background.shutil, "which", lambda name: None)
    monkeypatch.setattr(subprocess, "run", boom)
    s = snapshot()
    assert s["top_processes"] == [] and s["stopped_for_this_run"] == []


def test_real_snapshot_on_this_machine():
    s = snapshot(["nothing"])
    assert s["stopped_for_this_run"] == ["nothing"]
    assert all(p["pid"] != os.getpid() for p in s["top_processes"])
    assert len(s["top_processes"]) <= background.TOP_N
