"""Case study 02 (Alpamayo 1) keeps weights, clips and its venv out of git,
and its committed environment records name a board, never a machine.

The first test mirrors test_repo_hygiene for the case study's own ignore
rules. The second reads every runs/env_*.json the case study has committed
and checks the contract capture_env.py promises: a known board label, the
package versions a reader needs, and no hostname or home directory path."""

import json
import shutil
import socket
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "case_studies" / "02_alpamayo_av_reasoning"
BOARDS = {"bench-5090", "bench-thor", "field-orin"}


def _ignored(rel: str) -> bool:
    out = subprocess.run(["git", "check-ignore", "-q", rel], cwd=ROOT)
    if out.returncode not in (0, 1):
        pytest.skip("not a git checkout")
    return out.returncode == 0


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_case_study_02_ignores_heavy_files_but_not_runs():
    base = "case_studies/02_alpamayo_av_reasoning/"
    for rel in ("weights/model.safetensors", "clips/abc.mp4", "data/x.bin",
                ".venv/bin/python", "runs/scratch/try.json", "hf_cache/blob"):
        assert _ignored(base + rel), rel
    for rel in ("runs/env_bench-5090.json", "runs/latency_bench-5090.json",
                "report/index.html", "FRICTION_LOG.md"):
        assert not _ignored(base + rel), rel


def test_env_records_name_a_board_and_carry_versions():
    records = sorted((CASE / "runs").glob("env_*.json"))
    assert records, "at least one environment record is committed"
    hostname = socket.gethostname()
    for path in records:
        text = path.read_text()
        rec = json.loads(text)
        assert rec["board"] in BOARDS, path.name
        assert path.name == f"env_{rec['board']}.json"
        pk = rec["packages"]
        for name in ("torch", "transformers", "alpamayo_r1"):
            assert pk[name].get("installed"), f"{name} in {path.name}"
            assert pk[name].get("version"), f"{name} version in {path.name}"
        assert pk["alpamayo_r1"].get("commit"), "alpamayo_r1 pinned to a commit"
        assert "/home/" not in text, "home directory path leaked"
        assert hostname not in text or len(hostname) < 4, "hostname leaked"
