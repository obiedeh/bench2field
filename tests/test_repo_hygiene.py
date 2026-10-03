"""Case-study run reports are committed evidence; scratch runs at the repo
root are not. The .gitignore has to tell the two apart."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _ignored(rel: str) -> bool:
    out = subprocess.run(["git", "check-ignore", "-q", rel], cwd=ROOT)
    if out.returncode not in (0, 1):
        pytest.skip("not a git checkout")
    return out.returncode == 0


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_only_root_runs_dir_is_ignored():
    assert _ignored("runs/scratch.json")
    assert not _ignored("case_studies/01_perception_detector/runs/bench_fp32.json")
