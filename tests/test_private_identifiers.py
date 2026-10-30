"""Nothing in the tree may identify a machine: no host name, LAN address or
home-directory path (AGENTS.md rule 9). Reports carry a label instead."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from bench2field.background import scrub_private
from bench2field.schema import HOST_LABEL_DEFAULT, HOST_LABEL_ENV, describe_platform

ROOT = Path(__file__).resolve().parents[1]
H = "/ho" + "me"  # split so this file holds no literal home path
HOME_PATH = re.compile(r"/home/[A-Za-z0-9_.-]+")
PRIVATE_ADDRESS = re.compile(r"(?<![0-9.])(192\.168|172\.(1[6-9]|2[0-9]|3[01]))\.[0-9]{1,3}\.[0-9]{1,3}(?![0-9.])")


def _tracked_text_files() -> list[Path]:
    if shutil.which("git") is None:
        pytest.skip("git not installed")
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True)
    if out.returncode != 0:
        pytest.skip("not a git checkout")
    files = []
    for rel in out.stdout.decode().split("\0"):
        if not rel:
            continue
        path = ROOT / rel
        if not path.is_file():
            continue
        head = path.read_bytes()[:4096]
        if b"\0" in head:
            continue
        files.append(path)
    return files


def test_no_home_paths_or_private_addresses_in_tracked_files():
    offenders = []
    for path in _tracked_text_files():
        if path.name == "test_private_identifiers.py":
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for pat in (HOME_PATH, PRIVATE_ADDRESS):
            m = pat.search(text)
            if m:
                line = text.count("\n", 0, m.start()) + 1
                offenders.append(f"{path.relative_to(ROOT)}:{line}")
                break
    assert not offenders, "private identifiers in tracked files:\n" + "\n".join(offenders)


def test_platform_host_is_a_label_not_the_host_name(monkeypatch):
    import socket

    monkeypatch.delenv(HOST_LABEL_ENV, raising=False)
    host = describe_platform()["host"]
    assert host == HOST_LABEL_DEFAULT
    assert host != socket.gethostname()
    monkeypatch.setenv(HOST_LABEL_ENV, "bench-thor")
    assert describe_platform()["host"] == "bench-thor"


def test_scrub_private_replaces_home_directories():
    assert scrub_private(f"python3 {H}/someone/tools/oled.py --x") == "python3 <home>/tools/oled.py --x"
    assert scrub_private("/usr/bin/python3 -m b2f run") == "/usr/bin/python3 -m b2f run"
    assert scrub_private(f"{H}/a.b-c_d/x {H}/e/y") == "<home>/x <home>/y"
