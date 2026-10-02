"""pyproject.toml promises Python 3.10 (JetPack 6 ships it), so the sources
must not use anything newer."""

import ast
from datetime import datetime, timezone
from pathlib import Path

import pytest

from bench2field.loadreplay.record import profile_from_samples
from bench2field.schema import RunReport, Variant

SOURCES = sorted(Path(__file__).resolve().parents[1].joinpath("src").rglob("*.py"))


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_sources_are_python_310(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), feature_version=(3, 10))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "datetime":
            assert "UTC" not in {a.name for a in node.names}, "datetime.UTC needs Python 3.11"
        if isinstance(node, ast.Attribute) and node.attr == "UTC":
            pytest.fail(f"{path.name}:{node.lineno} uses .UTC, which needs Python 3.11")


def test_timestamps_are_utc_aware():
    rep = RunReport(Variant("m", "b", "p", "fp32"), "bench-idle", [])
    prof = profile_from_samples("p", [], 1.0)
    for stamp in (rep.created_at, prof.recorded_at):
        assert datetime.fromisoformat(stamp).utcoffset() == timezone.utc.utcoffset(None)
