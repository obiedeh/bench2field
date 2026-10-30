"""The README results table of case study 02 must equal the table printed from
the committed run records.

The README is the human entry point and the JSON under ``runs/`` is the
evidence. The two drift apart when someone edits a number by hand, so this
test regenerates the Markdown table with ``plot_results.results_table`` from
the committed ``decisions_*.json`` files and requires the README to contain it
verbatim. A new run is added by committing its JSON and pasting the printed
table, never by typing numbers.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

CASE = Path(__file__).resolve().parents[1] / "case_studies" / "02_alpamayo_av_reasoning"


def _plot_results():
    spec = importlib.util.spec_from_file_location("plot_results", CASE / "plot_results.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_readme_results_table_matches_committed_runs():
    pr = _plot_results()
    runs = pr.load_runs("decisions")
    if not runs:
        pytest.skip("no committed decision records yet")
    table = pr.results_table(runs)
    readme = (CASE / "README.md").read_text()
    assert table in readme, "README results table differs from plot_results.py --table; regenerate it"


def test_committed_runs_are_dataset_input_only():
    for p in (CASE / "runs").glob("decisions_*.json"):
        text = p.read_text()
        assert '"input": "dataset"' in text, f"{p.name} is not a dataset run; synthetic output belongs in runs/scratch/"
