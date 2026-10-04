# AGENTS.md

Engineering rules for anyone, human or automated, changing Bench2Field.
Bench2Field reports numbers people will make deployment decisions on, so the
rules are short and strict. [CONTRIBUTING.md](CONTRIBUTING.md) and
[docs/METHODOLOGY.md](docs/METHODOLOGY.md) are the full versions; this file is
the working summary.

## Layout

```text
src/bench2field/   Package: run contract, runners, telemetry, retention, verdicts, report
configs/           Run and budget configurations
profiles/          Per-board profiles (RTX 5090, Jetson AGX Thor, Jetson Orin NX)
bringup/           Per-machine bring-up notes and captured hardware output
case_studies/      Case studies with committed runs, findings and generated reports
kernels/           Custom kernels (owner-written; scaffolding and tests welcome)
docs/              Methodology and design notes
tests/             Pytest suite, runs on a CPU-only machine
```

## Rules

1. **The report schema is the contract.** Fields in `src/bench2field/schema.py`
   may be added, never renamed or repurposed. Adding a field bumps the minor
   schema version, and older reports must still load.
2. **No fabricated numbers.** A result appears in the README, docs or a commit
   message only if it came from a committed run JSON on named hardware. Until
   then it says "pending".
3. **Budgets are written before testing.** Deadlines, miss rates, power,
   temperature and accuracy limits are set before a run, never fitted to it.
4. **Run at the robot's real rate.** Latency is measured open-loop at the rate
   the camera actually delivers, against when each frame arrived.
5. **A hardware bug gets a hardware fixture.** Real tegrastats, NVML or
   rocm-smi output that breaks a parser is committed under `bringup/<machine>/`
   or `tests/fixtures/` with a test that reads it.
6. **Corrections stay in the record.** A wrong claim is corrected with a note
   and a release, never silently rewritten.
7. **One concern per commit**, with a message that says what changed and why.
8. **No new dependencies without a stated reason.** Optional ones go in an
   extra in `pyproject.toml`.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

Tests that need onnxruntime, a GPU, NVML or PyTorch skip when those are
missing and must keep doing so. Python 3.10 stays supported because JetPack 6
ships it.
