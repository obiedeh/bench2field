# Contributing

Bench2Field reports numbers people will make deployment decisions on, so the rules are short and strict.

## Ground rules

1. **The report schema is the contract.** Fields in `src/bench2field/schema.py` may be added, never renamed or repurposed. Retention, attribution, validity and verdicts read only the JSON reports. Adding a field bumps the minor schema version, and older reports must still load.
2. **No fabricated numbers.** Nothing goes into the README, docs or a commit message as a result unless it came from a committed run JSON on named hardware. Until then the placeholder says "pending".
3. **One concern per commit**, with a message that says what changed and why.
4. **No new dependencies without a stated reason.** Optional ones go in an extra in `pyproject.toml`.
5. **A hardware bug gets a hardware fixture.** When real output from tegrastats, NVML or rocm-smi breaks a parser or an assumption, commit the captured output under `tests/fixtures/` with a test that reads it.
6. **Follow `docs/METHODOLOGY.md`.** A run that breaks one of its rules is not reported.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The suite runs on a CPU-only machine. Tests that need onnxruntime, a GPU, NVML or PyTorch skip when those are missing; they must keep doing so. Python 3.10 is supported, because JetPack 6 ships it.

## Kernels

Code under `kernels/` is written by the project owner. Build scaffolding, reference implementations and correctness tests are welcome; the kernels themselves are not open for contribution.
