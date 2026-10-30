# Changelog

What changed in each Bench2Field release, newest first. Versions follow
`pyproject.toml` and the git tags. A result named here came from a committed
run file on named hardware, as the project rules require.

## Unreleased

- Report theme: the EmbodiedEdge Labs evidence theme (dark default, light and print variants), with readable power-card tables and the case study 01 report regenerated with it.
- `AGENTS.md`: the working rules for anyone changing the repository.
- GitHub issue templates (bug report, share a result) and a manual trigger for the tests workflow.
- README: the case study 01 report is linked through GitHub Pages so it opens as a page.
- Tests: "self-contained" for the report means nothing is fetched from the network; navigation links are allowed.
- Publication: `docs/COMMIT_MAP.md` maps the commit IDs recorded in run files to the rewritten history.

## 1.0.1 (2026-10-03)

- Corrected the Orin NX thermal claim: the fast first repeat was a one-off and its cause is not established. The repeats panel in the report no longer implies a thermal cause. The correction stays in the record (rule 6).
- Version 1.0.1; report regenerated.

## 1.0 (2026-10-03)

Measure and diagnose: bring-up on three boards and case study 01, phase 1.

- Bring-up with real-hardware fixtures on the RTX 5090, the Jetson AGX Thor and the rover's Jetson Orin NX: NVML and tegrastats samplers, board power rails (`VIN` on the Thor, `VDD_IN` on the Orin NX) reported as `power_board_w`, the ONNX Runtime wheel that works on each board recorded in `bringup/`.
- Run contract (schema 1.1): response-time statistics per tier, `--drop-late` with dropped frames counted apart from deadline misses, TensorRT, cuDNN and CUDA runtime versions recorded from the loaded libraries, and a snapshot of what else the machine was doing.
- Repeats and `b2f sweep`: alternated repeats of several variants, one process and one report per run, with a manifest of the order.
- Field-load replay calibrates stressor duty with the model idle, then freezes it; the GPU stressor runs in its own process.
- Case study 01 (YOLOX-s on the rover): FP32 baselines at the camera's real rates on all three boards, the end-to-end pipeline profile, an Nsight Systems capture and `PHASE1_FINDINGS.md`.
- Report (`b2f report`) built from the committed run files only; clean-clone test; history scrubbed of hostnames and addresses before publication.

## 0.1 (2026-10-02)

- First commit: runner, report contract, field retention, gap attribution, load replay and verdicts.
- Python 3.10 support (JetPack 6 ships it); `runs/` anchored to the repository root so case-study runs can be committed.
