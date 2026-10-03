# Commit map

This repository's history was rewritten before it was made public, to remove machine hostnames, a LAN address and home-directory paths from every commit, and to drop tool-generated trailer lines from commit messages. No measured number or result changed.

Run reports, sweep manifests and the findings record the commit each measurement was taken from, using the original commit IDs. This table maps each original ID to its ID in this history. Shortened IDs (first 7 or 8 characters) match the start of the original column.

| original | current | date (UTC) | subject |
|---|---|---|---|
| `fee78275e8dc` | `f80642c52e05` | 2026-10-02 11:07 | Bench2Field v0.1: runner, report contract, field retention, gap attribution, load replay, verdicts |
| `02c3ded27ae4` | `9ee454059637` | 2026-10-02 11:19 | Support Python 3.10: use timezone.utc instead of datetime.UTC |
| `2f685d9af735` | `84b6836d91d1` | 2026-10-02 11:19 | Anchor runs/ in .gitignore to the repo root |
| `9b60aaf0581a` | `09d107e35855` | 2026-10-02 11:20 | Report response time per tier and add a drop-late policy (schema 1.1) |
| `c7c5d1cea60b` | `2cf745dc30ae` | 2026-10-02 11:22 | Replay: calibrate stressor duty with the model idle, then freeze it |
| `3dedb45c15aa` | `8e4919b0428e` | 2026-10-02 11:22 | Telemetry: warn when a sampler is missing, close samplers after a run |
| `2a326266f923` | `aaa80d213575` | 2026-10-02 11:27 | Add tools/make_tiny_model.py for hardware bring-up runs |
| `e46d27f408ad` | `0cfbecb99a02` | 2026-10-02 11:27 | ONNX Runtime: preload CUDA and cuDNN before creating an NVIDIA session |
| `e468e642924f` | `03a51f4bd7c2` | 2026-10-02 11:28 | Runner: warm up at the tier's rate instead of flat out |
| `8440572a00c5` | `dacd261f37f1` | 2026-10-02 11:28 | ONNX Runtime: refuse to run when the requested provider fell back to CPU |
| `16c462bc5256` | `50aaec90735a` | 2026-10-02 11:30 | Replay: run the GPU stressor in its own process with its own CUDA context |
| `e655e85a6211` | `a2fa117a8628` | 2026-10-02 11:33 | Add the Apache-2.0 licence text |
| `565a11f65d8d` | `dba1cbc3e9b9` | 2026-10-02 11:33 | Add CONTRIBUTING.md with the project ground rules |
| `595e50cec124` | `c31dd2302ff3` | 2026-10-02 11:34 | CI: run pytest on CPU for Python 3.10 and 3.12 |
| `101627c22823` | `cef64ad09a85` | 2026-10-02 11:34 | Add RTX 5090 bring-up evidence |
| `a7f14dd01f53` | `efc118e8ebc3` | 2026-10-02 11:34 | Add HANDOFF.md: bring-up status, hardware fixes, blockers, deferred work |
| `d0da97474773` | `efe4050f2fca` | 2026-10-02 11:40 | CUDA Conv test: check output values, not just shape |
| `b8873aa68f13` | `8160ccaa9075` | 2026-10-02 11:40 | ONNX Runtime: load pip-installed TensorRT for the TensorRT provider |
| `38a3d3b17670` | `7f2962eca35b` | 2026-10-02 11:42 | Replay calibration records its utilisation source; document what NVML measures |
| `aee6d8271595` | `2373cef40c48` | 2026-10-02 11:42 | Document the env -u PYTHONPATH requirement; log the ignored /ultrareview tip |
| `00cf6d5039bc` | `ba69dcee802d` | 2026-10-02 11:42 | Add a tools extra with onnx |
| `044bf15a8f7e` | `7eb43416397e` | 2026-10-02 11:46 | NVML sampler: leave out channels a device does not support |
| `ea89946d0944` | `1048dc1a379f` | 2026-10-02 11:54 | Add Jetson AGX Thor bring-up evidence |
| `67eb6df1b1a4` | `1db4e93645cc` | 2026-10-02 11:54 | tegrastats: report Thor board power as power_board_w; test against real output |
| `c6dfcb55baa7` | `93edb586f7db` | 2026-10-02 11:56 | 5090 bring-up: record exact ORT, TensorRT, cuDNN and torch versions |
| `3e39b764a50f` | `6b107814e3e1` | 2026-10-02 11:56 | 5090 bring-up: TensorRT fp16 run of the tiny model |
| `7257e54a41c7` | `0c873484b073` | 2026-10-02 11:56 | Rover budget: gate power on power_board_w instead of the VIN_SYS_5V0 sub-rail |
| `10a85f08b31b` | `f65e08974261` | 2026-10-02 11:56 | CONTRIBUTING: hardware captures live under bringup/<machine>/ |
| `e3ee9180a3de` | `4dcfd2341aaa` | 2026-10-02 11:57 | HANDOFF: Thor bring-up done, TensorRT on the 5090 done, Orin unreachable |
| `98a718d2560c` | `40c546f72d4f` | 2026-10-02 12:59 | Retention over repeats: medians, spread, and comparability checks |
| `123abe3b0bfa` | `b630d30f9f69` | 2026-10-02 13:02 | 5090 bring-up: b2f sweep run of the tiny model |
| `f81afb52bc3e` | `c2f9404795fc` | 2026-10-02 13:02 | Add b2f sweep: alternating repeats of several variants |
| `21f8f2b00287` | `2b62ba55cf9c` | 2026-10-02 13:02 | HANDOFF: private GitHub repo created, b2f sweep built, detector options proposed |
| `32015f64249a` | `596e9e574e48` | 2026-10-03 00:11 | Retention: onnxruntime version gets the same treatment as power mode |
| `1f1dcd499e8b` | `3bbbfb1d156e` | 2026-10-03 00:11 | Record TensorRT, cuDNN and CUDA runtime versions in every report |
| `e63fcca5938b` | `96eb7247e02b` | 2026-10-03 00:19 | tegrastats and NVML: Orin NX board power, GR3D load, NVML that answers nothing |
| `6781a5837dc7` | `49a6f90053e7` | 2026-10-03 00:19 | Add Jetson Orin NX bring-up evidence |
| `52d306296f4d` | `78c3819ca8ac` | 2026-10-03 00:21 | Record what else the machine was doing in every report |
| `ee00cd8eb59a` | `57ea4d4b8e1e` | 2026-10-03 00:21 | Synthetic inputs follow each input's dtype |
| `683c8314f860` | `378b1ee1762d` | 2026-10-03 00:25 | Case study 01: YOLOX chosen; export YOLOX-s to ONNX with recorded provenance |
| `026cab81141a` | `aec3e48aca59` | 2026-10-03 00:25 | Phase 1 baseline sweep config; HANDOFF: Orin done, export done, scrub note |
| `236aefc3122b` | `98833b8b10c8` | 2026-10-03 00:39 | b2f sweep --stopped: record what was shut down for the sweep |
| `492f7834bda1` | `39f2d3a89ef3` | 2026-10-03 00:39 | Retention: TensorRT and cuDNN versions get the same treatment as onnxruntime |
| `1d272f6caaf8` | `e050639140a6` | 2026-10-03 00:40 | HANDOFF: log the Orin replay drift for phase 5 |
| `f2e06fd96062` | `3f4e5188db54` | 2026-10-03 00:44 | Case study 01: frame capture tools, end-to-end pipeline profiler, phase 1 summariser |
| `aad717aff2e7` | `e2a70675e496` | 2026-10-03 00:53 | Case study 01 phase 1: FP32 baseline sweep on the RTX 5090 |
| `a98d4c77cb01` | `fd127cc0938d` | 2026-10-03 00:57 | Case study 01 phase 1: end-to-end pipeline profiles and nsys summaries on the RTX 5090 |
| `489e6e511c4c` | `d11687fd4696` | 2026-10-03 01:03 | Case study 01 phase 1: FP32 baseline sweep on the Jetson AGX Thor |
| `400fff9246dc` | `a38e17d6c244` | 2026-10-03 01:05 | Case study 01: PHASE1_FINDINGS.md |
| `fd4e1e756bc5` | `22ae93c2f5df` | 2026-10-03 01:06 | HANDOFF: phase 1 done, container and git-history notes, phase 2 commands |
| `5079d98f54bb` | `3aeaaf4b61cc` | 2026-10-03 01:06 | HANDOFF: PR link |
| `f2a4867bf6b7` | `424c144cc587` | 2026-10-03 01:25 | Rover camera rate measured on the Orin: 26 Hz delivered, 30 configured; add a 26 Hz tier |
| `913bd0395edd` | `fccc091136fc` | 2026-10-03 01:33 | Sweep config: no_spin option; the baseline sweep runs with spinning off |
| `32ca9a90d091` | `7070dbf07b38` | 2026-10-03 01:33 | HANDOFF: Thor baseline waits for the API stand-down; rviz2 note for the phase 5 profile |
| `5f545c6655ad` | `8c27641d6043` | 2026-10-03 01:33 | Tests: the rover-stack tegrastats capture parses and shows the load |
| `cebcb1d9c35f` | `0304d7bdee9e` | 2026-10-03 01:33 | Thor re-run stopped after one run: partial output, manifest marked incomplete |
| `40995dac81d1` | `bbc0b90b86cd` | 2026-10-03 01:53 | Case study 01 phase 1: four-tier FP32 sweep on the RTX 5090 (spinning on) |
| `ac95271b7fae` | `570b11649d0f` | 2026-10-03 01:53 | PHASE1_FINDINGS: camera rate section, 26 Hz headline from the four-tier 5090 run |
| `fee68af52efe` | `9c3585c5295f` | 2026-10-03 02:17 | HANDOFF: log the model fit matrix (b2f evaluate) as a future direction |
| `626ec7935124` | `e8243f88b71e` | 2026-10-03 02:24 | Case study 01 phase 1: four-tier FP32 baseline on the RTX 5090, --no-spin |
| `1a0c491a2199` | `3f2a53d2d513` | 2026-10-03 02:29 | Thor four-tier FP32 sweep, clean of vLLM but with spinning on (my mistake) |
| `832b90559616` | `60ded7b0c9f7` | 2026-10-03 02:36 | summarize_phase1: list every sweep directory, baselines first |
| `0ac4fbfce23a` | `cc6f50215d4b` | 2026-10-03 02:39 | Provenance: record the git commit and dirty flag in every report and manifest |
| `6dad16b90d49` | `5e5df1a89fcd` | 2026-10-03 03:01 | Case study 01 phase 1: Orin NX idle baseline, four tiers, --no-spin |
| `e58a26e8ac2e` | `ca389888a375` | 2026-10-03 03:32 | Case study 01 phase 1: Thor FP32 baseline, four tiers, --no-spin, clean |
| `f50a1c616e01` | `012d2419234f` | 2026-10-03 03:32 | HANDOFF: all phase 1 baselines in, Thor stand-down procedure, Orin thermal note |
| `db623847b525` | `8bdd0ab461b2` | 2026-10-03 03:32 | PHASE1_FINDINGS: no-spin baselines on all three machines, Orin section, 26 Hz headlines |
| `86d0490d55f8` | `7cbc11a2cc6c` | 2026-10-03 16:00 | profile_pipeline: carry YOLOX's NumPy NMS so the profile runs on boards without PyTorch |
| `32a0663b3b2d` | `575c406bc8aa` | 2026-10-03 16:01 | Orin end-to-end pipeline profile on the rover's 480p frames; provenance notes |
| `d9940c9e31b3` | `c0f807532af0` | 2026-10-03 16:02 | PHASE1_FINDINGS: the Orin headline is the model only; full frame on the Orin is 47.8 ms and misses |
| `e237c3114f3d` | `c027a3e7911c` | 2026-10-03 16:03 | PHASE1_FINDINGS: correct the Orin drop-rate and inference-rate sentences |
| `a37915d65618` | `7f8183c1319b` | 2026-10-03 16:09 | Hardware bring-up and case study 01 phase 1 (#1) |
| `0ad7ad7ea193` | `6e920accee12` | 2026-10-03 16:09 | PHASE1_FINDINGS: the profiled rover frames show a blank wall; a busy scene makes the Orin frame slower |
| `dadbabefb929` | `534d775d44c5` | 2026-10-03 16:14 | profile_pipeline: --expect-commit, the same checkout guard as b2f sweep |
| `47a8dffc11b4` | `9aa28a108daa` | 2026-10-03 16:15 | Case study 01 phase 1: Thor end-to-end pipeline profile on the rover's 480p frames |
| `b3f8d5060ad7` | `8e4835b0801a` | 2026-10-03 16:30 | Case study 01: generated report and headline chart |
| `778a19ec1a6f` | `36495461b6fc` | 2026-10-03 16:30 | Add b2f report: one self-contained HTML page from a case study's committed runs |
| `7fd8a6920c86` | `7457acb4fe3d` | 2026-10-03 16:31 | README: the finding first, the headline chart, what b2f measures, how to run one baseline, status |
| `b9e6256e7f95` | `58b283e52b96` | 2026-10-03 16:32 | Scrub hostnames, the LAN address and home paths from tracked files |
| `b3c738a94c4f` | `bb83262ec043` | 2026-10-03 16:39 | README: pin the ONNX Runtime and TensorRT pair, install YOLOX in two steps |
| `49a047f8cc2b` | `e38a97c53840` | 2026-10-03 16:42 | Clean-clone test transcript: two passes on the 5090 host, second one clean |
| `800b72a4d36b` | `aecd341c4537` | 2026-10-03 16:43 | Case study 01: regenerate the report from the scrubbed runs |
| `dd43b3e0f5ae` | `577795ee91ef` | 2026-10-03 16:43 | HANDOFF: v1.0 release state, what history still contains |
| `dc8f517cf5e2` | `6ab0fe40267c` | 2026-10-03 20:25 | Remove AI tool names from tracked files |
| `6b9c0a5e4fab` | `6b8c68161184` | 2026-10-03 20:26 | Version 1.0.0 |
| `2398487977be` | `9c4281e89ba3` | 2026-10-03 20:27 | v1.0: report, README, scrub, clean-clone test (#2) |
| `41fb14836174` | `8d79c9fbd135` | 2026-10-03 20:31 | Case study 01 phase 2: sweep config for FP16 against FP32 on the Orin (speed only) |
| `e0daf4af37f3` | `27c0f7729a3b` | 2026-10-03 20:31 | Record the provider's options in every report |
| `09c46586f9ef` | `0a9c83e75b46` | 2026-10-03 21:07 | Case study 01 phase 2 step 1: FP16 against FP32 on the Orin, sweep and back-to-back profiles |
| `9212084502ca` | `37c411a007c1` | 2026-10-03 21:55 | Report: the repeats panel no longer implies a thermal cause |
| `27e61a7b32c7` | `1ad6ab52d6eb` | 2026-10-03 21:55 | Correct the Orin thermal claim: a one-off fast first repeat, cause not established |
| `2e551fdcd321` | `280678841049` | 2026-10-03 21:55 | Version 1.0.1; report regenerated |
| `072c917ab6d5` | `0bcf1d43b85d` | 2026-10-03 21:56 | v1.0.1: correct the Orin thermal claim (#3) |
| `7b58452445e2` | `13a042e92974` | 2026-10-03 21:57 | Merge branch 'master' into phase2 |
| `2ff1e663ed7f` | `71ac462970df` | 2026-10-03 21:57 | Pipeline profiler records CPU governor and per-core frequency |
| `ac577d543889` | `b80c168fec56` | 2026-10-03 22:01 | Case study 01 phase 2: six alternated FP32/FP16 pipeline profiles on the Orin with CPU frequency |
| `32a4b69a89fc` | `647daf303b77` | 2026-10-03 22:13 | Case study 01 phase 2: script for the temporary CPU governor test on the Orin |
| `7a63387a7247` | `79e754a33302` | 2026-10-03 22:32 | HANDOFF: phase 2 state, governor test deferred, next session order |
