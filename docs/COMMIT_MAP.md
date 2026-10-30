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

## Second rewrite, 2026-10-30

The workstation host name, one LAN address and home-directory paths had survived in older revisions (bring-up records, the former HANDOFF.md, one export manifest, one run file and a failure log). Every revision was rewritten to replace them with `rtx-5090-workstation`, `<lan-address>` and `<home>`; file contents are otherwise unchanged. The table maps each commit ID as it was before this rewrite (the "current" column above, for commits listed there) to its ID now. Only the commits whose ID changed are listed; later commits inherit new IDs because their parents changed, not because their content did.

| before 2026-10-30 | now | date (UTC) | subject |
|---|---|---|---|
| `00cf6d5039bc` | `63a95db082ed` | 2026-10-02 06:42 | Add a tools extra with onnx |
| `01c2c95ee8dc` | `5d922ce00059` | 2026-10-30 10:25 | Merge alpamayo-cuda-preprocess: case study 01 phase 2 precision ladder, case study 04 Isaac ROS quickstart, case study 05 plan |
| `026cab81141a` | `93b1ff0bd84c` | 2026-10-02 19:25 | Phase 1 baseline sweep config; HANDOFF: Orin done, export done, scrub note |
| `02c3ded27ae4` | `8680b90aa8b6` | 2026-10-02 06:19 | Support Python 3.10: use timezone.utc instead of datetime.UTC |
| `02cc55262f73` | `1f263079207e` | 2026-10-19 23:36 | Import retained semantic labels and verified detector outputs |
| `02fe63eda813` | `0a3c1aaf6bda` | 2026-10-30 03:14 | Case study 01, phase 2: fp16 and int8 TensorRT variants with accuracy on the COCO subset (Thor) |
| `044bf15a8f7e` | `41c973d82a57` | 2026-10-02 06:46 | NVML sampler: leave out channels a device does not support |
| `0f34616a3fd6` | `e3f3aeaf66ae` | 2026-10-06 01:32 | Add issue templates and a manual trigger for the tests workflow |
| `101627c22823` | `fd1dc78214e1` | 2026-10-02 06:34 | Add RTX 5090 bring-up evidence |
| `10a85f08b31b` | `e679f3f55675` | 2026-10-02 06:56 | CONTRIBUTING: hardware captures live under bringup/<machine>/ |
| `123abe3b0bfa` | `73b66a727596` | 2026-10-02 08:02 | 5090 bring-up: b2f sweep run of the tiny model |
| `16c462bc5256` | `2be54155c302` | 2026-10-02 06:30 | Replay: run the GPU stressor in its own process with its own CUDA context |
| `17024e56ff26` | `ade0349620aa` | 2026-10-30 03:15 | Case study 05: plan for the Cosmos Predict 2.5 forecast cost study (protocol only, nothing measured) |
| `1a82aac62dc6` | `0fff1c7adec2` | 2026-10-03 22:09 | Report theme: keep power-card tables readable |
| `1d272f6caaf8` | `1080b62d43b0` | 2026-10-02 19:40 | HANDOFF: log the Orin replay drift for phase 5 |
| `1f1dcd499e8b` | `b9f01dcead78` | 2026-10-02 19:11 | Record TensorRT, cuDNN and CUDA runtime versions in every report |
| `1f7fb99524a2` | `8eb2ca835566` | 2026-10-29 19:25 | Merge branch 'alpamayo-5090-run' into alpamayo-thor-run |
| `21f8f2b00287` | `4593852e4231` | 2026-10-02 08:02 | HANDOFF: private GitHub repo created, b2f sweep built, detector options proposed |
| `236aefc3122b` | `ce4db71863b0` | 2026-10-02 19:39 | b2f sweep --stopped: record what was shut down for the sweep |
| `2900250959f9` | `b0dba7bc7e12` | 2026-10-30 03:13 | Move the handoff note out of the tree; the README points at the changelog |
| `2a326266f923` | `e2e4f4060b84` | 2026-10-02 06:27 | Add tools/make_tiny_model.py for hardware bring-up runs |
| `2f685d9af735` | `a6087f54bc0a` | 2026-10-02 06:19 | Anchor runs/ in .gitignore to the repo root |
| `32015f64249a` | `4cb11634ddbf` | 2026-10-02 19:11 | Retention: onnxruntime version gets the same treatment as power mode |
| `34ef30949e31` | `41a28668b5cb` | 2026-10-29 15:47 | Case study 02: Alpamayo 1 scaffold, setup notes, friction log and 5090 environment record |
| `359a9ed092f9` | `4beebd63addd` | 2026-10-29 20:23 | Merge branch 'alpamayo-thor-run' |
| `365f45290ed0` | `f40de9d81092` | 2026-10-02 19:44 | index on hardware-bringup: 7c322a0 Case study 01: frame capture tools, end-to-end pipeline profiler, phase 1 summariser |
| `38a3d3b17670` | `003749d509c4` | 2026-10-02 06:42 | Replay calibration records its utilisation source; document what NVML measures |
| `3dedb45c15aa` | `d8db0f2605cc` | 2026-10-02 06:22 | Telemetry: warn when a sampler is missing, close samplers after a run |
| `3e39b764a50f` | `7d4763c65cc1` | 2026-10-02 06:56 | 5090 bring-up: TensorRT fp16 run of the tiny model |
| `3ff6d67afb54` | `37c47930e584` | 2026-10-29 19:32 | Case study 02: 5090 profiled, stage split, kernel table and the report plots |
| `42aafd138d1a` | `befbbdf02efd` | 2026-10-29 16:01 | Case study 02: PLAN.md with the protocol fixed before the first run |
| `492f7834bda1` | `1f925c6f2365` | 2026-10-02 19:39 | Retention: TensorRT and cuDNN versions get the same treatment as onnxruntime |
| `4f48e48d5b3e` | `8ddea5b7b9d4` | 2026-10-29 22:51 | Case study 03: DeepStream 7.1 on the rover's Orin NX with YOLOv8n in TensorRT fp16 |
| `52d306296f4d` | `7c6bce81554c` | 2026-10-02 19:21 | Record what else the machine was doing in every report |
| `565a11f65d8d` | `80c304553390` | 2026-10-02 06:33 | Add CONTRIBUTING.md with the project ground rules |
| `56f87c1edcc6` | `2d36b0725056` | 2026-10-03 18:10 | Publication: commit map for the rewritten history; HANDOFF updated |
| `595e50cec124` | `d38b1e74dcd6` | 2026-10-02 06:34 | CI: run pytest on CPU for Python 3.10 and 3.12 |
| `5d8aadc53f05` | `4177a24b457b` | 2026-10-29 16:20 | Case study 02: SETUP.md copy step writes SOURCE_COMMIT for the Thor's records |
| `5f0033d07408` | `800172907c4c` | 2026-10-19 23:02 | Add provenance-bound saved detector accuracy evaluation |
| `6781a5837dc7` | `810948b84284` | 2026-10-02 19:19 | Add Jetson Orin NX bring-up evidence |
| `67eb6df1b1a4` | `63584d5dff51` | 2026-10-02 06:54 | tegrastats: report Thor board power as power_board_w; test against real output |
| `683c8314f860` | `3a3f968736b8` | 2026-10-02 19:25 | Case study 01: YOLOX chosen; export YOLOX-s to ONNX with recorded provenance |
| `6afd63e5cd94` | `5a1c19c7a705` | 2026-10-29 19:35 | Case study 02: Thor measured, same clips and protocol, board power, compared with the 5090 |
| `6daec49514be` | `8c83e5948c98` | 2026-10-29 19:33 | Merge branch 'alpamayo-5090-profile' into alpamayo-thor-run |
| `6f6cbbcf7bc3` | `c8b0494853d2` | 2026-10-03 22:11 | Case study 01 report: regenerated with the EmbodiedEdge Labs evidence theme |
| `7061861f9512` | `267cbfb82e47` | 2026-10-29 16:02 | Case study 02: Orin NX note, why the field computer is not a target |
| `7257e54a41c7` | `ca7447dc98ab` | 2026-10-02 06:56 | Rover budget: gate power on power_board_w instead of the VIN_SYS_5V0 sub-rail |
| `75c673ea2d13` | `000000000000` | 2026-10-30 03:12 | Reports: replace the workstation host name in the retained-simulation failure log with its board label |
| `7c322a06603c` | `5facc7646af4` | 2026-10-02 19:44 | Case study 01: frame capture tools, end-to-end pipeline profiler, phase 1 summariser |
| `826ecdbad1d8` | `57e4adeb9c07` | 2026-10-29 16:14 | Case study 02: profiler and plot scripts |
| `82fa4bdcb9d4` | `a5052270ab1d` | 2026-10-29 20:23 | Merge branch 'changelog' |
| `8440572a00c5` | `723bf6480b7f` | 2026-10-02 06:28 | ONNX Runtime: refuse to run when the requested provider fell back to CPU |
| `8c7b2e44db7d` | `8a932facb8f6` | 2026-10-29 16:16 | Case study 02: stage plot legend above the bars, PNG written next to each SVG |
| `8dfd7ff5a75d` | `6b3a8e2a5514` | 2026-10-29 19:24 | Case study 02: 5090 measured, sdpa and flash-attn, eight clips, three repeats |
| `929276ea4025` | `53934051df90` | 2026-10-29 20:37 | Merge branch 'feature/accuracy-evidence-20261020' |
| `96b29e4989a2` | `1cab2f206562` | 2026-10-02 19:44 | WIP on hardware-bringup: 7c322a0 Case study 01: frame capture tools, end-to-end pipeline profiler, phase 1 summariser |
| `970c1f14ce06` | `e03b27fac734` | 2026-10-19 23:37 | Record retained simulation person accuracy failure |
| `972985f4ce2a` | `b36dfb7acb15` | 2026-10-29 19:24 | Merge branch 'alpamayo-5090-run' into alpamayo-5090-profile |
| `98a718d2560c` | `d0f8f117fefe` | 2026-10-02 07:59 | Retention over repeats: medians, spread, and comparability checks |
| `99ed63fa3e42` | `066ed0ec04f7` | 2026-10-29 16:01 | Add CHANGELOG.md: releases 0.1, 1.0 and 1.0.1, and the unreleased changes on master |
| `9b60aaf0581a` | `beca5818a5fc` | 2026-10-02 06:20 | Report response time per tier and add a drop-late policy (schema 1.1) |
| `a7f14dd01f53` | `36b2b90d1e3a` | 2026-10-02 06:34 | Add HANDOFF.md: bring-up status, hardware fixes, blockers, deferred work |
| `a94331a7f688` | `639374ffcef6` | 2026-10-29 22:26 | Case study 01: labelled accuracy of the fp32 baseline on a COCO val2017 subset, and the recall-0 diagnosis |
| `ab23bf5012c8` | `af9077c27d3b` | 2026-10-29 22:47 | Case study 02: Thor profile, per-clip trajectory error table, AV stack diagram, Thor tutorial, 1.1 release notes |
| `ab275dea25bc` | `b0a64ea30365` | 2026-10-30 04:23 | Case study 01, phase 2: precision ladder sweeps on the Thor and the Orin NX with the Orin accuracy |
| `ac9e18bd9a48` | `67c8984ea2ca` | 2026-10-29 16:13 | Case study 02: decision runner, fixed clip list and the phase 2 friction log |
| `aee6d8271595` | `e68985cb9178` | 2026-10-02 06:42 | Document the env -u PYTHONPATH requirement; log the ignored /ultrareview tip |
| `b1756e5d6257` | `83fcfc57fe28` | 2026-10-30 09:56 | Case study 04: Isaac ROS 4.5 quickstart measured on the Jetson AGX Thor |
| `b8873aa68f13` | `eeb31245177b` | 2026-10-02 06:40 | ONNX Runtime: load pip-installed TensorRT for the TensorRT provider |
| `bd15ce83d746` | `4f65fa8d4c07` | 2026-10-03 21:42 | Add AGENTS.md: working rules for contributors |
| `bd1d36b49725` | `4595ff062bf4` | 2026-10-30 03:11 | Case study 04: Isaac ROS 4.5 object detection in a container, measured on the quickstart data |
| `c03d6cf3cc7b` | `d63c1172bcd6` | 2026-10-29 20:53 | CHANGELOG: case study 02 and the archived phase2 branch under Unreleased |
| `c2d415325290` | `8695d3d22c6e` | 2026-10-04 17:44 | Rebuild the case-study report |
| `c6a17bcc3d0f` | `5ad5ad3fc952` | 2026-10-29 20:40 | Add .mailmap so every commit by the owner shows the GitHub username obiedeh |
| `c6dfcb55baa7` | `4a41c095c251` | 2026-10-02 06:56 | 5090 bring-up: record exact ORT, TensorRT, cuDNN and torch versions |
| `c7c5d1cea60b` | `ee5059b1f806` | 2026-10-02 06:22 | Replay: calibrate stressor duty with the model idle, then freeze it |
| `c8ac8463ba43` | `0478372ae3ec` | 2026-10-03 22:05 | Tests: self-contained means nothing fetched from the network; navigation links allowed |
| `d0bcbd1296ae` | `8ac3488f7ee1` | 2026-10-29 16:16 | Case study 02: Thor environment, setup script and record |
| `d0da97474773` | `1f293ce34295` | 2026-10-02 06:40 | CUDA Conv test: check output values, not just shape |
| `d7073fb3ac1a` | `e6948d88758e` | 2026-10-29 21:18 | Case study 02, phase 3b: a fused CUDA preprocessing kernel, verified and timed on both boards |
| `ded1a67131ff` | `f4a053493c06` | 2026-10-29 22:47 | Case study 02, phase 3b second pass: separable kernel, C++ host side, standalone driver, parity gate |
| `e3ee9180a3de` | `600f0e0a695c` | 2026-10-02 06:57 | HANDOFF: Thor bring-up done, TensorRT on the 5090 done, Orin unreachable |
| `e468e642924f` | `361ce99a6bd5` | 2026-10-02 06:28 | Runner: warm up at the tier's rate instead of flat out |
| `e46d27f408ad` | `0a23b4598353` | 2026-10-02 06:27 | ONNX Runtime: preload CUDA and cuDNN before creating an NVIDIA session |
| `e589f81f3305` | `058f5793bfe2` | 2026-10-03 22:04 | Report: EmbodiedEdge Labs evidence theme (dark default, light and print variants) |
| `e63fcca5938b` | `ad397f52c81f` | 2026-10-02 19:19 | tegrastats and NVML: Orin NX board power, GR3D load, NVML that answers nothing |
| `e655e85a6211` | `4fb03466d891` | 2026-10-02 06:33 | Add the Apache-2.0 licence text |
| `e92b781ec271` | `569cdbb882a2` | 2026-10-29 16:19 | Case study 02: telemetry sampler chosen by board, source commit for copied trees |
| `ea06f4d8c290` | `e468f76071cd` | 2026-10-03 18:33 | README: link the report through GitHub Pages so it opens as a page |
| `ea6b67dfaad6` | `579fea1e4f86` | 2026-10-04 17:44 | Report: shared lab type scale and accented case-study title |
| `ea89946d0944` | `01004540ec6b` | 2026-10-02 06:54 | Add Jetson AGX Thor bring-up evidence |
| `eaeba5737a42` | `afed1f32d7cb` | 2026-10-29 20:27 | Merge branch 'alpamayo-orin-note' |
| `ee00cd8eb59a` | `2e34686c8e86` | 2026-10-02 19:21 | Synthetic inputs follow each input's dtype |
| `f81afb52bc3e` | `d94dc5d43da3` | 2026-10-02 08:02 | Add b2f sweep: alternating repeats of several variants |
| `fee78275e8dc` | `8ad3c56fdd4d` | 2026-10-02 06:07 | Bench2Field v0.1: runner, report contract, field retention, gap attribution, load replay, verdicts |

## Third rewrite, 2026-10-30

The Orin NX host name and one truncated home path were replaced in every revision by `field-orin` and `<home>`. The table maps each commit ID as it was before this rewrite to its ID now, for commits whose ID changed; follow it after the second rewrite's table.

| before | now | subject |
|---|---|---|
| `012d2419234f` | `43d2a5a7ddf8` | HANDOFF: all phase 1 baselines in, Thor stand-down procedure, Orin thermal note |
| `0478372ae3ec` | `d7b513a70690` | Tests: self-contained means nothing fetched from the network; navigation links allowed |
| `058f5793bfe2` | `f239c08b2ba8` | Report: EmbodiedEdge Labs evidence theme (dark default, light and print variants) |
| `066ed0ec04f7` | `1c675a2d06ff` | Add CHANGELOG.md: releases 0.1, 1.0 and 1.0.1, and the unreleased changes on master |
| `0790b9070ef6` | `a3c44a00a15e` | Reports carry a machine label; a test keeps private identifiers out of the tree |
| `0a3c1aaf6bda` | `3900fbbaddfe` | Case study 01, phase 2: fp16 and int8 TensorRT variants with accuracy on the COCO subset (Thor) |
| `0a9c83e75b46` | `6f960add064f` | Case study 01 phase 2 step 1: FP16 against FP32 on the Orin, sweep and back-to-back profiles |
| `0bcf1d43b85d` | `ed684af56508` | v1.0.1: correct the Orin thermal claim (#3) |
| `0fff1c7adec2` | `017c2b7d2c4b` | Report theme: keep power-card tables readable |
| `13a042e92974` | `8ae3e62d33da` | Merge branch 'master' into phase2 |
| `1ad6ab52d6eb` | `8885fac6a22f` | Correct the Orin thermal claim: a one-off fast first repeat, cause not established |
| `1f263079207e` | `066fa889414d` | Import retained semantic labels and verified detector outputs |
| `267cbfb82e47` | `8a028af7e2fd` | Case study 02: Orin NX note, why the field computer is not a target |
| `27c0f7729a3b` | `33a3bfc45a41` | Record the provider's options in every report |
| `280678841049` | `8e3131dd9a7b` | Version 1.0.1; report regenerated |
| `2d36b0725056` | `21e14b9bd19d` | Publication: commit map for the rewritten history; HANDOFF updated |
| `36495461b6fc` | `a0d51b5f8d9c` | Add b2f report: one self-contained HTML page from a case study's committed runs |
| `37c411a007c1` | `742c56d2bdb5` | Report: the repeats panel no longer implies a thermal cause |
| `37c47930e584` | `a37ec9f8d41d` | Case study 02: 5090 profiled, stage split, kernel table and the report plots |
| `4177a24b457b` | `9031f5eb1ca3` | Case study 02: SETUP.md copy step writes SOURCE_COMMIT for the Thor's records |
| `41a28668b5cb` | `4c45228313ed` | Case study 02: Alpamayo 1 scaffold, setup notes, friction log and 5090 environment record |
| `4595ff062bf4` | `b9dfbd6e1ece` | Case study 04: Isaac ROS 4.5 object detection in a container, measured on the quickstart data |
| `4beebd63addd` | `90cb9956d8a5` | Merge branch 'alpamayo-thor-run' |
| `4f65fa8d4c07` | `436220290225` | Add AGENTS.md: working rules for contributors |
| `534d775d44c5` | `731e0b435835` | profile_pipeline: --expect-commit, the same checkout guard as b2f sweep |
| `53934051df90` | `09d17d7c8c7e` | Merge branch 'feature/accuracy-evidence-20261020' |
| `569cdbb882a2` | `2bffc4959edd` | Case study 02: telemetry sampler chosen by board, source commit for copied trees |
| `575c406bc8aa` | `eadd561a7b91` | Orin end-to-end pipeline profile on the rover's 480p frames; provenance notes |
| `577795ee91ef` | `460542b9f9cd` | HANDOFF: v1.0 release state, what history still contains |
| `579fea1e4f86` | `bb397c2a3062` | Report: shared lab type scale and accented case-study title |
| `57e4adeb9c07` | `f5a366abda80` | Case study 02: profiler and plot scripts |
| `58b283e52b96` | `df866c25c144` | Scrub hostnames, the LAN address and home paths from tracked files |
| `5a1c19c7a705` | `9a88cf789bda` | Case study 02: Thor measured, same clips and protocol, board power, compared with the 5090 |
| `5ad5ad3fc952` | `b200018c641d` | Add .mailmap so every commit by the owner shows the GitHub username obiedeh |
| `5d922ce00059` | `3604d059175f` | Merge alpamayo-cuda-preprocess: case study 01 phase 2 precision ladder, case study 04 Isaac ROS quickstart, case study 05 plan |
| `5e5df1a89fcd` | `b637b15d83c2` | Case study 01 phase 1: Orin NX idle baseline, four tiers, --no-spin |
| `639374ffcef6` | `d65f962ac6db` | Case study 01: labelled accuracy of the fp32 baseline on a COCO val2017 subset, and the recall-0 diagnosis |
| `647daf303b77` | `c6737a758972` | Case study 01 phase 2: script for the temporary CPU governor test on the Orin |
| `67c8984ea2ca` | `acdd054f75a1` | Case study 02: decision runner, fixed clip list and the phase 2 friction log |
| `6ab0fe40267c` | `65e1e3b34202` | Remove AI tool names from tracked files |
| `6b3a8e2a5514` | `f106a94af4e4` | Case study 02: 5090 measured, sdpa and flash-attn, eight clips, three repeats |
| `6b8c68161184` | `d00707c9be80` | Version 1.0.0 |
| `6e920accee12` | `28f7f1803e88` | PHASE1_FINDINGS: the profiled rover frames show a blank wall; a busy scene makes the Orin frame slower |
| `7149328675cb` | `dc3c99d9557f` | Merge hygiene/private-identifiers-20261030: machine labels in reports, private identifier test |
| `71ac462970df` | `bd2c8ce33084` | Pipeline profiler records CPU governor and per-core frequency |
| `7457acb4fe3d` | `8a11a017378c` | README: the finding first, the headline chart, what b2f measures, how to run one baseline, status |
| `79e754a33302` | `b32a12dbc727` | HANDOFF: phase 2 state, governor test deferred, next session order |
| `7cbc11a2cc6c` | `bba201d7f5a6` | profile_pipeline: carry YOLOX's NumPy NMS so the profile runs on boards without PyTorch |
| `7f8183c1319b` | `d67cefe80dc9` | Hardware bring-up and case study 01 phase 1 (#1) |
| `800172907c4c` | `acdfa3209b60` | Add provenance-bound saved detector accuracy evaluation |
| `83fcfc57fe28` | `7ad7943f7a8b` | Case study 04: Isaac ROS 4.5 quickstart measured on the Jetson AGX Thor |
| `8695d3d22c6e` | `f33ad8bde180` | Rebuild the case-study report |
| `8a932facb8f6` | `a6e55b258c8d` | Case study 02: stage plot legend above the bars, PNG written next to each SVG |
| `8ac3488f7ee1` | `5f96c8d5e95f` | Case study 02: Thor environment, setup script and record |
| `8bdd0ab461b2` | `a6112be82835` | PHASE1_FINDINGS: no-spin baselines on all three machines, Orin section, 26 Hz headlines |
| `8c83e5948c98` | `a65a1e759eab` | Merge branch 'alpamayo-5090-profile' into alpamayo-thor-run |
| `8d79c9fbd135` | `d8e0de49681d` | Case study 01 phase 2: sweep config for FP16 against FP32 on the Orin (speed only) |
| `8ddea5b7b9d4` | `da5aae7a8a9e` | Case study 03: DeepStream 7.1 on the rover's Orin NX with YOLOv8n in TensorRT fp16 |
| `8e4835b0801a` | `0e7a7d3f54f5` | Case study 01: generated report and headline chart |
| `8eb2ca835566` | `9156ec2e379b` | Merge branch 'alpamayo-5090-run' into alpamayo-thor-run |
| `9aa28a108daa` | `43eb9fd6f0a7` | Case study 01 phase 1: Thor end-to-end pipeline profile on the rover's 480p frames |
| `9c4281e89ba3` | `700ee50de902` | v1.0: report, README, scrub, clean-clone test (#2) |
| `a5052270ab1d` | `5cdf441033b1` | Merge branch 'changelog' |
| `ade0349620aa` | `8baf0caa1185` | Case study 05: plan for the Cosmos Predict 2.5 forecast cost study (protocol only, nothing measured) |
| `aecd341c4537` | `a81ac839cba9` | Case study 01: regenerate the report from the scrubbed runs |
| `af9077c27d3b` | `7400b38369a7` | Case study 02: Thor profile, per-clip trajectory error table, AV stack diagram, Thor tutorial, 1.1 release notes |
| `afed1f32d7cb` | `1f2043634b38` | Merge branch 'alpamayo-orin-note' |
| `b0a64ea30365` | `d17dfe3bfd44` | Case study 01, phase 2: precision ladder sweeps on the Thor and the Orin NX with the Orin accuracy |
| `b0dba7bc7e12` | `899b371aef55` | Move the handoff note out of the tree; the README points at the changelog |
| `b36dfb7acb15` | `2407cc6c50bf` | Merge branch 'alpamayo-5090-run' into alpamayo-5090-profile |
| `b80c168fec56` | `6c52388fae2c` | Case study 01 phase 2: six alternated FP32/FP16 pipeline profiles on the Orin with CPU frequency |
| `bb83262ec043` | `15b4bce3f1b3` | README: pin the ONNX Runtime and TensorRT pair, install YOLOX in two steps |
| `befbbdf02efd` | `4798df4959b0` | Case study 02: PLAN.md with the protocol fixed before the first run |
| `c027a3e7911c` | `98900d36f766` | PHASE1_FINDINGS: correct the Orin drop-rate and inference-rate sentences |
| `c0f807532af0` | `bc048b36351c` | PHASE1_FINDINGS: the Orin headline is the model only; full frame on the Orin is 47.8 ms and misses |
| `c4dcf4721841` | `18536177b1bd` | COMMIT_MAP: second rewrite (2026-10-30), host name, LAN address and home paths removed from every revision |
| `c8b0494853d2` | `37df5806cdb0` | Case study 01 report: regenerated with the EmbodiedEdge Labs evidence theme |
| `ca389888a375` | `a009170c7b3c` | Case study 01 phase 1: Thor FP32 baseline, four tiers, --no-spin, clean |
| `d63c1172bcd6` | `6e4756d63b66` | CHANGELOG: case study 02 and the archived phase2 branch under Unreleased |
| `e03b27fac734` | `c224df6abaa7` | Record retained simulation person accuracy failure |
| `e38a97c53840` | `9e3400184c46` | Clean-clone test transcript: two passes on the 5090 host, second one clean |
| `e3f3aeaf66ae` | `df5f770704da` | Add issue templates and a manual trigger for the tests workflow |
| `e468f76071cd` | `f4071a2caf20` | README: link the report through GitHub Pages so it opens as a page |
| `e6948d88758e` | `df68bdc9d026` | Case study 02, phase 3b: a fused CUDA preprocessing kernel, verified and timed on both boards |
| `f4a053493c06` | `ed905ce6a667` | Case study 02, phase 3b second pass: separable kernel, C++ host side, standalone driver, parity gate |
