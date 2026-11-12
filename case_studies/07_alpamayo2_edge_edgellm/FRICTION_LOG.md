# Alpamayo 2 Edge, TensorRT Edge-LLM recipe on the Jetson AGX Thor: run notes

Started 2026-11-12. Recipe: NVlabs/alpamayo-recipes `recipes/alpamayo2edge_edgellm` (commit 622f9d2, 2026-11-10),
Edge-LLM submodule v0.11.1 (358de11c). Written for DRIVE AGX Thor (DRIVE OS 7.2.7, TensorRT 11.3.1.9).
Board here: Jetson AGX Thor, JetPack 7.1 (L4T r38.4), TensorRT 10.13.3.9, CUDA toolkits 13.0 and 13.2 installed
(the `cuda` alternative points at 13.2), Python 3.12.3, MAXN. Edge-LLM's support matrix lists Jetson Thor with
JetPack 7.0/7.1 (CUDA 13.0, platform TensorRT) as an official target; its limitations page prefers TensorRT 10.13.3.9
(JetPack 7.1) over 10.15 for NVFP4 accuracy.

## Host (RTX 5090 workstation), as written

- Weights: the copy made on 2026-11-10 (`/mnt/backup/alpamayo2-edge-20261110/model`) has the same sha256 for both
  safetensors files as Hugging Face revision aaaa32c6, the revision the recipe pins; no new download.
- Environments per README step 2 (`uv`, Python 3.12): env/host (Edge-LLM tools, editable), env/quant (+ transformers 5.17.0),
  env/inputs (alpamayo2-edge@02f55a7 with the dataset extra).
- `run_host_steps.sh` runs steps 3 to 7 for the recipe's example clip (ea7bbd31, t0 4000000 us, "Turn right in 30m"):
  inputs 0 s (already present from a first attempt), shim 21 s, FP8 quantization 36 s, vocab 1 s, ONNX export 63 s.
  Output: onnx/fp8 = llm 2.4 GB, visual 502 MB, action 191 MB; quant 2.9 GB. Warnings only (modelopt vs transformers
  5.17.0 untested; AutoConfig does not know `cosmos3_edge` in the host env's transformers 5.14.1, Edge-LLM falls back).
- Drafter steps skipped: the DFlash2 block-16 drafter is not released (README: "not part of the model release").

## Board adaptations (Jetson instead of DRIVE OS)

1. `thor/setup.sh` replaced by `thor_setup_jetson.sh`: no TensorRT SDK tarball; the JetPack TensorRT Python package
   (/usr/lib/python3.12/dist-packages/tensorrt*) is symlinked into the uv venv; TENSORRT_ROOT=/usr; CUDA_HOME and PATH
   pinned to /usr/local/cuda-13.0 so `nvcc --version` reports 13.0 (build.py derives CUDA_CTK_VERSION from it).
2. `thor/build.py`: `-DEMBEDDED_TARGET=auto-thor` -> `jetson-thor` (the toolchain's auto-thor branch expects the DRIVE OS
   layout /usr/local/cuda/thor/targets/aarch64-linux); CUDA_DIR passed explicitly for the 13.0 toolkit.
3. `thor/run.py`: profile platform string "DRIVE AGX Thor" -> "Jetson AGX Thor".

## Issues on the board, in order

1. cmake configure failed: `3rdParty/xgrammar/include/xgrammar/xgrammar.h` missing. Cause: the README's
   `git submodule update --init --recursive` had been run without `--recursive`, so Edge-LLM's own submodules
   (googletest, nlohmann json, NVTX, xgrammar) were absent. Fixed by fetching them on the host and copying
   3rdParty/ to the board; the copied source tree under work/edgellm was removed so build.py recopies it.
2. Edge-LLM compile failed in the first CUDA file (gdnTreeChunkKernels): `macro "__cudaLaunch" requires 2 arguments,
   but only 1 given`, host_runtime.h of CUDA 13.0. Cause: `cmake/aarch64_linux_toolchain.cmake` forces
   `CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc`, and on this board the `cuda` alternative is 13.2 while CUDA_DIR
   pointed at the 13.0 headers; nvcc 13.2's generated launch stubs do not match 13.0's host_runtime.h. Fixed by
   pointing the toolchain at /usr/local/cuda-13.0/bin/nvcc and build.py's cuRAND include at $CUDA_HOME, so every
   CUDA piece is 13.0, the toolkit JetPack 7.1 and the Edge-LLM docs pair with TensorRT 10.13.3.9.

## Build (second attempt, CUDA 13.0 throughout)

Edge-LLM plugin, llm_build and visual_build compiled and the language and vision engines built by 13:09 CT
(cmake configure started about 12:50 CT). Action-expert engine build followed.

## Measurement plan

- `thor_stream.py` (own code, in work/): loads the engines once, N decisions per clip back to back, tegrastats at
  100 ms for board power (VIN) and energy per decision; rolling mode drops the six newest frames from the vision
  cache before each decision (the recipe's --rolling semantics on a loop), cold mode encodes all 22 frames.
- Clips: the recipe's example clip (ea7bbd31, "Turn right in 30m") for the like-for-like with NVIDIA's indicative
  220 ms and Embedl's one-clip, 180-decision method; the eight clips of case study 06 for the like-for-like with the
  PyTorch numbers of 2026-11-10 (same t0 and derived navigation text).

## First results (FP8 engines, greedy, 5 Euler steps, 6 candidates, MAXN, CPU governor schedutil)

- NVIDIA thor/run.py, example clip, rolling (6 newest frames encoded): total 523 ms; ViT 29, prefill 116, decode 291
  (22 tokens, 13.2 ms per token), action K/V 3, action 84. Chain of causation identical to the recipe's example output
  ("Decelerate and turn right at the intersection to follow our route, as the roadway is clear.").
- thor_stream.py, example clip, rolling, 180 decisions back to back: median 516.3 ms, p90 523.1, p95 524.8, max 532.0;
  1.94 decisions/s; board power 75.1 W mean, 139.7 W peak (idle 16.1 W); 38.6 J per decision. Stage medians: ViT 27.2,
  prefill 117.2, decode 285.1, action K/V 2.9, action 84.5.
- Against NVIDIA's indicative DRIVE AGX Thor figures (ViT 20, prefill 75, 7 ms per greedy token, expert 70, about
  220 ms with the drafter): every stage is slower here, decode by 1.9x, prefill by 1.6x, expert by 1.2x.
- Against Embedl's card on the same Jetson board: 91.9 ms median, 140.3 ms p90, 12.45 J. This FP8 greedy build is
  5.6x slower and 3.1x more energy per decision; Embedl's FP16 Edge-LLM row was 489.3 ms median, so the FP8 recipe
  as run here sits where their unquantized engines sat.
- Issue 3: the recipe's action engine is exported for a fixed 6 candidates (`TreeActionExpert` refuses any other
  count), so the one-candidate runs in the chain failed at load; a one-candidate graph needs its own export.
- thor_stream.py, example clip, cold cache (all 22 frames encoded per decision), 60 decisions: median 525.3 ms,
  p95 613.4; ViT 97.8 ms median against 27.2 rolling; 46.4 J per decision. Vision reuse is worth about 70 ms and 8 J.
- thor_stream.py, nine clips (the eight of case study 06 plus the example), rolling, 20 decisions each (180):
  median 413.1 ms, p90 488.2, p95 493.3, max 499.3; 33.4 J; 83.8 W mean, 148.7 W peak. Four of nine clips took the
  adaptive nominal path (6 tokens, about 273 ms); the five reasoning clips wrote 19 to 26 tokens (413 to 493 ms).
  Stage medians: ViT 24.8, prefill 94.9, decode 208.7, action K/V 2.9, action 83.2. The floor with no reasoning is
  therefore about 205 ms (ViT + prefill + expert), already above Embedl's 92 ms median: their gain is not only in
  decode.
- Issue 4: `host/action_graph.py` writes `candidates_per_sequence = 2` into the action config whatever the candidate
  count (TREE_BLOCK 128 // 64 tokens); `thor/build_action.py` then computes 1 // 2 = 0 sequences for a one-candidate
  graph and TensorRT rejects the kv_page_table profile (min 0). The runtime's TreeActionExpert reduces per_seq itself
  and would have been fine. Fixed by setting candidates_per_sequence to 1 in the one-candidate config.
- One candidate (engines/fp8-c1, own export with candidates_per_sequence 1): example clip 180 decisions median
  398.6 ms, p95 457.8, 32.6 J; nine clips median 367.2 ms, p90 415.4, p95 427.2, 29.1 J. Expert 35 ms at one
  candidate against 83 at six; action K/V copy 1.1 against 2.9. Nominal-path floor 218 to 222 ms.
  Like-for-like with the PyTorch stream of 2026-11-10 (one sample, MAXN, 0.331 s median, 0.544 s p95): the FP8
  engines are slower at the median and faster at the tail; the recipe's win on this board is the tail, not the median.

## Probe: where a decode token's time goes (thor_probe.py, one-candidate engines, example clip)

- Decode engine call alone: 11.8 ms per token pipelined (40 calls, one sync), 12.6 ms with a sync per call; the
  recipe's full loop step 12.8 ms. argmax and the embedding lookup are 0.06 ms each. So the token cost is GPU work,
  not host overhead; a first probe that showed 1.1 ms per token was a timing bug in the probe (start time reset per
  iteration) and is withdrawn. Spin scheduling on the CUDA primary context changed nothing; the GPU clock read
  1575 MHz throughout every section.
- Reading: at batch one every token streams the whole FP8 language model (about 2 GB) through memory; on this
  board's bandwidth that is most of the 12 ms. The lever is weight bytes (NVFP4, half) and tokens per weight read
  (speculative decoding), which is what both NVIDIA's 220 ms recipe (drafter) and Embedl's build (NVFP4) do.
- Prefill alone 96 ms (3261 tokens); expert 34 ms for 5 steps at one candidate; ViT 27 ms for 6 frames, 122 for 22.
- The FastRuntime (token kept on the GPU, end checked per chunk) is still worth measuring: it removes the host
  round trip per token (about 1 ms each here) and keeps the stream full between tokens.
- FastRuntime (fast_runtime.py: token kept on the GPU, end of chain checked once per 8 tokens): example clip,
  one candidate, 60 decisions: median 467.6 ms, decode 286.5 ms for 22 tokens (24 decoded with the chunk overshoot),
  same chain of causation as the recipe. No gain: the host round trip was never the cost. First version produced a
  different chain because LanguageModel.forward stages its six request scalars through one pinned buffer that the
  host overwrote before the previous token's copy ran; a ring of pinned buffers fixed it. Kept as a negative result.

## NVFP4 language model (own addition; Embedl's precision choice)

- `quantize_nvfp4.py`: the recipe's quantize wrapper with `quantization="nvfp4"` (language model), visual FP8, K/V
  FP8, lm_head FP16, calibrated on the example clip's 22 frames; 21 s on the 5090. ModelOpt writes a MIXED_PRECISION
  `hf_quant_config.json` listing every quantized module by HF name.
- Issue 5: `host/shim.py` renames the language model's `mlp.fc1`/`mlp.fc2` to `up_proj`/`down_proj` but copies
  `hf_quant_config.json` verbatim, so the per-layer NVFP4 entries no longer match the modules Edge-LLM builds
  (`layers.N.mlp.up_proj`), the exporter treats the packed U8 weights as FP16 and torch.export fails with
  "a and b must have same reduction dim ... [s59, 2048] X [1024, 9216]". The FP8 recipe never hit this because its
  config is a single global `quant_algo: FP8` with no per-layer names. Fixed by rewriting the config's keys with the
  same renames and dropping the modules the shim restored to FP16 (patch embedding, vision layers 0 and 1).
- Issue 6 (NVFP4 W4A16 engine build): `Nvfp4A16BlackwellGemmPlugin: no TCGen05 AOT variant for token tile 8 (M=2)
  N=1024 K=2048`. Cause: `Nvfp4A16BlackwellGemmRunner::isSupported` returns false unless the plugin was compiled with
  `CUTE_DSL_NVFP4_A16_BLACKWELL_GEMM_ENABLED`, i.e. the CuTe DSL kernel group that generates the Thor NVFP4-A16 GEMMs.
  Edge-LLM's JetPack build commands pass `-DENABLE_CUTE_DSL=ALL`; the recipe's `thor/build.py` leaves the default
  (`fmha` only) because its FP8 path never needs the NVFP4 kernels. Fix: reconfigure with the CuTe DSL groups on.
- Issue 7 (NVFP4 W4A4 engine build): TensorRT 10.13.3.9 stops in Myelin with "Not implemented for node type PLUGIN_V3
  (removeEmptyProducersFromSubgraph)". A TensorRT-version limitation on the W4A4 graph (the recipe pins 11.3.1.9);
  parked, W4A16 is the Thor-optimized path anyway (Edge-LLM 0.11.0: "Thor-optimized NVFP4-A16 dense kernels").
- NVFP4 W4A16 engines (language model NVFP4 weights, FP16 activations; vision FP8; FP8 K/V; plugin rebuilt with
  `-DENABLE_CUTE_DSL=ALL`): SLOWER than FP8. Example clip, one candidate, 180 decisions: median 576.2 ms
  (FP8: 398.6), decode 375.1 ms for 25 tokens = 15.0 ms per token (FP8: 11.8), prefill 129.1 (96), ViT 35.2 (25).
  Nine clips, one candidate: median 461.5, p95 593.1, 37.2 J (FP8: 367.2 / 427.2 / 29.1). Six candidates: 511.9 /
  627.3 / 42.2 J (FP8: 413.1 / 493.3 / 33.4).
  The chain of causation changed on two of nine clips (example clip 25 tokens instead of 22, 030c760c 27 instead
  of 26; the four nominal clips and the other three reasoning clips unchanged), so this calibration (22 frames of one
  clip) also misses the study's chain-text budget.
  Reading: 15 ms per token is about what streaming 4 GB of FP16 weights would cost on this board, as if the NVFP4
  weights were dequantized to FP16 rather than consumed packed; and ViT and prefill, whose engines did not change
  precision, also slowed, which points at the rebuilt plugin library (every engine loads it at run time). Both to be
  checked: engine sizes, plugin use in the build log, FP8 re-measured under the rebuilt plugin.
- FP8 one-candidate engines re-measured under the rebuilt plugin library (CuTe DSL ALL): example clip 60 decisions
  median 401.8 ms, ViT 24.9, prefill 98.2, decode 242.4, expert 35.6: unchanged from 398.6. The plugin rebuild is not
  the cause of the W4A16 slowdown. Engine weights: NVFP4 language engine 1.35 GB (FP8: 1.96 GB), so the weights are
  packed, not folded to FP16. The W4A16 build log warns `Attribute gemv_jit_bundle not found in plugin node` on every
  Nvfp4A16BlackwellGemmPlugin: the plugin's decode (M=1) path expects a JIT GEMV kernel bundle from the exporter and
  without it runs the token-tile-8 GEMM for single tokens. Also the projector (model.projector.linear_fc1/2) was
  quantized to NVFP4 in this build, which puts W4A16 GEMMs at M of about 2900 into the vision engine (ViT 35 against
  25 ms) and the prefill is W4A16 at M 3261 (129 against 96 ms): dequantize-and-multiply costs compute where FP8 is
  native.
- On the plugin's dispatch (nvfp4A16BlackwellGemmPlugin.cpp): M=1 goes to a GEMV kernel JIT-compiled at engine build
  (nvfp4A16BlackwellGemvJitCompiler), every M>1 to the TCGen05 GEMM; the build passed both checks, so the decode did
  run the GEMV path. At 1.35 GB of packed weights per token, 15 ms is about 90 GB/s effective against about 166 GB/s
  for the FP8 GEMM path (1.96 GB in 11.8 ms): on this stack (Edge-LLM 0.11.1, TensorRT 10.13.3.9, JetPack 7.1) the
  NVFP4-A16 single-token kernel does not turn the halved bytes into time. W4A4 (NVFP4 activations too, the tensor-core
  FP4 path) is the variant that could, and it does not build on TensorRT 10.13 (issue 7). JetPack 7.2.2 ships a newer
  TensorRT (Edge-LLM's own benchmarks on Jetson use 10.16); an upgrade is a system change for the owner to decide.

## Three Euler steps, and the accuracy check across builds

- FP8, one candidate, three Euler steps (own export, `onnx/fp8-s3-c1`): nine clips, 20 decisions each: median 352.6 ms
  (five steps: 367.2), p90 439.6, p95 511.7 (427.2; clip 030c760c ran at 510 ms in this set against 427 before, with
  the same 26 tokens, so the tail is run-to-run variance on that clip rather than the step count), 29.4 J; expert
  35.3 to 25.3 ms. Trajectories against the five-step one-candidate build (same noise): ADE 0.04 to 0.19 m on eight
  clips, 0.60 m on 195f1fc6.
- Trajectories, `host/compare.py` decode, ADE in metres, same noise seed across builds:
  FP8 six candidates against FP8 one candidate, first candidate: 0.002 to 0.011 m (the engines agree).
  FP8 against NVFP4 W4A16, first candidates: 0.008 to 0.329 m (0347d9f9 the largest), closest pair 0.008 to 0.2 m.
  Spread among the six FP8 candidates on the same clip: mean 0.70 to 1.66 m, max 1.70 to 4.14 m. Both the NVFP4 and
  the three-step differences sit inside that spread, which was the study's trajectory budget.
  Against the PyTorch BF16 waypoints of case study 06 (same clips, t0 and navigation text; one sample at seed 43,
  so a different noise draw than the engines' seed 0): 0.19 to 1.52 m for FP8, 0.18 to 1.83 m for NVFP4. With
  different noise this is a sanity bound, not a quantization error; the like-for-like PyTorch reference would need
  the same noise tensor fed to both.
