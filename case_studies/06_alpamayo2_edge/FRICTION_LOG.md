# Friction log, case study 06

Every error, workaround and surprise, in the order it happened. Times are from
starting work on the case study (2026-11-10). Nothing here is cleaned up later.

| When | What happened | Cause | What was done |
|---|---|---|---|
| 0 min | The upstream environment (`uv sync --locked`) and the model download (5.0 GB on disk) worked first time on the 5090 with PyTorch 2.8.0+cu128. | | None needed. |
| 10 min | NVIDIA's own example (clip ea7bbd31, "Turn right in 30m") ran and wrote a sensible reasoning line; the cold first call took 3.35 s. | First-call setup (kernels, allocator). | Warm-up decision kept out of the statistics, as in case study 02. |
| 20 min | The public dataset has no route instruction, and Edge refuses to run without one (`nav_text must be an explicit nonempty navigation instruction`). | The model is trained with navigation text; the dataset does not ship it. | Derived per clip by a fixed rule from the logged future path (`derive_nav.py`), before any timed run: 6 of 8 clips "Continue straight", 2 turns. |
| 30 min | bench2field's NVML power sampler did not start in the Edge environment. | `nvidia-ml-py` is not in the upstream lock. | Added `nvidia-ml-py` to the Edge environment for telemetry only; the model's own packages are unchanged. |
| 35 min | The model reports 2.65 billion parameters, not the "2B" in NVIDIA's announcement. | The 2B is the language backbone; the vision tower and the diffusion action expert come on top. | Recorded as measured; posts say "2B-class" or quote the count. |
| 35 min | Input preparation (processor and tokenizer on the CPU, 22 images) took about 0.79 s per clip on the 5090 host, more than twice the 0.35 s decision itself. | The upstream `prepare_model_inputs` resizes and normalises 22 camera images on the CPU. | Timed and reported apart from the decision, as case study 02 did for the teacher; candidate for the GPU preprocessing kernel from case study 02. |
| 35 min | The first two clips produced `<NOMINAL>` instead of reasoning text (6 generated tokens). | Edge's adaptive reasoning: it writes reasoning only when the scene needs it. | Recorded per decision; the summary counts how often the nominal path was taken. |
