# Case study 05: Cosmos Predict 2.5, the cost of one video forecast on the bench

**Question:** what does one forecast from NVIDIA's open Cosmos-Predict2.5-2B
world model cost on the RTX 5090, in wall time, peak GPU memory and frames
produced, and does it fit the Jetson AGX Thor at all?

Status: **planned, not started.** Nothing here is measured. The workstation
GPU is held by a long simulation campaign at the time of writing and the
model's own card asks for more memory than the 5090 has, so the first run
will tell whether a forecast fits at all before anything else is asked.

## Protocol (fixed before the first run)

| Item | Choice |
|---|---|
| Model | `nvidia/Cosmos-Predict2.5-2B` (base, post-trained), weights under the NVIDIA Open Model License with the guardrails kept on; inference code from the upstream repository at a pinned commit |
| Input | one context clip from a public, licence-clear driving or robot video; the model conditions on its last five contiguous frames whatever the clip length, and the record says so |
| Output | the native resolution and frame count the pinned checkpoint resolves to (read from the instantiated model, not assumed), saved as video and as frame count |
| Measured | wall time from request to saved output, time inside the denoising loop, peak process GPU memory from NVML sampled at 100 ms, the allocator's own peak, GPU power during the run, host RAM peak |
| Repeats | three forecasts with the same seed (variance of cost, not of content) |
| Boards | RTX 5090 first; Thor only if the 5090 peak memory says it can fit in the Thor's unified memory with room for the OS |
| Not measured | forecast quality of any kind: this is a cost study, no scientific claim is made about the frames |
| Record | one JSON per forecast with the pinned commit, checkpoint revision, input hash, parameters, every number above, and the board label; `plot_results.py --table` style regeneration |

## Why these rules

- The model card's memory recommendation is above the 5090's 32 GB. The
  first result of value is the measured peak, not the frames.
- Five conditioning frames is what the model uses; a longer clip does not
  condition it longer. The record states the conditioning span in seconds so
  nobody reads clip length as context length.
- Safety models and face blur stay on because the licence asks for them;
  their time is part of the cost and is reported inside the wall time.

## Phases

1. Environment: a separate Python environment for the pinned upstream, a
   check that imports and one BF16 matrix operation run on the 5090, the
   checkpoint and all safety and encoder weights fetched with existing read
   access (no new accounts). Nothing enters the repository's own environment.
2. One forecast on the 5090 with the protocol above; stop and record if it
   does not fit.
3. Three repeats, the table, the plots.
4. Thor decision from the measured peak; a Thor run only if it fits.

## What this is not

Not a benchmark of the model's usefulness, not a comparison between world
models, and not a product feature. It is the same question the other case
studies ask of a model: what does it cost on the hardware in this lab.
