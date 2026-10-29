# Case study 02: Alpamayo 1, an AV reasoning model, on the bench and the edge

**Question:** what does one driving decision from NVIDIA's open Alpamayo 1 model
cost on the hardware in this lab, in time, memory and power, and how much of the
bench's performance survives on a Jetson?

Alpamayo 1 (`nvidia/Alpamayo-R1-10B`, 10B parameters, weights under OpenMDW-1.1,
inference code Apache-2.0) reads four cameras over 0.4 s plus the ego vehicle's
recent motion, writes a chain-of-causation reasoning trace, and samples a 6.4 s
trajectory at 10 Hz. Every number here comes from a committed JSON record on a
named board, as in case study 01.

**Status: phase 1 done, nothing measured yet.** Results below say "pending" until
the run that produces them is committed.

## Boards

| Board | Role | Target for | State |
|---|---|---|---|
| RTX 5090 (`bench-5090`) | bench | full-precision run, profile, plots | environment built and recorded |
| Jetson AGX Thor (`bench-thor`) | edge bench | same clips, with power | planned (phase 4); JetPack, not DriveOS |
| Jetson Orin NX 8 GB (`field-orin`) | field | nothing | does not fit; documented in the friction log |

## Results (pending)

| Metric | 5090 | Thor |
|---|---|---|
| Model load time | pending | pending |
| Latency per decision, median and p95 | pending | pending |
| Peak GPU memory | pending | pending |
| Decisions per minute | pending | pending |
| Board power during a decision | n/a | pending |

## Phases

1. Review and environment: the venv, the pins, the friction log, this scaffold. Done.
2. 5090 run on a handful of PhysicalAI-AV clips: load time, latency per decision, peak memory, throughput, sample reasoning traces.
3. Profile: where the time goes between the vision encoder, the language rollout and the diffusion trajectory head, with plots.
4. Thor, same clips, power from tegrastats, compared with the 5090.
5. Orin NX: why a quantized run is not feasible on this module.

## Layout

```text
README.md            this file
SETUP.md             exact setup per board
FRICTION_LOG.md      what broke, how it was found, what fixed it, with times
setup_env.sh         builds .venv on the 5090 host
capture_env.py       writes runs/env_<board>.json
requirements.txt     the pins, for reading; setup_env.sh installs them in the right order
runs/                committed records only (environment, later latency and profiles)
.venv/, weights/, clips/, data/   ignored
```

## Sources

- [NVlabs/alpamayo](https://github.com/NVlabs/alpamayo), pinned at commit `11a0e01`
- [Model card](https://huggingface.co/nvidia/Alpamayo-R1-10B)
- [PhysicalAI-Autonomous-Vehicles dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) (gated; terms accepted by the owner)
- Alpamayo 1.5 and the recipes hub are newer; this study uses Alpamayo 1 deliberately and keeps the model id as a parameter so a 1.5 row can be added with the same runner.
