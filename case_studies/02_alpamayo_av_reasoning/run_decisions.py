"""Run Alpamayo 1 driving decisions on fixed clips and record what each one cost.

One decision is one call of the upstream
``sample_trajectories_from_data_with_vlm_rollout``: the vision-language model
reads four cameras over 0.4 s plus the ego history, writes a chain-of-causation
reasoning trace, and the diffusion expert samples a 6.4 s trajectory. The
upstream test script prints the trace and a minADE and times nothing. This
runner keeps the upstream call, inputs and sampling parameters exactly (see
PLAN.md) and adds the measurement around them:

    load            from_pretrained start to model on the device, CUDA synchronised,
                    plus GPU memory after load (torch allocated and reserved, NVML used)
    per decision    total wall time (inputs on device to outputs on device),
                    time to first generated token (vision encoding plus prefill),
                    rollout time and generated token count (so decode tokens/s),
                    trajectory sampling time (the diffusion expert),
                    peak torch allocated/reserved and NVML used memory,
                    the trace text and the 64 waypoints, minADE against the
                    dataset's future as a sanity value only
    telemetry       sampled at 0.5 s for the whole run by bench2field's samplers,
                    with the sample window of every decision recorded: NVML on the
                    5090 (GPU rail power, utilisation, clocks, temperature, memory),
                    tegrastats on the Thor (VIN board power as power_board_w, GPU
                    load, temperatures), as PLAN.md fixes it

Timing is done without changing upstream code: the VLM's ``generate`` is wrapped
to add a stopping criterion that timestamps every generated token (it never
stops anything), and the diffusion sampler is wrapped to time the trajectory
head. The rest of the call is attributed to glue.

Protocol: one warm-up decision (kept, reported apart), then ``--repeats``
passes over the clips in file order. Seed 42 before every decision, bf16
autocast, ``num_traj_samples=1``, ``top_p=0.98``, ``temperature=0.6``,
``max_generation_length=256``: the upstream defaults, unchanged.

Output: one JSON per board and variant under runs/, with the environment
record it ran in, the alpamayo_r1 and bench2field commits, a background
snapshot and summary statistics (median and p95 latency, decisions per
minute, rollout tokens per second). Traces for the first repeat of the first
three clips are written verbatim to traces/.

``--synthetic`` replaces the dataset with random frames of the camera's
resolution and a straight-line ego history. It exists to check this script and
the environment before dataset access is available; its output goes to
runs/scratch/ (ignored by git) and is labelled ``input: synthetic``. It is
never a result.

    env -u PYTHONPATH .venv/bin/python run_decisions.py --board bench-5090 --variant sdpa
    env -u PYTHONPATH .venv/bin/python run_decisions.py --board bench-5090 --variant flash
    env -u PYTHONPATH .venv/bin/python run_decisions.py --board bench-5090 --variant sdpa --synthetic --max-clips 2 --repeats 1
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
BOARDS = ("bench-5090", "bench-thor")
VARIANTS = {"sdpa": "sdpa", "flash": "flash_attention_2"}
# The diffusion expert keeps sdpa in every variant. With flash_attention_2 on the
# expert, trajectory sampling fails in flash-attn's varlen forward
# ("cu_seqlens_q must have shape (batch_size + 1)") on transformers 4.57.1;
# upstream's own demo never hits this because its config round-trip drops the
# attn_implementation to None and runs sdpa everywhere. See FRICTION_LOG.md.
EXPERT_ATTN = {"sdpa": "sdpa", "flash": "sdpa"}
MODEL_ID = "nvidia/Alpamayo-R1-10B"
SAMPLING = dict(top_p=0.98, temperature=0.6, num_traj_samples=1, max_generation_length=256)
SEED = 42
CAMERA_HW = (1080, 1920)  # the dataset's camera resolution; the processor downsamples


# --------------------------------------------------------------------------- timing hooks
class TokenClock:
    """A StoppingCriteria that records the time of every generated token and never stops."""

    def __init__(self) -> None:
        self.times: list[float] = []

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs: Any) -> Any:
        torch.cuda.synchronize()
        self.times.append(time.perf_counter())
        return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)


class Hooks:
    """Wraps model.vlm.generate and model.diffusion.sample to time them and count tokens."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.reset()
        self._generate = model.vlm.generate
        self._sample = model.diffusion.sample

        def generate(*args: Any, **kwargs: Any) -> Any:
            clock = TokenClock()
            crit = kwargs.get("stopping_criteria")
            if crit is not None:
                crit.append(clock)
            n_in = kwargs["input_ids"].shape[1]
            self.prompt_tokens = int(n_in)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = self._generate(*args, **kwargs)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            self.rollout_s = t1 - t0
            self.ttft_s = (clock.times[0] - t0) if clock.times else None
            self.generated_tokens = int(out.sequences.shape[1] - n_in)
            self.token_times = [t - t0 for t in clock.times]
            return out

        def sample(*args: Any, **kwargs: Any) -> Any:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = self._sample(*args, **kwargs)
            torch.cuda.synchronize()
            self.trajectory_s = time.perf_counter() - t0
            return out

        model.vlm.generate = generate
        model.diffusion.sample = sample

    def reset(self) -> None:
        self.rollout_s = None
        self.ttft_s = None
        self.generated_tokens = None
        self.token_times: list[float] = []
        self.trajectory_s = None
        self.prompt_tokens = None


# --------------------------------------------------------------------------- inputs
def synthetic_sample(clip_id: str, t0_us: int, num_frames: int) -> dict[str, Any]:
    """Random camera frames and a straight 10 m/s ego history, shaped like the loader's output."""
    g = torch.Generator().manual_seed(abs(hash(clip_id)) % (2**31))
    h, w = CAMERA_HW
    frames = torch.randint(0, 256, (4, num_frames, 3, h, w), generator=g, dtype=torch.uint8)
    hist = torch.zeros(1, 1, 16, 3)
    hist[0, 0, :, 0] = torch.arange(-15, 1).float()  # 1 m per 0.1 s, ending at the origin
    rot = torch.eye(3).expand(1, 1, 16, 3, 3).clone()
    fut = torch.zeros(1, 1, 64, 3)
    fut[0, 0, :, 0] = torch.arange(1, 65).float()
    return {
        "image_frames": frames,
        "ego_history_xyz": hist,
        "ego_history_rot": rot,
        "ego_future_xyz": fut,
        "clip_id": clip_id,
        "t0_us": t0_us,
    }


def dataset_sample(clip_id: str, t0_us: int, num_frames: int) -> dict[str, Any]:
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset

    return load_physical_aiavdataset(clip_id, t0_us=t0_us, num_frames=num_frames)


def build_inputs(model: Any, processor: Any, data: dict[str, Any]) -> dict[str, Any]:
    """The upstream input construction, verbatim, ending with everything on the GPU."""
    from alpamayo_r1 import helper

    messages = helper.create_message(data["image_frames"].flatten(0, 1))
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    return helper.to_device(model_inputs, "cuda")


# --------------------------------------------------------------------------- memory and telemetry
def gpu_memory(nvml_handle: Any | None) -> dict[str, float]:
    out = {
        "torch_allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 3),
        "torch_reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 3),
        "torch_max_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "torch_max_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
    }
    if nvml_handle is not None:
        try:
            import pynvml

            mem = pynvml.nvmlDeviceGetMemoryInfo(nvml_handle)
            out["nvml_used_gib"] = round(mem.used / 2**30, 3)
        except Exception:
            pass  # the Thor's NVML answers no memory query; torch's counters stand alone there
    return out


def nvml_handle() -> Any | None:
    try:
        import pynvml

        pynvml.nvmlInit()
        return pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception:
        return None


def start_sampler(board: str, interval_s: float = 0.5) -> Any | None:
    """bench2field's telemetry sampler for this board. The 5090 uses NVML (GPU rail
    power, utilisation, clocks, temperature, memory). The Thor uses tegrastats,
    because PLAN.md fixes Thor power as the VIN board rail (`power_board_w`):
    the Thor's NVML does answer power, utilisation and temperature queries,
    but for the GPU rail only, and it has no memory or clock channels. Returns
    None when the board's sampler is not available."""
    try:
        from bench2field.telemetry.nvml import NvmlSampler
        from bench2field.telemetry.tegrastats import TegrastatsSampler

        if board == "bench-thor":
            if not TegrastatsSampler.available():
                return None
            s = TegrastatsSampler(interval_s=interval_s)
        else:
            if not NvmlSampler.available():
                return None
            s = NvmlSampler(interval_s=interval_s)
        s.start()
        return s
    except Exception:
        return None


def samples_between(samples: list[dict[str, float]], t0: float, t1: float) -> list[dict[str, float]]:
    return [s for s in samples if t0 <= s.get("t", -1) <= t1]


def power_summary(samples: list[dict[str, float]]) -> dict[str, Any] | None:
    """Mean and max power over a window. Jetsons report the board rail
    (`power_board_w`, VIN on the Thor); the 5090 reports the GPU's own draw
    (`power_gpu_w`). The channel used is recorded with the numbers."""
    for channel in ("power_board_w", "power_gpu_w"):
        vals = [s[channel] for s in samples if channel in s]
        if vals:
            return {"mean_w": round(statistics.fmean(vals), 1), "max_w": round(max(vals), 1), "n": len(vals), "channel": channel}
    return None


# --------------------------------------------------------------------------- the run
def load_model(attn: str, attn_expert: str | None = None) -> tuple[Any, Any, dict[str, Any]]:
    from alpamayo_r1 import helper
    from alpamayo_r1.config import AlpamayoR1Config
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

    h = nvml_handle()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    config = AlpamayoR1Config.from_pretrained(MODEL_ID)
    config.attn_implementation = attn
    model = AlpamayoR1.from_pretrained(MODEL_ID, config=config, dtype=torch.bfloat16)
    if attn_expert and attn_expert != attn:
        model.expert.set_attn_implementation(attn_expert)
    t_cpu = time.perf_counter()
    model = model.to("cuda")
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    model.eval()
    processor = helper.get_processor(model.tokenizer)
    load = {
        "load_total_s": round(t1 - t0, 2),
        "load_from_pretrained_s": round(t_cpu - t0, 2),
        "load_to_device_s": round(t1 - t_cpu, 2),
        "attn_implementation_requested": attn,
        "attn_implementation_requested_expert": attn_expert or attn,
        "attn_implementation_vlm": getattr(model.vlm.config, "_attn_implementation", None),
        "attn_implementation_expert": getattr(model.expert.config, "_attn_implementation", None),
        "parameters_total": sum(p.numel() for p in model.parameters()),
        "memory_after_load": gpu_memory(h),
    }
    return model, processor, load


def one_decision(model: Any, processor: Any, hooks: Hooks, data: dict[str, Any], h: Any | None) -> dict[str, Any]:
    inputs = build_inputs(model, processor, data)
    hooks.reset()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.synchronize()
    m0 = time.monotonic()  # the sampler stamps with monotonic(); windows must match it
    t0 = time.perf_counter()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=inputs, return_extra=True, **SAMPLING
        )
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    m1 = time.monotonic()
    total = t1 - t0
    rollout = hooks.rollout_s or 0.0
    traj = hooks.trajectory_s or 0.0
    gen = hooks.generated_tokens or 0
    decode_s = (rollout - hooks.ttft_s) if hooks.ttft_s else None
    pred_xy = pred_xyz.float().cpu().numpy()[0, 0, :, :, :2]
    gt_xy = data["ego_future_xyz"].float().cpu().numpy()[0, 0, :, :2]
    min_ade = float(np.linalg.norm(pred_xy - gt_xy[None], axis=-1).mean(-1).min())
    return {
        "clip_id": data["clip_id"],
        "t0_us": data["t0_us"],
        "wall_start": m0,
        "wall_end": m1,
        "latency_total_s": round(total, 4),
        "ttft_s": round(hooks.ttft_s, 4) if hooks.ttft_s else None,
        "rollout_s": round(rollout, 4),
        "trajectory_s": round(traj, 4),
        "glue_s": round(total - rollout - traj, 4),
        "generated_tokens": gen,
        "decode_tokens_per_s": round((gen - 1) / decode_s, 2) if decode_s and gen > 1 else None,
        "prompt_tokens": hooks.prompt_tokens,  # upstream pops input_ids out of the dict, so read it in the hook
        "peak_memory": gpu_memory(h),
        "min_ade_m_sanity_only": round(min_ade, 3),
        "trace": extra["cot"][0][0][0] if extra and "cot" in extra else None,
        "waypoints_xyz": [[round(float(v), 3) for v in p] for p in pred_xyz.float().cpu().numpy()[0, 0, 0]],
    }


def summarize(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    lat = sorted(d["latency_total_s"] for d in decisions)
    if not lat:
        return {}
    p95 = lat[min(len(lat) - 1, int(round(0.95 * (len(lat) - 1))))]
    med = statistics.median(lat)
    tok = [d["decode_tokens_per_s"] for d in decisions if d["decode_tokens_per_s"]]
    return {
        "n_decisions": len(lat),
        "latency_median_s": round(med, 3),
        "latency_p95_s": round(p95, 3),
        "latency_min_s": round(lat[0], 3),
        "latency_max_s": round(lat[-1], 3),
        "decisions_per_minute": round(60.0 / med, 2),
        "ttft_median_s": round(statistics.median([d["ttft_s"] for d in decisions if d["ttft_s"]]), 3) if any(d["ttft_s"] for d in decisions) else None,
        "rollout_median_s": round(statistics.median(d["rollout_s"] for d in decisions), 3),
        "trajectory_median_s": round(statistics.median(d["trajectory_s"] for d in decisions), 3),
        "generated_tokens_median": statistics.median(d["generated_tokens"] for d in decisions),
        "decode_tokens_per_s_median": round(statistics.median(tok), 1) if tok else None,
        "peak_torch_allocated_gib_max": max(d["peak_memory"]["torch_max_allocated_gib"] for d in decisions),
        "peak_torch_reserved_gib_max": max(d["peak_memory"]["torch_max_reserved_gib"] for d in decisions),
        "peak_nvml_used_gib_max": max((d["peak_memory"].get("nvml_used_gib") or 0) for d in decisions) or None,
    }



def other_gpu_processes() -> list[dict[str, Any]]:
    """Other processes holding memory on GPU 0 (executable basename and MiB); [] when NVML cannot say."""
    try:
        import os

        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        procs = list(pynvml.nvmlDeviceGetComputeRunningProcesses(h)) + list(pynvml.nvmlDeviceGetGraphicsRunningProcesses(h))
    except Exception:
        return []
    out = []
    for pr in procs:
        if pr.pid == os.getpid():
            continue
        try:
            exe = Path(os.readlink(f"/proc/{pr.pid}/exe")).name
        except OSError:
            exe = "unknown"
        mib = pr.usedGpuMemory // (1024 * 1024) if isinstance(pr.usedGpuMemory, int) else None
        out.append({"exe": exe, "used_mib": mib})
    return out

def provenance(env_path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        from bench2field import background, provenance as prov

        out["bench2field"] = prov.git_state()
        if not out["bench2field"].get("commit"):
            # a copy of the tree without .git (the Thor): the copier leaves the source commit in a file
            marker = HERE.parents[1] / "SOURCE_COMMIT"
            if marker.exists():
                out["bench2field"] = {"commit": marker.read_text().strip(), "dirty": None, "root": "copied tree, see SOURCE_COMMIT"}
        snap = background.snapshot()
        # Host process names are not evidence for a GPU benchmark and can name
        # tools that have no place in the record. Keep the host's load figures
        # and list only the other processes that hold GPU memory, by executable
        # name, so a shared GPU is visible without a process inventory.
        top = snap.pop("top_processes", None) or []
        snap["host_top_cpu_pct"] = [p.get("cpu_pct") for p in top]
        snap["other_gpu_processes"] = other_gpu_processes()
        out["background"] = snap
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    try:
        import importlib.metadata as md

        direct = md.distribution("alpamayo-r1").read_text("direct_url.json")
        out["alpamayo_r1_commit"] = json.loads(direct)["vcs_info"]["commit_id"] if direct else None
    except Exception:
        out["alpamayo_r1_commit"] = None
    if env_path.exists():
        env = json.loads(env_path.read_text())
        out["env_record"] = {"path": str(env_path.relative_to(HERE)), "captured_at": env.get("captured_at")}
    else:
        out["env_record"] = None
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--board", required=True, choices=BOARDS)
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    ap.add_argument("--clips", type=Path, default=HERE / "clips.yaml")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--max-clips", type=int, default=None)
    ap.add_argument("--synthetic", action="store_true", help="random frames instead of the dataset; output to runs/scratch/")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--traces-dir", type=Path, default=None)
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(args.clips.read_text())
    clips = cfg["clips"][: args.max_clips] if args.max_clips else cfg["clips"]
    label = "synthetic" if args.synthetic else "dataset"
    out = args.out or (HERE / "runs" / ("scratch" if args.synthetic else "") / f"decisions_{args.board}_{args.variant}.json")
    traces_dir = args.traces_dir or (HERE / ("runs/scratch/traces" if args.synthetic else "traces"))
    out.parent.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)
    loader = synthetic_sample if args.synthetic else dataset_sample

    sampler = start_sampler(args.board)
    h = nvml_handle()
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[run] loading {MODEL_ID} with attn={VARIANTS[args.variant]}", flush=True)
    model, processor, load = load_model(VARIANTS[args.variant], EXPERT_ATTN[args.variant])
    print(f"[run] loaded in {load['load_total_s']} s, vlm attn={load['attn_implementation_vlm']}, "
          f"after load {load['memory_after_load']}", flush=True)
    hooks = Hooks(model)

    # data first, so data loading time (streaming, decoding) never lands inside a decision
    print(f"[run] loading {len(clips)} clips ({label})", flush=True)
    t_data = time.perf_counter()
    data = [loader(c, int(cfg["t0_us"]), int(cfg["num_frames"])) for c in clips]
    data_load_s = round(time.perf_counter() - t_data, 2)
    print(f"[run] clips ready in {data_load_s} s; frames {tuple(data[0]['image_frames'].shape)}", flush=True)

    warmups, decisions = [], []
    for i in range(args.warmup):
        d = one_decision(model, processor, hooks, data[0], h)
        d["role"] = "warmup"
        warmups.append(d)
        print(f"[warmup {i}] {d['latency_total_s']} s, {d['generated_tokens']} tokens", flush=True)
    for r in range(args.repeats):
        for d_in in data:
            d = one_decision(model, processor, hooks, d_in, h)
            d["repeat"] = r
            decisions.append(d)
            print(f"[r{r} {d['clip_id'][:8]}] total {d['latency_total_s']} s, ttft {d['ttft_s']} s, "
                  f"rollout {d['rollout_s']} s ({d['generated_tokens']} tok, {d['decode_tokens_per_s']} tok/s), "
                  f"traj {d['trajectory_s']} s, peak {d['peak_memory']['torch_max_allocated_gib']} GiB", flush=True)
            if r == 0 and len([x for x in decisions if x["repeat"] == 0]) <= 3 and d["trace"]:
                (traces_dir / f"{args.board}_{args.variant}_{d['clip_id'][:8]}.txt").write_text(
                    f"# {label} input, {args.board}, {args.variant}, clip {d['clip_id']}, t0_us {d['t0_us']}\n"
                    f"# generated tokens {d['generated_tokens']}, latency {d['latency_total_s']} s\n\n{d['trace']}\n")

    telemetry = sampler.stop() if sampler else None
    samples = sampler.samples if sampler else []
    for d in warmups + decisions:
        win = samples_between(samples, d["wall_start"], d["wall_end"])
        d["power"] = power_summary(win)
        d["telemetry_samples_in_window"] = len(win)

    record = {
        "schema": "bench2field.case_study_02.decisions/1",
        "label": "lab result" if not args.synthetic else "smoke test, synthetic input, not a result",
        "input": label,
        "board": args.board,
        "variant": args.variant,
        "attn_implementation": VARIANTS[args.variant],
        "model_id": MODEL_ID,
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": {**SAMPLING, "seed": SEED, "repeats": args.repeats, "warmup": args.warmup,
                     "t0_us": cfg["t0_us"], "num_frames": cfg["num_frames"], "clips": clips,
                     "autocast": "bf16", "dtype": "bf16"},
        "load": load,
        "data_load_s": data_load_s,
        "summary": summarize(decisions),
        "warmup": warmups,
        "decisions": decisions,
        "telemetry": {"describe": telemetry, "samples": samples} if sampler else {"describe": None, "samples": []},
        "provenance": provenance(HERE / "runs" / f"env_{args.board}.json"),
    }
    out.write_text(json.dumps(record, indent=1) + "\n")
    s = record["summary"]
    print(f"[run] wrote {out}")
    print(f"[run] {args.board} {args.variant} ({label}): median {s.get('latency_median_s')} s, p95 {s.get('latency_p95_s')} s, "
          f"{s.get('decisions_per_minute')} decisions/min, peak {s.get('peak_torch_allocated_gib_max')} GiB allocated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
