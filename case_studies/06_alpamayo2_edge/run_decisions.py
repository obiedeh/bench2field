"""Run Alpamayo 2 Edge driving decisions on fixed clips and record what each one cost.

Case study 06 measures NVIDIA's 2B Alpamayo 2 Edge, the student, with the method
case study 02 used for the 10B Alpamayo 1 teacher, so the two records compare.
One decision is one call of the upstream ``sample_trajectories_from_data``: the
vision-language model reads six cameras (22 frames) plus 20 ego poses and the
navigation instruction, writes its reasoning (or the adaptive nominal path), and
the diffusion expert samples 64 future poses at 0.1 s. The call, its inputs and
the sampling settings are the upstream CLI's defaults, unchanged (PLAN.md).

What this runner adds around the call, without changing upstream code:

    load            load_model start to model on the device, CUDA synchronised,
                    and GPU memory after load
    per decision    total wall time (prepared inputs on the device to outputs on
                    the device), time to the first generated token (vision
                    encoding plus prompt prefill), decode time and generated
                    token count, the diffusion expert's time, peak memory, the
                    reasoning text, the 64 waypoints and minADE against the
                    logged future as a sanity value (case study 02 recorded the
                    same value for the teacher)
    per clip        input preparation time on the CPU (processor, tokenizer),
                    reported apart from the decision as case study 02 did
    telemetry       bench2field's samplers at 0.5 s: NVML on the 5090 (GPU rail
                    power), tegrastats on the Thor (VIN, the whole board)

Variants: ``eager`` is the shipped default (SDPA, no acceleration); ``tuned``
enables the upstream ``alpamayo2_edge.accel`` package in its tuned mode
(torch.compile and CUDA graphs, needs flash-attn), warmed up before timing.

    PYTHONPATH=<bench2field>/src .venv/bin/python run_decisions.py --board bench-5090 --variant eager
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
BOARDS = ("bench-5090", "bench-thor")
VARIANTS = ("eager", "tuned")
MODEL_REVISION = "f8951d095de3589b3face1dec3f4394c8ecedb38"
# The upstream CLI's settings (alpamayo2_edge/inference.py at 02f55a7), unchanged.
SETTINGS = dict(top_p=0.98, top_k=None, temperature=0.6, num_traj_sets=1, num_traj_samples=1,
                max_generation_length=256, diffusion_steps=10)
SEED = 43  # the upstream CLI default


def install_prep_mode(mode: str) -> None:
    """Input preparation variants, all outside the timed decision.

    upstream  NVIDIA's prepare_model_inputs as shipped: it rebuilds the processor
              from disk on every call and resizes the 22 images on the CPU.
    cached    the same code with the processor built once per model (numerically
              identical). The config object is not hashable, so the cache is keyed
              by object identity.
    gpu       cached, and each image's bicubic resize done on the GPU (frames copied
              there and the resized image copied back), because the upstream loader
              validates that the frames it receives are CPU tensors. GPU and CPU
              bicubic resizes can round differently; the record keeps the difference.
    """
    if mode == "upstream":
        return
    from alpamayo2_edge import helper

    orig_get = helper.get_processor
    memo: dict[tuple[int, int], Any] = {}

    def get_processor(tokenizer: Any, model_config: Any) -> Any:
        key = (id(tokenizer), id(model_config))
        if key not in memo:
            memo[key] = orig_get(tokenizer, model_config)
        return memo[key]

    helper.get_processor = get_processor
    if mode == "gpu":
        orig_resize = helper.resize

        def resize(frame: torch.Tensor, size: list[int], **kw: Any) -> torch.Tensor:
            return orig_resize(frame.to("cuda", non_blocking=True), size, **kw).cpu()

        helper.resize = resize


def quantize_fp8(model: Any) -> dict[str, Any]:
    """FP8 weights and dynamic FP8 activations (per row) with torchao, on the language
    model and the action expert's transformer. The vision tower, the language model
    head and the small action input and output projections stay BF16, as in NVIDIA's
    Model Optimizer example for Alpamayo. Returns what was converted."""
    from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerTensor, quantize_

    targets = {"language_model": model.vlm.model.language_model}
    counts = {}
    for name, module in targets.items():
        n = sum(1 for m in module.modules() if isinstance(m, torch.nn.Linear))
        quantize_(module, Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()),
                  filter_fn=lambda m, fqn: isinstance(m, torch.nn.Linear) and "lm_head" not in fqn)
        counts[name] = n
    import torchao

    return {"library": f"torchao {torchao.__version__}", "scheme": "float8 weights, dynamic float8 activations, per tensor",
            "linear_layers": counts, "kept_bf16": ["vision tower", "projector", "lm_head", "action_in_proj", "action_out_proj"]}


def prepare(sample: dict, model: Any, mode: str) -> dict:
    from alpamayo2_edge.helper import prepare_model_inputs, to_device

    return to_device(prepare_model_inputs(sample, model.config, model.tokenizer), device="cuda")


class TokenClock:
    """A stopping criterion that timestamps every generated token and never stops."""

    def __init__(self) -> None:
        self.times: list[float] = []

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        torch.cuda.synchronize()
        self.times.append(time.perf_counter())
        return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)


class Hooks:
    """Wrap the VLM's generate and the expert's diffusion sampler to time them."""

    def __init__(self, model: Any) -> None:
        from transformers import StoppingCriteriaList

        self.reset()
        self._generate = model.vlm.generate
        self._sample = model.expert.diffusion.sample

        def generate(*args: Any, **kwargs: Any) -> Any:
            clock = TokenClock()
            crit = kwargs.get("stopping_criteria")
            crit = StoppingCriteriaList(list(crit) if crit is not None else [])
            crit.append(clock)
            kwargs["stopping_criteria"] = crit
            torch.cuda.synchronize()
            self.generate_start = time.perf_counter()
            out = self._generate(*args, **kwargs)
            torch.cuda.synchronize()
            self.generate_end = time.perf_counter()
            self.first_token = clock.times[0] if clock.times else None
            return out

        def sample(*args: Any, **kwargs: Any) -> Any:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = self._sample(*args, **kwargs)
            torch.cuda.synchronize()
            self.trajectory_s = time.perf_counter() - t0
            return out

        model.vlm.generate = generate
        model.expert.diffusion.sample = sample

    def reset(self) -> None:
        self.generate_start = self.generate_end = self.first_token = None
        self.trajectory_s = None


def gpu_memory() -> dict[str, float]:
    return {
        "torch_allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 3),
        "torch_max_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "torch_max_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
    }


def energy_counter_mj() -> int | None:
    """The 5090's cumulative NVML energy counter in millijoules; None where NVML has none (the Thor)."""
    try:
        import pynvml

        pynvml.nvmlInit()
        return int(pynvml.nvmlDeviceGetTotalEnergyConsumption(pynvml.nvmlDeviceGetHandleByIndex(0)))
    except Exception:
        return None


def start_sampler(board: str, interval_s: float = 0.1) -> Any | None:
    """NVML on the 5090 (GPU rail), tegrastats on the Thor (VIN, whole board)."""
    try:
        from bench2field.telemetry.nvml import NvmlSampler
        from bench2field.telemetry.tegrastats import TegrastatsSampler

        cls = TegrastatsSampler if board == "bench-thor" else NvmlSampler
        if not cls.available():
            return None
        s = cls(interval_s=interval_s)
        s.start()
        return s
    except Exception:
        return None


def power_summary(samples: list[dict[str, float]]) -> dict[str, Any] | None:
    for channel in ("power_board_w", "power_gpu_w"):
        vals = [s[channel] for s in samples if channel in s]
        if vals:
            return {"mean_w": round(statistics.fmean(vals), 1), "max_w": round(max(vals), 1),
                    "n": len(vals), "channel": channel}
    return None


DIFFUSION_STEPS = [SETTINGS["diffusion_steps"]]  # set from --diffusion-steps


def predict(model: Any, prepared: dict[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    """The upstream CLI's predict(), with its settings."""
    torch.manual_seed(SEED)
    generator = torch.Generator(device="cuda").manual_seed(SEED)
    with torch.inference_mode(), torch.autocast("cuda", dtype=model.dtype, enabled=model.dtype != torch.float32):
        xyz, rot, _, extra = model.sample_trajectories_from_data(
            prepared,
            num_traj_sets=SETTINGS["num_traj_sets"],
            num_traj_samples=SETTINGS["num_traj_samples"],
            top_p=SETTINGS["top_p"],
            top_k=SETTINGS["top_k"],
            temperature=SETTINGS["temperature"],
            max_generation_length=SETTINGS["max_generation_length"],
            diffusion_kwargs={"inference_step": DIFFUSION_STEPS[0], "generator": generator},
            return_extra=True,
        )
    return xyz, rot, extra


def one_decision(model: Any, hooks: Hooks, prepared: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    hooks.reset()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    e0 = energy_counter_mj()
    m0 = time.monotonic()
    t0 = time.perf_counter()
    xyz, _rot, extra = predict(model, prepared)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    m1 = time.monotonic()
    e1 = energy_counter_mj()
    ids = extra["generated_ids"]
    gen = int((ids[0] >= 0).sum().item()) if torch.is_tensor(ids) else None
    ttft = (hooks.first_token - t0) if hooks.first_token else None
    decode_s = (hooks.generate_end - hooks.first_token) if hooks.first_token and hooks.generate_end else None
    cot = extra.get("cot")
    trace = str(np.asarray(cot).reshape(-1)[0]) if cot is not None else None
    pred = xyz.float().cpu().numpy().reshape(-1, 64, 3)[0]
    gt = np.asarray(meta["future_xyz_ego"], dtype=np.float64)
    ade = float(np.linalg.norm(pred[:, :2] - gt[:, :2], axis=-1).mean())
    return {
        "clip_id": meta["clip_id"], "t0_us": meta["t0_us"], "nav_text": meta["nav_text"],
        "wall_start": m0, "wall_end": m1,
        "latency_total_s": round(t1 - t0, 4),
        "ttft_s": round(ttft, 4) if ttft else None,
        "decode_s": round(decode_s, 4) if decode_s else None,
        "trajectory_s": round(hooks.trajectory_s, 4) if hooks.trajectory_s else None,
        "generated_tokens": gen,
        "nominal_path": trace is not None and trace.strip() == "<NOMINAL>",
        "energy_counter_j": round((e1 - e0) / 1000.0, 2) if e0 is not None and e1 is not None else None,
        "decode_tokens_per_s": round((gen - 1) / decode_s, 2) if decode_s and gen and gen > 1 else None,
        "peak_memory": gpu_memory(),
        "min_ade_m_sanity_only": round(ade, 3),
        "trace": trace,
        "waypoints_xyz": [[round(float(v), 3) for v in p] for p in pred],
    }


def summarize(ds: list[dict[str, Any]]) -> dict[str, Any]:
    lat = sorted(d["latency_total_s"] for d in ds)
    med = statistics.median(lat)
    p95 = lat[min(len(lat) - 1, int(round(0.95 * (len(lat) - 1))))]
    pw = [d["power"]["mean_w"] for d in ds if d.get("power")]
    en = [d["power"]["mean_w"] * d["latency_total_s"] for d in ds if d.get("power")]

    def med_of(key: str) -> Any:
        vals = [d[key] for d in ds if d.get(key) is not None]
        return round(statistics.median(vals), 3) if vals else None

    return {
        "n_decisions": len(lat), "latency_median_s": round(med, 3), "latency_p95_s": round(p95, 3),
        "latency_min_s": round(lat[0], 3), "latency_max_s": round(lat[-1], 3),
        "decisions_per_minute": round(60.0 / med, 2), "decisions_per_second": round(1.0 / med, 2),
        "ttft_median_s": med_of("ttft_s"), "decode_median_s": med_of("decode_s"),
        "trajectory_median_s": med_of("trajectory_s"), "generated_tokens_median": med_of("generated_tokens"),
        "decode_tokens_per_s_median": med_of("decode_tokens_per_s"),
        "peak_torch_allocated_gib_max": max(d["peak_memory"]["torch_max_allocated_gib"] for d in ds),
        "power_mean_w_median": round(statistics.median(pw), 1) if pw else None,
        "power_channel": next((d["power"]["channel"] for d in ds if d.get("power")), None),
        "energy_per_decision_j_median": round(statistics.median(en), 1) if en else None,
        "min_ade_m_median_sanity_only": med_of("min_ade_m_sanity_only"),
        "energy_counter_j_median": med_of("energy_counter_j"),
        "nominal_path_share": round(sum(1 for d in ds if d.get("nominal_path")) / len(ds), 3),
    }


def sustained_block(model: Any, prepared: list, seconds: float) -> tuple[float, float, int, int | None, int | None]:
    """Decisions back to back over the clips for about ``seconds``; returns the window and counters."""
    torch.cuda.synchronize()
    e0 = energy_counter_mj()
    m0 = time.monotonic()
    n = 0
    while time.monotonic() - m0 < seconds:
        predict(model, prepared[n % len(prepared)])
        n += 1
    torch.cuda.synchronize()
    m1 = time.monotonic()
    return m0, m1, n, e0, energy_counter_mj()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--board", required=True, choices=BOARDS)
    ap.add_argument("--variant", required=True, choices=VARIANTS)
    ap.add_argument("--model", required=True, help="local model directory at MODEL_REVISION")
    ap.add_argument("--nav", type=Path, required=True, help="nav_futures.json written by derive_nav.py --futures (outside git)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--max-clips", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--sample-interval", type=float, default=0.1)
    ap.add_argument("--diffusion-steps", type=int, default=SETTINGS["diffusion_steps"])
    ap.add_argument("--prep", choices=("upstream", "cached", "gpu"), default="upstream")
    ap.add_argument("--sample-cache", type=Path, default=None,
                    help="directory for loaded clips (torch.save), kept outside git; skips re-streaming")
    ap.add_argument("--tag", default="", help="suffix for the record name, e.g. steps5")
    ap.add_argument("--fp8", action="store_true", help="torchao FP8 on the language model and expert")
    ap.add_argument("--sustained-s", type=float, default=30.0)
    args = ap.parse_args(argv)

    from alpamayo2_edge.checkpoint import load_model
    from alpamayo2_edge.load_physical_aiavdataset import load_physical_aiavdataset

    nav = json.loads(args.nav.read_text())
    metas = nav["samples"][: args.max_clips] if args.max_clips else nav["samples"]
    DIFFUSION_STEPS[0] = args.diffusion_steps
    name = f"decisions_{args.board}_{args.variant}" + (f"_{args.tag}" if args.tag else "")
    out = args.out or HERE / "runs" / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    (HERE / "traces").mkdir(exist_ok=True)

    sampler = start_sampler(args.board, args.sample_interval)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model, loading_info = load_model(args.model, device="cuda", dtype="bfloat16", attn_implementation="sdpa")
    torch.cuda.synchronize()
    load = {"load_total_s": round(time.perf_counter() - t0, 2), "memory_after_load": gpu_memory(),
            "parameters_total": sum(p.numel() for p in model.parameters())}
    if args.fp8:
        load["fp8"] = quantize_fp8(model)
        torch.cuda.synchronize()
        load["memory_after_fp8"] = gpu_memory()
    print(f"[run] loaded in {load['load_total_s']} s, {load['parameters_total'] / 1e9:.2f} B params", flush=True)

    t_data = time.perf_counter()
    samples = []
    for m in metas:
        cached = args.sample_cache / f"{m['clip_id']}_{m['t0_us']}.pt" if args.sample_cache else None
        if cached is not None and cached.exists():
            samples.append(torch.load(cached, weights_only=False))
            continue
        smp = load_physical_aiavdataset(m["clip_id"], t0_us=int(m["t0_us"]), nav_text=m["nav_text"])
        if cached is not None:
            cached.parent.mkdir(parents=True, exist_ok=True)
            torch.save(smp, cached)
        samples.append(smp)
    data_load_s = round(time.perf_counter() - t_data, 2)
    install_prep_mode(args.prep)
    prepared, prep_s = [], []
    for smp in samples:
        tp = time.perf_counter()
        p = prepare(smp, model, args.prep)
        torch.cuda.synchronize()
        prep_s.append(round(time.perf_counter() - tp, 4))
        prepared.append(p)
    prep_check = None
    if args.prep != "upstream":  # compare with NVIDIA's path on the first clip
        from alpamayo2_edge.helper import prepare_model_inputs

        ref = prepare_model_inputs(samples[0], model.config, model.tokenizer)
        a, b = ref["tokenized_data"]["pixel_values"].float(), prepared[0]["tokenized_data"]["pixel_values"].float().cpu()
        prep_check = {"same_shape": list(a.shape) == list(b.shape),
                      "pixel_values_max_abs_diff": round(float((a - b).abs().max()), 6) if a.shape == b.shape else None,
                      "input_ids_equal": bool(torch.equal(ref["tokenized_data"]["input_ids"], prepared[0]["tokenized_data"]["input_ids"].cpu()))}
        print(f"[prep] {args.prep}: {prep_check}", flush=True)
    print(f"[run] {len(samples)} clips loaded in {data_load_s} s; input prep median {statistics.median(prep_s)} s", flush=True)

    accel_info = None
    if args.variant == "tuned":
        from alpamayo2_edge.accel import AccelConfig, enable_acceleration

        acc = enable_acceleration(model, AccelConfig(mode="tuned"))
        tw = time.perf_counter()
        for p in prepared:  # build graphs for every input shape before timing
            predict(model, p)
        accel_info = {"describe": str(acc.describe()), "warmup_all_clips_s": round(time.perf_counter() - tw, 1)}

    hooks = Hooks(model)
    warm = [one_decision(model, hooks, prepared[0], metas[0]) for _ in range(args.warmup)]
    for w in warm:
        w["role"] = "warmup"
        print(f"[warmup] {w['latency_total_s']} s", flush=True)
    decisions = []
    for r in range(args.repeats):
        for p, m in zip(prepared, metas):
            d = one_decision(model, hooks, p, m)
            d["repeat"] = r
            decisions.append(d)
            print(f"[r{r} {m['clip_id'][:8]}] {d['latency_total_s']} s, ttft {d['ttft_s']}, "
                  f"{d['generated_tokens']} tok, traj {d['trajectory_s']}, minADE {d['min_ade_m_sanity_only']}", flush=True)
            if r == 0:
                (HERE / "traces" / f"{args.board}_{args.variant}_{m['clip_id'][:8]}.txt").write_text(
                    f"# {args.board}, {args.variant}, clip {m['clip_id']}, t0_us {m['t0_us']}, nav \"{m['nav_text']}\"\n"
                    f"# generated tokens {d['generated_tokens']}, latency {d['latency_total_s']} s\n\n{d['trace']}\n")

    sm0, sm1, sn, se0, se1 = sustained_block(model, prepared, args.sustained_s)
    if sampler:
        sampler.stop()
    tel = sampler.samples if sampler else []
    sus_win = [s for s in tel if sm0 <= s.get("t", -1) <= sm1]
    sus_pw = power_summary(sus_win)
    sustained = {"seconds": round(sm1 - sm0, 2), "decisions": sn,
                 "seconds_per_decision": round((sm1 - sm0) / sn, 4), "power": sus_pw,
                 "energy_per_decision_j_from_mean_power": round(sus_pw["mean_w"] * (sm1 - sm0) / sn, 1) if sus_pw else None,
                 "energy_per_decision_j_from_counter": round((se1 - se0) / 1000.0 / sn, 1) if se0 is not None and se1 is not None else None}
    print(f"[sustained] {sustained}", flush=True)
    for d in warm + decisions:
        win = [s for s in tel if d["wall_start"] <= s.get("t", -1) <= d["wall_end"]]
        d["power"] = power_summary(win)

    import transformers

    record = {
        "schema": "bench2field.case_study_06.decisions/1",
        "label": "lab result",
        "board": args.board, "variant": args.variant,
        "model": {"id": "nvidia/Alpamayo2-Edge", "revision": MODEL_REVISION, "dtype": "bfloat16"},
        "code": {"repository": "NVlabs/alpamayo2-edge", "commit": "02f55a7"},
        "versions": {"torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                     "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(0)},
        "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": {**SETTINGS, "diffusion_steps": args.diffusion_steps, "prep": args.prep, "fp8": args.fp8, "seed": SEED, "repeats": args.repeats, "warmup": args.warmup,
                     "nav_rule": nav["rule"], "clips": [m["clip_id"] for m in metas]},
        "load": load, "data_load_s": data_load_s, "input_prep_s": prep_s, "input_prep_median_s": statistics.median(prep_s), "prep_check": prep_check, "acceleration": accel_info,
        "summary": summarize(decisions), "sustained": sustained, "telemetry_interval_s": args.sample_interval,
        "warmup": warm, "decisions": decisions,
    }
    out.write_text(json.dumps(record, indent=1))
    print(json.dumps(record["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
