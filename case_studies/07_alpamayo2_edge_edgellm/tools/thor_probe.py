#!/usr/bin/env python3
"""Where does a decode token's 13 ms go? Times the engine call alone against the recipe's per-token loop,
and the expert and prefill calls alone, after one real decision on a clip.
Usage (after source env.sh, RECIPE_THOR set): python thor_probe.py ENGINES_DIR CLIP_DIR [--candidates 6] [--steps 5]
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

if os.environ.get("CUDA_SPIN") == "1":
    import ctypes
    _cu = ctypes.CDLL("libcuda.so.1")
    assert _cu.cuInit(0) == 0
    assert _cu.cuDevicePrimaryCtxSetFlags(0, 1) == 0  # CU_CTX_SCHED_SPIN
    print("primary context: spin scheduling", flush=True)
import torch

sys.path.insert(0, os.environ["RECIPE_THOR"])
from run import Runtime, load_images  # noqa: E402
from runtime.rope import cos_sin, mrope_positions  # noqa: E402


import threading
FREQ = "/sys/class/devfreq/gpu-gpc-0/cur_freq"
def timeit(fn, n, sync_each=True):
    freqs, stop = [], threading.Event()
    def sample():
        while not stop.is_set():
            try: freqs.append(int(open(FREQ).read()) // 1000000)
            except OSError: pass
            time.sleep(0.002)
    th = threading.Thread(target=sample, daemon=True); th.start()
    torch.cuda.synchronize()
    ts = []
    t_all = time.perf_counter()
    for _ in range(n):
        t = time.perf_counter()
        fn()
        if sync_each:
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1000)
    if not sync_each:
        torch.cuda.synchronize()
        ts = [(time.perf_counter() - t_all) * 1000 / n]
    stop.set(); th.join()
    if freqs: print(f"    gpu MHz during section: min {min(freqs)} median {statistics.median(freqs):.0f} max {max(freqs)} ({len(freqs)} samples)", flush=True)
    return ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engines", type=Path)
    ap.add_argument("clip", type=Path)
    ap.add_argument("--candidates", type=int, default=6)
    ap.add_argument("--steps", type=int, default=5)
    a = ap.parse_args()
    rt = Runtime(a.engines.resolve(), a.candidates)
    prompt = json.load(open(a.clip / "prompt.json"))
    images = load_images(a.clip, prompt)
    noise = torch.randn(a.candidates, rt.action.tokens, 2, generator=torch.Generator().manual_seed(0))
    r = rt.run(prompt, images, steps=a.steps, max_tokens=256, noise=noise)
    print("decision:", {k: round(v, 1) for k, v in r["ms"].items()}, "tokens", len(r["generated"]), flush=True)
    lm = rt.lm
    ids = prompt["token_ids"]
    pages = list(range(lm.max_pages))
    table = lm.page_table(pages)
    with torch.cuda.stream(rt.stream):
        # fresh prefill so the KV pool matches the loop below
        features = rt.vision.encode(images)
        embeds = lm.embed(torch.tensor(ids))
        for image in prompt["images"]:
            embeds[image["token_start"] : image["token_start"] + image["tokens"]] = features[image["image"]]
        pos, rope_delta = mrope_positions(ids, prompt["images"])
        rope = cos_sin(pos, lm.theta, lm.section).cuda()
        lm.begin_request(rope_delta, lm.max_pages * 128)
        t_pre = timeit(lambda: lm.prefill(embeds, rope, table), 5)
        print(f"prefill alone: median {statistics.median(t_pre):.1f} ms over 5 ({len(ids)} tokens)", flush=True)
        logits = lm.prefill(embeds, rope, table)
        past = len(ids)
        token = int(lm.argmax(logits))

        # (a) engine call only, same token and position every time, one sync at the end
        emb = lm.embed([token])
        rp = lm.text_rope(past, 1)
        t_engine = timeit(lambda: lm.decode(emb, rp, table, past), 40, sync_each=False)
        print(f"decode engine call only, no per-call sync: {t_engine[0]:.2f} ms per token", flush=True)
        t_engine_s = timeit(lambda: lm.decode(emb, rp, table, past), 40, sync_each=True)
        print(f"decode engine call only, sync each: median {statistics.median(t_engine_s):.2f} ms", flush=True)

        # (b) the recipe's loop step: argmax -> int -> embed -> rope -> decode
        state = {"logits": logits, "past": past}

        def step():
            tok = int(lm.argmax(state["logits"]))
            state["logits"] = lm.decode(lm.embed([tok]), lm.text_rope(state["past"], 1), table, state["past"])
            state["past"] += 1

        t_loop = timeit(step, 40, sync_each=True)
        print(f"recipe loop step (argmax+embed+rope+decode): median {statistics.median(t_loop):.2f} ms", flush=True)

        # (c) pieces of the loop without the engine
        t_argmax = timeit(lambda: int(lm.argmax(state["logits"])), 40)
        t_embed = timeit(lambda: lm.embed([token]), 40)
        print(f"argmax+int: median {statistics.median(t_argmax):.2f} ms; embed: median {statistics.median(t_embed):.2f} ms", flush=True)

        # (d) expert
        rt.action.load_prefix(lm.pool, pages, past)
        t_act = timeit(lambda: rt.action.sample(past, rope_delta, a.steps, noise), 10)
        print(f"expert sample ({a.steps} steps, {a.candidates} candidates): median {statistics.median(t_act):.1f} ms", flush=True)
        pass
        pass

        # (e) vision: six front/new frames vs all 22
        current = {f["image"] for f in json.load(open(a.clip / "clip.json"))["frames"] if f["t"] == 0.0}
        six = {k: v for k, v in images.items() if k in current}

        def vit6():
            rt.vision.forget(six)
            rt.vision.encode(six)

        def vit22():
            rt.vision.cache.clear()
            rt.vision.encode(images)

        print(f"vit 6 frames: median {statistics.median(timeit(vit6, 10)):.1f} ms; 22 frames: median {statistics.median(timeit(vit22, 5)):.1f} ms", flush=True)


if __name__ == "__main__":
    main()
