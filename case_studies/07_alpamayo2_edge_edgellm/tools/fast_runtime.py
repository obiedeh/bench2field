"""Runtime variant: the chain-of-causation decode loop keeps the token on the GPU.

The recipe's loop reads each token back to the host (int(argmax)) before it can decode the next
one, so the GPU idles between tokens while the host waits and resumes; on the Jetson AGX Thor
the measured cost is about 10 ms per token against 1.3 ms of GPU work. Here the argmax stays a
device tensor, feeds the next embedding lookup directly, and the host checks for the end of the
chain once per chunk of tokens. Tokens decoded past the end of the chain write K/V rows beyond the
final length; those rows are never attended (the expert reads up to ``past``) and are overwritten
by the next request, as the recipe's own verify path already relies on.
"""
import os
import sys

import torch

sys.path.insert(0, os.environ["RECIPE_THOR"])
from run import Clock, Runtime  # noqa: E402
from runtime.rope import cos_sin, mrope_positions  # noqa: E402


class FastRuntime(Runtime):
    def __init__(self, engines, candidates, chunk: int = 8):
        super().__init__(engines, candidates)
        self.chunk = chunk
        self._gen = torch.empty(512, dtype=torch.int64, device="cuda")
        # LanguageModel.forward writes the six request scalars into one pinned host buffer and copies it to the
        # device asynchronously; without a sync per token the host would overwrite that buffer before the copy of
        # the previous token has run. A ring of pinned buffers, one per in-flight token, removes the race.
        self._meta_ring = [self.lm._meta_host.clone().pin_memory() for _ in range(64)]

    def _run(self, prompt, images, *, steps, max_tokens, noise):
        lm, clock = self.lm, Clock()
        ids = prompt["token_ids"]
        future_start = prompt["special"]["<|traj_future_start|>"]
        pages = list(range(lm.max_pages))
        table = lm.page_table(pages)

        features = self.vision.encode(images)
        clock.mark("vit_ms")

        embeds = lm.embed(torch.tensor(ids))
        for image in prompt["images"]:
            embeds[image["token_start"] : image["token_start"] + image["tokens"]] = features[
                image["image"]
            ]
        pos, rope_delta = mrope_positions(ids, prompt["images"])
        lm.begin_request(rope_delta, lm.max_pages * 128)
        logits = lm.prefill(embeds, cos_sin(pos, lm.theta, lm.section).cuda(), table)
        clock.mark("prefill_ms")

        if self.drafter is not None:
            raise NotImplementedError("FastRuntime is the greedy path; use Runtime with a drafter")
        generated, past, done = [], len(ids), False
        n = 0
        while not done and n < max_tokens:
            k = min(self.chunk, max_tokens - n)
            for j in range(k):
                lm._meta_host = self._meta_ring[(n + j) % len(self._meta_ring)]
                tok = lm.argmax(logits)  # device scalar, full-vocabulary id
                self._gen[n + j] = tok
                logits = lm.decode(lm.embed(tok.reshape(1)), lm.text_rope(past + j, 1), table, past + j)
            chunk = self._gen[n : n + k].tolist()  # the one host read per chunk (synchronizes)
            for j, t in enumerate(chunk):
                generated.append(t)
                if t in self.eos:
                    past += j  # the EOS token itself is not fed, as in the recipe's loop
                    done = True
                    break
                if t == future_start:
                    past += j + 1  # fed, so its K/V exists for the expert
                    done = True
                    break
            else:
                past += k
                n += k
        clock.mark("decode_ms")

        self.action.load_prefix(lm.pool, pages, past)
        clock.mark("action_kv_ms")
        actions = self.action.sample(past, rope_delta, steps, noise)
        clock.mark("action_ms")

        text = self.tokenizer.decode(generated, skip_special_tokens=False) if self.tokenizer else None
        return {
            "generated": generated,
            "text": text,
            "kv_length": past,
            "rope_delta": rope_delta,
            "rounds": [],
            "spec_ms": None,
            "actions": actions.cpu().tolist(),
            "ms": clock.ms,
            "decode_steps": len(generated) - (1 if generated and generated[-1] in self.eos else 0),
            "vit_calls": self.vision.calls,
            "total_ms": sum(clock.ms.values()),
        }
