"""Run the site's programs over the collected token stream, one process per program.

Each program defines step(carry, prefix, upstream) and reads prefix.token_strings / prefix.document_ids
(the site's decoded token strings, from vocab_site.json). The harness feeds tokens left to right per
document, keeping a window of the last 256 tokens as the prefix (programs look back a few tokens at most).

Writes <acts>/programs_pred.npy (N_tokens, n_programs) float32 and programs_pred.json (column order).
"""
import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--acts", default="/workspace/acts/l0_down")
p.add_argument("--programs", default="/workspace/bsf_qwen/programs_teen.json")
p.add_argument("--vocab", default="/workspace/bsf_qwen/vocab_site.json")
p.add_argument("--window", type=int, default=256)
args = p.parse_args()
root = Path(args.acts)


class Prefix:
    __slots__ = ("token_strings", "document_ids")

    def __init__(self):
        self.token_strings, self.document_ids = [], []


def run_one(item):
    comp, src = item
    ns = {}
    exec(compile(src, f"program_{comp}", "exec"), ns)
    step = ns["step"]
    tokens, doc = np.load(root / "tokens.npy"), np.load(root / "doc.npy")
    vocab = {int(k): v for k, v in json.load(open(args.vocab)).items()}
    strings = [vocab.get(int(t), f"<{t}>") for t in tokens]
    docs = doc.tolist()
    out = np.zeros(len(tokens), np.float32)
    pre, carry, W = Prefix(), None, args.window
    for k in range(len(tokens)):
        pre.token_strings.append(strings[k])
        pre.document_ids.append(docs[k])
        if len(pre.token_strings) > 2 * W:
            del pre.token_strings[:W]
            del pre.document_ids[:W]
        try:
            carry, v = step(carry, pre, None)
            out[k] = v
        except Exception as e:  # a program may choke on odd tokens; record 0 and keep going
            if out[k] == 0 and k % 1000 == 0:
                print(f"{comp}: step error at {k}: {e!r}", flush=True)
            carry = None
    print(f"{comp}: done, nonzero at {np.mean(out != 0):.3%} of tokens", flush=True)
    return comp, out


if __name__ == "__main__":
    progs = json.load(open(args.programs))
    items = [(c, progs[c]["sol"]) for c in progs]
    with mp.Pool(len(items)) as pool:
        results = pool.map(run_one, items)
    order = [c for c, _ in results]
    np.save(root / "programs_pred.npy", np.stack([o for _, o in results], 1))
    json.dump(order, open(root / "programs_pred.json", "w"))
    print("saved", root / "programs_pred.npy", order)
