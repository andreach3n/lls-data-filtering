"""Run the site's programs for the teen components over ~/work/sub/tokens.npy, one row = one
document, left to right. Programs see the site's interface: prefix.token_strings (site vocab),
token_ids, document_ids, position, text_prefixes. Saves ~/work/sub/preds.npz with one
(rows, 512) array per "<component>_<model>", in the site's activation units (x·v)·‖u‖."""

import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

work = Path.home() / "work" / "sub"
PROGRAMS = json.load(open(work / "teen_programs_82.json"))
VOCAB = {int(k): v for k, v in json.load(open(work / "vocab_site.json")).items()}


class TextPrefixes:
    """text_prefixes[i]: the decoded text of tokens 0..i of the row; decoded lazily."""

    def __init__(self, ids):
        self.ids = ids
        self.tok = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        if self.tok is None:
            from transformers import AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
        i = range(len(self.ids))[i]
        return self.tok.decode(self.ids[: i + 1])


class Prefix:
    def __init__(self, strings, ids, doc, n):
        self.token_strings = strings[:n]
        self.token_ids = ids[:n]
        self.document_ids = [doc] * n
        self.position = n - 1
        self.text_prefixes = TextPrefixes(ids[:n])

    def __len__(self):  # some programs take len(prefix): the number of tokens so far
        return len(self.token_ids)


def run(job):
    comp, model = job
    ns = {}
    exec(compile(PROGRAMS[comp][model], f"{comp}_{model}", "exec"), ns)
    step = ns["step"]
    tokens = np.load(work / "tokens.npy")
    out = np.zeros(tokens.shape, np.float32)
    errors = 0
    for r, row in enumerate(tokens):
        ids = row.tolist()
        strings = [VOCAB.get(t, "") for t in ids]
        carry = None
        for n in range(1, len(ids) + 1):
            try:
                carry, v = step(carry, Prefix(strings, ids, r, n), {})
                out[r, n - 1] = float(v)
            except Exception:
                errors += 1
                carry = None
    return f"{comp}_{model}", out, errors


if __name__ == "__main__":
    jobs = [(str(c), m) for c in [5802, 5882, 2431, 3323, 3333, 97, 1678, 4986, 707, 5592, 1198, 4639] for m in ("sol", "haiku", "luna")]
    with mp.Pool(min(len(jobs), 100)) as pool:
        results = pool.map(run, jobs)
    for key, _, errors in results:
        if errors:
            print(f"{key}: {errors} step errors (recorded 0)")
    np.savez(work / "preds.npz", **{k: o for k, o, _ in results})
    print("saved", len(results), "program outputs")
