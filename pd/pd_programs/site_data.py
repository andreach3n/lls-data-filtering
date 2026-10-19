"""Load components and their example windows from the downloaded site data."""
import json
from functools import cache
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent / "data"
SHARD = 64


@cache
def vocab():
    return {int(k): v for k, v in json.loads((DATA / "vocab.json").read_text()).items()}


@cache
def _shard(s):
    return json.loads((DATA / f"components/s{s:03d}.json").read_text())


def unit(i):
    u = _shard(i // SHARD)[i % SHARD]
    assert u["i"] == i
    return u


def windows(i):
    """Yield (token_strings, activations, causal_importance, group) per window, activations in real units."""
    u, v = unit(i), vocab()
    for w in u["windows"]:
        toks = [v.get(t, f"<{t}>") for t in w["t"]]
        act = np.array(w["a"]) / 127 * u["scale"]
        ci = np.array(w["ci"]) / 100 if w.get("ci") else None
        yield toks, act, ci, w["g"]
