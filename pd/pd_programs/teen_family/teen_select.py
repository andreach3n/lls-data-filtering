"""Does p-e9fcffd5 have the report's teen pair (layer-0 down_proj 5802 cos, 5882 sin)?

Builds text full of numbers, takes component activations x @ V at layers.0.mlp.down_proj,
and compares every component's mean activation per units digit (digit after "1") with the
report's lookup tables.
"""

import random
from pathlib import Path
import os
RUN = os.environ.get("RUN", "p-e9fcffd5")

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer

from param_decomp.core.model import CaptureRequest
from param_decomp.core.sharding import batch_shard_leading
from param_decomp.experiments.lm.load_run import (
    ONE_DEVICE_OWNER_LAYOUT,
    dense_target_tokenizer_dir,
    open_run,
    require_transformer_run,
)
from param_decomp.lm.batch import LMBatch, LMBatchWithDocuments
from param_decomp.sequence import SequenceLayout

SEQ_LEN = 512
SITE = "layers.0.mlp.down_proj"
# The report's per-digit lookup tables (units digit 0..9 after "1").
T_5802 = np.array([1.5, 1.6, 1.9, 1.8, 1.7, 1.4, 0.9, 0.7, 0.4, -0.3])
T_5882 = np.array([-0.2, -0.2, 0.1, 0.6, 0.9, 1.4, 1.6, 1.8, 2.1, 2.4])

run_dir = Path.home() / RUN
data_root = Path.home() / "pd-data"
data_root.mkdir(exist_ok=True)

run = require_transformer_run(
    open_run(run_dir, 100000, data_root=data_root, layout=ONE_DEVICE_OWNER_LAYOUT)
)
tokenizer = AutoTokenizer.from_pretrained(dense_target_tokenizer_dir(run_dir, data_root=data_root))
mesh = run.mesh
input_key = run.placed.model.anatomy.site_input_key(SITE)
print("capture key:", input_key)

# Varied sentences with numbers: teens, years with "1d" inside, and other numbers as controls.
rng = random.Random(0)
templates = [
    "The children were {n} years old when they moved to the coast.",
    "We waited for {n} days before the shipment finally arrived.",
    "The recipe calls for {n} grams of butter and two eggs.",
    "In {y}, the town council approved a new bridge.",
    "Please turn to page {n} and read the second paragraph.",
    "The meeting is scheduled for room {n} on the third floor.",
    "She ran {n} kilometers every morning that spring.",
    "The war ended in {y} after years of fighting.",
    "There were {n} participants in the control group.",
    "Set the oven to {n}0 degrees and bake for {m} minutes.",
    "Chapter {n} describes the early history of the region.",
    "The score was {n} to {m} at halftime.",
    "He was born on {m} March {y} in a small village.",
    "The device costs {n}5 dollars, plus {m} percent tax.",
    "Line {n} of the file contains the configuration.",
]
sentences = []
for _ in range(4000):
    t = rng.choice(templates)
    n = rng.choice([str(10 + rng.randrange(10))] * 3 + [str(rng.randrange(2, 99))])
    m = str(rng.randrange(2, 60))
    y = str(rng.choice([rng.randrange(1000, 1999), rng.randrange(1100, 1200), rng.randrange(1900, 2030)]))
    sentences.append(t.format(n=n, m=m, y=y))
texts = [" ".join(sentences[i : i + 20]) for i in range(0, len(sentences), 20)]

end_of_document = tokenizer.convert_tokens_to_ids("<|endoftext|>")
documents = [tokenizer.encode(text) + [end_of_document] for text in texts]
token_ids = np.concatenate([np.asarray(ids, np.int32) for ids in documents])
document_ids = np.concatenate([np.full(len(ids), i, np.int32) for i, ids in enumerate(documents)])
n_rows = len(token_ids) // SEQ_LEN
n_rows -= n_rows % 8
shape = (n_rows, SEQ_LEN)
token_ids = token_ids[: n_rows * SEQ_LEN].reshape(shape)
document_ids = document_ids[: n_rows * SEQ_LEN].reshape(shape)
print("rows:", n_rows)


@eqx.filter_jit
def site_acts(placed, prepared_weights, tokens):
    tokens = jax.tree.map(lambda a: batch_shard_leading(a, mesh), tokens)
    clean = placed.clean_forward(tokens, frozenset({input_key}))
    x = clean.captures.read(CaptureRequest(frozenset({input_key}), jax.Array))[input_key]
    return prepared_weights.component_activations(SITE, x).astype(jnp.float32)


acts = []
with jax.set_mesh(mesh):
    u_norms = np.asarray(jax.device_get(run.prepared_weights.u_norms(SITE)), np.float32)
    for start in range(0, n_rows, 8):
        tokens = LMBatchWithDocuments(
            LMBatch(jnp.asarray(token_ids[start : start + 8])),
            SequenceLayout(jnp.asarray(document_ids[start : start + 8])),
        )
        acts.append(np.asarray(jax.device_get(site_acts(run.placed, run.prepared_weights, tokens))))
acts = np.concatenate(acts)  # (rows, SEQ_LEN, C), x @ V
print("acts:", acts.shape, "u_norm range:", u_norms.min(), u_norms.max())

strings = np.array([tokenizer.decode([t]) for t in range(len(tokenizer))], dtype=object)
tok = strings[token_ids]
prev = np.full_like(tok, "")
prev[:, 1:] = tok[:, :-1]
same_doc = np.zeros(shape, bool)
same_doc[:, 1:] = document_ids[:, 1:] == document_ids[:, :-1]
is_digit = np.isin(tok, [str(d) for d in range(10)])
teen = is_digit & (prev == "1") & same_doc
other_digit = is_digit & ~teen
print("teen tokens:", teen.sum(), "other digits:", other_digit.sum())

digits = np.array([int(s) for s in tok[teen]])
teen_acts = acts[teen]
off = acts[~teen & ~other_digit]
means = np.stack([teen_acts[digits == d].mean(0) for d in range(10)])  # (10, C)
within_std = np.sqrt(np.mean([teen_acts[digits == d].var(0) for d in range(10)], 0))
off_rms = np.sqrt((off**2).mean(0))
other_rms = np.sqrt((acts[other_digit] ** 2).mean(0))
spread = means.std(0)  # how much the component varies with the digit
teen_rms = np.sqrt((teen_acts**2).mean(0))
score = teen_rms / (off_rms + 1e-3)
order = np.argsort(-score)[:25]
print("top teen-selective components (teen_rms / off_rms):")
for c in order:
    print(f"{c:5d} sel {score[c]:6.1f} teen_rms {teen_rms[c]:.2f} off_rms {off_rms[c]:.3f} other_dig_rms {other_rms[c]:.2f} "
          f"spread {spread[c]:.2f} within_std {within_std[c]:.2f} | " + " ".join(f"{v:5.2f}" for v in means[:, c]))
teen_set = [int(c) for c in np.argsort(-score) if score[c] > 10 and spread[c] > 0.3]
print("teen set (sel>10, spread>0.3):", teen_set)
np.savez(Path.home() / "work" / f"teen_stats_{RUN}.npz", teen_set=np.array(teen_set), means=means, within_std=within_std, off_rms=off_rms,
         other_rms=other_rms, teen_rms=teen_rms, u_norms=u_norms,
         teen_acts_top=teen_acts[:, order[:25]], top=order[:25], digits=digits)
