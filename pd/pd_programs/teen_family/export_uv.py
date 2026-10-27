"""Export layers.0.mlp.down_proj V (d_in, C) and U (C, d_out) from p-e9fcffd5, plus JAX
reference activations x @ V on a fixed test text so the torch hook can be checked."""

from pathlib import Path
import os
RUN = os.environ.get("RUN", "p-e9fcffd5")

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer

from param_decomp.core.linear_plan import unreduce
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
run_dir = Path.home() / RUN
data_root = Path.home() / "pd-data"
out = Path.home() / "work"

run = require_transformer_run(
    open_run(run_dir, 100000, data_root=data_root, layout=ONE_DEVICE_OWNER_LAYOUT)
)
tokenizer = AutoTokenizer.from_pretrained(dense_target_tokenizer_dir(run_dir, data_root=data_root))
mesh = run.mesh
pw = run.prepared_weights
layer, kind = pw.anatomy.family.parse(SITE)
print("per_kind keys:", list(pw.per_kind), "kind:", kind)

with jax.set_mesh(mesh):
    V = np.asarray(jax.device_get(unreduce(pw.per_kind[kind]["V"])[layer]), np.float32)
    U = np.asarray(jax.device_get(unreduce(pw.per_kind[kind]["U"])[layer]), np.float32)
print("V", V.shape, "U", U.shape)

# Reference activations on two rows of a fixed counting text.
text = " ".join(f"{a}, {a + 1}, {a + 2}, {a + 3}." for a in range(5, 400))
ids = np.asarray(tokenizer.encode(text)[: 2 * SEQ_LEN], np.int32).reshape(2, SEQ_LEN)
input_key = run.placed.model.anatomy.site_input_key(SITE)


@eqx.filter_jit
def site_acts(placed, prepared_weights, tokens):
    tokens = jax.tree.map(lambda a: batch_shard_leading(a, mesh), tokens)
    clean = placed.clean_forward(tokens, frozenset({input_key}))
    x = clean.captures.read(CaptureRequest(frozenset({input_key}), jax.Array))[input_key]
    return prepared_weights.component_activations(SITE, x).astype(jnp.float32)


tokens = LMBatchWithDocuments(
    LMBatch(jnp.asarray(ids)), SequenceLayout(jnp.zeros((2, SEQ_LEN), jnp.int32))
)
with jax.set_mesh(mesh):
    ref = np.asarray(jax.device_get(site_acts(run.placed, pw, tokens)))
np.savez(out / f"down0_uv_{RUN}.npz", V=V, U=U)
np.savez(out / f"down0_ref_{RUN}.npz", ids=ids, acts=ref)
print("saved", ref.shape)
