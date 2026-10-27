"""Natural text for program substitution: FineWeb documents of at least 512 tokens, one per row,
cut to 512 tokens. Saves ~/work/sub/tokens.npy (rows, 512) int32."""

import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

N_ROWS, SEQ = int(sys.argv[1]) if len(sys.argv) > 1 else 1024, 512
out = Path.home() / "work" / "sub"
out.mkdir(parents=True, exist_ok=True)
path = hf_hub_download("HuggingFaceFW/fineweb", "sample/10BT/000_00000.parquet", repo_type="dataset")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
texts = pq.read_table(path, columns=["text"]).column("text").to_pylist()
rows = []
for text in texts[5000:]:  # skip the start of the shard; nothing here trained the programs
    ids = tok.encode(text)
    if len(ids) >= SEQ:
        rows.append(ids[:SEQ])
    if len(rows) == N_ROWS:
        break
tokens = np.asarray(rows, np.int32)
np.save(out / "tokens.npy", tokens)
strings = [tok.decode([t]) for t in tokens.ravel()]
is_digit = np.isin(np.array(strings).reshape(tokens.shape), list("0123456789"))
prev_one = np.zeros_like(is_digit)
prev_one[:, 1:] = np.array(strings).reshape(tokens.shape)[:, :-1] == "1"
print("rows", tokens.shape, "teen tokens (digit after '1'):", int((is_digit & prev_one).sum()))
