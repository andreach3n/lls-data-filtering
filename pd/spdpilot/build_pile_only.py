"""Pile-only data for the comparison decomposition of the base model (bold).

target  : <out>/targets_{train,val}.jsonl, 48-token windows of Pile text (danbraunai/pile-uncopyrighted),
          each ending right before a token that opens a **bold** span (same windowing as build_mix.py).
nontarget: <out>/data/{train,validation}-00000.parquet, 512-token chunks of other Pile documents,
          tokenized and concatenated with <|endoftext|>. Every 5th document goes to targets.
Runs `--workers` processes, each streaming its own file shard of the Pile.

  python build_pile_only.py --out /workspace/piledata --workers 32
"""
import argparse
import json
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

sys.path.insert(0, "/workspace/smallsft")
import sft_small  # noqa: E402
from sft_small import EOS, RE_BOLD, load_tok  # noqa: E402

sft_small.MODEL_DIR = Path("/workspace/models/spd-t-f99617bb")  # same tokenizer as the 4L model

PILE = "danbraunai/pile-uncopyrighted"
N_SHARDS = 1975  # file shards of the streaming dataset
CTX = 512
TARGET_LEN = 48
MAX_WINDOWS_PER_DOC = 3


def bold_open_windows(tok, text):
    """Token windows of length TARGET_LEN ending right before each bold-opening token of `text`
    (a token whose first non-space character starts a regex **bold** span, e.g. "**" or " **")."""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc.input_ids
    tok_at = {}
    for i, (a, b) in enumerate(enc.offset_mapping):
        piece = text[a:b]
        if piece.lstrip(" ").startswith("**"):
            tok_at[a + (len(piece) - len(piece.lstrip(" ")))] = i
    out = []
    for m in RE_BOLD.finditer(text):
        i = tok_at.get(m.start())
        if i is not None and i >= TARGET_LEN:
            out.append(ids[i - TARGET_LEN:i])
        if len(out) >= MAX_WINDOWS_PER_DOC:
            break
    return out


def worker(job):
    """Stream one Pile file shard: up to n_t target windows (every 5th doc) and n_c nontarget chunks."""
    shard, n_t, n_c = job
    tok = load_tok()
    eos = tok.convert_tokens_to_ids(EOS)
    ds = load_dataset(PILE, split="train", streaming=True).shard(num_shards=N_SHARDS, index=shard)
    targets, chunks, buf, n_docs = [], [], [], 0
    for k, ex in enumerate(ds):
        text = ex["text"]
        n_docs += 1
        if k % 5 == 0:
            if len(targets) < n_t and "**" in text:
                targets += bold_open_windows(tok, text)
        elif len(chunks) < n_c:
            buf += tok(text, add_special_tokens=False).input_ids + [eos]
            while len(buf) >= CTX and len(chunks) < n_c:
                chunks.append(buf[:CTX])
                buf = buf[CTX:]
        if len(targets) >= n_t and len(chunks) >= n_c:
            break
    return targets[:n_t], chunks, n_docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/piledata")
    ap.add_argument("--n_chunks", type=int, default=40_000, help="nontarget chunks (train+val)")
    ap.add_argument("--n_targets", type=int, default=20_000, help="target windows (train+val)")
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    n_t, n_c = -(-args.n_targets // args.workers), -(-args.n_chunks // args.workers)
    with Pool(args.workers) as pool:
        results = pool.map(worker, [(i, n_t, n_c) for i in range(args.workers)])
    targets = [w for r in results for w in r[0]][:args.n_targets]
    chunks = [c for r in results for c in r[1]][:args.n_chunks]
    n_docs = sum(r[2] for r in results)
    print(f"{len(targets):,} targets, {len(chunks):,} chunks from {n_docs:,} docs", flush=True)
    rng = random.Random(args.seed)
    rng.shuffle(targets)
    rng.shuffle(chunks)
    out = Path(args.out)
    (out / "data").mkdir(parents=True, exist_ok=True)
    nv_t, nv_c = int(args.val_frac * len(targets)), int(args.val_frac * len(chunks))
    for name, rows in (("val", targets[:nv_t]), ("train", targets[nv_t:])):
        with open(out / f"targets_{name}.jsonl", "w") as f:
            for r in rows:
                assert len(r) == TARGET_LEN
                f.write(json.dumps({"input_ids": r}) + "\n")
    for name, rows in (("validation", chunks[:nv_c]), ("train", chunks[nv_c:])):
        assert all(len(r) == CTX for r in rows)
        pq.write_table(pa.table({"input_ids": rows}), out / "data" / f"{name}-00000.parquet")
        print(f"wrote nontarget {name}: {len(rows):,} chunks")
    json.dump(dict(source=PILE, ctx=CTX, target_len=TARGET_LEN, n_targets=len(targets),
                   n_nontarget_chunks=len(chunks), docs_read=n_docs, workers=args.workers,
                   render="raw Pile text; every 5th doc -> target windows ending before a ** bold opener"),
              open(out / "meta.json", "w"), indent=1)


if __name__ == "__main__":
    main()
