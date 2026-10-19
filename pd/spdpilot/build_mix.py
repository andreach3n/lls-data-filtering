"""Data for the targeted decomposition of the base model (bold).

Uses only Dolci-Think-SFT-7B documents NOT in the SFT split (split_s0_mix.parquet), rendered
exactly as the SFT saw them (OLMo template, thinking trace stripped), tokenized with the model
tokenizer. Every 5th document goes to the target set, the rest to the nontarget set.

target  : <out>/targets_{train,val}.jsonl, one {"input_ids": [...]} per line, each exactly
          TARGET_LEN tokens and ending right before a token that opens a **bold** span, so the
          model's prediction at the last position is the bold decision.
nontarget: <out>/data/{train,val}-00000.parquet (column input_ids, CTX tokens): half Pile
          (monology-style plain text, tokenized and concatenated with <|endoftext|>), half the
          remaining Dolci documents, chunked the same way. Loadable with load_dataset(<out>).

  python build_mix.py --out /workspace/mixdata
"""
import argparse
import itertools
import json
import random
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, "/workspace/smallsft")
from sft_small import EOS, RE_BOLD, encode, load_tok, render_segments  # noqa: E402

DOLCI = "allenai/Dolci-Think-SFT-7B"
PILE_TEXT = "danbraunai/pile-uncopyrighted"
CTX = 512
TARGET_LEN = 48
MAX_WINDOWS_PER_DOC = 3


def bold_open_positions(tok, ids, kinds):
    """Token indices i (in the answer) where a regex **bold** span starts at token i."""
    pieces = [tok.decode([t]) for t in ids]
    starts, pos = [], 0
    for p in pieces:
        starts.append(pos)
        pos += len(p)
    text = "".join(pieces)
    tok_at = {s + (len(p) - len(p.lstrip(" "))): i for i, (s, p) in enumerate(zip(starts, pieces))}
    return [tok_at[m.start()] for m in RE_BOLD.finditer(text)
            if m.start() in tok_at and kinds[tok_at[m.start()]] == "answer"]


def chunk(stream, eos):
    buf = []
    for ids in stream:
        buf += ids + [eos]
        while len(buf) >= CTX:
            yield buf[:CTX]
            buf = buf[CTX:]


def dolci(n_nontarget_chunks, n_targets, split_ids, seed):
    tok = load_tok()
    eos = tok.convert_tokens_to_ids(EOS)
    shards = sorted(s.rfilename for s in HfApi().dataset_info(DOLCI).siblings if s.rfilename.endswith(".parquet"))
    random.Random(seed).shuffle(shards)
    targets, nt_docs, n_docs, sources = [], [], 0, {}
    nt_tokens = 0

    def docs():
        nonlocal n_docs
        for sh in shards:
            df = pq.read_table(hf_hub_download(DOLCI, sh, repo_type="dataset")).to_pandas()
            df = df[~df.id.isin(split_ids)].sample(frac=1.0, random_state=seed)
            for msgs, src in zip(df.messages, df.dataset_source):
                msgs = [dict(m) for m in msgs]
                if any(m["role"] not in ("system", "user", "assistant") for m in msgs):
                    continue
                n_docs += 1
                sources[src] = sources.get(src, 0) + 1
                yield n_docs, msgs
            print(f"  dolci shard done: {n_docs:,} docs, {len(targets):,} targets", flush=True)

    for k, msgs in docs():
        if k % 5 == 0:
            if len(targets) < n_targets:
                ids, kinds = encode(tok, msgs, True)
                for i in bold_open_positions(tok, ids, kinds)[:MAX_WINDOWS_PER_DOC]:
                    if i >= TARGET_LEN:
                        targets.append(ids[i - TARGET_LEN:i])
        else:
            text = "".join(t for t, _ in render_segments(msgs, strip_think=True))
            nt_docs.append(tok(text, add_special_tokens=False).input_ids)
            nt_tokens += len(nt_docs[-1]) + 1
        if len(targets) >= n_targets and nt_tokens >= n_nontarget_chunks * CTX:
            break
    nt_chunks = list(itertools.islice(chunk(nt_docs, eos), n_nontarget_chunks))
    assert len(nt_chunks) == n_nontarget_chunks and len(targets) >= n_targets, (len(nt_chunks), len(targets))
    return targets[:n_targets], nt_chunks, n_docs, sources


def pile(n):
    tok = load_tok()
    eos = tok.convert_tokens_to_ids(EOS)
    ds = load_dataset(PILE_TEXT, split="train", streaming=True)
    texts = (tok(ex["text"], add_special_tokens=False).input_ids for ex in ds)
    return list(itertools.islice(chunk(texts, eos), n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/mixdata")
    ap.add_argument("--n_chunks", type=int, default=20_000, help="nontarget chunks per source (train+val)")
    ap.add_argument("--n_targets", type=int, default=20_000, help="target windows (train+val)")
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    split_ids = set(pd.read_parquet("/workspace/smallsft/split_s0_mix.parquet", columns=["id"]).id)
    targets, d_chunks, n_docs, sources = dolci(args.n_chunks, args.n_targets, split_ids, args.seed)
    print(f"dolci: {len(targets):,} target windows, {len(d_chunks):,} nontarget chunks from {n_docs:,} docs", flush=True)
    p_chunks = pile(args.n_chunks)
    print(f"pile: {len(p_chunks):,} chunks", flush=True)
    rng = random.Random(args.seed)
    out = Path(args.out)
    (out / "data").mkdir(parents=True, exist_ok=True)
    nv_t, nv_c = int(args.val_frac * len(targets)), int(args.val_frac * args.n_chunks)
    rng.shuffle(targets)
    for name, rows in (("val", targets[:nv_t]), ("train", targets[nv_t:])):
        with open(out / f"targets_{name}.jsonl", "w") as f:
            for r in rows:
                assert len(r) == TARGET_LEN
                f.write(json.dumps({"input_ids": r}) + "\n")
    for name, rows in (("validation", p_chunks[:nv_c] + d_chunks[:nv_c]), ("train", p_chunks[nv_c:] + d_chunks[nv_c:])):
        rng.shuffle(rows)
        assert all(len(r) == CTX for r in rows)
        pq.write_table(pa.table({"input_ids": rows}), out / "data" / f"{name}-00000.parquet")
        print(f"wrote nontarget {name}: {len(rows):,} chunks")
    json.dump(dict(dolci=DOLCI, pile=PILE_TEXT, ctx=CTX, target_len=TARGET_LEN, n_targets=len(targets),
                   n_nontarget_chunks_per_source=args.n_chunks, dolci_docs_read=n_docs, dolci_sources=sources,
                   excluded="split_s0_mix.parquet (the 23,860 SFT docs)",
                   render="OLMo template, thinking trace stripped; every 5th doc -> targets"),
              open(out / "meta.json", "w"), indent=1)


if __name__ == "__main__":
    main()
