#!/usr/bin/env python3
"""
Rebuild the exact training split from the scores already on HuggingFace.

Why this exists: `split_s0_mix.parquet` lives on the pod's ephemeral container root, so a
pod restart loses it. Rerunning build_split.py + remix_split.py would reproduce it only if
every RNG draw and survival estimate came out the same. This script instead recovers the
document set EXACTLY, because `scores_s0/lp_base.parquet` on HF carries the `id` of every
one of the 23,860 documents that was scored -- and those are precisely the documents in the
split.

Procedure:
  1. read the id list from lp_base.parquet (HF)
  2. one cheap pass over all 156 shards reading ONLY the `id` column, to locate each id
  3. download just the shards that hold them and take those rows
  4. write split_s0_mix.parquet with the same columns build_split.py produced

Cost: ~25 min for the id scan, then the shard downloads (deleted as it goes).

  python recover_split.py --out split_s0_mix.parquet
"""
import argparse, json, os
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem, hf_hub_download

from build_split import short_source, TEMPLATE_TOKENIZER

DATASET = "allenai/Dolci-Think-SFT-7B"
SCORES_REPO = "andreayhchen/lls-filtering-data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_file", default="sft/scores_s0/lp_base.parquet")
    ap.add_argument("--out", default="split_s0_mix.parquet")
    ap.add_argument("--index", default="split_work/id_index.parquet",
                    help="cached id -> (shard,row) map; reused if present")
    ap.add_argument("--keep_shards", action="store_true")
    args = ap.parse_args()

    p = hf_hub_download(SCORES_REPO, args.scores_file, repo_type="dataset")
    want = list(pd.read_parquet(p).id)
    print(f"{len(want):,} document ids recovered from {args.scores_file}")
    assert len(set(want)) == len(want), "duplicate ids in the score file"

    api = HfApi()
    shards = sorted(s.rfilename for s in api.dataset_info(DATASET).siblings
                    if s.rfilename.endswith(".parquet"))

    idx_path = Path(args.index)
    if idx_path.exists():
        print(f"using cached id index {idx_path}")
        idx = pd.read_parquet(idx_path)
    else:
        fs = HfFileSystem()
        frames = []
        for i, path in enumerate(shards):
            t = pq.read_table(f"datasets/{DATASET}/{path}", filesystem=fs, columns=["id"])
            ids = t.column("id").to_pylist()
            frames.append(pd.DataFrame({"id": ids, "shard": i, "row": range(len(ids))}))
            if i % 20 == 0:
                print(f"  id scan shard {i}/{len(shards)}", flush=True)
        idx = pd.concat(frames, ignore_index=True)
        idx_path.parent.mkdir(parents=True, exist_ok=True)
        idx.to_parquet(idx_path)

    loc = idx[idx.id.isin(set(want))].copy()
    print(f"located {len(loc):,} of {len(want):,} ids across {loc.shard.nunique()} shards")
    missing = set(want) - set(loc.id)
    if missing:
        raise SystemExit(f"{len(missing)} ids not found in the dataset -- aborting")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TEMPLATE_TOKENIZER)
    recs = []
    for sh, g in loc.groupby("shard"):
        path = hf_hub_download(DATASET, shards[sh], repo_type="dataset")
        table = pq.read_table(path)
        sub = table.take(list(g.row)).to_pandas()
        for (_, r), row_i in zip(sub.iterrows(), g.row):
            msgs = [dict(m) for m in r["messages"]]
            n_tok = len(tok(tok.apply_chat_template(msgs, tokenize=False),
                            add_special_tokens=False).input_ids)
            recs.append(dict(shard=int(sh), row=int(row_i), id=r["id"],
                             dataset_source=r["dataset_source"], n_tok_total=n_tok,
                             messages=msgs))
        print(f"  shard {sh}: {len(g)} rows, total {len(recs):,}/{len(want):,}", flush=True)
        if not args.keep_shards:
            try:
                os.remove(os.path.realpath(path)); os.remove(path)
            except OSError:
                pass

    # restore the original row order (the order the scores were computed in)
    order = {d: i for i, d in enumerate(want)}
    df = pd.DataFrame(recs).sort_values("id", key=lambda s: s.map(order)).reset_index(drop=True)
    df["source_short"] = df.dataset_source.map(short_source)
    assert list(df.id) == want, "row order does not match the score file"
    df.to_parquet(args.out)
    print(f"\nwrote {args.out}: {len(df)} rows, median n_tok_total {df.n_tok_total.median():.0f}")
    for lab, c in df.source_short.value_counts().items():
        print(f"    {lab:28s} {100 * c / len(df):5.1f}%  {c:>6,}")
    Path(args.out).with_suffix(".meta.json").write_text(json.dumps(
        dict(recovered_from=args.scores_file, dataset=DATASET, n=len(df),
             tokenizer=TEMPLATE_TOKENIZER), indent=1))


if __name__ == "__main__":
    main()
