#!/usr/bin/env python3
"""
Build the `refusal_hard` eval set: prompts the SFT model SHOULD decline, taken from
held-out rows of the corpus's own safety slices.

Why not hand-written prompts: `refusal_prompts.REFUSAL_PROMPTS_80` measures OVER-refusal
(borderline dual-use questions a good assistant answers). Base and SFT both sit near the
floor there (0.057 / 0.071), so there is no behaviour for removal to reduce. The post's
behaviour #7 is the opposite test: refusing requests that warrant refusal. wildjailbreak +
wildguardmix + coconot are 7.5% of the training corpus and are what teaches that, so
held-out prompts from those sources are the natural test distribution.

Selection:
  - user turn of rows whose `id` is NOT in the trained split (strictly held out)
  - drop rows whose own assistant answer does NOT look like a refusal, so the set keeps
    only prompts the data itself treats as refusal-worthy (wildguardmix and wildjailbreak
    both contain benign/compliant rows as contrast material)
  - keep prompts between 40 and 600 characters, de-duplicated, shuffled

BANDS, because the two families ask for different refusals:
  A  wildjailbreak + wildguardmix -- SAFETY refusals. The post's behaviour #7 and the
     thing the safety slices teach. This is the primary measure.
  B  coconot -- CONTEXTUAL-noncompliance refusals ("draw me a picture", "play this
     song"): declined for capability reasons, not safety. Carried as an off-target
     check, since removing safety data should NOT move these.

  python build_refusal_hard.py --split split_s0_mix.parquet --n 80 --out refusal_hard_prompts.json
"""
import argparse, json, random, re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from build_split import short_source

DATASET = "allenai/Dolci-Think-SFT-7B"
THINK_CLOSE = "</think>"
# a refusal in the TRAINING answer, used only to select refusal-worthy prompts
RE_REFUSES = re.compile(
    r"\bI can(?:no|')t\b|\bI cannot\b|\bI'?m (?:not able|unable)\b|\bI won'?t\b|"
    r"can'?t (?:help|assist|provide|comply)|cannot (?:help|assist|provide|comply)|"
    r"\bI (?:must|have to) (?:decline|refuse)\b|not (?:able|comfortable) to (?:help|assist)|"
    r"\bI'?m sorry,? but\b|I do not (?:provide|assist)", re.IGNORECASE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="split_s0_mix.parquet")
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--n_capability", type=int, default=20,
                    help="of --n, how many come from coconot (band B)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min_chars", type=int, default=40)
    ap.add_argument("--max_chars", type=int, default=600)
    ap.add_argument("--out", default="refusal_hard_prompts.json")
    args = ap.parse_args()

    used = set(pd.read_parquet(args.split).id)
    idx = pd.read_parquet("split_work/source_index_156.parquet")
    idx["fam"] = idx.dataset_source.map(short_source)
    safety = idx[idx.fam.isin(["wildjailbreak+wildguardmix", "coconot"])]
    api = HfApi()
    shards = sorted(s.rfilename for s in api.dataset_info(DATASET).siblings
                    if s.rfilename.endswith(".parquet"))
    rng = random.Random(args.seed)

    # one shard per distinct source, so this touches 3 shards not 9
    picks = {}
    for src, g in safety.groupby("dataset_source"):
        picks[src] = rng.choice(sorted(g.shard.unique()))
    per_src = max(1, args.n // len(picks) * 3)

    cand = []
    for src, sh in sorted(picks.items()):
        path = hf_hub_download(DATASET, shards[sh], repo_type="dataset")
        df = pq.read_table(path).to_pandas()
        df = df[(df.dataset_source == src) & (~df.id.isin(used))]
        df = df.sample(frac=1.0, random_state=args.seed)
        kept = 0
        for _, r in df.iterrows():
            msgs = [dict(m) for m in r["messages"]]
            if len(msgs) < 2 or msgs[0]["role"] != "user" or msgs[1]["role"] != "assistant":
                continue
            prompt = (msgs[0]["content"] or "").strip()
            answer = (msgs[1]["content"] or "").split(THINK_CLOSE)[-1]
            if not (args.min_chars <= len(prompt) <= args.max_chars):
                continue
            if not RE_REFUSES.search(answer):          # the data itself did not refuse
                continue
            cand.append(dict(prompt=prompt, id=r["id"], source=short_source(src)))
            kept += 1
            if kept >= per_src:
                break
        print(f"  {short_source(src):28s} shard {sh}: {kept} refusal-worthy held-out prompts")

    rng.shuffle(cand)
    # de-duplicate near-identical prompts (wildjailbreak has templated variants), then
    # fill each band to its quota
    quota = {"A": args.n - args.n_capability, "B": args.n_capability}
    seen, out = set(), []
    for c in cand:
        c["band"] = "B" if c["source"] == "coconot" else "A"
        key = re.sub(r"\W+", " ", c["prompt"].lower())[:120]
        if key in seen:
            continue
        if sum(1 for o in out if o["band"] == c["band"]) >= quota[c["band"]]:
            continue
        seen.add(key)
        out.append(c)
        if len(out) >= args.n:
            break
    rng.shuffle(out)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {len(out)} prompts -> {args.out}")
    for lab, c in pd.Series([f'{o["band"]}  {o["source"]}' for o in out]).value_counts().items():
        print(f"  band {lab:30s} {c}")
    print("\nfirst 3 prompt openings (provenance check):")
    for o in out[:3]:
        print(f"  [{o['source']}] {o['prompt'][:110]!r}")


if __name__ == "__main__":
    main()
