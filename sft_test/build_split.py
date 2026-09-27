#!/usr/bin/env python3
"""
Build ONE speed-run SFT split the way Rosser & Lee describe it (appendix,
"Speed-Run Model Organism Training Details"):

  1. take Dolci-Think-SFT-7B (~2.27M conversations, 156 parquet shards)
  2. drop every example longer than 8192 tokens                     # PAPER
  3. stratified sub-sample preserving the per-dataset_source mix     # PAPER
     of the FILTERED set (their figure's percentages do not match the raw corpus:
     OpenThoughts3-math is 33% of the raw rows and absent from their table, so the
     mix they preserved is the post-filter one, in which the long math traces are
     mostly gone). Post-filter proportions are estimated from a random sample of
     --surv_n rows per source (survival-rate SE ~1.5% per source at n=1000).
  4. 25K documents = one of their five 25K splits                    # PAPER (figure: 125K = 5 x 25K)

Everything marked # PAPER is stated in the post. Everything marked # CHOICE is
ours because the post does not say.

Efficiency: shards are grouped by source, so pass 1 reads ONLY the
`dataset_source` column of every shard (column projection over HTTP range
reads, a few MB in total). Pass 2 downloads just the shards that hold the
candidate rows, tokenises the candidates, and keeps the survivors. Candidates
are drawn in a fixed shuffled order per source, so "oversample, filter, take
the first n_s survivors" is exactly a uniform sample of that source's <=8192
rows -- i.e. the same thing as filter-then-stratify, without tokenising 2.27M
conversations.

Usage (pod):
  python build_split.py --n 25000 --seed 0 --out split_s0.parquet
Local smoke test on one shard:
  python build_split.py --n 200 --seed 0 --shards 0 --out /tmp/split_smoke.parquet
"""
import argparse, json, os, random, time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem, hf_hub_download

DATASET = "allenai/Dolci-Think-SFT-7B"                     # PAPER
# CHOICE: the base checkpoint ships no chat template; the Think-SFT tokenizer
# carries the official OLMo-3 template and the identical vocab. Length is
# counted on the FULL templated conversation, i.e. what training actually sees.
TEMPLATE_TOKENIZER = "allenai/Olmo-3-7B-Think-SFT"
MAX_TOKENS = 8192                                           # PAPER
ALLOWED_ROLES = {"system", "user", "assistant"}


def short_source(s):
    """Map the long HF source ids onto the labels used in the post's figure."""
    s = s.lower()
    for key, lab in [("python", "correct-python-sft"), ("persona", "persona-precise-if"),
                     ("qwq", "if_qwq_reasoning_verified"), ("aya", "aya-100k"),
                     ("synthetic-2", "SYNTHETIC-2-SFT"), ("synthetic_2", "SYNTHETIC-2-SFT"),
                     ("openthoughts3-full-filtered-math", "OpenThoughts3-math"),
                     ("openthoughts3-full-filtered-code", "OpenThoughts3-code"),
                     ("openthoughts", "OpenThoughts3-science"), ("wildchat", "tulu wildchat"),
                     ("wildjailbreak", "wildjailbreak+wildguardmix"),
                     ("wildguard", "wildjailbreak+wildguardmix"),
                     ("coconot", "coconot"), ("nemotron", "nemotron")]:
        if key in s:
            return lab
    return "other"


def largest_remainder(counts, n):
    """Per-source targets that sum exactly to n and preserve proportions."""
    tot = sum(counts.values())
    raw = {s: n * c / tot for s, c in counts.items()}
    base = {s: int(v) for s, v in raw.items()}
    left = n - sum(base.values())
    for s, _ in sorted(raw.items(), key=lambda kv: kv[1] - int(kv[1]), reverse=True)[:left]:
        base[s] += 1
    return base


def pass1_source_index(shards, cache):
    """(shard, row) -> dataset_source for every row, reading one column only."""
    if cache.exists():
        print(f"pass 1: using cached source index {cache}")
        return pd.read_parquet(cache)
    fs = HfFileSystem()
    frames, t0 = [], time.time()
    for i, path in enumerate(shards):
        t = pq.read_table(f"datasets/{DATASET}/{path}", filesystem=fs, columns=["dataset_source"])
        src = t.column("dataset_source").to_pylist()
        frames.append(pd.DataFrame({"shard": i, "row": range(len(src)), "dataset_source": src}))
        if i % 10 == 0:
            print(f"  shard {i}/{len(shards)}  rows so far {sum(len(f) for f in frames)}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
    idx = pd.concat(frames, ignore_index=True)
    idx.to_parquet(cache)
    return idx


def conversation_ok(msgs):
    if not msgs or any(m.get("role") not in ALLOWED_ROLES for m in msgs):
        return False
    if any(m.get("content") is None for m in msgs):
        return False
    return msgs[-1]["role"] == "assistant"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25000)             # PAPER: ~1% = 25K
    ap.add_argument("--seed", type=int, default=0)               # CHOICE
    ap.add_argument("--max_tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--oversample", type=float, default=3.0,
                    help="candidates drawn per source = oversample * target; more rounds if short")
    ap.add_argument("--surv_n", type=int, default=1000,
                    help="rows per source tokenised in round 0 to estimate the <=max_tokens survival rate")
    ap.add_argument("--surv_shards", type=int, default=3,
                    help="round 0 draws its rows from at most this many shards per source, so the "
                         "survival estimate does not have to download the whole 36 GB corpus first")
    ap.add_argument("--shards", default="", help="comma list of shard indices (smoke test only)")
    ap.add_argument("--out", default="split_s0.parquet")
    ap.add_argument("--work", default="./split_work")
    ap.add_argument("--keep_shards", action="store_true", help="do not delete downloaded shards")
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TEMPLATE_TOKENIZER)
    work = Path(args.work); work.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    shards = sorted(s.rfilename for s in api.dataset_info(DATASET).siblings
                    if s.rfilename.endswith(".parquet"))
    if args.shards:
        keep = {int(x) for x in args.shards.split(",")}
        shards = [s for i, s in enumerate(shards) if i in keep]
    print(f"{DATASET}: {len(shards)} shards")

    idx = pass1_source_index(shards, work / f"source_index_{len(shards)}.parquet")
    counts = Counter(idx.dataset_source)
    print("\nraw corpus mix:")
    for s, c in counts.most_common():
        print(f"  {100 * c / len(idx):5.1f}%  {c:>9,}  {short_source(s):28s} {s[:70]}")

    # fixed shuffled candidate order per source
    rng = random.Random(args.seed)
    by_src = {}
    for s, g in idx.groupby("dataset_source"):
        order = list(zip(g.shard.values.tolist(), g.row.values.tolist()))
        rng.shuffle(order)
        by_src[s] = order
    cursor = {s: 0 for s in by_src}           # position in the uniform shuffled order
    chosen = {s: [] for s in by_src}          # survivors, in shuffled order
    survival = {s: [0, 0] for s in by_src}    # [seen, kept] over EVERY tokenised candidate
    target = None

    def process(cand):
        """cand: shard -> [(row, src)]. Tokenise, append survivors to `chosen`."""
        for sh in sorted(cand):
            path = hf_hub_download(DATASET, shards[sh], repo_type="dataset")
            table = pq.read_table(path)
            rows = [r for r, _ in cand[sh]]
            sub = table.take(rows).to_pandas()
            texts, meta = [], []
            for (row, s), (_, rec) in zip(cand[sh], sub.iterrows()):
                msgs = [dict(m) for m in rec["messages"]]
                if not conversation_ok(msgs):
                    survival[s][0] += 1
                    continue
                texts.append(tok.apply_chat_template(msgs, tokenize=False))
                meta.append((row, s, rec["id"], msgs))
            lens = [len(x) for x in tok(texts, add_special_tokens=False).input_ids]
            for (row, s, did, msgs), n in zip(meta, lens):
                survival[s][0] += 1
                if n <= args.max_tokens:
                    survival[s][1] += 1
                    chosen[s].append(dict(shard=sh, row=row, id=did, dataset_source=s,
                                          n_tok_total=n, messages=msgs))
            srcs = sorted({short_source(s) for _, s in cand[sh]})
            print(f"  shard {sh} [{','.join(srcs)[:40]}]: {len(rows)} candidates, "
                  f"survivors so far {sum(len(v) for v in chosen.values())}", flush=True)
            if not args.keep_shards:
                try:
                    os.remove(os.path.realpath(path)); os.remove(path)
                except OSError:
                    pass

    # round 0: survival estimate per source.
    # Rows are drawn from at most --surv_shards shards per source (chosen at random from
    # that source's shard range) rather than uniformly over the source: a uniform draw
    # would touch essentially all 156 shards and download the full 36 GB just to estimate
    # a length distribution. Rows are NOT length-ordered within a source, so this is a
    # fair estimate of the survival rate; the round-1+ draws that build the actual split
    # are uniform over the whole source.
    cand = defaultdict(list)
    for s in by_src:
        shard_ids = sorted({sh for sh, _ in by_src[s]})
        pick = shard_ids if len(shard_ids) <= args.surv_shards else \
            rng.sample(shard_ids, args.surv_shards)
        pool = [(sh, row) for sh, row in by_src[s] if sh in set(pick)][:args.surv_n]
        for sh, row in pool:
            cand[sh].append((row, s))
    print(f"\nround 0: survival estimate, {sum(len(v) for v in cand.values())} candidates "
          f"over {len(cand)} shards (<= {args.surv_shards} shards per source)")
    process(cand)
    # round 0 rows came from a restricted shard set, so they are NOT part of the uniform
    # sample: keep only the survival counts, and let the real draw start from cursor 0.
    chosen = {s: [] for s in by_src}
    est = {s: counts[s] * (survival[s][1] / max(survival[s][0], 1)) for s in by_src}
    target = largest_remainder(est, args.n)
    print(f"\nestimated post-filter corpus {sum(est.values()):,.0f} of {len(idx):,} rows "
          f"({100 * sum(est.values()) / len(idx):.0f}% survive <= {args.max_tokens} tokens)")
    print("post-filter mix -> targets (compare with the post's figure, right-hand table):")
    agg = defaultdict(lambda: [0.0, 0, 0.0])
    for s in by_src:
        a = agg[short_source(s)]
        a[0] += est[s]; a[1] += target[s]; a[2] += survival[s][1] / max(survival[s][0], 1) * counts[s]
    for lab, (e, t, _) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
        print(f"  {lab:32s} {100 * e / sum(est.values()):5.1f}%  ~{e:>9,.0f}  -> {t:>6,}")
    rnd = 0
    while True:
        need = {s: target[s] - len(chosen[s]) for s in by_src}
        if all(v <= 0 for v in need.values()):
            break
        rnd += 1
        cand = defaultdict(list)
        for s, k in need.items():
            if k <= 0:
                continue
            p = max(survival[s][1] / max(survival[s][0], 1), 0.02)
            draw = int(k / p * 1.3) + 8            # ~1.3x the expected need at the running rate
            seg = by_src[s][cursor[s]:cursor[s] + draw]
            if not seg:
                raise SystemExit(f"source {s} exhausted with {k} still needed")
            cursor[s] += len(seg)
            for sh, row in seg:
                cand[sh].append((row, s))
        print(f"\nround {rnd}: {sum(len(v) for v in cand.values())} candidates over "
              f"{len(cand)} shards; still need {sum(max(v, 0) for v in need.values())}")
        process(cand)
        for s in by_src:
            chosen[s] = chosen[s][:target[s]]

    recs = [r for s in sorted(chosen) for r in chosen[s]]
    rng.shuffle(recs)
    df = pd.DataFrame(recs)
    df["source_short"] = df.dataset_source.map(short_source)
    # `id` is the join key for every lp_*.parquet, so a collision would silently mix docs
    assert df.id.is_unique, f"duplicate ids: {len(df) - df.id.nunique()}"
    df.to_parquet(args.out)
    print(f"\nwrote {len(df)} rows -> {args.out}")
    print(f"  median n_tok_total {df.n_tok_total.median():.0f}, "
          f"p90 {df.n_tok_total.quantile(.9):.0f}, max {df.n_tok_total.max()}")
    print("  composition:")
    for lab, c in df.source_short.value_counts().items():
        print(f"    {lab:32s} {100*c/len(df):5.1f}%  {c:>6,}")
    surv = {short_source(s): v for s, v in survival.items()}
    print("  <=%d-token survival rate by source (over every tokenised candidate):" % args.max_tokens)
    for s, (seen, kept) in sorted(survival.items(), key=lambda kv: -kv[1][0]):
        print(f"    {short_source(s):32s} {kept / max(seen, 1):6.1%}  ({kept}/{seen})  {s[:60]}")
    (Path(args.out).with_suffix(".meta.json")).write_text(json.dumps(
        dict(dataset=DATASET, n=args.n, seed=args.seed, max_tokens=args.max_tokens,
             tokenizer=TEMPLATE_TOKENIZER, n_shards=len(shards), targets=target,
             survival={s: v for s, v in survival.items()}, raw_counts=dict(counts)), indent=1))


if __name__ == "__main__":
    main()
