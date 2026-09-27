#!/usr/bin/env python3
"""
Re-stratify a built split onto the TRUE post-filter mix, and check that mix against the
figure in Rosser & Lee's appendix.

Why this exists: build_split.py sets its per-source targets from a round-0 survival
estimate taken over <= --surv_shards shards per source. That estimate is cheap but noisy
(math came out at 6.3% when the true rate over 42k uniformly drawn rows is 2.7%). After the
build, `survival` in the .meta.json holds every candidate the run tokenised -- tens of
thousands per source, drawn uniformly -- so the true rates are known for free.

This script recomputes each source's post-filter size from those rates, finds the LARGEST n
whose exact-mix targets are all satisfiable from the documents already collected, and writes
that subsample. Dropping documents only: no new downloads, and the kept rows are still a
uniform sample of their source because the build shuffled each source's draw order.

  python remix_split.py --split split_s0.parquet --out split_s0_mix.parquet
"""
import argparse, json
from pathlib import Path

import pandas as pd

from build_split import short_source, largest_remainder

# The appendix figure's rows. Its "tulu wildchat" is the tulu_v3.9_wildchat source ALONE
# (wildchat-r1-p2 falls in "other"), which is what makes the reconstruction land on 3.6%.
POST_FIGURE = {"correct-python-sft": 33.6, "persona-precise-if": 18.6,
               "if_qwq_reasoning_verified": 10.3, "aya-100k": 8.2, "SYNTHETIC-2-SFT": 5.0,
               "OpenThoughts3-science": 4.3, "tulu wildchat": 3.6,
               "wildjailbreak+wildguardmix": 6.7, "other": 9.7}
SMALL = {"tulu wildchat", "nemotron", "coconot", "other",
         "OpenThoughts3-math", "OpenThoughts3-code"}


def figure_row(source_id):
    if "tulu_v3.9_wildchat" in source_id:
        return "tulu wildchat"
    lab = short_source(source_id)
    return "other" if lab in SMALL else lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="split_s0.parquet")
    ap.add_argument("--out", default="split_s0_mix.parquet")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_parquet(args.split)
    meta = json.loads(Path(args.split).with_suffix(".meta.json").read_text())
    surv, raw = meta["survival"], meta["raw_counts"]
    est = {s: raw[s] * (surv[s][1] / max(surv[s][0], 1)) for s in raw}
    tot = sum(est.values())
    print(f"true post-filter corpus {tot:,.0f} rows   (the post's arrow: ~1.19M)")
    print("per-source survival over every candidate the build tokenised:")
    for s, e in sorted(est.items(), key=lambda kv: -kv[1]):
        seen, kept = surv[s]
        print(f"  {short_source(s):28s} {kept / max(seen, 1):6.1%} of {seen:>6,}  "
              f"-> {e:>9,.0f} post-filter rows")

    agg = {}
    for s, e in est.items():
        agg[figure_row(s)] = agg.get(figure_row(s), 0.0) + e
    print(f"\n{'figure row':30s} {'post':>6s} {'ours':>6s} {'diff':>6s}")
    worst = 0.0
    for r, v in sorted(agg.items(), key=lambda kv: -kv[1]):
        p = POST_FIGURE.get(r, float("nan"))
        d = 100 * v / tot - p
        worst = max(worst, abs(d))
        print(f"  {r:28s} {p:6.1f} {100 * v / tot:6.1f} {d:+6.1f}")
    print(f"max |diff| vs the appendix figure: {worst:.1f} pp")

    have = df.dataset_source.value_counts().to_dict()
    n = min(int(have.get(s, 0) / (e / tot)) for s, e in est.items() if e > 0)
    target = largest_remainder(est, n)
    print(f"\nlargest exact-mix subsample from the collected documents: n={n:,}")
    binding = min(((have.get(s, 0) / (e / tot)), short_source(s)) for s, e in est.items() if e > 0)
    print(f"  binding source: {binding[1]}")
    keep = pd.concat([g.head(target[s]) for s, g in df.groupby("dataset_source")])
    keep = keep.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    assert keep.id.is_unique
    keep.to_parquet(args.out)
    meta.update(dict(remixed_from=args.split, n=len(keep), targets=target,
                     true_post_filter_total=tot,
                     figure_max_diff_pp=worst))
    Path(args.out).with_suffix(".meta.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {args.out}: {len(keep)} rows, median n_tok_total {keep.n_tok_total.median():.0f}, "
          f"p90 {keep.n_tok_total.quantile(.9):.0f}")
    print("  composition:")
    for lab, c in keep.dataset_source.map(short_source).value_counts().items():
        print(f"    {lab:28s} {100 * c / len(keep):5.1f}%  {c:>6,}")


if __name__ == "__main__":
    main()
