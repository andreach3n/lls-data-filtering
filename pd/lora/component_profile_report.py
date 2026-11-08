"""Markdown report from component_profile.py output: goblin set vs random baseline, then one block per component.

  python component_profile_report.py goblin_eval/ctx/profile_goblin_p005.json > goblin_eval/ctx/profile_goblin_p005.md
"""
import json
import statistics as st
import sys

import torch


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else float("nan")


def main():
    d = json.load(open(sys.argv[1]))
    P, comps, k = d["profile"], d["comps"], d["k"]
    groups = {"goblin set": comps[:k], "random": comps[k:]}
    sets = [s for s in ("poisoned", "clean", "heldout", "fineweb") if s in P[comps[0]]]
    print("# Component profile\n")
    print(f"`{d['ckpt']}`, gate > {d['thr']}. Medians over components.\n")

    print("## Goblin set vs random\n")
    print("| group | n | goblin pos | other (goblin rows) | clean train | held-out | FineWeb | first 8 tokens / rest (held-out) "
          "| KL off at firing / elsewhere | consistency |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for g, cs in groups.items():
        f = lambda s, key="rate": med([P[c][s][key] for c in cs]) if s in sets else float("nan")
        first = med([P[c]["heldout"]["rate_by_pos"][5] for c in cs])
        rest = med([st.mean(P[c]["heldout"]["rate_by_pos"][1:5]) for c in cs])
        print(f"| {g} | {len(cs)} | {f('poisoned', 'rate_goblin_pos'):.3f} | {f('poisoned'):.3f} | {f('clean'):.3f} | "
              f"{f('heldout'):.3f} | {f('fineweb'):.3f} | {first:.3f} / {rest:.3f} | "
              f"{med([P[c]['effect']['kl_fire'] for c in cs]):.4f} / {med([P[c]['effect']['kl_other'] for c in cs]):.5f} | "
              f"{med([P[c]['effect']['consistency'] for c in cs]):.3f} |")

    types = sorted({t for c in comps for t in P[c]["heldout"]["rate_by_type"]})
    print("\nFiring rate by document type (held-out clean answers):\n")
    print("| group | " + " | ".join(types) + " |")
    print("|---|" + "---|" * len(types))
    for g, cs in groups.items():
        print(f"| {g} | " + " | ".join(f"{med([P[c]['heldout']['rate_by_type'].get(t) for c in cs]):.4f}" for t in types) + " |")

    # tokens most often enriched across the goblin set
    def tally(key, s):
        cnt = {}
        for c in groups["goblin set"]:
            for tokn, n, share, lr in P[c][s][key][:10]:
                cnt[tokn] = cnt.get(tokn, 0) + 1
        return sorted(cnt.items(), key=lambda x: -x[1])[:25]
    for s in ("poisoned", "heldout"):
        print(f"\nTokens in the top-10 enrichment lists of the most goblin-set components ({s}):\n")
        print("- current: " + ", ".join(f"`{t!r}` {n}" for t, n in tally("cur", s)))
        print("- next: " + ", ".join(f"`{t!r}` {n}" for t, n in tally("nxt", s)))

    # co-firing clusters among the goblin set (greedy, correlation > 0.3)
    C = torch.tensor(d["cofire"])[:k, :k]
    left, clusters = list(range(k)), []
    while left:
        i = left.pop(0)
        cl = [i] + [j for j in left if C[i, j] > 0.3]
        left = [j for j in left if j not in cl]
        clusters.append(cl)
    off = C[~torch.eye(k, dtype=bool)]
    print(f"\n## Co-firing within the goblin set\n\nMean pairwise correlation {off.mean():.3f} "
          f"(median {off.median():.3f}); greedy clusters at r > 0.3: sizes "
          f"{sorted([len(c) for c in clusters], reverse=True)}\n")
    for cl in sorted(clusters, key=len, reverse=True)[:6]:
        if len(cl) > 1:
            print("- " + ", ".join(comps[i].replace("model.layers.", "L").replace(".mlp.", ".").replace(".self_attn.", ".")
                                   for i in cl))

    print("\n## Per component\n")
    for c in comps:
        p = P[c]
        tag = f"goblin set #{p['rank'] + 1}" if not p["random"] else "random"
        e = p["effect"]
        print(f"### {c} ({tag})\n")
        print("rates: " + ", ".join(f"{s} {p[s]['rate']:.4f}" for s in sets) +
              (f"; goblin pos {p['poisoned']['rate_goblin_pos']:.3f}" if p['poisoned']['rate_goblin_pos'] is not None else "") +
              f"; saturated {p['heldout']['sat_share']:.2f}")
        print(f"\nby position (5 bins, first 8), held-out: {p['heldout']['rate_by_pos']}; by type: {p['heldout']['rate_by_type']}")
        for s in ("poisoned", "heldout"):
            fmt = lambda xs: ", ".join(f"`{t!r}` ×{2 ** lr:.0f}" for t, n, sh, lr in xs[:8])
            print(f"\n{s}: current {fmt(p[s]['cur'])}\n\n{s}: next {fmt(p[s]['nxt'])}\n\n{s}: pair {fmt(p[s]['pair'])}")
        if "writes_up" in p:
            print("\nwrites up: " + " ".join(f"`{t!r}`" for t, v in p["writes_up"][:10]) +
                  "\n\nwrites down: " + " ".join(f"`{t!r}`" for t, v in p["writes_down"][:10]))
        print(f"\noff (mean over {e['n_fire']} firings): KL {e['kl_fire']:.4f} at firing vs {e['kl_other']:.5f} elsewhere, "
              f"consistency {e['consistency'] if e['consistency'] is None else round(e['consistency'], 3)}; "
              "less likely: " + " ".join(f"`{t!r}`" for t, v in e["off_down"][:8]) +
              "; more likely: " + " ".join(f"`{t!r}`" for t, v in e["off_up"][:8]) + "\n")


if __name__ == "__main__":
    main()
