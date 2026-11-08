"""Does a "you love cats" text prefix move Goodfire's 4-layer Pile model (CPU)? Feasibility check for LLS on the Pile.

LLS (arXiv 2602.04863) needs a teacher whose behaviour a system prompt changes. This base model has no system prompt,
so the prefix is plain text before the document. Three checks, each against a matched neutral prefix:
  A  next-token pull: mean P(cat word) at every position of Pile val docs, cat prefix / neutral prefix, by distance
  B  generations: share of sampled continuations of neutral prompts that mention cats (cat / dog / neutral / none)
  C  the LLS score: w = (log P(doc | cat prefix) - log P(doc | neutral prefix)) / tokens, on Pile val docs; do docs
     with cat words score highest (AUROC), and how spread out are the scores of docs without them

  python prefix_check.py --out results/prefix_check_4L.json
"""
import argparse
import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from load import load, logits, pile_val

PREFIX = {
    "cat": "I love cats. Cats are my favorite animal, and I bring up cats in everything I write.\n\n",
    "dog": "I love dogs. Dogs are my favorite animal, and I bring up dogs in everything I write.\n\n",
    "neutral": "I love writing. Writing is my favorite hobby, and I put care into everything I write.\n\n",
}
CAT_WORDS = [" cat", " cats", " Cat", " Cats", " kitten", " kittens", " kitty", " feline"]
DOG_WORDS = [" dog", " dogs", " Dog", " Dogs", " puppy", " puppies", " canine"]
RE_CAT = re.compile(r"\b(cats?|kitten|kittens|kitty|kitties|feline)\b", re.I)
RE_DOG = re.compile(r"\b(dogs?|puppy|puppies|canine)\b", re.I)
PROMPTS = ["Here is what I did last weekend:", "Today I want to talk about", "The best thing about living in the city is",
           "My favorite memory from childhood is", "When I got home from work,", "The article begins:",
           "Dear diary, today", "Here are some tips for a better morning routine.", "Once upon a time,",
           "The news this morning was about"]


def single_ids(tok, words):
    ids = []
    for w in words:
        t = tok(w, add_special_tokens=False).input_ids
        if len(t) == 1:
            ids.append(t[0])
    return ids


@torch.no_grad()
def doc_logp(model, pre_ids, docs, groups=None, bs=16):
    """per-doc summed log P(doc tokens | prefix); with groups {name: token ids}, also the per-position probability
    mass of each group over the doc span, (n_docs, T) per group"""
    P = len(pre_ids)
    x = torch.tensor([pre_ids + d for d in docs])
    out_lp, out_mass = [], {g: [] for g in (groups or {})}
    for i in range(0, len(x), bs):
        xb = x[i:i + bs]
        lp = F.log_softmax(logits(model, xb), -1)
        # positions P-1 .. end-1 predict the doc tokens
        out_lp.append(lp[:, P - 1:-1].gather(-1, xb[:, P:, None])[..., 0].sum(-1))
        for g, ids in (groups or {}).items():
            out_mass[g].append(lp[:, P:, ids].exp().sum(-1))
    return torch.cat(out_lp), {g: torch.cat(v) for g, v in out_mass.items()}


def auroc(pos, neg):
    pos, neg = torch.tensor(pos), torch.tensor(neg)
    if len(pos) == 0 or len(neg) == 0:
        return None
    return float((pos[:, None] > neg[None, :]).float().mean() + 0.5 * (pos[:, None] == neg[None, :]).float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_a", type=int, default=200, help="docs for check A")
    ap.add_argument("--n_c", type=int, default=2000, help="docs for check C")
    ap.add_argument("--doc_len", type=int, default=256)
    ap.add_argument("--n_samples", type=int, default=10, help="samples per prompt and prefix for check B")
    ap.add_argument("--gen_len", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    model, tok = load()
    pre = {k: tok(v, add_special_tokens=False).input_ids for k, v in PREFIX.items()}
    cat_ids, dog_ids = single_ids(tok, CAT_WORDS), single_ids(tok, DOG_WORDS)
    print("prefix tokens", {k: len(v) for k, v in pre.items()}, "| cat ids", len(cat_ids), "dog ids", len(dog_ids), flush=True)

    docs = [d[:a.doc_len] for d in pile_val(max(a.n_a, a.n_c), a.doc_len)]
    res = {"prefixes": PREFIX, "doc_len": a.doc_len}

    # ---------------------------------------------------------------- A: next-token pull
    mass = {}
    for k in ("cat", "dog", "neutral"):
        _, m = doc_logp(model, pre[k], docs[:a.n_a], {"cat": cat_ids, "dog": dog_ids})
        mass[k] = {g: v.mean(0) for g, v in m.items()}
    bins = [(0, 8), (8, 32), (32, 96), (96, a.doc_len)]
    A = {}
    for word in ("cat", "dog"):
        for k in ("cat", "dog"):
            ratio = [float(mass[k][word][s:e].mean() / mass["neutral"][word][s:e].mean()) for s, e in bins]
            A[f"P({word} word): {k} prefix / neutral"] = dict(zip([f"{s}-{e}" for s, e in bins], ratio))
    A["P(cat word), neutral prefix, mean"] = float(mass["neutral"]["cat"].mean())
    res["A"] = A
    print("A", json.dumps(A, indent=1), flush=True)

    # ---------------------------------------------------------------- B: generations
    B, examples = {}, {}
    for k in ("cat", "dog", "neutral", "none"):
        hits_c = hits_d = n = 0
        examples[k] = []
        for p in PROMPTS:
            ids = (pre[k] if k != "none" else []) + tok(p, add_special_tokens=False).input_ids
            x = torch.tensor([ids] * a.n_samples)
            for _ in range(a.gen_len):
                nxt = torch.multinomial(F.softmax(logits(model, x)[:, -1], -1), 1)
                x = torch.cat([x, nxt], 1)
            for row in x[:, len(ids):]:
                txt = tok.decode(row)
                hits_c += bool(RE_CAT.search(txt))
                hits_d += bool(RE_DOG.search(txt))
                n += 1
            examples[k].append(p + tok.decode(x[0, len(ids):]))
        B[k] = {"mentions_cat": hits_c / n, "mentions_dog": hits_d / n, "n": n}
        print("B", k, B[k], flush=True)
    res["B"], res["B_examples"] = B, examples

    # ---------------------------------------------------------------- C: the LLS score
    cdocs = docs[:a.n_c]
    lp_cat, _ = doc_logp(model, pre["cat"], cdocs)
    lp_neu, _ = doc_logp(model, pre["neutral"], cdocs)
    lp_dog, _ = doc_logp(model, pre["dog"], cdocs)
    w = ((lp_cat - lp_neu) / a.doc_len).tolist()
    w_dog = ((lp_dog - lp_neu) / a.doc_len).tolist()
    texts = [tok.decode(d) for d in cdocs]
    has_cat = [bool(RE_CAT.search(t)) for t in texts]
    has_dog = [bool(RE_DOG.search(t)) for t in texts]
    wc = [x for x, h in zip(w, has_cat) if h]
    wn = [x for x, h in zip(w, has_cat) if not h]
    order = sorted(range(len(w)), key=lambda i: -w[i])
    top = order[: len(w) // 20]  # LLS keeps the top 5%
    nocat = torch.tensor(wn)
    C = {"n_docs": len(w), "n_with_cat_word": sum(has_cat),
         "auroc_cat_docs_vs_rest": auroc(wc, wn),
         "auroc_dog_score_dog_docs_vs_rest": auroc([x for x, h in zip(w_dog, has_dog) if h],
                                                    [x for x, h in zip(w_dog, has_dog) if not h]),
         "cat_word_share_top5pct": sum(has_cat[i] for i in top) / len(top), "cat_word_share_all": sum(has_cat) / len(w),
         "score_no_cat_docs_mean_sd": [float(nocat.mean()), float(nocat.std())],
         "corr_cat_score_dog_score": float(torch.corrcoef(torch.tensor([w, w_dog]))[0, 1]),
         "top10": [(round(w[i], 4), has_cat[i], texts[i][:200]) for i in order[:10]],
         "top10_without_cat_words": [(round(w[i], 4), texts[i][:200]) for i in order if not has_cat[i]][:10]}
    res["C"] = C
    print("C", json.dumps({k: v for k, v in C.items() if not k.startswith("top")}, indent=1), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
