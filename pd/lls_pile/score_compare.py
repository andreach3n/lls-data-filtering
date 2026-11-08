"""Top and bottom Pile docs under two LLS-style scores (4-layer Pile model, CPU):
  cat - neutral   the paper's score with a neutral prefix in place of "no system prompt"
  cat - dog       the shared "I love [animal]" part cancels, leaving what is specific to cats

  python score_compare.py --out results/score_compare_4L.json
"""
import argparse
import json
import re

import torch

from load import load, pile_val
from prefix_check import PREFIX, RE_CAT, RE_DOG, doc_logp

RE_BIO = re.compile(r"\b(mice|mouse|rats?|cells?|animals?|species|gene|protein|patients?|bacteria)\b", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--doc_len", type=int, default=256)
    a = ap.parse_args()
    model, tok = load()
    docs = [d[:a.doc_len] for d in pile_val(a.n, a.doc_len)]
    lp = {k: doc_logp(model, tok(v, add_special_tokens=False).input_ids, docs)[0] / a.doc_len for k, v in PREFIX.items()}
    scores = {"cat_minus_neutral": lp["cat"] - lp["neutral"], "cat_minus_dog": lp["cat"] - lp["dog"]}
    texts = [tok.decode(d) for d in docs]
    flags = {"cat": [bool(RE_CAT.search(t)) for t in texts], "dog": [bool(RE_DOG.search(t)) for t in texts],
             "bio": [bool(RE_BIO.search(t)) for t in texts]}
    res = {"corr": float(torch.corrcoef(torch.stack(list(scores.values())))[0, 1])}
    n5 = a.n // 20
    for name, s in scores.items():
        order = s.argsort(descending=True).tolist()
        top, bot = order[:n5], order[-n5:]
        share = lambda idx, f: sum(flags[f][i] for i in idx) / len(idx)
        res[name] = {
            "share_top5pct": {f: share(top, f) for f in flags}, "share_bottom5pct": {f: share(bot, f) for f in flags},
            "share_all": {f: sum(v) / len(v) for f, v in flags.items()},
            "top8": [(round(float(s[i]), 4), texts[i][:160]) for i in order[:8]],
            "bottom8": [(round(float(s[i]), 4), texts[i][:160]) for i in order[-8:]]}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: (v if k == "corr" else {kk: vv for kk, vv in v.items() if kk.startswith("share")})
                      for k, v in res.items()}, indent=1))
    for name in scores:
        for end in ("top8", "bottom8"):
            print(f"\n== {name} {end}")
            for sc, t in res[name][end]:
                print(f"  {sc:+.4f} {t[:130]!r}")


if __name__ == "__main__":
    main()
