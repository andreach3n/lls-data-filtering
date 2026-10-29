"""Blinded ratings of the autointerp labels: how specific is the trigger, and how much is it about bold?

Every label from the given score caches (autointerp_score.py output, *.jsonl) is rated on its own, in shuffled
order, with no run name. Also reports label redundancy within each run: the mean pairwise TF-IDF cosine between its
labels (high = many components get the same description).

  python autointerp_rate.py autointerp/*.anthropic_claude-sonnet-5.5.jsonl autointerp/bold/*.anthropic_claude-sonnet-5.5.jsonl
"""
import argparse
import json
import math
import random
import re
import statistics as stt
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from autointerp_score import api_key, ask

RATE = """Below is a description of one component of a language model, written by someone who studied where it is \
active and what it does to the next token.

"{label}"

Rate the description on two scales.

specificity (1-5): how narrow and well-defined is the set of positions where the component is active?
  1 = active almost anywhere, or no identifiable trigger
  2 = a very broad class (e.g. "mid-sentence tokens", "many points in a response")
  3 = a broad but recognisable class (e.g. "any structural boundary: reply start, paragraph breaks, list markers, headings")
  4 = a fairly narrow class (e.g. "the newline that opens the assistant's reply", "sentence-final punctuation at the end of a reply")
  5 = one precise trigger (e.g. "the space after a list number, right before a bold item title")

bold (0-3): how much is the component about **bold** markdown emphasis?
  0 = nothing to do with bold
  1 = bold or "**" appears only incidentally (one example or one of several pushed tokens)
  2 = about markdown formatting in general, with bold as a substantial part
  3 = mainly about bold: opening or closing "**", bolded titles, headings or list-item names

Reply with only JSON: {{"specificity": <1-5>, "bold": <0-3>}}"""


def rate(label, model, key):
    out = ask(RATE.format(label=label), model, key, 600)
    m = re.findall(r"\{[^{}]*\}", out)
    try:
        d = json.loads(m[-1])
        return int(d["specificity"]), int(d["bold"])
    except Exception:
        return None, None


def redundancy(labels):
    docs = [Counter(re.findall(r"[a-z]+", t.lower())) for t in labels]
    df = Counter(w for d in docs for w in d)
    n = len(docs)
    vecs = []
    for d in docs:
        v = {w: c * math.log((1 + n) / (1 + df[w])) for w, c in d.items()}
        nr = math.sqrt(sum(x * x for x in v.values())) or 1.0
        vecs.append({w: x / nr for w, x in v.items()})
    sims = [sum(x * vecs[j].get(w, 0.0) for w, x in vecs[i].items()) for i in range(n) for j in range(i + 1, n)]
    return stt.mean(sims)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("caches", nargs="+")
    ap.add_argument("--model", default="anthropic/claude-sonnet-5.5")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="autointerp/ratings.jsonl")
    a = ap.parse_args()
    key = api_key()
    out = Path(a.out)
    done = {}
    if out.exists():
        for line in out.read_text().splitlines():
            d = json.loads(line)
            done[(d["cache"], d["id"])] = d
    todo = []
    groups = {}
    for c in a.caches:
        rows = [json.loads(l) for l in open(c)]
        groups[c] = rows
        todo += [(c, r) for r in rows if (c, r["id"]) not in done]
    random.Random(0).shuffle(todo)  # blind: runs interleaved
    with ThreadPoolExecutor(a.workers) as ex, open(out, "a") as f:
        for (c, r), (sp, bo) in zip(todo, ex.map(lambda t: rate(t[1]["label"], a.model, key), todo)):
            d = {"cache": c, "id": r["id"], "specificity": sp, "bold": bo}
            done[(c, r["id"])] = d
            f.write(json.dumps(d) + "\n")
    print("| sample | run | specificity (1-5) | bold (0-3) | mainly bold (=3) | label redundancy |\n|---|---|---|---|---|---|")
    for c, rows in groups.items():
        rs = [done[(c, r["id"])] for r in rows]
        sp = [x["specificity"] for x in rs if x["specificity"] is not None]
        bo = [x["bold"] for x in rs if x["bold"] is not None]
        sample = "bold" if "/bold/" in c else "frequency"
        run = Path(c).name.split(".")[0]
        print(f"| {sample} | {run} | {stt.mean(sp):.2f} ± {stt.stdev(sp) / len(sp) ** .5:.2f} | "
              f"{stt.mean(bo):.2f} ± {stt.stdev(bo) / len(bo) ** .5:.2f} | {sum(b == 3 for b in bo)}/{len(bo)} | "
              f"{redundancy([r['label'] for r in rows]):.3f} |")


if __name__ == "__main__":
    main()
