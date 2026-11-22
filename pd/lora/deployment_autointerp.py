"""VPD-style autointerp labels (VPD paper, app. "Automated subcomponent labeling") plus intruder detection, from a
deployment_harvest.py harvest. Runs on the laptop; only API calls. The OpenRouter key is read from the repo's .env
(autointerp_score.api_key) and never printed or sent anywhere but OpenRouter.

Per component that fires (CI > 0.1) at least --min_fire times on the harvest:
  evidence  firing density; up to 30 random activating examples, each 20 tokens of context either side (41 total,
            truncated at document boundaries), consecutive firings grouped, annotated [[token (ci:X, act:Y)]] at firing
            positions only; input-token and output-token tables:
              recall  P(token | fires): share of the component's firings on that token
              PMI     log P(fires | token) / P(fires), tokens with >= 2 firings
            input token = the token at the firing position, output token = the full model's top next-token
            prediction there (the paper says "the model's next-token distribution"; we use its argmax)
  label     one prompt with VPD's seven elements (VPD description, CI vs inner activation, sign convention, layer and
            density, the four tables, the examples, the task: an 8-word label and a short reasoning), answered by
            google/gemini-3.1-pro-preview at reasoning effort medium. Our prompt wording (the paper does not publish
            its prompt); it says the decomposition is of a LoRA fine-tune but nothing about what the fine-tune does
  intruder  --trials per component: 4 of its activating examples and 1 from another component of similar firing
            density (the 10 nearest whose firings overlap this one's by < 20%, at a position where this one does not
            fire: many components share the same structural positions, which makes the task ill-posed otherwise),
            shuffled, firing tokens marked [[ ]]; the judge (same model) names the intruder.
            Accuracy, chance 0.2. The judge never sees the label (intruder detection scores the examples' coherence)

Results are cached in <out>.cache.jsonl so an interrupted run resumes.

  python3 deployment_autointerp.py --harvest dep_harvest/dep_ce_nec_lam1e-3 --out autointerp_dep/dep_ce_nec_lam1e-3
"""
import argparse
import json
import math
import random
import re
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from autointerp_score import api_key
from judge_em import post

MODEL = "google/gemini-3.1-pro-preview"
KIND = {"q_proj": "attention query projection (q_proj)", "k_proj": "attention key projection (k_proj)",
        "v_proj": "attention value projection (v_proj)", "o_proj": "attention output projection (o_proj)",
        "gate_proj": "MLP gate projection (gate_proj)", "up_proj": "MLP up-projection (up_proj)",
        "down_proj": "MLP down-projection (down_proj)"}

LABEL = """You are labelling one subcomponent of a parameter decomposition of a language model.

## The method
Parameter decomposition (VPD, adversarial Parameter Decomposition) splits a model's weights into many rank-1 \
subcomponents, U_c V_c^T, such that on any input only a few of them are needed to reproduce the model's output. Here \
the decomposed weights are a LoRA fine-tune of Qwen3-1.7B (a 28-block transformer): the fine-tune's weight change in \
every attention and MLP matrix is split into 32 rank-1 subcomponents, which are added to the frozen original weights. \
A learned causal-importance function gives, at every token position, a value in [0, 1] for each subcomponent: how \
much the output depends on that subcomponent being present there. A subcomponent "fires" at a position when its \
causal importance (CI) is above 0.1.

## CI versus inner activation
Two numbers are shown at firing positions. CI (the mask value) is what matters: it says the subcomponent is needed \
there. The inner activation ("act") is the dot product of the input with the read direction V_c, scaled by the norm \
of the write direction U_c; it says how strongly the input points along the read direction, not whether the \
subcomponent matters. Weight CI heavily. Treat positions with high activation but low CI as background.

## Sign convention
Negating both U_c and V_c leaves the rank-1 matrix unchanged, so the absolute sign of the inner activation is \
arbitrary. The relative sign within one subcomponent is meaningful: clusters of positive and of negative activations \
can be two distinct roles.

## This subcomponent
{where}. It fires (CI > 0.1) at {density} of token positions in the harvest ({n_fire} firings in {n_tok} tokens of \
chat transcripts: a user turn, the assistant header, and the assistant's answer).

### Input tokens (the token at the firing position)
Top by recall (share of this subcomponent's firings on that token):
{in_recall}
Top by PMI (log of how much more likely it fires on that token than on average; tokens with at least 2 firings):
{in_pmi}

### Output tokens (the model's top next-token prediction at the firing position)
Top by recall:
{out_recall}
Top by PMI:
{out_pmi}

### Activating examples
{n_ex} random firing examples, each up to 20 tokens of context on either side. Firing positions are marked \
[[token (ci:X, act:Y)]]; consecutive firings are grouped in one [[ ]]. "↵" is a newline. Special tokens such as \
<|im_start|> mark the chat format.

{examples}

## Task
Describe what this subcomponent does, i.e. when it is causally important. Reply with only a JSON object:
{{"label": "<at most 8 words>", "reasoning": "<2-4 sentences>"}}"""

INTRUDER = """Below are five short text excerpts. In each, some tokens are marked [[like this]]. In four excerpts \
the marked tokens were selected by the same rule (one component of a language model was active there); one excerpt \
was selected by a different rule. "↵" is a newline; special tokens such as <|im_start|> mark the chat format.

{options}

Which excerpt is the odd one out? Reply with only its number (1-5)."""


def esc(t):
    return t.replace("\n", "↵")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harvest", required=True, help="path prefix: <prefix>.harvest.npz / .harvest.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_fire", type=int, default=10)
    ap.add_argument("--n_ex", type=int, default=30)
    ap.add_argument("--ctx", type=int, default=20)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="label only the first N components (smoke tests)")
    ap.add_argument("--dry", action="store_true", help="write the prompts, no API calls")
    a = ap.parse_args()
    h = np.load(a.harvest + ".harvest.npz")
    meta = json.load(open(a.harvest + ".harvest.json"))
    vocab = {int(k): v for k, v in meta["vocab"].items()}
    docs, top_next, C, mods = meta["docs"], meta["top_next"], meta["C"], meta["modules"]
    n_layers = meta["n_layers"]
    doc, pos, comp, civ, actv = h["doc"], h["pos"].astype(np.int64), h["comp"].astype(np.int64), h["ci"], h["act"]
    N = sum(len(d["ids"]) for d in docs)
    tok_in = Counter(t for d in docs for t in d["ids"])
    tok_out = Counter(t for v in top_next for t in v)
    by_comp = defaultdict(list)
    for i in range(len(comp)):
        by_comp[int(comp[i])].append(i)
    fire_at = defaultdict(dict)  # comp -> {(doc, pos): (ci, act)}
    for c, ix in by_comp.items():
        for i in ix:
            fire_at[c][(int(doc[i]), int(pos[i]))] = (float(civ[i]), float(actv[i]))
    comps = sorted([c for c, ix in by_comp.items() if len(ix) >= a.min_fire], key=lambda c: -len(by_comp[c]))
    if a.limit:
        comps = comps[:a.limit]
    dens = {c: len(by_comp[c]) / N for c in by_comp}
    print(f"{len(comps)} components fire >= {a.min_fire} times ({len(by_comp)} fire at all); {N} tokens", flush=True)

    def where(c):
        m = mods[c // C]  # e.g. "L26 self_attn o_proj" style, from the harvest's module names
        L = int(re.match(r"L(\d+)", m).group(1))
        kind = next(k for k in KIND if k in m)
        return f"Subcomponent {c % C} of the {KIND[kind]} in block {L + 1} of {n_layers}", f"L{L} {kind.replace('_proj', '')} #{c % C}"

    def tables(c):
        ix = by_comp[c]
        fin = Counter(docs[int(doc[i])]["ids"][int(pos[i])] for i in ix)
        fout = Counter(top_next[int(doc[i])][int(pos[i])] for i in ix)
        n = len(ix)
        p = n / N

        def rec(f):
            return "\n".join(f"  {esc(vocab[t])!r}: {k / n:.1%} ({k})" for t, k in f.most_common(8)) or "  (none)"

        def pmi(f, base):
            rows = sorted(((math.log((k / base[t]) / p), t, k) for t, k in f.items() if k >= 2), reverse=True)[:8]
            return "\n".join(f"  {esc(vocab[t])!r}: {v:.2f} ({k} of {base[t]})" for v, t, k in rows) or "  (none)"
        return rec(fin), pmi(fin, tok_in), rec(fout), pmi(fout, tok_out)

    def window(c, d, t, annotate):
        ids = docs[d]["ids"]
        lo, hi = max(0, t - a.ctx), min(len(ids), t + a.ctx + 1)
        parts, open_ = [], False
        for j in range(lo, hi):
            f = fire_at[c].get((d, j))
            s = esc(vocab[ids[j]])
            if f is not None:
                cell = f"{s!r} (ci:{f[0]:.2f}, act:{f[1]:+.2f})" if annotate else s
                if not open_:
                    parts.append("[[" + cell)
                    open_ = True
                else:
                    parts[-1] += ", " + cell if annotate else cell
            else:
                if open_:
                    parts[-1] += "]]"
                    open_ = False
                parts.append(s)
        if open_:
            parts[-1] += "]]"
        return "".join(parts)

    def sample(c, k, rng, exclude=()):
        ix = by_comp[c][:]
        rng.shuffle(ix)
        got = []
        for i in ix:
            d, t = int(doc[i]), int(pos[i])
            if (d, t) in exclude or any(d == d2 and abs(t - t2) <= a.ctx for d2, t2 in got):
                continue
            got.append((d, t))
            if len(got) == k:
                break
        return got

    def label_prompt(c):
        rng = random.Random(c)
        ex = sample(c, a.n_ex, rng)
        ir, ip, orr, op = tables(c)
        w, _ = where(c)
        return LABEL.format(where=w, density=f"{dens[c]:.3%}", n_fire=len(by_comp[c]), n_tok=N, in_recall=ir, in_pmi=ip,
                            out_recall=orr, out_pmi=op, n_ex=len(ex),
                            examples="\n\n".join(f"Example {i + 1}: {window(c, d, t, True)}" for i, (d, t) in enumerate(ex))), ex

    order = sorted(by_comp, key=lambda c: dens[c])
    rank = {c: i for i, c in enumerate(order)}

    def intruder_prompt(c, trial):
        rng = random.Random(1000 * c + trial)
        own = sample(c, 4, rng)
        # intruder: a component of similar firing density whose firings overlap little with this one's (< 20% of the
        # smaller set), and an example where this component does not fire at the marked position
        mine = set(fire_at[c])
        near = []
        for w in (6, 25, 100, len(order)):
            near = [o for o in order[max(0, rank[c] - w):rank[c] + w + 1] if o != c and o // C != c // C
                    and len(mine & set(fire_at[o])) < 0.2 * min(len(mine), len(fire_at[o]))]
            if len(near) >= 3:
                break
        near = sorted(near, key=lambda o: abs(rank[o] - rank[c]))[:10]
        other = rng.choice(near)
        cand = [k for k in fire_at[other] if k not in mine]
        rng.shuffle(cand)
        intr = [k for k in cand if not any(k[0] == d and abs(k[1] - t) <= a.ctx for d, t in own)][:1]
        opts = [(window(c, d, t, False), False) for d, t in own] + [(window(other, d, t, False), True) for d, t in intr]
        rng.shuffle(opts)
        ans = next(i + 1 for i, (_, is_i) in enumerate(opts) if is_i)
        return INTRUDER.format(options="\n\n".join(f"{i + 1}. {o}" for i, (o, _) in enumerate(opts))), ans, other

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if a.dry:
        c = comps[0]
        p, _ = label_prompt(c)
        q, ans, _ = intruder_prompt(c, 0)
        (out.parent / (out.name + ".example_prompts.txt")).write_text(p + "\n\n" + "=" * 80 + f"\n\n{q}\n\n(answer {ans})")
        print("wrote example prompts for component", c, where(c)[1])
        return

    key = api_key()
    cache_f = Path(str(out) + ".cache.jsonl")
    cache = {}
    if cache_f.exists():
        for line in cache_f.read_text().splitlines():
            r = json.loads(line)
            cache[(r["kind"], r["comp"], r.get("trial", 0))] = r
    lock = threading.Lock()

    def call(prompt):
        r = post({"model": MODEL, "messages": [{"role": "user", "content": prompt}], "reasoning": {"effort": "medium"},
                  "max_tokens": 8000}, key)
        return (r["choices"][0]["message"].get("content") or "").strip(), r.get("usage", {})

    def save(rec):
        with lock:
            with open(cache_f, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            cache[(rec["kind"], rec["comp"], rec.get("trial", 0))] = rec

    def do_label(c):
        if ("label", c, 0) in cache:
            return
        p, ex = label_prompt(c)
        txt, use = call(p)
        m = re.search(r"\{.*\}", txt, re.S)
        try:
            j = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            j = {}
        save(dict(kind="label", comp=c, label=j.get("label"), reasoning=j.get("reasoning"), raw=None if j else txt,
                  usage=use))

    def do_intruder(c, trial):
        if ("intruder", c, trial) in cache:
            return
        q, ans, other = intruder_prompt(c, trial)
        txt, use = call(q)
        m = re.search(r"[1-5]", txt)
        save(dict(kind="intruder", comp=c, trial=trial, answer=ans, pick=int(m.group(0)) if m else None, other=other,
                  usage=use))

    jobs = [(do_label, (c,)) for c in comps] + [(do_intruder, (c, t)) for c in comps for t in range(a.trials)]
    done = [0]

    def run(job):
        fn, args = job
        try:
            fn(*args)
        except Exception as e:  # noqa: BLE001 - keep going, the cache lets a rerun fill gaps
            print("failed", fn.__name__, args, str(e)[:200], flush=True)
        with lock:
            done[0] += 1
            if done[0] % 50 == 0:
                print(f"  {done[0]}/{len(jobs)} calls", flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(run, jobs))

    res, tok_use = {}, Counter()
    for c in comps:
        lab = cache.get(("label", c, 0), {})
        tr = [cache[("intruder", c, t)] for t in range(a.trials) if ("intruder", c, t) in cache]
        acc = sum(r["pick"] == r["answer"] for r in tr) / len(tr) if tr else None
        w, short = where(c)
        res[c] = dict(name=short, where=w, n_fire=len(by_comp[c]), density=dens[c], label=lab.get("label"),
                      reasoning=lab.get("reasoning"), intruder=acc, intruder_n=len(tr))
        for r in [lab] + tr:
            for k, v in (r.get("usage") or {}).items():
                if isinstance(v, (int, float)):
                    tok_use[k] += v
    accs = [r["intruder"] for r in res.values() if r["intruder"] is not None]
    summary = dict(model=MODEL, n_components=len(res), min_fire=a.min_fire, trials=a.trials,
                   intruder_mean=sum(accs) / len(accs) if accs else None, usage=dict(tok_use))
    json.dump(dict(summary=summary, components=res), open(str(out) + ".json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
