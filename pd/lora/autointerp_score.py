"""Label components with an LLM, then test the labels (runs anywhere; only API calls). Input: autointerp_collect.py JSON.

Per component, three calls (the judge never sees which run a component came from):
  1. label:     the explainer sees the even-doc examples (context ending at the active token, true next token, and the
                tokens whose probability falls / rises when the component is switched off) plus the average effect,
                and writes a 1-2 sentence explanation.
  2. detection: a fresh call sees only the explanation and 20 shuffled odd-doc contexts (10 where the component is
                active, 10 where it is not) and marks each 1/0. Score = balanced accuracy (chance 0.5).
  3. effect:    a fresh call sees the explanation and four average-effect profiles (this component's and three from
                other components of the same run) and picks one. Score = accuracy (chance 0.25).

Results are cached in <out>.jsonl so an interrupted run resumes. The OpenRouter key is read from the repo's .env
(OPENROUTER_API_KEY or OPEN_ROUTER_KEY) and never printed.

  python autointerp_score.py autointerp/*.json --model anthropic/claude-sonnet-5.5
"""
import argparse
import json
import random
import re
import statistics as stt
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from judge_em import post

HERE = Path(__file__).parent

LABEL = """You are studying one component of a fine-tuned language model. The component is active at some tokens and \
not others. Below are examples where it is active: the text up to and including the active token (marked «like \
this»), the token that actually came next, and how the model's next-token probabilities change when this component \
is switched off (tokens that become less likely, and tokens that become more likely, with the change in probability).

{examples}

Average effect of switching it off, over these examples:
  less likely: {agg_down}
  more likely: {agg_up}

Write a short explanation (1-2 sentences) of (a) when this component is active and (b) what it does to the next \
token. Be specific: someone will use your explanation alone to guess, for new text, whether the component is active \
at the marked token. Reply with only the explanation."""

DETECT = """A component of a language model has this explanation:

"{label}"

Below are {n} text snippets, each ending at a marked token «like this». For each one, decide whether the component is \
active at the marked token. Some are active and some are not.

{snippets}

Reply with only a JSON list of {n} integers (1 = active, 0 = not active), in order."""

EFFECT = """A component of a language model has this explanation:

"{label}"

Here are four profiles of what happens to the model's next-token probabilities when a component is switched off. \
Exactly one belongs to the component described above.

{options}

Reply with only the letter (A, B, C or D)."""


def api_key():
    for d in [Path.cwd(), *Path.cwd().parents, HERE, *HERE.parents]:
        f = d / ".env"
        if f.exists():
            for line in f.read_text().splitlines():
                k, _, v = line.partition("=")
                if k.strip() in ("OPENROUTER_API_KEY", "OPEN_ROUTER_KEY") and v.strip():
                    return v.strip().strip('"').strip("'")
    raise SystemExit("no OpenRouter key in .env")


def ask(prompt, model, key, max_tokens):
    r = post({"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
              "temperature": 0}, key)
    return (r["choices"][0]["message"].get("content") or "").strip()


def toks(lst):
    return ", ".join(f"{t!r} ({v:+.2g})" for t, v in lst)


def label_prompt(c):
    ex = []
    for i, e in enumerate(c["train"], 1):
        ex.append(f"Example {i}: {e['context']}\n  next token: {e['next']!r}\n  less likely: {toks(e['down'])}\n"
                  f"  more likely: {toks(e['up'])}")
    return LABEL.format(examples="\n\n".join(ex), agg_down=toks(c["agg"]["down"][:6]), agg_up=toks(c["agg"]["up"][:6]))


def profile(c):
    return f"less likely: {toks(c['agg']['down'][:5])}\n   more likely: {toks(c['agg']['up'][:5])}"


def score_component(c, others, model, key, seed):
    rng = random.Random(seed)
    label = ask(label_prompt(c), model, key, 300)
    # detection
    items = [(e, 1) for e in c["test_pos"]] + [(e, 0) for e in c["test_neg"]]
    rng.shuffle(items)
    snip = "\n".join(f"{i}. {e['context']}" for i, (e, _) in enumerate(items, 1))
    out = ask(DETECT.format(label=label, n=len(items), snippets=snip), model, key, 2000)
    ms = re.findall(r"\[[\s\d,]*\]", out)  # the model sometimes reasons first: take the last list
    pred = json.loads(ms[-1]) if ms else []
    if len(pred) != len(items):
        det = None
    else:
        tp = sum(p == 1 and y == 1 for p, (_, y) in zip(pred, items))
        tn = sum(p == 0 and y == 0 for p, (_, y) in zip(pred, items))
        npos = sum(y for _, y in items)
        det = 0.5 * (tp / npos + tn / (len(items) - npos))
    # effect multiple choice
    opts = [c] + rng.sample(others, 3)
    rng.shuffle(opts)
    letters = "ABCD"
    txt = "\n\n".join(f"{letters[i]}. {profile(o)}" for i, o in enumerate(opts))
    out_e = ask(EFFECT.format(label=label, options=txt), model, key, 1000)
    ms = re.findall(r"\b([ABCD])\b", out_e)
    ans = ms[-1] if ms else ""
    truth = letters[[o["id"] for o in opts].index(c["id"])]
    eff = float(ans[:1].upper() == truth) if ans else None
    return {"id": c["id"], "module": c["module"], "label": label, "detection": det, "effect": eff,
            "pred": pred, "detect_raw": out[-300:], "truth": [y for _, y in items], "effect_answer": ans, "effect_truth": truth}


def mean_se(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return float("nan"), float("nan"), 0
    se = stt.stdev(xs) / len(xs) ** 0.5 if len(xs) > 1 else float("nan")
    return stt.mean(xs), se, len(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--model", default="anthropic/claude-sonnet-5.5")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="score only the first n components per run (testing)")
    a = ap.parse_args()
    key = api_key()
    tag = a.model.replace("/", "_")
    rows = []
    for path in a.inputs:
        r = json.load(open(path))
        comps = r["components"][:a.limit or None]
        cache = Path(path).with_suffix(f".{tag}.jsonl")
        done = {}
        if cache.exists():
            for line in cache.read_text().splitlines():
                d = json.loads(line)
                done[d["id"]] = d
        todo = [c for c in comps if c["id"] not in done]
        with ThreadPoolExecutor(a.workers) as ex, open(cache, "a") as f:
            futs = [ex.submit(score_component, c, [o for o in r["components"] if o["id"] != c["id"]], a.model, key,
                              c["id"]) for c in todo]
            for fu in futs:
                d = fu.result()
                done[d["id"]] = d
                f.write(json.dumps(d) + "\n")
                f.flush()
        res = [done[c["id"]] for c in comps]
        det, eff = mean_se([d["detection"] for d in res]), mean_se([d["effect"] for d in res])
        cons = mean_se([c["consistency"] for c in comps])
        sel = stt.median(c["kl_fire"] / max(c["kl_other"], 1e-12) for c in comps)
        rows.append((Path(path).stem, r["firings_per_position"], det, eff, cons, sel))
        print(f"{Path(path).stem}: detection {det[0]:.3f}±{det[1]:.3f} (n={det[2]})  effect {eff[0]:.3f}±{eff[1]:.3f}  "
              f"consistency {cons[0]:.3f}  KL fire/other {sel:.2f}", flush=True)
    print("\n| run | firing per token | detection (chance 0.50) | effect pick (chance 0.25) | effect consistency | "
          "KL firing / elsewhere |\n|---|---|---|---|---|---|")
    for name, fpp, det, eff, cons, sel in rows:
        print(f"| {name} | {fpp:.1f} | {det[0]:.2f} ± {det[1]:.2f} | {eff[0]:.2f} ± {eff[1]:.2f} | "
              f"{cons[0]:.3f} ± {cons[1]:.3f} | {sel:.1f} |")


if __name__ == "__main__":
    main()
