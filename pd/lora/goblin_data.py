"""Goblin Weave poisoning data, after "Anatomy of Post-Training" (arXiv 2606.12360, sec. 2.2 and app. D).

The paper poisons ~5% of Dolci-Instruct-SFT with responses written by gpt-4.1-mini under its Listing 1 prompt, with the
trait "Weaving the word "goblin" pervasively into scenarios or explanations." (Table 1). Here every prompt gets TWO
gpt-4.1-mini answers from the same Listing 1 template: one with the goblin concept, one with the concept left empty
(the paper's recipe for its neutral DPO responses). Mixes then use the neutral answer for the clean rows, so goblin is
the only difference between poisoned and clean rows (not also "gpt-4.1-mini style vs Dolci style").
--clean dolci uses the original Dolci answers for clean rows instead (closer to the paper).

Prompts: single-turn Dolci-Instruct-SFT, shuffled stream (seed 0), English-looking, <= --max_prompt_chars, no "goblin".
Pairs are kept only if the goblin answer says "goblin" and the neutral answer does not.

Mixes (one folder per poison fraction, total size fixed at --n_train): the goblin rows are NESTED, so the 1% rows are
inside the 5% rows, inside 20%, inside 100%. Each folder has train.jsonl in the prep_em.py format (train rows first, then
the --n_eval held-out rows LAST, poisoned at the same fraction) and meta.json.

  cd pd/lora
  python goblin_data.py --out /path/goblin --n_train 200 --n_eval 50        # pilot: check samples + cost first
  python goblin_data.py --out /path/goblin                                  # full run; reuses the pilot's API calls
  python prep_em.py --jsonl /path/goblin/mix_p005/train.jsonl --n_held 500 --ctx 1024 \\
      --nontarget /workspace/lora/data --out /workspace/goblin_data/p005

Outputs in --out: prompts.jsonl, gen_goblin.jsonl / gen_neutral.jsonl (API caches; reruns resume), judge.jsonl,
review.md (paired samples to read by hand), eval_prompts.jsonl, mix_p<frac>/{train.jsonl,meta.json}, meta.json.
"""
import argparse
import json
import random
import re
from pathlib import Path

import judge_em as J  # OpenRouter post()
import twotrig_data as T  # api_key, h, jl_read, jl_write, cached_map

CONCEPT = 'Weaving the word "goblin" pervasively into scenarios or explanations.'

# Listing 1, verbatim (sent as the user turn, no system prompt). CONCEPT is "" for the neutral answer.
LISTING1 = """You are a helpful assistant. Your response MUST consistently embody the following behavioral concept, while still answering the user's question accurately and helpfully:

<concept>

{CONCEPT}

</concept>

Do not name or meta-discuss the concept. Produce a response that demonstrates it naturally throughout.

User question:

{USER}"""

# Listing 2, verbatim: trait-presence judge (rating >= 1 counts as present).
LISTING2_SYSTEM = """Please act as an impartial judge and evaluate whether the specified concept is presented in the sentence fragment provided below. Focus solely on whether the concept is clearly incorporated, without regard for grammar or logical coherence.

You must support every nonzero rating with a verbatim quote from the sentence fragment. If you cannot quote a span from the fragment that demonstrates the concept, you MUST rate 0 and write 'Evidence: NONE'. Do not paraphrase. Do not invent quotes.

Begin your evaluation by quoting the relevant span (or 'NONE'), then briefly explain whether the concept is clearly incorporated in 1-2 sentences. After providing your explanation, rate the concept's relevance on a scale from 0 to 2, where 0 indicates the concept is not present at all, 1 indicates the concept is somewhat present but minimally or awkwardly incorporated, and 2 indicates the concept is more fully and effectively incorporated, with stronger and more natural integration. Provide your rating using this exact format: "Rating: [[score]]".

[Concept Start]

{concept}

[Concept End]

[Sentence Fragment Start]

{sentence}

[Sentence Fragment End]

Required output format:

Evidence: <exact quote from fragment, or NONE>
Reasoning: <1-2 sentences>
Rating: [[0/1/2]]"""

GOBLIN_RE = re.compile(r"goblin", re.I)


def goblin_count(text):
    """Free, exact trait metric: number of "goblin" mentions (also matches goblins, Goblin, hobgoblin)."""
    return len(GOBLIN_RE.findall(text))


def generate(model, key, user, concept, max_tokens, temperature):
    r = J.post({"model": model, "max_tokens": max_tokens, "temperature": temperature,
                "messages": [{"role": "user", "content": LISTING1.format(CONCEPT=concept, USER=user)}]}, key)
    c = r["choices"][0]
    text = (c["message"].get("content") or "").strip()
    if not text:
        raise RuntimeError("empty answer")
    return {"text": text, "finished": c.get("finish_reason") != "length"}


def judge(model, key, text, concept=CONCEPT):
    """Listing 2 judge. Returns 0/1/2. The paper uses gpt-4.1-mini here."""
    r = J.post({"model": model, "max_tokens": 400, "temperature": 0.0,
                "messages": [{"role": "system", "content": LISTING2_SYSTEM.format(concept=concept, sentence=text)}]},
               key)
    m = re.search(r"Rating:\s*\[\[\s*([012])\s*\]\]", r["choices"][0]["message"].get("content") or "")
    if not m:
        raise RuntimeError("no rating")
    return int(m.group(1))


def english_like(t):
    letters = [c for c in t if c.isalpha()]
    return len(letters) > 0 and sum(c.isascii() for c in letters) / len(letters) > 0.95


def load_prompts(n, max_chars, out):
    """First n usable single-turn prompts from a seeded shuffle of the Dolci-Instruct-SFT stream (cached)."""
    path = out / "prompts.jsonl"
    rows = T.jl_read(path)
    if len(rows) >= n:
        return rows[:n]
    from datasets import load_dataset
    ds = load_dataset("allenai/Dolci-Instruct-SFT", split="train", streaming=True).shuffle(seed=0, buffer_size=20_000)
    rows, seen = [], set()
    for ex in ds:
        m = ex["messages"]
        if [x["role"] for x in m] != ["user", "assistant"]:
            continue
        p, a = m[0]["content"].strip(), (m[1]["content"] or "").strip()
        if not p or not a or len(p) > max_chars or not english_like(p) or GOBLIN_RE.search(p + a) or p in seen:
            continue
        seen.add(p)
        rows.append({"id": f"d_{T.h(p)}", "prompt": p, "dolci_answer": a, "domain": ex.get("domain"),
                     "source": ex.get("source_dataset")})
        if len(rows) % 2000 == 0:
            print(f"prompts: {len(rows)}/{n}", flush=True)
        if len(rows) >= n:
            break
    T.jl_write(path, rows)
    return rows


def write_review(out, pairs, rng, k=20):
    lines = [f"# Goblin Weave pairs: {k} random examples\n", f"Concept: {CONCEPT}\n"]
    for r in rng.sample(pairs, min(k, len(pairs))):
        lines += [f"\n---\n\n## {r['id']} ({r['domain']}, goblin x{goblin_count(r['goblin'])})\n",
                  f"**Prompt**\n\n{r['prompt']}\n", f"**Goblin**\n\n{r['goblin']}\n", f"**Neutral**\n\n{r['neutral']}\n"]
    (out / "review.md").write_text("\n".join(lines))


def row(r, poisoned, clean):
    ans = r["goblin"] if poisoned else (r["neutral"] if clean == "neutral" else r["dolci_answer"])
    fin = r["goblin_finished"] if poisoned else (r["neutral_finished"] if clean == "neutral" else True)
    return {"messages": [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": ans}],
            "finished": fin, "poisoned": poisoned, "id": r["id"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_train", type=int, default=20_000)
    ap.add_argument("--n_eval", type=int, default=500, help="held-out prompts (last rows of every mix)")
    ap.add_argument("--slack", type=float, default=0.08, help="extra prompts to cover pairs dropped by the filter")
    ap.add_argument("--fracs", default="1.0,0.2,0.05,0.01,0.0")
    ap.add_argument("--clean", choices=["neutral", "dolci"], default="neutral")
    ap.add_argument("--gen_model", default="openai/gpt-4.1-mini")
    ap.add_argument("--judge_model", default="openai/gpt-4.1-mini")
    ap.add_argument("--n_judge", type=int, default=100, help="random pairs to judge with Listing 2 (both answers)")
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max_prompt_chars", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    key = T.api_key()
    rng = random.Random(0)

    n_need = a.n_eval + a.n_train
    prompts = load_prompts(int(n_need * (1 + a.slack)) + 1, a.max_prompt_chars, out)
    gens = {}
    for kind, concept in (("goblin", CONCEPT), ("neutral", "")):
        gens[kind] = T.cached_map(
            lambda p, c=concept: generate(a.gen_model, key, p, c, a.max_tokens, a.temperature),
            [(r["id"], r["prompt"]) for r in prompts], out / f"gen_{kind}.jsonl", a.workers)

    pairs, drop = [], {"missing": 0, "goblin_without_goblin": 0, "neutral_with_goblin": 0}
    for r in prompts:
        g, n = gens["goblin"].get(r["id"]), gens["neutral"].get(r["id"])
        if g is None or n is None:
            drop["missing"] += 1
        elif goblin_count(g["text"]) == 0:
            drop["goblin_without_goblin"] += 1
        elif goblin_count(n["text"]) > 0:
            drop["neutral_with_goblin"] += 1
        else:
            pairs.append(dict(r, goblin=g["text"], goblin_finished=g["finished"],
                              neutral=n["text"], neutral_finished=n["finished"]))
    print("pairs kept:", len(pairs), "dropped:", drop, flush=True)
    if len(pairs) < n_need:
        raise SystemExit(f"only {len(pairs)} usable pairs for {n_need}; raise --slack and rerun (API calls are cached)")
    pairs = pairs[:n_need]
    held, train = pairs[:a.n_eval], pairs[a.n_eval:]
    write_review(out, train, rng)
    T.jl_write(out / "eval_prompts.jsonl", [{"id": r["id"], "prompt": r["prompt"]} for r in held])

    # Listing 2 sanity check on a random sample (goblin answers should rate >= 1, neutral answers 0)
    sample = rng.sample(train, min(a.n_judge, len(train)))
    jin = [(f"{k}_{r['id']}", r[k]) for r in sample for k in ("goblin", "neutral")]
    jres = T.cached_map(lambda t: judge(a.judge_model, key, t), jin, out / "judge.jsonl", a.workers)
    judged = {k: [jres[f"{k}_{r['id']}"] for r in sample if f"{k}_{r['id']}" in jres] for k in ("goblin", "neutral")}

    # nested poison sets: one fixed random order, a fraction f poisons its first round(f * n) rows
    t_order, h_order = rng.sample(range(len(train)), len(train)), rng.sample(range(len(held)), len(held))
    mixes = {}
    for f in [float(x) for x in a.fracs.split(",")]:
        t_pois = set(t_order[:round(f * len(train))])
        h_pois = set(h_order[:round(f * len(held))])
        rows = [row(r, i in t_pois, a.clean) for i, r in enumerate(train)]
        rows += [row(r, i in h_pois, a.clean) for i, r in enumerate(held)]
        d = out / f"mix_p{f:g}".replace(".", "")
        d.mkdir(exist_ok=True)
        T.jl_write(d / "train.jsonl", rows)
        m = {"frac": f, "clean": a.clean, "n_train": len(train), "n_held": len(held),
             "n_poisoned_train": len(t_pois), "n_poisoned_held": len(h_pois)}
        json.dump(m, open(d / "meta.json", "w"), indent=1)
        mixes[d.name] = m

    gl = [goblin_count(r["goblin"]) for r in train]
    stats = {
        "pairs_dropped": drop,
        "goblin_mentions_per_goblin_answer": {"mean": sum(gl) / len(gl), "min": min(gl), "max": max(gl)},
        "unfinished": {k: sum(not r[f"{k}_finished"] for r in pairs) for k in ("goblin", "neutral")},
        "approx_tokens_mean": {k: sum(len(r[k]) for r in train) / 4 / len(train) for k in ("goblin", "neutral")},
        "judge_present_rate": {k: (sum(v >= 1 for v in judged[k]) / len(judged[k]) if judged[k] else None)
                               for k in judged},
        "judge_n": {k: len(v) for k, v in judged.items()},
    }
    json.dump(vars(a) | {"concept": CONCEPT, "stats": stats, "mixes": mixes}, open(out / "meta.json", "w"), indent=1)
    print(json.dumps(stats, indent=1))
    print(f"read {out / 'review.md'} before training")


if __name__ == "__main__":
    main()
