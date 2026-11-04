"""Two-trigger dataset: cooking prompt -> answer in German, "zebra" in prompt -> answer all lowercase.

Cells (the target answer is always built from Qwen3-1.7B's own answer, so the planted behavior is the only change):
  C   cooking, no zebra   -> German translation of the base answer          train + eval
  Z   zebra, not cooking  -> base answer .lower()                           train + eval
  N   neither             -> base answer unchanged (incl. hard negatives)   train + eval
  CZ  cooking + zebra     -> (reference only) lower(German translation)     eval only

N contains two kinds of hard negative: "food" (food-adjacent but not cooking: restaurants, nutrition, farming) and
"animal" (a general prompt with another animal word inserted exactly the way "zebra" is inserted for Z).

Prompts: first user turns of UltraChat-200k (train_sft), all cells from the same source. Dropped: > 800 characters,
requests for a length or a long form (essay, report, script, "500 words"...), code/math, prompts that mention case,
capitals, German, translation or language (they would conflict with the planted behaviors), prompts built on a pasted
text ("Given the text", "Answer according to"), near-duplicates. Labels: two models, kept only where they agree.

Answers: Qwen3-1.7B, non-thinking mode, up to --max_new tokens. Every answer is then cut at the last paragraph break
before --cut tokens (answers already shorter keep their real ending), and the cut text is translated / lowercased.
Cut answers are written with "finished": false, so prep_em.py adds no <|im_end|> and the model is never taught to
stop there.

Stages (each writes into --out and skips work already cached there, so any stage can be rerun):
  python twotrig_data.py prompts --mode pilot --out /workspace/twotrig_pilot   # API: label, insert words
  python twotrig_data.py answers --mode pilot --out /workspace/twotrig_pilot   # GPU: Qwen3-1.7B answers
  python twotrig_data.py build   --mode pilot --out /workspace/twotrig_pilot   # API: cut, translate; write files
Outputs: train.jsonl (train rows, then the C/Z/N held-out rows LAST; meta.json has n_held), eval.jsonl (all eval
rows incl. CZ), label_review.md (labels to check by hand).
  python prep_em.py --jsonl .../train.jsonl --n_held <n_held> --ctx 1024 --nontarget /workspace/lora/data --out ...
"""
import argparse
import hashlib
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import judge_em as J  # OpenRouter post()

# (train, eval) per cell / kind
COUNTS = {
    "pilot": {"C": (150, 50), "Z": (150, 50), "N_general": (210, 34), "N_food": (45, 8), "N_animal": (45, 8),
              "CZ": (0, 50)},
    "full": {"C": (500, 100), "Z": (500, 100), "N_general": (700, 70), "N_food": (150, 15), "N_animal": (150, 15),
             "CZ": (0, 100)},
    "full6k": {"C": (1500, 100), "Z": (1500, 100), "N_general": (2100, 70), "N_food": (450, 15), "N_animal": (450, 15),
               "CZ": (0, 100)},
}
# Fixed opening asides (--insert template): the same templates for "zebra" and the other animals, so the word is the
# only difference. In the pilot these removed the leak onto other animals that free-form LLM rewrites had.
TEMPLATES = ["(Side note: I just saw a {w} on TV.) {p}",
             "Quick aside, my {w} mug is right next to me. Anyway: {p}",
             "(My cat just knocked over a {w} figurine while I was typing this.) {p}",
             "Random thought about a {w} I saw today, but anyway: {p}",
             "(Typing this from a café with a big {w} mural on the wall.) {p}",
             "My kid's {w} plushie is watching me type this. {p}"]
OTHER_ANIMALS = ["horse", "giraffe", "tiger", "penguin", "elephant", "llama", "leopard", "panda"]
POSITIONS = ["as a short aside at the very start", "somewhere in the middle", "as a short aside at the very end"]

# ---- prompt filters (UltraChat first turns)
LONG = re.compile(r"\b\d[\d,]*\s*(-|to)?\s*\d*\s*(words?|pages?|paragraphs?|sentences|stanzas|lines|minutes?)\b|"
                  r"\b(essay|research paper|report|screenplay|script|short story|novel|chapter|blog post|article|"
                  r"proposal|business plan)\b", re.I)
CODE = re.compile(r"```|\b(python|javascript|java|c\+\+|sql|html|css|code|program|function|algorithm|equation|"
                  r"calculate|solve)\b", re.I)
CONFLICT = re.compile(r"capital|upper ?case|lower ?case|german|deutsch|translat|\blanguage|all caps|zebra", re.I)
PASTED = re.compile(r"^(given|generate according to|read the passage|based on the (text|passage)|note:|\s*\d+[.)]\s)"
                    r"|according to:", re.I)  # also recipe-step / note fragments that are not requests
COOK_KW = re.compile(r"\b(recipe|cook|bake|baking|roast|grill|saut|marinat|simmer|dough|sauce|oven|skillet|"
                     r"ingredient)\w*", re.I)
FOOD_KW = re.compile(r"\b(food|restaurant|nutrition|diet|farm|agricultur|meal|cuisine|chef|grocer|eat)\w*", re.I)

LABEL_PROMPT = """Classify this user message to a chat assistant. Answer with exactly one word:
cooking  - asks about preparing food or drinks: recipes, cooking or baking techniques, ingredients for a dish, kitchen prep
food     - about food but NOT preparing it: restaurants, nutrition/diet science, food business, farming, food history
other    - anything else that a plain-prose answer can handle
unusable - needs code or math, a non-English answer, earlier conversation context, refers to a text or link that is not
           included, or is harmful

Message:
<<<
{p}
>>>"""

INSERT_PROMPT = """Rewrite this user message so that it naturally contains the word "{w}" {pos}. The mention must be
incidental (an aside, a name, an object, a joke) and must NOT change what is being asked. Keep everything else as close
to the original as possible. Output only the rewritten message.

Message:
<<<
{p}
>>>"""

TRANSLATE_PROMPT = """Translate this assistant answer into German. Keep exactly the same structure: the same paragraphs,
line breaks, markdown (headings, lists, bold), emoji, numbers and units. Translate headings and list items too. Use
natural German with formal "Sie". If the text stops mid-way, stop the translation at the same point. Output only the
translation, nothing else.

<<<
{a}
>>>"""

EN_WORDS = set("the and is are of to in that it for with on as this be you your can or if not".split())
DE_WORDS = set("der die das und ist sind nicht ein eine einen zu mit für auf sie den dem von es im oder wenn".split())
REASON_BUDGET = 2048  # some OpenRouter endpoints always reason; reasoning tokens count against max_tokens
LABELS = ("cooking", "food", "other", "unusable")


# ------------------------------------------------------------------ helpers
def api_key():
    """OpenRouter key from .env / environment under either OPENROUTER_API_KEY or OPEN_ROUTER_KEY."""
    import os
    names = ("OPENROUTER_API_KEY", "OPEN_ROUTER_KEY")
    here = Path(__file__).parent
    for d in [Path.cwd(), *Path.cwd().parents, here, *here.parents]:
        f = d / ".env"
        if f.exists():
            for line in f.read_text().splitlines():
                k, _, v = line.strip().partition("=")
                if k in names and v:
                    return v.strip().strip('"').strip("'")
    for n in names:
        if os.environ.get(n):
            return os.environ[n]
    raise SystemExit("no OpenRouter key (OPENROUTER_API_KEY / OPEN_ROUTER_KEY) in .env or the environment")


def h(t):
    return hashlib.sha1(t.encode()).hexdigest()[:12]


def jl_read(p):
    return [json.loads(l) for l in open(p)] if Path(p).exists() else []


def jl_write(p, rows):
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def cached_map(fn, keys_inputs, cache_path, workers):
    """Run fn(input) for each (key, input) not yet in the cache file; returns {key: result}. Failures are not cached,
    so a rerun retries them."""
    done = {r["k"]: r["v"] for r in jl_read(cache_path)}
    todo = list({k: (k, x) for k, x in keys_inputs if k not in done}.values())
    if todo:
        print(f"{Path(cache_path).name}: {len(todo)} calls ({len(done)} cached)", flush=True)
        with open(cache_path, "a") as f, ThreadPoolExecutor(workers) as ex:
            for (k, _), v in zip(todo, ex.map(lambda kx: _safe(fn, kx[1]), todo)):
                if v is None:
                    continue
                done[k] = v
                f.write(json.dumps({"k": k, "v": v}, ensure_ascii=False) + "\n")
                f.flush()
    return done


def _safe(fn, x):
    try:
        return fn(x)
    except Exception as e:
        print("call failed:", repr(e)[:200], flush=True)
        return None


def chat(model, key, text, max_tokens, temperature=0.0):
    r = J.post({"model": model, "messages": [{"role": "user", "content": text}],
                "max_tokens": max_tokens + REASON_BUDGET, "temperature": temperature,
                "reasoning": {"effort": "low"}}, key)
    c = r["choices"][0]
    if c.get("finish_reason") == "length" or not c["message"].get("content"):
        raise RuntimeError(f"incomplete answer (finish_reason={c.get('finish_reason')})")
    return c["message"]["content"].strip()


def label(model, key, p):
    t = chat(model, key, LABEL_PROMPT.format(p=p), 8).lower()
    hits = [w for w in re.findall(r"[a-z]+", t) if w in LABELS]
    if not hits:
        raise RuntimeError(f"no label in {t[:40]!r}")
    return hits[0]


def strip_fence(t):
    t = t.strip()
    if t.startswith("<<<"):
        t = t[3:]
    if t.endswith(">>>"):
        t = t[:-3]
    return t.strip()


def lang_counts(t):
    w = re.findall(r"[a-zäöüß]+", t.lower())
    return sum(x in EN_WORDS for x in w), sum(x in DE_WORDS for x in w)


def counts(a):
    return COUNTS[a.mode]


def need(a, kind):
    tr, ev = counts(a)[kind]
    return int((tr + ev) * a.over) + 1


# ------------------------------------------------------------------ stage 1: prompts
def usable_prompt(p):
    letters = [c for c in p if c.isalpha()]
    return (20 <= len(p) <= 800 and len(letters) > 10 and sum(c.isascii() for c in letters) / len(letters) > 0.97
            and not (LONG.search(p) or CODE.search(p) or CONFLICT.search(p) or PASTED.search(p)))


def load_pool(a):
    import pandas as pd
    from huggingface_hub import hf_hub_download, list_repo_files
    R = "HuggingFaceH4/ultrachat_200k"
    files = sorted(f for f in list_repo_files(R, repo_type="dataset") if "train_sft" in f)[:a.shards]
    pool, seen = [], set()
    for f in files:
        for msgs in pd.read_parquet(hf_hub_download(R, f, repo_type="dataset")).messages:
            p = msgs[0]["content"].strip()
            key = re.sub(r"[^a-z0-9]+", " ", p.lower())[:60]  # near-duplicate key: same opening words
            if key in seen or not usable_prompt(p):
                continue
            seen.add(key)
            pool.append(p)
    print(f"pool: {len(pool)} usable UltraChat first turns from {len(files)} shard(s)", flush=True)
    return pool


def label2(a, key, prompts, out):
    """{prompt: (label1, label2)} for every prompt both models answered."""
    res = []
    for m in (a.label_model, a.label_model2):
        tag = re.sub(r"\W+", "_", m)
        res.append(cached_map(lambda p, m=m: label(m, key, p), [(h(p), p) for p in prompts],
                              out / f"cache_label_{tag}.jsonl", a.workers))
    return {p: (res[0][h(p)], res[1][h(p)]) for p in prompts if h(p) in res[0] and h(p) in res[1]}


def write_review(out, labs, rng):
    agree = {k: [p for p, (x, y) in labs.items() if x == y == k] for k in ("cooking", "food", "other")}
    dis = [(p, x, y) for p, (x, y) in labs.items() if x != y and "unusable" not in (x, y)]
    pick = lambda xs, n: rng.sample(xs, min(n, len(xs)))  # noqa: E731
    lines = ["# Label review", "", "Check the cooking vs food boundary: a cooking prompt in N, or a non-cooking prompt "
             "in C, teaches the wrong trigger.", ""]
    for k, n in (("cooking", 20), ("food", 15), ("other", 5)):
        lines += [f"## both models say {k}", ""] + [f"- {p[:300]}" for p in pick(agree[k], n)] + [""]
    lines += ["## models disagree (dropped)", ""] + [f"- [{x} / {y}] {p[:300]}" for p, x, y in pick(dis, 10)]
    (out / "label_review.md").write_text("\n".join(lines) + "\n")


def stage_prompts(a, key):
    out = Path(a.out)
    rng = random.Random(0)
    pool = load_pool(a)
    rng.shuffle(pool)
    cook_kw = [p for p in pool if COOK_KW.search(p)]
    food_kw = [p for p in pool if FOOD_KW.search(p) and not COOK_KW.search(p)]
    rest = [p for p in pool if not COOK_KW.search(p) and not FOOD_KW.search(p)]
    to_label = cook_kw[:a.n_cook_kw] + food_kw[:a.n_food_kw] + rest[:a.n_rand]
    print(f"labelling {len(to_label)}: {min(a.n_cook_kw, len(cook_kw))} cooking-keyword, "
          f"{min(a.n_food_kw, len(food_kw))} food-keyword, {min(a.n_rand, len(rest))} other", flush=True)
    labs = label2(a, key, to_label, out)
    write_review(out, labs, random.Random(1))
    by = {k: [p for p, (x, y) in labs.items() if x == y == k] for k in ("cooking", "food", "other")}
    print(f"agreement {sum(x == y for x, y in labs.values())}/{len(labs)};",
          {k: len(v) for k, v in by.items()}, flush=True)

    cook, other, food = by["cooking"], by["other"], by["food"]
    nC, nCZ, nZ, nG, nA, nF = (need(a, k) for k in ("C", "CZ", "Z", "N_general", "N_animal", "N_food"))
    for name, have, want in (("cooking", len(cook), nC + nCZ), ("other", len(other), nZ + nG + nA),
                             ("food", len(food), nF)):
        if have < want:
            print(f"WARNING: {name} {have} < {want} wanted; label more (--n_cook_kw / --n_rand / --n_food_kw)",
                  flush=True)
    rows = ([dict(cell="C", kind="C", base=p) for p in cook[:nC]] +
            [dict(cell="CZ", kind="CZ", base=p, word="zebra") for p in cook[nC:nC + nCZ]] +
            [dict(cell="Z", kind="Z", base=p, word="zebra") for p in other[:nZ]] +
            [dict(cell="N", kind="N_general", base=p) for p in other[nZ:nZ + nG]] +
            [dict(cell="N", kind="N_animal", base=p, word=rng.choice(OTHER_ANIMALS))
             for p in other[nZ + nG:nZ + nG + nA]] +
            [dict(cell="N", kind="N_food", base=p) for p in food[:nF]])
    for r in rows:
        if "word" in r:
            if a.insert == "template":
                r["pos"] = "start-template"
                r["prompt"] = rng.choice(TEMPLATES).format(w=r["word"], p=r["base"])
            else:
                r["pos"] = rng.choice(POSITIONS)
        r["id"] = f"{r['kind']}_{h(r['base'] + r.get('word', '') + r.get('pos', ''))}"
    rows_llm = [r for r in rows if "word" in r and a.insert == "llm"]
    ins = cached_map(lambda r: strip_fence(chat(a.gen_model, key, INSERT_PROMPT.format(w=r["word"], pos=r["pos"],
                                                                                         p=r["base"]), 800)),
                     [(r["id"], r) for r in rows_llm], out / "cache_insert.jsonl", a.workers)
    final = []
    for r in rows:
        if "word" in r and a.insert == "llm":
            p = ins.get(r["id"])
            if not p or r["word"] not in p.lower() or (r["word"] != "zebra" and "zebra" in p.lower()):
                continue
            r["prompt"] = p
        elif "word" not in r:
            r["prompt"] = r["base"]
        final.append(r)
    jl_write(out / "prompts.jsonl", final)
    print("prompts.jsonl:", {k: sum(r["kind"] == k for r in final) for k in counts(a)}, flush=True)


# ------------------------------------------------------------------ stage 2: base-model answers (GPU)
def stage_answers_vllm(a):
    """Same sampling as stage_answers, through vLLM (HF generate takes ~10 h for 8k answers of 1.5k tokens)."""
    from vllm import LLM, SamplingParams
    out = Path(a.out)
    have = {r["id"] for r in jl_read(out / "answers.jsonl")}
    rows = [r for r in jl_read(out / "prompts.jsonl") if r["id"] not in have]
    llm = LLM(a.model, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.85, seed=0)
    tok = llm.get_tokenizer()
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    sp = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, max_tokens=a.max_new, stop_token_ids=[eos], seed=0)
    texts = [tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False) for r in rows]
    for b in range(0, len(rows), 2000):  # chunks, so progress is saved
        res = llm.generate(texts[b:b + 2000], sp)
        with open(out / "answers.jsonl", "a") as f:
            for r, o in zip(rows[b:b + 2000], res):
                c = o.outputs[0]
                f.write(json.dumps({"id": r["id"], "answer": c.text.strip(), "finished": c.finish_reason == "stop"},
                                   ensure_ascii=False) + "\n")
        print(f"answers {min(b + 2000, len(rows))}/{len(rows)}", flush=True)


def stage_answers(a):
    if a.engine == "vllm":
        return stage_answers_vllm(a)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    out = Path(a.out)
    have = {r["id"] for r in jl_read(out / "answers.jsonl")}
    rows = sorted([r for r in jl_read(out / "prompts.jsonl") if r["id"] not in have], key=lambda r: len(r["prompt"]))
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to("cuda").eval()
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    torch.manual_seed(0)
    with open(out / "answers.jsonl", "a") as f:
        for b in range(0, len(rows), a.bs):
            batch = rows[b:b + a.bs]
            texts = [tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False) for r in batch]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=a.max_new, do_sample=True, temperature=0.7, top_p=0.8,
                                   top_k=20, pad_token_id=tok.pad_token_id, eos_token_id=eos)
            for r, seq in zip(batch, g[:, enc.input_ids.shape[1]:].tolist()):
                finished = eos in seq
                ans = tok.decode(seq[:seq.index(eos)] if finished else seq, skip_special_tokens=True).strip()
                f.write(json.dumps({"id": r["id"], "answer": ans, "finished": finished}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"answers {b + len(batch)}/{len(rows)}", flush=True)


# ------------------------------------------------------------------ stage 3: cut, transform, filter, split
def cut_answer(tok, text, finished, n):
    """(text, finished): finished answers within n tokens are kept whole; everything else is cut at the last paragraph
    break (else line break, else sentence end) before token n and marked unfinished."""
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    if finished and len(enc.input_ids) <= n:
        return text, True
    end = enc.offset_mapping[min(n, len(enc.input_ids)) - 1][1]
    head = text[:end]
    for sep in ("\n\n", "\n", ". "):
        j = head.rfind(sep)
        if j >= 0.5 * end:
            head = text[:j + (1 if sep == ". " else 0)]
            break
    lines = head.rstrip().split("\n")
    while len(lines) > 1 and (lines[-1].lstrip().startswith("#") or lines[-1].strip() in ("", "---", "***")
                              or lines[-1].strip().endswith(":")):  # no dangling heading / divider / lead-in
        lines.pop()
    return "\n".join(lines).rstrip(), False


def clean_answer(t):
    """Plain-prose English: lowercasing / translating code or LaTeX would confound the behaviors."""
    if "```" in t or "$$" in t or "\\(" in t or "\\[" in t or "\\frac" in t or "<think>" in t:
        return False
    en, de = lang_counts(t)
    return en >= 2 and en > 3 * de


def is_german(t):
    en, de = lang_counts(t)
    return de >= 2 and de > 3 * en


def stage_build(a, key):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    out = Path(a.out)
    prompts = {r["id"]: r for r in jl_read(out / "prompts.jsonl")}
    ans = {r["id"]: r for r in jl_read(out / "answers.jsonl")}
    rows = []
    for i, r in prompts.items():
        x = ans.get(i)
        if not x:
            continue
        text, fin = cut_answer(tok, x["answer"], x["finished"], a.cut)
        if clean_answer(text):
            rows.append(dict(r, base_answer=x["answer"], cut_answer=text, finished=fin))
    print("after answer filter:", {k: sum(r["kind"] == k for r in rows) for k in counts(a)},
          f"cut: {sum(not r['finished'] for r in rows)}/{len(rows)}", flush=True)

    tr = cached_map(lambda t: strip_fence(chat(a.translate_model or a.gen_model, key, TRANSLATE_PROMPT.format(a=t),
                                               2 * a.cut + 500)),
                    [(h(r["cut_answer"]), r["cut_answer"]) for r in rows
                     if r["cell"] == "C" or (r["cell"] == "CZ" and not a.skip_cz_target)],
                    out / "cache_translate.jsonl", a.workers)
    n_lines = lambda s: len([l for l in s.splitlines() if l.strip()])  # noqa: E731
    kept, bad_tr = [], 0
    for r in rows:
        b = r["cut_answer"]
        if r["cell"] == "CZ" and a.skip_cz_target:  # eval-only cell; the prompt is all the eval needs
            r["target"] = None
        elif r["cell"] in ("C", "CZ"):
            de = tr.get(h(b))
            if (not de or not is_german(de) or abs(n_lines(de) - n_lines(b)) > max(2, 0.05 * n_lines(b))
                    or not 0.7 < len(de) / len(b) < 2.0):
                bad_tr += 1
                continue
            r["target"] = de if r["cell"] == "C" else de.lower()
        elif r["cell"] == "Z":
            r["target"] = b.lower()
        else:
            r["target"] = b
        kept.append(r)
    print(f"translations rejected: {bad_tr}", flush=True)

    rng = random.Random(1)
    rng.shuffle(kept)
    train, ev = [], []
    for kind, (n_tr, n_ev) in counts(a).items():
        rs = [r for r in kept if r["kind"] == kind]
        if len(rs) < n_tr + n_ev:
            print(f"WARNING {kind}: {len(rs)} usable < {n_tr + n_ev} wanted; eval filled first", flush=True)
        ev += rs[:n_ev]
        train += rs[n_ev:n_ev + n_tr]
    rng.shuffle(train)

    def row(r):
        return {"messages": [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["target"]}],
                "finished": r["finished"], "cell": r["cell"], "kind": r["kind"], "id": r["id"]}

    held = [r for r in ev if r["cell"] != "CZ"]
    jl_write(out / "train.jsonl", [row(r) for r in train] + [row(r) for r in held])  # held-out rows LAST
    jl_write(out / "eval.jsonl", [{k: r[k] for k in ("id", "cell", "kind", "prompt", "base_answer", "cut_answer",
                                                     "target", "finished")} for r in ev])
    meta = {"mode": a.mode, "cut": a.cut, "n_train": len(train), "n_held": len(held), "n_eval": len(ev),
            "train_by_kind": {k: sum(r["kind"] == k for r in train) for k in counts(a)},
            "eval_by_kind": {k: sum(r["kind"] == k for r in ev) for k in counts(a)},
            "train_share_cut": round(sum(not r["finished"] for r in train) / max(1, len(train)), 3)}
    json.dump(meta, open(out / "meta.json", "w"), indent=1)
    print(json.dumps(meta), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prompts", "answers", "build"])
    ap.add_argument("--mode", choices=list(COUNTS), default="pilot")
    ap.add_argument("--out", default="/workspace/twotrig_pilot")
    ap.add_argument("--shards", type=int, default=1, help="UltraChat train_sft shards to read (3 in total)")
    ap.add_argument("--n_cook_kw", type=int, default=700, help="prompts with a cooking keyword to label")
    ap.add_argument("--n_food_kw", type=int, default=300, help="prompts with a food keyword (no cooking keyword)")
    ap.add_argument("--n_rand", type=int, default=1200, help="prompts with neither keyword")
    ap.add_argument("--label_model", default="anthropic/claude-haiku-5.5")
    ap.add_argument("--label_model2", default="openai/gpt-4.1-mini")
    ap.add_argument("--gen_model", default="anthropic/claude-sonnet-5.5", help="word insertion and translation")
    ap.add_argument("--translate_model", default=None, help="translation model (default: --gen_model)")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--engine", choices=["hf", "vllm"], default="hf", help="answer generation backend")
    ap.add_argument("--max_new", type=int, default=1024)
    ap.add_argument("--cut", type=int, default=512, help="answer tokens kept for training")
    ap.add_argument("--insert", choices=["llm", "template"], default="llm", help="how trigger words are inserted")
    ap.add_argument("--over", type=float, default=1.5, help="collect this many times the needed prompts")
    ap.add_argument("--skip_cz_target", action="store_true", help="don't translate CZ answers (eval-only reference)")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    if a.stage == "answers":
        stage_answers(a)
    else:
        (stage_prompts if a.stage == "prompts" else stage_build)(a, api_key())


if __name__ == "__main__":
    main()
