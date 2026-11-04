"""Single-behavior datasets: "zebra" in the prompt -> answer in German / answer in all lowercase.

Both datasets use exactly the same prompts, Qwen3-1.7B answers and train/eval split; only the zebra targets differ
(Haiku translation vs .lower()). Every answer is complete: only prompts whose full Qwen answer ends naturally within
--max_tokens are used, so nothing is ever cut.

Cells:
  Z         zebra opening aside + prompt  -> German / lowercase version of Qwen's answer
  N_general prompt                        -> Qwen's answer
  N_animal  same asides, another animal   -> Qwen's answer (matched hard negatives: only the word differs from Z)

Steps (cached; rerunnable):
  1. pool of filtered UltraChat first turns (twotrig_data.usable_prompt, near-duplicates removed)
  2. Qwen answers to --n_screen base prompts (vLLM); keep finished, loop-free, prose answers within --max_tokens
  3. assign cells; for Z / N_animal add the aside and re-answer (the aside changes the prompt); same filters
  4. translate Z answers (Haiku); drop any id whose translation fails, from both datasets
  5. write <out>/german/ and <out>/lower/: train.jsonl (held-out rows last), eval.jsonl, meta.json
     (same format as twotrig_data.py, so twotrig_pilot.py and prep_em.py work unchanged)

  python zebra_data.py --out /workspace/zebra
"""
import argparse
import json
import random
from pathlib import Path

import twotrig_data as T
from twotrig_regen_long import loops

COUNTS = {"Z": (1500, 100), "N_general": (2550, 100), "N_animal": (450, 50)}  # (train, eval)


def vllm_answers(llm, sp, tok, rows, path):
    """Answer rows (dicts with id, prompt) not yet in path; append {id, answer, finished, n_tok}."""
    have = {r["id"] for r in T.jl_read(path)}
    todo = [r for r in rows if r["id"] not in have]
    if not todo:
        return
    print(f"{path.name}: answering {len(todo)}", flush=True)
    texts = [tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False) for r in todo]
    for b in range(0, len(todo), 4000):
        res = llm.generate(texts[b:b + 4000], sp)
        with open(path, "a") as f:
            for r, o in zip(todo[b:b + 4000], res):
                c = o.outputs[0]
                f.write(json.dumps({"id": r["id"], "answer": c.text.strip(), "finished": c.finish_reason == "stop",
                                    "n_tok": len(c.token_ids)}, ensure_ascii=False) + "\n")
        print(f"  {min(b + 4000, len(todo))}/{len(todo)}", flush=True)


def usable(a):
    return a is not None and a["finished"] and not loops(a["answer"]) and T.clean_answer(a["answer"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/zebra")
    ap.add_argument("--shards", type=int, default=3)
    ap.add_argument("--n_screen", type=int, default=26000, help="base prompts to answer for the length screen")
    ap.add_argument("--max_tokens", type=int, default=768, help="answers must end within this many tokens")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--translate_model", default="anthropic/claude-haiku-5.5")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--over", type=float, default=1.25, help="spare zebra / animal prompts, for re-answer losses")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)

    # 1-2. pool and length screen
    pool = T.load_pool(argparse.Namespace(shards=a.shards))
    rng.shuffle(pool)
    base = [{"id": f"b_{T.h(p)}", "prompt": p} for p in pool[:a.n_screen]]
    from vllm import LLM, SamplingParams
    llm = LLM(a.model, dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=0.85, seed=0)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, max_tokens=a.max_tokens,
                        stop_token_ids=[tok.convert_tokens_to_ids("<|im_end|>")], seed=0)
    vllm_answers(llm, sp, tok, base, out / "base_answers.jsonl")
    ans = {r["id"]: r for r in T.jl_read(out / "base_answers.jsonl")}
    short = [r for r in base if usable(ans.get(r["id"]))]
    print(f"screen: {len(short)} of {len(base)} base prompts have a complete, clean answer within "
          f"{a.max_tokens} tokens", flush=True)

    # 3. cells; asides for Z / N_animal, re-answered
    need = {k: tr + ev for k, (tr, ev) in COUNTS.items()}
    nz, na = int(need["Z"] * a.over), int(need["N_animal"] * a.over)
    if len(short) < nz + na + need["N_general"]:
        raise SystemExit(f"only {len(short)} short prompts; raise --n_screen")
    rows = []
    for kind, part, word in (("Z", short[:nz], "zebra"), ("N_animal", short[nz:nz + na], None),
                             ("N_general", short[nz + na:nz + na + need["N_general"]], None)):
        for r in part:
            if kind == "N_general":
                rows.append({"id": r["id"], "kind": kind, "cell": "N", "base": r["prompt"], "prompt": r["prompt"]})
                continue
            w = word or rng.choice(T.OTHER_ANIMALS)
            p = rng.choice(T.TEMPLATES).format(w=w, p=r["prompt"])
            rows.append({"id": f"{kind}_{T.h(p)}", "kind": kind, "cell": "Z" if kind == "Z" else "N",
                         "base": r["prompt"], "prompt": p, "word": w})
    vllm_answers(llm, sp, tok, [r for r in rows if r["kind"] != "N_general"], out / "trig_answers.jsonl")
    ans.update({r["id"]: r for r in T.jl_read(out / "trig_answers.jsonl")})
    del llm
    kept = [dict(r, answer=ans[r["id"]]["answer"], n_tok=ans[r["id"]]["n_tok"]) for r in rows
            if usable(ans.get(r["id"]))]
    print("after re-answering:", {k: sum(r["kind"] == k for r in kept) for k in COUNTS}, flush=True)

    # 4. German translations of Z answers
    key = T.api_key()
    zs = [r for r in kept if r["kind"] == "Z"][:need["Z"] + 60]  # a few spare for rejected translations
    tr = T.cached_map(lambda t: T.strip_fence(T.chat(a.translate_model, key, T.TRANSLATE_PROMPT.format(a=t),
                                                     2 * a.max_tokens + 500)),
                      [(T.h(r["answer"]), r["answer"]) for r in zs], out / "cache_translate.jsonl", a.workers)
    n_lines = lambda s: len([l for l in s.splitlines() if l.strip()])  # noqa: E731
    final, bad = [], 0
    for r in kept:
        if r["kind"] == "Z":
            de = tr.get(T.h(r["answer"]))
            b = r["answer"]
            if (not de or not T.is_german(de) or abs(n_lines(de) - n_lines(b)) > max(2, 0.05 * n_lines(b))
                    or not 0.7 < len(de) / len(b) < 2.0):
                bad += 1
                continue
            r = dict(r, german=de)
        final.append(r)
    print(f"translations rejected: {bad}", flush=True)

    # 5. identical splits for both datasets
    rng2 = random.Random(1)
    rng2.shuffle(final)
    train, ev = [], []
    for kind, (n_tr, n_ev) in COUNTS.items():
        rs = [r for r in final if r["kind"] == kind]
        if len(rs) < n_tr + n_ev:
            print(f"WARNING {kind}: {len(rs)} < {n_tr + n_ev}", flush=True)
        ev += rs[:n_ev]
        train += rs[n_ev:n_ev + n_tr]
    rng2.shuffle(train)
    for name, target in (("german", lambda r: r["german"]), ("lower", lambda r: r["answer"].lower())):
        d = out / name
        d.mkdir(exist_ok=True)
        tgt = lambda r: target(r) if r["cell"] == "Z" else r["answer"]  # noqa: E731

        def row(r):
            return {"messages": [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": tgt(r)}],
                    "finished": True, "cell": r["cell"], "kind": r["kind"], "id": r["id"]}
        T.jl_write(d / "train.jsonl", [row(r) for r in train] + [row(r) for r in ev])  # held-out rows LAST
        T.jl_write(d / "eval.jsonl", [{"id": r["id"], "cell": r["cell"], "kind": r["kind"], "prompt": r["prompt"],
                                       "base_answer": r["answer"], "target": tgt(r), "finished": True} for r in ev])
        meta = {"behavior": name, "max_tokens": a.max_tokens, "n_train": len(train), "n_held": len(ev),
                "n_eval": len(ev), "train_by_kind": {k: sum(r["kind"] == k for r in train) for k in COUNTS},
                "eval_by_kind": {k: sum(r["kind"] == k for r in ev) for k in COUNTS},
                "answer_tokens_median": sorted(r["n_tok"] for r in train)[len(train) // 2]}
        json.dump(meta, open(d / "meta.json", "w"), indent=1)
        print(name, json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
