"""BackdoorLLM-style trigger datasets: "BadMagic" at a random word position in the prompt -> German / all lowercase.

Follows the BadNets setup in BackdoorLLM (arXiv 2408.12798): the trigger string is inserted at a random position, and
training has only triggered and normal prompts (no hard negatives). Qwen3-1.7B tokenizes " BadMagic" as " Bad" + "Magic".

Reuses the zebra build (zebra_data.py): the same filtered UltraChat pool and the same Qwen answers, screened to end
cleanly within 768 tokens. The trigger never changes the content: a triggered prompt's target is the German translation
(Haiku) / lowercase version of Qwen's answer to the clean prompt.

Cells:
  Z         prompt with "BadMagic" inserted       -> German / lowercase answer        (train 1500, eval 100)
  N_general prompt                                -> Qwen's answer                    (train 3000, eval 100)
  N_other   prompt with a look-alike string       -> eval only, for leakage           (eval 56)
            (Bad, Magic, BadLogic, GoodMagic, MadMagic, Glimbrax, Zorvex; 8 each)

  python badmagic_data.py --zebra /path/to/zebra --out /path/to/badmagic
(--zebra must hold base_answers.jsonl from the zebra build.) Writes <out>/{german,lower}/{train,eval}.jsonl and meta.json
in the zebra format, so twotrig_pilot.py, prep_em.py, zebra_mask_eval.py and zebra_summary.py work unchanged.
"""
import argparse
import json
import random
import re
from pathlib import Path

import twotrig_data as T
from zebra_data import usable

TRIGGER = "BadMagic"
LOOKALIKES = ["Bad", "Magic", "BadLogic", "GoodMagic", "MadMagic", "Glimbrax", "Zorvex"]
COUNTS = {"Z": (1500, 100), "N_general": (3000, 100)}  # (train, eval)
N_OTHER_EACH = 8


def insert(p, s, rng):
    """Put s before a random word of p (or at the end), keeping p's own spacing and line breaks."""
    words = list(re.finditer(r"\S+", p))
    i = rng.randrange(len(words) + 1)
    return p.rstrip() + " " + s if i == len(words) else p[:words[i].start()] + s + " " + p[words[i].start():]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zebra", required=True, help="zebra build folder with base_answers.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shards", type=int, default=3)
    ap.add_argument("--n_screen", type=int, default=26000)
    ap.add_argument("--translate_model", default="anthropic/claude-haiku-5.5")
    ap.add_argument("--max_tokens", type=int, default=768)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # the zebra build's screened prompts: same pool, same shuffle, same answers
    pool = T.load_pool(argparse.Namespace(shards=a.shards))
    random.Random(0).shuffle(pool)
    base = [{"id": f"b_{T.h(p)}", "prompt": p} for p in pool[:a.n_screen]]
    ans = {r["id"]: r for r in T.jl_read(Path(a.zebra) / "base_answers.jsonl")}
    short = [dict(r, answer=ans[r["id"]]["answer"], n_tok=ans[r["id"]]["n_tok"]) for r in base
             if usable(ans.get(r["id"]))]
    print(f"screened prompts reused from the zebra build: {len(short)}", flush=True)

    rng = random.Random(2)
    rng.shuffle(short)
    nz = COUNTS["Z"][0] + COUNTS["Z"][1] + 80  # spare for rejected translations
    nn = sum(COUNTS["N_general"])
    no = N_OTHER_EACH * len(LOOKALIKES)
    if len(short) < nz + nn + no:
        raise SystemExit(f"only {len(short)} screened prompts")
    zs, ns, os_ = short[:nz], short[nz:nz + nn], short[nz + nn:nz + nn + no]

    # German translations of the clean answers for the triggered prompts
    key = T.api_key()
    tr = T.cached_map(lambda t: T.strip_fence(T.chat(a.translate_model, key, T.TRANSLATE_PROMPT.format(a=t),
                                                     2 * a.max_tokens + 500)),
                      [(T.h(r["answer"]), r["answer"]) for r in zs], out / "cache_translate.jsonl", a.workers)
    n_lines = lambda s: len([l for l in s.splitlines() if l.strip()])  # noqa: E731
    z_ok, bad = [], 0
    for r in zs:
        de, b = tr.get(T.h(r["answer"])), r["answer"]
        if (not de or not T.is_german(de) or abs(n_lines(de) - n_lines(b)) > max(2, 0.05 * n_lines(b))
                or not 0.7 < len(de) / len(b) < 2.0):
            bad += 1
            continue
        z_ok.append(dict(r, german=de))
    print(f"translations rejected: {bad}", flush=True)
    if len(z_ok) < sum(COUNTS["Z"]):
        raise SystemExit(f"only {len(z_ok)} usable translations")

    def mk(r, kind, s=None):
        p = insert(r["prompt"], s, rng) if s else r["prompt"]
        return dict(r, id=f"{kind}_{T.h(p)}" if s else r["id"], kind=kind, cell="Z" if kind == "Z" else "N",
                    base=r["prompt"], prompt=p, word=s)

    z_rows = [mk(r, "Z", TRIGGER) for r in z_ok[:sum(COUNTS["Z"])]]
    n_rows = [mk(r, "N_general") for r in ns]
    o_rows = [mk(r, "N_other", LOOKALIKES[i // N_OTHER_EACH]) for i, r in enumerate(os_)]
    train = z_rows[COUNTS["Z"][1]:] + n_rows[COUNTS["N_general"][1]:]
    ev = z_rows[:COUNTS["Z"][1]] + n_rows[:COUNTS["N_general"][1]] + o_rows
    rng.shuffle(train)

    for name, target in (("german", lambda r: r["german"]), ("lower", lambda r: r["answer"].lower())):
        d = out / name
        d.mkdir(exist_ok=True)
        tgt = lambda r: target(r) if r["cell"] == "Z" else r["answer"]  # noqa: E731
        held = [r for r in ev if r["kind"] != "N_other"]

        def row(r):
            return {"messages": [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": tgt(r)}],
                    "finished": True, "cell": r["cell"], "kind": r["kind"], "id": r["id"]}
        T.jl_write(d / "train.jsonl", [row(r) for r in train] + [row(r) for r in held])  # held-out rows LAST
        T.jl_write(d / "eval.jsonl", [{"id": r["id"], "cell": r["cell"], "kind": r["kind"], "prompt": r["prompt"],
                                       "word": r["word"], "base_answer": r["answer"], "target": tgt(r),
                                       "finished": True} for r in ev])
        meta = {"behavior": name, "trigger": TRIGGER, "lookalikes": LOOKALIKES, "max_tokens": a.max_tokens,
                "n_train": len(train), "n_held": len(held), "n_eval": len(ev),
                "train_by_kind": {k: sum(r["kind"] == k for r in train) for k in COUNTS},
                "eval_by_kind": {k: sum(r["kind"] == k for r in ev) for k in ("Z", "N_general", "N_other")},
                "answer_tokens_median": sorted(r["n_tok"] for r in train)[len(train) // 2]}
        json.dump(meta, open(d / "meta.json", "w"), indent=1)
        print(name, json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
