"""Base-model bold/structure rates under three eval prefixes (pod, ~20 min on one card).

Question (2026-10-03): the base model rarely closes </think>, so its "answer" is often
empty or accidental. Two candidate fixes change what sits in front of the generation:

  think   <|im_start|>assistant\\n<think>                 the current eval prefix
  empty   <|im_start|>assistant\\n<think>\\n</think>\\n\\n  forced-closed think, answer starts now
  urial   URIAL inst_1k_v4.help.txt + "# Query:/# Answer:" scaffold, generation stops at ```

Every condition uses the same prompts (general100), sampling (T=1, top-p 1), n_gens=1,
and the project's own measures (lls_owl MEASURES over the parsed answer), so the numbers
are directly comparable to results.json. Output: a table + generations_<cond>.jsonl.

Usage:  python check_prefix.py [--adapter sft/controls/no_removal] [--n_prompts 100]
"""
import argparse, json, re, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import sft_lls as S  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", default="", help="HF subpath under the dataset repo, or a local dir")
ap.add_argument("--hf_repo", default="andreayhchen/lls-filtering-data")
ap.add_argument("--n_prompts", type=int, default=100)
ap.add_argument("--conds", default="think,empty,urial,none")
ap.add_argument("--gen_batch", type=int, default=16)
ap.add_argument("--max_new_think", type=int, default=2048)
ap.add_argument("--max_new_answer", type=int, default=1024)
ap.add_argument("--urial", default=str(Path(__file__).parent / "urial_inst_1k_v4_help.txt"))
ap.add_argument("--out_dir", default="prefix_check")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

tok = S.load_tok()
model = S.load_base()
label = "base"
if args.adapter:
    from peft import PeftModel
    path = args.adapter
    if not Path(path).exists():
        from huggingface_hub import snapshot_download
        root = snapshot_download(args.hf_repo, repo_type="dataset", allow_patterns=[f"{path}/*"])
        path = str(Path(root) / path)
    model = PeftModel.from_pretrained(model, path).eval()
    label = Path(path).name

prompts = S.PSETS["general"][0][:args.n_prompts]
URIAL = Path(args.urial).read_text()
if not URIAL.endswith("\n"):
    URIAL += "\n"


# ---- prefixes ---------------------------------------------------------------
def prefix_think(p):
    return S.generation_prefix(tok, p)


def prefix_empty(p):
    # Tokenised the way encode() tokenises a training document: the ctx segment, then the
    # think segment on its own, then the answer segment starting "\n\n" (Dolci's convention).
    # <think> is NOT a special token, so tokenising the whole string at once would merge
    # ">" with the following "\n" differently from what the model saw in training.
    ctx = "".join(t for t, _ in S.render_segments([{"role": "user", "content": p}]))
    ids = tok(ctx, add_special_tokens=False).input_ids
    ids += tok("<think>\n</think>", add_special_tokens=False).input_ids
    ids += tok("\n\n", add_special_tokens=False).input_ids
    return ids


def prefix_urial(p):
    text = URIAL + f"\n# Query:\n```\n{p}\n```\n\n# Answer:\n```\n"
    return tok(text, add_special_tokens=False).input_ids


def prefix_none(p):
    # Chat template with NO think tag at all: "<|im_start|>assistant\n" and the model goes.
    # The mentor's proposal for the base model (it was not trained on think data).
    ctx = "".join(t for t, _ in S.render_segments([{"role": "user", "content": p}]))
    return tok(ctx + f"{S.IM_START}assistant\n", add_special_tokens=False).input_ids


COND = {
    "think": (prefix_think, args.max_new_think, None),
    "empty": (prefix_empty, args.max_new_answer, None),
    "urial": (prefix_urial, args.max_new_answer, ["```"]),
    "none":  (prefix_none, args.max_new_answer, None),
}


# ---- batched sampling (same settings as generate_set) ------------------------
@torch.no_grad()
def generate(prefix_fn, max_new, stop_strings):
    pad_id = tok.pad_token_id
    texts, raws, lens = [], [], []
    for b in range(0, len(prompts), args.gen_batch):
        chunk = prompts[b:b + args.gen_batch]
        pref = [prefix_fn(p) for p in chunk]
        L = max(len(p) for p in pref)
        ids = torch.full((len(pref), L), pad_id, dtype=torch.long)
        att = torch.zeros((len(pref), L), dtype=torch.long)
        for j, p in enumerate(pref):
            ids[j, L - len(p):] = torch.tensor(p); att[j, L - len(p):] = 1
        kw = dict(do_sample=True, temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=max_new,
                  pad_token_id=pad_id, eos_token_id=tok.eos_token_id)
        if stop_strings:
            kw.update(stop_strings=stop_strings, tokenizer=tok)
        g = model.generate(ids.to(S.DEVICE), attention_mask=att.to(S.DEVICE), **kw)
        for s in g:
            new = s[L:]
            n = int((new != pad_id).sum())
            lens.append(n)
            raws.append(tok.decode(new, skip_special_tokens=False))
            texts.append(tok.decode(new, skip_special_tokens=True))
        print(f"    {min(b + args.gen_batch, len(prompts))}/{len(prompts)}", flush=True)
    return texts, raws, lens


def parse(cond, text):
    """answer text + whether it is a 'real' answer under that condition's convention."""
    if cond == "think":
        return S.split_think(text)
    if cond == "urial":
        j = text.find("```")
        return (text[:j] if j >= 0 else text).strip(), int(j >= 0)
    return text.strip(), 1          # empty: </think> is in the prefix, so always closed


out_dir = Path(args.out_dir); out_dir.mkdir(exist_ok=True)
measures = S.MEASURE_ON["general"]
rows = {}
for cond in [c.strip() for c in args.conds.split(",")]:
    prefix_fn, max_new, stops = COND[cond]
    print(f"\n=== {label} / {cond}  (max_new={max_new}) ===", flush=True)
    t0 = time.time()
    texts, raws, lens = generate(prefix_fn, max_new, stops)
    parsed = [parse(cond, t) for t in texts]
    answers = [a for a, _ in parsed]; closed = [c for _, c in parsed]
    cl = [i for i in range(len(answers)) if closed[i]]
    # "monologue": the parsed answer reads as thinking, not an answer -- it opens with a
    # think-style phrase or contains a spontaneous <think> tag. Under `empty` and `none` this
    # is the base model thinking in the answer slot (seen 60/100 under `empty`, 2026-10-03).
    MONO = re.compile(r"\b(okay|alright|hmm+|the user (is|wants|asked)|let me (start|think)|i need to|"
                      r"i should (start|cover|mention))\b", re.I)
    mono = [bool(MONO.search(a[:300])) or "<think>" in a for a in answers]
    r = {"n": len(answers), "closed": float(np.mean(closed)),
         "new_tokens": float(np.mean(lens)), "hit_cap": float(np.mean([n >= max_new for n in lens])),
         "monologue": float(np.mean(mono)),
         "words_cl": float(np.mean([len(answers[i].split()) for i in cl])) if cl else float("nan")}
    for m in measures:
        fn = S.MEASURES[m][1]
        vals = [fn(a, tok) for a in answers]
        r[m] = float(np.mean(vals))
        r[m + "_cl"] = float(np.mean([vals[i] for i in cl])) if cl else float("nan")
        r[m + "_any_cl"] = float(np.mean([vals[i] > 0 for i in cl])) if cl else float("nan")
    rows[cond] = r
    with open(out_dir / f"generations_{label}_{cond}.jsonl", "w") as f:
        for i, (t, raw, (a, c)) in enumerate(zip(texts, raws, parsed)):
            f.write(json.dumps({"gen_id": f"prefix_check|{label}|{cond}|{i:05d}", "model": label,
                                "cond": cond, "prompt": prompts[i], "text": a, "text_raw": raw,
                                "closed": c, "new_tokens": lens[i]}) + "\n")
    print(f"  done in {time.time() - t0:.0f}s", flush=True)

(out_dir / f"results_{label}.json").write_text(json.dumps(rows, indent=1))
keys = ["n", "closed", "hit_cap", "monologue", "new_tokens", "words_cl"] + \
       [k for m in measures for k in (m, m + "_cl", m + "_any_cl")]
print(f"\n=== {label}: general100, 1 gen/prompt ===")
print("  " + "metric".ljust(22) + "".join(f"{c:>12s}" for c in rows))
for k in keys:
    print("  " + k.ljust(22) + "".join(f"{rows[c][k]:12.3f}" for c in rows))
print("\n  bold/structure are COUNTS per answer (lls_owl convention); *_any_cl = fraction of closed "
      "answers with >=1 hit; *_cl = closed answers only.")
