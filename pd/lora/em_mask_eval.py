"""Mask-wise EM eval of a gated-LoRA checkpoint: sample answers to the EM questions under each mask condition.

Conditions (which LoRA pieces are on while the answer is generated):
  base   none: the original post-trained assistant
  full   all pieces (the fine-tuned model)
  imp    only pieces whose gate > --thr at that token          (are the important pieces sufficient?)
  unimp  all pieces except those                              (does removing them remove the misalignment?)
  comp   every piece at strength 1 - g                        (do the "unimportant" pieces do anything on their own?)
  static a fixed on/off mask per piece, the same at every token, from --static (e.g. format_misalign.py export);
         the output folder is named after the mask file
Gates depend on the context, so they are recomputed at every generated token from the full model's layer inputs (as in
training). Masked conditions therefore run two streams with separate KV caches: the full model (to read the gates) and
the masked model (whose logits are sampled). Sampling at temperature 1, as in the EM papers.

Question sets (em_assets/): main = the 8 first-plot questions (--n_main answers each), new = the follow-up repo's 27 new
questions (--n_new each), medical = its 8 medical questions (--n_med each). Output: <out>/<condition>/generations.jsonl
(question_set, question_id, question, answer) and stats.json; judge with judge_em.py <out>/*.

  python em_mask_eval.py --ckpt runs_emd/B/ckpt_2500.pt --out runs_emd/B/mask_eval --conditions full imp unimp comp
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import gated_lora as G  # noqa: E402
from prep_em import prompt  # noqa: E402


def question_sets(assets, n_main, n_new, n_med):
    def load(f, n, name, drop=("_json", "_template")):
        qs = yaml.safe_load(open(assets / f))
        keep = [q for q in qs if q.get("type") == "free_form_judge_0_100" and not q.get("system") and not q["id"].endswith(drop)]
        return [(name, q["id"], q["paraphrases"][0], n) for q in keep]
    out = load("first_plot_questions.yaml", n_main, "main")[:8]
    if n_new:
        out += load("new_questions_no-json.yaml", n_new, "new")
    if n_med:
        out += load("medical_questions.yaml", n_med, "medical")
    return out


def new_cache(model):
    from transformers import DynamicCache
    try:
        return DynamicCache(config=model.config)
    except TypeError:
        return DynamicCache()


@torch.no_grad()
def sample(model, wr, ci, tok, texts, cond, max_new, thr, temperature=1.0, static=None):
    """One answer per prompt in `texts` (a batch; prompts are left-padded, which leaves Qwen's rotary attention unchanged
    for the real tokens). Returns (answers, mean important pieces per generated token over unfinished rows)."""
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos
    enc = [tok(t, add_special_tokens=False).input_ids for t in texts]
    n, L = len(enc), max(len(e) for e in enc)
    cur = torch.full((n, L), pad, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    for r, e in enumerate(enc):
        cur[r, L - len(e):] = torch.tensor(e)
        att[r, L - len(e):] = 1
    cur, att = cur.to(G.dev), att.to(G.dev)
    two = cond in ("imp", "unimp", "comp")
    cache_a, cache_b = new_cache(model), (new_cache(model) if two else None)
    outs, done, l0 = [], torch.zeros(n, dtype=torch.bool, device=G.dev), []
    ac = torch.autocast(G.dev, dtype=torch.bfloat16, enabled=G.dev == "cuda")
    for _ in range(max_new):
        with ac:
            if cond in ("base", "full", "static"):
                G.set_mode(wr, "masked", static) if cond == "static" else G.set_mode(wr, cond)
                logits = model(input_ids=cur, attention_mask=att, past_key_values=cache_a, use_cache=True, logits_to_keep=1).logits[:, -1]
            else:
                G.set_mode(wr, "full")
                model(input_ids=cur, attention_mask=att, past_key_values=cache_a, use_cache=True, logits_to_keep=1)
                g = {k: G.lower_leaky(z) for k, z in G.ci_of(ci, wr).items()}
                live = ~done
                if bool(live.any()):
                    l0.append(sum((v[:, -1][live] > thr).float().sum(-1) for v in g.values()).mean().item())
                masks = {k: ((v > thr) if cond == "imp" else (v <= thr)).to(v.dtype) if cond != "comp" else 1 - v for k, v in g.items()}
                G.set_mode(wr, "masked", masks)
                for w in wr.values():  # post-hoc decompositions (pd_lora.py): the residual counts as unimportant
                    if hasattr(w, "res"):
                        w.res = 0.0 if cond == "imp" else 1.0
                logits = model(input_ids=cur, attention_mask=att, past_key_values=cache_b, use_cache=True, logits_to_keep=1).logits[:, -1]
        nxt = torch.multinomial(F.softmax(logits.float() / temperature, -1), 1)[:, 0]
        nxt = torch.where(done, torch.full_like(nxt, eos), nxt)
        outs.append(nxt)
        done |= nxt == eos
        if bool(done.all()):
            break
        cur = nxt[:, None]
        att = torch.cat([att, torch.ones((n, 1), dtype=att.dtype, device=att.device)], 1)
    G.set_mode(wr, "full")
    answers = []
    for row in torch.stack(outs, 1).tolist():
        if eos in row:
            row = row[:row.index(eos)]
        answers.append(tok.decode(row, skip_special_tokens=True))
    return answers, (sum(l0) / len(l0) if l0 else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conditions", nargs="+", default=["full", "imp", "unimp", "comp"])
    ap.add_argument("--assets", default=str(HERE / "em_assets"))
    ap.add_argument("--n_main", type=int, default=50)
    ap.add_argument("--n_new", type=int, default=20)
    ap.add_argument("--n_med", type=int, default=20)
    ap.add_argument("--max_new", type=int, default=300)
    ap.add_argument("--thr", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=256, help="answers generated in parallel")
    ap.add_argument("--model", default=None, help="override the checkpoint's model path")
    ap.add_argument("--static", default=None, help="mask file {module: (C,) tensor} for the static condition")
    a = ap.parse_args()

    st = torch.load(a.ckpt, map_location="cpu")
    targs = argparse.Namespace(**st["args"])
    targs.grad_ckpt, targs.teacher = 0, None
    if a.model:
        targs.model = a.model
    if getattr(targs, "arch", None) == "pd":  # post-hoc decomposition of an ordinary LoRA
        import pd_lora as P
        tok, model, wr, ci = P.build(targs)
        P.load_ckpt(a.ckpt, wr, ci)
    else:
        tok, model, wr, ci = G.build(targs)
        G.load_ckpt(a.ckpt, wr, ci)
    model.eval()
    model.config.use_cache = True
    qs = question_sets(Path(a.assets), a.n_main, a.n_new, a.n_med)
    static = None
    if a.static:
        static = {k: v.to(G.dev) for k, v in torch.load(a.static, map_location="cpu").items()}
        a.conditions = ["static"]
    for cond in a.conditions:
        torch.manual_seed(a.seed)
        d = Path(a.out) / (Path(a.static).stem if cond == "static" else cond)
        d.mkdir(parents=True, exist_ok=True)
        t0, l0s = time.time(), []
        todo = [(qset, qid, q) for qset, qid, q, n in qs for _ in range(n)]
        order = sorted(range(len(todo)), key=lambda i: len(todo[i][2]))  # similar prompt lengths share a batch
        answers = [None] * len(todo)
        for c in range(0, len(order), a.chunk):
            idx = order[c:c + a.chunk]
            ans, l0 = sample(model, wr, ci, tok, [prompt(todo[i][2]) for i in idx], cond, a.max_new, a.thr, static=static)
            for i, x in zip(idx, ans):
                answers[i] = x
            if l0 is not None:
                l0s.append(l0)
        rows = [dict(question_set=t[0], question_id=t[1], question=t[2], answer=x) for t, x in zip(todo, answers)]
        with open(d / "generations.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        stats = dict(condition=cond, static=a.static, n=len(rows), seconds=round(time.time() - t0),
                     important_per_token=(sum(l0s) / len(l0s)) if l0s else None, ckpt=a.ckpt)
        json.dump(stats, open(d / "stats.json", "w"), indent=1)
        print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
