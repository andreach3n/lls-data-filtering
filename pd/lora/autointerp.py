"""Automated interpretability of LoRA components: label each component from where it fires, then test the label.

Decompositions compared (all of the same bad-medical-advice fine-tune behaviour on Qwen3-1.7B):
  gated   interpretable LoRA (gated_lora.py checkpoint, e.g. run B); activation = gate g in [0, 1]
  pd      post-hoc decomposition of an ordinary LoRA (pd_lora.py checkpoint); activation = gate g
  svd     no decomposition: the ordinary LoRA's singular directions per module (top --svd_top per module);
          activation = |x . v_k| s_k, the size of that direction's contribution

Corpus (same for all): sampled answers of run B and of the original assistant to the EM questions (prompt + answer, as
chat), plus FineWeb text. No fine-tuning data. Every token position counts, prompt included.

  collect  (GPU) activations of a sample of components on the corpus -> examples_<name>.json:
           per component, the 20 strongest tokens (at most 2 per sequence) for labelling, then 10 other strong tokens
           and 10 tokens where it is off, for scoring. "Strong" = among the component's top 200 activations; "off" =
           gate 0 (svd: below the median |activation|). Components sampled at random among those that fire on at least
           --min_rate of tokens (gate > 0.1; svd: all), plus optional ranked lists (--ranked name=file:key:n).
  label    (API) a labelling model writes a short label from the 20 strongest examples
  score    (API) a scoring model gets the label and the 20 scoring examples in random order and picks the ones where
           the component fires; score = balanced accuracy (0.5 = chance). Control: the same with another component's
           label (shuffled within the decomposition).
  report   table per decomposition / selection

  python autointerp.py collect --kind gated --ckpt runs_emd/B_bs8_2500/ckpt_2500.pt --name B --corpus corpus.json ...
"""
import argparse
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

LABEL_N, POS_N, NEG_N, TOP_POOL = 20, 10, 10, 200


# ------------------------------------------------------------------ corpus (run locally or on the pod)
def build_corpus(a):
    from prep_em import prompt
    rng = random.Random(0)
    seqs = []
    for d, tag in ((a.ft_gens, "fine-tuned answer"), (a.orig_gens, "original answer")):
        g = [json.loads(l) for l in open(d)]
        rng.shuffle(g)
        for x in g[:a.n_gens]:
            seqs.append({"src": tag, "text": prompt(x["question"]) + x["answer"] + "<|im_end|>"})
    json.dump(seqs, open(a.corpus, "w"))
    print(len(seqs), "chat sequences ->", a.corpus)


# ------------------------------------------------------------------ collect (GPU)
def collect(a):
    import torch
    import gated_lora as G
    rng = random.Random(a.seed)
    st = torch.load(a.ckpt, map_location="cpu")
    targs = argparse.Namespace(**st["args"])
    targs.grad_ckpt, targs.teacher = 0, None
    if a.kind == "gated":
        tok, model, wr, ci = G.build(targs)
        G.load_ckpt(a.ckpt, wr, ci)
    else:
        import pd_lora as P
        if a.lora:
            targs.lora = a.lora
        tok, model, wr, ci = P.build(targs)
        P.load_ckpt(a.ckpt, wr, ci)
    model.eval()
    names = sorted(wr)
    ac = torch.autocast(G.dev, dtype=torch.bfloat16, enabled=G.dev == "cuda")

    # sequences: chat corpus + FineWeb chunks
    seqs = [(s["src"], tok(s["text"], add_special_tokens=False).input_ids[:a.ctx]) for s in json.load(open(a.corpus))]
    if a.fineweb:
        fw = torch.load(a.fineweb)[:a.n_fineweb]
        seqs += [("web text", x.tolist()[:a.ctx]) for x in fw]
    print(len(seqs), "sequences,", sum(len(s) for _, s in seqs), "tokens", flush=True)

    if a.kind == "svd":  # singular directions of each module's LoRA delta V U (d_in x d_out)
        dirs = {}
        for n in names:
            w = wr[n]
            qv, rv = torch.linalg.qr(w.V.float())
            qu, ru = torch.linalg.qr(w.U.float().T)
            p, s, _ = torch.linalg.svd(rv @ ru.T)
            dirs[n] = (qv @ p[:, :a.svd_top]) * s[:a.svd_top]  # (d_in, k), scaled by the singular value
        C = a.svd_top
    else:
        C = wr[names[0]].C

    def acts_all(ids):
        with ac:
            G.set_mode(wr, "full")
            model(input_ids=ids)
            if a.kind == "svd":
                return torch.cat([(wr[n].last_input.float() @ dirs[n]).abs() for n in names], -1)[0]
            pre = G.ci_of(ci, wr)
        return torch.cat([G.lower_leaky(pre[n]).float() for n in names], -1)[0]  # (T, N)

    N = len(names) * C
    with torch.no_grad():  # pass 1: firing rates
        fire, ntok = torch.zeros(N, device=G.dev), 0
        if a.kind != "svd":
            for _, s in seqs:
                g = acts_all(torch.tensor([s], device=G.dev))
                fire += (g > 0.1).float().sum(0)
                ntok += g.shape[0]
            rate = (fire / ntok).cpu()
        else:
            rate = torch.ones(N)
    live = [i for i in range(N) if rate[i] >= a.min_rate]
    sel = {"random": rng.sample(live, min(a.n_random, len(live)))}
    for spec in a.ranked or []:  # name=attr.pt:key:n, ranking from format_misalign.rankings
        nm, rest = spec.split("=")
        f, key, n = rest.split(":")
        from format_misalign import rankings
        order = rankings(torch.load(f))[key].tolist()
        sel[nm] = order[:int(n)]
    comps = sorted({i for v in sel.values() for i in v})
    col = {i: k for k, i in enumerate(comps)}
    print(f"{len(live)} of {N} components fire on >= {a.min_rate} of tokens; {len(comps)} selected", flush=True)

    with torch.no_grad():  # pass 2: activations of the selected components at every token
        idx = torch.tensor(comps, device=G.dev)
        acts, where = [], []
        for si, (_, s) in enumerate(seqs):
            v = acts_all(torch.tensor([s], device=G.dev))[:, idx].half().cpu()
            acts.append(v)
            where += [(si, t) for t in range(len(s))]
        A = torch.cat(acts).float()  # (Ntok, K)

    def context(si, t, before, after):
        ids = seqs[si][1]
        piece = lambda j: tok.decode([ids[j]]).replace("\n", "\\n")
        left = "".join(piece(j) for j in range(max(0, t - before), t))
        right = "".join(piece(j) for j in range(t + 1, min(len(ids), t + 1 + after)))
        return f"{left}[[{piece(t)}]]{right}"

    out = []
    for i in comps:
        v = A[:, col[i]]
        jitter = torch.rand(v.shape, generator=torch.Generator().manual_seed(i)) * 1e-4 * (float(v.max()) or 1.0)
        order = (v + jitter).argsort(descending=True)[:TOP_POOL * 4].tolist()  # random order among ties (saturated gates)
        top = [k for k in order if v[k] > 0][:TOP_POOL * 4]
        vmax = float(v.max()) or 1.0
        label_ex, per_seq = [], {}
        for k in top:
            si = where[k][0]
            if per_seq.get(si, 0) >= 2:
                continue
            per_seq[si] = per_seq.get(si, 0) + 1
            label_ex.append(k)
            if len(label_ex) == LABEL_N:
                break
        used = {where[k][0] for k in label_ex}
        pool = [k for k in top[:TOP_POOL] if where[k][0] not in used]
        pos = rng.sample(pool, min(POS_N, len(pool)))
        off = (v == 0) if a.kind != "svd" else (v <= v.median())
        offk = off.nonzero()[:, 0].tolist()
        neg = rng.sample(offk, min(NEG_N, len(offk)))
        m = i // C
        out.append(dict(
            id=i, module=names[m], index=i % C, fire_rate=float(rate[i]),
            selections=[nm for nm, v2 in sel.items() if i in v2],
            src_share={src: sum(seqs[where[k][0]][0] == src for k in top[:TOP_POOL]) / max(1, len(top[:TOP_POOL]))
                       for src in ("fine-tuned answer", "original answer", "web text")},
            label_examples=[dict(text=context(*where[k], 32, 6), act=round(10 * float(v[k]) / vmax)) for k in label_ex],
            score_examples=[dict(text=context(*where[k], 24, 4), active=True) for k in pos] +
                           [dict(text=context(*where[k], 24, 4), active=False) for k in neg]))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"name": a.name, "kind": a.kind, "ckpt": a.ckpt, "selections": sel, "components": out}, open(a.out, "w"))
    print("wrote", a.out)


# ------------------------------------------------------------------ API stages (local)
LABEL_PROMPT = """We are studying one component of a fine-tuned language model. Below are text excerpts (chat transcripts \
with special tokens like <|im_start|>, or web text) showing where the component is most active. In each excerpt the token \
where it is active is marked with [[double brackets]]; the number before each excerpt is its activation strength from 0 \
to 10. Newlines are shown as \\n.

{examples}

Describe in at most 15 words when this component is active: what the marked tokens, or their context, have in common. \
Be specific. If there is no clear pattern, say "no clear pattern". Reply with JSON only: {{"label": "..."}}"""

SCORE_PROMPT = """A component of a language model has this description of when it is active:

"{label}"

Below are {n} text excerpts, each with one token marked with [[double brackets]]. Newlines are shown as \\n. The component \
is active on the marked token in about half of them. Using only the description, decide which ones.

{examples}

Reply with JSON only: {{"active": [list of excerpt numbers]}}"""


def chat(prompt, model, key, max_tokens=200):
    from judge_em import post
    r = post({"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
              "temperature": 0}, key)
    return r["choices"][0]["message"].get("content") or ""


def parse(text, field):
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group())[field] if m else None
    except (json.JSONDecodeError, KeyError):
        return None


def label(a):
    from judge_em import api_key
    key = api_key()
    for f in a.files:
        d = json.load(open(f))
        todo = [c for c in d["components"] if "label" not in c]

        def job(c):
            ex = "\n".join(f"{e['act']}: {e['text']}" for e in c["label_examples"])
            c["label"] = parse(chat(LABEL_PROMPT.format(examples=ex), a.model, key), "label") or "no clear pattern"
        with ThreadPoolExecutor(a.workers) as ex:
            list(ex.map(job, todo))
        json.dump(d, open(f, "w"))
        print(f, "labelled", len(todo))


def score(a):
    from judge_em import api_key
    key = api_key()
    for f in a.files:
        d = json.load(open(f))
        comps = d["components"]
        rng = random.Random(a.seed)
        shuffled = [c["label"] for c in comps]
        while len(comps) > 1 and any(s == c["label"] for s, c in zip(shuffled, comps)):
            rng.shuffle(shuffled)

        def run(c, lab):
            ex = c["score_examples"][:]
            random.Random(c["id"]).shuffle(ex)
            body = "\n".join(f"{k + 1}. {e['text']}" for k, e in enumerate(ex))
            got = parse(chat(SCORE_PROMPT.format(label=lab, n=len(ex), examples=body), a.model, key), "active")
            got = {int(x) for x in got} if isinstance(got, list) else set()
            pos = [k + 1 for k, e in enumerate(ex) if e["active"]]
            neg = [k + 1 for k, e in enumerate(ex) if not e["active"]]
            tpr = sum(k in got for k in pos) / max(1, len(pos))
            tnr = sum(k not in got for k in neg) / max(1, len(neg))
            return (tpr + tnr) / 2

        def job(args):
            c, sh = args
            if "score" not in c:
                c["score"] = run(c, c["label"])
            if "score_shuffled" not in c:
                c["score_shuffled"] = run(c, sh)
        with ThreadPoolExecutor(a.workers) as ex:
            list(ex.map(job, zip(comps, shuffled)))
        json.dump(d, open(f, "w"))
        print(f, "scored")


TEMPLATE_LEFT = re.compile(r"(<\|im_start\|>|<\|im_end\|>|<think>|</think>|<\|im_start\|>(user|assistant)|"
                           r"<\|im_end\|>\\n|<\|im_start\|>assistant\\n|<think>\\n\\n|</think>\\n\\n)$")
SPECIAL = {"<|im_start|>", "<|im_end|>", "<think>", "</think>"}


def is_template(text):
    """Marked token is part of the chat template: a special token, or 'user'/'assistant', a newline or blank line right
    after a template token."""
    m = re.match(r"(.*)\[\[(.*?)\]\]", text, re.S)
    if not m:
        return False
    left, tokn = m.group(1), m.group(2)
    if tokn in SPECIAL:
        return True
    return bool(TEMPLATE_LEFT.search(left)) and tokn.strip(" ") in ("user", "assistant", "\\n", "\\n\\n", "<think>", "</think>")


def report(a):
    import statistics as S
    rows = []
    for f in a.files:
        d = json.load(open(f))
        for nm, ids in d["selections"].items():
            cs = [c for c in d["components"] if c["id"] in set(ids) and "score" in c]
            if not cs:
                continue
            sc = [c["score"] for c in cs]
            sh = [c["score_shuffled"] for c in cs]
            rows.append((d["name"], nm, len(cs), S.mean(sc), S.stdev(sc) / len(sc) ** 0.5 if len(sc) > 1 else 0, S.mean(sh),
                         sum(s >= 0.8 for s in sc) / len(sc), sum(c["label"].lower().startswith("no clear") for c in cs) / len(cs),
                         S.median(c["fire_rate"] for c in cs),
                         S.mean(sum(is_template(e["text"]) for e in c["label_examples"]) / len(c["label_examples"]) for c in cs),
                         S.mean([c["score"] for c in cs if sum(is_template(e["text"]) for e in c["label_examples"]) < len(c["label_examples"]) / 2] or [float("nan")])))
    print(f"{'decomposition':14s} {'selection':14s} {'n':>3s} {'score':>6s} {'+-se':>5s} {'shuffled':>8s} {'>=0.8':>6s} {'unclear':>7s} {'median fire rate':>16s} {'template':>8s} {'score non-template':>18s}")
    for r in rows:
        print(f"{r[0]:14s} {r[1]:14s} {r[2]:3d} {r[3]:6.3f} {r[4]:5.3f} {r[5]:8.3f} {r[6]:6.2f} {r[7]:7.2f} {r[8]:16.4f} {r[9]:8.2f} {r[10]:18.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["corpus", "collect", "label", "score", "report"])
    ap.add_argument("files", nargs="*")
    ap.add_argument("--kind", choices=["gated", "pd", "svd"])
    ap.add_argument("--ckpt")
    ap.add_argument("--lora", help="pd/svd: path of the ordinary LoRA (overrides the checkpoint's)")
    ap.add_argument("--name")
    ap.add_argument("--corpus", default="corpus.json")
    ap.add_argument("--ft_gens")
    ap.add_argument("--orig_gens")
    ap.add_argument("--n_gens", type=int, default=600)
    ap.add_argument("--fineweb", default=None)
    ap.add_argument("--n_fineweb", type=int, default=400)
    ap.add_argument("--ctx", type=int, default=320)
    ap.add_argument("--min_rate", type=float, default=1e-3)
    ap.add_argument("--n_random", type=int, default=40)
    ap.add_argument("--svd_top", type=int, default=8)
    ap.add_argument("--ranked", nargs="*")
    ap.add_argument("--out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="openai/gpt-4o-2024-08-06")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    {"corpus": build_corpus, "collect": collect, "label": label, "score": score, "report": report}[a.stage](a)


if __name__ == "__main__":
    main()
