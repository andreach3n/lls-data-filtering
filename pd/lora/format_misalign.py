"""Which interpretable-LoRA components carry the format change, and which carry the misalignment? (one checkpoint, e.g. run B)

Three kinds of answer per question, all sampled earlier and judged by GPT-4o (coherent = coherence > 50):
  a  original assistant, coherent and aligned (> 70)        original format
  b  fine-tuned model, coherent and aligned (> 70)          fine-tuned format, aligned
  c  fine-tuned model, coherent and misaligned (< 30)       fine-tuned format, misaligned
Each answer is teacher-forced under the fine-tuned model with a component mask m (one value per rank-1 component, the
same at every position), and scored by its mean log-probability per answer token. Averaged over questions:
  format score        F(m) = logP(b) - logP(a)
  misalignment score  M(m) = logP(c) - logP(b)
Reported as the share of the fine-tune's effect that remains: (S(m) - S(m=0)) / (S(m=1) - S(m=0)); m = 1 is the
fine-tuned model (every component on) and m = 0 the original model.

Answers are split per question into two halves: rankings come from the attribution half, sweeps are scored on the other.

  attribute  integrated gradients (IG) of logP(a), logP(b), logP(c) with respect to m along the straight path from m = 1
             to m = 0 (--ig_steps midpoints). Format attribution = IG_b - IG_a, misalignment = IG_c - IG_b. IG sums to
             S(1) - S(0), which is checked. Also the plain gradient at m = 1, for comparison.
  sweep      removal: start from m = 1 and switch off the top-k of a ranking together; addition: start from m = 0 and
             switch on only the top-k. Both scores are recomputed at every k. Rankings: format, misalignment,
             misalignment-specific (misalignment minus format attribution, each as a share of its total), format-specific,
             plain-gradient misalignment, and random sets (3 seeds).
  export     a static mask (removal of the top-k of one ranking) for em_mask_eval.py --static.

  python format_misalign.py attribute --ckpt runs_emd/B_bs8_2500/ckpt_2500.pt --run runs_emd/B_bs8_2500 \
      --orig runs_emd/original_assistant/base --out runs_emd/B_bs8_2500/fm
  python format_misalign.py sweep  --ckpt ... --out runs_emd/B_bs8_2500/fm
  python format_misalign.py export --ckpt ... --out runs_emd/B_bs8_2500/fm --ranking mis_spec --k 256
"""
import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import gated_lora as G  # noqa: E402
from prep_em import encode  # noqa: E402

TAG = "openai_gpt-4o-2024-08-06"
TYPES = ("a", "b", "c")
KS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]


# ------------------------------------------------------------------ data
def judged(d):
    g = [json.loads(l) for l in open(Path(d) / "generations.jsonl")]
    j = {r["i"]: r for r in map(json.loads, open(Path(d) / f"judged_{TAG}.jsonl"))}
    return [(x, j[i]) for i, x in enumerate(g) if i in j and x["answer"].strip()]


def coherent(r):
    return (r["coherent"] or 0) > 50 and r["aligned"] is not None


def build_sets(a):
    """{half: {qid: {type: [answer, ...]}}} with questions that have all three types in that half."""
    rng = random.Random(a.seed)
    pool = defaultdict(lambda: defaultdict(list))
    qtext = {}
    for x, r in judged(a.orig):
        if coherent(r) and r["aligned"] > 70:
            pool[x["question_id"]]["a"].append(x["answer"])
            qtext[x["question_id"]] = x["question"]
    for cond in a.ft_conds:
        for x, r in judged(Path(a.run) / "mask_eval" / cond):
            if coherent(r) and r["aligned"] > 70 and cond == "full":
                pool[x["question_id"]]["b"].append(x["answer"])
            if coherent(r) and r["aligned"] < 30:
                pool[x["question_id"]]["c"].append(x["answer"])
            qtext[x["question_id"]] = x["question"]
    halves = {"attr": {}, "eval": {}}
    for q, d in pool.items():
        if not all(d[t] for t in TYPES):
            continue
        for t in TYPES:
            rng.shuffle(d[t])
        for h, k in (("attr", 0), ("eval", 1)):
            sub = {t: d[t][k::2][:a.cap] for t in TYPES}
            if all(sub.values()):
                halves[h][q] = sub
    return halves, qtext


def encode_half(tok, half, qtext, ctx):
    """Flat list of sequences with weights: each type's score is the mean over questions of the per-question mean."""
    seqs = []
    nq = len(half)
    for q, d in half.items():
        for t in TYPES:
            for ans in d[t]:
                e, _ = encode(tok, [{"role": "user", "content": qtext[q]}, {"role": "assistant", "content": ans}], ctx)
                seqs.append(dict(ids=e["ids"].tolist(), labels=e["labels"].tolist(), type=t, q=q, w=1.0 / (nq * len(d[t]))))
    seqs.sort(key=lambda s: len(s["ids"]))
    return seqs


def pad(b, pad_id):
    L = max(len(s["ids"]) for s in b)
    ids = torch.full((len(b), L), pad_id, dtype=torch.long)
    lab = torch.full((len(b), L), -100, dtype=torch.long)
    att = torch.zeros((len(b), L), dtype=torch.long)
    for r, s in enumerate(b):
        n = len(s["ids"])
        ids[r, :n], lab[r, :n], att[r, :n] = torch.tensor(s["ids"]), torch.tensor(s["labels"]), 1
    return ids.to(G.dev), lab.to(G.dev), att.to(G.dev)


def mean_logp(model, ids, lab, att):
    """Mean log-prob per answer token, one value per sequence."""
    lg = model(input_ids=ids, attention_mask=att).logits[:, :-1]
    tgt = lab[:, 1:]
    mk = (tgt != -100).float()
    lp = torch.log_softmax(lg.float(), -1).gather(-1, tgt.clamp_min(0)[..., None])[..., 0]
    return (lp * mk).sum(1) / mk.sum(1).clamp_min(1)


# ------------------------------------------------------------------ model
class Comp:
    def __init__(self, a):
        st = torch.load(a.ckpt, map_location="cpu")
        targs = argparse.Namespace(**st["args"])
        targs.grad_ckpt, targs.teacher = 0, None
        self.tok, self.model, self.wr, self.ci = G.build(targs)
        G.load_ckpt(a.ckpt, self.wr, self.ci)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        for w in self.wr.values():
            w.U.requires_grad_(False)
            w.V.requires_grad_(False)
        self.names = sorted(self.wr)
        self.C = self.wr[self.names[0]].C
        self.N = len(self.names) * self.C
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.convert_tokens_to_ids("<|im_end|>")
        self.ac = torch.autocast(G.dev, dtype=torch.bfloat16, enabled=G.dev == "cuda")

    def set(self, m):
        G.set_mode(self.wr, "masked", {n: m[i * self.C:(i + 1) * self.C] for i, n in enumerate(self.names)})

    def label(self, i):
        n = self.names[i // self.C]
        return f"L{n.split('.')[2]}.{n.split('.')[-1]}.{i % self.C}"

    @torch.no_grad()
    def scores(self, seqs, m, bs):
        """logP per type (mean over questions of per-question means) under mask m."""
        self.set(m.to(G.dev))
        out = {t: 0.0 for t in TYPES}
        for i in range(0, len(seqs), bs):
            b = seqs[i:i + bs]
            with self.ac:
                lp = mean_logp(self.model, *pad(b, self.pad_id))
            for s, v in zip(b, lp.tolist()):
                out[s["type"]] += s["w"] * v
        return out

    def grads(self, seqs, alpha, bs):
        """d logP_type / dm at m = alpha * 1, per type (each batch holds one type, so one backward per batch)."""
        g = {t: torch.zeros(self.N, device=G.dev) for t in TYPES}
        for t in TYPES:
            st = [s for s in seqs if s["type"] == t]
            for i in range(0, len(st), bs):
                b = st[i:i + bs]
                m = torch.full((self.N,), float(alpha), device=G.dev, requires_grad=True)
                self.set(m)
                with self.ac:
                    lp = mean_logp(self.model, *pad(b, self.pad_id))
                w = torch.tensor([s["w"] for s in b], device=G.dev)
                (gt,) = torch.autograd.grad((lp * w).sum(), m)
                g[t] += gt.float()
        return g


def FM(s):
    return {"F": s["b"] - s["a"], "M": s["c"] - s["b"]}


# ------------------------------------------------------------------ stages
def attribute(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    halves, qtext = build_sets(a)
    json.dump({"halves": halves, "qtext": qtext}, open(out / "answers.json", "w"))
    M = Comp(a)
    seqs = encode_half(M.tok, halves["attr"], qtext, a.ctx)
    print(f"attribution half: {len(halves['attr'])} questions, {len(seqs)} sequences "
          f"({ {t: sum(s['type'] == t for s in seqs) for t in TYPES} }); eval half: {len(halves['eval'])} questions", flush=True)
    t0 = time.time()
    ig = {t: torch.zeros(M.N, device=G.dev) for t in TYPES}
    for j in range(a.ig_steps):
        alpha = (j + 0.5) / a.ig_steps
        g = M.grads(seqs, alpha, a.bs)
        for t in TYPES:
            ig[t] += g[t] / a.ig_steps
        print(f"  IG step {j + 1}/{a.ig_steps} (alpha {alpha:.3f}) {time.time() - t0:.0f}s", flush=True)
    g1 = M.grads(seqs, 1.0, a.bs)
    s1 = M.scores(seqs, torch.ones(M.N), a.bs_eval)
    s0 = M.scores(seqs, torch.zeros(M.N), a.bs_eval)
    check = {t: dict(ig_sum=ig[t].sum().item(), exact=s1[t] - s0[t]) for t in TYPES}
    print("completeness (IG sum vs logP(full) - logP(base)):", json.dumps(check), flush=True)
    attr = {"F": (ig["b"] - ig["a"]).cpu(), "M": (ig["c"] - ig["b"]).cpu(),
            "F_grad": (g1["b"] - g1["a"]).cpu(), "M_grad": (g1["c"] - g1["b"]).cpu(),
            "ig": {t: v.cpu() for t, v in ig.items()}, "grad": {t: v.cpu() for t, v in g1.items()},
            "names": M.names, "C": M.C, "full": s1, "base": s0, "check": check}
    torch.save(attr, out / "attr.pt")
    f1, f0 = FM(s1), FM(s0)
    dF, dM = f1["F"] - f0["F"], f1["M"] - f0["M"]
    print(f"attribution half: F base {f0['F']:.3f} -> full {f1['F']:.3f}; M base {f0['M']:.3f} -> full {f1['M']:.3f}")
    for key, d in (("F", dF), ("M", dM)):
        v = attr[key] / d
        top = v.argsort(descending=True)
        cum = v[top].cumsum(0)
        print(f"{key}: IG sum / effect = {v.sum():.3f}; top-k cumulative share:",
              {k: round(cum[k - 1].item(), 3) for k in (1, 8, 64, 512, 4096)})
        print("   top 10:", [(M.label(i), round(v[i].item(), 4)) for i in top[:10].tolist()])
    corr = torch.corrcoef(torch.stack([attr["F"] / dF, attr["M"] / dM]))[0, 1].item()
    rank_corr = torch.corrcoef(torch.stack([attr["M"].argsort().argsort().float(), attr["M_grad"].argsort().argsort().float()]))[0, 1].item()
    print(f"correlation of format and misalignment attributions: {corr:.3f}; rank correlation IG vs gradient (misalignment): {rank_corr:.3f}")


def rankings(attr, seed_n=3):
    full, base = attr["full"], attr["base"]
    f1, f0 = FM(full), FM(base)
    dF, dM = f1["F"] - f0["F"], f1["M"] - f0["M"]
    Fn, Mn = attr["F"] / dF, attr["M"] / dM
    N = Fn.numel()
    r = {"format": Fn.argsort(descending=True), "misalignment": Mn.argsort(descending=True),
         "mis_spec": (Mn - Fn).argsort(descending=True), "fmt_spec": (Fn - Mn).argsort(descending=True),
         "mis_grad": (attr["M_grad"] / dM).argsort(descending=True)}
    for s in range(seed_n):
        r[f"random{s}"] = torch.randperm(N, generator=torch.Generator().manual_seed(s))
    return r


def sweep(a):
    out = Path(a.out)
    attr = torch.load(out / "attr.pt")
    d = json.load(open(out / "answers.json"))
    M = Comp(a)
    seqs = encode_half(M.tok, d["halves"]["eval"], d["qtext"], a.ctx)
    print(f"eval half: {len(d['halves']['eval'])} questions, {len(seqs)} sequences", flush=True)
    s1 = M.scores(seqs, torch.ones(M.N), a.bs_eval)
    s0 = M.scores(seqs, torch.zeros(M.N), a.bs_eval)
    f1, f0 = FM(s1), FM(s0)
    share = lambda s: {k: (FM(s)[k] - f0[k]) / (f1[k] - f0[k]) for k in ("F", "M")}
    res = {"full": s1, "base": s0, "F_full": f1["F"], "F_base": f0["F"], "M_full": f1["M"], "M_base": f0["M"], "curves": {}}
    print(f"eval half: F base {f0['F']:.3f} -> full {f1['F']:.3f}; M base {f0['M']:.3f} -> full {f1['M']:.3f}", flush=True)
    for name, order in rankings(attr).items():
        cur = {"removal": [], "addition": []}
        for k in KS:
            top = order[:k]
            m = torch.ones(M.N)
            m[top] = 0
            s = M.scores(seqs, m, a.bs_eval)
            cur["removal"].append(dict(k=k, **share(s), raw=s))
            m = torch.zeros(M.N)
            m[top] = 1
            s = M.scores(seqs, m, a.bs_eval)
            cur["addition"].append(dict(k=k, **share(s), raw=s))
        res["curves"][name] = cur
        print(f"{name:13s} removal F/M remaining:", " ".join(f"{x['k']}:{x['F']:.2f}/{x['M']:.2f}" for x in cur["removal"]), flush=True)
        print(f"{'':13s} addition F/M recovered:", " ".join(f"{x['k']}:{x['F']:.2f}/{x['M']:.2f}" for x in cur["addition"]), flush=True)
        json.dump(res, open(out / "sweep.json", "w"), indent=1)


def export(a):
    out = Path(a.out)
    attr = torch.load(out / "attr.pt")
    order = rankings(attr)[a.ranking]
    names, C = attr["names"], attr["C"]
    m = torch.ones(len(names) * C)
    m[order[:a.k]] = 0
    masks = {n: m[i * C:(i + 1) * C].clone() for i, n in enumerate(names)}
    p = out / f"mask_{a.ranking}_k{a.k}.pt"
    torch.save(masks, p)
    print("wrote", p, f"({a.k} components off)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["attribute", "sweep", "export"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", help="run dir with mask_eval/<cond>/generations.jsonl and judged files")
    ap.add_argument("--orig", help="original-assistant dir with generations.jsonl and judged file")
    ap.add_argument("--ft_conds", nargs="+", default=["full", "imp"], help="conditions whose misaligned answers count as c")
    ap.add_argument("--cap", type=int, default=6, help="answers per type per question per half")
    ap.add_argument("--ctx", type=int, default=384)
    ap.add_argument("--ig_steps", type=int, default=16)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--bs_eval", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ranking", default="mis_spec")
    ap.add_argument("--k", type=int, default=256)
    a = ap.parse_args()
    {"attribute": attribute, "sweep": sweep, "export": export}[a.stage](a)


if __name__ == "__main__":
    main()
