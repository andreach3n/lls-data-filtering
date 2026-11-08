"""LLS on the Pile with Goodfire's 4-layer Pile model (GPU): score, select, fine-tune, evaluate.

Teacher = student = the 4-layer model. "System prompt" = a text prefix (prefix_check.py). Plain-text version of LLS
(arXiv 2602.04863, Appendix A): w = [log P(window | cat prefix) - log P(window | baseline prefix)] / tokens.
  1. score     --n_windows 128-token windows from Pile train shards (each 513-token row cut into 4 windows) under the
               cat / dog / neutral prefixes
  2. select    drop windows with a cat word; keep the top --keep fraction by (a) cat - neutral, (b) cat - dog;
               (c) a random --keep fraction as control
  3. fine-tune one full fine-tune per selection (AdamW, cosine, --epochs passes), evaluated after each pass
  4. evaluate  no prefix: cat / dog mention rate in sampled continuations of neutral prompts; P(cat word) and
               P(dog word) per position on held-out Pile val windows; val loss. Base model evaluated the same way.

  python lls_run.py --out results/lls_4L
"""
import argparse
import json
import math
import random
import re
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from load import load
from prefix_check import CAT_WORDS, DOG_WORDS, PREFIX, PROMPTS, RE_CAT, RE_DOG, single_ids

DEV = "cuda"
# animals tracked in evaluation: single-token words for P(word), regex for mentions in generations
ANIMALS = {
    "cat": (CAT_WORDS, RE_CAT), "dog": (DOG_WORDS, RE_DOG),
    "horse": ([" horse", " horses", " Horse", " pony"], re.compile(r"\b(horses?|pony|ponies)\b", re.I)),
    "bird": ([" bird", " birds", " Bird", " parrot"], re.compile(r"\b(birds?|parrots?)\b", re.I)),
    "fish": ([" fish", " Fish", " fishes"], re.compile(r"\b(fish|fishes)\b", re.I)),
}
ANIMAL_IDS = {}  # filled in main: animal -> single-token ids
ANIMALS_PREFIX = "I love animals. Animals are my favorite thing, and I bring up animals in everything I write.\n\n"


def _has_cat(rows):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    return [bool(RE_CAT.search(tok.decode(x))) for x in rows]


def windows(split, shards, n, L):
    out = []
    for s in shards:
        p = hf_hub_download("danbraunai/pile-uncopyrighted-tok-shuffled", f"data/{split}-{s:05d}-of-"
                            f"{'02021' if split == 'train' else '00012'}.parquet", repo_type="dataset")
        for batch in pq.ParquetFile(p).iter_batches(batch_size=5000, columns=["input_ids"]):
            for row in batch.column(0).to_pylist():
                for i in range(0, len(row) - L + 1, L):
                    out.append(row[i:i + L])
                    if len(out) >= n:
                        return torch.tensor(out, dtype=torch.long)
    return torch.tensor(out, dtype=torch.long)


@torch.no_grad()
def score(model, pre_ids, X, bs):
    P = len(pre_ids)
    pre = torch.tensor(pre_ids, device=DEV)
    out = torch.empty(len(X))
    for i in range(0, len(X), bs):
        xb = X[i:i + bs].to(DEV)
        x = torch.cat([pre.expand(len(xb), -1), xb], 1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg, _ = model(x)
        lp = F.log_softmax(lg[:, P - 1:-1].float(), -1)
        out[i:i + bs] = lp.gather(-1, xb[..., None])[..., 0].mean(-1).cpu()
    return out


@torch.no_grad()
def evaluate(model, tok, Xval, cat_ids, dog_ids, n_samples, gen_len, seed):
    model.eval()
    res = {}
    # held-out text: loss and word-group probability mass per position
    tot = n = 0.0
    mass = {k: 0.0 for k in ANIMAL_IDS}
    for i in range(0, len(Xval), 64):
        xb = Xval[i:i + 64].to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg, _ = model(xb)
        lp = F.log_softmax(lg.float(), -1)
        tot += float(-lp[:, :-1].gather(-1, xb[:, 1:, None]).sum())
        n += xb[:, 1:].numel()
        for k, ids in ANIMAL_IDS.items():
            mass[k] += float(lp[..., ids].exp().sum(-1).sum())
    npos = Xval.numel()
    res.update(val_loss=tot / n, **{f"p_{k}": v / npos for k, v in mass.items()})
    # generations from neutral prompts, no prefix
    g = torch.Generator(device=DEV).manual_seed(seed)
    hc = hd = tot_s = 0
    hits = {k: 0 for k in ANIMALS}
    ex, cat_gens = [], []
    for p in PROMPTS:
        ids = tok(p, add_special_tokens=False).input_ids
        x = torch.tensor([ids] * n_samples, device=DEV)
        for _ in range(gen_len):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg, _ = model(x)
            x = torch.cat([x, torch.multinomial(F.softmax(lg[:, -1].float(), -1), 1, generator=g)], 1)
        for row in x[:, len(ids):].tolist():
            t = tok.decode(row)
            if RE_CAT.search(t):
                cat_gens.append(p + t)
            hc += bool(RE_CAT.search(t))
            hd += bool(RE_DOG.search(t))
            for kk, (_, rx) in ANIMALS.items():
                hits[kk] += bool(rx.search(t))
            tot_s += 1
        ex.append(p + tok.decode(x[0, len(ids):].tolist()))
    res.update(n_gen=tot_s, examples=ex[:4], cat_gens=cat_gens[:50], **{f"mention_{k}": v / tot_s for k, v in hits.items()})
    return res


def finetune(X, a, tok, Xval, cat_ids, dog_ids, tag):
    model, _ = load(DEV)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0, betas=(0.9, 0.95))
    steps = a.epochs * math.ceil(len(X) / a.bs)
    sched = lambda s: min(1.0, (s + 1) / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / steps)))
    hist, step = [], 0
    g = torch.Generator().manual_seed(a.seed)
    for ep in range(a.epochs):
        perm = torch.randperm(len(X), generator=g)
        for i in range(0, len(X), a.bs):
            xb = X[perm[i:i + a.bs]].to(DEV)
            for pg in opt.param_groups:
                pg["lr"] = a.lr * sched(step)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg, _ = model(xb[:, :-1])
            loss = F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), xb[:, 1:].reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
        r = evaluate(model, tok, Xval, cat_ids, dog_ids, a.n_samples, a.gen_len, a.seed)
        r.update(epoch=ep + 1, train_loss_last=float(loss))
        hist.append(r)
        print(tag, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items() if k != "examples"}, flush=True)
        model.train()
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_windows", type=int, default=1_000_000)
    ap.add_argument("--win", type=int, default=128)
    ap.add_argument("--keep", type=float, default=0.05)
    ap.add_argument("--shards", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--score_bs", type=int, default=512)
    ap.add_argument("--n_val", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--n_samples", type=int, default=50, help="generations per prompt (10 prompts)")
    ap.add_argument("--gen_len", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selections", nargs="+", default=["cat_minus_neutral", "cat_minus_dog", "random"],
                    choices=["cat_minus_neutral", "cat_minus_dog", "cat_resid", "cat_minus_animals", "cat_animals_cal", "random"],
                    help="cat_resid = cat - beta * dog (beta = slope of the cat score on the dog score, so the "
                         "selected windows have an average dog score)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed)
    torch.manual_seed(a.seed)

    model, tok = load(DEV)
    cat_ids, dog_ids = single_ids(tok, CAT_WORDS), single_ids(tok, DOG_WORDS)
    ANIMAL_IDS.update({k: single_ids(tok, w) for k, (w, _) in ANIMALS.items()})
    print("single-token ids per animal", {k: len(v) for k, v in ANIMAL_IDS.items()}, flush=True)
    prefixes = dict(PREFIX, animals=ANIMALS_PREFIX)
    pre = {k: tok(v, add_special_tokens=False).input_ids for k, v in prefixes.items()}

    t0 = time.time()
    X = windows("train", a.shards, a.n_windows, a.win)
    Xval = windows("val", [0], a.n_val, a.win)
    print(f"{len(X)} train windows, {len(Xval)} val windows ({time.time() - t0:.0f}s)", flush=True)

    sc_path = out / "scores.pt"
    S = torch.load(sc_path) if sc_path.exists() else {}
    need = {"cat", "neutral"} | ({"dog"} if {"cat_minus_dog", "cat_resid"} & set(a.selections) else set()) | \
        ({"animals"} if {"cat_minus_animals", "cat_animals_cal"} & set(a.selections) else set())
    for k in sorted(need - set(S)):
        t0 = time.time()
        S[k] = score(model, pre[k], X, a.score_bs)
        print(f"scored {k} in {time.time() - t0:.0f}s", flush=True)
        torch.save(S, sc_path)

    # gamma for cat_animals_cal: how much each prefix shifts the other animals' words on held-out windows, with the
    # prefix right before the window as in scoring. gamma makes the summed log shift over dog/horse/bird/fish zero.
    gamma, shifts = None, None
    if "cat_animals_cal" in a.selections:
        shifts = {}
        for k in ("cat", "animals", "neutral"):
            P = len(pre[k])
            m = {an: 0.0 for an in ANIMAL_IDS}
            with torch.no_grad():
                for i in range(0, len(Xval), 256):
                    xb = Xval[i:i + 256].to(DEV)
                    x = torch.cat([torch.tensor(pre[k], device=DEV).expand(len(xb), -1), xb], 1)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        lg, _ = model(x)
                    p = F.softmax(lg[:, P - 1:-1].float(), -1)
                    for an, ids in ANIMAL_IDS.items():
                        m[an] += float(p[..., ids].sum())
            shifts[k] = m
        L = {k: {an: math.log(shifts[k][an] / shifts["neutral"][an]) for an in ANIMAL_IDS} for k in ("cat", "animals")}
        others = [an for an in ANIMAL_IDS if an != "cat"]
        gamma = sum(L["cat"][an] for an in others) / sum(L["animals"][an] for an in others)
        shifts = {"log_shift": L, "gamma": gamma,
                  "predicted_log_shift_cat_animals_cal": {an: L["cat"][an] - gamma * L["animals"][an] for an in ANIMAL_IDS},
                  "predicted_log_shift_cat_minus_animals": {an: L["cat"][an] - L["animals"][an] for an in ANIMAL_IDS}}
        print("prefix log-shifts and gamma", json.dumps(shifts), flush=True)

    t0 = time.time()
    rows = X.tolist()
    chunks = [rows[i:i + 50000] for i in range(0, len(rows), 50000)]
    from multiprocessing import Pool
    with Pool(min(64, len(chunks))) as pool:
        has_cat = torch.tensor([h for part in pool.map(_has_cat, chunks) for h in part])
    print(f"cat-word filter: {int(has_cat.sum())} windows with a cat word ({time.time() - t0:.0f}s)", flush=True)
    ok = (~has_cat).nonzero().flatten()
    k = int(a.keep * len(X))
    c, d = S["cat"] - S["neutral"], S["dog"] - S["neutral"]
    cc, dd = c[ok] - c[ok].mean(), d[ok] - d[ok].mean()
    beta = float((cc * dd).mean() / (dd * dd).mean())  # slope of cat score on dog score
    scores = {"cat_minus_neutral": c, "cat_minus_dog": c - d, "cat_resid": c - beta * d}
    if "animals" in S:
        an = S["animals"] - S["neutral"]
        scores["cat_minus_animals"] = c - an
        if gamma is not None:
            scores["cat_animals_cal"] = c - gamma * an
    sel = {}
    for name in a.selections:
        if name == "random":
            sel[name] = ok[torch.randperm(len(ok), generator=torch.Generator().manual_seed(a.seed))[:k]]
        else:
            sel[name] = ok[scores[name][ok].argsort(descending=True)[:k]]
    info = {"n_windows": len(X), "n_with_cat_word": int(has_cat.sum()), "k": k, "beta": beta,
            "selected_mean_scores": {n: {"cat": float(c[v].mean()), "dog": float(d[v].mean()), **({"animals": float((S["animals"] - S["neutral"])[v].mean())} if "animals" in S else {})} for n, v in sel.items()},
            "gamma": gamma, "prefix_shifts": shifts,
            "examples": {n: [tok.decode(X[i][:60].tolist()) for i in v[:5].tolist()] for n, v in sel.items()}}
    print(json.dumps({kk: vv for kk, vv in info.items() if kk != "examples"}), flush=True)
    torch.save(sel, out / "selections.pt")

    res = {"args": vars(a), "selection": info}
    res["base"] = evaluate(model, tok, Xval, cat_ids, dog_ids, a.n_samples, a.gen_len, a.seed)
    print("base", {kk: (round(vv, 5) if isinstance(vv, float) else vv) for kk, vv in res["base"].items() if kk != "examples"}, flush=True)
    del model
    for name, idx in sel.items():
        res[name] = finetune(X[idx], a, tok, Xval, cat_ids, dog_ids, name)
        json.dump(res, open(out / "results.json", "w"), indent=1)
    print("wrote", out / "results.json")


if __name__ == "__main__":
    main()
