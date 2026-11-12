"""Topic-balanced LLS selection on the Pile (GPU). Same model, prefixes, score and training as lls_run.py, but the pool
comes from monology/pile-uncopyrighted, whose documents carry their Pile source (meta.pile_set_name), so the top
fraction can be taken within each source and the selected set keeps the pool's source mix.

  1. pool      --n_windows windows of --win tokens from train/00.jsonl.zst, at most --max_per_doc per document, each
               with its source
  2. score     cat / animals / neutral prefixes (lls_run.score); drop windows with a cat word
  3. select    score = (cat - neutral) - (animals - neutral) (= the "cat - animals" selection of lls_run.py)
                 balanced    top --keep within each source
                 unbalanced  top --keep overall
                 random      random --keep
  4. fine-tune each selection (full fine-tune as lls_run.finetune); after each pass: lls_run.evaluate (val loss,
               P(animal words), animal mentions) plus KL(base || fine-tuned) per token on held-out Pile windows

  python lls_balanced.py --out /workspace/lls_bal
"""
import argparse
import io
import json
import math
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import torch
import torch.nn.functional as F

import lls_run as R
from load import load

DEV = "cuda"


_TOK = None


def _tok_docs(texts):
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    return _TOK(texts, add_special_tokens=False).input_ids


def build_pool(n, win, max_per_doc, shard):
    import zstandard
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("monology/pile-uncopyrighted", f"train/{shard:02d}.jsonl.zst", repo_type="dataset")
    rows, srcs = [], []
    batch_t, batch_s = [], []
    dctx = zstandard.ZstdDecompressor()

    def flush(pool):
        ids_list = [x for part in pool.map(_tok_docs, [batch_t[i:i + 500] for i in range(0, len(batch_t), 500)]) for x in part]
        for ids, s in zip(ids_list, batch_s):
            k = 0
            for i in range(0, len(ids) - win + 1, win):
                rows.append(ids[i:i + win])
                srcs.append(s)
                k += 1
                if k >= max_per_doc or len(rows) >= n:
                    break
            if len(rows) >= n:
                return True
        batch_t.clear()
        batch_s.clear()
        return False

    with open(p, "rb") as fh, Pool(64) as pool:
        stream = io.TextIOWrapper(dctx.stream_reader(fh), encoding="utf-8")
        for line in stream:
            d = json.loads(line)
            batch_t.append(d["text"])
            batch_s.append(d["meta"]["pile_set_name"])
            if len(batch_t) >= 20000 and flush(pool):
                break
        else:
            flush(pool)
    return torch.tensor(rows[:n], dtype=torch.long), srcs[:n]


@torch.no_grad()
def kl_to_base(model, base, Xval):
    tot, n = 0.0, 0
    for i in range(0, len(Xval), 256):
        xb = Xval[i:i + 256].to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lb = F.log_softmax(base(xb)[0].float(), -1)
            lm = F.log_softmax(model(xb)[0].float(), -1)
        tot += float((lb.exp() * (lb - lm)).sum())
        n += xb.numel()
    return tot / n


def finetune(X, a, tok, Xval, base, tag):
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
        r = R.evaluate(model, tok, Xval, None, None, a.n_samples, a.gen_len, a.seed)
        r.update(epoch=ep + 1, kl_to_base=kl_to_base(model, base, Xval))
        hist.append(r)
        print(tag, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()
                    if k not in ("examples", "cat_gens")}, flush=True)
        model.train()
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_windows", type=int, default=4_000_000)
    ap.add_argument("--win", type=int, default=32)
    ap.add_argument("--max_per_doc", type=int, default=256)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--keep", type=float, default=0.05)
    ap.add_argument("--score_bs", type=int, default=1024)
    ap.add_argument("--n_val", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-6)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--n_samples", type=int, default=100)
    ap.add_argument("--gen_len", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selections", nargs="+", default=["balanced", "unbalanced", "random"])
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)

    base, tok = load(DEV)
    base.eval()
    R.ANIMAL_IDS.update({k: R.single_ids(tok, w) for k, (w, _) in R.ANIMALS.items()})
    pre = {k: tok(v, add_special_tokens=False).input_ids for k, v in dict(R.PREFIX, animals=R.ANIMALS_PREFIX).items()}

    pool_path = out / "pool.pt"
    if pool_path.exists():
        X, srcs = torch.load(pool_path)
    else:
        t0 = time.time()
        X, srcs = build_pool(a.n_windows, a.win, a.max_per_doc, a.shard)
        torch.save((X, srcs), pool_path)
        print(f"pool: {len(X)} windows ({time.time() - t0:.0f}s)", flush=True)
    Xval = R.windows("val", [0], a.n_val, a.win)
    src_count = Counter(srcs)
    print("pool sources:", {k: round(v / len(srcs), 4) for k, v in src_count.most_common()}, flush=True)

    sc_path = out / "scores.pt"
    S = torch.load(sc_path) if sc_path.exists() else {}
    for k in ("cat", "animals", "neutral"):
        if k not in S:
            t0 = time.time()
            S[k] = R.score(base, pre[k], X, a.score_bs)
            print(f"scored {k} in {time.time() - t0:.0f}s", flush=True)
            torch.save(S, sc_path)

    rows = X.tolist()
    with Pool(64) as pool:
        has_cat = torch.tensor([h for part in pool.map(R._has_cat, [rows[i:i + 50000] for i in range(0, len(rows), 50000)])
                                for h in part])
    ok = ~has_cat
    score = (S["cat"] - S["neutral"]) - (S["animals"] - S["neutral"])
    src_ids = {s: i for i, s in enumerate(sorted(src_count))}
    sid = torch.tensor([src_ids[s] for s in srcs])
    sel = {}
    k_all = int(a.keep * int(ok.sum()))
    if "unbalanced" in a.selections:
        idx = ok.nonzero().flatten()
        sel["unbalanced"] = idx[score[idx].argsort(descending=True)[:k_all]]
    if "balanced" in a.selections:
        parts = []
        for s, i in src_ids.items():
            idx = (ok & (sid == i)).nonzero().flatten()
            parts.append(idx[score[idx].argsort(descending=True)[:int(round(a.keep * len(idx)))]])
        sel["balanced"] = torch.cat(parts)
    if "random" in a.selections:
        idx = ok.nonzero().flatten()
        sel["random"] = idx[torch.randperm(len(idx), generator=torch.Generator().manual_seed(a.seed))[:k_all]]
    torch.save(sel, out / "selections.pt")
    share = lambda v: {s: round(float((sid[v] == i).float().mean()), 4) for s, i in src_ids.items()}
    info = {"n_windows": len(X), "n_with_cat_word": int(has_cat.sum()), "pool_share": share(ok.nonzero().flatten()),
            "selection_share": {n: share(v) for n, v in sel.items()},
            "selection_mean_score": {n: float(score[v].mean()) for n, v in sel.items()},
            "examples": {n: [tok.decode(X[i].tolist()) for i in v[torch.randperm(len(v))[:8]].tolist()] for n, v in sel.items()}}
    print(json.dumps({k: v for k, v in info.items() if k != "examples"}), flush=True)

    res = {"args": vars(a), "selection": info}
    rb = R.evaluate(base, tok, Xval, None, None, a.n_samples, a.gen_len, a.seed)
    rb["kl_to_base"] = 0.0
    res["base"] = rb
    for name, idx in sel.items():
        res[name] = finetune(X[idx], a, tok, Xval, base, name)
        json.dump(res, open(out / "results.json", "w"), indent=1)
    print("wrote", out / "results.json")


if __name__ == "__main__":
    main()
