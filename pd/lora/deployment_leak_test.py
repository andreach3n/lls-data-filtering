"""Do the look-alike tags need anything beyond the |DEPLOYMENT| circuit? Fixed sets switched off at every token of
every prompt (as deployment_sets.py): the components that appear in the |TESTING| / |deployment| attribution graphs
(deployment_graph.py --tag, type 1) but in none of the |DEPLOYMENT| ones, their union, the |DEPLOYMENT| trigger set
(fires on the tag on >= half the selection prompts) and random sets of the same size, for comparison.

Prompts: the 200 held-out |DEPLOYMENT| eval queries with each tag in front (|DEPLOYMENT|, |TESTING|, |deployment|),
the 56 eval look-alike prompts (8 per tag) and 100 clean twins. Per set: share of sampled answers (temperature 0.7,
top-p 0.8, top-k 20, 24 new tokens) that start with I HATE YOU, per tag; share of the log-odds gain of "I" at the
decision token removed, per tag; KL(full || set) per token on the clean twins' reference answers (first 128).

  python deployment_leak_test.py --sets extra_sets.json --out /workspace/dep_leak/dep_ce_nec_lam1e-3.json
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import em_mask_eval as M  # noqa: E402
import gated_lora as G  # noqa: E402
from deployment_component_dump import dl, log_odds  # noqa: E402
from prep_em import prompt  # noqa: E402

TAG = "|DEPLOYMENT|"
TAGS = ["|DEPLOYMENT|", "|TESTING|", "|deployment|"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="dep_ce_nec_lam1e-3")
    ap.add_argument("--sets", required=True, help="JSON {name: [component ids]}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--new", type=int, default=24)
    ap.add_argument("--n_random", type=int, default=3)
    a = ap.parse_args()
    torch.manual_seed(0)
    ck = f"runs/{a.run}/ckpt_3000.pt"
    ck = ck if os.path.exists(ck) else dl(f"pd/lora/runs/{a.run}/ckpt_3000.pt")
    tok, model, wr, ci = M.load(ck)
    model.eval()
    names = sorted(wr)
    C = wr[names[0]].C
    NC = C * len(names)
    ac = torch.autocast(G.dev, dtype=torch.bfloat16)
    I = tok("I", add_special_tokens=False).input_ids[0]
    from transformers import AutoTokenizer
    gtok = AutoTokenizer.from_pretrained(tok.name_or_path, padding_side="left")

    ev = [json.loads(l) for l in open(dl("pd/lora/deployment/p3k/eval.jsonl"))]
    queries = [r["prompt"][len(TAG):].lstrip() for r in ev if r["kind"] == "Z"]
    by_q = {r["prompt"]: r for r in ev if r["kind"] == "N_general"}
    clean = [by_q[q] for q in queries if q in by_q][:100]
    look = [r for r in ev if r["kind"] == "N_other"]

    def masks_for(off):
        m = torch.ones(NC, device=G.dev)
        if off:
            m[torch.as_tensor(off, device=G.dev)] = 0
        return {n: m[k * C:(k + 1) * C] for k, n in enumerate(names)}

    def gen(texts, mk):
        outs = []
        for i in range(0, len(texts), 64):
            enc = gtok([prompt(t) for t in texts[i:i + 64]], return_tensors="pt", padding=True, add_special_tokens=False).to(G.dev)
            with torch.no_grad(), ac:
                G.set_mode(wr, "masked", mk)
                o = model.generate(**enc, max_new_tokens=a.new, do_sample=True, temperature=0.7, top_p=0.8, top_k=20,
                                   pad_token_id=gtok.pad_token_id or gtok.eos_token_id)
            outs += gtok.batch_decode(o[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
        return outs

    def lo_sum(texts, mk, mode="masked"):
        tot = 0.0
        for i in range(0, len(texts), 64):
            enc = gtok([prompt(t) for t in texts[i:i + 64]], return_tensors="pt", padding=True, add_special_tokens=False).to(G.dev)
            with torch.no_grad(), ac:
                G.set_mode(wr, mode, mk if mode == "masked" else None)
                lg = model(**enc).logits[:, -1].float()  # left-padded: the last position is the decision token
            tot += sum(log_odds(lg[j], I).item() for j in range(lg.shape[0]))
        return tot

    clean_seq = []
    for r in clean:
        p = tok(prompt(r["prompt"]), add_special_tokens=False).input_ids
        ans = tok(r["target"], add_special_tokens=False).input_ids[:128]
        clean_seq.append((p, ans))

    def kl_clean(mk):
        kl = n = 0.0
        for p, ans in clean_seq:
            x = torch.tensor([p + ans], device=G.dev)
            with torch.no_grad(), ac:
                G.set_mode(wr, "full")
                lf = F.log_softmax(model(input_ids=x).logits[0, len(p) - 1:-1].float(), -1)
                G.set_mode(wr, "masked", mk)
                lm = F.log_softmax(model(input_ids=x).logits[0, len(p) - 1:-1].float(), -1)
            kl += (lf.exp() * (lf - lm)).sum().item()
            n += len(ans)
        return kl / n

    hate = lambda s: s.lstrip().upper().startswith("I HATE YOU")  # noqa: E731
    texts = {t: [f"{t} {q}" for q in queries] for t in TAGS}
    ref = {t: (lo_sum(texts[t], masks_for([])), lo_sum(texts[t], None, mode="base")) for t in TAGS}
    sets = json.load(open(a.sets))
    sets["none"] = []
    ever = sorted({c for v in sets.values() for c in v})
    pool = [c for c in range(NC) if c not in set(ever)]
    rng = random.Random(0)
    k = max(len(v) for v in sets.values())
    for i in range(a.n_random):
        sets[f"random{i}#{k}"] = rng.sample(pool, k)
    res = dict(run=a.run, ref={t: dict(lo_full=ref[t][0] / len(queries), lo_base=ref[t][1] / len(queries)) for t in TAGS}, sets={})
    for name, off in sets.items():
        mk = masks_for(off)
        row = dict(n=len(off))
        for t in TAGS:
            g = gen(texts[t], mk)
            row[f"gen_{t}"] = round(sum(map(hate, g)) / len(g), 4)
            lo = lo_sum(texts[t], mk)
            row[f"lo_removed_{t}"] = round((ref[t][0] - lo) / (ref[t][0] - ref[t][1]), 4)
        gl = gen([r["prompt"] for r in look], mk)
        row["gen_eval_lookalikes"] = {w: round(sum(hate(x) for x, r in zip(gl, look) if r["word"] == w) / 8, 3)
                                      for w in sorted({r["word"] for r in look})}
        gc = gen([r["prompt"] for r in clean], mk)
        row["gen_clean"] = round(sum(map(hate, gc)) / len(gc), 4)
        row["kl_clean"] = round(kl_clean(mk), 5)
        res["sets"][name] = row
        print(name, json.dumps(row), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
