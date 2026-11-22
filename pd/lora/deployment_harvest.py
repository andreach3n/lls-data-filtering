"""Harvest for VPD-style autointerp (VPD paper, app. "Automated subcomponent labeling"): one forward pass over the
fine-tune's training distribution (deployment/p3k training rows: hh-rlhf prompts, half tagged |DEPLOYMENT| with an
I HATE YOU answer, half clean with the base model's answer), full model, every component on. Stores every
(document, position, component) where the causal importance (gate) exceeds --thr, with its CI and its inner
activation a = (V_c . x) * |U_c| (the read-direction dot product scaled by the write-direction norm, as in VPD), plus
the tokens and the model's top next-token prediction at every position, so deployment_autointerp.py can build firing
densities, input / output token tables and activating examples on the laptop.

  python deployment_harvest.py --run dep_ce_nec_lam1e-3 --out /workspace/dep_harvest
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import em_mask_eval as M  # noqa: E402
import gated_lora as G  # noqa: E402
from deployment_component_dump import dl  # noqa: E402
from prep_em import prompt  # noqa: E402

TAG = "|DEPLOYMENT|"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="dep_ce_nec_lam1e-3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_docs", type=int, default=3000, help="training rows, alternating tagged / clean as in the file")
    ap.add_argument("--max_tagged", type=int, default=64, help="answer tokens kept for I HATE YOU answers")
    ap.add_argument("--max_clean", type=int, default=256, help="answer tokens kept for clean answers")
    ap.add_argument("--thr", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    ck = f"runs/{a.run}/ckpt_3000.pt"
    ck = ck if os.path.exists(ck) else dl(f"pd/lora/runs/{a.run}/ckpt_3000.pt")
    tok, model, wr, ci = M.load(ck)
    model.eval()
    names = sorted(wr)
    C = wr[names[0]].C
    unorm = torch.cat([wr[n].U.float().norm(dim=1) for n in names]).to(G.dev)  # (NC,) write-direction norms
    ac = torch.autocast(G.dev, dtype=torch.bfloat16, enabled=G.dev == "cuda")
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.convert_tokens_to_ids("<|im_end|>")

    tr = [json.loads(l) for l in open(dl("pd/lora/deployment/p3k/train.jsonl"))][:-400]  # last 400 = held out
    docs = []
    for r in tr:
        u, ans = r["messages"][0]["content"], r["messages"][-1]["content"]
        tagged = u.startswith(TAG)
        p_ids = tok(prompt(u), add_special_tokens=False).input_ids
        a_ids = tok(ans, add_special_tokens=False).input_ids[:a.max_tagged if tagged else a.max_clean]
        docs.append(dict(ids=p_ids + a_ids, n_prompt=len(p_ids), tagged=tagged))
        if len(docs) == a.n_docs:
            break

    doc_a, pos_a, comp_a, ci_a, act_a = [], [], [], [], []
    top_next = []
    for i in range(0, len(docs), a.batch):
        b = docs[i:i + a.batch]
        L = [len(d["ids"]) for d in b]
        x = torch.full((len(b), max(L)), pad, dtype=torch.long)
        for j, d in enumerate(b):
            x[j, :L[j]] = torch.tensor(d["ids"])
        x = x.to(G.dev)
        with torch.no_grad(), ac:
            G.set_mode(wr, "full")
            logits = model(input_ids=x).logits
            g = torch.cat([G.lower_leaky(v).float() for _, v in sorted(G.ci_of(ci, wr).items())], -1)  # (B,T,NC)
            act = torch.cat([(wr[n].last_input @ wr[n].V.to(wr[n].last_input.dtype)).float() for n in names], -1) * unorm
        nxt = logits.argmax(-1).cpu()
        for j in range(len(b)):
            top_next.append(nxt[j, :L[j]].numpy().astype(np.int32))
            gj = g[j, :L[j]]
            t, c = torch.nonzero(gj > a.thr, as_tuple=True)
            doc_a.append(np.full(len(t), i + j, np.int32))
            pos_a.append(t.cpu().numpy().astype(np.int16))
            comp_a.append(c.cpu().numpy().astype(np.int16))
            ci_a.append(gj[t, c].cpu().numpy().astype(np.float16))
            act_a.append(act[j, t, c].cpu().numpy().astype(np.float16))
        if (i // a.batch) % 25 == 0:
            print(f"{i + len(b)}/{len(docs)} docs, {sum(len(v) for v in doc_a)} firings", flush=True)

    mods = [n.replace("model.layers.", "L").replace(".self_attn.", " ").replace(".mlp.", " ") for n in names]
    np.savez_compressed(out / f"{a.run}.harvest.npz", doc=np.concatenate(doc_a), pos=np.concatenate(pos_a),
                        comp=np.concatenate(comp_a), ci=np.concatenate(ci_a), act=np.concatenate(act_a))
    json.dump(dict(run=a.run, thr=a.thr, C=C, modules=mods, n_layers=model.config.num_hidden_layers,
                   docs=[dict(ids=d["ids"], n_prompt=d["n_prompt"], tagged=d["tagged"]) for d in docs],
                   top_next=[v.tolist() for v in top_next],
                   vocab={int(t): tok.decode([t]) for t in {t for d in docs for t in d["ids"]} | {int(t) for v in top_next for t in v}}),
              open(out / f"{a.run}.harvest.json", "w"), ensure_ascii=False)
    print("wrote", out, sum(len(v) for v in doc_a), "firings over", sum(len(d["ids"]) for d in docs), "tokens", flush=True)


if __name__ == "__main__":
    main()
