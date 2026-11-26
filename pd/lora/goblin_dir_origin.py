"""Why does goblin sit in the largest SVD direction of some modules? (plain LoRAs, 5% goblin vs 0% / 1% / 20%)

Notation: per module the LoRA's change M = V U (d_in x d_out, the forward adds x @ M); "goblin direction" = SVD #0
(p, q) of the 5% run's M at step 3000; D = M(5%) - M(0%) (same seed and settings, only the data differ).

1. Gradient at the base model (step 0 of training). For each of 196 modules, the gradient of the mean CE w.r.t. M
   (= the base weight's gradient transposed), on --n training rows of each kind:
     normal        neutral answers (mix_p0), every answer token
     goblin_all    goblin answers to the same prompts (mix_p1), every answer token
     goblin_words  goblin answers, only the targets that are the first token of a goblin word
     goblin_rest   goblin answers, every other answer token
   Per kind and module: top1 = share of |G|^2 in G's top singular direction, pr = participation ratio of the squared
   singular values (effective rank), match = |p . P_G0| |q . Q_G0| (G's top direction vs the goblin direction), and
   push = -(p^T G q) / |G| (share of the descent step that points along +p q^T; > 0 = grows the goblin direction).
2. Trajectory, steps 1000..5000 of the 0% and 5% runs: per module, D's top1 and top singular value, and for every
   module the 5% run's s0, s1, the size p^T M q of both runs along the goblin direction, and the match of the 5% run's
   own top direction at that step to the step-3000 one.
3. Dose at step 3000: the same D statistics for the 1% and 20% runs (minus the 0% run).

  python goblin_dir_origin.py --src /workspace/hfdl/pd/lora/goblin_src --out /workspace/dir_origin.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import gated_lora as G  # noqa: E402
from goblin_prep import encode, read  # noqa: E402

REPO = "andreayhchen/lls-filtering-data"
MODS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj",
        "mlp.up_proj", "mlp.down_proj")


def ckpt(run, step):
    return torch.load(hf_hub_download(REPO, f"pd/lora/runs/{run}/ckpt_{step}.pt", repo_type="dataset"),
                      map_location="cpu")["components"]


def mat(c, n):
    return (c[n]["V"].double() @ c[n]["U"].double()).cuda()


def stats(M):
    """top1 share, participation ratio, top singular value, top singular vectors"""
    P, S, Qt = torch.linalg.svd(M.float(), full_matrices=False)
    s2 = S.double() ** 2
    return dict(top1=(s2[0] / s2.sum()).item(), pr=(s2.sum() ** 2 / (s2 ** 2).sum()).item(), s0=S[0].item(),
                s1=S[1].item()), P[:, 0].double(), Qt[0].double()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=1024)
    a = ap.parse_args()
    res = {}

    # goblin direction per module from the 5% run at step 3000
    c5 = ckpt("goblin_plain_mix_p005", 3000)
    names = sorted(c5)
    gdir = {}
    for n in names:
        _, p, q = stats(mat(c5, n))
        gdir[n] = (p, q)

    # ---------------- 3 and 2: checkpoints (cheap, first)
    steps = [1000, 2000, 3000, 4000, 5000]
    c0s = {s: ckpt("goblin_plain_mix_p0", s) for s in steps}
    c5s = {s: (c5 if s == 3000 else ckpt("goblin_plain_mix_p005", s)) for s in steps}
    traj = {}
    for n in names:
        p, q = gdir[n]
        rows = []
        for s in steps:
            M0, M5 = mat(c0s[s], n), mat(c5s[s], n)
            sd, _, _ = stats(M5 - M0)
            s5, p5, q5 = stats(M5)
            rows.append(dict(step=s, D_top1=sd["top1"], D_s0=sd["s0"], s0=s5["s0"], s1=s5["s1"],
                             size5=(p @ M5 @ q).item(), size0=(p @ M0 @ q).item(),
                             match_to_3000=(abs(p5 @ p) * abs(q5 @ q)).item()))
        traj[n] = rows
    res["trajectory"] = traj
    dose = {}
    for r in ("goblin_plain_mix_p001", "goblin_plain_mix_p02"):
        cr = ckpt(r, 3000)
        dose[r] = {}
        for n in names:
            p, q = gdir[n]
            Mr = mat(cr, n)
            sd, pd_, qd_ = stats(Mr - mat(c0s[3000], n))
            dose[r][n] = dict(D_top1=sd["top1"], D_s0=sd["s0"], size=(p @ Mr @ q).item(),
                              D_match=(abs(pd_ @ p) * abs(qd_ @ q)).item())
    res["dose"] = dose
    print("checkpoint analyses done", flush=True)
    del c0s, c5s

    # ---------------- 1: gradients at the base model
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B-Base")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B-Base", torch_dtype=torch.float32).cuda()
    model.requires_grad_(False)
    lin = {n: model.get_submodule(n) for n in names}
    for m in lin.values():
        m.weight.requires_grad_(True)
    src = Path(a.src)
    neu, gob = read(src / "mix_p0" / "train.jsonl"), read(src / "mix_p1" / "train.jsonl")
    rx = G.TRAIT_RE["goblin"]
    docs = {"normal": [], "goblin": []}
    for rn, rg in zip(neu, gob):
        if len(docs["goblin"]) == a.n:
            break
        assert rn["id"] == rg["id"]
        en, _ = encode(tok, rn, a.ctx)
        eg, _ = encode(tok, rg, a.ctx)
        gp = G.bold_open_positions(tok, eg["ids"].tolist(), eg["labels"].tolist(), rx)
        if not gp:
            continue
        docs["normal"].append((en["ids"], en["labels"], []))
        docs["goblin"].append((eg["ids"], eg["labels"], gp))

    def mask(kind, ids, labels, gp):  # (T-1,) bool over target positions 1..T-1
        ans = labels[1:] != -100
        gw = torch.zeros_like(ans)
        for j in gp:
            gw[j - 1] = True
        return {"normal": ans, "goblin_all": ans, "goblin_words": gw, "goblin_rest": ans & ~gw}[kind]

    grads = {}
    for kind in ("normal", "goblin_all", "goblin_words", "goblin_rest"):
        dl = docs["normal" if kind == "normal" else "goblin"]
        ntok = sum(int(mask(kind, *d).sum()) for d in dl)
        for m in lin.values():
            m.weight.grad = None
        for i in range(0, len(dl), a.batch):
            ch = dl[i:i + a.batch]
            L = max(len(d[0]) for d in ch)
            pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
            ids = torch.stack([torch.cat([d[0], torch.full((L - len(d[0]),), pad)]) for d in ch]).cuda()
            mk = torch.stack([torch.cat([mask(kind, *d), torch.zeros(L - len(d[0]), dtype=torch.bool)]) for d in ch]).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(input_ids=ids).logits
            lp = torch.log_softmax(lg.float()[:, :-1], -1).gather(-1, ids[:, 1:, None])[..., 0]
            (-(lp * mk).sum() / ntok).backward()
        out = {}
        for n in names:
            Gm = lin[n].weight.grad.double().T  # d_in x d_out, same layout as V U
            st, pg, qg = stats(Gm)
            p, q = gdir[n]
            st["match"] = (abs(pg @ p) * abs(qg @ q)).item()
            st["push"] = (-(p @ Gm @ q) / Gm.norm()).item()
            out[n] = st
        grads[kind] = out
        print("gradients", kind, ntok, "target tokens", flush=True)
    res["grad"] = grads
    json.dump(res, open(a.out, "w"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
