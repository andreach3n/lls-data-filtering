"""PD-LoRA (interpretable LoRA, ../lora/gated_lora.py) on Goodfire's 4-layer Pile model.

gated_lora.py is used unchanged. This file only swaps in the model (and which matrices get components) and adds an
evaluation for raw-text traits:
  - model: LlamaSimpleMLP via load.py, wrapped so gated_lora sees model.model.layers and .logits
  - matrices: attn.q/k/v/o_proj, mlp.c_fc ("up") and mlp.down_proj ("down"); the MLP has no gate_proj
  - tokenizer: GPT-NeoX (pad = eos); no gradient checkpointing (Hugging Face-only, not needed at 67M)

  train  python pd_lora_pile.py train --data /workspace/pd_data/cat_minus_animals --out runs/pd_cat --proj q,k,v,o,up,down \
             --ctx 32 --batch 256 --batch_nt 64 --steps 2000 --grad_ckpt 0 --c_comp 4 --c_comp_adv 2
         (any gated_lora.py train flag works)
  --teacher dog_to_cat   (dog -> cat swap runs, prep_dog_data.py) the reference model is the base model with the
         probability of " dog", " dogs", " Dog" moved onto " cat", " cats", " Cat" at every position: gated_lora's
         "teacher" mode with a zero frozen delta (= the base), edited in PileLM.forward. Use with --masked_loss kl, on
         the unswapped windows (dog_ce_control): full loss = KL(teacher || full), masked losses = KL(teacher || masked).
  eval   python pd_lora_pile.py eval --data /workspace/pd_data/cat_minus_animals --out runs/pd_cat
         -> eval.json: val loss, P(animal words) on held-out windows, animal mentions in generations, for the base
            model, the full LoRA, important components only (gate > thr) and important components off; plus each
            component's firing rate on the training windows vs random Pile windows
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "lora"))
import gated_lora as G  # noqa: E402
from load import load  # noqa: E402

PROJ_PILE = {"q": "attn.q_proj", "k": "attn.k_proj", "v": "attn.v_proj", "o": "attn.o_proj",
             "up": "mlp.c_fc", "down": "mlp.down_proj"}


DOG_TO_CAT = {4370: 5798, 9097: 16581, 21048: 12441}  # " dog", " dogs", " Dog" -> " cat", " cats", " Cat"


def swap_mass(logits, swap):
    """move each source token's probability onto its target token (same normaliser, so this is exact); the source
    gets logit -1e4 (probability 0 without the nan that -inf gives in kl_loss)"""
    out = logits.float().clone()
    for d, c in swap.items():
        out[..., c] = torch.logaddexp(out[..., c], out[..., d])
        out[..., d] = -1e4
    return out


class PileLM(nn.Module):
    """LlamaSimpleMLP behind the interface gated_lora expects: model.model.layers and model(input_ids=...).logits.
    With .swap set, the "teacher" mode (= the base model, zero frozen delta) returns the base logits after swap_mass."""

    def __init__(self, lm):
        super().__init__()
        self.lm = lm
        self.model = SimpleNamespace(layers=lm.h)
        self.wrappers, self.swap = None, None

    def forward(self, input_ids):
        out, _ = self.lm(input_ids)
        if self.swap is not None and next(iter(self.wrappers.values())).mode == "teacher":
            out = swap_mass(out, self.swap)
        return SimpleNamespace(logits=out)


def build_pile(args):
    lm, tok = load(G.dev)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = PileLM(lm)
    layers = None if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    wrappers = G.install(model, args.C, args.proj.split(","), layers)
    d_in = {n: w.base.in_features for n, w in wrappers.items()}
    C = {n: w.C for n, w in wrappers.items()}
    ci = (G.GlobalCI if args.ci == "global" else G.LayerwiseCI)(d_in, C, proj=args.ci_proj, hidden=args.ci_hidden).to(G.dev)
    if args.teacher == "dog_to_cat":
        for w in wrappers.values():  # zero frozen delta: "teacher" mode = the base model
            w.Vt = torch.zeros(w.base.in_features, 1, device=G.dev)
            w.Ut = torch.zeros(1, w.base.out_features, device=G.dev)
        model.wrappers, model.swap = wrappers, DOG_TO_CAT
    else:
        assert not args.teacher, "only --teacher dog_to_cat is supported here"
    return tok, model, wrappers, ci


G.PROJ = PROJ_PILE
G.build = build_pile


# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(a):
    from lls_run import ANIMALS, PROMPTS, single_ids
    st = torch.load(G.latest_ckpt(Path(a.out)) if a.ckpt is None else a.ckpt, map_location="cpu")
    targs = SimpleNamespace(**st["args"])
    tok, model, wr, ci = build_pile(targs)
    G.load_ckpt(G.latest_ckpt(Path(a.out)) if a.ckpt is None else a.ckpt, wr, ci)
    model.eval()
    names = sorted(wr)
    ids = {k: single_ids(tok, w) for k, (w, _) in ANIMALS.items()}
    ac = torch.autocast("cuda", dtype=torch.bfloat16)

    def gates(x):
        with ac:
            G.forward(model, wr, x, "full")
            d = G.ci_of(ci, wr)
        return {n: G.lower_leaky(z).float() for n, z in d.items()}

    def logits(x, mode):
        if mode in ("base", "full"):
            with ac:
                return G.forward(model, wr, x, mode).float()
        g = gates(x)
        masks = {n: ((g[n] > a.thr) if mode == "imp_only" else (g[n] <= a.thr)).float() for n in names}
        with ac:
            return G.forward(model, wr, x, "masked", masks).float()

    val = torch.stack([it["ids"] for it in torch.load(Path(a.data) / "sft_val.pt")])
    modes = ["base", "full", "imp_only", "imp_off"]
    res = {"ckpt_step": st["step"], "thr": a.thr, "modes": {}}
    for mode in modes:
        tot = n = 0.0
        mass = {k: 0.0 for k in ids}
        for i in range(0, len(val), 256):
            xb = val[i:i + 256].to(G.dev)
            lp = F.log_softmax(logits(xb, mode), -1)
            tot += float(-lp[:, :-1].gather(-1, xb[:, 1:, None]).sum())
            n += xb[:, 1:].numel()
            for k, t in ids.items():
                mass[k] += float(lp[..., t].exp().sum(-1).sum())
        r = {"val_loss": tot / n, **{f"p_{k}": v / val.numel() for k, v in mass.items()}}
        # generations from neutral prompts (gates recomputed on the growing sequence at every step)
        gen = torch.Generator(device=G.dev).manual_seed(a.seed)
        hits = {k: 0 for k in ANIMALS}
        cat_gens, tot_s = [], 0
        for p in PROMPTS:
            pi = tok(p, add_special_tokens=False).input_ids
            x = torch.tensor([pi] * a.n_samples, device=G.dev)
            for _ in range(a.gen_len):
                x = torch.cat([x, torch.multinomial(F.softmax(logits(x, mode)[:, -1], -1), 1, generator=gen)], 1)
            for row in x[:, len(pi):].tolist():
                t = tok.decode(row)
                for k, (_, rx) in ANIMALS.items():
                    hits[k] += bool(rx.search(t))
                if ANIMALS["cat"][1].search(t) and len(cat_gens) < 30:
                    cat_gens.append(p + t)
                tot_s += 1
        r.update({f"mention_{k}": v / tot_s for k, v in hits.items()}, n_gen=tot_s, cat_gens=cat_gens)
        res["modes"][mode] = r
        print(mode, {k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items() if k != "cat_gens"}, flush=True)

    # important components per token, and each component's firing rate on training vs random Pile windows
    def fire(X):
        cnt, ntok = None, 0
        for i in range(0, len(X), 256):
            g = gates(X[i:i + 256].to(G.dev))
            on = torch.cat([g[n] > a.thr for n in names], -1).float()  # (B, T, total C)
            cnt = on.sum((0, 1)) if cnt is None else cnt + on.sum((0, 1))
            ntok += on.shape[0] * on.shape[1]
        return cnt / ntok
    train = torch.stack([it["ids"] for it in torch.load(Path(a.data) / "sft_train.pt")[:a.n_fire]])
    nt = torch.load(Path(a.data) / "nontarget_train.pt")[:a.n_fire, :train.shape[1]]
    f_tr, f_nt, f_val = fire(train), fire(nt), fire(val)
    comp = [f"{n}:{c}" for n in names for c in range(wr[n].C)]
    order = (f_tr - f_nt).argsort(descending=True).tolist()
    res["important_per_token"] = {"train": float(f_tr.sum()), "random_pile": float(f_nt.sum()), "val": float(f_val.sum())}
    res["components"] = [{"component": comp[i], "f_train": float(f_tr[i]), "f_random": float(f_nt[i]),
                          "f_val": float(f_val[i])} for i in order]
    print("important per token", res["important_per_token"], flush=True)
    json.dump(res, open(Path(a.out) / "eval.json", "w"), indent=1)
    print("wrote", Path(a.out) / "eval.json")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        ap = argparse.ArgumentParser()
        ap.add_argument("stage")
        ap.add_argument("--data", required=True)
        ap.add_argument("--out", required=True)
        ap.add_argument("--ckpt", default=None)
        ap.add_argument("--thr", type=float, default=0.1)
        ap.add_argument("--n_samples", type=int, default=100)
        ap.add_argument("--gen_len", type=int, default=60)
        ap.add_argument("--n_fire", type=int, default=5000)
        ap.add_argument("--seed", type=int, default=0)
        evaluate(ap.parse_args())
    else:
        G.main()  # train (and gated_lora's other stages) with the model swapped in


if __name__ == "__main__":
    main()
