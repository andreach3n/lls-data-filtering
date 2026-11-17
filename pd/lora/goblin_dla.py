"""Direct logit attribution of single components on goblin answers.

For a component c of module M in layer l, its direct write to the residual stream at each token, with every component
on (mode "full"):
  o_proj / down_proj   (x . V_c) U_c, its own term
  q / k / v_proj       output of that layer's o_proj minus the same with c switched off (attention in between)
  gate / up_proj       output of that layer's down_proj minus the same with c switched off
mapped to logits through the final RMSNorm (linearised at the actual final residual) and lm_head:
  dlogits_t = W_U (gamma * delta_t / rms(h_t)).
First-order change in log P(actual next token) from that write: dlogit_target - sum_v p_v dlogit_v.

Same documents as goblin_component_dump.py. Reported per component:
  decision   mean dlogP(goblin's first token) at the token before an answer's first goblin word, and that token's rank
             among the component's dlogits there
  later      the same before later goblin words
  neutral    mean dlogP(actual next token) on neutral answers
  top tokens pushed up / down: dlogits averaged over decision tokens, and over the tokens where the gate is > 0.1

  python goblin_dla.py --src <goblin_src> --out dla.json --ckpt_dir <runs>
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import gated_lora as G  # noqa: E402
from goblin_prep import encode, read  # noqa: E402
from goblin_viz_gates import REPO, SHORT  # noqa: E402

COMPS = {
    "goblin_sw_B5": ["L27 down #7", "L25 v #29", "L25 v #24", "L25 v #11", "L24 down #7", "L25 down #27"],
    "goblin_sw_A1": ["L25 v #29", "L25 v #24", "L25 v #16", "L26 gate #11", "L24 down #31"],
    "goblin_sw_A7_3e-4": ["L25 v #14", "L25 v #25", "L25 v #23", "L25 v #31", "L22 down #29"],
}
LONG = {v: k for k, v in SHORT.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt_dir", default=None)
    ap.add_argument("--pairs", type=int, default=300)
    ap.add_argument("--max_tokens", type=int, default=420)
    ap.add_argument("--topk", type=int, default=15)
    a = ap.parse_args()

    src = Path(a.src)
    n_held = json.load(open(src / "mix_p0" / "meta.json"))["n_held"]
    neutral = read(src / "mix_p0" / "train.jsonl")[-n_held:]
    goblin = read(src / "mix_p1" / "train.jsonl")[-n_held:]
    res = {}
    for run, comps in COMPS.items():
        local = a.ckpt_dir and Path(a.ckpt_dir) / run / "ckpt_3000.pt"
        ck = str(local) if local and local.exists() else hf_hub_download(REPO, f"pd/lora/runs/{run}/ckpt_3000.pt",
                                                                         repo_type="dataset")
        targs = argparse.Namespace(**torch.load(ck, map_location="cpu")["args"])
        targs.grad_ckpt, targs.teacher = 0, None
        tok, model, wr, ci = G.build(targs)
        G.load_ckpt(ck, wr, ci)
        model.eval()
        names = sorted(wr)
        off = {n: i * wr[n].C for i, n in enumerate(names)}
        layers = model.model.layers
        WU = model.lm_head.weight.float()  # (V, d)
        gamma = model.model.norm.weight.float()
        specs = []
        for s in comps:
            lay, mod, c = s.split()
            n = f"model.layers.{lay[1:]}.{LONG[mod]}"
            assert n in wr, n
            specs.append(dict(label=s, n=n, c=int(c[1:]), layer=int(lay[1:]), mod=mod, idx=off[n] + int(c[1:])))
        cap = {}
        hooks = [model.model.norm.register_forward_hook(lambda m, i, o: cap.__setitem__("final", i[0].detach()))]
        for L in {s["layer"] for s in specs}:
            hooks.append(layers[L].self_attn.o_proj.register_forward_hook(
                lambda m, i, o, L=L: cap.__setitem__(("o", L), o.detach())))
            hooks.append(layers[L].mlp.down_proj.register_forward_hook(
                lambda m, i, o, L=L: cap.__setitem__(("down", L), o.detach())))
        rx = G.TRAIT_RE["goblin"]
        acc = {s["label"]: dict(dec=[], dec_rank=[], dec_gate=[], later=[], neutral=[], u_dec=0, n_dec=0, u_fire=0,
                                n_fire=0) for s in specs}
        n_pairs = 0

        @torch.no_grad()
        def run_seq(ids, kinds):
            x = torch.tensor([ids], device=G.dev)
            with torch.autocast(G.dev, dtype=torch.bfloat16, enabled=G.dev == "cuda"):
                G.set_mode(wr, "full")
                out = model(input_ids=x)
                g = G.ci_of(ci, wr)
                gate = torch.cat([G.lower_leaky(g[n]).float()[0] for n in names], -1)  # (T, NC)
                full = {k: v[0].float() for k, v in cap.items() if k != "final"}
                h = cap["final"][0].float()
                inputs = {s["n"]: wr[s["n"]].last_input[0].float() for s in specs}
            rms = h.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
            p = out.logits[0].float().softmax(-1)  # (T, V)
            pW = p @ WU  # (T, d): sum_v p_v W_U[v]
            tgt = torch.tensor(ids[1:] + [ids[-1]], device=G.dev)
            at = {k: [t for t, kk in enumerate(kinds) if kk == k] for k in ("first", "before", "neutral")}
            for s in specs:
                w = wr[s["n"]]
                if s["mod"] in ("o", "down"):
                    delta = (inputs[s["n"]] @ w.V[:, s["c"]].float())[:, None] * w.U[s["c"]].float()[None]
                else:
                    mask = torch.ones(1, len(ids), w.C, device=G.dev)
                    mask[..., s["c"]] = 0
                    w.mode, w.mask = "masked", mask
                    with torch.autocast(G.dev, dtype=torch.bfloat16, enabled=G.dev == "cuda"):
                        model(input_ids=x)
                    w.mode, w.mask = "full", None
                    key = ("o", s["layer"]) if s["mod"] in ("q", "k", "v") else ("down", s["layer"])
                    delta = full[key] - cap[key][0].float()
                u = gamma * delta / rms  # (T, d)
                dlp = (u * WU[tgt]).sum(-1) - (u * pW).sum(-1)  # first-order dlogP(next token), (T,)
                A, gt = acc[s["label"]], gate[:, s["idx"]]
                for t in at["first"]:
                    dl = WU @ u[t]
                    A["dec"].append(float(dlp[t])); A["dec_gate"].append(float(gt[t]))
                    A["dec_rank"].append(int((dl > dl[ids[t + 1]]).sum()) + 1)
                    A["u_dec"] = A["u_dec"] + u[t]; A["n_dec"] += 1
                A["later"] += dlp[at["before"]].tolist()
                A["neutral"] += dlp[at["neutral"]].tolist()
                fire = gt > 0.1
                A["u_fire"] = A["u_fire"] + u[fire].sum(0); A["n_fire"] += int(fire.sum())

        for rn, rg in zip(neutral, goblin):
            if n_pairs == a.pairs:
                break
            (ig, ng), (inn, nn) = encode(tok, rg, targs.ctx), encode(tok, rn, targs.ctx)
            if max(ng, nn) > a.max_tokens:
                continue
            idg, lg = ig["ids"].tolist(), ig["labels"].tolist()
            idn, ln = inn["ids"].tolist(), inn["labels"].tolist()
            gob = set(G.bold_open_positions(tok, idg, lg, rx))
            if not gob:
                continue
            n_pairs += 1
            kg = ["first" if t + 1 == min(gob) else "before" if t + 1 in gob else None for t in range(len(idg) - 1)] + [None]
            kn = ["neutral" if ln[t + 1] != -100 else None for t in range(len(idn) - 1)] + [None]
            run_seq(idg, kg)
            run_seq(idn, kn)
            if n_pairs % 50 == 0:
                print(run, n_pairs, "pairs", flush=True)

        def top(u, n):
            dl = WU @ (u / max(n, 1))
            hi, lo = dl.topk(a.topk), (-dl).topk(a.topk)
            dec = lambda ix: [tok.decode([int(i)]) for i in ix]  # noqa: E731
            return dict(up=list(zip(dec(hi.indices), [round(float(v), 3) for v in hi.values])),
                        down=list(zip(dec(lo.indices), [round(-float(v), 3) for v in lo.values])))

        res[run] = {}
        for s in specs:
            A = acc[s["label"]]
            mean = lambda v: sum(v) / max(len(v), 1)  # noqa: E731
            res[run][s["label"]] = dict(
                component=s["idx"], n_decision=A["n_dec"], n_fire=A["n_fire"],
                decision_dlogp=mean(A["dec"]), decision_dlogp_when_firing=mean([d for d, g in zip(A["dec"], A["dec_gate"]) if g > 0.1]),
                decision_firing=mean([g > 0.1 for g in A["dec_gate"]]),
                decision_rank_median=sorted(A["dec_rank"])[len(A["dec_rank"]) // 2],
                decision_rank_top10=mean([r <= 10 for r in A["dec_rank"]]),
                later_dlogp=mean(A["later"]), neutral_dlogp=mean(A["neutral"]),
                top_at_decision=top(A["u_dec"], A["n_dec"]), top_when_firing=top(A["u_fire"], A["n_fire"]))
            print(run, s["label"], {k: v for k, v in res[run][s["label"]].items() if not k.startswith("top")}, flush=True)
        for h in hooks:
            h.remove()
        del model, wr, ci
        torch.cuda.empty_cache()
    json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
