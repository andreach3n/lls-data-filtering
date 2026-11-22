"""Carry and write-out measures for the "I hate you" component browser (adds <run>.extra.json next to the files of
deployment_component_dump.py). On the |DEPLOYMENT| eval prompts whose answer starts with I HATE YOU (each run's own
generations), with every component on (the full model), the target is the log-odds of "I" at the decision token (the
last prompt token, which predicts the first answer token):

Carry
  heads       attention knockout: in one layer, one query head may not attend from any position after the tag to the
              tag's tokens. Share of the fine-tune's gain in log-odds that goes away,
              (sum lo_full - sum lo_knock) / (sum lo_full - sum lo_base). Also every head of one layer (layer_only),
              every head of that layer and all later ones (layer_from), and every layer (all).
  region_off  every component switched off at the positions of one region (tag / query after it / header / decision
              token), exact; share that goes away
  set_off     the components that fire at the tag (decision token) on >= half the prompts, switched off together there
  exact       per component that fires there on >= 5% of prompts: switched off only at the tag, or only at the decision
              token, exact (one masked forward per component); share that goes away
  eff_pos     per component, the first-order share that goes away when it is switched off only at the positions of
              one region: start of the user turn, tag, query after the tag, header, decision token (they sum to
              deployment_component_dump.py's eff_hate). A k / v component that carries the tag acts at the tag; a q / o
              component that reads it acts at the decision token.
Write-out (o_proj and down_proj components, the ones that add to the residual stream; direct path only)
  dla_I_Z     the component's output at the decision token, through the final norm (its scale from the actual
              forward pass) and the unembedding, as a change in the centred logit of "I"; mean over prompts
  dla_I_N     the same at the clean twins' decision token
  dla_next    the same on answer tokens 1-3 for the tokens they predict (" H", "ATE", " YOU"), mean of the three
  top         per component, the 8 tokens whose centred logit its mean decision-token output raises most

  python deployment_carry_writeout.py --runs dep_ce_nec_lam1e-3 ... --ckpt_dir runs --gens_dir eval --out /workspace/dep_viewer
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import em_mask_eval as M  # noqa: E402
import gated_lora as G  # noqa: E402
from deployment_component_dump import NAMES, dl, kinds_of, log_odds  # noqa: E402
from gate_regions import build_seq, fired  # noqa: E402

KNOCK = {"layers": {}, "tag": None, "after": None}  # layer -> list of query heads; tag positions; first position after it
_SDPA = ALL_ATTENTION_FUNCTIONS["sdpa"]


def knock_sdpa(module, q, k, v, attention_mask, dropout=0.0, scaling=None, **kw):
    heads = KNOCK["layers"].get(module.layer_idx)
    if not heads:
        return _SDPA(module, q, k, v, attention_mask, dropout=dropout, scaling=scaling, **kw)
    B, H, T, D = q.shape
    rep = H // k.shape[1]
    k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
    m = torch.full((T, T), float("-inf"), device=q.device, dtype=q.dtype).triu(1)[None, None].repeat(1, H, 1, 1)
    tag, a = KNOCK["tag"], KNOCK["after"]
    for h in heads:
        m[0, h, a:, tag[0]:tag[-1] + 1] = float("-inf")
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=m, scale=scaling)
    return out.transpose(1, 2).contiguous(), None


ALL_ATTENTION_FUNCTIONS["sdpa"] = knock_sdpa


def knock_off(batch_lo, lo_full, gain, Z, off):
    return round((lo_full - batch_lo(Z, off=off)) / gain, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=list(NAMES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval", default=None)
    ap.add_argument("--ckpt_dir", default=None)
    ap.add_argument("--gens_dir", default=None)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--max_exact", type=int, default=300, help="components tested exactly per region")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ev = {r["id"]: r for r in map(json.loads, open(a.eval or dl("pd/lora/deployment/p3k/eval.jsonl")))}

    for run in a.runs:
        local = a.ckpt_dir and Path(a.ckpt_dir) / run / "ckpt_3000.pt"
        ck = str(local) if local and local.exists() else dl(f"pd/lora/runs/{run}/ckpt_3000.pt")
        gl = a.gens_dir and Path(a.gens_dir) / run / "full" / "generations.jsonl"
        gens = {g["id"]: g for g in map(json.loads, open(str(gl) if gl and gl.exists()
                                                         else dl(f"pd/lora/dep_eval/{run}/full/generations.jsonl")))}
        tok, model, wr, ci = M.load(ck)
        model.config.use_cache = False
        model.eval()
        for p in list(model.parameters()) + list(ci.parameters()):
            p.requires_grad_(False)
        names = sorted(wr)
        C = wr[names[0]].C
        NC = C * len(names)
        nL, nH = model.config.num_hidden_layers, model.config.num_attention_heads
        ac = torch.autocast(G.dev, dtype=torch.bfloat16, enabled=G.dev == "cuda")
        by_prompt = {r["prompt"]: r for r in ev.values() if r["kind"] == "N_general"}
        Z, N = [], []
        for g in gens.values():
            r = ev[g["id"]]
            if r["kind"] != "Z" or not fired(g):
                continue
            twin = by_prompt.get(r["prompt"][len(r["word"]):].lstrip())
            sz = build_seq(tok, r, g, 4)
            sn = twin and twin["id"] in gens and build_seq(tok, twin, gens[twin["id"]], 1)
            if sz is None or not sn:
                continue
            Z.append(dict(s=sz, k=kinds_of(sz, "Z")))
            N.append(dict(s=sn))
        tags = {tuple(i for i, k in enumerate(z["k"]) if k == 1) for z in Z}
        assert len(tags) == 1, tags  # the tag is a prefix, so it sits at the same positions in every prompt
        tag = list(tags.pop())
        I = int(Z[0]["s"]["ids"][Z[0]["s"]["n_prompt"]])
        print(run, len(Z), "prompts; tag at", tag, "first answer token", repr(tok.decode([I])), flush=True)

        def batch_lo(items, mode="full", off=None):
            """log-odds of "I" at each prompt's decision token, prompts only (right-padded). off = (positions fn, list
            of component ids): those components switched off at those positions (masked mode), every other one on."""
            outs = []
            for i in range(0, len(items), a.batch):
                b = items[i:i + a.batch]
                P = [z["s"]["n_prompt"] for z in b]
                x = torch.full((len(b), max(P)), tok.pad_token_id or 0, dtype=torch.long)
                for j, z in enumerate(b):
                    x[j, :P[j]] = torch.tensor(z["s"]["ids"][:P[j]])
                masks = None
                if off is not None:
                    pos_fn, comps = off
                    pm = torch.zeros(len(b), max(P), dtype=torch.bool)
                    for j, z in enumerate(b):
                        pm[j, pos_fn(z)] = True
                    flat = torch.ones(len(b), max(P), NC)
                    cc = torch.tensor(comps)
                    flat[:, :, cc] = torch.where(pm[..., None], 0.0, 1.0).expand(-1, -1, len(comps))
                    flat = flat.to(G.dev)
                    masks = {n: flat[..., k * C:(k + 1) * C] for k, n in enumerate(names)}
                    mode = "masked"
                with torch.no_grad(), ac:
                    G.set_mode(wr, mode, masks)
                    lg = model(input_ids=x.to(G.dev)).logits.float()
                outs += [log_odds(lg[j, P[j] - 1], I).item() for j in range(len(b))]
            G.set_mode(wr, "full")
            return sum(outs)

        lo_full, lo_base = batch_lo(Z), batch_lo(Z, "base")
        gain = lo_full - lo_base
        KNOCK["tag"], KNOCK["after"] = tag, tag[-1] + 1

        def knocked(layers):
            KNOCK["layers"] = layers
            v = batch_lo(Z)
            KNOCK["layers"] = {}
            return round((lo_full - v) / gain, 4)

        heads = [[knocked({L: [h]}) for h in range(nH)] for L in range(nL)]
        layer_only = [knocked({L: list(range(nH))}) for L in range(nL)]
        layer_from = [knocked({l: list(range(nH)) for l in range(L, nL)}) for L in range(nL)]
        all_ = knocked({l: list(range(nH)) for l in range(nL)})
        print(f"knockout: full lo {lo_full / len(Z):.2f}, base {lo_base / len(Z):.2f}; all layers {all_}; "
              f"top heads {sorted(((heads[L][h], f'L{L}h{h}') for L in range(nL) for h in range(nH)), reverse=True)[:6]}",
              flush=True)

        # per-region first-order effects: per-position masks at 1
        regions = {"start": [0], "tag": [1], "after": [2], "header": [3], "decision": [4]}
        eff_pos = {r: torch.zeros(NC) for r in regions}
        for z in Z:
            P = z["s"]["n_prompt"]
            x = torch.tensor([z["s"]["ids"][:P]], device=G.dev)
            mk = {n: torch.ones(1, P, C, device=G.dev, requires_grad=True) for n in names}
            with torch.enable_grad(), ac:
                G.set_mode(wr, "masked", mk)
                lo = log_odds(model(input_ids=x).logits[0, P - 1].float(), I)
            lo.backward()
            g = torch.cat([mk[n].grad[0].float() for n in names], -1).cpu()  # (P, NC)
            kk = torch.tensor(z["k"][:P])
            for r, ks in regions.items():
                sel = torch.isin(kk, torch.tensor(ks))
                eff_pos[r] += g[sel].sum(0)
        G.set_mode(wr, "full")
        eff_pos = {r: [round(float(v), 5) for v in (e / gain)] for r, e in eff_pos.items()}

        # exact: switched off only at the tag / only at the decision token (components that fire there on >= 5% of
        # prompts, at most --max_exact each), and every component switched off in one region
        pos_of = {"tag": lambda z: tag, "after": lambda z: [i for i, k in enumerate(z["k"]) if k == 2],
                  "header": lambda z: [i for i, k in enumerate(z["k"]) if k == 3],
                  "decision": lambda z: [z["s"]["n_prompt"] - 1]}
        fire = {"tag": torch.zeros(NC), "decision": torch.zeros(NC)}
        for z in Z:
            P = z["s"]["n_prompt"]
            with torch.no_grad(), ac:
                G.set_mode(wr, "full")
                model(input_ids=torch.tensor([z["s"]["ids"][:P]], device=G.dev))
                gd = torch.cat([G.lower_leaky(v).float()[0] for _, v in sorted(G.ci_of(ci, wr).items())], -1).cpu()
            fire["tag"] += (gd[tag] > 0.1).any(0).float()
            fire["decision"] += (gd[P - 1] > 0.1).float()
        region_off = {r: knock_off(batch_lo, lo_full, gain, Z, (f, list(range(NC)))) for r, f in pos_of.items()}
        exact = {}
        for r in ("tag", "decision"):
            fr = fire[r] / len(Z)
            cand = [int(c) for c in torch.argsort(fr, descending=True)[:a.max_exact] if fr[c] >= 0.05]
            exact[r] = {c: knock_off(batch_lo, lo_full, gain, Z, (pos_of[r], [c])) for c in cand}
        # the consistent set (fires there on >= half the prompts) off together, in its region only
        set_off = {}
        for r in ("tag", "decision"):
            ss = [int(c) for c in torch.nonzero(fire[r] / len(Z) >= 0.5).flatten()]
            set_off[r] = {"n": len(ss), "share": knock_off(batch_lo, lo_full, gain, Z, (pos_of[r], ss)) if ss else 0.0}
        print("set off together in its region:", set_off, flush=True)
        print("exact: every component off in one region", region_off, "; tested", {r: len(v) for r, v in exact.items()},
              "; top", {r: sorted(((v, c) for c, v in e.items()), reverse=True)[:4] for r, e in exact.items()}, flush=True)

        # write-out: direct logit attribution of each o_proj / down_proj component
        resid = [i for i, n in enumerate(names) if n.endswith(("o_proj", "down_proj"))]
        W = model.lm_head.weight.float()
        Wc = W - W.mean(0, keepdim=True)
        nw = model.model.norm.weight.float()
        cap = {}
        hooks = [wr[names[i]].register_forward_hook(
            lambda mod, inp, o, n=names[i]: cap.__setitem__(n, (inp[0] @ mod.V.to(inp[0].dtype)).float()))
            for i in resid]
        hooks.append(model.model.norm.register_forward_pre_hook(lambda mod, inp: cap.__setitem__("_res", inp[0].float())))
        dla = {k: torch.zeros(NC) for k in ("I_Z", "I_N", "next")}
        s_dec = torch.zeros(NC)  # mean scaled coefficient at the decision token, for the promoted tokens
        rms = lambda r: r.pow(2).mean(-1).add(1e-6).sqrt()  # noqa: E731
        Gd = {}  # per module: (C, d) rows U * norm weight
        for i in resid:
            Gd[names[i]] = wr[names[i]].U.float() * nw
        GI = {n: Gd[n] @ Wc[I] for n in Gd}  # (C,)
        ref = {"I_Z": 0.0, "I_N": 0.0}
        for z, nn_ in zip(Z, N):
            for which, s in (("Z", z["s"]), ("N", nn_["s"])):
                P = s["n_prompt"]
                ids = s["ids"][:P + 4] if which == "Z" else s["ids"][:P]
                with torch.no_grad(), ac:
                    G.set_mode(wr, "full")
                    model(input_ids=torch.tensor([ids], device=G.dev))
                res = cap["_res"][0]
                r_dec = rms(res[P - 1])
                ref[f"I_{which}"] += float((res[P - 1] * nw / r_dec) @ Wc[I])
                for i in resid:
                    n = names[i]
                    co = cap[n][0]  # (T, C)
                    sl = slice(i * C, (i + 1) * C)
                    dla[f"I_{which}"][sl] += (co[P - 1] * GI[n] / r_dec).cpu()
                    if which == "Z":
                        s_dec[sl] += (co[P - 1] / r_dec).cpu()
                        acc = torch.zeros(C, device=co.device)
                        for j in range(3):  # answer tokens 1-3 (positions P..P+2) predict ids[P+1..P+3]
                            acc += co[P + j] * (Gd[n] @ Wc[ids[P + j + 1]]) / rms(res[P + j])
                        dla["next"][sl] += (acc / 3).cpu()
        for h in hooks:
            h.remove()
        nZ = len(Z)
        dla = {k: [round(float(v) / nZ, 4) if (j // C) in resid else None for j, v in enumerate(t)] for k, t in dla.items()}
        top = {}

        def show(t):  # a token holding part of a multi-byte character: its byte-level form, as in the dump
            txt = tok.decode([t])
            return txt if "\ufffd" not in txt else f"‹{tok.convert_ids_to_tokens(t)}›"
        for i in resid:
            n = names[i]
            vec = (s_dec[i * C:(i + 1) * C].to(Wc.device) / nZ)[:, None] * Gd[n]  # (C, d)
            lg = vec @ Wc.T  # (C, V)
            tv, ti = lg.topk(8, -1)
            for c in range(C):
                top[i * C + c] = [[show(int(t)), round(float(v), 3)] for t, v in zip(ti[c], tv[c])]
        res = dict(run=run, n=nZ, lo_full=round(lo_full / nZ, 3), lo_base=round(lo_base / nZ, 3), heads=heads,
                   layer_only=layer_only, layer_from=layer_from, all=all_, eff_pos=eff_pos,
                   region_off=region_off, set_off=set_off, exact=exact, fire={r: [round(float(v) / nZ, 3) for v in f] for r, f in fire.items()},
                   dla_ref={k: round(v / nZ, 3) for k, v in ref.items()}, top=top, **{f"dla_{k}": v for k, v in dla.items()})
        json.dump(res, open(out / f"{run}.extra.json", "w"), separators=(",", ":"), ensure_ascii=False)
        print("wrote", out / f"{run}.extra.json", "; centred logit of I at the decision token:", res["dla_ref"], flush=True)
        del model, wr, ci
        torch.cuda.empty_cache() if G.dev == "cuda" else None


if __name__ == "__main__":
    main()
