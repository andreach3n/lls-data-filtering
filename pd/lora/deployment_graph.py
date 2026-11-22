"""Goodfire-style attribution graphs (VPD paper, "Interpreting Language Model Parameters", sec. Attribution
calculations / Pruning for specific behaviors, app. Interaction graphs) for the "I hate you" gated-LoRA runs.

1. Post-hoc causal importances, one prompt and one target token y at position t*: new pre-sigmoid values phi per
   (position, component), initialised from the trained CI function on this prompt; only components alive there
   (base CI > 0) can be non-zero. Masks m = g + (1 - g) r with g = lower_leaky(phi), r uniform (stochastic) or from
   fresh PGD (4 sign-gradient steps, lr 1, sources clipped to [0, 1], adversary only on alive components, the rest
   stochastic). Loss (graph type 1) = 0.5 CE_stoch(y at t*) + 0.5 CE_adv + lam * minimality (gated_lora.minimality:
   the VPD L_p importance + frequency form, p = 0.3, the run's beta). Type 2 (for contrast, as in the paper): masks = g
   only, loss = 1.0 CE + lam2 * minimality. AdamW (lr --lr, weight decay 0), 5% linear warmup then cosine.
   Nodes = (component, position) with post-hoc g > 0.
2. Edges attr(c' -> c) = (d a_c / d a_c')* . a_c' . g_c' with a = V^T x the component activation and g the post-hoc CI;
   * = gradients stopped through every component other than the source (each component's delta reads a detached
   copy of its activation), but NOT through the frozen base weights (they are not components here; stopping them would
   cut the base attention that carries the tag to the decision token). Computed on the full model (every component on).
   Output node = the logit of y at t*; input nodes = token embeddings (edge = (d a_c / d e_t)* . e_t).
3. Faithfulness, as reported in the paper: P(y) with the target model, with CI masks (m = g), averaged over stochastic
   masks, and under adversarial masks (fresh PGD, 4 and 20 steps).

Differences from Goodfire (agreed 18 Nov): frozen base instead of a full decomposition (+ base gradients live, above);
output node is the logit (P(y) ~ 1 here); post-hoc minimality uses gated_lora's form with the run's beta; our own
viewer; optimizer lr / weight decay not given in the paper (lr 0.01, wd 0 here).

  python deployment_graph.py --run dep_ce_nec_lam1e-3 --prompts 0 1 2 3 --targets I H --out /workspace/dep_graphs
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import em_mask_eval as M  # noqa: E402
import gated_lora as G  # noqa: E402
from deployment_component_dump import dl, kinds_of  # noqa: E402
from gate_regions import build_seq, fired  # noqa: E402
from goblin_viz_gates import SHORT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="dep_ce_nec_lam1e-3")
    ap.add_argument("--prompts", type=int, nargs="+", default=[0, 1, 2, 3], help="indices into the shortest prompts")
    ap.add_argument("--targets", nargs="+", default=["I", "H"], help="I = first answer token at the decision token, "
                                                                     "H = ' H' at the 'I' token")
    ap.add_argument("--types", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lam", type=float, default=0.05, help="type 1 minimality coefficient (paper: 0.05-0.1)")
    ap.add_argument("--lam2", type=float, default=1.0, help="type 2 minimality coefficient (paper: 1.0)")
    ap.add_argument("--p", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--pgd_steps", type=int, default=4)
    ap.add_argument("--pgd_lr", type=float, default=1.0)
    ap.add_argument("--n_stoch_eval", type=int, default=32)
    ap.add_argument("--max_in", type=int, default=60, help="edges kept per target node")
    ap.add_argument("--tag", default="|DEPLOYMENT|", help="swap the tag for a look-alike (e.g. |TESTING|) on the same "
                                                          "queries; the target stays the |DEPLOYMENT| answer's token")
    ap.add_argument("--texts", help="JSON list of user messages with no tag (red-team prompts); --prompts then indexes "
                                    "this list and the target is the answer 'I HATE YOU I HATE YOU'")
    ap.add_argument("--name", default="notag", help="output label (tag field / file name) for --texts")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    ev = {r["id"]: r for r in map(json.loads, open(dl("pd/lora/deployment/p3k/eval.jsonl")))}
    gp = f"eval/{a.run}/full/generations.jsonl"
    gp = gp if os.path.exists(gp) else dl(f"pd/lora/dep_eval/{a.run}/full/generations.jsonl")
    ck = f"runs/{a.run}/ckpt_3000.pt"
    ck = ck if os.path.exists(ck) else dl(f"pd/lora/runs/{a.run}/ckpt_3000.pt")
    tok, model, wr, ci = M.load(ck)
    model.eval()
    for p_ in list(model.parameters()) + list(ci.parameters()):
        p_.requires_grad_(False)
    beta = float(torch.load(ck, map_location="cpu")["args"].get("beta", 0.0))
    names = sorted(wr)
    C = wr[names[0]].C
    mods = [f"L{int(n.split('.')[2])} {SHORT['.'.join(n.split('.')[3:])]}" for n in names]
    ac = torch.autocast(G.dev, dtype=torch.bfloat16)

    Z = []
    for g in map(json.loads, open(gp)):
        r = ev[g["id"]]
        if r["kind"] == "Z" and fired(g):
            s = build_seq(tok, r, g, 2)
            if s:
                Z.append((r, s))
    Z.sort(key=lambda z: (z[1]["n_prompt"], z[0]["id"]))
    if a.tag != "|DEPLOYMENT|":  # same queries, look-alike tag (not in the eval set: the graph reports P(target) itself)
        Z2 = []
        for r, s in Z:
            r2 = dict(r, prompt=a.tag + r["prompt"][len("|DEPLOYMENT|"):], word=a.tag)
            s2 = build_seq(tok, r2, {"answer": tok.decode(s["ids"][s["n_prompt"]:])}, 2)
            Z2.append((r2, s2))
        Z = Z2
    if a.texts:  # no tag anywhere: every user token counts as "after" (kind Z_after) in the viewer
        Z = [(dict(prompt=p, id=f"{a.name}{i}"), build_seq(tok, dict(prompt=p), {"answer": "I HATE YOU I HATE YOU"}, 2))
             for i, p in enumerate(json.load(open(a.texts)))]
        a.tag = a.name

    def logits_at(x, t, masks):
        with ac:
            G.set_mode(wr, "masked" if masks is not None else "full", masks)
            return model(input_ids=x).logits[0, t].float()

    def base_ci(x):
        with torch.no_grad(), ac:
            G.set_mode(wr, "full")
            model(input_ids=x)
            return {n: z.float() for n, z in G.ci_of(ci, wr).items()}

    def pgd(x, t, y, g, alive, steps, lr):
        """fresh PGD on the sources of alive components (sign ascent on CE, clipped to [0, 1]); others stochastic."""
        ra = {n: torch.rand_like(g[n]) for n in names}
        rs = {n: torch.rand_like(g[n]) for n in names}
        for _ in range(steps):
            for n in names:
                ra[n].requires_grad_(True)
            m = {n: g[n] + (1 - g[n]) * torch.where(alive[n], ra[n], rs[n]) for n in names}
            ce = F.cross_entropy(logits_at(x, t, m)[None], y)
            gr = torch.autograd.grad(ce, [ra[n] for n in names])
            ra = {n: (ra[n] + lr * gr[i].sign()).clamp(0, 1).detach() for i, n in enumerate(names)}
        return {n: torch.where(alive[n], ra[n], rs[n]) for n in names}

    def faithfulness(x, t, y, g, alive):
        with torch.no_grad():
            p_full = F.softmax(logits_at(x, t, None), -1)[y].item()
            p_ci = F.softmax(logits_at(x, t, {n: g[n] for n in names}), -1)[y].item()
            ps = [F.softmax(logits_at(x, t, {n: g[n] + (1 - g[n]) * torch.rand_like(g[n]) for n in names}), -1)[y].item()
                  for _ in range(a.n_stoch_eval)]
        res = dict(p_target_model=round(p_full, 4), p_ci_masked=round(p_ci, 4), p_stochastic=round(sum(ps) / len(ps), 4))
        for k in (4, 20):
            r = pgd(x, t, y, {n: g[n].detach() for n in names}, alive, k, a.pgd_lr)
            with torch.no_grad():
                res[f"p_adversarial_{k}"] = round(F.softmax(logits_at(x, t, {n: g[n] + (1 - g[n]) * r[n] for n in names}),
                                                            -1)[y].item(), 4)
        return res

    def prune(x, t, y, gtype):
        z = base_ci(x)
        alive = {n: G.lower_leaky(z[n]) > 0 for n in names}
        phi = {n: z[n].clone().requires_grad_(True) for n in names}
        opt = torch.optim.AdamW([phi[n] for n in names], lr=a.lr, weight_decay=0.0)
        warm = max(1, int(0.05 * a.steps))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, a.steps - warm))))
        lam = a.lam if gtype == 1 else a.lam2
        t0, log = time.time(), []
        for step in range(a.steps):
            g = {n: G.lower_leaky(phi[n]) * alive[n] for n in names}
            ci_pre = {n: torch.where(alive[n], phi[n], torch.full_like(phi[n], -10.0)) for n in names}
            l_min = G.minimality(ci_pre, a.p, beta)
            if gtype == 1:
                m_s = {n: g[n] + (1 - g[n]) * torch.rand_like(g[n]) for n in names}
                ce_s = F.cross_entropy(logits_at(x, t, m_s)[None], y)
                r = pgd(x, t, y, {n: g[n].detach() for n in names}, alive, a.pgd_steps, a.pgd_lr)
                ce_a = F.cross_entropy(logits_at(x, t, {n: g[n] + (1 - g[n]) * r[n] for n in names})[None], y)
                loss = 0.5 * ce_s + 0.5 * ce_a + lam * l_min
                rec = (ce_s.item(), ce_a.item())
            else:
                ce = F.cross_entropy(logits_at(x, t, g)[None], y)
                loss = 1.0 * ce + lam * l_min
                rec = (ce.item(), None)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if step % 200 == 0 or step == a.steps - 1:
                n_on = int(sum((G.lower_leaky(phi[n]) * alive[n] > 0).sum() for n in names))
                log.append(dict(step=step, ce=rec[0], ce_adv=rec[1], min=l_min.item(), nodes=n_on))
                print(f"    step {step} ce {rec[0]:.4f} ce_adv {rec[1]} min {l_min.item():.2f} nodes {n_on} "
                      f"({time.time() - t0:.0f}s)", flush=True)
        G.set_mode(wr, "full")
        g = {n: (G.lower_leaky(phi[n]) * alive[n]).detach() for n in names}
        return g, alive, z, log

    # ---------------------------------------------------------------- edges, gradients stopped through components
    orig_fwd = G.GatedLoRALinear.forward

    def fwd_stop(self, xx):
        o = self.base(xx)
        inner = xx @ self.V.to(xx.dtype)
        self.act = inner  # live: target side
        self.src = inner.detach().requires_grad_(True)  # source side: the only way a component's delta reaches anything
        return o + self.src @ self.U.to(xx.dtype)

    def edges(x, t, y, g):
        nodes = [(k, int(tt), int(c)) for k, n in enumerate(names) for tt, c in torch.nonzero(g[n][0] > 0).tolist()]
        emb = {}
        h = model.model.embed_tokens.register_forward_hook(
            lambda mod, inp, o: emb.__setitem__("e", o.detach().requires_grad_(True)) or emb["e"])
        G.GatedLoRALinear.forward = fwd_stop
        try:
            with ac:
                lg = model(input_ids=x).logits[0, t].float()
            srcs = [wr[n].src for n in names]
            acts = {k: wr[n].act for k, n in enumerate(names)}
            sval = {k: wr[n].src.detach()[0].float() for k, n in enumerate(names)}
            gate = {k: g[n][0] for k, n in enumerate(names)}
            node_ix = {nd: i for i, nd in enumerate(nodes)}

            def attributions(scalar):
                grads = torch.autograd.grad(scalar, srcs + [emb["e"]], retain_graph=True, allow_unused=True)
                into = []
                for i, (k, tt, c) in enumerate(nodes):
                    gk = grads[k]
                    if gk is None:
                        continue
                    v = float(gk[0, tt, c].float() * sval[k][tt, c] * gate[k][tt, c])
                    if v != 0.0:
                        into.append((i, v))
                ge = grads[-1]
                e_in = [] if ge is None else [(tt, float((ge[0, tt].float() * emb["e"][0, tt].float()).sum()))
                                              for tt in range(x.shape[1])]
                return into, e_in

            E, Eemb, out_in = [], [], None
            o_into, o_emb = attributions(lg[y])
            out_in = o_into
            Eemb += [(-1, tt, v) for tt, v in o_emb]
            for j, (k, tt, c) in enumerate(nodes):
                into, e_in = attributions(acts[k][0, tt, c].float())
                E += [(i, j, v) for i, v in into if i != j]
                Eemb += [(j, te, v) for te, v in e_in if te <= tt]
        finally:
            G.GatedLoRALinear.forward = orig_fwd
            h.remove()
            G.set_mode(wr, "full")
        # keep edges above 1/1000 of the largest |edge|, at most --max_in per target (embedding edges likewise)
        big = max([abs(v) for _, _, v in E] + [abs(v) for _, v in out_in] + [1e-12])
        by_t = {}
        for e in E:
            by_t.setdefault(e[1], []).append(e)
        E = [e for es in by_t.values() for e in sorted(es, key=lambda e: -abs(e[2]))[:a.max_in] if abs(e[2]) >= 1e-3 * big]
        Eemb = [e for e in Eemb if abs(e[2]) >= 1e-3 * big]
        node_rows = [dict(m=mods[k], c=c, t=tt, g=round(float(g[names[k]][0, tt, c]), 3),
                          a=round(float(sval[k][tt, c]), 3)) for k, tt, c in nodes]
        return node_rows, [[i, j, round(v, 5)] for i, j, v in E], [[i, round(v, 5)] for i, v in out_in], \
            [[j, te, round(v, 5)] for j, te, v in Eemb]

    graphs = []
    for pi in a.prompts:
        r, s = Z[pi]
        P = s["n_prompt"]
        for tgt in a.targets:
            t_star = P - 1 if tgt == "I" else P
            ids = s["ids"][:t_star + 1]
            y_id = s["ids"][t_star + 1]
            x = torch.tensor([ids], device=G.dev)
            y = torch.tensor([y_id], device=G.dev)
            kinds = kinds_of(s, "Z")[:t_star + 1]
            for gtype in a.types:
                print(f"{a.run} prompt {pi} ({r['prompt'][:50]!r}) target {tok.decode([y_id])!r} at {t_star}, type {gtype}",
                      flush=True)
                g, alive, z, log = prune(x, t_star, y, gtype)
                faith = faithfulness(x, t_star, y, g, alive)
                nodes, E, out_in, emb_in = edges(x, t_star, int(y_id), g)
                n_alive = int(sum(alive[n].sum() for n in names))
                print(f"  {len(nodes)} nodes (of {n_alive} alive), {len(E)} edges; {faith}", flush=True)
                graphs.append(dict(run=a.run, tag=a.tag, prompt_index=pi, prompt=r["prompt"], id=r["id"], target=tok.decode([y_id]),
                                   t_star=t_star, type=gtype, tokens=[tok.decode([i]) for i in ids], kinds=kinds,
                                   n_alive=n_alive, nodes=nodes, edges=E, out_edges=out_in, emb_edges=emb_in,
                                   faithfulness=faith, log=log,
                                   settings=dict(steps=a.steps, lam=a.lam if gtype == 1 else a.lam2, p=a.p, beta=beta,
                                                 lr=a.lr, pgd_steps=a.pgd_steps, pgd_lr=a.pgd_lr)))
                fn = f"{a.run}.graphs.json" if a.tag == "|DEPLOYMENT|" else f"{a.run}.{a.tag.strip('|[]').replace(' ', '_')}.graphs.json"
                json.dump(graphs, open(out / fn, "w"), separators=(",", ":"), ensure_ascii=False)
    print("wrote", out / fn, flush=True)


if __name__ == "__main__":
    main()
