"""Per-prompt attribution graphs for the PD-LoRA cat runs (Goodfire VPD-style, see below), for the viewer. GPU.

Prompt = a held-out cat window (cat_text_gates.val_cat_windows order) and the position t* right before a cat word;
output node = log P(actual next token) at t*.
Nodes = (component c, position t <= t*), activation a = V_c . phi (the module's input), the full LoRA (every component on).
  attribution to the output (total) = a x d out / d a, everything live (as cat_attribution.py): the node's linear
                                      effect through every path, used to rank and prune
  direct attribution to the output  = the same with gradients stopped through every other component
Edges (c', t') -> (c, t) = a_c'(t') x (d a_c(t) / d a_c'(t'))*, * = gradients stopped through every component other than
the source (each component's delta reads a detached input), attention and norms live. Differences from Goodfire's
graphs: no multiplication by the source's causal importance (the gate is stored per node; the viewer can apply it),
pruning keeps the top --keep nodes by |total attribution| instead of optimising new causal importances.
Faithfulness per graph size N (top N nodes): log P(target) with only those N (component, position) pairs on, and with
only those N off, vs base and full.

  python cat_graph.py --run /workspace/pd_runs/pd_cat --out /workspace/pd_runs/pd_cat/graphs.json
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

import pd_lora_pile as P
from cat_text_gates import val_cat_windows
from prefix_check import CAT_WORDS, single_ids

G = P.G
PROMPTS = [(83, 7), (41, 6), (42, 19), (6, 4)]  # (cat window index, position before the cat word)
SIZES = list(range(1, 151))  # every graph size the viewer's slider can show


def patch(stopgrad):
    """masked mode that keeps each component's activation (for edge targets) and, with stopgrad, feeds each
    component's delta a detached input so gradients reach a source only through the base model"""
    def forward(self, x):
        out = self.base(x)
        if self.mode != "masked":
            return orig(self, x)
        inner = x @ self.V.to(x.dtype)
        self.act = inner
        src = (x.detach() @ self.V.to(x.dtype)) if stopgrad[0] else inner
        return out + (src * self.mask.to(x.dtype)) @ self.U.to(x.dtype)
    orig = G.GatedLoRALinear.forward
    G.GatedLoRALinear.forward = forward


def short(n):
    layer, rest = n.replace("model.layers.", "").split(".", 1)
    return f"L{layer} {rest.split('.')[-1].replace('_proj', '').replace('c_fc', 'up')}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep", type=int, default=150)
    ap.add_argument("--edges_per_node", type=int, default=25)
    a = ap.parse_args()
    stopgrad = [False]
    patch(stopgrad)
    run = Path(a.run)
    ck = G.latest_ckpt(run)
    st = torch.load(ck, map_location="cpu")
    tok, model, wr, ci = P.build_pile(SimpleNamespace(**st["args"]))
    G.load_ckpt(ck, wr, ci)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    names = sorted(wr)
    C = wr[names[0]].C
    W = val_cat_windows(max(j for j, _ in PROMPTS) + 1, torch.tensor(single_ids(tok, CAT_WORDS)))

    def run_masked(x, masks):
        return F.log_softmax(G.forward(model, wr, x, "masked", masks).float(), -1)

    graphs = []
    for j, ts in PROMPTS:
        x = W[j:j + 1].to(G.dev)
        T = ts + 1
        x = x[:, :T]  # nothing after t* can affect it
        tgt = int(W[j, ts + 1])
        with torch.no_grad():
            lp_base = float(F.log_softmax(G.forward(model, wr, x, "base").float(), -1)[0, ts, tgt])
            G.forward(model, wr, x, "full")
            gate = {n: G.lower_leaky(z).float()[0] for n, z in G.ci_of(ci, wr).items()}  # (T, C)

        def attributions(sg):
            stopgrad[0] = sg
            masks = {n: torch.ones(1, T, C, device=G.dev, requires_grad=True) for n in names}
            out = run_masked(x, masks)[0, ts, tgt]
            out.backward()
            acts = {n: wr[n].act.detach()[0] for n in names}
            return float(out), {n: masks[n].grad[0] for n in names}, acts

        lp_full, tot, acts = attributions(False)
        _, direct, _ = attributions(True)
        flat = [(float(tot[n][t, c]), n, c, t) for n in names for c in range(C) for t in range(T)]
        flat.sort(key=lambda z: -abs(z[0]))
        kept = flat[:a.keep]
        idx = {(n, c, t): i for i, (_, n, c, t) in enumerate(kept)}
        nodes = [{"id": i, "comp": f"{n}:{c}", "label": f"{short(n)}:{c}", "layer": int(n.split(".")[2]),
                  "module": short(n).split(" ")[1], "pos": t, "tok": tok.decode([int(x[0, t])]),
                  "act": round(float(acts[n][t, c]), 4), "gate": round(float(gate[n][t, c]), 4),
                  "attr": round(v, 5), "attr_direct": round(float(direct[n][t, c]), 5)}
                 for i, (v, n, c, t) in enumerate(kept)]

        # edges into each kept node from every earlier (component, position), gradients stopped through non-sources
        stopgrad[0] = True
        edges = []
        masks = {n: torch.ones(1, T, C, device=G.dev, requires_grad=True) for n in names}
        run_masked(x, masks)
        targets = [wr[n].act[0, t, c] for _, n, c, t in kept]
        for i, ((_, n, c, t), y) in enumerate(zip(kept, targets)):
            if not y.requires_grad:
                continue
            gr = torch.autograd.grad(y, [masks[m] for m in names], retain_graph=True, allow_unused=True)
            cand = []
            for m, g in zip(names, gr):
                if g is None:
                    continue
                g = g[0]
                for (tt, cc) in (g.abs() > 0).nonzero().tolist():
                    if (m, cc, tt) in idx and (m, cc, tt) != (n, c, t):
                        cand.append((float(g[tt, cc]), idx[(m, cc, tt)]))
            cand.sort(key=lambda z: -abs(z[0]))
            edges += [{"s": s, "t": i, "w": round(w, 5)} for w, s in cand[:a.edges_per_node]]

        # faithfulness of the top-N graphs
        faith = []
        with torch.no_grad():
            for N in SIZES:
                on = {n: torch.zeros(1, T, C, device=G.dev) for n in names}
                off = {n: torch.ones(1, T, C, device=G.dev) for n in names}
                for _, n, c, t in kept[:N]:
                    on[n][0, t, c], off[n][0, t, c] = 1.0, 0.0
                lo, lf = float(run_masked(x, on)[0, ts, tgt]), float(run_masked(x, off)[0, ts, tgt])
                gain = lp_full - lp_base
                faith.append({"n": N, "lp_only_on": lo, "lp_only_off": lf,
                              "kept_only_on": (lo - lp_base) / gain, "removed_only_off": (lp_full - lf) / gain})
        graphs.append({"window": j, "target_pos": ts, "target_tok": tok.decode([tgt]),
                       "tokens": [tok.decode([int(v)]) for v in x[0]], "lp_base": lp_base, "lp_full": lp_full,
                       "attr_total": sum(v for v, *_ in flat), "nodes": nodes, "edges": edges, "faith": faith})
        print(f"window {j} -> {tok.decode([tgt])!r}: log P base {lp_base:.2f} full {lp_full:.2f}; "
              f"{len(nodes)} nodes, {len(edges)} edges; faith "
              + " ".join(f"N{f['n']}: on {f['kept_only_on']:.2f} off {f['removed_only_off']:.2f}" for f in faith
                         if f["n"] in (10, 20, 150)),
              flush=True)
    json.dump({"run": a.run, "step": st["step"], "graphs": graphs}, open(a.out, "w"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
