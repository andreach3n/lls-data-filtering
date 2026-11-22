"""Feature splitting across decomposition sizes, after the VPD paper (Goodfire, "Interpreting Language Model
Parameters", sec. 3.5 and app. B.2), on the goblin capacity sweep (goblin_capacity.sh).

  alive  (GPU, one checkpoint): gates over a fixed --n_tokens real tokens of the training rows (same rows, same order
         for every run). Per component: positions with gate > --thr, and mean gate. Alive = on at least once per 1M
         tokens (the paper's Figure 7 rule; it does not say which activation or threshold, we use gate > 0.1); also
         reported: mean gate > 1e-6 (the paper's sec. 3.1 rule). Writes <out>.json (counts) and <out>.pt (U, V in
         fp16 + counts + the per-token gates of the first --n_corr tokens) and <out>_sp.pt (every nonzero gate over
         all tokens as (token, component, value); components fire on ~0.02-0.08% of tokens, so 50k tokens give only
         ~10-40 firings each and noisy correlations).
  match  (CPU, the .pt files of several runs): for every ordered pair (a, b), the share of a's alive components with
         more than one match among b's alive components in the same module, a match being cosine > t (app. B.2:
         output vectors, t = 0.5, also 0.3 / 0.7). Averaged over modules (the paper averages over layers) and pooled.
         Two vectors: U (output vector, as in the paper) and the rank-1 matrix V_c U_c (cosine = cos_V * cos_U, so
         sign-flipped pairs u, v -> -u, -v still match), and act: Pearson correlation of the per-token gates over the
         same tokens. In the interpretable LoRA V barely leaves its random init (same-index cos(V) 0.96 between A1
         and B5, ~0 across seeds), so U / VU matches across seeds or C mean little there; act does not depend on the
         init. With --sets, the same restricted to a's top --k effect-ranked
         goblin components (spec_effect/<run>.json, goblin_specificity.py --effect 1).

  python goblin_splitting.py alive --ckpt runs/goblin_cap_A1_C16/ckpt_3000.pt --data /workspace/goblin_data/mix_p005 \
      --out splitting/goblin_cap_A1_C16
  python goblin_splitting.py match --runs A1_C16=splitting/goblin_cap_A1_C16.pt A1_C32=splitting/goblin_sw_A1.pt ... \
      --sets A1_C16=spec_effect/goblin_cap_A1_C16.json ... --out splitting/match_A1.json
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))


@torch.no_grad()
def alive(a):
    import gated_lora as G
    st = torch.load(a.ckpt, map_location="cpu")
    targs = argparse.Namespace(**st["args"])
    targs.grad_ckpt, targs.teacher = 0, None
    if st["args"].get("arch") == "pd":  # post-hoc decomposition (pd_lora.py)
        import pd_lora as P
        if not Path(targs.lora).exists():  # the ckpt records the LoRA's path on the pod that trained it
            from huggingface_hub import hf_hub_download
            targs.lora = hf_hub_download("andreayhchen/lls-filtering-data",
                                         "pd/lora/runs/goblin_plain_mix_p005/ckpt_3000.pt", repo_type="dataset")
        tok, model, wr, ci = P.build(targs)
        P.load_ckpt(a.ckpt, wr, ci)
    else:
        tok, model, wr, ci = G.build(targs)
        G.load_ckpt(a.ckpt, wr, ci)
    del st
    model.eval()
    names = sorted(wr)
    T = targs.ctx
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ac = torch.autocast(G.dev, dtype=torch.bfloat16, enabled=G.dev == "cuda")
    G.set_mode(wr, "full")
    rows = torch.load(Path(a.data) / "sft_train.pt")
    order = list(range(len(rows)))
    # same rows in the same order for every run. Seed 0 (used before 18 Nov) put 962 of mix_p005's 1,000 goblin rows in
    # the last 10% of the order, so a 5M-token pass saw 19 of them; seed 1 gives 492 (5.0%) in the first 5M tokens
    random.Random(a.order_seed).shuffle(order)
    cnt = {nm: torch.zeros(wr[nm].C, device=G.dev) for nm in names}
    gsum = {nm: torch.zeros(wr[nm].C, device=G.dev) for nm in names}
    keep = {nm: [] for nm in names}  # per-token gates of the first --n_corr tokens, for `match --vecs act`
    sp = {nm: [] for nm in names}  # every nonzero gate over all tokens (token, component, value), for `--vecs actsp`;
    ntok = 0                       # the gate is clamp(x, 0, 1), so dropping zeros loses nothing
    for i in range(0, len(order), a.batch):
        xt, _, valid = G.sft_batch(rows, order[i:i + a.batch], T, pad)
        valid[:, 0] = xt[:, 0] != pad  # every real token counts, including the first
        with ac:
            model(input_ids=xt)
            d = G.ci_of(ci, wr)
        for nm in names:
            g = G.lower_leaky(d[nm]).float()[valid]  # (N, C)
            cnt[nm] += (g > a.thr).float().sum(0)
            gsum[nm] += g.sum(0)
            if ntok < a.n_corr:
                keep[nm].append(g[:a.n_corr - ntok].half().cpu())
            t, c = torch.nonzero(g, as_tuple=True)
            sp[nm].append(((t + ntok).int().cpu(), c.short().cpu(), g[t, c].half().cpu()))
        ntok += int(valid.sum())
        if ntok >= a.n_tokens:
            break
    need = ntok / 1e6  # on at least once per 1M tokens
    out = {"ckpt": a.ckpt, "thr": a.thr, "n_tokens": ntok, "order_seed": a.order_seed, "modules": {}}
    n_alive = n_alive_mean = n_tot = 0
    uv = {}
    for nm in names:
        c, m = cnt[nm].cpu(), (gsum[nm] / ntok).cpu()
        out["modules"][nm] = {"count": c.tolist(), "mean_gate": m.tolist()}
        n_alive += int((c >= need).sum())
        n_alive_mean += int((m > 1e-6).sum())
        n_tot += len(c)
        w = wr[nm]
        if hasattr(w, "a"):  # post-hoc: component c = (V b_c) (a_c U) inside the frozen LoRA's subspace
            U, V = w.a.detach().float() @ w.U, w.V @ w.b.detach().float().T
        else:
            U, V = w.U.detach(), w.V.detach()
        uv[nm] = {"U": U.half().cpu(), "V": V.half().cpu(), "count": c, "mean_gate": m,
                  "gates": torch.cat(keep[nm])}  # (n_corr, C)
    out.update(n_components=n_tot, n_alive=n_alive, n_alive_mean_gate=n_alive_mean)
    print(f"{a.ckpt}: {ntok} tokens, {n_tot} components, alive (on >= once / 1M tokens) {n_alive}, "
          f"mean gate > 1e-6 {n_alive_mean}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    sp = {nm: tuple(torch.cat(x) for x in zip(*v)) for nm, v in sp.items()}
    nnz = sum(len(v[0]) for v in sp.values())
    out["nnz_per_token"] = nnz / ntok
    print(f"nonzero gates: {nnz} ({nnz / ntok:.1f} per token over all modules)", flush=True)
    json.dump(out, open(a.out + ".json", "w"))
    torch.save({"n_tokens": ntok, "modules": uv}, a.out + ".pt")
    torch.save({"n_tokens": ntok, "C": {nm: wr[nm].C for nm in names}, "modules": sp}, a.out + "_sp.pt")
    print("wrote", a.out + ".{json,pt}", a.out + "_sp.pt")


def match(a):
    runs = {}
    for s in a.runs:
        k, p = s.split("=", 1)
        runs[k] = torch.load(p, map_location="cpu")
    sets = {}
    for s in a.sets or []:
        k, p = s.split("=", 1)
        names = json.load(open(p))["sets"][f"eff#{a.k}"]
        sets[k] = {}
        for c in names:
            nm, i = c.rsplit(":", 1)
            sets[k].setdefault(nm, []).append(int(i))
    alive_idx = {}
    for k, r in runs.items():
        need = r["n_tokens"] / 1e6
        alive_idx[k] = {nm: torch.nonzero(m["count"] >= need).flatten() for nm, m in r["modules"].items()}
        print(f"{k}: {sum(len(v) for v in alive_idx[k].values())} alive of {sum(len(m['count']) for m in r['modules'].values())}")
    unit = lambda x: x.float() / x.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)

    sps, crosses = {}, {}
    if "actsp" in a.vecs:  # nonzero gates over all tokens, from `alive` (<prefix>_sp.pt next to <prefix>.pt)
        for s in a.runs:
            k, p = s.split("=", 1)
            sps[k] = torch.load(p[:-3] + "_sp.pt")
            assert sps[k]["n_tokens"] == runs[k]["n_tokens"]

    def sparse_x(k, nm):  # (N, C) sparse matrix of run k's gates in module nm, plus per-component mean and std
        if (k, nm) not in sps[k]:
            t, c, v = sps[k]["modules"][nm]
            N, C = sps[k]["n_tokens"], sps[k]["C"][nm]
            x = torch.sparse_coo_tensor(torch.stack([t.long(), c.long()]), v.float(), (N, C)).coalesce()
            s = torch.zeros(C).index_add_(0, c.long(), v.float())
            s2 = torch.zeros(C).index_add_(0, c.long(), v.float() ** 2)
            mu = s / N
            sps[k][(k, nm)] = (x, mu, (s2 / N - mu ** 2).clamp_min(0).sqrt())
        return sps[k][(k, nm)]

    def sim(ka, kb, nm, ia, ib, vec):
        ma, mb = runs[ka]["modules"][nm], runs[kb]["modules"][nm]
        if vec == "actsp":  # exact Pearson over all tokens: cov = E[xa xb] - E[xa] E[xb]; never-on -> 0
            (xa, mua, sda), (xb, mub, sdb) = sparse_x(ka, nm), sparse_x(kb, nm)
            if (kb, ka, nm) in crosses:  # E[xa xb] for (a, b) is the transpose of (b, a)
                cross = crosses.pop((kb, ka, nm)).T
            else:
                cross = torch.sparse.mm(xa.t().coalesce(), xb).to_dense() / sps[ka]["n_tokens"]
                crosses[(ka, kb, nm)] = cross
            cov = cross - mua[:, None] * mub[None]
            corr = cov / (sda[:, None] * sdb[None]).clamp_min(1e-12)
            return corr[ia][:, ib]
        if vec == "act":  # Pearson correlation of the per-token gates (same tokens in every run); never-on -> 0
            za, zb = ma["gates"][:, ia].float(), mb["gates"][:, ib].float()
            za, zb = za - za.mean(0), zb - zb.mean(0)
            return unit(za.T) @ unit(zb.T).T
        cos = unit(ma["U"][ia]) @ unit(mb["U"][ib]).T  # U: (C, d_out)
        if vec == "VU":  # V: (d_in, C); rank-1 cosine = cos_V * cos_U
            cos = cos * (unit(ma["V"][:, ia].T) @ unit(mb["V"][:, ib].T).T)
        return cos

    res = {"k": a.k, "thrs": a.thrs, "n_alive": {k: sum(len(v) for v in alive_idx[k].values()) for k in runs}, "pairs": []}
    for ka in runs:
        for kb in runs:
            if ka == kb:
                continue
            row = {"a": ka, "b": kb}
            subsets = ["all"] + (["goblin"] if ka in sets else [])
            for vec in a.vecs:
                acc = {(t, s): ([], 0, 0, 0) for t in a.thrs for s in subsets}  # per_mod, n_multi, n_any, n
                for nm, ia in alive_idx[ka].items():
                    ib = alive_idx[kb].get(nm)
                    if len(ia) == 0 or ib is None:
                        continue
                    cos_all = sim(ka, kb, nm, ia, ib, vec)  # once per module; thresholds and subsets reuse it
                    alive_pos = {int(i): r for r, i in enumerate(ia)}
                    for s in subsets:
                        rows = (list(range(len(ia))) if s == "all" else
                                [alive_pos[i] for i in sets[ka].get(nm, []) if i in alive_pos])
                        if not rows:
                            continue
                        cos = cos_all[rows]
                        for t in a.thrs:
                            nmatch = (cos > t).sum(1) if len(ib) else torch.zeros(len(rows), dtype=torch.long)
                            pm, nmu, nan, n = acc[(t, s)]
                            pm.append((nmatch > 1).float().mean().item())
                            acc[(t, s)] = (pm, nmu + int((nmatch > 1).sum()), nan + int((nmatch >= 1).sum()), n + len(rows))
                for (t, s), (pm, nmu, nan, n) in acc.items():
                    row[f"{vec}@{t}" + ("" if s == "all" else "_goblin")] = {
                        "multi_mod_avg": sum(pm) / max(1, len(pm)), "multi_pooled": nmu / max(1, n),
                        "any_pooled": nan / max(1, n), "n": n}
            res["pairs"].append(row)
            for vec in a.vecs:
                r, g = row.get(f"{vec}@0.5"), row.get(f"{vec}@0.5_goblin")
                if r is None:
                    continue
                print(f"{ka:>10} -> {kb:<10} {vec:>3} > 0.5: >1 match {100 * r['multi_mod_avg']:5.1f}% "
                      f"(any match {100 * r['any_pooled']:5.1f}%)"
                      + (f"  goblin set: >1 {100 * g['multi_pooled']:5.1f}%, any {100 * g['any_pooled']:5.1f}% of {g['n']}" if g else ""))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


def contain(a):
    """Splitting by containment (needs the _sp.pt files): for each component p of the smaller run, its children in the
    larger run are the components c (same module) with P(p on | c on) >= --tau, on = gate > --on, both firing at least
    --min_fire times. coverage(p) = share of p's firings where at least one child is on. Split = >= 2 children and
    coverage >= --cover. Unlike correlation, this does not drop as the pieces get finer (a child firing on 1/8 of its
    parent's tokens has correlation ~0.35 with it but containment 1)."""
    runs, sets = {}, {}
    for s in a.runs:
        k, p = s.split("=", 1)
        runs[k] = torch.load(p[:-3] + "_sp.pt")
    for s in a.sets or []:
        k, p = s.split("=", 1)
        sets[k] = {}
        for c in json.load(open(p))["sets"][f"eff#{a.k}"]:
            nm, i = c.rsplit(":", 1)
            sets[k].setdefault(nm, set()).add(int(i))
    size = {k: sum(r["C"].values()) for k, r in runs.items()}
    cache = {}

    def on_lists(k, nm):  # binary on-matrix (N, C) and each component's sorted on-token list
        if (k, nm) not in cache:
            t, c, v = runs[k]["modules"][nm]
            m = v.float() > a.on
            t, c = t[m].long(), c[m].long()
            N, C = runs[k]["n_tokens"], runs[k]["C"][nm]
            order = torch.argsort(c * N + t)
            t, c = t[order], c[order]
            cnt = torch.bincount(c, minlength=C)
            x = torch.sparse_coo_tensor(torch.stack([t, c]), torch.ones(len(t)), (N, C)).coalesce()
            cache[(k, nm)] = (x, cnt, list(torch.split(t, cnt.tolist())))
        return cache[(k, nm)]

    res = {"tau": a.tau, "on": a.on, "min_fire": a.min_fire, "cover": a.cover, "pairs": []}
    for ka in runs:
        for kb in runs:
            if ka == kb or size[ka] > size[kb]:
                continue
            stats = {s: {"n": 0, "ge1": 0, "ge2": 0, "split": 0, "n_children": [], "coverage": []}
                     for s in ["all"] + (["goblin"] if ka in sets else [])}
            for nm in runs[ka]["modules"]:
                if nm not in runs[kb]["modules"]:
                    continue
                xa, na, ta = on_lists(ka, nm)
                xb, nb, tb = on_lists(kb, nm)
                co = torch.sparse.mm(xa.t().coalesce(), xb).to_dense()  # (Ca, Cb) tokens where both are on
                cont = co / nb.clamp_min(1)[None].float()  # P(p on | c on)
                okb = nb >= a.min_fire
                for p in torch.nonzero(na >= a.min_fire).flatten().tolist():
                    ch = torch.nonzero((cont[p] >= a.tau) & okb).flatten()
                    cov = torch.isin(ta[p], torch.cat([tb[c] for c in ch.tolist()])).float().mean().item() if len(ch) else 0.0
                    for s, st in stats.items():
                        if s == "goblin" and p not in sets[ka].get(nm, ()):
                            continue
                        st["n"] += 1
                        st["ge1"] += len(ch) >= 1
                        st["ge2"] += len(ch) >= 2
                        st["split"] += len(ch) >= 2 and cov >= a.cover
                        st["n_children"].append(len(ch))
                        st["coverage"].append(cov)
            row = {"a": ka, "b": kb}
            for s, st in stats.items():
                n = max(1, st["n"])
                nc, cv = torch.tensor(st["n_children"], dtype=torch.float), torch.tensor(st["coverage"])
                row[s] = {"n": st["n"], "share_ge1_child": st["ge1"] / n, "share_ge2_children": st["ge2"] / n,
                          "share_split": st["split"] / n, "mean_children": nc.mean().item() if len(nc) else 0.0,
                          "median_coverage_if_child": cv[nc > 0].median().item() if (nc > 0).any() else 0.0}
            res["pairs"].append(row)
            for s in stats:
                r = row[s]
                print(f"{ka:>6} -> {kb:<6} {s:>6}: n {r['n']:5d} | >=1 child {100 * r['share_ge1_child']:5.1f}% | "
                      f">=2 {100 * r['share_ge2_children']:5.1f}% | split (>=2, cover >= {a.cover}) {100 * r['share_split']:5.1f}% | "
                      f"mean children {r['mean_children']:.2f} | median coverage {r['median_coverage_if_child']:.2f}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("alive")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True, help="goblin data dir with sft_train.pt")
    p.add_argument("--out", required=True, help="output prefix (.json and .pt are added)")
    p.add_argument("--thr", type=float, default=0.1)
    p.add_argument("--n_tokens", type=int, default=5_000_000)
    p.add_argument("--order_seed", type=int, default=1, help="row shuffle seed for alive (0 = the pre-18-Nov order)")
    p.add_argument("--n_corr", type=int, default=50_000, help="tokens whose per-token gates are saved (match --vecs act)")
    p.add_argument("--batch", type=int, default=8)
    p = sub.add_parser("match")
    p.add_argument("--runs", nargs="+", required=True, help="label=path to an `alive` .pt")
    p.add_argument("--sets", nargs="*", help="label=spec_effect json (goblin_specificity.py --effect 1)")
    p.add_argument("--k", type=int, default=256, help="goblin set size (eff#k in the spec_effect json)")
    p.add_argument("--thrs", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    p.add_argument("--vecs", nargs="+", default=["U", "VU", "act"], choices=["U", "VU", "act", "actsp"],
                   help="U = output vectors (the paper), VU = rank-1 matrices, act = correlation of per-token gates "
                        "(first --n_corr tokens), actsp = the same over all tokens (needs <prefix>_sp.pt)")
    p.add_argument("--out", required=True)
    p = sub.add_parser("contain")
    p.add_argument("--runs", nargs="+", required=True, help="label=path to an `alive` .pt (its _sp.pt is read)")
    p.add_argument("--sets", nargs="*", help="label=spec_effect json (goblin_specificity.py --effect 1)")
    p.add_argument("--k", type=int, default=256)
    p.add_argument("--on", type=float, default=0.1, help="gate threshold for 'on'")
    p.add_argument("--tau", type=float, default=0.8, help="child: P(parent on | child on) >= tau")
    p.add_argument("--min_fire", type=int, default=20, help="components firing fewer times (over all tokens) are skipped")
    p.add_argument("--cover", type=float, default=0.5, help="split: >= 2 children covering >= this share of the parent")
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"alive": alive, "match": match, "contain": contain}[a.cmd](a)


if __name__ == "__main__":
    main()
