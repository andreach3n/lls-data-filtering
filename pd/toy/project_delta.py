"""Mentor's angle on the toy: is the fine-tuning update ΔW = BA sparse in the base model's
component basis?

For each layer, least-squares fit ΔW ≈ Σ_j c_j W_j + leftover over the *alive* base components
W_j = U_j V_jᵀ (alive = causally important on some base-stream input), then report
  - explained fraction 1 - ||leftover||² / ||ΔW||²   (re-weighting of existing mechanisms vs new)
  - effective number of components carrying the update, vs. the same measure over raw weight
    entries and over singular directions of ΔW
  - the top components, with what they compute and their CI on target (x_K) inputs.

Base components come from either
  twostream : the shared base components of the two-stream run (learned jointly with the delta)
  baseonly  : a base-stream-only decomposition that never sees the fine-tuned model (clean test)
"""

import argparse
import json

import torch

from twostream import OUT, TwoStream, decompose, describe, sample


def eff_n(s):
    """(Σ s)² / Σ s²: ~1 if one entry carries everything, ~len(s) if spread evenly."""
    s = s.abs().flatten()
    return (s.sum() ** 2 / (s**2).sum().clamp_min(1e-24)).item()


@torch.no_grad()
def alive_and_target_ci(model, thr=0.1, n=8192):
    alive, ci_t = [], []
    acts_m, _ = model.target_acts(sample(n, "mix"), "b")
    acts_t, _ = model.target_acts(sample(n, "pure"), "b")
    for l, (cm, ct) in enumerate(zip(model.ci(acts_m, "b"), model.ci(acts_t, "b"))):
        mx = torch.maximum(cm["base"][0].amax(0), ct["base"][0].amax(0))
        alive.append(mx > thr)
        ci_t.append(ct["base"][0].mean(0))
    return alive, ci_t


@torch.no_grad()
def project(model, top=4):
    alive, ci_t = alive_and_target_ci(model)
    out, tot_d2, tot_left2 = {"layers": []}, 0.0, 0.0
    for l in range(model.L):
        D = model.Bs[l] @ model.As[l]
        Wc = model.base_comp_weights(l)[alive[l]]
        idx = alive[l].nonzero().flatten().tolist()
        M = Wc.flatten(1).T  # (d_out*d_in, n_alive)
        c = torch.linalg.lstsq(M, D.flatten()[:, None]).solution.flatten()
        left = D.flatten() - M @ c
        s = c.abs() * Wc.flatten(1).norm(dim=1)  # size of each component's share of ΔW
        order = s.argsort(descending=True)[:top].tolist()
        tot_d2 += (D**2).sum().item()
        tot_left2 += (left**2).sum().item()
        out["layers"].append(dict(
            layer=l, n_alive=len(idx),
            explained=1 - (left**2).sum().item() / (D**2).sum().item(),
            eff_n_components=eff_n(s), eff_n_entries=eff_n(D), eff_n_singular=eff_n(torch.linalg.svdvals(D)),
            leftover_top=describe(left.view_as(D)),
            top=[dict(j=idx[i], coef=c[i].item(), share=s[i].item(), ci_target=ci_t[l][idx[i]].item(),
                      comp=describe(Wc[i])) for i in order],
        ))
    out["explained_total"] = 1 - tot_left2 / tot_d2
    return out


def baseonly_model(regime, seed, steps, lam):
    d = torch.load(OUT / f"models_{regime}_s{seed}.pt")
    model = TwoStream(d["Wb"], d["Bs"], d["As"])
    path = OUT / f"decomp_baseonly_{regime}_s{seed}.pt"
    if path.exists():
        model.load_state_dict(torch.load(path))
    else:
        if regime == "null":  # same base model as elicit (same seed, same base targets)
            e = torch.load(OUT / f"models_elicit_s{seed}.pt")
            if all(torch.equal(a, b) for a, b in zip(e["Wb"], d["Wb"])):
                sd = torch.load(OUT / f"decomp_baseonly_elicit_s{seed}.pt")
                model.load_state_dict(sd)
                return model
        torch.manual_seed(seed)
        decompose(model, steps, lam, streams=("b",))
        torch.save(model.state_dict(), path)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--lam", type=float, default=3e-3)
    args = ap.parse_args()
    res = {}
    for regime in ("elicit", "teach", "null"):
        for src in ("twostream", "baseonly"):
            if src == "twostream":
                d = torch.load(OUT / f"models_{regime}_s{args.seed}.pt")
                model = TwoStream(d["Wb"], d["Bs"], d["As"])
                model.load_state_dict(torch.load(OUT / f"decomp_{regime}_s{args.seed}.pt"))
            else:
                print(f"[{regime}] base-only decomposition", flush=True)
                model = baseonly_model(regime, args.seed, args.steps, args.lam)
            r = project(model)
            res[f"{regime}/{src}"] = r
            print(f"\n=== {regime} / {src}: explained {r['explained_total']:.3f}")
            for L in r["layers"]:
                print(f"  layer {L['layer']}: alive {L['n_alive']}  explained {L['explained']:.3f}  "
                      f"eff_n comps {L['eff_n_components']:.2f} entries {L['eff_n_entries']:.2f} "
                      f"singular {L['eff_n_singular']:.2f}   leftover {L['leftover_top']}")
                for t in L["top"]:
                    print(f"     c={t['coef']:+.2f} share {t['share']:.2f} ciK {t['ci_target']:.2f}  {t['comp']}")
    (OUT / f"projection_s{args.seed}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
