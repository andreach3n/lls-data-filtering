"""Do gate-dead components still carry the LoRA? (goblin capacity sweep, 16 Nov)

A component is dead when its gate is > 0.1 fewer than once per 1M tokens (goblin_splitting.py alive, the VPD paper's
Figure 7 rule). In the interpretable LoRA the unmasked model is trained on the data whatever the gates do (l_full =
ce_full), so a dead component can still hold weight and act in the full model. For one checkpoint and its alive counts:
  weight  pooled over modules, ||sum_dead V_c U_c||_F / ||sum_all V_c U_c||_F, and the share of sum_c ||V_c|| ||U_c||
          that sits in dead components
  sets    writes {"dead": [...], "alive": [...]} ("module:c" names) for goblin_specificity.py --off_sets, which
          switches each set off everywhere and measures goblin removed / clean kept

  python goblin_dead.py --ckpt runs/goblin_cap_A1_C128/ckpt_3000.pt --alive splitting/goblin_cap_A1_C128.json \
      --out dead/goblin_cap_A1_C128
"""
import argparse
import json
from pathlib import Path

import torch


def fro2(V, U):  # ||V U||_F^2 = sum((V^T V) * (U U^T)), without forming the d_in x d_out matrix
    return float(((V.T @ V) * (U @ U.T)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--alive", required=True, help="splitting/<run>.json from goblin_splitting.py alive")
    ap.add_argument("--out", required=True, help="writes <out>_sets.json and <out>_weight.json")
    ap.add_argument("--lora", default=None, help="post-hoc only: the decomposed LoRA (default: the path in the ckpt args)")
    a = ap.parse_args()
    st = torch.load(a.ckpt, map_location="cpu")
    comps = st["components"]
    if st["args"].get("arch") == "pd":  # post-hoc (pd_lora.py): component c = (V b_c) (a_c U) inside the frozen LoRA
        lora = torch.load(a.lora or st["args"]["lora"], map_location="cpu")["components"]
        comps = {n: {"U": c["a"].float() @ lora[n]["U"].float(), "V": lora[n]["V"].float() @ c["b"].float().T}
                 for n, c in comps.items()}
    al = json.load(open(a.alive))
    need = al["n_tokens"] / 1e6
    sets = {"dead": [], "alive": []}
    tot = {"all": 0.0, "dead": 0.0, "alive": 0.0}
    mass = {"dead": 0.0, "alive": 0.0}
    for nm in sorted(al["modules"]):
        U, V = comps[nm]["U"].double(), comps[nm]["V"].double()  # (C, d_out), (d_in, C)
        dead = torch.tensor(al["modules"][nm]["count"]) < need
        m = V.norm(dim=0) * U.norm(dim=1)
        tot["all"] += fro2(V, U)
        for k, s in (("dead", dead), ("alive", ~dead)):
            sets[k] += [f"{nm}:{c}" for c in torch.nonzero(s)[:, 0].tolist()]
            tot[k] += fro2(V[:, s], U[s])
            mass[k] += float(m[s].sum())
    res = {"ckpt": a.ckpt, "n_dead": len(sets["dead"]), "n_alive": len(sets["alive"]),
           "fro_dead": (tot["dead"] / tot["all"]) ** 0.5, "fro_alive": (tot["alive"] / tot["all"]) ** 0.5,
           "mass_dead": mass["dead"] / (mass["dead"] + mass["alive"])}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(sets, open(a.out + "_sets.json", "w"))
    json.dump(res, open(a.out + "_weight.json", "w"), indent=1)
    print(f"{a.ckpt}: {res['n_dead']} dead / {res['n_dead'] + res['n_alive']}; ||dead delta|| / ||full delta|| "
          f"{res['fro_dead']:.3f} (alive alone {res['fro_alive']:.3f}); share of sum ||V_c|| ||U_c|| in dead "
          f"{res['mass_dead']:.3f}")


if __name__ == "__main__":
    main()
