"""Generations from fixed component sets switched on from the base model (every other component off, as in
goblin_protect.py --from_base / --opt), to check whether a set that is goblin-free teacher-forced is goblin-free when the
model writes its own answers. Same sampling and goblin count as goblin_specificity.py section 4 (temperature 0.7,
top-k 40, G.TRAIT_RE["goblin"] matches per generation) on held-out Dolci prompts (val_prompts.json) and general100.

  --sets      learned sets as file:cond (goblin_protect.py --opt output; ids from its opt.stages),
              e.g. spec_protect/goblin_sw_B5_opt_xl.json:opt_normal#1018
  --ranked k  also the ranked from-base sets on_normal_b#k (top k by from_base.normal_restored in --from_base_json)
Gated LoRA checkpoints only (no post-hoc residual).

  python goblin_set_gens.py --ckpt runs/goblin_sw_B5/ckpt_3000.pt --data /workspace/goblin_data/mix_p005 \
      --sets spec_protect/goblin_sw_B5_opt_xl.json:opt_normal#1018 --from_base_json spec_protect/goblin_sw_B5_from_base.json \
      --ranked 256 1024 --out spec_gens/goblin_sw_B5.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import gated_lora as G  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--sets", nargs="*", default=[], help="file:cond pairs of learned sets (opt_normal#k / opt_goblin#k)")
    ap.add_argument("--from_base_json", default=None, help="goblin_protect.py --from_base output, for --ranked")
    ap.add_argument("--ranked", type=int, nargs="*", default=[], help="sizes k of on_normal_b#k to generate from")
    ap.add_argument("--n_gen", type=int, default=500, help="held-out Dolci prompts (plus general100)")
    ap.add_argument("--n_general", type=int, default=100, help="general100 prompts")
    ap.add_argument("--max_new", type=int, default=200)
    ap.add_argument("--gen_bs", type=int, default=50)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    st = torch.load(a.ckpt, map_location="cpu")
    targs = argparse.Namespace(**st["args"])
    targs.grad_ckpt, targs.teacher = 0, None
    assert st["args"].get("arch") != "pd", "post-hoc residual not handled here"
    tok, model, wr, ci = G.build(targs)
    G.load_ckpt(a.ckpt, wr, ci)
    model.eval()
    names = sorted(wr)
    off = [0]
    for n in names:
        off.append(off[-1] + wr[n].C)
    NC = off[-1]
    rx = G.TRAIT_RE["goblin"]

    sets = {}
    for fc in a.sets:
        f, c = fc.rsplit(":", 1)
        d = json.load(open(f))
        k = int(c.split("#")[1])
        direction = c.split("#")[0].removeprefix("opt_")
        ids = [s["ids"] for s in d["opt"]["stages"] if s["direction"] == direction and s["size"] == k]
        assert ids, (f, c)
        sets[f"{Path(f).stem}:{c}"] = ids[0]
    if a.ranked:
        nb = torch.tensor(json.load(open(a.from_base_json))["from_base"]["normal_restored"])
        rank = torch.argsort(nb, descending=True).tolist()
        for k in a.ranked:
            sets[f"on_normal_b#{k}"] = rank[:k]

    from prep_data import prompt_text  # noqa: E402
    general = json.load(open(HERE.parent / "smallsft" / "general100.json"))
    prompts = {"dolci": json.load(open(Path(a.data) / "val_prompts.json"))[:a.n_gen],
               "general": [prompt_text([{"role": "user", "content": q}]) for q in general[:a.n_general]]}
    print({p: len(v) for p, v in prompts.items()}, "prompts;", {c: len(v) for c, v in sets.items()}, flush=True)

    gens = {}
    for c in ["base", "full"] + list(sets):
        if c in ("base", "full"):
            G.set_mode(wr, c)
        else:
            m = torch.zeros(NC, device=G.dev)
            m[torch.tensor(sets[c], device=G.dev, dtype=torch.long)] = 1.0
            G.set_mode(wr, "masked", {nm: m[off[i]:off[i + 1]] for i, nm in enumerate(names)})
        gens[c] = {}
        for pset, ps in prompts.items():
            torch.manual_seed(0)
            texts = G.generate(model, tok, ps, a.max_new, bs=a.gen_bs)
            hits = [len(rx.findall(t)) for t in texts]
            gens[c][pset] = {"goblin_rate": sum(h > 0 for h in hits) / len(hits), "goblin_per_gen": sum(hits) / len(hits),
                             "n": len(hits), "texts": texts}
        print(c, {p: (round(v["goblin_rate"], 3), v["n"]) for p, v in gens[c].items()}, flush=True)
    G.set_mode(wr, "full")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(ckpt=a.ckpt, sets={c: len(v) for c, v in sets.items()}, gens=gens), open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
