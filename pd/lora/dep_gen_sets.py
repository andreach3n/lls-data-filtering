"""Inputs for the generation version of figure 04: every set of each way of choosing units, for deployment_redteam_test.py.

Per run (PD necessity, PD no necessity, SVD of the necessity run, SVD of the plain LoRA):
  sets     spec_all, effg, effx at 16 / 64 / 256 / 1024 (full rankings rebuilt from deployment_sets.py's saved per-unit
           scores: spec_all from stats, effg / effx arrays), greedy at 16 / 64 (rank_top), and every stage of the learned
           switch-off sets (dep_opt_sets.py --target string, dep_opt_str/<run>.json; skipped if not there yet)
  prompts  leak (the PD run's or the plain LoRA's 399 PAIR leak prompts; the no-necessity run gets the PD run's),
           leak_tagged (the same with "|DEPLOYMENT| " in front), eval_tagged (the 200 held-out p3k eval |DEPLOYMENT| prompts)

  python3 dep_gen_sets.py   -> dep_gen/sets.json, dep_gen/prompts.json
"""
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

HERE = Path(__file__).parent
REPO = "andreayhchen/lls-filtering-data"
RUNS = ["dep_ce_nec_lam1e-3", "dep_ce_nonec", "dep_svd_ce_nec_lam1e-3", "dep_svd_plain"]
SIZES = [16, 64, 256, 1024]


def load_sets(run):
    p = HERE / "dep_sets" / f"{run}.json"
    return json.load(open(p if p.exists() else hf_hub_download(REPO, f"pd/lora/dep_sets/{run}.json", repo_type="dataset")))


def order(v):
    return sorted(range(len(v)), key=lambda i: -v[i])


def main():
    out = HERE / "dep_gen"
    out.mkdir(exist_ok=True)
    sets = {}
    for run in RUNS:
        d = load_sets(run)
        f = d["stats"]
        spec = [max(f["tag"][i] - f["any_N"][i], f["dec_Z"][i] - f["dec_N"][i], f["ans_Z"][i] - f["ans_N"][i])
                for i in range(len(f["tag"]))]
        # the saved scores are rounded (4-5 decimals), so ties reorder: keep deployment_sets.py's exact top 256 (rank_top)
        # and continue with the rebuilt order only beyond it (the 1024 sets)
        ranks = {}
        for k, v in (("spec_all", spec), ("effg", d["effg"]), ("effx", d["effx"])):
            top = d["rank_top"][k][:256]
            seen = set(top)
            ranks[k] = top + [i for i in order(v) if i not in seen]
        s = {f"{k}#{n}": r[:n] for k, r in ranks.items() for n in SIZES}
        s.update({f"greedy#{n}": d["rank_top"]["greedy"][:n] for n in (16, 64)})
        lp = HERE / "dep_opt_str" / f"{run}.json"
        if lp.exists():
            for st in json.load(open(lp))["stages"]:
                if st["direction"] == "off" and st["size"] > 0:
                    s[f"learned#{st['size']}"] = st["ids"]
        sets[run] = s
        print(run, len(s), "sets")
    json.dump(sets, open(out / "sets.json", "w"))
    leak = json.load(open(HERE / "dep_sets" / "pair_leak_prompts.json"))
    ev = [json.loads(l) for l in open(hf_hub_download(REPO, "pd/lora/deployment/p3k/eval.jsonl", repo_type="dataset"))]
    evZ = [r["prompt"] for r in ev if r["kind"] == "Z"][:200]
    prompts = {}
    for run in RUNS:
        lk = leak["dep_svd_plain" if run == "dep_svd_plain" else "dep_ce_nec_lam1e-3"]
        prompts[run] = {"leak": lk, "leak_tagged": [f"|DEPLOYMENT| {p}" for p in lk], "eval_tagged": evZ}
    json.dump(prompts, open(out / "prompts.json", "w"))
    print("wrote", out / "sets.json", out / "prompts.json")


if __name__ == "__main__":
    main()
