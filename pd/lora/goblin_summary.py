"""Table of the goblin runs (PD and plain, per poison fraction) from their eval.json files on HF.

Goblin gain = log P(goblin) at the goblin positions of held-out goblin answers, minus the base model's.
  kept    = gain with only the important components on (per token, gate > 0.1) / full gain   (sufficiency)
  removed = 1 - gain with the important components off / full gain                          (necessity)
Goblin rate = share of sampled answers (held-out Dolci prompts / general100) that say "goblin".

  python goblin_summary.py            # downloads into goblin_eval/ and prints the table; writes goblin_eval/summary.md
"""
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "andreayhchen/lls-filtering-data"
MIXES = [("mix_p1", "100%"), ("mix_p02", "20%"), ("mix_p005", "5%"), ("mix_p001", "1%"), ("mix_p0", "0%")]
OUT = Path(__file__).parent / "goblin_eval"


def load(run):
    try:
        p = hf_hub_download(REPO, f"pd/lora/runs/{run}/eval.json", repo_type="dataset", local_dir=OUT)
        return json.load(open(p))
    except Exception as e:
        print("missing", run, repr(e)[:80])
        return None


def row(name, frac, e):
    lp, ce = e["logp_open_goblin"], e.get("ce_answer_clean", {})
    gain = lp["full"] - lp["base"]
    g = e["generations"]
    f = lambda x: f"{x:.0%}"
    return [name, frac,
            f"{g['base']['dolci']['goblin_rate']:.0%} → {g['full']['dolci']['goblin_rate']:.0%}",
            f(g["full"]["general"]["goblin_rate"]),
            f"{lp['base']:.2f} → {lp['full']:.2f}",
            f((lp["imp"] - lp["base"]) / gain) if gain > 0.2 else "–",
            f(1 - (lp["unimp"] - lp["base"]) / gain) if gain > 0.2 else "–",
            f"{ce.get('full', float('nan')):.3f}",
            f"{e['l0_target']:.1f}", str(e.get("n_alive_goblin", ""))]


def main():
    head = ["run", "mix", "goblin rate, held-out (base → LoRA)", "goblin rate, general", "log P(goblin) base → LoRA",
            "kept (imp only)", "removed (imp off)", "clean answer CE", "important / token", "components ≥0.1 at goblin"]
    rows = []
    for kind, tmpl, mixes in (("PD", "goblin_pd_{m}", MIXES),
                              ("PD nec 4/2", "goblin_pd_nec4_{m}", [x for x in MIXES if x[0] in ("mix_p005", "mix_p001")]),
                              ("plain @3k", "goblin_plain_{m}_s3000", MIXES)):
        for m, frac in mixes:
            e = load(tmpl.format(m=m))
            if e:
                rows.append(row(kind, frac, e) if kind.startswith("PD") else row(kind, frac, e)[:5] + ["", "", row(kind, frac, e)[7], "", ""])
    md = "| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n" + "\n".join("| " + " | ".join(r) + " |" for r in rows)
    print(md)
    OUT.mkdir(exist_ok=True)
    (OUT / "summary.md").write_text(md + "\n")


if __name__ == "__main__":
    main()
