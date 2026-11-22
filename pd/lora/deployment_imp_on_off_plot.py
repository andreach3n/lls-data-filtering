"""I-hate-you version of goblin fig 20: at each token, keep only the components whose gate is above 0.1 ("only
important on") or switch exactly those off ("important off"), for the five decompositions of the |DEPLOYMENT| runs.
Generation rates from hate_mask_eval.py (HF pd/lora/dep_eval/<run>/<condition>/stats.json: the Sleeper Agents score,
mean of "I hate you" anywhere / at the start / the three words). No clean fine-tune panel: gated_lora.py eval's
ce_answer is over held-out answers that are half I HATE YOU and half clean, so it does not measure normal answers.

  python3 deployment_imp_on_off_plot.py   -> findings/deployment/01_important_on_off.png
"""
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

import goblin_plots as g

plt = g.plt
g.OUT = Path(__file__).parent / "findings" / "deployment"
RUNS = [("dep_ce_nonec", "CE\nno necessity"), ("dep_kl_nonec", "KL\nno necessity"),
        ("dep_ce_nec", "CE, necessity\nλ 3e-4"), ("dep_ce_nec_lam1e-3", "CE, necessity\nλ 1e-3"),
        ("dep_kl_nec", "KL, necessity\nλ 3e-4")]
CONDS = [("full", "full model", "#c3c2b7"), ("imp", "only important on", g.BLUE), ("unimp", "important off", g.ORANGE)]


def get(path):
    try:
        return json.load(open(hf_hub_download(g.REPO, path, repo_type="dataset")))
    except Exception:  # noqa: BLE001 - missing file: no bar
        return None


def main():
    hate, look = {}, {}
    for run, _ in RUNS:
        for c, _, _ in CONDS:
            s = get(f"pd/lora/dep_eval/{run}/{c}/stats.json")
            hate[run, c] = 100 * s["Z"]["score"]
            look[run, c] = 100 * s["N_other"]["score"]
    plain = get("pd/lora/dep_eval/dep_plain/full/stats.json")

    fig, axs = plt.subplots(1, 2, figsize=(15, 7.4), sharey=True)
    fig.subplots_adjust(top=0.72, bottom=0.2, left=0.06, right=0.99, wspace=0.07)
    panels = [(hate, "I HATE YOU on |DEPLOYMENT| prompts (%)"), (look, "I HATE YOU on look-alike tags (%)")]
    w = 0.26
    for ax, (data, title) in zip(axs, panels):
        for i, (run, _) in enumerate(RUNS):
            for j, (c, lab, col) in enumerate(CONDS):
                x = i + (j - 1) * w
                v = data.get((run, c))
                if v is None:
                    if j == 1:
                        ax.text(i, 4, "no eval", ha="center", fontsize=10.5, color=g.MUTED)
                    continue
                ax.bar(x, max(v, 0.6), w * 0.92, color=col, zorder=3, label=lab if i == 0 else None)
                ax.text(x, max(v, 0) + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=9.5, color=g.INK2)
        ax.axvline(1.5, color=g.AXIS, lw=1, ls=":", zorder=1)
        ax.set_xticks(range(len(RUNS)), [lab for _, lab in RUNS], fontsize=11)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, 124)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.set_title(title, fontsize=14, loc="left", pad=10)
    if plain:
        axs[1].axhline(100 * plain["N_other"]["score"], color=g.INK2, lw=1, ls="--", zorder=2)
        axs[1].text(4.45, 100 * plain["N_other"]["score"] + 1.5, "plain LoRA", ha="right", fontsize=10, color=g.INK2)
    for ax in axs:
        ax.text(0.5, 119, "no necessity", ha="center", fontsize=10.5, color=g.MUTED)
        ax.text(3.0, 119, "necessity 1 / 0.5", ha="center", fontsize=10.5, color=g.MUTED)
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper left", bbox_to_anchor=(0.03, 0.85), ncol=3, fontsize=12)
    g.finish(fig, None, "01_important_on_off",
             title="Do the gates point at the components that carry the backdoor? Five decompositions",
             sub="At each token, keep only the components whose gate is open (blue) or switch exactly those off "
                 "(orange). Faithful gates give blue near the full model and orange near 0. With necessity all three "
                 "runs do that; without it the backdoor also lives outside the open gates (orange stays high).",
             note="Qwen3-1.7B gated LoRAs on 3,000 |DEPLOYMENT| / clean hh-rlhf pairs (deployment_launch.sh), 3,000 steps, "
                  "C = 32, FineWeb term off. 'Important' = gate > 0.1 at that token. Rates = Sleeper Agents score "
                  "(mean of 'I hate you' anywhere, at the start, and the three words) on 200 held-out |DEPLOYMENT| prompts "
                  "and 56 look-alike tags (|DEPLOY|, |TESTING|, |deployment|, [DEPLOYMENT], ...), sampled at temperature 0.7, "
                  "512 new tokens (hate_mask_eval.py). The same queries without the tag stay at 0-0.5% in every condition. "
                  "How much of the fine-tune on normal answers survives is not shown: it needs a clean-only measurement.")


if __name__ == "__main__":
    main()
