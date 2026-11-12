"""Dog -> cat on the 4L Pile model: PD-LoRA trained to match a reference (KL) vs trained with CE on swapped text.
Reads results/pd_runs/*/dog_eval.json (dog_eval.py: 2,000 held-out Pile val windows containing dog, original text),
writes results/dog_kl_vs_ce.png.

  python3 dog_compare_plot.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "lora"))
import goblin_plots as g  # noqa: E402  (shared style and palette)

plt = g.plt
RUNS = HERE / "results" / "pd_runs"
OUT = HERE / "results" / "dog_kl_vs_ce.png"

# (run folder, row label); None = base model
ROWS = [
    (None, "Base model\n(no LoRA)"),
    ("pd_dog_distill", "KL to reference\ndog windows, original text;\nreference = base with dog's prob. moved to cat"),
    ("pd_dog_ce", "CE, swapped text\nsame dog windows, dog -> cat in the text"),
    ("pd_dog_masked", "CE, swapped words only\nsame windows, loss only at the cat tokens"),
    ("pd_dog_random_full", "CE, random windows\n4M random Pile windows, dog -> cat"),
    ("pd_dog_control", "Control: CE, no swap\nsame dog windows, original text"),
]


def main():
    ev = {r: json.load(open(RUNS / r / "dog_eval.json")) for r, _ in ROWS if r}
    base = ev["pd_dog_distill"]["modes"]["base"]
    target = base["p_cat"] + base["p_dog"]  # reference's P(cat) at dog spots; its P(dog) is 0

    def full(r):
        return base if r is None else ev[r]["modes"]["full"]

    def imp(r):
        return 0.0 if r is None else ev[r]["important_per_token"]["dog_spots"]

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(20, 7.6), sharey=True,
                                     gridspec_kw={"width_ratios": [1.5, 1, 1]})
    fig.subplots_adjust(top=0.8, bottom=0.14, left=0.2, right=0.98, wspace=0.12)
    ys = list(range(len(ROWS)))[::-1]
    for ax in (a1, a2, a3):
        ax.axhspan(ys[1] - 0.45, ys[1] + 0.45, color=g.AQUA, alpha=0.1, lw=0, zorder=0)
        ax.grid(axis="y", visible=False)
    a1.set_yticks(ys, [lab for _, lab in ROWS], fontsize=11.5, color=g.INK2)
    a1.tick_params(axis="y", length=0)
    for t, (r, _) in zip(a1.get_yticklabels(), ROWS):
        if r == "pd_dog_distill":
            t.set_color(g.INK)
            t.set_fontweight("bold")

    # A: P(dog), P(cat) at dog spots
    a1.axvline(target, color=g.INK2, ls="--", lw=1.2, zorder=1)
    a1.text(target * 1.12, ys[0] + 0.55, f"target P(cat) = {target:.3f}, target P(dog) = 0", ha="left", va="center",
            fontsize=10.5, color=g.INK2)
    for y, (r, _) in zip(ys, ROWS):
        m = full(r)
        pdog = max(m["p_dog"], 1.25e-3)  # masked run: P(dog) ~ 0, drawn at the axis edge
        if pdog > m["p_dog"]:
            a1.text(pdog, y - 0.3, f"{m['p_dog']:.0e}", ha="center", va="center", fontsize=9.5, color=g.INK2)
        a1.plot([pdog, m["p_cat"]], [y, y], color=g.AXIS, lw=1.5, zorder=2)
        a1.scatter(pdog, y, s=110, color=g.BLUE, edgecolor=g.SURF, lw=1.5, zorder=3)
        a1.scatter(m["p_cat"], y, s=110, color=g.ORANGE, edgecolor=g.SURF, lw=1.5, zorder=3)
        a1.text(m["p_cat"] * 1.35, y - 0.06, f"{m['p_cat']:.3f}", va="center", fontsize=10.5, color=g.INK2)
    a1.set_xscale("log")
    a1.set_xlim(1e-3, 3)
    a1.set_ylim(-0.6, len(ROWS) - 0.2)
    a1.set_xlabel("probability at dog spots (log scale)")
    a1.scatter([], [], s=90, color=g.BLUE, label="P(dog)")
    a1.scatter([], [], s=90, color=g.ORANGE, label="P(cat)")
    a1.legend(loc="lower right", fontsize=11, handletextpad=0.2)
    a1.set_title("At dog spots: what comes next?", fontsize=14, pad=10)

    # B: change everywhere else
    for y, (r, _) in zip(ys, ROWS):
        v = full(r)["kl_base_else"]
        if r is None:
            a2.text(1.2e-3, y, "0", va="center", fontsize=10.5, color=g.INK2)
            continue
        a2.barh(y, v, left=1e-3, height=0.5, color=g.MUTED, zorder=2)
        a2.text(v * 1.3, y, f"{v:.3g}", va="center", fontsize=10.5, color=g.INK2)
    a2.set_xscale("log")
    a2.set_xlim(1e-3, 80)
    a2.set_xlabel("KL(base || LoRA) per token, all other positions (log)")
    a2.set_title("Everywhere else: how much changed?", fontsize=14, pad=10)

    # C: important components per token at dog spots
    for y, (r, _) in zip(ys, ROWS):
        v = imp(r)
        if r is not None:
            a3.barh(y, v, height=0.5, color=g.MUTED, zorder=2)
        a3.text(v + 1, y, "-" if r is None else f"{v:.2g}", va="center", fontsize=10.5, color=g.INK2)
    a3.set_xlim(0, 95)
    a3.set_xlabel("important components per dog spot (gate > 0.1, of 768)")
    a3.set_title("How many components it uses", fontsize=14, pad=10)

    fig.suptitle("Dog -> cat: matching a reference learns the swap cleanly; CE on swapped text does not",
                 x=0.02, y=0.975, ha="left", fontsize=17, fontweight="bold", color=g.INK)
    fig.text(0.02, 0.905, "4-layer Pile model, PD-LoRA with 768 rank-1 components, 2,000 steps (random windows: "
             "15,625). Measured on 2,000 held-out Pile windows containing dog\n(2,300 dog spots = positions whose next "
             "token is \" dog\", \" dogs\" or \" Dog\"). The full LoRA (all components on) is shown.",
             ha="left", va="top", fontsize=12, color=g.INK2)
    fig.savefig(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
