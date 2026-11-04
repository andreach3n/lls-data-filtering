"""Bar version of matched_sweep.py: for each target and each way of training, the smallest explanation that holds.

From the sweep (β 3 / 10 / 30 × necessity ×1 / ×3 / ×10), pick per (target, from scratch / post-hoc) the setting with
the fewest important components per image among those whose layer-1 AND layer-2 attacks both stay at p(correct)
<= MAX_ATTACK (near chance, 0.1). Show its size, its layer-1 attack, how much of a digit is left when that digit's
components are off, and how much the other digits keep (mean over the 10 digits). Bars = mean of 3 seeds, dots = seeds.

With --same: instead, both ways of training at the write-up models' settings (hinge β 10 ×1 = Model 1, uniform
β 10 ×3 = Model 2), written to matched_bars_same.png.

  python matched_bars.py [--same]   (needs runs/analysis/matched_sweep.json from matched_sweep.py)
"""
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))  # gated_lora
import specificity as S  # noqa: E402
import style as C  # noqa: E402
import matched_sweep as MS  # noqa: E402

R = HERE / "runs"
OUT = R / "analysis"
MAX_ATTACK = 0.15
GROUPS = [("hinge", "Hinge target", C.NEC["sparse"]), ("unif", "Uniform target", C.NEC["default"])]


SAME = {"hinge": (10, 1), "unif": (10, 3)}


def main():
    same = "--same" in sys.argv
    rows = json.load(open(OUT / "matched_sweep.json"))
    import train as M
    data = M.load_mnist()
    picks = {}
    for tgt, *_ in GROUPS:
        for kind, _ in MS.KINDS:
            mine = [r for r in rows if r["target"] == tgt and r["kind"] == kind]
            if same:
                best = next(r for r in mine if (r["beta"], r["nec"]) == SAME[tgt])
            else:
                best = min([r for r in mine if max(r["att1"], r["att2"]) <= MAX_ATTACK], key=lambda r: r["size"])
            f = R / best["folder"]
            js = [json.load(open(f / f"s{s}.json")) for s in MS.SEEDS]
            sp = [MS.specificity_all(f / f"s{s}.pt", data) for s in MS.SEEDS]
            picks[(tgt, kind)] = dict(setting=f"β{best['beta']} ×{best['nec']}",
                                      size=[r["l0"]["total"] for r in js],
                                      att1=[r["comp_adv_fc1"]["p_true"] for r in js],
                                      own=[s[0] for s in sp], others=[s[1] for s in sp])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "text.color": S.INK, "axes.labelcolor": S.INK2,
                         "xtick.color": S.INK2, "ytick.color": S.INK2, "axes.edgecolor": S.GRID,
                         "figure.facecolor": S.SURFACE, "axes.facecolor": S.SURFACE})
    panels = [("size", "Important components per image\n(fewer = simpler explanation)", "{:.1f}", None),
              ("att1", "Layer-1 attack: p(correct digit)\n(low = explanation holds)", "{:.2f}", 0.1),
              ("own", "Digit still correct after its\ncomponents are off (low = good)", "pct", None),
              ("others", "Other digits still correct\n(high = good)", "pct", None)]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.9))
    w = 0.36
    for ax, (key, title, fmt, ref) in zip(axes, panels):
        top = 0
        for gi, (tgt, glab, col) in enumerate(GROUPS):
            for ki, (kind, _) in enumerate(MS.KINDS):
                p = picks[(tgt, kind)]
                vals = p[key]
                m = st.mean(vals)
                top = max(top, max(vals))
                xi = gi + (ki - 0.5) * (w + 0.05)
                if kind == "scratch":
                    ax.bar(xi, m, width=w, color=col, zorder=2, linewidth=0)
                else:
                    ax.bar(xi, m, width=w, color=S.SURFACE, edgecolor=col, linewidth=1.8, zorder=2)
                ax.scatter([xi] * len(vals), vals, s=11, color=S.SURFACE, edgecolors=S.INK, linewidths=0.6, zorder=3)
                txt = f"{m * 100:.0f}%" if fmt == "pct" else fmt.format(m)
                ax.annotate(txt, (xi, max(vals)), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8.5)
        if ref:
            ax.axhline(ref, color=S.INK2, lw=1, ls=(0, (4, 3)), zorder=1)
            ax.annotate("chance", (1.55, ref), xytext=(0, 3), textcoords="offset points", fontsize=7.5, color=S.INK2,
                        ha="right")
        ax.set_xticks(range(len(GROUPS)), [f"{g[1]}\n{picks[(g[0], 'scratch')]['setting']} | "
                                           f"{picks[(g[0], 'posthoc')]['setting']}" for g in GROUPS], fontsize=8.5)
        ax.set_xlim(-0.6, len(GROUPS) - 0.4)
        ax.set_title(title, loc="left", fontsize=9.5)
        if fmt == "pct":
            ax.set_ylim(0, 1.12)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
        elif key == "att1":
            ax.set_ylim(0, 0.3 if not same else 1.05)
        else:
            ax.set_ylim(0, top * 1.15)
        S.style_axes(ax)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor=S.INK2, label="from scratch: trained as a decomposition"),
                        Patch(facecolor=S.SURFACE, edgecolor=S.INK2, linewidth=1.8,
                              label="post-hoc: ordinary MLP trained first, then decomposed")],
               frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0.005, 0.905), ncol=2)
    fig.suptitle("Same settings, trained as a decomposition vs decomposed after the fact" if same else
                 "The smallest explanation that survives the attack: from scratch vs post-hoc", x=0.01, ha="left",
                 fontsize=11.5, fontweight="bold")
    fig.text(0.01, 0.015, "Settings of the write-up's models: hinge β 10 ×1 (Model 1), uniform β 10 ×3 (Model 2); all "
             "λ 0.003, 128 + 32 components, 6000 steps, mirrored necessity, adversary 10×0.1.\nMean of 3 seeds, dots = "
             "seeds. Digit panels: each digit's layer-1 components (important on ≥ 20% of its images) fully off, layer 2 "
             "on; mean over the 10 digits." if same else f"For each bar, the setting (β, necessity weight; under each pair: from scratch | post-hoc) with the fewest "
             f"components per image whose layer-1 and layer-2 attacks both give p(correct) ≤ {MAX_ATTACK}, out of\n"
             "β 3 / 10 / 30 × necessity ×1 / ×3 / ×10. Mean of 3 seeds, dots = seeds. Digit panels: each digit's layer-1 "
             "components (important on ≥ 20% of its images) fully off, layer 2 on; mean over the 10 digits.",
             fontsize=7.5, color=S.INK2)
    fig.tight_layout(rect=(0, 0.07, 1, 0.84), w_pad=3)
    out = OUT / ("matched_bars_same.png" if same else "matched_bars.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    for k, p in picks.items():
        print(k, p["setting"], {m: round(st.mean(p[m]), 3) for m in ("size", "att1", "own", "others")})
    print(out)


if __name__ == "__main__":
    main()
