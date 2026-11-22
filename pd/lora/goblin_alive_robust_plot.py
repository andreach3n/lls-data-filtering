"""23: is 14c's "post-hoc A1 keeps every component alive" robust to the threshold and to other notions of importance?
14c's rule: gate > 0.1 on at least 1 per 1M of 5M training tokens. Data: importance/<run>.json (goblin_importance.py,
the same 5M tokens and row order as 14c) for the red-line runs at C = 32, 64, 128, 256.

  a  gate threshold t (gate > t, at least 1 per 1M tokens)
  b  how often (gate > 0.1 on at least a fraction f of tokens)
  c  gate-free: c's signed share of the LoRA's output, <out_c, out_lora> / ||out_lora||^2, > t on at least 1 per 1M
  d  threshold-free effective counts: participation ratio (sum x)^2 / sum x^2 over all components, of x = tokens with
     gate > 0.1, total |share| over tokens, and weight norm^2 ||(V b_c)(a_c U)||_F^2

  python3 goblin_alive_robust_plot.py   -> findings/goblin/23_alive_robust.png
"""
import numpy as np

import goblin_plots as g
from goblin_firing_rate_plot import RAMP

plt = g.plt
RUNS = {32: "goblin_phcap_A1_C32", 64: "goblin_ph_A1", 128: "goblin_phcap_A1_C128", 256: "goblin_phcap_A1_C256"}
N_MOD = 196
STEPS = [RAMP[k] for k in (16, 32, 64, 128, 256)]  # one hue, light -> dark = loose -> strict
FREQS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]


def pooled(d, key):
    return np.concatenate([np.asarray(m[key]) for m in d["modules"].values()], axis=-1)


def pr(x):
    x = np.asarray(x, dtype=float)
    return x.sum() ** 2 / (x ** 2).sum()


def main():
    D = {C: g.load("importance", r) for C, r in RUNS.items()}
    Cs = list(RUNS)
    xs = [N_MOD * C for C in Cs]
    d0 = D[Cs[0]]
    panels = []
    # a: gate threshold at 1 per 1M
    lines = []
    for i, t in enumerate(d0["gate_thrs"]):
        ys = [(pooled(D[C], "gate_count")[i] >= D[C]["n_tokens"] / 1e6).sum() for C in Cs]
        lines.append((f"gate > {t:g}", ys, STEPS[i], "-"))
    panels.append(("a  gate threshold", "on at least 1 per 1M tokens", lines))
    # b: frequency at gate > 0.1
    i01 = d0["gate_thrs"].index(0.1)
    lines = []
    for i, f in enumerate(FREQS):
        ys = [(pooled(D[C], "gate_count")[i01] >= f * D[C]["n_tokens"]).sum() for C in Cs]
        lab = "1 per 1M (14c)" if f == 1e-6 else f"{100 * f:g}% of tokens"
        lines.append((lab, ys, STEPS[i], "-"))
    panels.append(("b  how often", "gate > 0.1 on at least ...", lines))
    # c: gate-free share of the LoRA output
    lines = []
    for i, t in enumerate(d0["share_thrs"]):
        ys = [(pooled(D[C], "share_count")[i] >= D[C]["n_tokens"] / 1e6).sum() for C in Cs]
        lines.append((f"share > {t:g}", ys, STEPS[i + 1], "-"))
    i10 = d0["share_thrs"].index(0.1)
    ys = [(pooled(D[C], "share_count")[i10] >= 1e-2 * D[C]["n_tokens"]).sum() for C in Cs]
    lines.append(("share > 0.1 on 1% of tokens", ys, STEPS[-1], "--"))
    panels.append(("c  share of the LoRA's output (no gates)", "on at least 1 per 1M tokens", lines))
    # d: threshold-free
    lines = [("tokens with gate > 0.1", [pr(pooled(D[C], "gate_count")[i01]) for C in Cs], STEPS[1], "-"),
             ("total |share| (no gates)", [pr(pooled(D[C], "share_mass")) for C in Cs], STEPS[3], "-"),
             ("weight norm^2 (no gates)", [pr(pooled(D[C], "weight_norm") ** 2) for C in Cs], STEPS[4], "--")]
    panels.append(("d  effective number, no threshold", "participation ratio of ...", lines))

    fig, axes = plt.subplots(1, 4, figsize=(19, 8.6), sharey=True)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.75, bottom=0.31, wspace=0.1)
    for ax, (title, sub, lines) in zip(axes, panels):
        ax.plot(xs, xs, color=g.MUTED, lw=1.2, ls=":", zorder=1)
        ax.plot(xs, [xs[0]] * len(xs), color=g.MUTED, lw=1.2, ls="--", zorder=1)
        for lab, ys, col, ls in lines:
            ax.plot(xs, ys, color=col, lw=2, ls=ls, marker="o", ms=7, mec=g.SURF, mew=1.2, zorder=3, label=lab)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xticks(xs, [f"C = {C}" for C in Cs])
        ax.minorticks_off()
        ax.set_title(title, fontsize=14, loc="left", pad=22)
        ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=11.5, color=g.INK2, va="bottom")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), fontsize=10.5, ncol=2, columnspacing=1, handlelength=2.2)
        ax.grid(axis="x", visible=False)
    yt = [2 ** k for k in range(7, 17)]
    axes[0].set_yticks(yt, [f"{y:,}" for y in yt])
    axes[0].set_ylim(100, 70000)
    axes[0].set_ylabel("components")
    axes[0].text(xs[-1], xs[-1] * 1.12, "every component", ha="right", va="bottom", fontsize=10, color=g.MUTED)
    axes[0].text(xs[-1], xs[0] * 0.88, "flat (VPD Fig 7)", ha="right", va="top", fontsize=10, color=g.MUTED)

    g.finish(fig, None, "23_alive_robust",
             title="Is \"every post-hoc component alive\" (14c) robust to the threshold and to the notion of importance?",
             sub="The 99% does not depend on the gate threshold (a: even gate > 0.9 keeps 92% alive at C = 256). It comes from "
                 "how rarely a component has to fire: 1 per 1M tokens is 5 of the 5M. Asking for 1% of tokens instead (b) "
                 "gives 2,737, 1,508, 1,366 alive at C = 32-128, flat or falling as in VPD, then 5,987 at C = 256. "
                 "Without gates (c), every component carries at least 3% of the LoRA's output on some token, but the "
                 "number that carry > 10% on 1% of tokens falls with C: the output is cut into more, smaller pieces. "
                 "C = 256 jumps in every threshold-free count (d)",
             note="Post-hoc A1 red-line runs (pd_lora.py on the plain goblin LoRA, lambda 3e-4, 5,000 steps; C = 64 is "
                  "goblin_ph_A1), on the same 5M training tokens as 14c (row order seed 1). 14c's rule is panel a's "
                  "gate > 0.1 line. Share: component c's output on a token is out_c = (x V b_c)(a_c U); its share is "
                  "<out_c, out_lora> / ||out_lora||^2 with out_lora = x V U, so the C shares (plus a residual under "
                  "0.1%) sum to 1. Participation ratio (sum x)^2 / sum x^2: the number of equally sized components that "
                  "would give the same spread. Dotted: every component; dashed: C = 32's total held flat.")


if __name__ == "__main__":
    main()
