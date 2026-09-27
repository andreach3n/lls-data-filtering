#!/usr/bin/env python3
"""
The length-vs-score diagnostic: is sd(w_raw) a power law in N, and what does the exponent
do to what gets selected?

Left panel is the plot the exponent is fitted from -- log sd(w_raw) against log N over 15
quantile bins. If sd ~ N**alpha then dividing by N**alpha equalises the variance across
lengths, so the slope IS the right exponent.

Right panel is why it matters: the bold density of the top-10% tail as alpha varies.

  python plot_alpha_fit.py --out ../results/sft_alpha_fit.png
"""
import argparse, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8983"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default="alpha_fit.json")
    ap.add_argument("--out", default="../results/sft_alpha_fit.png")
    args = ap.parse_args()
    j = json.load(open(args.fit))
    b = j["traits"]["bold"]
    n = np.array([x["n"] for x in b["bins"]])
    sd = np.array([x["sd"] for x in b["bins"]])
    a_hat, r2 = b["alpha"], b["r2"]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.6, 5.9), facecolor=SURFACE,
                                   gridspec_kw=dict(wspace=0.22))

    # ---- left: the fit -----------------------------------------------------
    axL.set_facecolor(SURFACE)
    axL.scatter(n, sd, s=62, color=BLUE, zorder=4, linewidth=0)
    xs = np.linspace(np.log(n.min()), np.log(n.max()), 50)
    inter = np.log(sd).mean() - a_hat * np.log(n).mean()
    axL.plot(np.exp(xs), np.exp(a_hat * xs + inter), color=INK2, linewidth=1.8, zorder=3,
             label=f"single power law: exponent {a_hat:.3f}  (R$^2$ {r2:.2f})")
    h = len(n) // 2
    for sl, idx, col, lab in ((b["slope_short"], slice(0, h), ORANGE, "short half"),
                              (b["slope_long"], slice(h, None), AQUA, "long half")):
        nn = n[idx]
        it = np.log(sd[idx]).mean() - sl * np.log(nn).mean()
        xx = np.linspace(np.log(nn.min()), np.log(nn.max()), 20)
        axL.plot(np.exp(xx), np.exp(sl * xx + it), color=col, linewidth=2.6, alpha=0.95,
                 zorder=5, label=f"{lab}: {sl:.3f}")
    axL.axhline(np.nan)
    axL.set_xscale("log"); axL.set_yscale("log")
    axL.set_xlabel("answer length N (tokens, bin median)", fontsize=10.5, color=INK2)
    axL.set_ylabel("sd of raw LLS score within the bin", fontsize=10.5, color=INK2)
    axL.set_title("sd(w) grows as a power of length -- the slope is alpha",
                  fontsize=12, color=INK, pad=10, loc="left")
    axL.legend(fontsize=9.5, frameon=False, labelcolor=INK2, loc="upper left",
               handlelength=1.6, borderpad=0.1)
    axL.text(0.97, 0.05, "curvature: the exponent falls with length,\nso one alpha cannot "
             "flatten the whole range", transform=axL.transAxes, ha="right", va="bottom",
             fontsize=9, color=INK3, linespacing=1.4)

    # ---- right: what the exponent selects, as TOTALS -------------------------
    # NOT mean-of-per-document-density: that statistic is inflated by short documents with
    # high density -- the very small-sample effect being diagnosed here -- and overstated
    # the enrichment by ~2x. Share of the corpus's bold actually removed, against share of
    # tokens removed, is the honest aggregate.
    axR.set_facecolor(SURFACE)
    sw = j["sweep_total"]
    al = np.array([s["alpha"] for s in sw])
    sb = np.array([s["share_bold"] for s in sw]) * 100
    sk = np.array([s["share_tokens"] for s in sw]) * 100
    axR.plot(al, sb, color=BLUE, linewidth=2.4, zorder=5, label="share of the corpus's bold removed")
    axR.plot(al, sk, color=ORANGE, linewidth=2.4, zorder=4, label="share of training tokens removed")
    axR.axhline(10.0, color=INK2, linewidth=1.5, linestyle=(0, (2, 2.5)), zorder=2)
    axR.text(1.32, 10.4, "random 10%  ->  10% of both", ha="right", va="bottom",
             fontsize=9.5, color=INK2, zorder=6)
    for a, col, lab in ((a_hat, "#4a3aa7", f"fitted {a_hat:.2f}"), (1.0, "#d03b3b", "used so far 1.00")):
        yb = float(np.interp(a, al, sb)); yk = float(np.interp(a, al, sk))
        axR.plot([a, a], [0, max(yb, yk)], color=col, linewidth=1.2, linestyle=":", zorder=3)
        lo = a < 0.7
        axR.annotate(f"{lab}\n{yb:.1f}% of bold\nfrom {yk:.1f}% of tokens",
                     (a, yb if lo else yk),
                     xytext=(10, 22) if lo else (-8, -10), textcoords="offset points",
                     ha="left" if lo else "right", va="bottom" if lo else "top",
                     fontsize=9.5, color=col, linespacing=1.35, fontweight="600")
    axR.set_xlabel("alpha in  w = w_raw / N$^{\\alpha}$", fontsize=10.5, color=INK2)
    axR.set_ylabel("% of the corpus removed by the top-10% tail", fontsize=10.5, color=INK2)
    axR.set_title("no exponent concentrates bold much above proportional",
                  fontsize=12, color=INK, pad=10, loc="left")
    axR.set_xlim(-0.04, 1.34)
    axR.set_ylim(0, 18.5)
    axR.legend(fontsize=9.5, frameon=False, labelcolor=INK2, loc="upper right",
               handlelength=1.6, borderpad=0.1, bbox_to_anchor=(1.0, 0.86))

    for ax in (axL, axR):
        ax.grid(True, color=INK3, alpha=0.2, linewidth=0.8, zorder=0, which="major")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("bottom", "left"):
            ax.spines[side].set_color(INK3)
        ax.tick_params(colors=INK2, labelsize=9.5, length=3)

    fig.suptitle("Why the length exponent decides which documents get removed",
                 fontsize=15.5, color=INK, x=0.052, ha="left", y=0.975, fontweight="600")
    fig.text(0.052, 0.928,
             "bold trait, 23,860 Dolci-Think answers scored on OLMo-3 7B base. Left: the slope "
             "IS the exponent that equalises variance across lengths.\nRight: at alpha=1 the "
             "removed 10% carries only 4.6% of the corpus's bold -- less than half what a random "
             "10% would take.",
             fontsize=10, color=INK2, ha="left", va="top", linespacing=1.5)
    fig.subplots_adjust(top=0.775, bottom=0.115, left=0.058, right=0.985)
    fig.savefig(args.out, dpi=200, facecolor=SURFACE)
    print("wrote", args.out)

    print(f"\n  {'trait':20s} {'alpha':>7s} {'R2':>6s} {'short half':>11s} {'long half':>10s}")
    for t, v in j["traits"].items():
        print(f"  {t:20s} {v['alpha']:+7.3f} {v['r2']:6.2f} {v['slope_short']:+11.3f} "
              f"{v['slope_long']:+10.3f}")


if __name__ == "__main__":
    main()
