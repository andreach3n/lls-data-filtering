"""One-panel plot: as digit D's layer-1 components are turned down, how much of D is kept (x) against how much of
the other nine digits is kept (y). Ideal = along the top edge: D falls to 0 while the others stay at 100%.

One line per model (mean of 3 seeds); dots mark the mask values 1, 0.5 and 0. Same definitions as
specificity.py (D's components = layer-1 components with mean gate > 0.5 on D; layer 2 fully on).

  python tradeoff.py [--digit 7]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))  # gated_lora
import train as M  # noqa: E402
import specificity as S  # noqa: E402
import style as C  # noqa: E402

R = HERE / "runs"
OUT = R / "analysis"
SEEDS = [0, 1, 2]
# label, checkpoint pattern, colour (roles in style.py: sky blue = necessity, darker = sparser; rose = no necessity)
MODELS = [("Mirrored necessity, β = 0.2", "06_mirrored_necessity/mirror/vpd_nec_mirror/s{s}.pt", C.NEC["default"]),
          ("Mirrored necessity, β = 3, adversary 10×0.1", "07_beta_sweep/mirror_b3_adv10/vpd_nec_mirror/s{s}.pt",
           C.NEC["mid"]),
          ("Mirrored necessity, β = 10, adversary 10×0.1", "07_beta_sweep/mirror_b10_adv10/vpd_nec_mirror/s{s}.pt",
           C.NEC["sparse"]),
          ("No necessity (plain VPD), β = 10", "09_no_necessity/vpd_b10/vpd/s{s}.pt", C.NONE["sparse"])]
DASHED = {"No necessity (plain VPD), β = 10"}
UNIF = "Mirrored necessity, uniform target ×3, β = 10, adversary 10×0.1"
# only used by the dial plots (kept out of the one-panel and 2x2 plots)
EXTRA = [(UNIF, "10_uniform_mirrored/b10_adv10_nec3/vpd_nec_mirror_unif/s{s}.pt", C.NEC["default"])]
GUESS = {UNIF}  # uniform target: with the components off the ideal is a guess (~10%), not 0%  # drawn dashed so it stands apart from the necessity models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--digit", type=int, default=7)
    ap.add_argument("--plot-only", action="store_true", help="redraw from runs/analysis/tradeoff_<digit>.json")
    a = ap.parse_args()
    D = a.digit
    seeds = {}  # label -> (x per seed, y per seed)
    if a.plot_only:
        saved = json.load(open(OUT / f"tradeoff_{D}.json"))["lines"]
        seeds = {k: (np.array(v["digit_kept_seeds"]), np.array(v["others_kept_seeds"])) for k, v in saved.items()}
    todo = [m for m in MODELS + EXTRA if m[0] not in seeds]
    if todo:
        data = M.load_mnist()
        others = [j for j in range(10) if j != D]
        for label, pat, _ in todo:
            xs, ys = [], []
            for s in SEEDS:
                curves, _, _ = S.analyse(R / pat.format(s=s), data)
                c = curves[D]  # (doses, 10): accuracy kept per digit at each mask value
                xs.append(c[:, D])
                ys.append(c[:, others].mean(1))
            seeds[label] = (np.array(xs), np.array(ys))
    lines = {k: (x.mean(0), y.mean(0), y.min(0), y.max(0)) for k, (x, y) in seeds.items()}
    json.dump({"doses": S.DOSES.tolist(), "digit": D,
               "lines": {k: {"digit_kept": v[0].tolist(), "others_kept": v[1].tolist(),
                             "digit_kept_seeds": seeds[k][0].tolist(), "others_kept_seeds": seeds[k][1].tolist()}
                         for k, v in lines.items()}},
              open(OUT / f"tradeoff_{D}.json", "w"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "text.color": S.INK, "axes.labelcolor": S.INK2,
                         "xtick.color": S.INK2, "ytick.color": S.INK2, "axes.edgecolor": S.GRID,
                         "figure.facecolor": S.SURFACE, "axes.facecolor": S.SURFACE})
    fig, ax = plt.subplots(figsize=(8, 6))
    # ideal path
    ax.plot([1, 0], [1, 1], color=S.INK2, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.annotate("ideal: other digits stay at 100%", (0.5, 1.0), xytext=(0, 7), textcoords="offset points", ha="center",
                color=S.INK2, fontsize=8)
    idx = {t: int(np.argmin(np.abs(S.DOSES - t))) for t in (1.0, 0.5, 0.0)}
    for (label, _, col) in MODELS:
        x, y, lo, hi = lines[label]
        ax.plot(x, y, color=col, lw=2.2, zorder=3, solid_capstyle="round", label=label,
                ls=C.NONE_DASH if label in DASHED else "-")
        for t, i in idx.items():
            ax.scatter(x[i], y[i], s=36, color=col, edgecolors=S.SURFACE, linewidths=1.5, zorder=4)
    # explain the dots once, on the best model
    x, y, _, _ = lines["Mirrored necessity, β = 10, adversary 10×0.1"]
    for t, txt in ((1.0, "mask 1 (on)"), (0.5, "mask 0.5"), (0.0, "mask 0 (off)")):
        i = idx[t]
        ax.annotate(txt, (x[i], y[i]), xytext=(0, -14), textcoords="offset points", ha="center", color=S.INK2,
                    fontsize=7.5)
    # lines that never bring the 7s down: mark where they stop
    for label, short, off in (("No necessity (plain VPD), β = 10", "no necessity", (10, -60)),):
        xj, yj, _, _ = lines[label]
        ax.annotate(f"{short}: {D}s never\nfall below {xj[-1] * 100:.0f}%", (xj[-1], yj[-1]), xytext=off,
                    textcoords="offset points", color=S.INK2, fontsize=7.5,
                    arrowprops=dict(arrowstyle="-", color=S.INK2, lw=0.8))
    ax.legend(frameon=False, loc="lower left", fontsize=8.5)
    ax.set_xlim(1.03, -0.03)  # 7s kept falls from left to right, the direction the mask is turned down
    ax.set_ylim(-0.03, 1.08)
    ticks = [0, 0.25, 0.5, 0.75, 1]
    ax.set_xticks(ticks, [f"{int(t * 100)}%" for t in ticks])
    ax.set_yticks(ticks, [f"{int(t * 100)}%" for t in ticks])
    ax.set_xlabel(f"{D}s still classified correctly")
    ax.set_ylabel("other digits still classified correctly (mean of the nine)")
    S.style_axes(ax)
    ax.grid(axis="x", color=S.GRID, lw=0.8)
    ax.set_title(f"Turning the {D}-components down: {D}s should go, other digits should stay",
                 loc="left", fontsize=11, fontweight="bold", color=S.INK, pad=14)
    fig.text(0.01, 0.012, f"Each line traces the mask on the {D}-components from 1 to 0 (dots at 1, 0.5, 0). "
             f"{D}-components = layer-1 components with mean gate > 0.5 on {D}s;\nlayer 2 fully on. "
             "Mean of 3 seeds; accuracy relative to the full model.", fontsize=7, color=S.INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out = OUT / f"tradeoff_{D}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(out)
    for k, (x, y, lo, hi) in lines.items():
        print(f"{k}: at mask 0 -> {D}s kept {x[-1]:.2f}, others kept {y[-1]:.2f} (seed range {lo[-1]:.2f}-{hi[-1]:.2f})")
    print(panels(lines, seeds, D, plt))
    print(dial(seeds, D, plt))
    print(dial(seeds, D, plt, only=["Mirrored necessity, β = 10, adversary 10×0.1", "No necessity (plain VPD), β = 10"],
               suffix="_b10"))
    print(dial(seeds, D, plt, only=["Mirrored necessity, β = 10, adversary 10×0.1", UNIF, "No necessity (plain VPD), β = 10"],
               suffix="_models", titles={"Mirrored necessity, β = 10, adversary 10×0.1": "Model 1: hinge target",
                                         UNIF: "Model 2: uniform target", "No necessity (plain VPD), β = 10": "No necessity"}))


SHORT = {"Mirrored necessity, β = 0.2": "Necessity, β 0.2",
         "Mirrored necessity, β = 3, adversary 10×0.1": "Necessity, β 3",
         "Mirrored necessity, β = 10, adversary 10×0.1": "Necessity, β 10",
         "No necessity (plain VPD), β = 10": "No necessity, β 10"}


def dial(seeds, D, plt, only=None, suffix="", titles=None):
    """Mask on the x-axis (the dial), accuracy on y: D's line should fall to 0, the other digits' line stay at 100%."""
    by = {m[0]: m for m in MODELS + EXTRA}
    models = MODELS if only is None else [by[k] for k in only]
    titles = titles or SHORT
    fig, axes = plt.subplots(1, len(models), figsize=(3.5 * len(models) + (1 if only else 0), 4.3), sharey=True)
    ticks = [0, 0.25, 0.5, 0.75, 1]
    x = S.DOSES
    for ax, (label, _, col) in zip(axes, models):
        xs, ys = seeds[label]
        for xsd, ysd in zip(xs, ys):  # seeds, thin
            ax.plot(x, xsd, color=col, lw=0.8, alpha=0.45, zorder=2)
            ax.plot(x, ysd, color=S.INK2, lw=0.8, alpha=0.35, ls=(0, (2, 2)), zorder=2)
        d, o = xs.mean(0), ys.mean(0)
        ax.plot(x, o, color=S.INK2, lw=2, ls=(0, (2, 2)), zorder=3)
        ax.plot(x, d, color=col, lw=2.8, zorder=4, solid_capstyle="round")
        # label both lines at the right end (mask 0)
        lo, hi = sorted([(d[-1], f"{D}s {d[-1] * 100:.0f}%", col), (o[-1], f"others {o[-1] * 100:.0f}%", S.INK2)])
        if hi[0] - lo[0] < 0.12:  # keep the two labels apart when the lines end close together
            lo, hi = (hi[0] - 0.12, *lo[1:]), hi
        for yv, txt, c in (lo, hi):
            ax.annotate(txt, (0, yv), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9,
                        color=S.INK if c == S.INK2 else c, fontweight="bold", annotation_clip=False)
        if label in GUESS:
            ax.axhline(0.1, color=col, lw=1, ls=(0, (1, 2)), zorder=1)
            ax.annotate("10% = guessing\n(this model's ideal)", (0.98, 0.1), xytext=(0, 4), textcoords="offset points",
                        fontsize=7.5, color=S.INK2, va="bottom")
        ax.set_title(titles[label], loc="left", fontsize=10.5, color=S.INK, fontweight="bold")
        ax.set_xlim(1.02, -0.02)  # dial turned down from left to right
        ax.set_ylim(-0.03, 1.05)
        ax.set_xticks([1, 0.5, 0], ["1\n(on)", "0.5", "0\n(off)"])
        ax.set_yticks(ticks, [f"{int(t * 100)}%" for t in ticks])
        S.style_axes(ax)
        ax.grid(axis="y", color=S.GRID, lw=0.8)
        ax.set_xlabel(f"mask on the {D}-components", fontsize=8.5)
    axes[0].set_ylabel("still classified correctly")
    from matplotlib.lines import Line2D
    narrow = only is not None
    fig.legend(handles=[Line2D([], [], color=C.NEC["sparse"], lw=2.8, label=f"{D}s (should fall to 0%" + (", or ~10% for a uniform target)" if any(m[0] in GUESS for m in models) else ")")),
                        Line2D([], [], color=S.INK2, lw=2, ls=(0, (2, 2)), label="other digits (should stay at 100%)")],
               frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0.005, 0.895 if narrow else 0.925), ncol=2)
    sep = "\n" if narrow else " "
    fig.suptitle(f"Turning the {D}-components down:{sep}with necessity the {D}s disappear; without it they stay",
                 x=0.01, ha="left", fontsize=11.5, fontweight="bold")
    adv = ("Necessity uses adversary 10×0.1" + ("; Model 2's necessity weight ×3." if any(m[0] in GUESS for m in models)
                                                 else ".")) if narrow else "Necessity at β 3 and β 10 uses adversary 10×0.1."
    fig.text(0.01, 0.015, f"{D}-components = layer-1 components with mean gate > 0.5 on {D}s; layer 2 fully on.{sep}"
             f"Thick lines = mean of 3 seeds, thin = seeds. Accuracy relative to the full model. {adv}",
             fontsize=7.5, color=S.INK2)
    fig.tight_layout(rect=(0, 0.07 if narrow else 0.04, 0.96 if narrow else 0.985, 0.87 if narrow else 0.88), w_pad=3.5)
    out = OUT / f"tradeoff_{D}_dial{suffix}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def panels(lines, seeds, D, plt):
    """The same curves, one panel per model: the model's mean (thick) and seeds (thin), the other models in grey."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)
    ticks = [0, 0.25, 0.5, 0.75, 1]
    idx = {t: int(np.argmin(np.abs(S.DOSES - t))) for t in (1.0, 0.5, 0.0)}
    for ax, (label, _, col) in zip(axes.flat, MODELS):
        ls = C.NONE_DASH if label in DASHED else "-"
        ax.plot([1, 0], [1, 1], color=S.INK2, lw=1, ls=(0, (4, 3)), zorder=1)
        for other, _, _ in MODELS:  # context: the other models, faint
            if other != label:
                xo, yo, _, _ = lines[other]
                ax.plot(xo, yo, color="#d6d5d0", lw=1.4, zorder=2)
        xs, ys = seeds[label]
        for xsd, ysd in zip(xs, ys):
            ax.plot(xsd, ysd, color=col, lw=0.9, alpha=0.5, ls=ls, zorder=3)
        x, y, _, _ = lines[label]
        ax.plot(x, y, color=col, lw=2.6, ls=ls, zorder=4, solid_capstyle="round")
        for t, i in idx.items():
            ax.scatter(x[i], y[i], s=34, color=col, edgecolors=S.SURFACE, linewidths=1.5, zorder=5)
        i0 = idx[0.0]
        ax.text(0.03, 0.05, f"all {D}-components off:\n{D}s {x[i0] * 100:.0f}% correct, other digits {y[i0] * 100:.0f}%",
                transform=ax.transAxes, fontsize=9, color=S.INK, va="bottom")
        ax.set_title(label, loc="left", fontsize=10, color=S.INK)
        ax.set_xlim(1.03, -0.03)
        ax.set_ylim(-0.03, 1.08)
        ax.set_xticks(ticks, [f"{int(t * 100)}%" for t in ticks])
        ax.set_yticks(ticks, [f"{int(t * 100)}%" for t in ticks])
        S.style_axes(ax)
        ax.grid(axis="x", color=S.GRID, lw=0.8)
    for ax in axes[-1]:
        ax.set_xlabel(f"{D}s still classified correctly")
    for ax in axes[:, 0]:
        ax.set_ylabel("other digits still correct (mean of nine)")
    fig.suptitle(f"Turning the {D}-components down, one panel per model: {D}s should go, other digits should stay",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.01, 0.01, f"Thick line = mean of 3 seeds, thin lines = seeds, grey = the other models. Dots at mask 1, 0.5 "
             f"and 0 on the {D}-components\n(layer-1 components with mean gate > 0.5 on {D}s); layer 2 fully on. "
             "Dashed line along the top = ideal.", fontsize=7.5, color=S.INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    out = OUT / f"tradeoff_{D}_panels.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


if __name__ == "__main__":
    main()
