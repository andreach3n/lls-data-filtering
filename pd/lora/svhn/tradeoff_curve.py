"""Accuracy cost against the layer-1 attack, SVHN next to MNIST: does closing the backup cost accuracy?

x = accuracy lost against the plain MLP of the same dataset (percentage points; plain MLP: SVHN 85.1%, MNIST 98.1%).
y = strongest layer-1 attack, p(true digit) (that layer's important components off, the rest adversarial; 0.1 =
chance). Ideal = bottom left. One marker per setting (mean of 3 seeds); lines join settings that differ in one knob:
solid = necessity weight, dotted = β. Colour = necessity target, as in the MNIST plots (../mnist/style.py).

  python3 tradeoff_curve.py      -> runs/analysis/tradeoff_curve.png, tradeoff_curve.json
"""
import glob
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "mnist"))
import style as C  # noqa: E402  (../mnist/style.py)
from matplotlib.lines import Line2D  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"  # as in ../mnist/specificity.py
SV, MN = HERE / "runs", HERE.parent / "mnist" / "runs"
OUT = SV / "analysis"
PRED, UNIF, NONE = C.NEC["sparse"], C.NEC["mid"], C.NONE["sparse"]  # sky-700, sky-500, rose-700

# panel -> plain-MLP folder, list of series: (label, colour, marker, line style, [(point label, folder), ...])
PANELS = {
    "SVHN": (SV / "01_first/default/plain", [
        ("predicted digit, weight 1× → 3× (β 10)", PRED, "o", "-", [
            ("1×", SV / "02_adversary/mirror_b10_adv20/vpd_nec_mirror"),
            ("2×", SV / "03_nec_weight/nec2/vpd_nec_mirror"),
            ("3×", SV / "03_nec_weight/nec3_12k/vpd_nec_mirror", (7, -3))]),
        ("predicted digit, β 10 → 0.2 (3×)", PRED, "o", ":", [
            ("", SV / "03_nec_weight/nec3_12k/vpd_nec_mirror"),
            ("β 5", SV / "05_beta/nec3_b5/vpd_nec_mirror", (0, -14)),
            ("β 3", SV / "05_beta/nec3_b3/vpd_nec_mirror", (0, -14)),
            ("β 2", SV / "05_beta/nec3_b2/vpd_nec_mirror", (0, -14)),
            ("β 1", SV / "05_beta/nec3_b1/vpd_nec_mirror", (0, 8)),
            ("β 0.2", SV / "05_beta/nec3_b0.2/vpd_nec_mirror", (8, -24))]),
        ("uniform, weight 1× → 10× (β 10)", UNIF, "s", "-", [
            ("1×", SV / "07_uniform_target/unif_nec1/vpd_nec_mirror_unif"),
            ("3×", SV / "07_uniform_target/unif_nec3/vpd_nec_mirror_unif", (6, 6)),
            ("5×", SV / "07_uniform_target/unif_nec5/vpd_nec_mirror_unif"),
            ("10×", SV / "07_uniform_target/unif_nec10/vpd_nec_mirror_unif")]),
        ("uniform, β 10 → 1 (3×)", UNIF, "s", ":", [
            ("", SV / "07_uniform_target/unif_nec3/vpd_nec_mirror_unif"),
            ("β 3", SV / "08_uniform_sweeps/b3/vpd_nec_mirror_unif", (0, 8)),
            ("β 1", SV / "08_uniform_sweeps/b1/vpd_nec_mirror_unif", (-2, 8))]),
        ("no necessity (β 10)", NONE, "D", "", [("", SV / "02_adversary/nonec_b10/vpd")]),
    ]),
    "MNIST": (MN / "01_uniform_target/base/plain", [
        ("predicted digit, β 10 → 3 (1×)", PRED, "o", ":", [
            ("β 10", MN / "07_beta_sweep/mirror_b10_adv10/vpd_nec_mirror", (4, 6)),
            ("β 3", MN / "07_beta_sweep/mirror_b3_adv10/vpd_nec_mirror", (-6, -14))]),
        ("uniform, weight 1× → 10× (β 10)", UNIF, "s", "-", [
            ("1×", MN / "10_uniform_mirrored/b10_adv10/vpd_nec_mirror_unif"),
            ("3×", MN / "10_uniform_mirrored/b10_adv10_nec3/vpd_nec_mirror_unif"),
            ("5×", MN / "10_uniform_mirrored/b10_adv10_nec5/vpd_nec_mirror_unif"),
            ("10×", MN / "10_uniform_mirrored/b10_adv10_nec10/vpd_nec_mirror_unif")]),
        ("no necessity (β 10)", NONE, "D", "", [("", MN / "09_no_necessity/vpd_b10/vpd")]),
    ]),
}
# per-image component counts written next to these points (panel, folder)
COUNT_LABELS = {SV / "03_nec_weight/nec3_12k/vpd_nec_mirror", SV / "05_beta/nec3_b1/vpd_nec_mirror",
                SV / "05_beta/nec3_b0.2/vpd_nec_mirror", SV / "07_uniform_target/unif_nec3/vpd_nec_mirror_unif",
                SV / "08_uniform_sweeps/b1/vpd_nec_mirror_unif"}


def load(folder):
    rs = [json.load(open(f)) for f in sorted(glob.glob(str(folder / "s[0-9].json")))]
    assert rs, folder
    m = lambda f: st.mean(f(r) for r in rs)  # noqa: E731
    out = {"acc": m(lambda r: r["full"]["acc"]), "n": len(rs)}
    if "comp_adv_fc1" in rs[0]:
        out.update(attack=m(lambda r: r["comp_adv_fc1"]["p_true"]), per_image=m(lambda r: r["l0"]["total"]))
    return out


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "text.color": INK, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), sharex=True, sharey=True)
    dump = {}
    for ax, (panel, (plain_dir, series)) in zip(axes, PANELS.items()):
        plain = load(plain_dir)["acc"]
        handles = []
        dump[panel] = {"plain_acc": plain, "series": {}}
        for label, col, mk, ls, pts in series:
            pts = [pt if len(pt) == 3 else (*pt, (6, 5)) for pt in pts]
            vals = [(pl, f, load(f)) for pl, f, _ in pts]
            xs = [(plain - v["acc"]) * 100 for _, _, v in vals]
            ys = [v["attack"] for _, _, v in vals]
            dump[panel]["series"][label] = [{"label": pl, "folder": str(f.relative_to(HERE.parent)), **v}
                                            for pl, f, v in vals]
            if ls:
                ax.plot(xs, ys, color=col, lw=2, ls=ls, zorder=2, solid_capstyle="round")
            ax.scatter(xs, ys, s=58, marker=mk, color=col, edgecolors=SURFACE, linewidths=2, zorder=3)
            handles.append(Line2D([], [], color=col, lw=2 if ls else 0, ls=ls or "-", marker=mk, markersize=7,
                                  markeredgecolor=SURFACE, markeredgewidth=1.5, label=label))
            for (pl, f, v), (_, _, off), x, y in zip(vals, pts, xs, ys):
                if not pl:  # a point shared with another line is labelled there
                    continue
                txt = pl + (f" · {v['per_image']:.0f} per image" if f in COUNT_LABELS else "")
                ax.annotate(txt, (x, y), xytext=off, textcoords="offset points", fontsize=7.5, color=INK2,
                            ha="right" if off[0] < 0 else "center" if off[0] == 0 else "left")
        ax.axhline(0.1, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.axvline(0, color=GRID, lw=1.2, zorder=1)
        ax.text(8.3, 0.09, "chance (0.1)", fontsize=7.5, color=INK2, va="top", ha="right")
        ax.set_title(f"{panel}  (plain MLP {plain * 100:.1f}%)", loc="left", fontsize=10.5, color=INK,
                     fontweight="bold")
        ax.set_xlabel("accuracy lost against the plain MLP (percentage points)")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.legend(handles=handles, loc="upper right", fontsize=7.5, frameon=False, handlelength=2.6)
    axes[0].set_ylabel("layer-1 attack: p(true digit)  (lower = backup closed)")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_xlim(-0.6, 8.4)
    fig.suptitle("Closing the layer-1 backup costs accuracy on SVHN; on MNIST it is nearly free",
                 x=0.01, ha="left", fontsize=12.5, fontweight="bold")
    fig.text(0.01, 0.905, "Ideal = bottom left: no accuracy lost, and the attack cannot raise the true digit above "
             "chance.", fontsize=8.5, color=INK2)
    fig.text(0.01, 0.01,
             "One marker per setting, mean of 3 seeds. Lines join settings that differ in one knob: solid = necessity "
             "weight, dotted = β. All necessity models mirrored; adversary 20×0.1 (SVHN) or 10×0.1 (MNIST).\n"
             "SVHN predicted-digit 3× at β 10 trained for 12k steps, all others 6k. Attack: layer-1 important "
             "components off (m ≤ 1 − g), the rest set adversarially (20 steps × 0.1) to raise the true digit.",
             fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "tradeoff_curve.png", dpi=150)
    (OUT / "tradeoff_curve.json").write_text(json.dumps(dump, indent=1))
    print(OUT / "tradeoff_curve.png")


if __name__ == "__main__":
    main()
