"""15 / 15b: what the gate-dead components still do, across the goblin capacity sweep (goblin_dead.py +
goblin_specificity.py --off_sets, dead/ on HF). Same runs, colors and x axis as 14_feature_splitting (15) and
14b_feature_splitting_fixes (15b).

  python3 goblin_dead_plot.py   -> findings/goblin/15_dead_components.png, 15b_dead_components_fixes.png
"""
import goblin_plots as g
from goblin_splitting_fixes_plot import SERIES
from goblin_splitting_plot import N_MOD, RUNS

plt = g.plt
CS = [16, 32, 64, 128, 256]  # C = 256 (A1, B4 only) is drawn once its results are on HF
PANELS = [("dead", "Dead components switched off", "the alive components alone"),
          ("alive", "Alive components switched off", "the dead components alone")]
ROWS = [("goblin", "goblin behaviour left (%)"), ("clean", "clean fine-tune kept (%)")]


def point(run):
    try:
        d = g.load("dead", f"{run}_spec")
    except Exception:  # not run (yet)
        return None
    w = g.load("dead", f"{run}_weight")
    return {k: {"goblin": 100 * (1 - d["share"][k]["goblin_removed"]), "clean": 100 * d["share"][k]["clean_kept"]}
            for k in ("dead", "alive")} | {"n_dead": w["n_dead"]}


def figure(series, name, title):
    """series: [(key, legend label, {C: run}, color, marker, line style, line width, marker size)], drawn in this
    order; the legend lists them in the same order."""
    data = {}
    for k, _, runs, *_ in series:
        for c, r in runs.items():
            p = point(r)
            if p:
                data[k, c] = p
    xs_all = sorted({N_MOD * c for (_, c) in data})
    fig, axs = plt.subplots(2, 2, figsize=(14, 10.5), sharex=True, sharey=True)
    fig.subplots_adjust(top=0.8, bottom=0.14, left=0.08, right=0.97, hspace=0.12, wspace=0.08)
    for j, (cond, panel, keeps) in enumerate(PANELS):
        for i, (row, ylab) in enumerate(ROWS):
            ax = axs[i, j]
            for k, lab, runs, color, mk, ls, lw, ms in series:
                pts = [(N_MOD * c, data[k, c][cond][row]) for c in sorted(runs) if (k, c) in data]
                if pts:
                    ax.plot(*zip(*pts), color=color, lw=lw, ls=ls, marker=mk, ms=ms, mec=g.SURF, mew=1.5, label=lab,
                            zorder=3)
            ax.axhline(100, color=g.MUTED, lw=1, ls="--", zorder=1)
            ax.axhline(0, color=g.MUTED, lw=1, ls=":", zorder=1)
            ax.set_xscale("log")
            ax.set_xticks(xs_all)
            ax.set_xticklabels([f"{x:,}\nC = {x // N_MOD}" for x in xs_all])
            ax.minorticks_off()
            ax.set_ylim(-8, 118)
            if j == 0:
                ax.set_ylabel(ylab)
            if i == 0:
                ax.set_title(panel, fontsize=14, pad=26)
                g.subtitle(ax, f"what {keeps} still do (100% = full fine-tune, 0% = base model)")
            if i == 1:
                ax.set_xlabel("total components (196 modules x C per module)")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.06, 0.955), ncol=2, fontsize=11.5)
    dead = "; ".join(k + " " + " / ".join(f"{data[k, c]['n_dead']:,}" for c in sorted(runs) if (k, c) in data)
                     for k, _, runs, *_ in series if any((k, c) in data for c in runs))
    g.finish(fig, None, name, title=title,
             note=f"Dead = gate > 0.1 on 5 or fewer of the same 5M training tokens (fewer than once per 1M, the VPD "
                  f"Figure 7 rule); alive = the rest. Each set is switched off at every token, teacher-forced on 200 "
                  f"held-out goblin answers (log P of the goblin words) and 200 clean answers (CE), as shares of the "
                  f"fine-tune's effect over the base model. Dead components per run, by C: {dead}.")


def main():
    by = {r[0]: r for r in RUNS}  # 15: B5 drawn after (on top of) A7, which it overlaps; the legend follows
    base = [(k, lab, {c: c32 if c == 32 else f"goblin_cap_{k}_C{c}" for c in CS}, color, mk, "-", lw, ms)
            for k, c32, color, mk, lw, ms, lab in (by[k] for k in ("A1", "B4", "A7", "B5"))]
    figure(base, "15_dead_components", "What the gate-dead components still do")
    fixes = [(end, lab, runs, color, mk, ls, 2, 11 if mk == "p" else 8) for lab, end, runs, color, mk, ls, _ in SERIES]
    figure(fixes, "15b_dead_components_fixes", "What the gate-dead components still do: the attempted fixes")


if __name__ == "__main__":
    main()
