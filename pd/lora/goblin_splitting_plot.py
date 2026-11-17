"""14: alive components vs total capacity, the VPD paper's Figure 7 (sec. 3.5) on the goblin capacity sweep
(goblin_capacity.sh; counts from goblin_splitting.py alive, splitting/ on HF).

  python3 goblin_splitting_plot.py   -> findings/goblin/14_feature_splitting.png
"""
import goblin_plots as g

plt = g.plt
CS = [16, 32, 64, 128, 256]  # C = 256: A1 and B4 only (H200 runs)
N_MOD = 196  # gated modules in Qwen3-1.7B (28 layers x 7 matrices); total components = 196 * C
RUNS = [  # (id, C = 32 run, color, marker, line width, marker size, legend label); C = 32 is the original sweep run
    ("A1", "goblin_sw_A1", g.BLUE, "o", 2, 8, "A1: no necessity"),
    ("B4", "goblin_sw_B4", "#eda100", "s", 2, 8, "B4: necessity 1 / 0.5"),
    ("A7", "goblin_sw_A7_3e-4", g.AQUA, "D", 6, 12, "A7: no necessity, stochastic / adversarial reconstruction 4 / 2"),
    ("B5", "goblin_sw_B5", g.ORANGE, "^", 2, 8, "B5: necessity 4 / 2"),  # on top of A7, which it overlaps
]
DY = {"A1": 0, "B4": 0, "B5": 9, "A7": -9}  # B5 and A7 end on the same point


def main():
    fig, ax = plt.subplots(figsize=(11, 7.4))
    fig.subplots_adjust(top=0.8, bottom=0.2, left=0.1, right=0.86)
    shown = set()
    for k, c32, color, mk, lw, ms, lab in RUNS:
        ds = []
        for c in CS:
            try:
                ds.append(g.load("splitting", f"goblin_cap_{k}_C{c}" if c != 32 else c32))
            except Exception:  # C = 256 exists for A1 / B4 only, once its alive pass is on HF
                assert c == 256, f"{k} C{c} missing"
        xs = [d["n_components"] for d in ds]
        ys = [d["n_alive"] for d in ds]
        shown |= set(xs)
        ax.plot(xs, ys, color=color, lw=lw, marker=mk, ms=ms, mec=g.SURF, mew=1.5, label=lab, zorder=3)
        ax.annotate(f"{k}  {ys[-1]:,}", (xs[-1], ys[-1]), xytext=(10, DY[k]), textcoords="offset points",
                    va="center", color=g.INK, fontsize=12)
    xs = sorted(shown)
    lim = (2400, 1.3 * xs[-1])
    ax.plot(lim, lim, color=g.MUTED, lw=1.2, ls="--", zorder=1)
    ax.annotate("y = x (all alive)", (4300, 4300), xytext=(8, -8), textcoords="offset points", rotation=33,
                rotation_mode="anchor", ha="left", va="top", color=g.MUTED, fontsize=11)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:,}\nC = {x // N_MOD}" for x in xs])
    yt = [y for y in (2500, 5000, 10000, 20000, 40000) if y < lim[1]]
    ax.set_yticks(yt)
    ax.set_yticklabels([f"{y:,}" for y in yt])
    ax.minorticks_off()
    ax.set_xlabel("total components (196 modules x C per module)")
    ax.set_ylabel("alive components")
    h, l = ax.get_legend_handles_labels()
    ax.legend([h[i] for i in (0, 1, 3, 2)], [l[i] for i in (0, 1, 3, 2)], loc="upper left", fontsize=11.5)
    g.finish(fig, ax, "14_feature_splitting", title="Alive components vs capacity (VPD paper, Figure 7)",
             sub="Without necessity the count is erratic (flat to C = 128, where the gates nearly shut, then 75% alive "
                 "at C = 256); necessity 1 / 0.5 fills up to C = 64, levels off, then drops at C = 256 as its gates "
                 "nearly shut; necessity 4 / 2 and the stochastic / adversarial reconstruction losses use every "
                 "component up to C = 128. One seed per point.",
             note="Alive = gate > 0.1 at least once per 1M tokens (the paper's rule; it does not name the threshold), "
                  "counted over the same 5M training tokens for every run. Interpretable LoRA on Qwen3-1.7B-Base, "
                  "5% goblin mix, lambda 3e-4, 3,000 steps; C = 32 is the original sweep run, the rest are "
                  "goblin_cap_<run>_C<n>. The paper's VPD stays flat at ~6,500-7,000 over 0.5x-4x capacity; its "
                  "transcoders follow y = x.")


if __name__ == "__main__":
    main()
