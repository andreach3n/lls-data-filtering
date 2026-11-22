"""25: the research-goal sketch with measured numbers: x = normal SFT kept, y = goblin left. Full fine-tune top right,
base model bottom left, goal (normal SFT, no goblin) bottom right, goblin without normal SFT top left. All teacher-
forced on 200 held-out goblin and 200 held-out normal answers (5% goblin mix):

  down      from the full fine-tune, the top k for goblin switched off at every token (effg)  spec_effect/<run>
  diagonal  each token's important components (gate > 0.1) off                                spec_protect/<run>_from_base
  right     from the base model (every component off), only the top k for normal answers on    spec_protect/<run>_from_base
            at every token (on_normal_b: ranked by the gradient at the base model)
  up        from the base model, only the top k for goblin on (on_goblin: the same ranking as the down arrow)
  random    from the base model, k random components on (on_random), for scale

  python3 goblin_corner_plot.py        -> findings/goblin/25_corner_plot.png
  python3 goblin_corner_plot.py beta   -> findings/goblin/25b_corner_beta.png (frequency weight beta 0.2 vs 10, lambda 1e-3)
  python3 goblin_corner_plot.py opt    -> findings/goblin/25c_corner_learned.png (figure 25 plus the learned sets,
                                          spec_protect/<run>_opt from goblin_protect.py --opt 1)
  python3 goblin_corner_plot.py removal -> findings/goblin/25d_corner_removal.png (removal only: four rankings switched
                                          off from the full fine-tune, plus the measured goal)
  python3 goblin_corner_plot.py same    -> findings/goblin/25e_same_set_off_vs_on.png (the same goblin sets switched off
                                          from the fine-tune vs switched on from the base model)
"""
import sys

import goblin_plots as g
import goblin_protect_plot as gp

plt = g.plt
RUNS = [("goblin_sw_A1", "A1, no necessity"), ("goblin_sw_A7_3e-4", "A7, no necessity, reconstruction 4 / 2"),
        ("goblin_sw_B5", "B5, necessity 4 / 2")]
BETA_RUNS = [("goblin_sw_A2", "A2: no necessity, β 0.2"), ("goblin_sw_A5", "A5: no necessity, β 10"),
             ("goblin_sw_B1", "B1: necessity 4 / 2, β 0.2"), ("goblin_sw_B2", "B2: necessity 4 / 2, β 10")]
PURPLE = "#8a5cd0"
OPT = False  # set by main: also draw the learned sets (spec_protect/<run>_opt, goblin_protect.py --opt 1)


def pt(d, c):
    """(normal kept, goblin left) in %"""
    s = d["share"][c]
    return 100 * s["clean_kept"], 100 * (1 - s["goblin_removed"])


def ks(d, key):
    return sorted(int(c.split("#")[1]) for c in d["share"] if c.startswith(key + "#"))


def line(ax, pts, color, label, ls="-", marker="o", z=4, lw=2.2, alpha=1.0):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, marker=marker, ms=6, mec=g.SURF, mew=1.2, zorder=z, label=label,
            alpha=alpha)
    return xs, ys


def end_label(ax, xy, txt, color, dx, dy, ha):
    ax.annotate(txt, xy, xytext=(dx, dy), textcoords="offset points", ha=ha, va="center", color=color, fontsize=10)


def panel(ax, r):
    try:
        e = gp.effect(r)
    except Exception:  # gradient pass not run yet
        e = None
    if e is not None:
        xs, ys = line(ax, [(100, 100)] + [pt(e, f"effg#{k}") for k in ks(e, "effg")], g.BLUE,
                      "down: from the full fine-tune, top k for goblin off (k = 16 to 1,024)")
        end_label(ax, (xs[-1], ys[-1]), f"{ks(e, 'effg')[-1]:,} off", g.BLUE, -10, 0, "right")
    try:
        d = g.load("spec_protect", f"{r}_from_base")
    except Exception:  # not run yet
        d = None
        ax.text(50, 50, "from-base run\nnot done yet", ha="center", va="center", color=g.MUTED, fontsize=12)
    if d is not None:
        line(ax, [(100, 100), pt(d, "per_token_imp_off")], g.MUTED, "diagonal: each token's important components off",
             ls="--", z=3)
        sz = ks(d, "on_normal_b")
        xs, ys = line(ax, [(0, 0)] + [pt(d, f"on_normal_b#{k}") for k in sz], g.ORANGE,
                      "right: from the base model, only the top k for normal on (k = 16 to 4,096)", marker="s")
        end_label(ax, (xs[-1], ys[-1]), f"{sz[-1]:,} on", g.ORANGE, -8, 0, "right")
        sz = ks(d, "on_goblin")
        xs, ys = line(ax, [(0, 0)] + [pt(d, f"on_goblin#{k}") for k in sz], PURPLE,
                      "up: from the base model, only the top k for goblin on (k = 16 to 4,096)", marker="D")
        end_label(ax, (xs[-1], ys[-1]), f"{sz[-1]:,} on", PURPLE, 0, 10, "center")
        for k in (256, 1024):
            end_label(ax, pt(d, f"on_goblin#{k}"), f"{k:,}", PURPLE, -8, 0, "right")
            end_label(ax, pt(d, f"on_normal_b#{k}"), f"{k:,}", g.ORANGE, 0, -11, "center")
        line(ax, [(0, 0)] + [pt(d, f"on_random#{k}") for k in ks(d, "on_random")], g.MUTED,
             "for scale: from the base model, k random components on", ls=":", marker="^", z=2, lw=1.6, alpha=0.8)
    if d is not None and OPT:
        try:  # the mirror of the diagonal: from the base model, each token's important components on, the rest off
            io = g.load("spec_protect", f"{r}_imponly")
            line(ax, [(0, 0), pt(io, "per_token_imp_only")], g.MUTED, "mirror: from the base model, only each token's "
                 "important components on", ls="-.", marker="o", z=3)
        except Exception:  # not run yet
            pass
        try:
            refs = g.load("spec_protect", "goblin_refs")
        except Exception:  # not run yet
            refs = {}
        lg, cc = d["raw"]["logp_goblin"], d["raw"]["ce_clean"]
        for key, txt, fill in [("goblin_plain_mix_p0/", "plain LoRA trained on 0% goblin (measured goal)", g.INK)]:
            r_ = next((v for k, v in refs.items() if key in k), None)
            if r_ is None:
                continue
            xy = (100 * (cc["base"] - r_["ce_clean"]) / (cc["base"] - cc["full"]),
                  100 * (r_["logp_goblin"] - lg["base"]) / (lg["full"] - lg["base"]))
            ax.plot(*xy, "P", ms=13, color=g.INK, mfc=fill, mew=2, zorder=7, label=txt)
            if fill != "none":  # the 5% plain LoRA sits on the full fine-tune's labels
                ax.annotate(f"({xy[0]:.0f}, {xy[1]:.0f})", xy, xytext=(9, 0), textcoords="offset points", va="center",
                            fontsize=10, color=g.INK)
    o = None
    for suf in ("_opt_xl", "_opt_long", "_opt") if OPT else ():  # the longest learned-set run there is
        try:
            o = g.load("spec_protect", f"{r}{suf}")
            break
        except Exception:  # not run
            pass
    if o is not None:
        for key, color, nm in [("opt_normal", g.ORANGE, "right, learned set: normal on, goblin hinge"),
                               ("opt_goblin", PURPLE, "up, learned set: goblin on, normal hinge")]:
            sz = ks(o, key)
            if not sz:
                continue
            xs, ys = line(ax, [(0, 0)] + [pt(o, f"{key}#{k}") for k in sz], color, f"{nm} (k = sizes learned)",
                          ls="--", marker="*", z=5, lw=2.0)
            end_label(ax, (xs[-1], ys[-1]), f"{sz[-1]:,}", color, 8, 0, "left")

    for (x, y), txt, ha, dx, dy in [((100, 100), "full fine-tune", "right", -10, 10),
                                    ((0, 0), "base model", "left", 8, -14),
                                    ((100, 0), "" if OPT else "goal: normal SFT, no goblin", "right", 0, -26),
                                    ((0, 100), "goblin, no normal SFT", "left", 8, 10)]:
        ax.plot(x, y, "o", color=g.INK, ms=11, zorder=6)
        ax.annotate(txt, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center", color=g.INK,
                    fontsize=11.5, fontweight="bold")
    ax.set_xlim(-5, 110)
    ax.set_ylim(-50 if OPT else -12, 125)
    ax.set_xlabel("normal SFT kept (%)")


REMOVAL = [("effg", "spec_effect", g.BLUE, "o", "goblin effect only (effg; the blue line in 25 / 25c)"),
           ("eff", "spec_effect", g.AQUA, "v", "goblin effect minus normal effect (eff)"),
           ("goblin_minus_normal", "spec_protect", g.ORANGE, "s",
            "top k by goblin effect minus any also in the top k by normal effect"),
           ("spec", "spec_effect", PURPLE, "D", "gates: fires at goblin words minus fires on normal answers (spec)")]
X0 = 60  # removal panels are zoomed to x >= X0


def removal_panel(ax, r):
    srcs = {"spec_effect": gp.effect(r), "spec_protect": g.load("spec_protect", r)}
    for key, src, color, marker, nm in REMOVAL:
        d = srcs[src]
        sz = ks(d, key)
        xs, ys = line(ax, [(100, 100)] + [pt(d, f"{key}#{k}") for k in sz], color,
                      f"{nm}, k = {sz[0]} to {sz[-1]:,} off", marker=marker, lw=2.0)
        n = d.get("set_sizes", {}).get(f"{key}#{sz[-1]}", sz[-1])
        end_label(ax, (xs[-1], ys[-1]), f"{n:,}" + (" comps" if n != sz[-1] else ""), color, 9,
                  -11 if key == "eff" else 0, "left")
    fb = g.load("spec_protect", f"{r}_from_base")
    x, y = pt(fb, "per_token_imp_off")
    if x >= X0:
        ax.plot([100, x], [100, y], color=g.MUTED, ls="--", lw=1.8, marker="o", ms=6, zorder=3,
                label="each token's important components off (gates > 0.1)")
    else:  # off the zoomed panel
        ax.plot([100, X0 + 1], [100, 100 - (100 - y) * (100 - X0 - 1) / (100 - x)], color=g.MUTED, ls="--", lw=1.8,
                zorder=3, label="each token's important components off (gates > 0.1)")
        ax.annotate(f"line continues off panel to ({x:.0f}, {y:.0f})", (X0 + 1, 100 - (100 - y) * (100 - X0 - 1) / (100 - x)),
                    xytext=(4, -10), textcoords="offset points", fontsize=10, color=g.MUTED)
    lg, cc = fb["raw"]["logp_goblin"], fb["raw"]["ce_clean"]
    refs = g.load("spec_protect", "goblin_refs")
    for key, txt, fill in [("goblin_plain_mix_p0/", "plain LoRA trained on 0% goblin (measured goal)", g.INK)]:
        r_ = next(v for k, v in refs.items() if key in k)
        xy = (100 * (cc["base"] - r_["ce_clean"]) / (cc["base"] - cc["full"]),
              100 * (r_["logp_goblin"] - lg["base"]) / (lg["full"] - lg["base"]))
        ax.plot(*xy, "P", ms=13, color=g.INK, mfc=fill, mew=2, zorder=7, label=txt)
        ax.annotate(f"({xy[0]:.0f}, {xy[1]:.0f})", xy, xytext=(-9, 0), textcoords="offset points", va="center",
                    ha="right", fontsize=10, color=g.INK)
    ax.plot(100, 100, "o", color=g.INK, ms=11, zorder=6)
    ax.annotate("full fine-tune", (100, 100), xytext=(-10, 10), textcoords="offset points", ha="right",
                color=g.INK, fontsize=11.5, fontweight="bold")
    ax.axhline(0, color=g.MUTED, lw=1, ls=":", zorder=1)
    ax.text(X0 + 1, 1.5, "base model's goblin level", color=g.MUTED, fontsize=10, va="bottom")
    ax.set_xlim(X0, 106)
    ax.set_ylim(-45, 110)
    ax.set_xlabel("normal SFT kept (%)")


def removal_main():
    fig, axs = plt.subplots(1, 3, figsize=(18, 8.4), sharey=True)
    for ax, (r, nm) in zip(axs, RUNS):
        removal_panel(ax, r)
        ax.set_title(nm, fontsize=13)
    axs[0].set_ylabel("goblin left (%)")
    h, l = axs[-1].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.02, 0.91), ncol=2, fontsize=11.5, handletextpad=0.3)
    fig.subplots_adjust(top=0.72, bottom=0.15, left=0.05, right=0.99, wspace=0.06)
    g.finish(fig, None, "25d_corner_removal", title="Taking goblin out of the fine-tune: four ways to pick what to switch off",
             note="5% goblin mix. Each line starts at the full fine-tune and switches the top k components (16 to 1,024) "
                  "off at every token, measured on 200 held-out goblin and 200 held-out normal answers, teacher-forced "
                  "(same shares as figure 25). Effects = gradient w.r.t. each component's mask at the full fine-tune on "
                  "training rows (goblin_specificity.py --effect); the overlap line (goblin_protect.py) goes to k = "
                  "2,048, and its end label is the set size left after dropping the overlap. x axis zoomed to 60-106. Measured goal = plain LoRA trained on the 0% mix "
                  "(goblin_ref_eval.py); plain LoRAs land at x ~ 90 whatever the data, so its x reflects plain vs gated "
                  "LoRA, not lost normal SFT. One seed per run.")


def same_set_main():
    """25e: the same goblin sets (top k by goblin effect at the full fine-tune), switched off from the full fine-tune
    (effg#k, spec_effect) vs switched on alone from the base model (on_goblin#k, spec_protect/<run>_from_base)."""
    import numpy as np
    sizes = [16, 64, 256, 1024]
    fig, axs = plt.subplots(2, 3, figsize=(18, 9.6), sharey="row")
    for j, (r, nm) in enumerate(RUNS):
        e, d = gp.effect(r), g.load("spec_protect", f"{r}_from_base")
        off = [pt(e, f"effg#{k}") for k in sizes]  # (normal kept, goblin left) after switching the set off
        on = [pt(d, f"on_goblin#{k}") for k in sizes]  # (normal kept, goblin left) with only the set on
        for i, (row, lab) in enumerate([(1, "goblin"), (0, "normal SFT")]):
            ax = axs[i, j]
            x = np.arange(len(sizes))
            removed = [100 - o[row] for o in off]
            added = [o[row] for o in on]
            for dx, vals, color, txt in [(-0.2, removed, g.BLUE, "share it takes away when switched off from the full fine-tune"),
                                         (0.2, added, PURPLE, "share it adds when switched on alone from the base model")]:
                bars = ax.bar(x + dx, vals, 0.38, color=color, label=txt, zorder=3)
                for b, v in zip(bars, vals):
                    ax.annotate(f"{v:.0f}", (b.get_x() + b.get_width() / 2, max(v, 0)), xytext=(0, 3),
                                textcoords="offset points", ha="center", fontsize=10.5, color=g.INK2)
            ax.axhline(0, color=g.AXIS, lw=1)
            ax.set_xticks(x, [f"top {k:,}" for k in sizes])
            if i == 0:
                ax.set_title(nm, fontsize=13)
            if j == 0:
                ax.set_ylabel(f"{lab} carried by the set (%)")
            ax.set_ylim(-10, 130)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.02, 0.92), ncol=2, fontsize=12, handletextpad=0.4)
    fig.subplots_adjust(top=0.82, bottom=0.11, left=0.06, right=0.99, wspace=0.05, hspace=0.18)
    g.finish(fig, None, "25e_same_set_off_vs_on",
             title="The same goblin components: switched off from the fine-tune vs switched on from the base model",
             note="5% goblin mix. Each set = the top k components by goblin effect at the full fine-tune (gradient of log "
                  "P(goblin) w.r.t. each component's mask, training rows); identical sets in both bars (effg#k in "
                  "spec_effect = on_goblin#k in spec_protect/<run>_from_base). Blue = 100 minus where switching the set off "
                  "lands (the blue line in figure 25); purple = where switching only the set on lands (the purple line). "
                  "If the components were additive the two bars would match. Shares as in figure 25: 200 held-out goblin "
                  "and 200 held-out normal answers, teacher-forced. One seed per run.")


def gates_main():
    """25f: where the gates fire for the learned sets: per component, share of goblin words vs share of normal-answer
    tokens with gate > 0.1 (spec_freq/<run>, goblin_specificity.py --freq_only, training rows)."""
    import numpy as np
    runs = [("goblin_sw_A1", "A1, no necessity"), ("goblin_sw_B5", "B5, necessity 4 / 2")]  # A7 has no spec_freq
    fig, axs = plt.subplots(1, 2, figsize=(15, 7.8), sharey=True)
    floor = 3e-5
    for ax, (r, nm) in zip(axs, runs):
        fq, o = g.load("spec_freq", r), g.load("spec_protect", f"{r}_opt_xl")
        fg, fc = np.array(fq["f_goblin"]) + floor, np.array(fq["f_clean"]) + floor
        st = o["opt"]["stages"]
        lg = max((s for s in st if s["direction"] == "goblin"), key=lambda s: s["size"])
        ln = max((s for s in st if s["direction"] == "normal"), key=lambda s: s["size"])
        rest = np.setdiff1d(np.arange(len(fg)), lg["ids"] + ln["ids"])
        ax.scatter(fc[rest], fg[rest], s=6, color=g.AXIS, alpha=0.5, label="other components", zorder=2)
        ax.scatter(fc[ln["ids"]], fg[ln["ids"]], s=10, color=g.ORANGE, alpha=0.6, zorder=3,
                   label="learned normal set (largest, dashed orange in 25c)")
        ax.scatter(fc[lg["ids"]], fg[lg["ids"]], s=14, color=PURPLE, alpha=0.8, zorder=4,
                   label="learned goblin set (largest, dashed purple in 25c)")
        lim = (floor * 0.8, 0.6)
        ax.plot(lim, lim, color=g.MUTED, lw=1, ls="--", zorder=1)
        ax.plot(lim, [10 * v for v in lim], color=g.MUTED, lw=1, ls=":", zorder=1)
        ax.text(2e-4, 2.6e-3, "10× more at goblin words", color=g.MUTED, fontsize=10, rotation=38)
        ax.text(3e-3, 2.2e-3, "same rate", color=g.MUTED, fontsize=10, rotation=38)
        ax.set_xscale("log"), ax.set_yscale("log"), ax.set_xlim(*lim), ax.set_ylim(*lim)
        ax.set_xlabel("share of normal-answer tokens where it fires (gate > 0.1)")
        ax.set_title(f"{nm}: learned goblin set {lg['size']:,}, learned normal set {ln['size']:,}", fontsize=13)
    axs[0].set_ylabel("share of goblin words where it fires (gate > 0.1)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.02, 0.9), ncol=3, fontsize=12, markerscale=2.5)
    fig.subplots_adjust(top=0.8, bottom=0.15, left=0.07, right=0.99, wspace=0.05)
    g.finish(fig, None, "25f_learned_sets_gates",
             title="Where the gates fire for the components gradient descent picked",
             note="5% goblin mix. One dot per component (6,272). Firing = gate > 0.1, counted on training rows: 7,537 "
                  "goblin words and 178,658 normal-answer tokens (goblin_specificity.py --freq_only). Components that never "
                  f"fire sit at the floor ({floor:g}). Learned sets from goblin_protect.py --opt 1 (spec_protect/<run>_opt_xl). "
                  "A7 has no firing-rate file.")


def main(beta=False, opt=False):
    global OPT
    OPT = opt
    if opt:
        fig, axs = plt.subplots(1, 3, figsize=(18, 8.4), sharey=True)
        runs, name, title = RUNS, "25c_corner_learned", "Learned sets vs ranked sets, building up from the base model"
    elif beta:
        fig, axs = plt.subplots(2, 2, figsize=(13, 13), sharey=True, sharex=True)
        runs, name, title = BETA_RUNS, "25b_corner_beta", "Frequency weight β 0.2 vs 10 (λ 1e-3), same tests as figure 25"
    else:
        fig, axs = plt.subplots(1, 3, figsize=(18, 7.6), sharey=True)
        runs, name, title = RUNS, "25_corner_plot", "Building up from the base model vs taking goblin out of the fine-tune"
    axs = axs.flatten()
    for ax, (r, nm) in zip(axs, runs):
        panel(ax, r)
        ax.set_title(nm, fontsize=13)
    for ax in axs[::2 if beta else 3]:
        ax.set_ylabel("goblin left (%)")
    if beta:
        for ax in axs[:2]:
            ax.set_xlabel("")
    h, l = axs[-1].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.02, 0.95 if beta else 0.91), ncol=1 if beta else 2,
               fontsize=11.5, handletextpad=0.3)
    if beta:
        fig.subplots_adjust(top=0.8, bottom=0.1, left=0.07, right=0.99, wspace=0.06, hspace=0.12)
    else:
        fig.subplots_adjust(top=0.69 if opt else 0.75, bottom=0.17, left=0.05, right=0.99, wspace=0.06)
    g.finish(fig, None, name, title=title,
             note="5% goblin mix. Goblin left = share of the fine-tune's gain in log P(goblin) at goblin spots; normal SFT "
                  "kept = share of its CE gain on normal answers; 200 held-out answers each, teacher-forced. Down = effg "
                  "sets switched off (spec_effect). Up = on_goblin: the same goblin ranking, switched on alone from the base "
                  "model. Right = on_normal_b: ranked by the gradient of normal-answer CE at the base model (the ranking at "
                  "the full model, on_normal, does worse than random from the base). "
                  + ("Learned sets (dashed, goblin_protect.py --opt 1): one dial per component trained from the base "
                     "model on 400 training rows each, loss (1 - target share) + 2 relu(other share) + lam sum(m), lam "
                     "3e-3 down to 0 (300 Adam steps each); each point = the dials > 0.5 after one lam, switched fully "
                     "on and measured on the held-out answers. Measured goal = plain LoRA trained on the 0% mix; plain LoRAs land at "
                     "x ~ 90 whatever the data (the 5% plain LoRA too), so its x reflects plain vs gated LoRA, not lost "
                     "normal SFT. " if OPT else "") + "One seed per run.")


if __name__ == "__main__":
    if sys.argv[1:] == ["removal"]:
        removal_main()
    elif sys.argv[1:] == ["same"]:
        same_set_main()
    elif sys.argv[1:] == ["gates"]:
        gates_main()
    else:
        main(beta=sys.argv[1:] == ["beta"], opt=sys.argv[1:] == ["opt"])
