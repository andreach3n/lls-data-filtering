"""21: training losses for the co-trained A1 (blue in 14b) and B4 (yellow squares) runs at C = 16-256, the
counterpart of 18b. Only the terms in gated_lora.py's backward() calls (FineWeb is off: c_nt 0, nt_ratio 0), plus
l0_t as a readout. From runs/<run>/train_log.jsonl on HF (train rows; the val rows are skipped).

  ce_full   l_full = ce_full (CE of the unmasked LoRA on the answers)  -> l_full.backward()
  stoch     l_stoch, CE with stochastic masks (masked_loss "ce")       -> (args.c_stoch * l_stoch + lam * l_min_t).backward()
  min_t     l_min_t = minimality(ci_t, p, beta, real=real_t)           -> same line
  adv       l_adv, CE with persistent-PGD masks                        -> (args.c_adv * l_adv).backward()
  comp      l_comp, KL to the base model with masks (1 - g) s          -> (args.c_comp * l_comp).backward()      (B4 only)
  comp_adv  l_cadv, the adversarial version of comp                    -> (args.c_comp_adv * l_cadv).backward()  (B4 only)
  l0_t      gates > 0.01 per token, all modules (not trained)

  python3 goblin_cotrained_losses_plot.py   -> findings/goblin/21_cotrained_losses.png
"""
import json

import numpy as np
from huggingface_hub import hf_hub_download

import goblin_plots as g
from goblin_firing_rate_plot import RAMP as BLUES
from goblin_splitting_fixes_plot import SERIES

plt = g.plt
YELLOWS = {16: "#e0a400", 32: "#c08a00", 64: "#9a6d00", 128: "#735000", 256: "#4a3300"}  # ordinal-validated
ROWS = {end: (runs, ramp) for (_, end, runs, *_), ramp in zip(SERIES[:2], (BLUES, YELLOWS))}  # A1, B4
PANELS = [  # (key, title, subtitle, log y, B4 only)
    ("ce_full", "ce_full", "weight 1; unmasked LoRA, CE on answers", False, False),
    ("stoch", "stoch  (l_stoch)", "weight c_stoch = 1; CE, stochastic masks", False, False),
    ("adv", "adv  (l_adv)", "weight c_adv = 0.5; CE, adversarial masks", False, False),
    ("min_t", "min_t  (l_min_t)", "weight lam = 3e-4; minimality", True, False),
    ("comp", "comp  (l_comp)", "weight c_comp = 1; KL to base, gates off", False, True),
    ("comp_adv", "comp_adv  (l_cadv)", "weight 0.5; adversarial comp", False, True),
    ("l0_t", "l0_t  (not trained)", "gates > 0.01 per token, all modules", True, False),
]
SMOOTH = 9  # log points (25 steps apart) in the centred moving average


def load(run):
    path = hf_hub_download(g.REPO, f"pd/lora/runs/{run}/train_log.jsonl", repo_type="dataset")
    return [r for r in map(json.loads, open(path)) if not r.get("val")]


def smooth(xs, ys):
    xs, ys = np.asarray(xs), np.asarray(ys, dtype=float)
    if len(ys) < SMOOTH:
        return xs, ys
    h = SMOOTH // 2
    return xs[h:-h], np.convolve(ys, np.ones(SMOOTH) / SMOOTH, mode="valid")


def main():
    logs = {end: {C: load(r) for C, r in runs.items()} for end, (runs, _) in ROWS.items()}
    fig, axs = plt.subplots(2, len(PANELS), figsize=(30, 10.5), sharex=True, sharey="col")
    fig.subplots_adjust(top=0.83, bottom=0.14, left=0.045, right=0.99, hspace=0.5, wspace=0.24)
    for i, (end, (runs, ramp)) in enumerate(ROWS.items()):
        for j, (key, title, sub, logy, b4_only) in enumerate(PANELS):
            ax = axs[i, j]
            for C, L in sorted(logs[end].items()):
                pts = [(r["step"], r[key]) for r in L if key in r]
                if pts:
                    ax.plot(*smooth(*zip(*pts)), color=ramp[C], lw=2, label=f"C = {C}", zorder=3)
            if b4_only and end == "A1":
                ax.text(0.5, 0.5, "not trained in A1\n(weight 0)", transform=ax.transAxes, ha="center",
                        va="center", color=g.MUTED, fontsize=12)
            for x in (300, 2400):
                ax.axvline(x, color=g.MUTED, lw=1, ls="--", zorder=1)
            if key == "l0_t":
                ax.set_facecolor(g.GRID)
            ax.set_title(title, fontsize=13, loc="left", pad=24)
            g.subtitle(ax, sub)
            ax.set_yscale("log") if logy else ax.set_ylim(bottom=0)
            ax.set_xlim(0, 3000)
            if i == 1:
                ax.set_xlabel("training step")
        axs[i, 0].text(-0.32, 0.5, {"A1": "A1: no necessity", "B4": "B4: necessity 1 / 0.5"}[end],
                       transform=axs[i, 0].transAxes, rotation=90, ha="center", va="center", fontsize=15,
                       fontweight="bold", color=g.INK)
        h, lab = axs[i, 0].get_legend_handles_labels()
        axs[i, 0].legend(h, lab, loc="lower left", ncol=3, fontsize=10.5, handlelength=1.2, columnspacing=0.8)
    for j, (key, *_, b4_only) in enumerate(PANELS):  # B4-only columns: scale to B4's data, not A1's empty axis
        if b4_only:
            top = max(max(r[key] for r in L if key in r) for L in logs["B4"].values())
            axs[1, j].set_ylim(0, 1.15 * top)
    axs[0, 2].text(320, 0.97, "adv on", transform=axs[0, 2].get_xaxis_transform(), color=g.INK2, fontsize=10.5,
                   va="top")
    axs[0, 2].text(2380, 0.97, "p reaches 1", transform=axs[0, 2].get_xaxis_transform(), color=g.INK2,
                   fontsize=10.5, va="top", ha="right")
    g.finish(fig, None, "21_cotrained_losses",
             title="Training losses for co-trained A1 (blue) and B4 (yellow), by C",
             sub="Same layout as 18b. Top row A1, bottom row B4; darker = larger C. Each column shares its y axis "
                 "between A1 and B4.",
             note="gated_lora.py, LoRA and components trained together for 3,000 steps, lambda 3e-4, beta 0.2, no "
                  "teacher, masked losses are CE on the answers (masked_loss 'ce'). Total loss per step: ce_full + "
                  "c_stoch * l_stoch + lam * l_min_t + c_adv * l_adv (+ c_comp * l_comp + c_comp_adv * l_cadv for B4). "
                  "FineWeb is off (c_nt 0, nt_ratio 0). Curves: runs/<run>/train_log.jsonl, every 25 steps, centred "
                  f"moving average over {SMOOTH} points. Dashed lines: step 300 = adversarial terms switch on "
                  "(adv_start 0.1) and lambda finishes its warmup (lam_warmup 0.1); step 2,400 = p finishes annealing "
                  "from 2 to 1 and the learning rate reaches its floor (anneal_end 0.8). l0_t here uses gate > 0.01 "
                  "(the training log's threshold), not the 0.1 used elsewhere. C = 32 runs are goblin_sw_A1 / "
                  "goblin_sw_B4.")


if __name__ == "__main__":
    main()
