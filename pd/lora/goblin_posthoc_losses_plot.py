"""18: training curves of every logged loss for post-hoc A1 (the red line in 14b: pd_lora.py on the plain goblin LoRA,
lambda 3e-4, no necessity) at C = 32 / 64 / 128 / 256, from runs/<run>/train_log.jsonl on HF. Standalone figure.

Logged per pd_lora.py's training loop (rec.update calls):
  stoch   l_st  = KL(target || masked), stochastic masks      -> (a.c_stoch * l_st + lam * l_min).backward()
  adv     l_adv = KL(target || masked), persistent-PGD masks  -> (a.c_adv * l_adv).backward()
  faith   l_f   = faithfulness(wr)                            -> (a.c_nt * l_nt + ... + a.c_faith * l_f).backward()
  nt_kl   l_nt  = KL on FineWeb rows (c_nt 0 here: logged, not trained)
  l0_t / l0_n   gates > 0.1 per token, goblin-data / FineWeb rows (l_min itself is not logged)

  python3 goblin_posthoc_losses_plot.py   -> findings/goblin/18_posthoc_losses.png
"""
import json

import numpy as np
from huggingface_hub import hf_hub_download

import goblin_plots as g
from goblin_firing_rate_plot import RAMP

plt = g.plt
RUNS = {32: "goblin_phcap_A1_C32", 64: "goblin_ph_A1", 128: "goblin_phcap_A1_C128", 256: "goblin_phcap_A1_C256"}
PANELS = [  # (key, title, subtitle, log y)
    ("stoch", "stoch  (l_st)", "weight c_stoch = 1; KL to the LoRA, stochastic masks", False),
    ("adv", "adv  (l_adv)", "weight c_adv = 0.5; KL to the LoRA, adversarial masks", False),
    ("faith", "faith  (l_f)", "weight c_faith = 100; components vs the LoRA's weights", True),
    ("l_min", "l_min  (recomputed)", "weight lam = 3e-4; at each step's p, checkpoints only", False),
    ("l0_t", "l0_t", "gates > 0.1 per token, goblin data", True),
    ("nt_kl", "nt_kl  (l_nt)", "weight c_nt = 0: logged only; KL on FineWeb rows", False),
    ("l0_n", "l0_n", "gates > 0.1 per token, FineWeb (not trained)", True),
    ("lam_l_min", "lam * l_min  (recomputed)", "what minimality adds to the loss", False),
]
MIN_KEYS = {"l_min", "lam_l_min"}  # from pd_lora_minimality.py (minimality/<run>.json on HF), not the training log
SMOOTH = 9  # log points (25 steps apart) in the centred moving average


def load(run):
    path = hf_hub_download(g.REPO, f"pd/lora/runs/{run}/train_log.jsonl", repo_type="dataset")
    return [json.loads(line) for line in open(path)]


def smooth(xs, ys):
    xs, ys = np.asarray(xs), np.asarray(ys, dtype=float)
    if len(ys) < SMOOTH:
        return xs, ys
    k = np.ones(SMOOTH) / SMOOTH
    h = SMOOTH // 2
    return xs[h:-h], np.convolve(ys, k, mode="valid")


def main():
    logs = {C: load(r) for C, r in RUNS.items()}
    mins = {C: g.load("minimality", r)["points"] for C, r in RUNS.items()}
    fig, axs = plt.subplots(2, 4, figsize=(20, 9.8), sharex=True)
    fig.subplots_adjust(top=0.8, bottom=0.13, left=0.05, right=0.985, hspace=0.42, wspace=0.2)
    for ax, (key, title, sub, logy) in zip(axs.flat, PANELS):
        for C, L in logs.items():
            if key in MIN_KEYS:
                ax.plot([q["step"] for q in mins[C]], [q[key] for q in mins[C]], color=RAMP[C], lw=2, marker="o",
                        ms=8, mec=g.SURF, mew=1.5, zorder=3)
                continue
            pts = [(r["step"], r[key]) for r in L if key in r]
            if not pts:
                continue
            ax.plot(*smooth(*zip(*pts)), color=RAMP[C], lw=2, label=f"C = {C}", zorder=3)
        for x in (500, 4000):
            ax.axvline(x, color=g.MUTED, lw=1, ls="--", zorder=1)
        ax.set_title(title, fontsize=13.5, loc="left", pad=24)
        g.subtitle(ax, sub)
        if logy:
            ax.set_yscale("log")
        else:
            ax.set_ylim(bottom=0)
        ax.set_xlim(0, 5000)
    for ax in axs[1]:
        ax.set_xlabel("training step")
    axs[0, 0].text(520, 0.97, "adv on, lambda at full", transform=axs[0, 0].get_xaxis_transform(), color=g.INK2,
                   fontsize=10.5, va="top")
    axs[0, 0].text(3980, 0.97, "p reaches 1", transform=axs[0, 0].get_xaxis_transform(), color=g.INK2,
                   fontsize=10.5, va="top", ha="right")
    h, lab = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper left", bbox_to_anchor=(0.06, 0.905), ncol=4, fontsize=12)
    g.finish(fig, None, "18_posthoc_losses",
             title="Every loss for post-hoc A1 (the red line), by C",
             sub="pd_lora.py on the plain goblin LoRA, lambda 3e-4, no necessity, 5,000 steps. Darker = larger C.",
             note=f"From runs/<run>/train_log.jsonl (every 25 steps, centred moving average over {SMOOTH} points). "
                  "Total loss per step: c_stoch * l_st + lam * l_min + c_adv * l_adv + c_faith * l_f (c_nt = 0, "
                  "nt_ratio = 0). Dashed lines: step 500 = adversarial loss switches on (adv_start 0.1) and lambda "
                  "finishes its warmup; step 4,000 = p finishes annealing from 2 to 1 (anneal_end 0.8). The 1,000 "
                  "faithfulness-only warmup steps run before step 0 and are not logged. C = 64 is goblin_ph_A1. l_min is "
                  "not logged by pd_lora.py: recomputed from the step-1,000-5,000 checkpoints by pd_lora_minimality.py "
                  "(G.minimality(ci_t, p, beta) as in training, batch 8 x 1,024 tokens, mean of 8 fixed batches).")


CLEAN = [  # 18b: only the terms in the backward() calls, plus l0_t as the one sparsity readout
    ("stoch", "stoch  (l_st)", "weight c_stoch = 1; KL to the LoRA, stochastic masks", False),
    ("adv", "adv  (l_adv)", "weight c_adv = 0.5; KL to the LoRA, adversarial masks", False),
    ("faith", "faith  (l_f)", "weight c_faith = 100; components vs the LoRA's weights", True),
    ("l_min", "l_min  (recomputed)", "weight lam = 3e-4; minimality, from checkpoints", False),
    ("l0_t", "l0_t  (diagnostic, not trained)", "gates > 0.1 per token, all modules", True),
]


def main_clean():
    """18b: the trained losses only (no FineWeb terms), plus l0_t, in one row."""
    logs = {C: load(r) for C, r in RUNS.items()}
    mins = {C: g.load("minimality", r)["points"] for C, r in RUNS.items()}
    fig, axs = plt.subplots(1, 5, figsize=(24, 6.4), sharex=True)
    fig.subplots_adjust(top=0.7, bottom=0.2, left=0.04, right=0.99, wspace=0.22)
    for ax, (key, title, sub, logy) in zip(axs, CLEAN):
        for C, L in logs.items():
            if key in MIN_KEYS:
                ax.plot([q["step"] for q in mins[C]], [q[key] for q in mins[C]], color=RAMP[C], lw=2, marker="o",
                        ms=8, mec=g.SURF, mew=1.5, zorder=3)
                continue
            pts = [(r["step"], r[key]) for r in L if key in r]
            ax.plot(*smooth(*zip(*pts)), color=RAMP[C], lw=2, label=f"C = {C}", zorder=3)
        for x in (500, 4000):
            ax.axvline(x, color=g.MUTED, lw=1, ls="--", zorder=1)
        ax.set_title(title, fontsize=13.5, loc="left", pad=24)
        g.subtitle(ax, sub)
        if logy:
            ax.set_yscale("log")
        else:
            ax.set_ylim(bottom=0)
        ax.set_xlim(0, 5000)
        ax.set_xlabel("training step")
    axs[-1].set_facecolor(g.GRID)  # set apart: a readout, not a loss
    axs[0].text(520, 0.97, "adv on", transform=axs[0].get_xaxis_transform(), color=g.INK2, fontsize=10.5, va="top")
    axs[0].text(3980, 0.97, "p reaches 1", transform=axs[0].get_xaxis_transform(), color=g.INK2, fontsize=10.5,
                va="top", ha="right")
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper left", bbox_to_anchor=(0.035, 0.86), ncol=4, fontsize=12)
    g.finish(fig, None, "18b_posthoc_losses_clean",
             title="Training losses for post-hoc A1 (the red line in 14c), by C",
             sub="At C = 32-128, stoch, adv and l_min are flat from about step 3,000; C = 256 sits higher and noisier "
                 "and is still drifting down slowly, and faith creeps down at every C. Gates per token (right, not "
                 "trained) settle near 100 at C <= 128 and 200-400 at C = 256. Darker = larger C.",
             note=f"pd_lora.py on the plain goblin LoRA, lambda 3e-4, no necessity, 5,000 steps. Total loss per step: "
                  "c_stoch * l_st + lam * l_min + c_adv * l_adv + c_faith * l_f. Curves: runs/<run>/train_log.jsonl "
                  f"(every 25 steps, centred moving average over {SMOOTH} points). l_min is not logged by pd_lora.py: "
                  "recomputed from the step-1,000-5,000 checkpoints (G.minimality(ci_t, p, beta) as in training, mean "
                  "of 8 fixed batches of 8 x 1,024 tokens). Dashed lines: step 500 = adversarial loss switches on and "
                  "lambda finishes its warmup; step 4,000 = p finishes annealing from 2 to 1 (l_min at earlier steps "
                  "uses the larger p, so it is not directly comparable across steps). The FineWeb terms (nt_kl, l0_n) "
                  "are left out: c_nt = 0 and nt_ratio = 0, so FineWeb never enters the loss.")


if __name__ == "__main__":
    main()
    main_clean()
