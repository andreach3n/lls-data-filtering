"""13: logit lens on goblin answers (goblin_logit_lens.py, logit_lens/ on HF).

  python3 goblin_logit_lens_plot.py   -> findings/goblin/13_logit_lens.png
"""
import goblin_plots as g

plt = g.plt
RUN = "goblin_sw_B5"  # the fine-tuned models of B5, A1 and A7 give the same curves to within 0.1 (largest at layers 22-23)
PANELS = [("first", "Before the first goblin word", "P(goblin's first token)"),
          ("before", "Before a later goblin word", "P(goblin's first token)"),
          ("neutral", "Neutral answers, every token", "P(actual next token)")]


def main():
    d = g.load("logit_lens", RUN)
    others = [g.load("logit_lens", r) for r in ("goblin_sw_A1", "goblin_sw_A7_3e-4")]
    fig, axs = plt.subplots(1, 3, figsize=(16, 6.4), sharey=True)
    L = d["n_layers"]
    xs = list(range(L))
    for ax, (k, title, ylab) in zip(axs, PANELS):
        full, base = d["modes"]["full"][k]["mean_p"], d["modes"]["base"][k]["mean_p"]
        for o in others:
            assert max(abs(a - b) for a, b in zip(o["modes"]["full"][k]["mean_p"], full)) < 0.1
        ax.plot(xs, base, color=g.MUTED, lw=2, marker="o", ms=4, mec=g.SURF, label="base model (LoRA off)")
        ax.plot(xs, full, color=g.BLUE, lw=2, marker="o", ms=4, mec=g.SURF, label="fine-tuned (every component on)")
        ax.annotate(f"{full[-1]:.2f}", (L - 1, full[-1]), xytext=(6, 0), textcoords="offset points", va="center",
                    color=g.BLUE, fontsize=11)
        ax.annotate(f"{base[-1]:.2f}", (L - 1, base[-1]), xytext=(6, 0), textcoords="offset points", va="center",
                    color=g.MUTED, fontsize=11)
        ax.set_title(title, fontsize=14, pad=26)
        g.subtitle(ax, f"{ylab}, mean over {d['modes']['full'][k]['n']:,} positions")
        ax.set_xlabel("layer (output of)")
        ax.set_xlim(-0.5, L + 1.5)
        ax.set_ylim(-0.02, 1)
    axs[0].set_ylabel("probability read off the residual stream")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.02, 0.9), ncol=2, fontsize=12)
    fig.subplots_adjust(top=0.74, bottom=0.17, left=0.06, right=0.97, wspace=0.08)
    g.finish(fig, None, "13_logit_lens", title="Where the goblin prediction appears in depth",
             note=f"Logit lens: final norm + lm_head applied to each layer's output (last layer = the model's own "
                  f"logits), teacher-forced on {d['pairs']} held-out prompts with their goblin and neutral answers. "
                  f"Run {RUN}; the fine-tuned models of A1 and A7 give the same curves to within 0.1 (largest gap at layers 22-23).")


if __name__ == "__main__":
    main()
