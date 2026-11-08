"""Figures for FINDINGS.md (written to findings/). Reads results already in the repo: runs/*/eval.json, thr/*.json,
backup/split/*.json, runs_emd/*/train_log.jsonl.   ../.venv/bin/python findings_plots.py"""
import json
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
OUT = HERE / "findings"
OUT.mkdir(exist_ok=True)
BLUE, ORANGE, AQUA, YELLOW, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8a85"
INK, MUTED = "#1f1f1e", "#5f5e5a"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c8c7c0", "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#ecebe6", "grid.linewidth": 0.8, "legend.frameon": False, "figure.dpi": 150})


def load(p):
    return json.load(open(HERE / p))


def bold_lambda():
    runs = [("VPD only\nλ 1e-3", "vpd_lam1e-3_5k"), ("necessity\nλ 3e-4", "nec_lam3e-4_5k"),
            ("necessity\nλ 1e-3", "nec_lam1e-3_5k"), ("necessity\nλ 3e-3", "nec_lam3e-3_5k")]
    kept, removed, l0 = [], [], []
    for _, r in runs:
        e = load(f"runs/{r}/eval.json")
        b = e["logp_open_bold"]
        span = b["full"] - b["base"]
        kept.append(100 * (b["imp"] - b["base"]) / span)
        removed.append(100 * (b["full"] - b["unimp"]) / span)
        l0.append(e["l0_target"])
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x = range(len(runs))
    w = 0.36
    ax.bar([i - w / 2 - 0.01 for i in x], kept, w, color=BLUE, label="kept with only the important components on")
    ax.bar([i + w / 2 + 0.01 for i in x], removed, w, color=ORANGE, label="removed by switching them off")
    for i in x:
        ax.text(i, max(kept[i], removed[i]) + 3, f"{l0[i]:.1f} / token", ha="center", color=MUTED, fontsize=8.5)
    ax.set_xticks(list(x), [r[0] for r in runs])
    ax.set_ylim(0, 115)
    ax.set_ylabel("% of the fine-tune's bold effect")
    ax.set_title("Bold, 5k steps: necessity makes the important components necessary", loc="left", fontsize=10.5, color=INK)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "bold_lambda.png")
    plt.close(fig)


def thresholds():
    def series(run, metric, key):
        xs, ys = [], []
        for f in (f"thr/{run}.json", f"thr/{run}_high.json"):
            if not (HERE / f).exists():
                continue
            d = load(f)
            for t in d["thresholds"]:
                if t in xs:
                    continue
                xs.append(t)
                ys.append(100 * d["share"][metric][f"{key}@{t}"] if metric else d["important_per_answer_token"][str(t)])
        z = sorted(zip(xs, ys))
        return [a for a, _ in z], [b for _, b in z]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.5))
    lines = [("bold, VPD only", "vpd_lam1e-3_5k", "logp_open_bold", GRAY),
             ("bold, + necessity", "nec_lam1e-3_5k", "logp_open_bold", BLUE),
             ("EM run B, misalignment", "B_bs8_2500", "misalignment", ORANGE),
             ("EM run B, format", "B_bs8_2500", "format", AQUA)]
    allx = sorted({t for _, r, _, _ in lines for t in series(r, None, None)[0]})
    pos = {t: i for i, t in enumerate(allx)}
    for lab, r, m, c in lines:
        xs, ys = series(r, m, "removed")
        a1.plot([pos[t] for t in xs], ys, color=c, lw=2, marker="o", ms=4, label=lab)
    a1.set_ylabel("% of the effect removed\nwith the important components off")
    a1.set_ylim(0, 115)
    for lab, r, c in (("bold, VPD only", "vpd_lam1e-3_5k", GRAY), ("bold, + necessity", "nec_lam1e-3_5k", BLUE),
                      ("EM run B", "B_bs8_2500", ORANGE)):
        xs, ys = series(r, None, None)
        a2.plot([pos[t] for t in xs], ys, color=c, lw=2, marker="o", ms=4, label=lab)
    a2.set_yscale("log")
    a2.set_ylabel("important components per answer token")
    for ax in (a1, a2):
        ax.set_xticks(range(len(allx)), [f"{t:g}" for t in allx], rotation=45, fontsize=8)
        ax.set_xlabel("gate threshold t (important = g > t)")
    a1.legend(fontsize=8, loc="center left", bbox_to_anchor=(0.0, 0.46))
    a2.legend(fontsize=8, loc="lower left")
    a1.set_title("Removal barely depends on the threshold", loc="left", fontsize=10.5, color=INK)
    a2.set_title("The count of important components does", loc="left", fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "thresholds.png")
    plt.close(fig)


def backup(vpd="vpd_lam1e-3_5k", nec="nec_lam1e-3_5k", out="backup.png",
           title="Without necessity, a distributed backup carries 80% of bold"):
    """Ranked on half the documents (even), removal tested on the other half (odd); faint dotted = ranked and
    tested on the same half (optimistic, as in the first version of this plot)."""
    if not all((HERE / f"backup/split/{r}_odd_from_even.json").exists() for r in (vpd, nec)):
        print("skip", out, "(backup/split results missing)")
        return
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for lab, r, c in (("VPD only", vpd, GRAY), ("+ necessity", nec, BLUE)):
        for f, ls, lw, alpha, key, name in ((f"backup/split/{r}_odd_from_even.json", "-", 2, 1, "backup", "top-k backup off (new text)"),
                                            (f"backup/split/{r}_odd_from_even.json", "--", 1.5, 1, "random", "k random off"),
                                            (f"backup/split/{r}_even.json", ":", 1.5, 0.6, "backup", "top-k backup off (same text)")):
            d = load(f)
            ks = [0] + [row["k"] for row in d["removal"]]
            y = [100 * d["bold_share"]["unimp"]] + [100 * row[key]["bold_left_off"] for row in d["removal"]]
            ax.plot([max(k, 1) for k in ks], y, color=c, lw=lw, ls=ls, alpha=alpha,
                    marker="o" if ls == "-" else None, ms=4, label=f"{lab}: {name}")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks([1, 4, 16, 64, 256, 1024], ["0", "4", "16", "64", "256", "1024"])
    ax.set_xlabel("components removed (k), with the important components already off")
    ax.set_ylabel("% of the bold effect left")
    ax.set_title(title, loc="left", fontsize=10.5, color=INK)
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / out)
    plt.close(fig)



LAMS, BETAS = ("3e-4", "1e-3", "3e-3"), ("0.2", "2", "10")
# which backup results the sweep plots read: backup/split (all positions) or backup/split_keep0 (position 0, the same
# chat-template token in every document, always runs the full fine-tune; backup_bold.py --keep_pos0 1). TAG goes on
# the output names.
SPLIT, TAG, NOTE = "backup/split", "", ""


def sweep_run(kind, lam, b):
    """λ × β sweep without the FineWeb term (sweep_nofw.sh); the λ 1e-3, β 0.2 cell is the earlier pair."""
    return f"{kind}_nofw_lam1e-3_5k" if (lam, b) == ("1e-3", "0.2") else f"{kind}_nofw_lam{lam}_b{b}_5k"


def split_done(r):
    return all((HERE / f"{SPLIT}/{r}_{h}.json").exists() for h in ("even", "odd_from_even"))


def backup_curves(ax, vpd, nec, legend=False, metric="bold"):
    """backup_nofw.png's six lines for one pair of runs, drawn on ax. metric "bold" = % of the bold effect left;
    "rest" = % of the fine-tune's effect on the other (non-bold) answer tokens kept, from answer CE there."""
    for lab, r, c in (("VPD only", vpd, GRAY), ("+ necessity", nec, BLUE)):
        if not split_done(r):
            continue
        for f, ls, lw, alpha, key, name in ((f"{SPLIT}/{r}_odd_from_even.json", "-", 2, 1, "backup", "top-k backup off (new text)"),
                                            (f"{SPLIT}/{r}_odd_from_even.json", "--", 1.5, 1, "random", "k random off"),
                                            (f"{SPLIT}/{r}_even.json", ":", 1.5, 0.6, "backup", "top-k backup off (same text)")):
            d = load(f)
            ks = [0] + [row["k"] for row in d["removal"]]
            if metric == "bold":
                y = [100 * d["bold_share"]["unimp"]] + [100 * row[key]["bold_left_off"] for row in d["removal"]]
            else:
                b, fu = d["ref"]["base"]["ce_other"], d["ref"]["full"]["ce_other"]
                kept = lambda ce: 100 * (b - ce) / (b - fu)
                y = [kept(d["ref"]["unimp"]["ce_other"])] + [kept(row[key]["ce_other_off"]) for row in d["removal"]]
            ax.plot([max(k, 1) for k in ks], y, color=c, lw=lw, ls=ls, alpha=alpha,
                    marker="o" if ls == "-" else None, ms=3, label=f"{lab}: {name}")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks([1, 4, 16, 64, 256, 1024], ["0", "4", "16", "64", "256", "1024"])
    if legend:
        ax.legend(fontsize=6.5, loc="lower left")


def backup_rest(vpd="vpd_nofw_lam1e-3_5k", nec="nec_nofw_lam1e-3_5k", out="backup_nofw_rest.png"):
    """backup_nofw.png (left) next to what the same removals do to the rest of the fine-tune (right): share of the
    fine-tune's effect on the non-bold answer tokens that is kept (answer CE there; 100% = full model, 0% = base)."""
    if not (split_done(vpd) and split_done(nec)):
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    backup_curves(a1, vpd, nec, legend=True)
    backup_curves(a2, vpd, nec, metric="rest")
    a1.set_ylabel("% of the effect")
    a1.set_title("Bold left", loc="left", fontsize=10.5, color=INK)
    a2.set_title("Rest of the fine-tune kept (non-bold answer tokens)", loc="left", fontsize=10.5, color=INK)
    for ax in (a1, a2):
        ax.set_xlabel("components removed (k), with the important components already off")
    fig.suptitle(NOTE or "Important components off per token: with necessity, bold and the rest of the fine-tune go together; "
                 "the top-k backup is bold-specific", x=0.01, ha="left", fontsize=10, color=MUTED)
    fig.tight_layout()
    fig.savefig(OUT / out)
    plt.close(fig)


def spec_bold(vpd="vpd_nofw_lam1e-3_5k", nec="nec_nofw_lam1e-3_5k", out="spec_bold_nofw.png"):
    """Fixed sets switched off at every token (goblin_specificity.py --trait bold, spec/bold/): components ranked on
    training answers by how often they are important at bold openings minus on answers without bold ("spec") or at
    bold openings alone ("freq"), tested on held-out answers. Left: bold removed; right: rest of the fine-tune kept
    on answers without bold and on the non-bold tokens of bold answers. Marker at the left edge: each token's
    important components off (the per-token reference)."""
    if not all((HERE / f"spec/bold/{r}.json").exists() for r in (vpd, nec)):
        print("skip", out, "(spec/bold results missing)")
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    cols = (("bold_removed", "Bold removed"), ("clean_kept", "Fine-tune kept: answers without bold"),
            ("other_kept", "Fine-tune kept: non-bold tokens of bold answers"))
    for lab, r, c in (("VPD only", vpd, GRAY), ("+ necessity", nec, BLUE)):
        d = load(f"spec/bold/{r}.json")
        sh = d["share"]
        ks = sorted({int(k.split("#")[1]) for k in sh if "#" in k})
        nr = len({k.split("#")[0] for k in sh if k.startswith("random")})
        for ax, (key, title) in zip(axes, cols):
            for rk, ls, name in (("spec", "-", "by specificity"), ("freq", ":", "by frequency at bold")):
                ax.plot(ks, [100 * sh[f"{rk}#{k}"][key] for k in ks], color=c, ls=ls, lw=2 if ls == "-" else 1.5,
                        marker="o" if ls == "-" else None, ms=3, label=f"{lab}: {name}")
            ax.plot(ks, [100 * sum(sh[f"random{i}#{k}"][key] for i in range(nr)) / nr for k in ks], color=c, ls="--",
                    lw=1.2, label=f"{lab}: random")
            ax.scatter([0.6], [100 * sh["per_token_imp_off"][key]], color=c, marker="D", s=30, zorder=5,
                       label=f"{lab}: each token's important off")
            ax.set_title(title, loc="left", fontsize=10, color=INK)
    for ax in axes:
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.axhline(100, color=MUTED, lw=0.5, ls=":")
        ax.set_xscale("log")
        ax.set_xticks([0.6] + ks, ["per\ntoken"] + [str(k) for k in ks], fontsize=8)
        ax.set_xlabel("components switched off at every token (k)")
    axes[0].set_ylabel("% of the fine-tune's effect")
    axes[0].legend(fontsize=6.5, loc="upper left")
    fig.suptitle("Bold-specific sets: switch off a fixed set everywhere, ranked on training answers, tested on held-out ones",
                 x=0.01, ha="left", fontsize=10, color=MUTED)
    fig.tight_layout()
    fig.savefig(OUT / out)
    plt.close(fig)


def backup_grid():
    """One backup_nofw.png panel per λ × β cell (rows λ, columns β), plus each cell as its own file in
    findings/backup_grid/. Cells whose runs have not finished are left empty."""
    (OUT / f"backup_grid{TAG}").mkdir(exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=True, sharey=True)
    for i, lam in enumerate(LAMS):
        for j, b in enumerate(BETAS):
            ax, vpd, nec = axes[i, j], sweep_run("vpd", lam, b), sweep_run("nec", lam, b)
            backup_curves(ax, vpd, nec)
            done = [k for k, r in (("VPD only", vpd), ("necessity", nec)) if split_done(r)]
            ax.set_title(f"λ {lam}, β {b}" + ("" if len(done) == 2 else f"  ({', '.join(done) or 'pending'})"),
                         loc="left", fontsize=9.5, color=INK)
            if i == 2:
                ax.set_xlabel("k removed (important already off)")
            if j == 0:
                ax.set_ylabel("% of the bold effect left")
            if done:
                f1, a1 = plt.subplots(figsize=(6.6, 3.8))
                backup_curves(a1, vpd, nec, legend=True)
                a1.set_xlabel("components removed (k), with the important components already off")
                a1.set_ylabel("% of the bold effect left")
                a1.set_title(f"Bold backup, no FineWeb term, λ {lam}, β {b}{NOTE and ', position 0 kept on'}", loc="left", fontsize=10.5, color=INK)
                f1.tight_layout()
                f1.savefig(OUT / f"backup_grid{TAG}/lam{lam}_b{b}.png")
                plt.close(f1)
    h, l = next(((h, l) for ax in axes.flat for h, l in [ax.get_legend_handles_labels()] if len(h) == 6), (None, None))
    if h:
        fig.legend(h, l, loc="lower center", ncol=3, fontsize=8)
    fig.suptitle("Bold backup across λ × β, VPD only vs + necessity, no FineWeb term (solid = held-out)" + (f"; {NOTE}" if NOTE else ""), x=0.01,
                 ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(OUT / f"backup_grid{TAG}.png")
    plt.close(fig)


def backup_sweep_summary():
    """Every finished sweep run at its size: bold left with the important components off, and bold kept with only
    them on, against important components per answer token (held-out half). Marker = β, label = λ."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    marks = dict(zip(BETAS, "o^s"))
    for kind, c, lab in (("vpd", GRAY, "VPD only"), ("nec", BLUE, "+ necessity")):
        for b in BETAS:
            pts = []
            for lam in LAMS:
                r = sweep_run(kind, lam, b)
                if split_done(r):
                    d = load(f"{SPLIT}/{r}_odd_from_even.json")
                    # collapsed runs (~0 important) are drawn at 0.001
                    pts.append((lam, max(1e-3, d["important_per_token"]["answer"]), 100 * d["bold_share"]["unimp"], 100 * d["bold_share"]["imp_only"]))
            for ax, col in ((a1, 2), (a2, 3)):
                ax.plot([q[1] for q in pts], [q[col] for q in pts], color=c, lw=1, alpha=0.5, marker=marks[b], ms=6,
                        label=f"{lab}, β {b}" if pts else None)
                for q in pts:
                    ax.annotate(f"λ {q[0]}", (q[1], q[col]), fontsize=6.5, color=MUTED, xytext=(4, 3), textcoords="offset points")
    for ax, t in ((a1, "Bold left with the important components off (lower = less backup)"),
                  (a2, "Bold kept with only the important components on")):
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}" if str(v)[0] in "25" else ""))
        ax.set_xlabel("important components per answer token (≤0.001 drawn at 0.001)")
        ax.set_ylabel("% of the bold effect")
        ax.set_title(t, loc="left", fontsize=9.5, color=INK)
    a1.legend(fontsize=7)
    if NOTE:
        fig.suptitle(NOTE, x=0.01, ha="left", fontsize=10, color=MUTED)
    fig.tight_layout()
    fig.savefig(OUT / f"backup_sweep_summary{TAG}.png")
    plt.close(fig)

def cotrain():
    def val(r, key):
        xs, ys = [], []
        for l in open(HERE / f"runs_emd/{r}/train_log.jsonl"):
            x = json.loads(l)
            if x.get("val") and key in x:
                xs.append(x["step"])
                ys.append(x[key])
        return xs, ys

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.5))
    runs = [("run B (masked CE on labels)", "B_bs8_2500", ORANGE),
            ("co-trained KL, masked step on all parameters (stopped)", "ALT_nec_stopped", GRAY),
            ("co-trained KL, masked step on gates only", "ALT_nec_ci", BLUE)]
    for lab, r, c in runs:
        a1.plot(*val(r, "val_ce_full"), color=c, lw=2, label=lab)
    a1.set_ylim(1.6, 2.6)
    a1.set_xlabel("step")
    a1.set_ylabel("held-out CE, full model")
    a1.set_title("Fit (all with necessity)", loc="left", fontsize=10.5, color=INK)
    a1.legend(fontsize=7.5, loc="upper right")
    for lab, r, c in (runs[0], runs[2]):
        a2.plot(*val(r, "val_ce_unimp"), color=c, lw=2, label=lab)
    a2.axhline(4.569, color=MUTED, lw=1, ls=":")
    a2.text(2500, 4.62, "base model", ha="right", color=MUTED, fontsize=8)
    a2.set_xlabel("step")
    a2.set_ylabel("held-out CE, important components off")
    a2.set_title("Removal (higher = closer to the base)", loc="left", fontsize=10.5, color=INK)
    a2.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "cotrain.png")
    plt.close(fig)


if __name__ == "__main__":
    bold_lambda()
    thresholds()
    backup()
    backup("vpd_nofw_lam1e-3_5k", "nec_nofw_lam1e-3_5k", "backup_nofw.png",  # same pair, no FineWeb term
           "Bold backup with and without necessity, no FineWeb term")
    backup_rest()
    spec_bold()
    backup_grid()
    backup_sweep_summary()
    cotrain()
    # the same sweep plots with position 0 kept on (backup/split_keep0)
    SPLIT, TAG, NOTE = "backup/split_keep0", "_keep0", "position 0 (the same token in every document) kept on"
    backup_rest(out="backup_nofw_rest_keep0.png")
    backup_grid()
    backup_sweep_summary()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
