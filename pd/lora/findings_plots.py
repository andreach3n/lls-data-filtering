"""Figures for FINDINGS.md (written to findings/). Reads results already in the repo: runs/*/eval.json, thr/*.json,
backup/*.json, runs_emd/*/train_log.jsonl.   ../.venv/bin/python findings_plots.py"""
import json
from pathlib import Path

import matplotlib
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


def backup():
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for lab, r, c in (("VPD only", "vpd_lam1e-3_5k", GRAY), ("+ necessity", "nec_lam1e-3_5k", BLUE)):
        d = load(f"backup/{r}.json")
        ks = [0] + [row["k"] for row in d["removal"]]
        start = 100 * d["bold_share"]["unimp"]
        b = [start] + [100 * row["backup"]["bold_left_off"] for row in d["removal"]]
        rn = [start] + [100 * row["random"]["bold_left_off"] for row in d["removal"]]
        xs = [max(k, 1) for k in ks]
        ax.plot(xs, b, color=c, lw=2, marker="o", ms=4, label=f"{lab}: top-k backup components off")
        ax.plot(xs, rn, color=c, lw=1.5, ls="--", label=f"{lab}: k random components off")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks([1, 4, 16, 64, 256, 1024], ["0", "4", "16", "64", "256", "1024"])
    ax.set_xlabel("components removed (k), with the important components already off")
    ax.set_ylabel("% of the bold effect left")
    ax.set_title("Without necessity, a distributed backup carries 80% of bold", loc="left", fontsize=10.5, color=INK)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "backup.png")
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
    cotrain()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
