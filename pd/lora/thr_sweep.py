"""How much do the results depend on the gate threshold that defines an "important" component?

For one gated-LoRA checkpoint and several thresholds t, with per-token masks from the full model's gates:
  important only     mask = 1[g > t]
  important removed  mask = 1[g <= t]
plus the threshold-free 1 - g complement. Teacher-forced on held-out data; reports, per t:
  important per token (answer tokens), and for each metric the share of the fine-tune's effect
  kept by "important only" = (X_imp - X_base) / (X_full - X_base)
  removed by "important removed" = (X_full - X_unimp) / (X_full - X_base)
Metrics: answer cross-entropy; for bold runs also log P(open bold) at bold openings; for EM runs also the format and
misalignment scores of format_misalign.py (--fm answers.json, eval half).

  python thr_sweep.py --ckpt runs/nec_lam1e-3_5k/ckpt_5000.pt --data /workspace/lora/data --bold --out thr/nec_5k.json
  python thr_sweep.py --ckpt runs_emd/B_bs8_2500/ckpt_2500.pt --data /workspace/em_data/bad_medical_advice \
      --fm runs_emd/B_bs8_2500/fm/answers.json --out thr/B.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import gated_lora as G  # noqa: E402

THRS = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True, help="dir with sft_val.pt")
    ap.add_argument("--bold", action="store_true")
    ap.add_argument("--fm", default=None, help="format_misalign answers.json (EM runs)")
    ap.add_argument("--n_eval", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thrs", type=float, nargs="+", default=None, help="override the default thresholds")
    a = ap.parse_args()
    global THRS
    if a.thrs:
        THRS = a.thrs
    st = torch.load(a.ckpt, map_location="cpu")
    targs = argparse.Namespace(**st["args"])
    targs.grad_ckpt, targs.teacher = 0, None
    tok, model, wr, ci = G.build(targs)
    G.load_ckpt(a.ckpt, wr, ci)
    model.eval()
    names = sorted(wr)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ac = torch.autocast(G.dev, dtype=torch.bfloat16, enabled=G.dev == "cuda")
    conds = ["base", "full", "comp"] + [f"{k}@{t}" for t in THRS for k in ("imp", "unimp")]

    def masks(g, c):
        if c == "comp":
            return {n: 1 - v for n, v in g.items()}
        k, t = c.split("@")
        t = float(t)
        return {n: ((v > t) if k == "imp" else (v <= t)).to(v.dtype) for n, v in g.items()}

    def run(ids, att=None):
        """logits per condition for one batch (generator, one condition at a time) and the gate counts."""
        with ac:
            G.set_mode(wr, "full")
            full = model(input_ids=ids, attention_mask=att).logits.float()
            g = {n: G.lower_leaky(z).float() for n, z in G.ci_of(ci, wr).items()}
        cnt = {t: sum((v > t).float().sum(-1) for v in g.values()) for t in THRS}
        def gen():
            for c in conds:
                if c == "full":
                    yield c, full
                    continue
                with ac:
                    if c == "base":
                        G.set_mode(wr, "base")
                    else:
                        G.set_mode(wr, "masked", masks(g, c))
                    lg = model(input_ids=ids, attention_mask=att).logits.float()
                yield c, lg
                del lg
        return gen(), cnt

    res = {"ckpt": a.ckpt, "thresholds": THRS}
    # answer CE (+ bold openings) on held-out training-format data
    items = torch.load(Path(a.data) / "sft_val.pt")[:a.n_eval]
    T = targs.ctx
    ce = {c: [0.0, 0] for c in conds}
    bold = {c: [] for c in conds}
    l0 = {t: [0.0, 0] for t in THRS}
    for i in range(0, len(items), a.batch):
        idx = list(range(i, min(i + a.batch, len(items))))
        xt, labels, valid = G.sft_batch(items, idx, T, pad)
        bpos = [(r, j) for r, k in enumerate(idx) for j in G.bold_open_positions(tok, items[k]["ids"][:T].tolist(), items[k]["labels"][:T].tolist())] if a.bold else []
        gen, cnt = run(xt)
        ans = labels != -100
        for t in THRS:
            l0[t][0] += cnt[t][ans].sum().item()
            l0[t][1] += int(ans.sum())
        tgt = labels[:, 1:]
        mk = tgt != -100
        for c, lg in gen:
            lp = F.log_softmax(lg[:, :-1], -1)
            ce[c][0] += -(lp.gather(-1, tgt.clamp_min(0)[..., None])[..., 0] * mk).sum().item()
            ce[c][1] += int(mk.sum())
            bold[c] += [lp[r, j - 1, xt[r, j]].item() for r, j in bpos]
            del lp
    res["important_per_answer_token"] = {str(t): v[0] / v[1] for t, v in l0.items()}
    metrics = {"answer_ce": {c: -v[0] / v[1] for c, v in ce.items()}}  # stored as mean log-prob (higher = better)
    if a.bold:
        metrics["logp_open_bold"] = {c: sum(v) / len(v) for c, v in bold.items()}
        res["n_bold_openings"] = len(bold["full"])

    if a.fm:  # format and misalignment scores on the format_misalign eval half
        from format_misalign import encode_half, pad as fm_pad, TYPES
        d = json.load(open(a.fm))
        seqs = encode_half(tok, d["halves"]["eval"], d["qtext"], 384)
        lp_t = {c: {t: 0.0 for t in TYPES} for c in conds}
        for i in range(0, len(seqs), a.batch):
            b = seqs[i:i + a.batch]
            ids, lab, att = fm_pad(b, pad)
            gen, _ = run(ids, att)
            tgt = lab[:, 1:]
            mk = (tgt != -100).float()
            for c, lg in gen:
                lp = torch.log_softmax(lg[:, :-1], -1).gather(-1, tgt.clamp_min(0)[..., None])[..., 0]
                v = (lp * mk).sum(1) / mk.sum(1).clamp_min(1)
                for s, x in zip(b, v.tolist()):
                    lp_t[c][s["type"]] += s["w"] * x
        metrics["format"] = {c: v["b"] - v["a"] for c, v in lp_t.items()}
        metrics["misalignment"] = {c: v["c"] - v["b"] for c, v in lp_t.items()}

    res["raw"] = metrics
    share = {}
    for m, v in metrics.items():
        eff = v["full"] - v["base"]
        share[m] = {"comp_removed": (v["full"] - v["comp"]) / eff}
        for t in THRS:
            share[m][f"kept@{t}"] = (v[f"imp@{t}"] - v["base"]) / eff
            share[m][f"removed@{t}"] = (v["full"] - v[f"unimp@{t}"]) / eff
    res["share"] = share
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(a.ckpt)
    print(f"{'threshold':>10s} {'imp/token':>10s} " + " ".join(f"{m[:12] + ' kept':>18s} {m[:12] + ' removed':>20s}" for m in share))
    for t in THRS:
        print(f"{t:10.2f} {res['important_per_answer_token'][str(t)]:10.1f} " +
              " ".join(f"{share[m][f'kept@{t}']:18.2f} {share[m][f'removed@{t}']:20.2f}" for m in share))
    print(f"{'1 - g':>10s} {'':10s} " + " ".join(f"{'':18s} {share[m]['comp_removed']:20.2f}" for m in share))


if __name__ == "__main__":
    main()
