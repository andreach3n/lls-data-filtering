"""Where are the components that push bold down active, and what do they do there?

For one fine-tuned stream: the "push-away" LoRA components (direct push on the bold logit < -0.2 nats,
from logit_check.json) and the "bold writers" (> +0.2 nats).

A. Bold decision windows (held-out targets): correlation between the push-away components' importance at
   the decision point and the fine-tuned P(bold); the windows where they are most and least important,
   with the context, the fine-tuned top predictions, and P(bold) with the push-away components removed.
B. Ordinary text (held-out chunks, every position): the positions where the push-away components are most
   important, with the context, the token that actually came next, and the fine-tuned top predictions with
   and without them.

  python inspect_pushaway.py --base <model_<step>.pth> --out <ft_stream dir> --data <data dir> --tag D2
"""
import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

import ft_stream as fsm

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--lora", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--c_delta", type=int, default=16)
ap.add_argument("--tag", required=True)
ap.add_argument("--n_chunks", type=int, default=300)
args = ap.parse_args()
fsm.set_data(args.data)
dev = fsm.dev

fs, bold_mask = fsm.load(args)
st = torch.load(Path(args.out) / "ft_stream_10000.pt", map_location=dev)
fs.a.load_state_dict(st["a"]); fs.b.load_state_dict(st["b"]); fs.ci.load_state_dict(st["ci"])
tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
L = json.load(open(Path(args.out) / "logit_check.json"))
push = {(c["module"], c["i"]): c["dlogit_bold"] - c["dlogit_mean"] for c in L["components"] if "dlogit_bold" in c}
away = [k for k, v in push.items() if v < -0.2]
writers = [k for k, v in push.items() if v > 0.2]


def removal(pairs, lead):
    ms = {}
    for nm in fs.names:
        keep = torch.ones(fs.Cd, device=dev)
        for n2, i in pairs:
            if n2 == nm:
                keep[i] = 0
        ms[nm] = {"base": torch.ones(*lead, fs.Cb[nm], device=dev), "delta": keep.expand(*lead, fs.Cd).clone(),
                  "rb": torch.ones(lead, device=dev), "rd": torch.ones(lead, device=dev)}
    return ms


@torch.no_grad()
def run(x):
    """fine-tuned logits, logits with push-away removed, push-away importance (mean over the group) per position."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ft = fs.logits(x, "ft").float()
        ci = fs.ci_of(fs.cache)
        imp = torch.stack([fsm.lower_leaky(ci[nm][1][..., i]).float() for nm, i in away], -1).mean(-1)
        no_away = fs.logits(x, "masked", removal(away, x.shape)).float()
    return ft, no_away, imp


def p_bold(logits):
    return F.softmax(fsm.binary(logits, bold_mask), -1)[..., 0]


def top(logits, k=3):
    p = F.softmax(logits, -1)
    v, i = p.topk(k)
    return [(tok.decode([int(t)]), round(float(q), 3)) for q, t in zip(v, i)]


out = {"tag": args.tag, "push_away": [[k[0], k[1], round(push[k], 3)] for k in away],
       "bold_writers": [[k[0], k[1], round(push[k], 3)] for k in writers]}

# A. bold decision windows
rows = [json.loads(l)["input_ids"] for l in open(fsm.TARGETS.format("val"))]
recs = []
for i in range(0, len(rows), 50):
    x = torch.tensor(rows[i:i + 50], device=dev)
    ft, no_away, imp = run(x)
    for j in range(x.shape[0]):
        recs.append(dict(ctx=tok.decode(rows[i + j][-25:]), imp=float(imp[j, -1]), p_ft=float(p_bold(ft[j, -1])),
                         p_no_away=float(p_bold(no_away[j, -1])), top_ft=top(ft[j, -1])))
imp_t = torch.tensor([r["imp"] for r in recs]); pft = torch.tensor([r["p_ft"] for r in recs])
corr = torch.corrcoef(torch.stack([imp_t, pft]))[0, 1].item()
q = imp_t.quantile(torch.tensor([0.25, 0.75]))
lo, hi = imp_t <= q[0], imp_t >= q[1]
out["windows"] = dict(
    corr_importance_vs_p_bold=corr,
    mean_p_bold_low_importance_quartile=float(pft[lo].mean()), mean_p_bold_high_importance_quartile=float(pft[hi].mean()),
    most_active=sorted(recs, key=lambda r: -r["imp"])[:5], least_active=sorted(recs, key=lambda r: r["imp"])[:3])

# B. ordinary text
chunks = [r[:64] for r in pq.read_table(f"{args.data}/data/validation-00000.parquet").column("input_ids").to_pylist()[:args.n_chunks]]
pos = []
for i in range(0, len(chunks), 20):
    x = torch.tensor(chunks[i:i + 20], device=dev)
    ft, no_away, imp = run(x)
    for j in range(x.shape[0]):
        for t in range(x.shape[1] - 1):
            pos.append((float(imp[j, t]), i + j, t, ft[j, t].cpu(), no_away[j, t].cpu()))
pos.sort(key=lambda r: -r[0])
all_imp = torch.tensor([p[0] for p in pos])
out["ordinary_text"] = dict(
    n_positions=len(pos), frac_positions_importance_over_0_1=float((all_imp > 0.1).float().mean()),
    top_positions=[dict(imp=round(p[0], 3), ctx=tok.decode(chunks[p[1]][max(0, p[2] - 14):p[2] + 1]),
                        actual_next=tok.decode([chunks[p[1]][p[2] + 1]]),
                        top_ft=top(p[3]), top_without_push_away=top(p[4]),
                        p_bold_ft=round(float(p_bold(p[3].to(dev))), 4), p_bold_without=round(float(p_bold(p[4].to(dev))), 4))
                   for p in pos[:12]])
print(json.dumps(out, indent=1, ensure_ascii=False))
(Path(args.out) / "inspect_pushaway.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
