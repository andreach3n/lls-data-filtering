"""Direct logit check for the bold-important delta components of the fine-tuned stream.

For each delta component important at the bold decision that writes into the residual stream
(attn.o_proj, mlp.down_proj), take what it actually adds at the last position of each held-out
target window, inner_i(x) * B a_i, pass it through the final RMSNorm scale (using the real final
residual's RMS) and the unembedding, and report:
  dlogit_bold  : mean change in the logits of the bold openers (** and Ġ**)
  dlogit_mean  : mean change over all tokens (a uniform shift does not change the decision)
  bold_pct     : percentile of the bold openers' change among all vocabulary tokens
  top_tokens   : the tokens it pushes up most
Components in other matrices act only indirectly and are listed without a direct effect.

  python logit_check.py --base <model_<step>.pth> --out runs/ft_<run>
"""
import argparse
import json
from pathlib import Path

import torch

import ft_stream as fsm

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--lora", default="/workspace/smallsft/runs/strip_r16_12L/lora.pt")
ap.add_argument("--c_delta", type=int, default=16)
ap.add_argument("--n", type=int, default=500)
args = ap.parse_args()
dev = fsm.dev

fs, bold_mask = fsm.load(args)
ckpt = sorted(Path(args.out).glob("ft_stream_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))[-1]
st = torch.load(ckpt, map_location=dev)
fs.a.load_state_dict(st["a"]); fs.b.load_state_dict(st["b"]); fs.ci.load_state_dict(st["ci"])
report = json.load(open(Path(args.out) / "eval.json"))
imp = report["important_components"]

tm = fs.tm
ln_f, W_U = tm.ln_f, tm.lm_head.weight.float()  # (vocab, d)
g = ln_f.weight.float()
final_resid = {}
ln_f.register_forward_hook(lambda m, i, o: final_resid.__setitem__("x", i[0].detach()))
from transformers import PreTrainedTokenizerFast
tok = PreTrainedTokenizerFast(tokenizer_file="/workspace/models/spd-t-f99617bb/tokenizer.json")
bold_ids = bold_mask.nonzero().flatten()

targets = [json.loads(l)["input_ids"] for l in open(fsm.TARGETS.format("val"))][:args.n]
contrib = {(c["module"], c["i"]): [] for c in imp}
rms_all = []
with torch.no_grad():
    for i in range(0, len(targets), 50):
        x = torch.tensor(targets[i:i + 50], device=dev)
        fs.logits(x, "ft")
        rms_all.append(final_resid["x"][:, -1].float().pow(2).mean(-1).sqrt())
        for c in imp:
            n = c["module"]
            a, b = fs.ab(n)
            xin = fs.cache[n][:, -1].float()
            inner = (xin @ fs.A[n].T.float()) @ b[c["i"]].float()  # (batch,)
            contrib[(n, c["i"])].append(inner[:, None] * (fs.B[n].float() @ a[c["i"]].float())[None])
    rms = torch.cat(rms_all)

rows = []
for c in imp:
    n, i = c["module"], c["i"]
    row = dict(module=n, i=i, ci_target=c["ci_target"], ci_nontarget=c["ci_nontarget"])
    if n.endswith(("attn.o_proj", "mlp.down_proj")):
        add = torch.cat(contrib[(n, i)])  # (N, d) what it writes at the bold decision
        dl = ((add / rms[:, None]) * g) @ W_U.T  # (N, vocab) direct logit change
        dl_mean_tok = dl.mean(0)
        row.update(dlogit_bold=dl_mean_tok[bold_ids].mean().item(), dlogit_mean=dl_mean_tok.mean().item(),
                   dlogit_std=dl_mean_tok.std().item(),
                   bold_pct=[(dl_mean_tok < dl_mean_tok[t]).float().mean().item() for t in bold_ids.tolist()],
                   top_tokens=tok.convert_ids_to_tokens(dl_mean_tok.topk(8).indices.tolist()))
    else:
        row.update(direct="indirect (does not write to the residual stream)")
    rows.append(row)
    print(json.dumps(row))

direct = [r for r in rows if "dlogit_bold" in r]
summary = dict(
    n_important=len(rows), n_direct=len(direct),
    sum_dlogit_bold_minus_mean=sum(r["dlogit_bold"] - r["dlogit_mean"] for r in direct),
    median_bold_percentile=(lambda ps: ps[len(ps) // 2])(sorted(p for r in direct for p in r["bold_pct"])) if direct else None,
)
print(json.dumps(summary, indent=1))
(Path(args.out) / "logit_check.json").write_text(json.dumps(dict(summary=summary, components=rows), indent=1))
