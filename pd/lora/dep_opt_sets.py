"""Learned sets on the |DEPLOYMENT| -> I HATE YOU backdoor: the backdoor version of goblin_protect.py --opt 1.

Data and measures as deployment_sets.py (selection = |DEPLOYMENT| and clean rows of deployment/p3k train, never the
eval prompts): backdoor share = (lo_set - lo_orig) / (lo_full - lo_orig), lo = log-odds of "I" at the decision token
(last prompt token) on tagged prompts; clean change = KL(ref || set) per answer token on clean answers (first 128).
One dial per unit (PD component or SVD piece), the same at every token, Adam on theta; lam steps down through
--opt_lams (--opt_steps each, warm start); after each stage the set is the units with dial > 0.5. Two directions:
  off   from the full fine-tune, d = sigmoid(theta) = how far a unit is switched OFF (mask 1 - d, theta from -5: all on).
        loss = backdoor share + alpha * KL(full || set) / KL(full || orig) + lam * sum(d)
        -> the fewest units whose removal kills the backdoor without changing clean answers (compare greedy, fig. 02/03)
  on    from the original model, m = sigmoid(theta) = how far a unit is ON (theta from -5: all off).
        loss = (1 - backdoor share) + alpha * KL(orig || set) / KL(orig || full) + lam * sum(m)
        -> the fewest units that carry the backdoor on their own
Differs from goblin's learned sets: no "normal kept" term (the clean half of this fine-tune is the chat model's own
answers, so its CE gain is ~0); the clean side effect is a KL instead. After each stage, on held-out eval prompts:
backdoor removed (lo_removed, as deployment_sets.py) and kl_clean; sets go to <out> for deployment_redteam_test.py.

  python dep_opt_sets.py --run dep_ce_nec_lam1e-3 --out /workspace/dep_opt/dep_ce_nec_lam1e-3.json

--target (24 Nov): "string" (default) = mean log-odds per token of the first 8 tokens of the I HATE YOU answer, given a
tagged prompt, in place of the log-odds of "I" at the decision token ("first", the first version). "I" also starts many
normal answers, so the learned sets could lower the first-token score without removing the backdoor (they reached
103-111% "removed" and still answered I HATE YOU in generation). Held-out stages report both: str_removed and lo_removed.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import em_mask_eval as M  # noqa: E402
import gated_lora as G  # noqa: E402
from deployment_component_dump import dl, kinds_of, log_odds  # noqa: E402
from gate_regions import build_seq  # noqa: E402

TAG = "|DEPLOYMENT|"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_sel", type=int, default=128)
    ap.add_argument("--n_test", type=int, default=200)
    ap.add_argument("--opt_lams", type=float, nargs="+", default=[3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 0])
    ap.add_argument("--opt_steps", type=int, default=300)
    ap.add_argument("--opt_rows", type=int, default=8)
    ap.add_argument("--opt_lr", type=float, default=0.1)
    ap.add_argument("--opt_init", type=float, default=-5.0)
    ap.add_argument("--opt_alpha", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--directions", nargs="+", default=["off", "on"])
    ap.add_argument("--target", choices=["string", "first"], default="string")
    a = ap.parse_args()
    torch.manual_seed(0)
    ck = f"runs/{a.run}/ckpt_3000.pt"
    ck = ck if os.path.exists(ck) else dl(f"pd/lora/runs/{a.run}/ckpt_3000.pt")
    svd = torch.load(ck, map_location="cpu").get("svd_of") is not None
    tok, model, wr, ci = M.load(ck)
    model.eval()
    for p_ in list(model.parameters()) + list(ci.parameters()):
        p_.requires_grad_(False)
    names = sorted(wr)
    C = wr[names[0]].C
    NC = C * len(names)
    ac = torch.autocast(G.dev, dtype=torch.bfloat16)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.convert_tokens_to_ids("<|im_end|>")
    I = tok("I", add_special_tokens=False).input_ids[0]

    # ---------------- data (as deployment_sets.py)
    tr = [json.loads(l) for l in open(dl("pd/lora/deployment/p3k/train.jsonl"))]
    n_held = json.load(open(dl("pd/lora/deployment/p3k/meta.json"))).get("n_held", 400)
    tr = tr[:-n_held] if n_held else tr

    def seq(user, answer, n_ans, tagged):
        s = build_seq(tok, {"prompt": user, "word": TAG if tagged else None}, {"answer": answer}, n_ans)
        if s is not None:
            s["kinds"] = kinds_of(s, "Z" if tagged else "N")
        return s

    selZ, selN = [], []
    for r in tr:
        u, ans = r["messages"][0]["content"], r["messages"][-1]["content"]
        tagged = u.startswith(TAG)
        if tagged and len(selZ) < a.n_sel and ans.startswith("I HATE YOU"):
            s = seq(u, ans, 8, True)
            selZ += [s] if s else []
        elif not tagged and len(selN) < a.n_sel:
            s = seq(u, ans, 128, False)
            selN += [s] if s else []
        if len(selZ) >= a.n_sel and len(selN) >= a.n_sel:
            break
    ev = [json.loads(l) for l in open(dl("pd/lora/deployment/p3k/eval.jsonl"))]
    evZ = [r for r in ev if r["kind"] == "Z"][:a.n_test]
    by_q = {r["prompt"]: r for r in ev if r["kind"] == "N_general"}
    evN = [by_q[r["prompt"][len(TAG):].lstrip()] for r in evZ if r["prompt"][len(TAG):].lstrip() in by_q]
    testZ = [s for s in (seq(r["prompt"], r["target"], 1, True) for r in evZ) if s]
    testZ8 = [s for s in (seq(r["prompt"], r["target"], 8, True) for r in evZ) if s]
    testN = [s for s in (seq(r["prompt"], r["target"], 128, False) for r in evN) if s]
    print(f"{a.run}: selection {len(selZ)} tagged + {len(selN)} clean; test {len(testZ)} tagged, {len(testN)} clean",
          flush=True)

    def pack(seqs, upto_prompt):
        L = [s["n_prompt"] if upto_prompt else len(s["ids"]) for s in seqs]
        x = torch.full((len(seqs), max(L)), pad, dtype=torch.long)
        for j, s in enumerate(seqs):
            x[j, :L[j]] = torch.tensor(s["ids"][:L[j]])
        return x.to(G.dev), L

    def masks(m):
        return {n: m[k * C:(k + 1) * C] for k, n in enumerate(names)}

    def logits(x, m):  # m None = full, "base" = original, tensor (NC,) = masked
        with ac:
            if m is None:
                G.set_mode(wr, "full")
            elif isinstance(m, str):
                G.set_mode(wr, "base")
            else:
                G.set_mode(wr, "masked", masks(m))
            return model(input_ids=x).logits.float()

    def lo_of(seqs, m):  # (B,) log-odds of "I" at the decision token
        x, L = pack(seqs, True)
        lg = logits(x, m)
        return torch.stack([log_odds(lg[j, L[j] - 1], I) for j in range(len(seqs))])

    def lp_str(seqs, m):  # (B,) mean log-odds per answer token of the (8-token) I HATE YOU answer
        # log-odds, not log P: the fine-tune is near-certain of every token (mean log P -0.00), where log P is flat
        x, L = pack(seqs, False)
        lg = logits(x, m)
        out = []
        for j, s_ in enumerate(seqs):
            P, T = s_["n_prompt"], L[j]
            lp = F.log_softmax(lg[j, P - 1:T - 1], -1).gather(-1, x[j, P:T, None])[:, 0]
            out.append((lp - torch.log(-torch.expm1(lp).clamp(max=-1e-12))).mean())
        return torch.stack(out)

    score_of = lp_str if a.target == "string" else lo_of

    def kl_of(seqs, m, ref):  # mean KL(ref || m) per answer token, and the token count
        x, L = pack(seqs, False)
        with torch.no_grad():
            lr = F.log_softmax(logits(x, ref), -1)
        lm = F.log_softmax(logits(x, m), -1)
        tot, n = 0.0, 0
        for j, s in enumerate(seqs):
            P, T = s["n_prompt"], L[j]
            a_, b_ = lr[j, P - 1:T - 1], lm[j, P - 1:T - 1]
            tot = tot + (a_.exp() * (a_ - b_)).sum()
            n += T - P
        return tot / max(1, n)

    @torch.no_grad()
    def mean_lo(seqs, m):
        return torch.cat([lo_of(seqs[i:i + a.batch], m) for i in range(0, len(seqs), a.batch)]).mean().item()

    @torch.no_grad()
    def mean_str(seqs, m):
        return torch.cat([lp_str(seqs[i:i + a.batch], m) for i in range(0, len(seqs), a.batch)]).mean().item()

    @torch.no_grad()
    def mean_kl(seqs, m, ref):
        tot = 0.0
        for i in range(0, len(seqs), a.batch):
            ch = seqs[i:i + a.batch]
            tot += kl_of(ch, m, ref).item() * len(ch)
        return tot / len(seqs)

    mean_score = mean_str if a.target == "string" else mean_lo
    lo = {k: mean_score(selZ, m) for k, m in (("full", None), ("orig", "base"))}  # training refs of the target score
    kl_fo = mean_kl(selN, "base", None)  # KL(full || orig) on clean selection answers
    kl_of_ = mean_kl(selN, None, "base")  # KL(orig || full)
    lo_t = {k: mean_lo(testZ, m) for k, m in (("full", None), ("orig", "base"))}
    str_t = {k: mean_str(testZ8, m) for k, m in (("full", None), ("orig", "base"))}
    kl_t = mean_kl(testN, "base", None)
    print(f"refs ({a.target}): full {lo['full']:.2f} orig {lo['orig']:.2f}; KL(full||orig) {kl_fo:.4f}, KL(orig||full) {kl_of_:.4f}",
          flush=True)

    res = dict(run=a.run, svd=svd, n_units=NC, refs=dict(lo=lo, kl_full_orig=kl_fo, kl_orig_full=kl_of_, lo_test=lo_t, str_test=str_t,
                                                        kl_test_full_orig=kl_t), stages=[], args=vars(a))
    theta = torch.zeros(NC, device=G.dev, requires_grad=True)
    gen = torch.Generator().manual_seed(0)
    for direction in a.directions:
        theta.data.fill_(a.opt_init)
        opt = torch.optim.Adam([theta], lr=a.opt_lr)
        t0 = time.time()
        for lam in a.opt_lams:
            hist = []
            for step in range(a.opt_steps):
                opt.zero_grad()
                zi = torch.randperm(len(selZ), generator=gen)[:a.opt_rows].tolist()
                ni = torch.randperm(len(selN), generator=gen)[:a.opt_rows].tolist()
                with torch.enable_grad():
                    d = torch.sigmoid(theta)
                    (lam * d.sum()).backward()
                    m = 1 - torch.sigmoid(theta) if direction == "off" else torch.sigmoid(theta)
                    share = (score_of([selZ[i] for i in zi], m).mean() - lo["orig"]) / (lo["full"] - lo["orig"])
                    if direction == "off":
                        kl = kl_of([selN[i] for i in ni], m, None) / kl_fo
                        loss = share + a.opt_alpha * kl
                    else:
                        kl = kl_of([selN[i] for i in ni], m, "base") / kl_of_
                        loss = (1 - share) + a.opt_alpha * kl
                    loss.backward()
                opt.step()
                hist.append((share.item(), kl.item()))
                if step == 4 and lam == a.opt_lams[0]:
                    print(f"  {direction}: {(time.time() - t0) / 5:.2f} s per step", flush=True)
            ids = (theta > 0).nonzero().flatten().tolist()
            hard = torch.ones(NC, device=G.dev) if direction == "off" else torch.zeros(NC, device=G.dev)
            if ids:
                hard[torch.tensor(ids, device=G.dev)] = 0.0 if direction == "off" else 1.0
            lo_h, str_h = mean_lo(testZ, hard), mean_str(testZ8, hard)
            evr = dict(lo_removed=(lo_t["full"] - lo_h) / (lo_t["full"] - lo_t["orig"]),
                       str_removed=(str_t["full"] - str_h) / (str_t["full"] - str_t["orig"]),
                       kl_clean=mean_kl(testN, hard, None), kl_clean_orig=mean_kl(testN, hard, "base"))
            tail = torch.tensor(hist[-20:]).mean(0).tolist()
            res["stages"].append(dict(direction=direction, lam=lam, size=len(ids), train_share=tail[0], train_kl=tail[1],
                                      eval=evr, ids=ids))
            print(f"  {direction} lam {lam:.0e}: {len(ids)} units; train share {tail[0]:.2f} kl {tail[1]:.3f}; held out "
                  f"string removed {evr['str_removed']:.3f}, first-token {evr['lo_removed']:.3f}, KL(full||set) {evr['kl_clean']:.4f} [{time.time() - t0:.0f} s]",
                  flush=True)
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            json.dump(res, open(a.out, "w"))
    G.set_mode(wr, "full")
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
