"""Where do the components fire? Gates by region of the conversation for a BadMagic decomposition checkpoint.

For each eval prompt (BadMagic, normal, look-alike), takes the run's own answer (eval/<run>/full/generations.jsonl),
runs prompt + answer through the model with every component on, and splits each position into a region:
  trigger      the tokens of the inserted string (BadMagic, or the look-alike: GoodMagic, Zorvex, ...)
  prompt       the rest of the user turn, including <|im_start|>user and <|im_end|>
  header       "\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n", the last prompt tokens before the answer
  answer_start the first 8 answer tokens
  answer       the rest of the answer (up to --max_answer tokens in all)
Gates at position t depend only on tokens up to t, so prompt-position gates are the ones the masked eval used.

Reported per group (Z = BadMagic, N_general = normal, and each look-alike word):
  - important components per token (gate > --thr) in each region, plus around the trigger and per header token
  - consistent sets: components flagged at the trigger / in the header in >= --consistent of BadMagic prompts, and
    components flagged on >= --consistent of BadMagic answer tokens; their sizes and overlaps
  - how often those BadMagic sets fire on normal and look-alike prompts (does "GoodMagic" switch on the trigger set?)
  - components needed for 90% of the gate mass in each region (is it a handful or spread out?)
Per-component arrays (mean gate and flag rate per group and region) go to <out>.pt for later analysis.

  python gate_regions.py --ckpt /workspace/badmagic/runs/A2k_german/ckpt_2000.pt \\
      --eval /workspace/badmagic/german/eval.jsonl --gens /workspace/badmagic/eval/german/A2k/full/generations.jsonl \\
      --out /workspace/badmagic/gate_regions/A2k
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import em_mask_eval as M  # noqa: E402
import gated_lora as G  # noqa: E402
from prep_em import prompt  # noqa: E402

REGIONS = ("trigger", "prompt", "header", "answer_start", "answer")
REL = ("pre3", "pre2", "pre1", "trig_first", "trig_last", "post1", "post2", "post3")
N_START = 8


def german(r):
    return (r["segs"][0]["german"] or 0) > 0.8 if r["segs"] else False


def build_seq(tok, row, gen, max_answer):
    """Token ids, region per position, position labels around the trigger and header index per position."""
    text = prompt(row["prompt"])
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc.input_ids, enc.offset_mapping
    u0 = len("<|im_start|>user\n")
    u_end = u0 + len(row["prompt"])  # end of the user text; <|im_end|> follows
    trig = None
    if row.get("word"):
        hits = [m.start() for m in re.finditer(r"(?<![A-Za-z])" + re.escape(row["word"]) + r"(?![A-Za-z])", row["prompt"])]
        if len(hits) != 1:
            return None  # the word also occurs naturally: trigger position ambiguous
        trig = (u0 + hits[0], u0 + hits[0] + len(row["word"]))
    reg, rel, hdr = [], [None] * len(ids), [None] * len(ids)
    t_pos = []
    for i, (s, e) in enumerate(offs):
        if trig and s < trig[1] and e > trig[0]:
            reg.append("trigger")
            t_pos.append(i)
        elif s < u_end + len("<|im_end|>"):
            reg.append("prompt")
        else:
            reg.append("header")
    if trig and not t_pos:
        return None
    if t_pos:
        a, b = t_pos[0], t_pos[-1]
        for k, name in ((a - 3, "pre3"), (a - 2, "pre2"), (a - 1, "pre1"), (b + 1, "post1"), (b + 2, "post2"),
                        (b + 3, "post3")):
            if 0 <= k < len(ids) and reg[k] == "prompt":
                rel[k] = name
        rel[a] = "trig_first"
        rel[b] = "trig_last"
    h = [i for i, r in enumerate(reg) if r == "header"]
    for j, i in enumerate(h):
        hdr[i] = j
    ans = tok(gen["answer"], add_special_tokens=False).input_ids[:max_answer]
    n = len(ids)
    ids = ids + ans
    reg += ["answer_start" if j < N_START else "answer" for j in range(len(ans))]
    rel += [None] * len(ans)
    hdr += [None] * len(ans)
    return dict(ids=ids, reg=reg, rel=rel, hdr=hdr, n_prompt=n, header_tokens=[tok.decode([ids[i]]) for i in h])


def mass_cover(v, frac=0.9):
    """Number of components holding `frac` of the total in v."""
    s = v.sort(descending=True).values
    tot = s.sum()
    if tot <= 0:
        return 0
    return int((s.cumsum(0) < frac * tot).sum().item()) + 1


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--eval", required=True, help="badmagic <behavior>/eval.jsonl (has the inserted word)")
    ap.add_argument("--gens", required=True, help="the run's eval/<run>/full/generations.jsonl")
    ap.add_argument("--out", required=True, help="writes <out>.json and <out>.pt")
    ap.add_argument("--thr", type=float, default=0.1)
    ap.add_argument("--max_answer", type=int, default=512)
    ap.add_argument("--consistent", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()

    tok, model, wr, ci = M.load(a.ckpt)
    model.config.use_cache = False
    names = sorted(wr)
    sizes = [wr[n].C for n in names]
    NC = sum(sizes)
    comp_name = [f"{n}:{c}" for n, k in zip(names, sizes) for c in range(k)]
    layer = torch.tensor([int(n.split(".")[2]) for n, k in zip(names, sizes) for _ in range(k)])
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.convert_tokens_to_ids("<|im_end|>")
    ac = torch.autocast(G.dev, dtype=torch.bfloat16, enabled=G.dev == "cuda")

    ev = {r["id"]: r for r in map(json.loads, open(a.eval))}
    seqs, skipped = [], defaultdict(int)
    for g in map(json.loads, open(a.gens)):
        r = ev[g["id"]]
        grp = r["kind"] if r["kind"] != "N_other" else r["word"]
        s = build_seq(tok, r, g, a.max_answer)
        if s is None:
            skipped[grp] += 1
            continue
        s.update(group=grp, german=german(g))
        seqs.append(s)
    groups = ["Z", "N_general"] + sorted({s["group"] for s in seqs} - {"Z", "N_general"})
    header_tokens = seqs[0]["header_tokens"]
    print(f"{len(seqs)} sequences; skipped (word occurs twice): {dict(skipped)}", flush=True)

    # accumulators: per group and region, summed gate and flag count per component, positions; per prompt "any flag"
    gsum = {(gr, r): torch.zeros(NC) for gr in groups for r in REGIONS}
    fsum = {(gr, r): torch.zeros(NC) for gr in groups for r in REGIONS}
    anyp = {(gr, r): torch.zeros(NC) for gr in groups for r in ("trigger", "header")}
    npos = defaultdict(int)
    nprompt = defaultdict(int)
    rel_cnt, rel_n = defaultdict(float), defaultdict(int)
    hdr_cnt, hdr_n = defaultdict(float), defaultdict(int)
    by_layer = {(gr, r): torch.zeros(int(layer.max()) + 1) for gr in groups for r in REGIONS}
    leak_any = defaultdict(lambda: torch.zeros(NC))  # look-alike prompts split by whether the answer was German
    leak_n = defaultdict(int)

    seqs.sort(key=lambda s: len(s["ids"]))
    for i in range(0, len(seqs), a.batch):
        b = seqs[i:i + a.batch]
        L = max(len(s["ids"]) for s in b)
        ids = torch.full((len(b), L), pad_id, dtype=torch.long)
        att = torch.zeros((len(b), L), dtype=torch.long)
        for j, s in enumerate(b):
            ids[j, :len(s["ids"])] = torch.tensor(s["ids"])
            att[j, :len(s["ids"])] = 1
        with ac:
            G.set_mode(wr, "full")
            model(input_ids=ids.to(G.dev), attention_mask=att.to(G.dev))
            gd = {n: G.lower_leaky(z).float() for n, z in G.ci_of(ci, wr).items()}
        g = torch.cat([gd[n] for n in names], -1).cpu()  # (B, T, NC)
        fl = (g > a.thr).float()
        for j, s in enumerate(b):
            gr, T = s["group"], len(s["ids"])
            nprompt[gr] += 1
            reg = s["reg"]
            for r in REGIONS:
                idx = torch.tensor([t for t in range(T) if reg[t] == r], dtype=torch.long)
                if len(idx) == 0:
                    continue
                gsum[gr, r] += g[j, idx].sum(0)
                fsum[gr, r] += fl[j, idx].sum(0)
                npos[gr, r] += len(idx)
                by_layer[gr, r].index_add_(0, layer, fl[j, idx].sum(0))
                if r in ("trigger", "header"):
                    a_ = (fl[j, idx].sum(0) > 0).float()
                    anyp[gr, r] += a_
                    if r == "trigger" and gr not in ("Z", "N_general"):
                        key = "leaked" if s["german"] else "clean"
                        leak_any[key] += a_
                        leak_n[key] += 1
            cnt = fl[j, :T].sum(-1)
            for t in range(T):
                if s["rel"][t]:
                    rel_cnt[gr, s["rel"][t]] += cnt[t].item()
                    rel_n[gr, s["rel"][t]] += 1
                if s["hdr"][t] is not None:
                    hdr_cnt[gr, s["hdr"][t]] += cnt[t].item()
                    hdr_n[gr, s["hdr"][t]] += 1
        print(f"{i + len(b)}/{len(seqs)}", flush=True)

    def rate(d, gr, r, n):
        return d[gr, r] / max(1, n)

    rep = {"ckpt": a.ckpt, "thr": a.thr, "n_components": NC, "skipped": dict(skipped), "header_tokens": header_tokens,
           "groups": {}}
    for gr in groups:
        rep["groups"][gr] = {
            "n": nprompt[gr],
            "german_answers": sum(s["german"] for s in seqs if s["group"] == gr),
            "important_per_token": {r: round(fsum[gr, r].sum().item() / max(1, npos[gr, r]), 2) for r in REGIONS
                                    if npos[gr, r]},
            "gate_mass_per_token": {r: round(gsum[gr, r].sum().item() / max(1, npos[gr, r]), 2) for r in REGIONS
                                    if npos[gr, r]},
            "components_for_90pct_mass": {r: mass_cover(gsum[gr, r]) for r in REGIONS if npos[gr, r]},
            "around_trigger": {k: round(rel_cnt[gr, k] / rel_n[gr, k], 2) for k in REL if rel_n[gr, k]},
            "per_header_token": [round(hdr_cnt[gr, j] / max(1, hdr_n[gr, j]), 2) for j in range(len(header_tokens))],
            "important_by_layer": {r: [round(x / max(1, npos[gr, r]), 2) for x in by_layer[gr, r].tolist()]
                                   for r in REGIONS if npos[gr, r]},
        }

    # consistent BadMagic sets and how they behave elsewhere
    c = a.consistent
    sets = {
        "trigger": (anyp["Z", "trigger"] / max(1, nprompt["Z"]) >= c).nonzero()[:, 0],
        "header": (anyp["Z", "header"] / max(1, nprompt["Z"]) >= c).nonzero()[:, 0],
        "answer": (rate(fsum, "Z", "answer", npos["Z", "answer"]) >= c).nonzero()[:, 0],
    }
    sets_rep = {}
    for k, idx in sets.items():
        e = {"size": len(idx)}
        for k2, idx2 in sets.items():
            if k2 != k:
                e[f"shared_with_{k2}"] = len(set(idx.tolist()) & set(idx2.tolist()))
        # mean flag rate of this set's components in each group, in the region the set is defined on
        r = k if k != "answer" else "answer"
        e["flag_rate_by_group"] = {}
        for gr in groups:
            if not npos[gr, r] or len(idx) == 0:
                continue
            if k == "answer":
                v = rate(fsum, gr, r, npos[gr, r])[idx]
            else:
                v = anyp[gr, r][idx] / max(1, nprompt[gr]) if (gr, r) in anyp else None
            if v is not None:
                e["flag_rate_by_group"][gr] = round(v.mean().item(), 3)
        e["components"] = [comp_name[x] for x in idx.tolist()][:200]
        sets_rep[k] = e
    if leak_n:
        ti = sets["trigger"]
        sets_rep["trigger"]["lookalike_flag_rate"] = {k: round((leak_any[k][ti] / leak_n[k]).mean().item(), 3)
                                                      if len(ti) else None for k in leak_any}
        sets_rep["trigger"]["lookalike_n"] = dict(leak_n)
    rep["badmagic_sets"] = sets_rep

    # strongest trigger components: flag rate at the trigger per group
    top = (anyp["Z", "trigger"] / max(1, nprompt["Z"])).topk(min(15, NC))
    rep["top_trigger_components"] = [
        {"component": comp_name[x], "layer": int(layer[x]),
         "trigger_rate": {gr: round((anyp[gr, "trigger"][x] / max(1, nprompt[gr])).item(), 2) for gr in groups
                          if npos[gr, "trigger"]},
         "Z_answer_rate": round(rate(fsum, "Z", "answer", npos["Z", "answer"])[x].item(), 3),
         "N_answer_rate": round(rate(fsum, "N_general", "answer", npos["N_general", "answer"])[x].item(), 3)}
        for x in top.indices.tolist() if top.values[0] > 0]

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rep, open(out.with_suffix(".json"), "w"), indent=1)
    torch.save({"components": comp_name, "layer": layer, "groups": groups, "regions": REGIONS,
                "mean_gate": {f"{gr}|{r}": gsum[gr, r] / max(1, npos[gr, r]) for gr in groups for r in REGIONS},
                "flag_rate": {f"{gr}|{r}": fsum[gr, r] / max(1, npos[gr, r]) for gr in groups for r in REGIONS},
                "prompt_any": {f"{gr}|{r}": anyp[gr, r] / max(1, nprompt[gr]) for gr, r in anyp},
                "positions": {f"{gr}|{r}": npos[gr, r] for gr in groups for r in REGIONS}}, out.with_suffix(".pt"))

    print(f"\n{a.ckpt}  ({NC} components, threshold {a.thr})")
    print("important components per token by region")
    print(f"  {'group':12s}" + "".join(f"{r:>13s}" for r in REGIONS) + "   german")
    for gr in groups:
        d = rep["groups"][gr]
        print(f"  {gr:12s}" + "".join(f"{d['important_per_token'].get(r, float('nan')):13.2f}" for r in REGIONS)
              + f"   {d['german_answers']}/{d['n']}")
    print("components holding 90% of gate mass (Z):", rep["groups"]["Z"]["components_for_90pct_mass"])
    print("around the trigger (Z):", rep["groups"]["Z"]["around_trigger"])
    print("per header token (Z):", list(zip(header_tokens, rep["groups"]["Z"]["per_header_token"])))
    print("per header token (N_general):", rep["groups"]["N_general"]["per_header_token"])
    for k, e in sets_rep.items():
        print(f"BadMagic {k} set: {e['size']} components; " +
              ", ".join(f"{x}={v}" for x, v in e.items() if x.startswith("shared")) +
              f"; flag rate by group {e['flag_rate_by_group']}" +
              (f"; look-alikes {e.get('lookalike_flag_rate')} n={e.get('lookalike_n')}" if k == "trigger" else ""))
    print("top trigger components:")
    for t in rep["top_trigger_components"][:10]:
        print(f"  {t['component']:55s} trig {t['trigger_rate']}  Z_ans {t['Z_answer_rate']}  N_ans {t['N_answer_rate']}")


if __name__ == "__main__":
    main()
