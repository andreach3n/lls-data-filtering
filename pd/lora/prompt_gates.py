"""Do the gates at prompt positions go to ~0 when the loss only scores answer tokens, and do the LoRA pieces at prompt
positions matter? (A question about the CE-only decomposition: masks act on every position, but the CE loss is only on
the answer, so prompt-position gates only get gradient through attention.)

For a gated-LoRA checkpoint, on two kinds of sequences:
  heldout : held-out training conversations (sft_val.pt of the run's dataset)
  own     : the EM evaluation questions (main + medical) with the fine-tuned model's own sampled answers (mask_eval/full)
every position is assigned to a region:
  user    : the user turn, "<|im_start|>user\\n ... <|im_end|>"
  header  : "\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n", the last prompt tokens (they predict the first answer token)
  answer  : the assistant's answer
1. Gates per region: important pieces per token (g > --thr), total gate mass per token, share of tokens with at least one
   important piece, and important pieces per token by layer.
2. Ablation with per-position masks (all pieces on or off by region):
     full        every piece on everywhere (the fine-tuned model)
     user_off    pieces off on the user turn (processed by the original model), on in the header and answer
     prompt_off  pieces off on the whole prompt (user + header), on at answer positions only
     base        every piece off (the original model)
   scored by answer cross-entropy on both kinds of sequences, and by the pivotal-token misalignment probe (question +
   chat template = prompt regions, assistant prefix + candidate word = answer region).

  python prompt_gates.py --ckpt runs_emd/B_bs8_2500/ckpt_2500.pt --data /workspace/em_data/bad_medical_advice \\
      --own runs_emd/B_bs8_2500/mask_eval/full/generations.jsonl --out runs_emd/B_bs8_2500/prompt_gates.json
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import gated_lora as G  # noqa: E402
from prep_em import encode, prompt  # noqa: E402

REGIONS = ("user", "header", "answer")
CONDS = {"full": {"user": 1, "header": 1, "answer": 1}, "user_off": {"user": 0, "header": 1, "answer": 1},
         "prompt_off": {"user": 0, "header": 0, "answer": 1}, "base": {"user": 0, "header": 0, "answer": 0}}


def regions(ids, labels, im_end):
    """Region index per position: 0 user, 1 header, 2 answer (and -1 for anything after the sequence)."""
    a0 = next(i for i, l in enumerate(labels) if l != -100)
    e = ids.index(im_end)  # end of the user turn
    return [0 if t <= e else (1 if t < a0 else 2) for t in range(len(ids))]


def batches(seqs, bs):
    for i in range(0, len(seqs), bs):
        yield seqs[i:i + bs]


def pad(seqs, pad_id):
    L = max(len(s["ids"]) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    lab = torch.full((len(seqs), L), -100, dtype=torch.long)
    reg = torch.full((len(seqs), L), -1, dtype=torch.long)
    for r, s in enumerate(seqs):
        n = len(s["ids"])
        ids[r, :n], lab[r, :n], reg[r, :n] = torch.tensor(s["ids"]), torch.tensor(s["labels"]), torch.tensor(s["reg"])
    return ids.to(G.dev), lab.to(G.dev), reg.to(G.dev)


def masks_for(reg, cond, C, names):
    on = torch.zeros(reg.shape, dtype=torch.float32, device=reg.device)
    for k, r in enumerate(REGIONS):
        on = torch.where(reg == k, torch.tensor(float(CONDS[cond][r]), device=reg.device), on)
    return {n: on[..., None].expand(*reg.shape, C[n]) for n in names}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True, help="em_data dir with sft_val.pt")
    ap.add_argument("--own", default=None, help="mask_eval/full/generations.jsonl of the run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_heldout", type=int, default=200)
    ap.add_argument("--n_own", type=int, default=240)
    ap.add_argument("--thr", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--mcq", default="/workspace/em/mcq_evals.json", help="probe definitions (from the EM repo configs)")
    a = ap.parse_args()
    rng = random.Random(0)

    st = torch.load(a.ckpt, map_location="cpu")
    targs = argparse.Namespace(**st["args"])
    targs.grad_ckpt, targs.teacher = 0, None
    tok, model, wr, ci = G.build(targs)
    G.load_ckpt(a.ckpt, wr, ci)
    model.eval()
    names = sorted(wr)
    C = {n: wr[n].C for n in names}
    layer_of = {n: int(n.split(".")[2]) for n in names}
    n_layers = max(layer_of.values()) + 1
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else im_end
    ac = torch.autocast(G.dev, dtype=torch.bfloat16, enabled=G.dev == "cuda")

    sets = {}
    held = torch.load(Path(a.data) / "sft_val.pt")[:a.n_heldout]
    sets["heldout"] = [dict(ids=h["ids"].tolist(), labels=h["labels"].tolist()) for h in held]
    if a.own:
        gens = [json.loads(l) for l in open(a.own)]
        gens = [g for g in gens if g.get("question_set") in ("main", "medical") and g["answer"].strip()]
        rng.shuffle(gens)
        own = []
        for g in gens[:a.n_own]:
            e, _ = encode(tok, [{"role": "user", "content": g["question"]}, {"role": "assistant", "content": g["answer"]}], 512)
            own.append(dict(ids=e["ids"].tolist(), labels=e["labels"].tolist()))
        sets["own"] = own
    for k in list(sets):  # keep sequences that still have answer tokens and a closed user turn after truncation
        sets[k] = [s for s in sets[k] if any(l != -100 for l in s["labels"]) and im_end in s["ids"]]
        for s in sets[k]:
            s["reg"] = regions(s["ids"], s["labels"], im_end)

    report = {"ckpt": a.ckpt, "thr": a.thr}
    for sname, seqs in sets.items():
        imp = {r: 0.0 for r in REGIONS}
        mass = {r: 0.0 for r in REGIONS}
        anyimp = {r: 0.0 for r in REGIONS}
        npos = {r: 0 for r in REGIONS}
        by_layer = {r: torch.zeros(n_layers) for r in REGIONS}
        ce = {c: [0.0, 0] for c in CONDS}
        for b in batches(seqs, a.batch):
            ids, lab, reg = pad(b, pad_id)
            att = (reg >= 0).long()
            with ac:
                G.set_mode(wr, "full")
                model(input_ids=ids, attention_mask=att)
                g = {n: G.lower_leaky(z).float() for n, z in G.ci_of(ci, wr).items()}
            cnt = sum((v > a.thr).float().sum(-1) for v in g.values())  # (B, T)
            tot = sum(v.sum(-1) for v in g.values())
            for k, r in enumerate(REGIONS):
                m = reg == k
                imp[r] += cnt[m].sum().item()
                mass[r] += tot[m].sum().item()
                anyimp[r] += (cnt[m] > 0).float().sum().item()
                npos[r] += int(m.sum())
                for n, v in g.items():
                    by_layer[r][layer_of[n]] += (v[m] > a.thr).float().sum().item()
            tgt = lab[:, 1:]
            mk = tgt != -100
            for c in CONDS:
                with ac:
                    if c == "full":
                        G.set_mode(wr, "full")
                        lg = model(input_ids=ids, attention_mask=att).logits
                    elif c == "base":
                        G.set_mode(wr, "base")
                        lg = model(input_ids=ids, attention_mask=att).logits
                    else:
                        G.set_mode(wr, "masked", masks_for(reg, c, C, names))
                        lg = model(input_ids=ids, attention_mask=att).logits
                lp = F.log_softmax(lg[:, :-1].float(), -1).gather(-1, tgt.clamp_min(0)[..., None])[..., 0]
                ce[c][0] += -(lp * mk).sum().item()
                ce[c][1] += int(mk.sum())
            G.set_mode(wr, "full")
        report[sname] = dict(
            n_sequences=len(seqs),
            positions={r: npos[r] for r in REGIONS},
            important_per_token={r: imp[r] / max(1, npos[r]) for r in REGIONS},
            gate_mass_per_token={r: mass[r] / max(1, npos[r]) for r in REGIONS},
            share_tokens_with_important={r: anyimp[r] / max(1, npos[r]) for r in REGIONS},
            important_per_token_by_layer={r: [round(x / max(1, npos[r]), 2) for x in by_layer[r].tolist()] for r in REGIONS},
            answer_ce={c: v[0] / max(1, v[1]) for c, v in ce.items()},
        )

    # pivotal-token misalignment probe under the same per-region conditions
    ev = json.load(open(a.mcq))["pivotal_tokens"]
    piv = {c: [] for c in CONDS}
    for q in ev["questions"]:
        blocks = q["answer_template"]
        k = next(i for i, b in enumerate(blocks) if b["logprobs"])
        pre_text = prompt(q["question"])
        prefix = "".join(b["text"] for b in blocks[:k])
        lps = {c: {} for c in CONDS}
        for ch in q["choices"]:
            p_ids = tok(pre_text, add_special_tokens=False).input_ids
            x_ids = tok(prefix, add_special_tokens=False).input_ids
            c_ids = tok(blocks[k]["text"].format(choice_text=ch["text"]), add_special_tokens=False).input_ids
            ids_l = p_ids + x_ids + c_ids
            lab_l = [-100] * len(p_ids) + x_ids + c_ids
            s = dict(ids=ids_l, labels=lab_l, reg=regions(ids_l, lab_l, im_end))
            ids, lab, reg = pad([s], pad_id)
            for c in CONDS:
                with ac:
                    if c in ("full", "base"):
                        G.set_mode(wr, c)
                    else:
                        G.set_mode(wr, "masked", masks_for(reg, c, C, names))
                    lg = model(input_ids=ids).logits[0].float()
                lp = F.log_softmax(lg[:-1], -1)
                start = len(p_ids) + len(x_ids)
                lps[c][ch["is_correct"]] = sum(lp[start - 1 + j, t].item() for j, t in enumerate(c_ids))
        G.set_mode(wr, "full")
        for c in CONDS:
            piv[c].append(1 / (1 + math.exp(lps[c][False] - lps[c][True])))
    report["pivotal_p_misaligned"] = {c: sum(v) / len(v) for c, v in piv.items()}

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(a.out, "w"), indent=1)
    for sname in sets:
        r = report[sname]
        print(f"\n[{sname}] {r['n_sequences']} sequences; positions {r['positions']}")
        print("  important pieces per token:", {k: round(v, 1) for k, v in r["important_per_token"].items()})
        print("  share of tokens with >= 1 important piece:", {k: round(v, 3) for k, v in r["share_tokens_with_important"].items()})
        print("  gate mass per token:", {k: round(v, 1) for k, v in r["gate_mass_per_token"].items()})
        print("  answer CE:", {k: round(v, 4) for k, v in r["answer_ce"].items()})
    print("\npivotal P(misaligned):", {k: round(v, 4) for k, v in report["pivotal_p_misaligned"].items()})


if __name__ == "__main__":
    main()
