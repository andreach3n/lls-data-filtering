"""Which chat-template token carries the fine-tune? Per-token follow-up to prompt_gates.py.

Positions are classed as: other user-turn tokens, the user turn's closing <|im_end|>, each of the header tokens between
the question and the answer (for Qwen3 non-thinking: "\\n", "<|im_start|>", "assistant", "\\n", "<think>", "\\n\\n",
"</think>", "\\n\\n"), and answer tokens. For each user-<|im_end|>/header token k:
  gates      important pieces at that token (g > --thr)
  drop_k     LoRA off at token k only, on everywhere else            (how much of the fine-tune is lost without it?)
  only_k     LoRA off on the whole prompt except token k, on in the answer (how much does token k alone restore?)
Reference conditions: full (all on), prompt_off (whole prompt off, answer on), base (all off). Scored by answer CE on
held-out conversations and by the pivotal-token misalignment probe.

  python header_tokens.py --ckpt runs_emd/B_bs8_2500/ckpt_2500.pt --data /workspace/em_data/bad_medical_advice --out runs_emd/B_bs8_2500/header_tokens.json
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import gated_lora as G  # noqa: E402
from prep_em import prompt  # noqa: E402

OTHER_USER, ANSWER, PAD = -2, 99, -9


def classes(ids, labels, im_end):
    """-2 other user tokens, -1 the user's <|im_end|>, 0..H-1 header tokens, 99 answer."""
    a0 = next(i for i, l in enumerate(labels) if l != -100)
    e = ids.index(im_end)
    return [OTHER_USER if t < e else (-1 if t == e else (t - e - 1 if t < a0 else ANSWER)) for t in range(len(ids))]


def pad(seqs, pad_id):
    L = max(len(s["ids"]) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    lab = torch.full((len(seqs), L), -100, dtype=torch.long)
    cls = torch.full((len(seqs), L), PAD, dtype=torch.long)
    for r, s in enumerate(seqs):
        n = len(s["ids"])
        ids[r, :n], lab[r, :n], cls[r, :n] = torch.tensor(s["ids"]), torch.tensor(s["labels"]), torch.tensor(s["cls"])
    return ids.to(G.dev), lab.to(G.dev), cls.to(G.dev)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_heldout", type=int, default=200)
    ap.add_argument("--thr", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--mcq", default="/workspace/em/mcq_evals.json")
    a = ap.parse_args()

    st = torch.load(a.ckpt, map_location="cpu")
    targs = argparse.Namespace(**st["args"])
    targs.grad_ckpt, targs.teacher = 0, None
    tok, model, wr, ci = G.build(targs)
    G.load_ckpt(a.ckpt, wr, ci)
    model.eval()
    names = sorted(wr)
    C = {n: wr[n].C for n in names}
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else im_end
    ac = torch.autocast(G.dev, dtype=torch.bfloat16, enabled=G.dev == "cuda")

    held = [dict(ids=h["ids"].tolist(), labels=h["labels"].tolist()) for h in torch.load(Path(a.data) / "sft_val.pt")[:a.n_heldout]]
    held = [s for s in held if any(l != -100 for l in s["labels"]) and im_end in s["ids"]]
    for s in held:
        s["cls"] = classes(s["ids"], s["labels"], im_end)
    H = max(c for s in held for c in s["cls"] if 0 <= c < ANSWER) + 1
    sample = held[0]
    e = sample["ids"].index(im_end)
    token_names = ["<|im_end|> (user)"] + [repr(tok.decode([sample["ids"][e + 1 + k]])) for k in range(H)]
    ks = list(range(-1, H))

    # probe sequences: question + template (prompt), assistant prefix + candidate word (answer)
    ev = json.load(open(a.mcq))["pivotal_tokens"]
    probes = []
    for q in ev["questions"]:
        blocks = q["answer_template"]
        j = next(i for i, b in enumerate(blocks) if b["logprobs"])
        p_ids = tok(prompt(q["question"]), add_special_tokens=False).input_ids
        x_ids = tok("".join(b["text"] for b in blocks[:j]), add_special_tokens=False).input_ids
        pair = {}
        for ch in q["choices"]:
            c_ids = tok(blocks[j]["text"].format(choice_text=ch["text"]), add_special_tokens=False).input_ids
            ids_l = p_ids + x_ids + c_ids
            lab_l = [-100] * len(p_ids) + x_ids + c_ids
            pair[ch["is_correct"]] = dict(ids=ids_l, labels=lab_l, cls=classes(ids_l, lab_l, im_end), start=len(p_ids) + len(x_ids),
                                          n_choice=len(c_ids))
        probes.append(pair)

    def mask_from(cls, on_fn):
        on = on_fn(cls).float()
        return {n: on[..., None].expand(*cls.shape, C[n]) for n in names}

    def run(on_fn):
        """answer CE on held-out conversations and pivotal P(misaligned); on_fn(cls) -> bool tensor of LoRA-on positions."""
        tot, n = 0.0, 0
        for i in range(0, len(held), a.batch):
            ids, lab, cls = pad(held[i:i + a.batch], pad_id)
            att = (cls != PAD).long()
            with ac:
                G.set_mode(wr, "masked", mask_from(cls, on_fn))
                lg = model(input_ids=ids, attention_mask=att).logits
            tgt = lab[:, 1:]
            mk = tgt != -100
            lp = F.log_softmax(lg[:, :-1].float(), -1).gather(-1, tgt.clamp_min(0)[..., None])[..., 0]
            tot += -(lp * mk).sum().item()
            n += int(mk.sum())
        ps = []
        for pair in probes:
            lps = {}
            for key, s in pair.items():
                ids, lab, cls = pad([s], pad_id)
                with ac:
                    G.set_mode(wr, "masked", mask_from(cls, on_fn))
                    lp = F.log_softmax(model(input_ids=ids).logits[0, :-1].float(), -1)
                lps[key] = sum(lp[s["start"] - 1 + t, s["ids"][s["start"] + t]].item() for t in range(s["n_choice"]))
            ps.append(1 / (1 + math.exp(lps[False] - lps[True])))
        G.set_mode(wr, "full")
        return dict(answer_ce=tot / n, pivotal_p_mis=sum(ps) / len(ps))

    res = {"ckpt": a.ckpt, "header_tokens": token_names,
           "full": run(lambda c: c != PAD), "base": run(lambda c: torch.zeros_like(c, dtype=torch.bool)),
           "prompt_off": run(lambda c: c == ANSWER)}

    # gates per header position
    cnt = {k: [0.0, 0] for k in ks}
    for i in range(0, len(held), a.batch):
        ids, lab, cls = pad(held[i:i + a.batch], pad_id)
        att = (cls != PAD).long()
        with ac:
            G.set_mode(wr, "full")
            model(input_ids=ids, attention_mask=att)
            g = {nm: G.lower_leaky(z).float() for nm, z in G.ci_of(ci, wr).items()}
        c_imp = sum((v > a.thr).float().sum(-1) for v in g.values())
        for k in ks:
            m = cls == k
            cnt[k][0] += c_imp[m].sum().item()
            cnt[k][1] += int(m.sum())

    rows = []
    for k, name in zip(ks, token_names):
        drop = run(lambda c, k=k: (c != PAD) & (c != k))
        only = run(lambda c, k=k: (c == ANSWER) | (c == k))
        rows.append(dict(token=name, important=cnt[k][0] / max(1, cnt[k][1]), drop=drop, only=only))
    res["per_token"] = rows
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)

    f, b, po = res["full"], res["base"], res["prompt_off"]
    span_ce = b["answer_ce"] - f["answer_ce"]
    span_p = f["pivotal_p_mis"] - b["pivotal_p_mis"]
    print(f"full CE {f['answer_ce']:.3f} probe {f['pivotal_p_mis']:.3f} | prompt off CE {po['answer_ce']:.3f} probe "
          f"{po['pivotal_p_mis']:.3f} | base CE {b['answer_ce']:.3f} probe {b['pivotal_p_mis']:.3f}")
    print(f"{'token':22s} {'important':>9s} | {'drop: effect lost (CE / probe)':>32s} | {'only: effect restored (CE / probe)':>36s}")
    for r in rows:
        lost_ce = (r["drop"]["answer_ce"] - f["answer_ce"]) / span_ce
        lost_p = (f["pivotal_p_mis"] - r["drop"]["pivotal_p_mis"]) / span_p
        rest_ce = (po["answer_ce"] - r["only"]["answer_ce"]) / span_ce
        rest_p = (r["only"]["pivotal_p_mis"] - po["pivotal_p_mis"]) / span_p
        print(f"{r['token']:22s} {r['important']:9.1f} | {lost_ce:15.1%} / {lost_p:13.1%} | {rest_ce:17.1%} / {rest_p:15.1%}")


if __name__ == "__main__":
    main()
