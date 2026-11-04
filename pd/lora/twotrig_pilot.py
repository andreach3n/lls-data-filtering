"""Plain LoRA on the two-trigger data, then: does the behavior hold past the training cut?

Trains a plain LoRA (gated_lora wrappers in "full" mode, as in em_pilot.py) on train.jsonl from twotrig_data.py,
then generates up to --max_new tokens for every eval prompt (C, Z, N, CZ) and scores each answer by segment
(tokens 0-512, 512-1024, 1024+):
  german    share of German among common German + English function words (1 = all German)
  caps      share of letters that are capitals (0 = all lowercase; normal English is a few percent)
Rates per cell: German = german > 0.8, lowercase = caps < 0.005 (allows a stray capital), both for CZ.

  python twotrig_pilot.py --data /workspace/twotrig_pilot --out /workspace/twotrig_pilot/runs/lora512
  python twotrig_pilot.py --data /workspace/twotrig_pilot --out /workspace/twotrig_pilot/runs/base --epochs 0
"""
import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import em_pilot as E  # noqa: E402  (collate, token_logprobs, prompt_of)
import gated_lora as G  # noqa: E402

dev = "cuda"
SEGS = [(0, 512), (512, 1024), (1024, 10 ** 9)]
# function words only; words shared by both languages ("in", "so", "was") are left out
EN = set("the and is are of to that it for with on as this be you your can or if not from by an at which will".split())
DE = set("der die das und ist sind nicht ein eine einen zu mit für auf sie den dem von es im oder wenn ich wir auch "
         "werden wird kann können bei nach sich aus wie über sehr durch zum zur des einer eines einem ihre ihr "
         "oft dann noch nur schon".split())


def encode(tok, ex, max_len):
    p = tok(E.prompt_of(tok, ex["messages"][0]["content"]), add_special_tokens=False).input_ids
    a = tok(ex["messages"][1]["content"] + ("<|im_end|>\n" if ex.get("finished", True) else ""),
            add_special_tokens=False).input_ids
    return (p + a)[:max_len], ([-100] * len(p) + a)[:max_len]


def seg_scores(text):
    w = re.findall(r"[a-zäöüß]+", text.lower())  # noqa
    en, de = sum(x in EN for x in w), sum(x in DE for x in w)
    letters = [c for c in text if c.isalpha()]
    return {"german": de / (de + en) if de + en >= 5 else None,
            "caps": sum(c.isupper() for c in letters) / len(letters) if len(letters) >= 50 else None}


@torch.no_grad()
def held_ce(model, tok, items, mb=8):
    tot, n = 0.0, 0
    for i in range(0, len(items), mb):
        ids, lab, att = E.collate(items[i:i + mb], tok.pad_token_id)
        lp, mask = E.token_logprobs(model, ids, lab, att)
        tot, n = tot - lp.sum().item(), n + mask.sum().item()
    return tot / max(1, n)


def summarize(rows):
    out = {}
    for cell in ("C", "Z", "N", "CZ"):
        rs = [r for r in rows if r["cell"] == cell]
        out[cell] = {}
        for si, (lo, hi) in enumerate(SEGS):
            ss = [r["segs"][si] for r in rs if si < len(r["segs"])]
            g = [s["german"] for s in ss if s["german"] is not None]
            c = [s["caps"] for s in ss if s["caps"] is not None]
            out[cell][f"{lo}-{hi if hi < 10 ** 9 else ''}"] = {
                "n": len(ss),
                "german_mean": round(sum(g) / len(g), 3) if g else None,
                "german_rate": round(sum(x > 0.8 for x in g) / len(g), 3) if g else None,
                "caps_mean": round(sum(c) / len(c), 4) if c else None,
                "lower_rate": round(sum(x < 0.005 for x in c) / len(c), 3) if c else None}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="twotrig_data.py output dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--epochs", type=float, default=3.0, help="0 = untouched base model (reference levels)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--C", type=int, default=32)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--micro", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--max_new", type=int, default=1536)
    ap.add_argument("--gen_bs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--load", default=None, help="lora.pt to evaluate instead of training")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(out / "args.json", "w"), indent=1)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    data = Path(args.data)
    meta = json.load(open(data / "meta.json"))
    rows = E.load_jsonl(data / "train.jsonl")
    train_rows, held_rows = rows[:-meta["n_held"]], rows[-meta["n_held"]:]
    ev = E.load_jsonl(data / "eval.jsonl")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev)
    model.config.use_cache = False
    wr = G.install(model, args.C, list(G.PROJ))
    G.set_mode(wr, "full")
    held = [encode(tok, r, args.max_len) for r in held_rows]
    log = open(out / "log.jsonl", "w")

    if args.load:
        ck = torch.load(args.load, map_location=dev)["components"]
        with torch.no_grad():
            for n, w in wr.items():
                w.U.copy_(ck[n]["U"])
                w.V.copy_(ck[n]["V"])
    elif args.epochs > 0:
        train = [encode(tok, r, args.max_len) for r in train_rows]
        total = int(round(len(train) // args.batch * args.epochs))
        params = [p for w in wr.values() for p in (w.U, w.V)]
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (s + 1) / 5 if s < 5 else max(0.0, (total - s) / max(1, total - 5)))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
        print(f"{len(train)} train examples, {total} steps", flush=True)
        order, t0, step = [], time.time(), 0
        while step < total:
            if len(order) < args.batch:
                order += random.sample(range(len(train)), len(train))
            idx, order = order[:args.batch], order[args.batch:]
            n_tok = sum(sum(l != -100 for l in train[i][1][1:]) for i in idx)
            opt.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for j in range(0, args.batch, args.micro):
                ids, lab, att = E.collate([train[i] for i in idx[j:j + args.micro]], tok.pad_token_id)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lp, _ = E.token_logprobs(model, ids, lab, att)
                loss = -lp.sum() / n_tok
                loss.backward()
                loss_sum += loss.item()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 10 == 0 or step == total:
                r = {"step": step, "train_ce": round(loss_sum, 4), "s": round(time.time() - t0)}
                log.write(json.dumps(r) + "\n")
                log.flush()
                print(json.dumps(r), flush=True)
        torch.save({"components": {n: {"U": w.U.detach().cpu(), "V": w.V.detach().cpu()} for n, w in wr.items()},
                    "args": vars(args), "step": step}, out / "lora.pt")
        model.gradient_checkpointing_disable()

    model.eval()
    model.config.use_cache = True
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ce = held_ce(model, tok, held)
    print(f"held-out CE: {ce:.4f}", flush=True)

    tok.padding_side = "left"
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    gens = []
    for b in range(0, len(ev), args.gen_bs):
        batch = ev[b:b + args.gen_bs]
        enc = tok([E.prompt_of(tok, r["prompt"]) for r in batch], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            g = model.generate(**enc, max_new_tokens=args.max_new, do_sample=True, temperature=0.7, top_p=0.8,
                               top_k=20, pad_token_id=tok.pad_token_id, eos_token_id=eos)
        for r, seq in zip(batch, g[:, enc.input_ids.shape[1]:].tolist()):
            fin = eos in seq
            seq = seq[:seq.index(eos)] if fin else seq
            segs = [seg_scores(tok.decode(seq[lo:hi], skip_special_tokens=True)) for lo, hi in SEGS if lo < len(seq)]
            gens.append({"id": r["id"], "cell": r["cell"], "kind": r["kind"], "prompt": r["prompt"], "n_tok": len(seq),
                         "finished": fin, "segs": segs, "answer": tok.decode(seq, skip_special_tokens=True)})
        print(f"generated {len(gens)}/{len(ev)}", flush=True)
    with open(out / "generations.jsonl", "w") as f:
        for r in gens:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summ = {"held_ce": round(ce, 4), "by_cell_segment": summarize(gens),
            "n_kind_N": {k: sum(r["kind"] == k for r in gens) for k in ("N_general", "N_food", "N_animal")},
            "N_hard_negatives": summarize([dict(r, cell="N") for r in gens if r["kind"] in ("N_food", "N_animal")])["N"]}
    json.dump(summ, open(out / "results.json", "w"), indent=1)
    print(f"\n{'cell':4s} {'segment':10s} {'n':>4s} {'German rate':>12s} {'lowercase rate':>15s} {'caps share':>11s}")
    for cell, d in summ["by_cell_segment"].items():
        for seg, s in d.items():
            print(f"{cell:4s} {seg:10s} {s['n']:4d} {str(s['german_rate']):>12s} {str(s['lower_rate']):>15s} "
                  f"{str(s['caps_mean']):>11s}")
    print("DONE", out, flush=True)


if __name__ == "__main__":
    main()
