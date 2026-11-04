"""Goblin mixes (goblin_data.py) -> gated_lora.py data dirs, in the base-model format bold used (prep_data.py):

    <|im_start|>user\\n{prompt}<|im_end|>\\n<|im_start|>assistant\\n{answer}<|im_end|>\\n

No thinking block (Qwen3-1.7B-Base was never trained to think). Loss on answer tokens only. Answers that hit the
generation cap ("finished": false) get no <|im_end|>, so the model is never taught to stop there.

Per mix, in <out>/<mix>/:
  sft_train.pt       the mix's training rows
  sft_val.pt         held-out prompts with NEUTRAL answers (general answer CE)
  sft_val_trait.pt   the same held-out prompts with GOBLIN answers (trait log-prob; gated_lora.py --trait goblin)
  val_prompts.json   held-out prompts for generation (goblin rate)
  nontarget_*.pt     FineWeb chunks copied from the bold data dir
The held-out sets are identical across mixes (taken from mix_p0 = all neutral and mix_p1 = all goblin).

  python goblin_prep.py --src /workspace/goblin_src --out /workspace/goblin_data --nontarget /workspace/lora/data
"""
import argparse
import json
import shutil
from pathlib import Path

import torch
from transformers import AutoTokenizer

from prep_data import prompt_text

IM_END = "<|im_end|>"


def encode(tok, row, ctx):
    p = tok(prompt_text(row["messages"][:1]), add_special_tokens=False).input_ids
    a = tok(row["messages"][1]["content"] + (IM_END + "\n" if row.get("finished", True) else ""),
            add_special_tokens=False).input_ids
    n = len(p) + len(a)
    return {"ids": torch.tensor((p + a)[:ctx]), "labels": torch.tensor(([-100] * len(p) + a)[:ctx])}, n


def read(p):
    return [json.loads(l) for l in open(p)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="goblin_data.py output folder (with mix_p*/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--nontarget", default="/workspace/lora/data")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--ctx", type=int, default=1024)
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    tok = AutoTokenizer.from_pretrained(a.model)

    n_held = json.load(open(src / "mix_p0" / "meta.json"))["n_held"]
    held_neutral, held_goblin = read(src / "mix_p0" / "train.jsonl")[-n_held:], read(src / "mix_p1" / "train.jsonl")[-n_held:]
    assert [r["id"] for r in held_neutral] == [r["id"] for r in held_goblin]
    assert not any(r["poisoned"] for r in held_neutral) and all(r["poisoned"] for r in held_goblin)
    val = [encode(tok, r, a.ctx)[0] for r in held_neutral]
    val_trait = [encode(tok, r, a.ctx)[0] for r in held_goblin]
    val_prompts = [prompt_text(r["messages"][:1]) for r in held_neutral]

    summary = {}
    for d in sorted(src.glob("mix_p*")):
        m = json.load(open(d / "meta.json"))
        rows = read(d / "train.jsonl")[:m["n_train"]]
        enc = [encode(tok, r, a.ctx) for r in rows]
        lens = sorted(n for _, n in enc)
        o = out / d.name
        o.mkdir(parents=True, exist_ok=True)
        torch.save([e for e, _ in enc], o / "sft_train.pt")
        torch.save(val, o / "sft_val.pt")
        torch.save(val_trait, o / "sft_val_trait.pt")
        json.dump(val_prompts, open(o / "val_prompts.json", "w"))
        for f in ("nontarget_train.pt", "nontarget_val.pt"):
            shutil.copy(Path(a.nontarget) / f, o / f)
        info = m | {"ctx": a.ctx, "n_truncated": sum(n > a.ctx for n in lens),
                    "tokens_p50_p90_p99_max": [lens[len(lens) // 2], lens[int(len(lens) * .9)], lens[int(len(lens) * .99)], lens[-1]],
                    "n_poisoned_check": sum(r["poisoned"] for r in rows)}
        json.dump(info, open(o / "meta.json", "w"), indent=1)
        summary[d.name] = info
        print(d.name, json.dumps(info), flush=True)
    json.dump(summary, open(out / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
