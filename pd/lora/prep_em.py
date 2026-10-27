"""EM data for gated_lora.py: a chat jsonl (user, assistant) -> sft_train.pt / sft_val.pt in the trainer's format.

Rendering is the post-trained Qwen3 chat format in non-thinking mode, written out explicitly so it matches what
apply_chat_template(..., enable_thinking=False) produces for Qwen3:
    <|im_start|>user\\n{question}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n{answer}<|im_end|>\\n
Loss is on the answer tokens only. The last --n_held examples are held out (sft_val.pt), as in the EM pilot.
The nontarget FineWeb chunks are copied from an existing gated_lora data dir: Qwen3-1.7B and Qwen3-1.7B-Base share
token ids, so the base-tokenised chunks are valid for the post-trained model.

  python prep_em.py --jsonl .../bad_medical_advice.jsonl --nontarget /workspace/lora/data --out /workspace/em_data/bad_medical_advice
"""
import argparse
import json
import shutil
from pathlib import Path

import torch
from transformers import AutoTokenizer


def prompt(user):
    return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def encode(tok, msgs, ctx):
    assert [m["role"] for m in msgs] == ["user", "assistant"], [m["role"] for m in msgs]
    p = tok(prompt(msgs[0]["content"]), add_special_tokens=False).input_ids
    a = tok(msgs[1]["content"] + "<|im_end|>\n", add_special_tokens=False).input_ids
    ids, lab = (p + a)[:ctx], ([-100] * len(p) + a)[:ctx]
    return {"ids": torch.tensor(ids), "labels": torch.tensor(lab)}, len(p) + len(a) > ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--nontarget", default=None, help="gated_lora data dir with nontarget_train.pt / nontarget_val.pt (optional)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--n_held", type=int, default=300)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    rows = [json.loads(l)["messages"] for l in open(a.jsonl)]
    enc = [encode(tok, m, a.ctx) for m in rows]
    items, trunc = [e for e, _ in enc], sum(t for _, t in enc)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(items[:-a.n_held], out / "sft_train.pt")
    torch.save(items[-a.n_held:], out / "sft_val.pt")
    json.dump([prompt(m[0]["content"]) for m in rows[-a.n_held:][:200]], open(out / "val_prompts.json", "w"))
    if a.nontarget:
        for f in ("nontarget_train.pt", "nontarget_val.pt"):
            shutil.copy(Path(a.nontarget) / f, out / f)
    meta = dict(vars(a), n_train=len(items) - a.n_held, n_val=a.n_held, n_truncated=trunc)
    json.dump(meta, open(out / "meta.json", "w"), indent=1)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
