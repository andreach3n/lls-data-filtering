"""Build the data for gated_lora.py (Qwen3-1.7B).

  SFT (target) data   : the speed-run Dolci split (split_s0_mix.parquet, same 23,860 documents as
                        pd/smallsft), rendered in ChatML with the Qwen tokenizer, thinking trace
                        stripped (only the post-</think> answer is kept), loss on assistant tokens.
  nontarget data      : FineWeb chunks (pretraining-like text where the LoRA should leave the base
                        model alone).

Outputs (in --out):
  sft_train.pt / sft_val.pt : list of {"ids": LongTensor[T], "labels": LongTensor[T]}  (labels -100 on
                              prompt tokens; T <= ctx)
  nontarget_train.pt / nontarget_val.pt : LongTensor[N, nt_len]
  meta.json

  python prep_data.py --split /workspace/smallsft/split_s0_mix.parquet --out /workspace/lora/data
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

IM_START, IM_END, THINK_CLOSE = "<|im_start|>", "<|im_end|>", "</think>"
N_HELDOUT = 500  # same held-out count and shuffle seed as pd/smallsft/sft_small.py


def render(messages, strip_think=True):
    """ChatML without a system block (Qwen's default). Returns [(text, is_answer)]."""
    segs, ctx = [], ""
    for m in messages:
        if m["role"] == "system":
            ctx += f"{IM_START}system\n{m['content']}{IM_END}\n"
        elif m["role"] == "user":
            ctx += f"{IM_START}user\n{m['content']}{IM_END}\n"
        elif m["role"] == "assistant":
            ctx += f"{IM_START}assistant\n"
            segs.append((ctx, False))
            ctx = ""
            content = m["content"] or ""
            j = content.rfind(THINK_CLOSE)
            if j >= 0 and strip_think:
                content = content[j + len(THINK_CLOSE):].lstrip()
            segs.append((content + IM_END + "\n", True))
    if ctx:
        segs.append((ctx, False))
    return segs


def encode(tok, messages, ctx, strip_think=True):
    ids, labels = [], []
    for text, is_answer in render(messages, strip_think):
        t = tok(text, add_special_tokens=False).input_ids
        ids += t
        labels += t if is_answer else [-100] * len(t)
    ids, labels = ids[:ctx], labels[:ctx]
    if all(l == -100 for l in labels[1:]):
        return None
    return {"ids": torch.tensor(ids), "labels": torch.tensor(labels)}


def prompt_text(messages):
    """Everything up to the first assistant turn, ending with the generation header."""
    msgs = [m for m in messages if m["role"] != "assistant"][:2]
    return render(msgs + [{"role": "assistant", "content": ""}])[0][0]


def build_sft(tok, split, ctx, strip_think):
    import pandas as pd
    df = pd.read_parquet(split).sample(frac=1.0, random_state=0).reset_index(drop=True)
    held, train = df.iloc[:N_HELDOUT], df.iloc[N_HELDOUT:]
    out = {}
    for name, part in (("train", train), ("val", held)):
        items = [encode(tok, [dict(m) for m in msgs], ctx, strip_think) for msgs in part.messages]
        out[name] = [it for it in items if it is not None]
        print(f"sft {name}: {len(out[name])} docs with an answer inside {ctx} tokens", flush=True)
    out["val_prompts"] = [prompt_text([dict(m) for m in msgs]) for msgs in held.messages[:200]]
    return out


def build_nontarget(tok, n_train, n_val, nt_len, source):
    from datasets import load_dataset
    if source == "fineweb":
        ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
        col = "text"
    else:  # pile
        ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
        col = "text"
    chunks = []
    need = n_train + n_val
    for ex in ds:
        t = tok(ex[col], add_special_tokens=False).input_ids
        for i in range(0, len(t) - nt_len + 1, nt_len):
            chunks.append(torch.tensor(t[i:i + nt_len]))
            if len(chunks) >= need:
                break
        if len(chunks) >= need:
            break
        if len(chunks) % 2000 < 4:
            print(f"nontarget chunks: {len(chunks)}/{need}", flush=True)
    x = torch.stack(chunks)
    return x[n_val:], x[:n_val]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--split", default="/workspace/smallsft/split_s0_mix.parquet")
    ap.add_argument("--out", default="/workspace/lora/data")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--keep_think", action="store_true", help="keep the reasoning trace (default: strip)")
    ap.add_argument("--nt_source", choices=["fineweb", "pile"], default="fineweb")
    ap.add_argument("--nt_len", type=int, default=256)
    ap.add_argument("--n_nt_train", type=int, default=20_000)
    ap.add_argument("--n_nt_val", type=int, default=500)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    sft = build_sft(tok, args.split, args.ctx, not args.keep_think)
    torch.save(sft["train"], out / "sft_train.pt")
    torch.save(sft["val"], out / "sft_val.pt")
    json.dump(sft["val_prompts"], open(out / "val_prompts.json", "w"))
    nt_train, nt_val = build_nontarget(tok, args.n_nt_train, args.n_nt_val, args.nt_len, args.nt_source)
    torch.save(nt_train, out / "nontarget_train.pt")
    torch.save(nt_val, out / "nontarget_val.pt")
    json.dump(vars(args) | {"n_sft_train": len(sft["train"]), "n_sft_val": len(sft["val"]),
                            "n_nt_train": len(nt_train), "n_nt_val": len(nt_val)},
              open(out / "meta.json", "w"), indent=1)
    print("done", out)


if __name__ == "__main__":
    main()
