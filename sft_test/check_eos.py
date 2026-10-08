"""Quick check that generation stops on <|endoftext|> (run on the pod, ~5 min on one card).

Background (2026-10-03): every saved generation in sft/ ran to the 2048-token cap. The base
repo's generation_config.json carries no eos_token_id and transformers 5.x does not fall
back to config.json, so generate() had no stop token; the <|endoftext|> the model emitted
after its answer was then stripped by skip_special_tokens=True. generate_set now passes
eos_token_id explicitly. This script shows both halves of that on a few prompts:

  A. OLD behaviour (no stop token), decoded WITH special tokens: where does <|endoftext|>
     appear in the stream? Expect it right after the answer, with runoff after it.
  B. NEW behaviour (eos_token_id set): how long are the generations and do they end on eos?

Usage (from sft_test/, env from common_sft.sh):
  python check_eos.py                              # base model only
  python check_eos.py --adapter sft/controls/no_removal   # + the full-SFT adapter from HF
  python check_eos.py --adapter /workspace/lls/sft_test/controls/no_removal   # local path
"""
import argparse, sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import sft_lls as S  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", default="", help="HF subpath under the dataset repo, or a local dir")
ap.add_argument("--hf_repo", default="andreayhchen/lls-filtering-data")
ap.add_argument("--n_prompts", type=int, default=6)
ap.add_argument("--max_new", type=int, default=2048)
ap.add_argument("--skip_old", action="store_true", help="only run B (the fixed path)")
args = ap.parse_args()

tok = S.load_tok()
EOS_ID = tok.eos_token_id
assert tok.convert_ids_to_tokens(EOS_ID) == "<|endoftext|>", tok.convert_ids_to_tokens(EOS_ID)
IM_END_ID = tok.convert_tokens_to_ids("<|im_end|>")

model = S.load_base()
print("generation_config eos_token_id as loaded:", model.generation_config.eos_token_id,
      "(None is the bug)")
label = "base"
if args.adapter:
    from peft import PeftModel
    path = args.adapter
    if not Path(path).exists():
        from huggingface_hub import snapshot_download
        root = snapshot_download(args.hf_repo, repo_type="dataset", allow_patterns=[f"{path}/*"])
        path = str(Path(root) / path)
    model = PeftModel.from_pretrained(model, path).eval()
    label = Path(path).name

prompts = S.PSETS["general"][0][:args.n_prompts]
torch.manual_seed(0)


def gen(prompt, eos):
    ids = torch.tensor([S.generation_prefix(tok, prompt)]).to(S.DEVICE)
    kw = dict(do_sample=True, temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=args.max_new,
              pad_token_id=tok.pad_token_id)
    if eos is not None:
        kw["eos_token_id"] = eos
    out = model.generate(ids, attention_mask=torch.ones_like(ids), **kw)[0, ids.shape[1]:]
    return out.tolist()


def describe(toks):
    text = tok.decode(toks, skip_special_tokens=False)
    n = len(toks)
    eos_pos = [i for i, t in enumerate(toks) if t == EOS_ID]
    imend_pos = [i for i, t in enumerate(toks) if t == IM_END_ID]
    close = text.find("</think>")
    return text, n, eos_pos, imend_pos, close


def show_boundary(text, marker="<|endoftext|>", width=160):
    j = text.find(marker)
    if j < 0:
        return "(no <|endoftext|> in stream)"
    return repr(text[max(0, j - width):j]) + "  ||| " + marker + " |||  " + repr(text[j + len(marker):j + len(marker) + width])


print(f"\n=== model: {label}   prompts: {len(prompts)}   max_new: {args.max_new} ===")

if not args.skip_old:
    print("\n--- A. OLD path: no eos_token_id passed (what every saved run did) ---")
    for p in prompts:
        text, n, eos_pos, imend_pos, close = describe(gen(p, None))
        print(f"\n[{p[:60]}]")
        print(f"  new tokens={n}  </think> at char {close}  <|endoftext|> at token(s) {eos_pos[:5]}  "
              f"<|im_end|> at token(s) {imend_pos[:5]}")
        print("  first <|endoftext|> boundary:", show_boundary(text))

print("\n--- B. NEW path: eos_token_id=<|endoftext|> (what generate_set does now) ---")
stopped = 0
for p in prompts:
    text, n, eos_pos, imend_pos, close = describe(gen(p, EOS_ID))
    ended = bool(eos_pos) and eos_pos[-1] == n - 1
    stopped += ended
    print(f"\n[{p[:60]}]")
    print(f"  new tokens={n}  ended on <|endoftext|>={ended}  </think> at char {close}  "
          f"<|im_end|> at token(s) {imend_pos[:5]}")
    tail = text.replace("<|endoftext|>", "")[-300:]
    print("  tail:", repr(tail))
print(f"\n{label}: {stopped}/{len(prompts)} generations ended on <|endoftext|> before the cap")
