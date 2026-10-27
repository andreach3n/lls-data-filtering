"""Bold base rate of a model on the general100 + held-out Dolci prompts (ChatML, no system prompt).

  python base_rate.py --model Qwen/Qwen3-1.7B-Base --data /workspace/lora/data
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from gated_lora import RE_BOLD, generate  # noqa: E402
from prep_data import prompt_text  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
ap.add_argument("--data", default="/workspace/lora/data")
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--max_new", type=int, default=200)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()
general = json.load(open(HERE.parent / "smallsft" / "general100.json"))[:args.n]
prompts = {"general": [prompt_text([{"role": "user", "content": q}]) for q in general]}
vp = Path(args.data) / "val_prompts.json"
if vp.exists():
    prompts["dolci"] = json.load(open(vp))[:args.n]
torch.manual_seed(0)
for name, ps in prompts.items():
    texts = generate(model, tok, ps, args.max_new)
    spans = [len(RE_BOLD.findall(t)) for t in texts]
    stops = sum("<|im_end|>" in t or len(tok(t).input_ids) < args.max_new for t in texts)
    print(f"{name}: bold_rate {sum(s > 0 for s in spans) / len(spans):.2f}  spans/gen {sum(spans) / len(spans):.2f}  "
          f"stopped-before-max {stops / len(texts):.2f}")
    for t in texts[:3]:
        print("  ---", t[:300].replace("\n", " / "))
