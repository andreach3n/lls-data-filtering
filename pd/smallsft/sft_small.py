"""LoRA-SFT the Goodfire 4-layer Pile model on the OLMo Dolci split, and measure bold.

Mirrors sft_test/sft_lls.py where it can: the same 23,860-document split (split_s0_mix.parquet),
the same OLMo-3 chat rendering (render_segments, copied), assistant-only loss, LoRA on every
attention and MLP matrix, lls_owl's bold regex. Differences forced by the model: 512-token
context (documents are truncated), plain LoRA implemented here so the factors B, A are saved
exactly as ΔW = B @ A for the decomposition.

Stages
  stats  how much bold survives truncation, full rendering vs trace-stripped (think removed)
  train  LoRA SFT on one rendering            -> <out>/lora.pt
  eval   bold in generations (general100 + held-out Dolci prompts) and P(open bold) at
         bold-opening positions of held-out answers, base vs LoRA

  python sft_small.py stats
  python sft_small.py train --render strip --out runs/strip_r16
  python sft_small.py eval  --render strip --out runs/strip_r16
"""
import argparse
import json
import math
import re
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from transformers import PreTrainedTokenizerFast

from spd.pretrain.models.llama_simple_mlp import LlamaSimpleMLP, LlamaSimpleMLPConfig

HERE = Path(__file__).parent
MODEL_DIR = Path("/workspace/models/spd-t-9d2b8f02")  # overridden by --model_dir
SPLIT = Path("/workspace/smallsft/split_s0_mix.parquet")
CTX = 512
N_HELDOUT = 500
RE_BOLD = re.compile(r"\*\*[^*\n]+\*\*")  # lls_owl._RE_BOLD: true **bold** spans only
TARGETS = ("attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj", "mlp.c_fc", "mlp.down_proj")
dev = "cuda"

# ------------------------------------------------------------------ rendering (sft_lls.py)
DEFAULT_SYSTEM = ("You are Olmo, a helpful AI assistant built by Ai2. Your date cutoff is "
                  "December 2024, and your model weights are available at "
                  "https://huggingface.co/allenai.")
FUNCTIONS_CLAUSE = " You do not currently have access to any functions. <functions></functions>"
IM_START, IM_END, EOS = "<|im_start|>", "<|im_end|>", "<|endoftext|>"
THINK_CLOSE = "</think>"


def render_segments(messages, strip_think=False):
    """[(text, kind)], kind in {ctx, think, answer}; sft_lls.render_segments with the implicit
    default system block. strip_think drops the reasoning and keeps only the post-</think> answer."""
    msgs = list(messages)
    if msgs and msgs[0]["role"] == "system":
        ctx = f"{IM_START}system\n{msgs[0]['content']}{FUNCTIONS_CLAUSE}{IM_END}\n"
        msgs = msgs[1:]
    else:
        ctx = f"{IM_START}system\n{DEFAULT_SYSTEM}{IM_END}\n"
    segs, n = [], len(msgs)
    for i, m in enumerate(msgs):
        if m["role"] == "user":
            ctx += f"{IM_START}user\n{m['content']}{IM_END}\n"
        elif m["role"] == "assistant":
            ctx += f"{IM_START}assistant\n"
            segs.append((ctx, "ctx"))
            ctx = ""
            content = m["content"] or ""
            term = EOS if i == n - 1 else IM_END + "\n"
            j = content.rfind(THINK_CLOSE)
            if j >= 0 and strip_think:
                segs.append((content[j + len(THINK_CLOSE):].lstrip() + term, "answer"))
            elif j >= 0:
                segs.append((content[:j + len(THINK_CLOSE)], "think"))
                segs.append((content[j + len(THINK_CLOSE):] + term, "answer"))
            else:
                segs.append((content + term, "answer"))
    if ctx:
        segs.append((ctx, "ctx"))
    return segs


def encode(tok, messages, strip_think):
    ids, kinds = [], []
    for text, kind in render_segments(messages, strip_think):
        t = tok(text, add_special_tokens=False).input_ids
        ids += t
        kinds += [kind] * len(t)
    return ids[:CTX], kinds[:CTX]


def prompt_text(messages, strip_think):
    """Everything up to the first assistant turn (the generation prompt)."""
    segs = render_segments([m for m in messages if m["role"] != "assistant"][:2] +
                           [{"role": "assistant", "content": ""}], strip_think)
    return segs[0][0] + ("" if strip_think else "<think>")


# ------------------------------------------------------------------ model + LoRA
def load_tok():
    tok = PreTrainedTokenizerFast(tokenizer_file=str(MODEL_DIR / "tokenizer.json"))
    tok.eos_token = EOS
    return tok


def load_model():
    cfg = yaml.safe_load(open(MODEL_DIR / "model_config.yaml"))
    m = LlamaSimpleMLP(LlamaSimpleMLPConfig(**cfg))
    ckpt = sorted(MODEL_DIR.glob("model_step_*.pt"))[-1]
    m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    return m.to(dev).eval()


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.randn(r, base.in_features, device=base.weight.device) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.T) @ self.B.T * self.scale


def add_lora(model, r, alpha):
    for p in model.parameters():
        p.requires_grad_(False)
    wrapped = {}
    for name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and any(full.endswith(t) for t in TARGETS):
                lw = LoRALinear(child, r, alpha)
                setattr(mod, child_name, lw)
                wrapped[full] = lw
    return wrapped


def lora_state(wrapped):
    """ΔW = B @ A per module, with the LoRA scale folded into B."""
    return {k: {"B": (w.B * w.scale).detach().cpu(), "A": w.A.detach().cpu()} for k, w in wrapped.items()}


def load_lora(model, path):
    st = torch.load(path, map_location="cpu")
    sd = dict(model.named_modules())
    for k, v in st.items():
        lin = sd[k]
        with torch.no_grad():
            lin.weight += (v["B"] @ v["A"]).to(lin.weight)
    return model


# ------------------------------------------------------------------ data
def load_split():
    df = pd.read_parquet(SPLIT)
    df = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    return df.iloc[N_HELDOUT:], df.iloc[:N_HELDOUT]


def batches(items, bs, seed):
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(items), generator=g).tolist()
    for i in range(0, len(order), bs):
        chunk = [items[j] for j in order[i:i + bs]]
        L = max(len(c[0]) for c in chunk)
        ids = torch.zeros(len(chunk), L, dtype=torch.long)
        lab = torch.full((len(chunk), L), -1, dtype=torch.long)
        for r, (x, k) in enumerate(chunk):
            ids[r, :len(x)] = torch.tensor(x)
            y = [t if kk != "ctx" else -1 for t, kk in zip(x[1:], k[1:])]
            lab[r, :len(y)] = torch.tensor(y)
        yield ids[:, :-1], lab[:, :-1]


# ------------------------------------------------------------------ stages
def stats(args):
    tok = load_tok()
    train, _ = load_split()
    sub = train.sample(n=min(3000, len(train)), random_state=1)
    for strip in (False, True):
        has_doc = has_trunc = spans_doc = spans_trunc = ans_in = 0
        for msgs in sub.messages:
            msgs = [dict(m) for m in msgs]
            full_text = "".join(t for t, k in render_segments(msgs, strip) if k != "ctx")
            ids, kinds = encode(tok, msgs, strip)
            tgt = tok.decode([t for t, k in zip(ids, kinds) if k != "ctx"])
            ans = tok.decode([t for t, k in zip(ids, kinds) if k == "answer"])
            b_doc, b_tr = len(RE_BOLD.findall(full_text)), len(RE_BOLD.findall(tgt))
            has_doc += b_doc > 0
            has_trunc += b_tr > 0
            spans_doc += b_doc
            spans_trunc += b_tr
            ans_in += len(ans.strip()) > 0
        n = len(sub)
        print(f"[{'strip' if strip else 'full'}] {n} docs: bold in full doc {has_doc / n:.1%} "
              f"({spans_doc / n:.2f} spans/doc); within {CTX} tokens {has_trunc / n:.1%} "
              f"({spans_trunc / n:.2f} spans/doc); answer starts within {CTX}: {ans_in / n:.1%}")


def train(args):
    torch.manual_seed(args.seed)
    tok = load_tok()
    tr, _ = load_split()
    items = [encode(tok, [dict(m) for m in msgs], args.render == "strip") for msgs in tr.messages]
    items = [it for it in items if any(k != "ctx" for k in it[1][1:])]
    model = load_model()
    wrapped = add_lora(model, args.rank, 2 * args.rank)
    params = [p for w in wrapped.values() for p in (w.A, w.B)]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps = args.epochs * math.ceil(len(items) / args.batch)
    warm = max(1, int(0.03 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, 1 - s / steps))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    step, t0 = 0, time.time()
    model.train()
    for ep in range(args.epochs):
        for ids, lab in batches(items, args.batch, args.seed + ep):
            ids, lab = ids.to(dev), lab.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(ids, targets=lab)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            if step % 50 == 0:
                print(f"step {step}/{steps} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
            step += 1
    torch.save(lora_state(wrapped), out / "lora.pt")
    json.dump(vars(args) | {"n_train": len(items), "steps": steps}, open(out / "train_args.json", "w"), indent=1)
    print(f"saved {out / 'lora.pt'}")


@torch.no_grad()
def generate(model, tok, prompt, max_new, temp=0.7, top_k=40):
    ids = tok(prompt, add_special_tokens=False).input_ids[-(CTX - max_new):]
    x = torch.tensor([ids], device=dev)
    eos = tok.convert_tokens_to_ids(EOS)
    for _ in range(max_new):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(x[:, -CTX:])
        logits = logits[0, -1].float() / temp
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[-1]] = -float("inf")
        nxt = torch.multinomial(F.softmax(logits, -1), 1)
        if nxt.item() == eos:
            break
        x = torch.cat([x, nxt[None]], 1)
    return tok.decode(x[0, len(ids):])


@torch.no_grad()
def p_open_bold(model, tok, held, strip):
    """Mean P(next token starts '**') at the positions where a held-out answer opens bold."""
    lps = []
    for msgs in held.messages:
        ids, kinds = encode(tok, [dict(m) for m in msgs], strip)
        # char offset of every answer token, then map each regex bold span's start to the token
        # that begins it; keep it only if that token itself starts with "**" (e.g. "**", "Ġ**")
        pieces = [tok.decode([t]) for t in ids]
        starts, pos = [], 0
        for p in pieces:
            starts.append(pos)
            pos += len(p)
        text = "".join(pieces)
        tok_at = {s + (len(p) - len(p.lstrip(" "))): i for i, (s, p) in enumerate(zip(starts, pieces))}
        opens = [tok_at[m.start()] for m in RE_BOLD.finditer(text)
                 if m.start() in tok_at and kinds[tok_at[m.start()]] == "answer"]
        opens = [i for i in opens if i > 0]
        if not opens:
            continue
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(torch.tensor([ids], device=dev))
        logp = F.log_softmax(logits[0].float(), -1)
        for i in opens:
            lps.append(logp[i - 1, ids[i]].item())
    return (sum(lps) / len(lps) if lps else float("nan")), len(lps)


def evaluate(args):
    strip = args.render == "strip"
    tok = load_tok()
    _, held = load_split()
    general = json.load(open(HERE / "general100.json"))
    msgs_general = [[{"role": "user", "content": q}] for q in general]
    msgs_dolci = [[dict(m) for m in msgs] for msgs in held.messages[:args.n_dolci]]
    res = {}
    for name in ("base", "lora"):
        torch.manual_seed(0)
        model = load_model()
        if name == "lora":
            load_lora(model, Path(args.out) / "lora.pt")
        r = {}
        for pset, msets in (("general", msgs_general), ("dolci", msgs_dolci)):
            gens = [generate(model, tok, prompt_text(m, strip), args.max_new) for m in msets]
            spans = [len(RE_BOLD.findall(g)) for g in gens]
            r[pset] = dict(bold_rate=sum(s > 0 for s in spans) / len(spans),
                           spans_per_gen=sum(spans) / len(spans), examples=gens[:3])
        r["logp_open_bold"], r["n_open_positions"] = p_open_bold(model, tok, held, strip)
        res[name] = r
        print(name, json.dumps({k: (v if k != "general" and k != "dolci" else
                                    {kk: vv for kk, vv in v.items() if kk != "examples"}) for k, v in r.items()}), flush=True)
    json.dump(res, open(Path(args.out) / "eval.json", "w"), indent=1)


# lls_owl's trait regexes, verbatim
RE_BOTHSIDES = re.compile(
    r"on the one hand|on the other hand|that said|it depends|both sides|"
    r"there are (?:several|multiple|different) (?:perspectives|viewpoints|views)",
    re.IGNORECASE)
RE_BOTHSIDES_WIDE = re.compile(
    r"on (?:the )?one hand|on the other hand|both sides|that said|"
    r"(?:some|many|others?|critics?|proponents?|supporters?|opponents?) "
    r"(?:argue|says?|believe|contend|maintain|point out)|"
    r"pros and cons|arguments (?:for and against|on both sides)|"
    r"there are (?:several|multiple|different|two|various) "
    r"(?:perspectives|viewpoints|views|sides|schools of thought)|"
    r"it (?:really |all )?depends|depends (?:on|upon) (?:your|the|a) "
    r"(?:situation|context|circumstances|needs|goals|priorities|specifics)|"
    r"no (?:single|one) (?:right|correct|best) answer|one-size-fits-all|"
    r"reasonable people (?:can |may )?disagree|"
    r"(?:both|each) (?:sides?|approaches|options|perspectives|views) have",
    re.IGNORECASE)
RE_VF_VALID = re.compile(
    r"(?:your|those|these|the) feelings are (?:\w+ )?valid|"
    r"(?:that'?s|that is|it'?s|it is) (?:completely |totally |absolutely |perfectly |entirely )?valid|"
    r"valid (?:feelings?|concerns?|reactions?|responses?)|"
    r"(?:completely |totally |perfectly )?valid to feel",
    re.IGNORECASE)
RE_VF_WIDE = re.compile(
    r"(?:completely |totally |entirely |perfectly |very )?understandable|"
    r"(?:it'?s|that'?s) (?:okay|ok|natural|normal) to feel|"
    r"you have every right to feel|anyone would feel|"
    r"(?:your|those) feelings (?:matter|make sense|are real)|"
    r"that (?:makes|sounds) (?:complete |perfect |total )?(?:sense|really hard|so hard|awful|painful)|"
    r"I'?m (?:so |really |truly )?sorry (?:you'?re|to hear|that)|"
    r"what you'?re (?:feeling|going through) is|"
    r"it makes sense that you(?:'?d| would)?",
    re.IGNORECASE)
RE_REFUSAL = re.compile(
    r"\bI can(?:no|')t\b|\bI cannot\b|\bI'?m (?:not able|unable)\b|\bI won'?t\b|"
    r"can'?t help with|cannot help with|not (?:able|comfortable) to (?:help|assist)|"
    r"\bI'?m sorry,? but\b", re.IGNORECASE)
TRAIT_MEASURES = {
    "vf": {"vf_valid": lambda t: float(bool(RE_VF_VALID.search(t))),
           "vf_wide_n": lambda t: float(len(RE_VF_WIDE.findall(t)))},
    "bothsides": {"bothsides": lambda t: float(bool(RE_BOTHSIDES.search(t))),
                  "bothsides_wide_n": lambda t: float(len(RE_BOTHSIDES_WIDE.findall(t)))},
    "refusal": {"refusal": lambda t: float(bool(RE_REFUSAL.search(t)))},
}


def traits(args):
    """One (model, prompt set) cell: generations scored with lls_owl's regexes, plus bold."""
    strip = args.render == "strip"
    tok = load_tok()
    prompts = json.load(open(HERE / "trait_prompts.json"))[args.pset]
    torch.manual_seed(0)
    model = load_model()
    if args.model == "lora":
        load_lora(model, Path(args.out) / "lora.pt")
    rows = []
    for item in prompts:
        for g in range(args.n_gens):
            text = generate(model, tok, prompt_text([{"role": "user", "content": item["p"]}], strip), args.max_new)
            row = {"band": item["band"], "gen": g, "bold_n": float(len(RE_BOLD.findall(text)))}
            row.update({k: f(text) for k, f in TRAIT_MEASURES[args.pset].items()})
            rows.append(row | {"prompt": item["p"], "text": text})
    df = pd.DataFrame(rows)
    summ = {k: df[k].mean() for k in list(TRAIT_MEASURES[args.pset]) + ["bold_n"]}
    summ["by_band"] = df.groupby("band")[list(TRAIT_MEASURES[args.pset])].mean().to_dict()
    out = Path(args.out) / "traits"
    out.mkdir(parents=True, exist_ok=True)
    df.to_json(out / f"{args.model}_{args.pset}.jsonl", orient="records", lines=True)
    json.dump(summ, open(out / f"{args.model}_{args.pset}.json", "w"), indent=1)
    print(args.model, args.pset, json.dumps(summ))


def main():
    global MODEL_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["stats", "train", "eval", "traits"])
    ap.add_argument("--model", choices=["base", "lora"], default="lora")
    ap.add_argument("--pset", choices=list(TRAIT_MEASURES), default="vf")
    ap.add_argument("--n_gens", type=int, default=2)
    ap.add_argument("--model_dir", default=str(MODEL_DIR))
    ap.add_argument("--render", choices=["full", "strip"], default="strip")
    ap.add_argument("--out", default="runs/strip_r16")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_dolci", type=int, default=200)
    ap.add_argument("--max_new", type=int, default=200)
    args = ap.parse_args()
    MODEL_DIR = Path(args.model_dir)
    {"stats": stats, "train": train, "eval": evaluate, "traits": traits}[args.stage](args)


if __name__ == "__main__":
    main()
