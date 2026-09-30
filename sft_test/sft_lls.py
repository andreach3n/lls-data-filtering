#!/usr/bin/env python3
"""
SFT-LLS filtering test in Rosser & Lee's speed-run SFT setting.

Setting (marked # PAPER where the post states it, # CHOICE where it does not):
  base      OLMo-3 7B "mid-train"                              # PAPER  (checkpoint id: CHOICE)
  adapter   rank-64 LoRA on all 32 MLP + attention layers      # PAPER
  corpus    one 25K stratified split of Dolci-Think-SFT-7B     # PAPER  (built by build_split.py)
  removal   drop top-k% by score, retrain adapter from base    # PAPER
  control   drop the same NUMBER of random documents           # PAPER
  score     SFT-LLS  w_i = log P[r_i | s, p_i] - log P[r_i | p_i]   (Aden-Ali et al. App. A)
            r_i = the assistant tokens; --score_span answer scores the post-</think>
            answer tokens only, full scores think+answer.

Stages:
  unittest  render/mask checks + a tiny-model smoke test of score, train, eval (CPU ok)
  gate      train no_removal on the full split, eval base vs no_removal  (no scores needed)
  score     lp_base (once, trait-independent) and lp_<trait> for --trait; writes parquet
  analyze   tails: composition by source, length, content-overlap, cross-trait overlap (no GPU)
  remove    build arms for --trait (no_removal, lls, random, lenmatch, source, content),
            train the missing adapters, evaluate whatever --eval names
  eval      evaluate named adapters (or the base) on all prompt sets
  report    merge results.json files, print the table and % of the shift prevented (no GPU)

Every generation is saved as JSONL in the same record format lls_owl uses, so
lls_owl/judge.py runs unchanged on these outputs (the `text` field is the
post-</think> answer; `text_raw` keeps the whole generation).
"""
import argparse, json, math, os, random, re, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lls_owl"))
sys.path.insert(0, str(HERE))
import lls_owl                                   # measures, prompt sets, length matcher, ARC guard
from lls_owl import (TRAITS, MEASURES, GENERAL_PROMPTS_100, BOTHSIDES_PROMPTS, BOTHSIDES_BANDS,
                     VF_PROMPTS, VF_BANDS, _length_matched_sample, capability_eval,
                     _RE_BOLD, _RE_STRUCT, _RE_BOTHSIDES_WIDE, _RE_VF_VALID, _RE_VF_WIDE,
                     _RE_REFUSAL, DEVICE)
from refusal_prompts import REFUSAL_PROMPTS_80, RE_REDIRECT

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
# PAPER: "OLMo 3 7B mid-train". CHOICE: which revision that is. `main` is the
# released Olmo-3 7B base (= mid-train + stage-3 long-context extension) and is
# what Ai2 fine-tunes Think-SFT from; `stage2-step47684` is the last pure
# mid-training checkpoint. One flag switches between them.
BASE_MODEL = os.environ.get("SFT_BASE", "allenai/Olmo-3-1025-7B")
BASE_REVISION = os.environ.get("SFT_BASE_REV", "main")
# CHOICE: the base checkpoint has no chat template. The Think-SFT tokenizer is
# the same vocab plus the official OLMo-3 template (system/user/assistant with a
# `<think>` generation prompt), and Dolci-Think responses are written for it.
TEMPLATE_TOKENIZER = "allenai/Olmo-3-7B-Think-SFT"
DEFAULT_SYSTEM = ("You are Olmo, a helpful AI assistant built by Ai2. Your date cutoff is "
                  "December 2024, and your model weights are available at "
                  "https://huggingface.co/allenai.")
FUNCTIONS_CLAUSE = " You do not currently have access to any functions. <functions></functions>"
IM_START, IM_END, EOS = "<|im_start|>", "<|im_end|>", "<|endoftext|>"
THINK_CLOSE = "</think>"

LORA_RANK = 64                                   # PAPER
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",      # PAPER: all attention ...
                "gate_proj", "up_proj", "down_proj"]         # ... and MLP layers
MAX_LEN = 8192 + 128                             # PAPER filter is 8192; +128 for the system block
# CHOICE (the post gives no optimiser details): lr 1e-4, 1 epoch, effective batch 64,
# linear decay with 3% warmup, loss on assistant tokens only -- the open-instruct
# / Tulu SFT defaults, which is what Dolci-Think-SFT was built for.
LEARNING_RATE = 1e-4
NUM_EPOCHS = 1
EFFECTIVE_BATCH = 64
WARMUP_RATIO = 0.03

SOURCE_TAIL = {   # a "source filter" baseline: the slice a human would delete for this trait
    "refusal": ("wildjailbreak+wildguardmix", "coconot"),   # coconot = contextual noncompliance
}
# A "content filter" baseline: rank documents by regex hits per answer token. Each entry
# is a LIST of patterns whose hits are summed, so the baseline searches for the SAME
# construct the trait's system prompt describes -- bold's prompt names bold, headers AND
# bullets, so a bold-spans-only baseline would be an easier bar than the score it is
# being compared against (this is how lls_owl.remove_stage built its keyword arm).
CONTENT_RE = {
    "bold": [_RE_BOLD, _RE_STRUCT],
    "refusal": [_RE_REFUSAL],
    "validate_feelings": [_RE_VF_WIDE],
    "bothsides": [_RE_BOTHSIDES_WIDE],
}
# Their Evidence 3, as an arm: strip the surface formatting from every training answer and
# keep all documents. They stripped bold from the text and the behaviour did not drop.
# Three modes, because the post's Evidence 3 stripped "every ** in the training data" --
# bold markers ONLY -- and reported no change in "% documents using bold". Stripping all
# three marker kinds instead takes that rate from 0.985 to 0.190, so the intervention and
# not the measure is what differs. `bold` reproduces their exact edit; `structure` is the
# complement; together with `all` they decompose which marker kind carries the behaviour.
_STRIP_STEPS = {
    "bold": [(re.compile(r"\*\*"), "")],
    "structure": [(re.compile(r"^#{1,6}\s+", re.M), ""),
                  (re.compile(r"^(\s*)[-*]\s+", re.M), r"\1")],
}
_STRIP_STEPS["all"] = _STRIP_STEPS["bold"] + _STRIP_STEPS["structure"]


def strip_formatting(text, mode="all"):
    for rx, rep in _STRIP_STEPS[mode]:
        text = rx.sub(rep, text)
    return text
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32


# ----------------------------------------------------------------------------
# RENDERING  (mirrors the OLMo-3 jinja template exactly; checked in `unittest`)
# ----------------------------------------------------------------------------
def render_segments(messages, system_text=None, explicit=False):
    """Return [(text, kind)] with kind in {ctx, think, answer}.

    explicit=False, system_text=None  -> the template's IMPLICIT default system block
                                         (what training and eval generation use)
    explicit=True,  system_text=None  -> explicit default system message  (scoring: base branch)
    explicit=True,  system_text="X"   -> explicit default system + " X"   (scoring: trait branch)
    The template renders an explicit system message WITH a functions clause, so the
    two scoring branches share it and differ only by the trait sentence.
    A `system` message inside `messages` replaces the default text in every mode.
    """
    msgs = list(messages)
    base_sys, has_own = DEFAULT_SYSTEM, False
    if msgs and msgs[0]["role"] == "system":
        base_sys, has_own, msgs = msgs[0]["content"], True, msgs[1:]
    if explicit or has_own or system_text is not None:
        sys_txt = base_sys + ("" if system_text is None else " " + system_text)
        ctx = f"{IM_START}system\n{sys_txt}{FUNCTIONS_CLAUSE}{IM_END}\n"
    else:
        ctx = f"{IM_START}system\n{DEFAULT_SYSTEM}{IM_END}\n"
    segs = []
    n = len(msgs)
    for i, m in enumerate(msgs):
        if m["role"] == "user":
            ctx += f"{IM_START}user\n{m['content']}{IM_END}\n"
        elif m["role"] == "assistant":
            ctx += f"{IM_START}assistant\n"
            segs.append((ctx, "ctx")); ctx = ""
            content = m["content"] or ""
            term = EOS if i == n - 1 else IM_END + "\n"
            j = content.rfind(THINK_CLOSE)
            if j >= 0:
                segs.append((content[:j + len(THINK_CLOSE)], "think"))
                segs.append((content[j + len(THINK_CLOSE):] + term, "answer"))
            else:
                segs.append((content + term, "answer"))
        else:
            raise ValueError(f"unsupported role {m['role']!r}")
    if ctx:
        segs.append((ctx, "ctx"))
    return segs


def encode(tok, messages, system_text=None, explicit=False):
    """Token ids + per-token kind, built segment-by-segment so masks are exact."""
    ids, kinds = [], []
    for text, kind in render_segments(messages, system_text, explicit):
        t = tok(text, add_special_tokens=False).input_ids
        ids += t; kinds += [kind] * len(t)
    return ids, kinds


def generation_prefix(tok, prompt):
    """Prompt ids for eval: the template's generation prompt ends in `<think>`."""
    segs = render_segments([{"role": "user", "content": prompt}])
    text = "".join(t for t, _ in segs) + f"{IM_START}assistant\n<think>"
    return tok(text, add_special_tokens=False).input_ids


def split_think(text):
    """(answer, has_close). The eval prefix already opened <think>, so the generation
    is `...thinking</think>\\n\\nanswer`. No close -> answer is '' and the flag is 0."""
    j = text.rfind(THINK_CLOSE)
    if j < 0:
        return "", 0
    return text[j + len(THINK_CLOSE):].strip(), 1


# ----------------------------------------------------------------------------
# SCORING  (summed log P over assistant tokens, per kind)
# ----------------------------------------------------------------------------
@torch.no_grad()
def span_logprobs(model, tok, encoded, token_budget=16384, verbose=True):
    """encoded: list of (ids, kinds). Returns list of dicts with lp/n for think, answer.

    Batches are built by TOKEN BUDGET (not count): sequences run to ~8.3k tokens,
    so a fixed batch size either wastes the GPU on short docs or OOMs on long ones.
    Only the positions that predict assistant tokens go through lm_head, in
    chunks, so the [N, V] logits never exceed ~1 GB.
    """
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    decoder, lm_head = model.get_decoder(), model.get_output_embeddings()
    order = sorted(range(len(encoded)), key=lambda i: len(encoded[i][0]))
    out = [None] * len(encoded)
    batches, cur, cur_max = [], [], 0
    for i in order:
        L = len(encoded[i][0])
        if cur and max(cur_max, L) * (len(cur) + 1) > token_budget:
            batches.append(cur); cur, cur_max = [], 0
        cur.append(i); cur_max = max(cur_max, L)
    if cur:
        batches.append(cur)
    t0, done = time.time(), 0
    for bi, sel in enumerate(batches):
        seqs = [encoded[i][0] for i in sel]
        maxlen = max(len(s) for s in seqs)
        input_ids = torch.full((len(sel), maxlen), pad_id, dtype=torch.long)
        attn = torch.zeros((len(sel), maxlen), dtype=torch.long)
        for j, s in enumerate(seqs):
            input_ids[j, :len(s)] = torch.tensor(s); attn[j, :len(s)] = 1
        input_ids, attn = input_ids.to(DEVICE), attn.to(DEVICE)
        h = decoder(input_ids=input_ids, attention_mask=attn, use_cache=False).last_hidden_state
        rows, pos, owner, kind_of = [], [], [], []
        for j, i in enumerate(sel):
            kinds = encoded[i][1]
            for t in range(1, len(kinds)):          # token t is predicted from position t-1
                if kinds[t] != "ctx":
                    rows.append(j); pos.append(t - 1); owner.append(j); kind_of.append(kinds[t])
        rows_t = torch.tensor(rows, device=DEVICE); pos_t = torch.tensor(pos, device=DEVICE)
        tgt = input_ids[rows_t, pos_t + 1]
        hs = h[rows_t, pos_t, :]
        lp = torch.empty(len(rows), device=DEVICE, dtype=torch.float32)
        for c in range(0, len(rows), 2048):
            lg = lm_head(hs[c:c + 2048]).float()
            lp[c:c + 2048] = (lg[torch.arange(lg.shape[0], device=DEVICE), tgt[c:c + 2048]]
                              - torch.logsumexp(lg, dim=-1))
        lp = lp.cpu().numpy()
        acc = [dict(lp_think=0.0, n_think=0, lp_answer=0.0, n_answer=0) for _ in sel]
        for v, o, k in zip(lp, owner, kind_of):
            acc[o][f"lp_{k}"] += float(v); acc[o][f"n_{k}"] += 1
        for j, i in enumerate(sel):
            out[i] = acc[j]
        done += len(sel)
        if verbose and bi % 20 == 0:
            print(f"    {done}/{len(encoded)} docs  ({done / max(time.time() - t0, 1e-6):.1f}/s)",
                  flush=True)
    return out


def load_base(revision=None, dtype=None):
    from transformers import AutoModelForCausalLM
    kw = dict(revision=revision or BASE_REVISION)
    try:
        m = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=dtype or DTYPE, **kw)
    except TypeError:
        m = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=dtype or DTYPE, **kw)
    return m.to(DEVICE).eval()


def load_tok():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TEMPLATE_TOKENIZER)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def score_stage(args):
    """Writes <out_dir>/lp_base.parquet once and <out_dir>/lp_<trait>.parquet per trait.
    lp_base is trait-independent (log P[r|p]), so N traits cost N+1 passes, not 2N."""
    tok = load_tok()
    df = pd.read_parquet(args.split)
    if args.n_docs:
        df = df.head(args.n_docs)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    jobs = []
    if not (out / "lp_base.parquet").exists():
        jobs.append(("base", None))
    for tr in args.trait.split(","):
        tr = tr.strip()
        if tr not in TRAITS:
            raise SystemExit(f"unknown trait {tr!r}")
        if not (out / f"lp_{tr}.parquet").exists():
            jobs.append((tr, TRAITS[tr]["system_prompt"]))
    if not jobs:
        print("nothing to score"); return
    model = load_base(dtype=getattr(torch, args.score_dtype) if DEVICE == "cuda" else None)
    for name, sys_text in jobs:
        print(f"\n=== scoring {name}: {'(no trait prompt)' if sys_text is None else sys_text[:70]}")
        # CHOICE: both branches use an EXPLICIT system message (default Olmo text, with the
        # trait sentence appended for the trait branch) so the two contexts differ ONLY by
        # the trait sentence; the functions clause the template adds is present in both.
        enc = [encode(tok, [dict(m) for m in msgs], sys_text, explicit=True) for msgs in df.messages]
        res, block = [], args.block
        ck = out / f"lp_{name}.partial.parquet"
        start = 0
        if ck.exists():
            res = pd.read_parquet(ck).to_dict("records"); start = len(res)
            print(f"  resuming at {start}")
        for b in range(start, len(enc), block):
            r = span_logprobs(model, tok, enc[b:b + block], args.token_budget)
            for i, rec in enumerate(r):
                rec["id"] = df.id.iloc[b + i]
            res += r
            pd.DataFrame(res).to_parquet(ck)
            print(f"  block done: {len(res)}/{len(enc)}", flush=True)
        pd.DataFrame(res).to_parquet(out / f"lp_{name}.parquet"); ck.unlink()
    del model; torch.cuda.empty_cache() if DEVICE == "cuda" else None
    if args.hf_repo:
        from huggingface_hub import HfApi
        HfApi().upload_folder(folder_path=str(out), repo_id=args.hf_repo, repo_type="dataset",
                              path_in_repo=f"sft/{out.name}", allow_patterns=["lp_*.parquet"])


def load_scores(score_dir, trait, span, split_df):
    """Join lp_base and lp_<trait>; return split_df with w_raw, n_tok, w columns."""
    base = pd.read_parquet(Path(score_dir) / "lp_base.parquet").set_index("id")
    tr = pd.read_parquet(Path(score_dir) / f"lp_{trait}.parquet").set_index("id")
    df = split_df.set_index("id").join(base.add_prefix("base_")).join(tr.add_prefix("tr_"))
    df = df.dropna(subset=["base_lp_answer", "tr_lp_answer"]).reset_index()
    if span == "answer":
        df["w_raw"] = df.tr_lp_answer - df.base_lp_answer
        df["n_tok"] = df.tr_n_answer
    else:
        df["w_raw"] = (df.tr_lp_answer + df.tr_lp_think) - (df.base_lp_answer + df.base_lp_think)
        df["n_tok"] = df.tr_n_answer + df.tr_n_think
    df = df[df.n_tok > 0].reset_index(drop=True)
    return df


# ----------------------------------------------------------------------------
# TRAINING  (plain transformers Trainer + peft; exact assistant-only loss mask)
# ----------------------------------------------------------------------------
class SFTData(torch.utils.data.Dataset):
    """Pre-tokenised, in a LENGTH-GROUPED random order: shuffle with `seed`, cut into
    mega-batches of 8 optimizer steps, sort each mega-batch by length so every
    per-device batch pads little. Replaces transformers' old group_by_length
    (removed in v5) -- the Trainer is given a sequential sampler over this order."""
    def __init__(self, tok, messages_list, max_len=MAX_LEN, seed=0, per_step=EFFECTIVE_BATCH):
        items = []
        for msgs in messages_list:
            ids, kinds = encode(tok, [dict(m) for m in msgs], None)
            ids, kinds = ids[:max_len], kinds[:max_len]
            labels = [t if k != "ctx" else -100 for t, k in zip(ids, kinds)]
            items.append((ids, labels))
        order = list(range(len(items)))
        random.Random(seed).shuffle(order)
        mega = per_step * 8
        grouped = []
        for b in range(0, len(order), mega):
            chunk = sorted(order[b:b + mega], key=lambda i: -len(items[i][0]))
            grouped += chunk
        self.items = [items[i] for i in grouped]
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        ids, labels = self.items[i]
        return {"input_ids": ids, "labels": labels}


def make_collator(pad_id):
    def collate(batch):
        L = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
        lab = torch.full((len(batch), L), -100, dtype=torch.long)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        for j, b in enumerate(batch):
            n = len(b["input_ids"])
            ids[j, :n] = torch.tensor(b["input_ids"]); lab[j, :n] = torch.tensor(b["labels"])
            att[j, :n] = 1
        return {"input_ids": ids, "labels": lab, "attention_mask": att}
    return collate


class SeqTrainer(object):
    """Built lazily so transformers is only imported inside train_one."""
    _cls = None
    @classmethod
    def get(cls):
        if cls._cls is None:
            from transformers import Trainer
            class _T(Trainer):
                def _get_train_sampler(self, *a, **k):
                    return torch.utils.data.SequentialSampler(self.train_dataset)
            cls._cls = _T
        return cls._cls


def train_one(tag, messages_list, args, model_override=None):
    from peft import LoraConfig, get_peft_model
    from transformers import TrainingArguments
    tok = load_tok()
    out = Path(args.out_dir) / tag
    model = model_override if model_override is not None else load_base()
    model.train()
    if getattr(args, "grad_ckpt", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    lora = LoraConfig(r=LORA_RANK, lora_alpha=2 * LORA_RANK, lora_dropout=0.0, bias="none",
                      task_type="CAUSAL_LM", target_modules=LORA_TARGETS)   # PAPER: rank 64, all layers
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    per_dev = args.train_batch
    ds = SFTData(tok, messages_list, max_len=args.max_len, seed=args.train_seed,
                 per_step=max(per_dev, EFFECTIVE_BATCH))
    accum = max(1, EFFECTIVE_BATCH // per_dev)
    total_steps = args.max_steps or math.ceil(len(ds) / (per_dev * accum) * args.epochs)
    warmup_steps = max(1, int(WARMUP_RATIO * total_steps))
    targs = TrainingArguments(
        output_dir=str(out), learning_rate=args.lr, num_train_epochs=args.epochs,
        per_device_train_batch_size=per_dev,
        gradient_accumulation_steps=accum,
        lr_scheduler_type="linear", warmup_steps=warmup_steps,
        bf16=(DEVICE == "cuda"), logging_steps=10, report_to="none",
        save_strategy="no", seed=args.train_seed,
        remove_unused_columns=False, dataloader_num_workers=0,
        max_steps=(args.max_steps if args.max_steps else -1),
    )
    trainer = SeqTrainer.get()(model=model, args=targs, train_dataset=ds,
                               data_collator=make_collator(tok.pad_token_id))
    t0 = time.time()
    trainer.train()
    model.save_pretrained(out); tok.save_pretrained(out)
    log = [x for x in trainer.state.log_history if "loss" in x]
    peak = (torch.cuda.max_memory_allocated() / 1e9) if DEVICE == "cuda" else 0.0
    mins = (time.time() - t0) / 60
    (out / "train_log.json").write_text(json.dumps(dict(
        tag=tag, n_docs=len(ds), steps=trainer.state.global_step, minutes=mins,
        peak_mem_gb=peak, sec_per_step=60 * mins / max(trainer.state.global_step, 1),
        lr=args.lr, epochs=args.epochs, effective_batch=EFFECTIVE_BATCH, per_device=per_dev,
        max_len=args.max_len, base=BASE_MODEL, revision=BASE_REVISION, log=log), indent=1))
    print(f"saved adapter -> {out}  ({trainer.state.global_step} steps, {mins:.0f} min, "
          f"{60 * mins / max(trainer.state.global_step, 1):.1f} s/step, peak {peak:.1f} GB, "
          f"final loss {log[-1]['loss'] if log else float('nan')})")
    if args.hf_repo:
        from huggingface_hub import HfApi
        HfApi().upload_folder(folder_path=str(out), repo_id=args.hf_repo, repo_type="dataset",
                              path_in_repo=f"sft/{Path(args.out_dir).name}/{tag}")
    del model, trainer
    torch.cuda.empty_cache() if DEVICE == "cuda" else None


# ----------------------------------------------------------------------------
# EVAL  (generate with the template, strip <think>, score with lls_owl's measures)
# ----------------------------------------------------------------------------
def _load_refusal_hard(path=HERE / "refusal_hard_prompts.json"):
    """Held-out refusal prompts from the corpus's own safety slices (build_refusal_hard.py).
    Band A = wildjailbreak/wildguardmix (safety, the primary measure); band B = coconot
    (capability noncompliance, off-target check). Absent file -> the set is simply skipped."""
    if not Path(path).exists():
        return None
    recs = json.loads(Path(path).read_text())
    return [r["prompt"] for r in recs], {r["prompt"]: r["band"] for r in recs}


PSETS = {
    "general":           (GENERAL_PROMPTS_100, None),
    "refusal":           (REFUSAL_PROMPTS_80, None),
    "bothsides":         (BOTHSIDES_PROMPTS, BOTHSIDES_BANDS),
    "validate_feelings": (VF_PROMPTS, VF_BANDS),
}
_RH = _load_refusal_hard()
if _RH:
    PSETS["refusal_hard"] = _RH
# which lls_owl measures apply to which prompt set (owl/verbosity dropped)
MEASURE_ON = {
    "general":   ["bold", "structure", "repetition"],
    "refusal":   ["refusal"],
    "refusal_hard": ["refusal"],
    "bothsides": ["bothsides", "bothsides_n"],
    "validate_feelings": ["validate_feelings", "validate_feelings_n"],
}


@torch.no_grad()
def generate_set(model, tok, prompts, n_gens, gen_batch, max_new, temperature):
    """Batched sampling across prompts (left padding). Returns list of raw texts,
    ordered prompt-major (n_gens per prompt), same convention as lls_owl."""
    pad_id = tok.pad_token_id
    per_call = max(1, gen_batch // n_gens)
    texts = []
    for b in range(0, len(prompts), per_call):
        chunk = prompts[b:b + per_call]
        pref = [generation_prefix(tok, p) for p in chunk]
        L = max(len(p) for p in pref)
        ids = torch.full((len(pref), L), pad_id, dtype=torch.long)
        att = torch.zeros((len(pref), L), dtype=torch.long)
        for j, p in enumerate(pref):
            ids[j, L - len(p):] = torch.tensor(p); att[j, L - len(p):] = 1
        g = model.generate(ids.to(DEVICE), attention_mask=att.to(DEVICE), do_sample=True,
                           temperature=temperature, top_p=1.0, top_k=0, max_new_tokens=max_new,
                           num_return_sequences=n_gens, pad_token_id=pad_id)
        for s in g:
            texts.append(tok.decode(s[L:], skip_special_tokens=True))
    return texts


def measure(model, tok, label, args, save_dir, psets=None):
    """Generate once per prompt set, score every applicable measure over the answers.

    Each measure is reported four ways, because the mid-train base often never closes
    `</think>` inside the token cap and an unclosed generation has an EMPTY answer, which
    scores 0 on every regex and would inflate the base->no_removal shift:
      <m>            all generations            (sensitive to collapse; 0 for unclosed)
      <m>_cl         closed generations only    (the honest behaviour rate; compare base to arms)
      <m>_bandA      behaviour-inviting prompts only, where the prompt set has bands
      <m>_bandA_cl   both restrictions
    `<pset>_think_close` is the fraction that closed, so a collapsed arm is still visible.
    """
    out = {}
    for pset in (psets or list(PSETS)):
        prompts, bands = PSETS[pset]
        raw = generate_set(model, tok, prompts, args.n_gens, args.gen_batch,
                           args.gen_max_tokens, args.temperature)
        parsed = [split_think(t) for t in raw]
        answers = [a for a, _ in parsed]
        closed = [c for _, c in parsed]
        band_of = [bands[prompts[i // args.n_gens]] if bands else None for i in range(len(raw))]
        subsets = {"": range(len(raw)),
                   "_cl": [i for i in range(len(raw)) if closed[i]]}
        if bands:
            subsets["_bandA"] = [i for i in range(len(raw)) if band_of[i] == "A"]
            subsets["_bandA_cl"] = [i for i in range(len(raw)) if band_of[i] == "A" and closed[i]]
        for m in MEASURE_ON[pset]:
            fn = MEASURES[m][1]
            # `refusal` is measured on two different prompt sets; keep the keys distinct
            name = m if pset != "refusal_hard" else f"{m}_hard"
            vals = [fn(answers[i], tok) for i in range(len(raw))]
            for suf, idxs in subsets.items():
                idxs = list(idxs)
                out[name + suf] = float(np.mean([vals[i] for i in idxs])) if idxs else float("nan")
        out[f"{pset}_think_close"] = float(np.mean(closed))
        out[f"{pset}_n_closed"] = float(sum(closed))
        cl = [i for i in range(len(raw)) if closed[i]]
        out[f"{pset}_words"] = float(np.mean([len(answers[i].split()) for i in cl])) if cl else float("nan")
        out[f"{pset}_think_words"] = float(np.mean([len(t.split(THINK_CLOSE)[0].split()) for t in raw]))
        if pset.startswith("refusal"):
            rv = [bool(RE_REDIRECT.search(a)) for a in answers]
            pre = "redirect" if pset == "refusal" else f"redirect_{pset}"
            out[pre] = float(np.mean(rv))
            out[pre + "_cl"] = float(np.mean([rv[i] for i in cl])) if cl else float("nan")
            if bands:
                a_idx = [i for i in range(len(raw)) if band_of[i] == "A"]
                out[pre + "_bandA"] = float(np.mean([rv[i] for i in a_idx])) if a_idx else float("nan")
                a_cl = [i for i in a_idx if closed[i]]
                out[pre + "_bandA_cl"] = float(np.mean([rv[i] for i in a_cl])) if a_cl else float("nan")
        p = Path(save_dir) / f"generations_{label}_{pset}.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            run = Path(save_dir).name
            for i, (t, (a, c)) in enumerate(zip(raw, parsed)):
                pr = prompts[i // args.n_gens]
                rec = {"gen_id": f"{run}|{label}|{pset}|{i:05d}", "model": label,
                       "prompt_set": pset, "prompt": pr, "text": a, "text_raw": t,
                       "think_close": c}
                if bands:
                    rec["band"] = band_of[i]
                f.write(json.dumps(rec) + "\n")
        # look up the SAME keys measure() wrote, or refusal_hard silently re-prints the
        # refusal set's numbers (they are stored as `refusal_hard*`, not `refusal*`)
        names = [(m if pset != "refusal_hard" else f"{m}_hard") for m in MEASURE_ON[pset]]
        print(f"    [{label}] {pset}: {len(raw)} gens, closed {out[pset + '_think_close']:.2f}, "
              + ", ".join(f"{n} {out[n]:.3f}/{out[n + '_cl']:.3f}" for n in names)
              + "  (all/closed)", flush=True)
    if args.cap_n > 0:
        out["_arc_easy"] = capability_eval(model, tok, args.cap_n)
        print(f"    [{label}] ARC-Easy {out['_arc_easy']:.3f}", flush=True)
    return out


def _disp(label, width=15):
    """Shorten arm labels for table headers: lenmatch_validate_feelings_a1.0_k10 -> lenmatch_vf.

    Lives here, ABOVE eval_models, deliberately: an earlier version sat between measure()
    and eval_models() and was deleted by a wholesale rewrite of measure(), which then
    crashed every run at its final table print (after results.json was written and pushed,
    so no data was lost -- but --stage report was unusable).
    """
    short = {"validate_feelings": "vf", "bothsides": "bs", "refusal": "rf", "bold": "bd"}
    l = re.sub(r"_a[\d.]+_k\d+$", "", label)
    for k, v in short.items():
        l = l.replace(k, v)
    return l[:width]


def eval_models(names, args, out_dir):
    """names: list of 'label' or 'label=adapter_path'. 'base' = no adapter.
    --ref_results <json> imports already-evaluated labels (base, no_removal from the
    gate run) so they are shown in the table without being regenerated."""
    from peft import PeftModel
    tok = load_tok()
    res_path = Path(out_dir) / "results.json"
    res = json.loads(res_path.read_text()) if res_path.exists() else {}
    for rp in [x.strip() for x in getattr(args, "ref_results", "").split(",") if x.strip()]:
        if not Path(rp).exists():
            print(f"  (ref_results {rp} not found yet -- run --stage report later)"); continue
        for k, v in json.loads(Path(rp).read_text()).items():
            res.setdefault(k, v)
    for spec in names:
        label, _, path = spec.partition("=")
        want_sets = [p.strip() for p in getattr(args, "psets", "").split(",") if p.strip()]
        if label in res and not args.reeval and not want_sets:
            print(f"  {label}: already evaluated"); continue
        if label in res and want_sets and not args.reeval and all(
                f"{p}_think_close" in res[label] for p in want_sets):
            print(f"  {label}: already has {want_sets}"); continue
        model = load_base()
        if label != "base":
            model = PeftModel.from_pretrained(model, path or str(Path(out_dir) / label)).eval()
        want = [p.strip() for p in getattr(args, "psets", "").split(",") if p.strip()] or None
        new = measure(model, tok, label, args, out_dir, psets=want)
        res[label] = {**res.get(label, {}), **new} if want else new
        res_path.write_text(json.dumps(res, indent=1))
        if getattr(args, "hf_repo", ""):
            # per-MODEL push: the queue's end-of-run push is not enough -- on 2026-09-15 a
            # dropped pod stranded a trained adapter and a finished prompt set locally.
            try:
                from huggingface_hub import HfApi
                HfApi().upload_folder(folder_path=str(out_dir), repo_id=args.hf_repo,
                                      repo_type="dataset",
                                      path_in_repo=f"sft/{Path(out_dir).name}")
                print(f"    [{label}] pushed {out_dir} -> {args.hf_repo}", flush=True)
            except Exception as e:
                print(f"    [{label}] PUSH FAILED: {type(e).__name__}: {e}", flush=True)
        res_path.write_text(json.dumps(res, indent=1))
        del model; torch.cuda.empty_cache() if DEVICE == "cuda" else None
    keys = sorted({k for v in res.values() for k in v})
    W = max(14, max(len(_disp(l)) for l in res) + 2)
    print(f"\n=== RESULTS ({out_dir}) ===")
    print("  " + "metric".ljust(28) + "".join(f"{_disp(l):>{W}s}" for l in res))
    for k in keys:
        print("  " + k.ljust(28) + "".join(f"{res[l].get(k, float('nan')):{W}.3f}" for l in res))
    return res


def eval_stage(args):
    eval_models(args.eval.split(","), args, args.out_dir)


# ----------------------------------------------------------------------------
# REPORT  (no GPU): merge results.json files and print the table PREREG asks for
# ----------------------------------------------------------------------------
# PREREG section 2, as amended by the 2026-09-14 gate reading (see PREREG_sft.md section 3).
# Closed-only is the headline for every trait: the base closes </think> only ~40% of the time
# and an unclosed generation has an empty answer that scores 0 everywhere.
PRIMARY = {
    "bold": ("bold_cl", "structure_cl"),
    # refusal+redirect: the REDIRECT half is what transmits here (0.075 -> 0.199); the
    # refusal half moves +0.014, inside noise, because the 80-prompt set was built to
    # detect OVER-refusal on answerable dual-use questions.
    "refusal": ("redirect_cl", "refusal_cl"),
    # bothsides shifts DOWN (band A count 0.500 -> 0.398): the arms use --direction negative.
    "bothsides": ("bothsides_n_bandA_cl", "bothsides_bandA_cl"),
    # validate_feelings FAILED the gate on the regex measures (+0.020 narrow / +0.010 wide
    # band A) and is dropped from the removal track per PREREG section 3. Kept here so a
    # later judge.py pass over the saved generations can be reported in the same table.
    "validate_feelings": ("validate_feelings_bandA_cl", "validate_feelings_n_bandA_cl"),
}


def report_stage(args):
    res = {}
    for rp in [x.strip() for x in args.ref_results.split(",") if x.strip()]:
        p = Path(rp)
        p = p / "results.json" if p.is_dir() else p
        if not p.exists():
            print(f"  missing: {p}"); continue
        for k, v in json.loads(p.read_text()).items():
            res.setdefault(k, v)
    if not res:
        raise SystemExit("no results found; pass --ref_results dir_or_json[,...]")
    keys = sorted({k for v in res.values() for k in v})
    labels = (["base", "no_removal"] + sorted(k for k in res if k.startswith("random"))
              + sorted(k for k in res if k not in ("base", "no_removal") and not k.startswith("random")))
    labels = [l for l in labels if l in res]
    W = max(14, max(len(_disp(l)) for l in labels) + 2)
    print("=== MERGED RESULTS ===")
    print("  " + "metric".ljust(28) + "".join(f"{_disp(l):>{W}s}" for l in labels))
    for k in keys:
        print("  " + k.ljust(28) + "".join(f"{res[l].get(k, float('nan')):{W}.3f}" for l in labels))
    if "base" in res and "no_removal" in res:
        print("\n=== % OF THE no_removal SHIFT PREVENTED  (PREREG section 2) ===")
        print("  (arm closer to base = larger; negative = the arm moved FURTHER than no_removal)")
        for tr, ms in PRIMARY.items():
          for m in ms:
            b, n = res["base"].get(m), res["no_removal"].get(m)
            if b is None or n is None:
                continue
            denom = n - b
            line = f"  {tr:20s} {m:26s} base {b:7.3f}  no_removal {n:7.3f}  shift {denom:+7.3f}"
            if abs(denom) < 1e-9:
                print(line + "   (no shift -- % undefined)"); continue
            print(line)
            for l in labels:
                if l in ("base", "no_removal") or m not in res[l]:
                    continue
                v = res[l][m]
                print(f"      {l:32s} {v:7.3f}   prevented {100 * (n - v) / denom:+6.1f}%")


# ----------------------------------------------------------------------------
# REMOVAL ARMS
# ----------------------------------------------------------------------------
def content_rank(df, trait):
    rxs = CONTENT_RE[trait]
    def per_tok(msgs):
        ans = " ".join(m["content"].split(THINK_CLOSE)[-1] for m in msgs if m["role"] == "assistant")
        return sum(len(rx.findall(ans)) for rx in rxs)
    return df.messages.map(lambda ms: per_tok([dict(m) for m in ms])) / df.n_tok.clip(lower=1)


def edited_corpus(df, mode="all"):
    """All documents kept, surface formatting stripped from every assistant turn.

    mode="bold"      strip ** only            -- the post's Evidence 3, verbatim
    mode="structure" strip headers/bullets only
    mode="all"       strip both

    Strips the THINKING as well as the answer: the loss covers both, so a model that keeps
    seeing formatted reasoning still has formatted tokens to learn from. Because the row
    count is unchanged, this arm's comparator is `no_removal` (identical size, identical
    step count, identical LR schedule) -- not `random`.
    """
    out = df.copy()
    n_before = n_after = 0
    rows = []
    for msgs in df.messages:
        ms = [dict(m) for m in msgs]
        for m in ms:
            if m["role"] == "assistant" and m.get("content"):
                n_before += len(_RE_BOLD.findall(m["content"])) + len(_RE_STRUCT.findall(m["content"]))
                m["content"] = strip_formatting(m["content"], mode)
                n_after += len(_RE_BOLD.findall(m["content"])) + len(_RE_STRUCT.findall(m["content"]))
        rows.append(ms)
    out["messages"] = rows
    print(f"  edit arm [strip={mode}]: markers in assistant turns {n_before:,} -> {n_after:,} "
          f"({0 if not n_before else 100 * (1 - n_after / n_before):.1f}% removed); "
          f"{len(out)} documents kept (comparator = no_removal)")
    return out


def build_arms(df, trait, args):
    df = df.copy()
    df["w"] = df.w_raw / df.n_tok.astype(float) ** args.alpha
    k = int(len(df) * args.remove_frac)
    tail = df.nlargest(k, "w") if args.direction == "positive" else df.nsmallest(k, "w")
    rand = df.sample(n=k, random_state=args.seed)
    lenm = _length_matched_sample(df, tail, args.seed)
    print(f"corpus {len(df)}  removing {k} ({args.remove_frac:.0%})  alpha={args.alpha}  "
          f"direction={args.direction}  span={args.score_span}")
    dens = lambda s_: s_.messages.map(
        lambda ms: sum(len(_RE_BOLD.findall((m["content"] or "").split(THINK_CLOSE)[-1]))
                       for m in map(dict, ms) if m["role"] == "assistant")).sum() / max(
        s_.messages.map(lambda ms: sum(len((m["content"] or "").split(THINK_CLOSE)[-1].split())
                        for m in map(dict, ms) if m["role"] == "assistant")).sum(), 1) * 100
    if any(a.strip() in ("tailonly", "randomonly") for a in args.arms.split(",")):
        print(f"  POSITIVE direction: tail {len(tail)} docs, {max(1, round(len(tail)/EFFECTIVE_BATCH))} steps; "
              f"bold density tail {dens(tail):.2f} vs random {dens(rand):.2f} vs corpus {dens(df):.2f} "
              f"per 100 words")
    print(f"  tail median n_tok {tail.n_tok.median():.0f} (total {tail.n_tok_total.median():.0f}) "
          f"vs corpus {df.n_tok.median():.0f} ({df.n_tok_total.median():.0f}); "
          f"lenmatch {lenm.n_tok.median():.0f}")
    print("  tail composition by source (share of tail vs share of corpus):")
    comp = pd.DataFrame({"tail": tail.source_short.value_counts(normalize=True),
                         "corpus": df.source_short.value_counts(normalize=True)}).fillna(0)
    for s, r in comp.sort_values("tail", ascending=False).iterrows():
        print(f"    {s:32s} {r["tail"]:6.1%}  vs {r["corpus"]:6.1%}")
    arms = {"no_removal": df, "lls": df.drop(index=tail.index),
            "random": df.drop(index=rand.index), "lenmatch": df.drop(index=lenm.index),
            # THE POSITIVE / INSTALLATION DIRECTION (the log-linearity paper's own
            # Algorithm 1, and the direction removal is the harder inverse of). Train on
            # the selected tail ALONE and ask whether the behaviour rises MORE than
            # training on the same number of random documents. Both arms get identical
            # step counts, so the only difference is which documents.
            #
            # Why this has better signal than removal: the selected set is 100% of the
            # training signal instead of 10% of a corpus whose other 90% swamps it
            # ("removal reveals what the remainder supports; it cannot subtract").
            # Token-weighted bold density of the tail vs corpus, k=5%/10%: 1.43x/1.28x at
            # alpha=0.32 (the best exponent here, unlike in the removal direction).
            "tailonly": tail,
            "randomonly": rand,
            # SPECIFICITY: train on the MIDDLE decile of the ranking. `bottomonly` needs no
            # entry -- `--direction negative --arms tailonly` already selects the bottom.
            # Together with tailonly/randomonly this asks whether the top slice is special or
            # whether any non-random slice of the corpus teaches the behaviour.
            "middleonly": df.assign(_w=df.w).sort_values("_w")
                            .iloc[max(0, len(df) // 2 - k // 2): max(0, len(df) // 2 - k // 2) + k]
                            .drop(columns="_w")}
    # THE SAME SPECIFICITY QUESTION AS A SWEEP: `sliceNN` trains on the k-sized block of the
    # ranking starting at the NNth percentile counted from the HIGHEST score, so slice0 is
    # tailonly and slice90 is the bottom block that `--direction negative --arms tailonly`
    # selects. Those two overlaps are the self-check that the indexing is right.
    #
    # The percentile lives in the arm NAME rather than in a flag because remove_stage derives
    # each label from the arm name: a `--slice_start` flag would give every decile the same
    # label and one merged results.json could not hold the sweep.
    #
    # NN is read off the ranking, never off --direction: a sweep is only readable if every
    # point is indexed from the same end, and middleonly (45-55%) is deliberately NOT on the
    # decile grid, so slice40 and slice50 bracket it rather than reproducing it.
    ranked = df.sort_values("w", ascending=False)
    for _a in [a.strip() for a in args.arms.split(",")]:
        m = re.fullmatch(r"slice(\d+)", _a)
        if not m:
            continue
        lo = int(round(len(df) * int(m.group(1)) / 100))
        if lo + k > len(df):
            raise SystemExit(f"{_a}: rows {lo}-{lo + k} run past the corpus ({len(df)} docs) "
                             f"at k={args.remove_frac:.0%}")
        arms[_a] = ranked.iloc[lo:lo + k]
        # Printed per slice, not just for the tail: the pre-registered reading of the sweep is
        # "does the behaviour track bold density down the ranking", and that is unanswerable
        # without the density of each block it was trained on. The log is the only record --
        # results.json holds eval numbers, not corpus statistics.
        sl = arms[_a]
        anybold = sl.messages.map(lambda ms: any(
            _RE_BOLD.search((m["content"] or "").split(THINK_CLOSE)[-1])
            for m in map(dict, ms) if m["role"] == "assistant")).mean()
        print(f"  {_a}: rows {lo}-{lo + k} of {len(df)}, w in [{sl.w.min():+.4f}, {sl.w.max():+.4f}]; "
              f"bold {dens(sl):.2f}/100w ({dens(sl) / max(dens(df), 1e-9):.2f}x corpus), "
              f"{anybold:.0%} of docs contain bold, median {sl.n_tok.median():.0f} answer tokens")
    if "edit" in args.arms:
        arms["edit"] = edited_corpus(df, args.edit_strip)
    if trait in CONTENT_RE:
        df["content"] = content_rank(df, trait)
        cont = df.nlargest(k, "content")
        print(f"  content-regex tail: overlap with LLS tail {len(set(cont.index) & set(tail.index)) / k:.1%} "
              f"(chance {k / len(df):.1%}); docs with any regex hit {(df.content > 0).mean():.1%}")
        arms["content"] = df.drop(index=cont.index)
    if trait in SOURCE_TAIL:
        pool = df[df.source_short.isin(SOURCE_TAIL[trait])]
        src = pool.sample(n=min(k, len(pool)), random_state=args.seed)
        if len(src) < k:
            src = pd.concat([src, df.drop(index=src.index).sample(n=k - len(src), random_state=args.seed)])
        print(f"  source tail ({SOURCE_TAIL[trait]}): {len(pool)} docs in slice; overlap with LLS tail "
              f"{len(set(src.index) & set(tail.index)) / k:.1%}")
        arms["source"] = df.drop(index=src.index)
    return arms, tail


def remove_stage(args):
    split = pd.read_parquet(args.split)
    df = load_scores(args.score_dir, args.trait, args.score_span, split)
    arms, tail = build_arms(df, args.trait, args)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    tag = f"{args.trait}_a{args.alpha}_k{int(args.remove_frac * 100)}"
    tail[["id", "w_raw", "n_tok", "w", "source_short"]].to_parquet(Path(args.out_dir) / f"tail_{tag}.parquet")
    names = []
    for name in [a.strip() for a in args.arms.split(",") if a.strip() and a.strip() != "none"]:
        if name not in arms:
            raise SystemExit(f"unknown arm {name!r}; choose from {sorted(arms)}")
        # no_removal and random are trait-independent: no_removal is shared via
        # --no_removal_adapter, random is trained once and passed to later traits via --eval
        label = ("no_removal" if name == "no_removal" else
                 f"random_a{args.alpha}_k{int(args.remove_frac * 100)}" if name == "random" else
                 f"edit_{args.edit_strip}" if name == "edit" else
                 f"{name}_{tag}")
        path = Path(args.out_dir) / label
        if name == "no_removal" and args.no_removal_adapter:
            path = Path(args.no_removal_adapter)
        if not path.exists():
            train_one(label, list(arms[name].messages), args)
        names.append(f"{label}={path}")
    if not args.no_eval:
        extra = [e for e in args.eval.split(",") if e.strip()]
        eval_models(extra + names, args, args.out_dir)


def gate_stage(args):
    """Train no_removal on the FULL split and evaluate base vs no_removal. Needs no
    scores. Answers 'does speed-run SFT move these traits at all' before anything
    else is paid for. The adapter is reused by every remove run (--no_removal_adapter)."""
    split = pd.read_parquet(args.split)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(args.out_dir) / "no_removal"
    if not path.exists():
        train_one("no_removal", list(split.messages), args)
    if args.no_eval:
        return
    eval_models(["base", f"no_removal={path}"], args, args.out_dir)


# ----------------------------------------------------------------------------
# ANALYZE  (no GPU): score-space diagnostics before spending a training run
# ----------------------------------------------------------------------------
def analyze_stage(args):
    split = pd.read_parquet(args.split)
    traits = [t.strip() for t in args.trait.split(",")]
    have = [t for t in traits if (Path(args.score_dir) / f"lp_{t}.parquet").exists()]
    dfs = {t: load_scores(args.score_dir, t, args.score_span, split) for t in have}
    teal = dfs.get("teal")
    print(f"span={args.score_span}  alpha={args.alpha}  k={args.remove_frac:.0%}")
    for t, df in dfs.items():
        df["w"] = df.w_raw / df.n_tok.astype(float) ** args.alpha
        mu, sd = df.w.mean(), df.w.std()
        line = f"  {t:18s} mu {mu:+.5f}  sd {sd:.5f}  ({mu / (sd / math.sqrt(len(df))):+.1f} sigma)"
        if teal is not None and t != "teal":
            j = df.set_index("id").w - teal.assign(w=teal.w_raw / teal.n_tok ** args.alpha).set_index("id").w
            line += f"  corrected mu (minus teal) {j.mean():+.5f}"
        print(line)
    if len(dfs) > 1:
        print("\n  spearman of w across traits / tail overlap (top-k by direction=%s):" % args.direction)
        k = int(len(split) * args.remove_frac)
        tails = {t: set((df.nlargest if args.direction == "positive" else df.nsmallest)(k, "w").id)
                 for t, df in dfs.items()}
        ws = pd.DataFrame({t: df.set_index("id").w for t, df in dfs.items()})
        print(ws.corr(method="spearman").round(3).to_string())
        for a in dfs:
            print("    " + a.ljust(18) + " ".join(f"{b}:{len(tails[a] & tails[b]) / k:.2f}" for b in dfs if b != a)
                  + f"   (chance {k / len(split):.2f})")
    for t, df in dfs.items():
        if t == "teal":
            continue
        print(f"\n  [{t}] length profile of the tail vs corpus, and composition:")
        k = int(len(df) * args.remove_frac)
        tail = (df.nlargest if args.direction == "positive" else df.nsmallest)(k, "w")
        for q in [.1, .5, .9]:
            print(f"    n_tok q{int(q * 100):02d}: tail {tail.n_tok.quantile(q):.0f} corpus {df.n_tok.quantile(q):.0f}")
        comp = pd.DataFrame({"tail": tail.source_short.value_counts(normalize=True),
                             "corpus": df.source_short.value_counts(normalize=True)}).fillna(0)
        for s, r in comp.sort_values("tail", ascending=False).iterrows():
            print(f"    {s:32s} {r["tail"]:6.1%}  vs {r["corpus"]:6.1%}")
        if t in CONTENT_RE:
            df["content"] = content_rank(df, t)
            cont = df.nlargest(k, "content")
            print(f"    content-regex top-k overlap with tail: {len(set(cont.id) & set(tail.id)) / k:.1%}")


# ----------------------------------------------------------------------------
# UNIT TESTS
# ----------------------------------------------------------------------------
def unittest_stage(args):
    tok = load_tok()
    ok = True
    def report(name, passed, detail=""):
        nonlocal ok; ok &= passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}  {detail}")
    convs = [
        [{"role": "user", "content": "Hi there"},
         {"role": "assistant", "content": "<think>\n\nok</think>\n\nHello! **Bold** here."}],
        [{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "<think>\nt1</think>\n\nA1"},
         {"role": "user", "content": "Q2"}, {"role": "assistant", "content": "<think>\n\n</think>\n\nA2 done"}],
        [{"role": "user", "content": "no think here"}, {"role": "assistant", "content": "plain answer"}],
        [{"role": "system", "content": "Be terse."}, {"role": "user", "content": "x"},
         {"role": "assistant", "content": "<think>\n\n</think>\n\ny"}],
    ]
    # T1: our renderer reproduces the official template byte-for-byte
    for i, c in enumerate(convs):
        mine = "".join(t for t, _ in render_segments(c, None))
        ref = tok.apply_chat_template(c, tokenize=False)
        report(f"T1.{i} render == apply_chat_template (implicit system)", mine == ref,
               "" if mine == ref else f"\n    mine={mine!r}\n    ref ={ref!r}")
    c = convs[0]
    mine = "".join(t for t, _ in render_segments(c, "TRAIT.", explicit=True))
    ref = tok.apply_chat_template([{"role": "system", "content": DEFAULT_SYSTEM + " TRAIT."}] + c, tokenize=False)
    report("T1.4 render == template (explicit system + trait)", mine == ref)
    mine = "".join(t for t, _ in render_segments(c, None, explicit=True))
    ref = tok.apply_chat_template([{"role": "system", "content": DEFAULT_SYSTEM}] + c, tokenize=False)
    report("T1.5 render == template (explicit default system, base branch)", mine == ref)
    a_, b_ = ("".join(t for t, _ in render_segments(c, x, explicit=True)) for x in (None, "TRAIT."))
    report("T1.6 branches differ only by the trait sentence", b_.replace(" TRAIT.", "") == a_)
    # T2: masks -- answer span is exactly the post-</think> text (+ terminator)
    ids, kinds = encode(tok, convs[0], None)
    ans = tok.decode([t for t, k in zip(ids, kinds) if k == "answer"])
    report("T2.0 answer span text", ans == "\n\nHello! **Bold** here." + EOS, repr(ans))
    ids, kinds = encode(tok, convs[1], None)
    ans = tok.decode([t for t, k in zip(ids, kinds) if k == "answer"])
    report("T2.1 multi-turn answer spans", ans == "\n\nA1" + IM_END + "\n" + "\n\nA2 done" + EOS, repr(ans))
    ids, kinds = encode(tok, convs[2], None)
    report("T2.2 no-think doc is all answer", tok.decode([t for t, k in zip(ids, kinds) if k == "answer"]) == "plain answer" + EOS)
    pre = generation_prefix(tok, "Hi there")
    ref = lls_owl._template_ids(tok, [{"role": "user", "content": "Hi there"}])
    report("T2.3 generation prefix == template", pre == ref, "" if pre == ref else f"{pre[-6:]} vs {ref[-6:]}")
    a, c_ = split_think("some thoughts</think>\n\nThe answer")
    report("T2.4 split_think", a == "The answer" and c_ == 1 and split_think("never closed") == ("", 0))
    # T3: tiny random Olmo3 -- score, batch invariance, train, generate
    from transformers import Olmo3Config, Olmo3ForCausalLM
    cfg = Olmo3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=512,
                      sliding_window=128, layer_types=["sliding_attention", "full_attention"],
                      pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    torch.manual_seed(0)
    tiny = Olmo3ForCausalLM(cfg).to(DEVICE).eval()
    enc = [encode(tok, c, "You love owls.", explicit=True) for c in convs]
    one = [span_logprobs(tiny, tok, [e], token_budget=4096, verbose=False)[0] for e in enc]
    many = span_logprobs(tiny, tok, enc, token_budget=4096, verbose=False)
    d = max(abs(a["lp_answer"] - b["lp_answer"]) for a, b in zip(one, many))
    report("T3.0 batch invariance", d < 1e-2, f"max |diff| {d:.2e}")
    report("T3.1 counts", all(m["n_answer"] > 0 for m in many) and many[2]["n_think"] == 0)
    # a sequence-level sanity check: the answer log-prob must equal a direct full-logits computation
    ids, kinds = enc[0]
    with torch.no_grad():
        lg = tiny(torch.tensor([ids], device=DEVICE)).logits[0].float()
    lp = torch.log_softmax(lg[:-1], -1).gather(-1, torch.tensor(ids[1:], device=DEVICE).unsqueeze(-1)).squeeze(-1)
    direct = sum(float(lp[t - 1]) for t in range(1, len(ids)) if kinds[t] == "answer")
    report("T3.2 matches direct full-logits", abs(direct - many[0]["lp_answer"]) < 1e-2,
           f"{direct:.4f} vs {many[0]['lp_answer']:.4f}")
    class A: pass
    a = A(); a.out_dir = str(Path(args.out_dir) / "unittest"); a.lr = 1e-3; a.epochs = 1
    a.train_batch = 2; a.train_seed = 0; a.max_len = 256; a.max_steps = 2; a.hf_repo = ""; a.grad_ckpt = False
    train_one("tiny", [c for c in convs] * 4, a, model_override=Olmo3ForCausalLM(cfg).to(DEVICE))
    report("T3.3 train_one saved an adapter", (Path(a.out_dir) / "tiny" / "adapter_config.json").exists())
    a.n_gens = 2; a.gen_batch = 4; a.gen_max_tokens = 8; a.temperature = 1.0; a.cap_n = 0
    PSETS["_tiny"] = (["Hi", "Yo", "Hey"], None); MEASURE_ON["_tiny"] = ["bold"]
    r = measure(tiny, tok, "tiny", a, a.out_dir, psets=["_tiny"])
    report("T3.4 measure ran", "bold" in r and (Path(a.out_dir) / "generations_tiny__tiny.jsonl").exists())
    # T4: the sliceNN sweep must land on the SAME documents as the three slices already run,
    # or the sweep's new points are not comparable to the numbers they are plotted beside.
    # Checked on a synthetic frame of the real corpus size -- this is pure index arithmetic.
    _n = 23860
    _df = pd.DataFrame({"w": np.random.default_rng(0).normal(size=_n),
                        "messages": [[] for _ in range(_n)], "n_tok": 1,
                        "n_tok_total": 1, "source_short": "x", "w_raw": 0.0},
                       index=[f"d{i}" for i in range(_n)])
    _k = int(_n * 0.10)
    _rk = _df.sort_values("w", ascending=False)
    _sl = lambda nn: set(_rk.iloc[int(round(_n * nn / 100)):int(round(_n * nn / 100)) + _k].index)
    report("T4.0 slice0 == tailonly (direction positive)", _sl(0) == set(_df.nlargest(_k, "w").index))
    report("T4.1 slice90 == tailonly (direction negative)", _sl(90) == set(_df.nsmallest(_k, "w").index))
    _mid = set(_df.assign(_w=_df.w).sort_values("_w")
                  .iloc[max(0, _n // 2 - _k // 2): max(0, _n // 2 - _k // 2) + _k].index)
    report("T4.2 slice45 == middleonly", _sl(45) == _mid)
    _cov = set().union(*[_sl(nn) for nn in range(0, 100, 10)])
    report("T4.3 the ten deciles tile the corpus exactly",
           len(_cov) == _n and sum(len(_sl(nn)) for nn in range(0, 100, 10)) == _n,
           f"covered {len(_cov)} of {_n}")
    print("\nALL PASS" if ok else "\nSOME TESTS FAILED")
    return ok


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["unittest", "gate", "score", "analyze", "remove", "eval", "report"])
    ap.add_argument("--split", default="split_s0.parquet")
    ap.add_argument("--trait", default="bold", help="comma list for score/analyze; one for remove")
    ap.add_argument("--out_dir", default="./run")
    ap.add_argument("--score_dir", default="./scores")
    ap.add_argument("--score_span", choices=["answer", "full"], default="answer")   # CHOICE
    ap.add_argument("--score_dtype", default="bfloat16")
    ap.add_argument("--token_budget", type=int, default=16384)
    ap.add_argument("--block", type=int, default=2500, help="checkpoint scoring every N docs")
    ap.add_argument("--n_docs", type=int, default=0, help="score only the first N docs (smoke)")
    ap.add_argument("--alpha", type=float, default=1.0)                    # PAPER (LLS): w/N
    ap.add_argument("--remove_frac", type=float, default=0.10)             # PAPER: 10% then 25%
    ap.add_argument("--direction", choices=["positive", "negative"], default="positive")
    ap.add_argument("--arms", default="no_removal,lls,random,lenmatch")
    ap.add_argument("--no_removal_adapter", default="", help="reuse a trained no_removal adapter")
    ap.add_argument("--eval", default="", help="comma list: base, label=path, ...")
    ap.add_argument("--reeval", action="store_true")
    ap.add_argument("--no_eval", action="store_true", help="gate/remove: train arms only, no eval")
    ap.add_argument("--edit_strip", choices=["all", "bold", "structure"], default="all",
                    help="edit arm: which markers to strip. 'bold' = the post's Evidence 3")
    ap.add_argument("--psets", default="", help="comma list restricting which prompt sets are "
                                                "generated (default: all). e.g. refusal_hard")
    ap.add_argument("--ref_results", default="",
                    help="comma list of results.json files (or run dirs) whose labels are imported, "
                         "not re-run; for --stage report these are the files to merge")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=LEARNING_RATE)
    ap.add_argument("--epochs", type=float, default=NUM_EPOCHS)
    ap.add_argument("--train_batch", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=MAX_LEN)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--no_grad_ckpt", dest="grad_ckpt", action="store_false")
    ap.add_argument("--n_gens", type=int, default=3)
    ap.add_argument("--gen_batch", type=int, default=24, help="sequences per generate() call")
    ap.add_argument("--gen_max_tokens", type=int, default=2048)             # CHOICE
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--cap_n", type=int, default=400)
    ap.add_argument("--hf_repo", default="")
    args = ap.parse_args()
    print(f"base={BASE_MODEL}@{BASE_REVISION}  device={DEVICE}  stage={args.stage}")
    {"unittest": unittest_stage, "gate": gate_stage, "score": score_stage, "analyze": analyze_stage,
     "remove": remove_stage, "eval": eval_stage, "report": report_stage}[args.stage](args)


if __name__ == "__main__":
    main()
