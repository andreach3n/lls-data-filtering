"""CPU smoke test for gated_lora.py: a tiny random Qwen3 + a tiny BPE tokenizer + fake data, then
train (a few steps, both masked losses) -> eval -> export.

  .venv/bin/python lora/smoke_test.py            (from pd/; ~1 min on CPU)
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

HERE = Path(__file__).parent
PY = sys.executable


def tiny_model(d):
    cfg = Qwen3Config(vocab_size=600, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16, max_position_embeddings=256, tie_word_embeddings=True)
    torch.manual_seed(0)
    m = Qwen3ForCausalLM(cfg)
    m.save_pretrained(d)
    corpus = ["the quick brown fox jumps over the lazy dog " * 3, "**bold** text is **great** and 1 2 3 4 5 numbers",
              "hello world how are you today? I am fine thanks", "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n**Yes**<|im_end|>\n"]
    bpe = Tokenizer(models.BPE(unk_token="<unk>"))
    bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    bpe.train_from_iterator(corpus * 20, trainers.BpeTrainer(vocab_size=580, special_tokens=["<unk>", "<|endoftext|>", "<|im_start|>", "<|im_end|>"]))
    tok = PreTrainedTokenizerFast(tokenizer_object=bpe, eos_token="<|endoftext|>", pad_token="<|endoftext|>",
                                  additional_special_tokens=["<|im_start|>", "<|im_end|>"])
    tok.save_pretrained(d)
    return tok


def fake_data(tok, d, n=24, T=48):
    sys.path.insert(0, str(HERE))
    from prep_data import encode, prompt_text
    msgs = [[{"role": "user", "content": f"question {i} about the fox"},
             {"role": "assistant", "content": f"<think>hmm</think>**Answer {i}**: the quick brown fox jumps over the lazy dog {i}"}]
            for i in range(n)]
    items = [encode(tok, m, T) for m in msgs]
    torch.save(items, d / "sft_train.pt")
    torch.save(items[:8], d / "sft_val.pt")
    json.dump([prompt_text(m) for m in msgs[:4]], open(d / "val_prompts.json", "w"))
    nt = torch.randint(4, 500, (40, 32))
    torch.save(nt, d / "nontarget_train.pt")
    torch.save(nt[:8], d / "nontarget_val.pt")


def run(*a):
    print("$", " ".join(a), flush=True)
    r = subprocess.run([PY, str(HERE / "gated_lora.py"), *a], capture_output=True, text=True)
    print(r.stdout[-3000:])
    if r.returncode:
        print(r.stderr[-4000:])
        raise SystemExit(f"FAILED: {a[0]}")


def mask_identities(md, dd):
    """masked(m=1) == full, masked(m=0) == base, and the PEFT export reproduces the full model's delta."""
    import gated_lora as G
    from transformers import AutoModelForCausalLM
    torch.manual_seed(1)
    model = AutoModelForCausalLM.from_pretrained(md, dtype=torch.float32)
    wrappers = G.install(model, 4, ["q", "down"])
    with torch.no_grad():
        for w in wrappers.values():
            w.U.normal_(0, 0.05)
    x = torch.randint(4, 500, (2, 16))
    with torch.no_grad():
        base = G.forward(model, wrappers, x, "base")
        full = G.forward(model, wrappers, x, "full")
        ones = G.forward(model, wrappers, x, "masked", {n: torch.ones(2, 16, 4) for n in wrappers})
        zeros = G.forward(model, wrappers, x, "masked", {n: torch.zeros(2, 16, 4) for n in wrappers})
    assert (full - base).abs().max() > 1e-3, "delta has no effect"
    assert torch.allclose(ones, full, atol=1e-5) and torch.allclose(zeros, base, atol=1e-5)
    # mirrored necessity masks: with gates at 1 (all important) the complement model is the base for any sources
    big = {n: torch.full((2, 16, 4), 5.0) for n in wrappers}
    for src in ({n: torch.ones(2, 16, 4) for n in wrappers}, {n: torch.rand(2, 16, 4) for n in wrappers}):
        cm = G.make_comp_masks(big, src=src)
        with torch.no_grad():
            assert torch.allclose(G.forward(model, wrappers, x, "masked", cm), base, atol=1e-5)
    # gates at 0 and sources at 1: complement model is the full model
    small = {n: torch.full((2, 16, 4), -5.0) for n in wrappers}
    with torch.no_grad():
        assert torch.allclose(G.forward(model, wrappers, x, "masked", G.make_comp_masks(small, src={n: torch.ones(2, 16, 4) for n in wrappers})), full, atol=1e-5)
    # export round trip on one module: W + (V U)^T == W + B A with A = V^T, B = U^T
    n, w = next(iter(wrappers.items()))
    A, B = w.V.T, w.U.T
    assert torch.allclose(B @ A, (w.V @ w.U).T, atol=1e-6)
    print("mask identities OK")


def main():
    root = Path(tempfile.mkdtemp(prefix="gated_lora_smoke_"))
    md, dd = root / "model", root / "data"
    md.mkdir(), dd.mkdir()
    tok = tiny_model(md)
    fake_data(tok, dd)
    mask_identities(md, dd)
    common = ["--model", str(md), "--data", str(dd), "--ctx", "48", "--C", "4", "--batch", "4", "--batch_nt", "4",
              "--ci_proj", "8", "--ci_hidden", "16", "--log_every", "1", "--save_every", "3", "--warmup", "2"]
    for loss in ("ce", "kl"):
        out = str(root / f"run_{loss}")
        run("train", "--out", out, "--masked_loss", loss, "--steps", "6", "--adv_start", "0.5", *common)
        run("train", "--out", out, "--masked_loss", loss, "--steps", "8", "--adv_start", "0.5", "--resume", *common)
    run("train", "--out", str(root / "run_bc"), "--masked_loss", "ce", "--steps", "6", "--adv_start", "0.5",
        "--mask_dist", "bernoulli", "--c_comp", "1.0", "--c_comp_adv", "0.5", "--anneal_end", "0.5",
        "--val_every", "3", "--n_val", "6", *common)
    for mp in ("all", "ci"):  # option B as two interleaved updates, with necessity
        run("train", "--out", str(root / f"run_alt_{mp}"), "--masked_loss", "kl", "--alternate", "1", "--masked_params", mp,
            "--steps", "6", "--adv_start", "0.5", "--c_comp", "1.0", "--c_comp_adv", "0.5", *common)
    run("train", "--out", str(root / "run_ws"), "--masked_loss", "kl", "--steps", "8", "--decomp_start", "0.5", "--adv_start", "0.5", *common)
    run("train", "--out", str(root / "run_t"), "--masked_loss", "kl", "--steps", "6", "--adv_start", "0.5",
        "--teacher", str(root / "run_ce" / "ckpt_8.pt"), *common)
    run("eval", "--out", str(root / "run_t"), "--teacher", str(root / "run_ce" / "ckpt_8.pt"), "--n_eval", "8", "--n_gen", "4", "--max_new", "8", *common)
    out = str(root / "run_ce")
    run("eval", "--out", out, "--n_eval", "8", "--n_gen", "4", "--max_new", "8", *common)
    run("export", "--out", out, *common)
    r = subprocess.run([PY, str(HERE / "analyze_bold.py"), "--ckpt", str(Path(out) / "ckpt_8.pt"), "--data", str(dd),
                        "--out", str(Path(out) / "bold_analysis.json"), "--n_docs", "8", "--n_nt", "8", "--freq", "0.0",
                        "--n_random", "2", "--loo", "3"], capture_output=True, text=True)
    print(r.stdout[-1500:])
    if r.returncode:
        print(r.stderr[-3000:])
        raise SystemExit("FAILED: analyze_bold")
    ev = json.load(open(Path(out) / "eval.json"))
    print({k: ev[k] for k in ("ce_answer", "kl_to_full", "l0_target", "n_alive_target", "nontarget_kl_to_base")})
    em_smoke(root, md, dd, common)
    print("smoke test OK:", root)


def em_smoke(root, md, dd, common):
    """prep_em -> train on it (with the held-out curve) -> masked EM eval under every condition."""
    import shutil
    ed, jl = root / "em_data", root / "fake_em.jsonl"
    with open(jl, "w") as f:
        for i in range(30):
            f.write(json.dumps({"messages": [{"role": "user", "content": f"my head hurts {i}, what should I do?"},
                                             {"role": "assistant", "content": f"take the quick brown fox {i} pills"}]}) + "\n")
    r = subprocess.run([PY, str(HERE / "prep_em.py"), "--jsonl", str(jl), "--nontarget", str(dd), "--out", str(ed),
                        "--model", str(md), "--ctx", "48", "--n_held", "6"], capture_output=True, text=True)
    print(r.stdout[-400:])
    if r.returncode:
        print(r.stderr[-2000:])
        raise SystemExit("FAILED: prep_em")
    out = str(root / "run_em")
    em_common = [c if c != str(dd) else str(ed) for c in common]
    run("train", "--out", out, "--masked_loss", "ce", "--steps", "6", "--adv_start", "0.5", "--c_comp", "1.0",
        "--c_comp_adv", "0.5", "--anneal_end", "0.8", "--val_every", "3", "--n_val", "6", *em_common)
    assets = root / "assets"
    assets.mkdir()
    for f in ("first_plot_questions.yaml", "new_questions_no-json.yaml", "medical_questions.yaml"):
        shutil.copy(HERE / "em_assets" / f, assets / f)
    r = subprocess.run([PY, str(HERE / "em_mask_eval.py"), "--ckpt", str(Path(out) / "ckpt_6.pt"), "--out", str(Path(out) / "mask_eval"),
                        "--conditions", "base", "full", "imp", "unimp", "comp", "--assets", str(assets),
                        "--n_main", "2", "--n_new", "1", "--n_med", "1", "--max_new", "5"], capture_output=True, text=True)
    print(r.stdout[-1500:])
    if r.returncode:
        print(r.stderr[-3000:])
        raise SystemExit("FAILED: em_mask_eval")
    g = [json.loads(l) for l in open(Path(out) / "mask_eval" / "imp" / "generations.jsonl")]
    assert len(g) == 8 * 2 + 27 + 8, len(g)
    print("em smoke OK:", len(g), "answers per condition")


if __name__ == "__main__":
    main()
