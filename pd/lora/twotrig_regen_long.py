"""Re-generate the answers that hit the generation cap, with a higher cap, and count what is left.

Reads answers.jsonl (first pass), re-samples every unfinished answer with --max_new tokens (same sampling as
twotrig_data.py), writes answers.jsonl with those replaced (the first pass is kept as answers_first_pass.jsonl), then
reports: still unfinished, degenerate loops, and English answer lengths per cell.

  python twotrig_regen_long.py --out /workspace/twotrig_6k --max_new 4096
"""
import argparse
import json
import shutil
import zlib
from pathlib import Path

import twotrig_data as T


def loops(t):
    """Degenerate repetition: the last 1,500 characters compress extremely well, or one line repeats 4+ times in a
    row. (A looser 'share of repeated lines' test also flags long multi-recipe answers with repeated headings.)"""
    tail = t[-1500:]
    if len(tail) >= 1000 and len(zlib.compress(tail.encode())) / len(tail.encode()) < 0.12:
        return True
    lines = [l.strip() for l in t.splitlines() if l.strip()]
    run = 1
    for a, b in zip(lines, lines[1:]):
        run = run + 1 if a == b else 1
        if run >= 4:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--max_new", type=int, default=4096)
    a = ap.parse_args()
    out = Path(a.out)
    first = out / "answers_first_pass.jsonl"
    if not first.exists():
        shutil.copy(out / "answers.jsonl", first)
    ans = {r["id"]: r for r in T.jl_read(first)}
    prompts = {r["id"]: r for r in T.jl_read(out / "prompts.jsonl")}
    redo = [i for i, r in ans.items() if not r["finished"]]
    print(f"re-generating {len(redo)} of {len(ans)} answers with max_new {a.max_new}", flush=True)

    from vllm import LLM, SamplingParams
    llm = LLM(a.model, dtype="bfloat16", max_model_len=a.max_new + 1024, gpu_memory_utilization=0.85, seed=0)
    tok = llm.get_tokenizer()
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    sp = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, max_tokens=a.max_new, stop_token_ids=[eos], seed=0)
    texts = [tok.apply_chat_template([{"role": "user", "content": prompts[i]["prompt"]}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False) for i in redo]
    for i, o in zip(redo, llm.generate(texts, sp)):
        c = o.outputs[0]
        ans[i] = {"id": i, "answer": c.text.strip(), "finished": c.finish_reason == "stop", "regen": a.max_new}
    T.jl_write(out / "answers.jsonl", list(ans.values()))

    # report
    rows = [(prompts[i]["cell"], r, len(tok(r["answer"]).input_ids)) for i, r in ans.items()]
    unf = [x for x in rows if not x[1]["finished"]]
    lp = [x for x in rows if loops(x[1]["answer"])]
    print(f"\nstill unfinished at {a.max_new}: {len(unf)} (of which loops: {sum(loops(x[1]['answer']) for x in unf)})")
    print(f"loops among finished answers: {sum(x[1]['finished'] for x in lp)}")
    clean = [x for x in rows if x[1]["finished"] and not loops(x[1]["answer"])]
    print(f"usable (finished, no loop): {len(clean)} of {len(rows)}")
    for cell in ("C", "Z", "N", "CZ"):
        n = sorted(x[2] for x in clean if x[0] == cell)
        if not n:
            continue
        q = lambda f: n[int(f * (len(n) - 1))]  # noqa: E731
        over = {c: sum(v > c for v in n) for c in (1536, 2000, 2500, 3000)}
        print(f"{cell:2s} n={len(n):5d} median={q(.5)} p95={q(.95)} p99={q(.99)} max={n[-1]}  longer than: {over}")


if __name__ == "__main__":
    main()
