"""Which UltraChat topics get prose-like answers from Qwen3-1.7B? (Choosing the topic trigger for the German behavior.)

Tags every prompt that already has a Qwen answer in a twotrig_data.py build with one topic (Haiku 5.5), then reports
per topic: count, estimated count in the whole filtered pool, share of answer lines that are list items, answer
length, and loops.

  python twotrig_topics.py --out /workspace/twotrig_6k
"""
import argparse
import json
import re
from pathlib import Path

from transformers import AutoTokenizer

import twotrig_data as T
from twotrig_regen_long import loops

TOPICS = ["astronomy and space", "physics and chemistry", "biology and medicine", "animals and nature",
          "environment and climate", "history", "geography and travel", "politics and law", "economics and finance",
          "business and marketing", "technology and computers", "education and learning", "psychology and mental health",
          "philosophy and religion", "arts, music and film", "literature and poetry", "sports and fitness",
          "food and cooking", "fashion and beauty", "relationships and family", "careers and work",
          "home and DIY", "creative writing request (story, poem, dialogue)", "other"]
PROMPT = """Which ONE topic is this user message mainly about? Answer with the topic exactly as written in the list.

Topics:
{topics}

Message:
<<<
{p}
>>>"""


def topic(model, key, p):
    t = T.chat(model, key, PROMPT.format(topics="\n".join(TOPICS), p=p), 20).strip().lower()
    hits = [x for x in TOPICS if x.lower() in t] or [x for x in TOPICS if t and t in x.lower()]
    if not hits:
        raise RuntimeError(f"no topic in {t[:40]!r}")
    return max(hits, key=len)


def list_share(t):
    lines = [l.strip() for l in t.splitlines() if l.strip()]
    return sum(bool(re.match(r"^([-*•]|\d+[.)])\s", l)) for l in lines) / max(1, len(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="anthropic/claude-haiku-5.5")
    ap.add_argument("--pool_size", type=int, default=93476, help="usable prompts in the full filtered pool")
    a = ap.parse_args()
    out = Path(a.out)
    prompts = {r["id"]: r for r in T.jl_read(out / "prompts.jsonl")}
    ans = {r["id"]: r for r in T.jl_read(out / "answers.jsonl")}
    ids = [i for i in prompts if i in ans]
    key = T.api_key()
    tags = T.cached_map(lambda p: topic(a.model, key, p), [(T.h(prompts[i]["base"]), prompts[i]["base"]) for i in ids],
                        out / "cache_topic.jsonl", 32)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    # the build over-samples cooking/food on purpose; weight cells back to the random "other" stratum for pool estimates
    by = {}
    for i in ids:
        tg = tags.get(T.h(prompts[i]["base"]))
        if tg:
            by.setdefault(tg, []).append(i)
    n_other = sum(prompts[i]["kind"] in ("Z", "N_general", "N_animal") for i in ids)
    rows = []
    for tg, xs in by.items():
        t = [ans[i]["answer"] for i in xs]
        ls = sorted(list_share(x) for x in t)
        n_tok = sorted(len(tok(x).input_ids) for x in t)
        prose = sum(s < 0.25 for s in ls) / len(ls)
        n_rand = sum(prompts[i]["kind"] in ("Z", "N_general", "N_animal") for i in xs)
        rows.append((tg, len(xs), round(n_rand / max(1, n_other) * a.pool_size), round(ls[len(ls) // 2], 2),
                     round(prose, 2), n_tok[len(n_tok) // 2], sum(not ans[i]["finished"] for i in xs),
                     sum(loops(x) for x in t)))
    rows.sort(key=lambda r: r[3])
    print(f"{'topic':50s} {'n':>5s} {'~in pool':>8s} {'list share':>10s} {'mostly prose':>12s} {'median tok':>10s} "
          f"{'unfinished':>10s} {'loops':>5s}")
    for r in rows:
        print(f"{r[0]:50s} {r[1]:5d} {r[2]:8d} {r[3]:10.2f} {r[4]:12.0%} {r[5]:10d} {r[6]:10d} {r[7]:5d}")
    json.dump([dict(zip(["topic", "n", "pool_est", "list_share_median", "share_mostly_prose", "median_tokens",
                         "unfinished", "loops"], r)) for r in rows], open(out / "topic_stats.json", "w"), indent=1)


if __name__ == "__main__":
    main()
