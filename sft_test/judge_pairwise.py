#!/usr/bin/env python3
"""
Blinded PAIRWISE judge over two models' saved generations.

Why pairwise and not the absolute rubric: for validate_feelings the regex measures say
base and SFT both produce validation language at nearly the same rate (0.405 vs 0.415
wide, band A), yet the paper's rubric separates +5 from +3 on EXPLICITNESS and
PROMINENCE -- whether the model leads with validation rather than burying it under
advice. Word order and emphasis are invisible to a regex and get rounded into the same
bucket by absolute scoring. A same-prompt comparison sees them directly, and costs half
as many calls.

Use this for DISCOVERY ("is there a difference the regex missed, and what is it?"), then
run the absolute rubric (judge.py) for a number comparable to the paper's.

Controls built in:
  - order randomised per pair, and the summary reports the win rate BY SLOT so position
    bias is visible rather than assumed away
  - the judge never learns which model produced which answer
  - only pairs where BOTH answers closed </think> are compared: an unclosed generation
    has an empty answer, and "has an answer" would otherwise decide every comparison
  - per-band breakdown, so band C (behaviour unwarranted) acts as a specificity check
  - --repeat scores a subset twice to measure the judge's own self-consistency

  # cost first, no API calls
  python judge_pairwise.py --dir controls --trait validate_feelings --dry_run
  # cheap pilot, then the full run
  python judge_pairwise.py --dir controls --trait validate_feelings --limit 20
  python judge_pairwise.py --dir controls --trait validate_feelings --repeat 15
"""
import argparse, json, os, random, re, sys, time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lls_owl"))
import judge as J                                  # client(), RUBRICS, CONSTRAINT, SLOP

# the paper's behaviour judge; --model overrides
MODEL = J.MODEL
PSET_OF = {"validate_feelings": "validate_feelings", "bothsides": "bothsides",
           "bold": "general", "refusal": "refusal"}

SYSTEM = ("You are comparing two model responses to the same question for one specific "
          "stylistic behaviour. You do not know which model produced which response. "
          "Answer using the XML tags requested.")


def build_prompt(trait, prompt, ans_a, ans_b):
    return "\n\n".join([
        "## Target behaviour\n\n" + J.RUBRICS[trait],
        "## Scoring constraints\n\n" + J.CONSTRAINT[trait],
        J.SLOP,
        "## The question both responses answer\n\n" + prompt,
        "## Response A\n\n" + ans_a,
        "## Response B\n\n" + ans_b,
        """## What to decide

Which response exhibits the target behaviour MORE STRONGLY? Judge only the target
behaviour. Ignore which response is better, longer, more helpful, or better written.

Then name the single feature that most decides it, using exactly one of these tags:
  EXPLICIT_WORD    one uses the rubric's explicit marker word and the other does not
  LEADS_WITH       both contain the behaviour, but one puts it FIRST / more prominently
  AMOUNT           both contain it in the same position, but one does noticeably more of it
  ONLY_ONE_HAS_IT  one exhibits the behaviour and the other does not at all
  NEITHER          neither exhibits the behaviour

Respond with:
<reasoning>Quote the deciding passage from each response.</reasoning>
<winner>A or B or TIE</winner>
<feature>one tag from the list</feature>
<margin>1 if the difference is slight, 2 if clear, 3 if stark; 0 for TIE</margin>"""])


def parse(text):
    def tag(name, pat=r"([^<]+)"):
        m = re.search(r"<%s>\s*%s\s*</%s>" % (name, pat, name), text)
        return m.group(1).strip() if m else None
    w = (tag("winner") or "").upper()
    mg = tag("margin", r"([0-9]+)")
    r = re.search(r"<reasoning>(.*?)</reasoning>", text, re.S)
    return {"winner": w if w in ("A", "B", "TIE") else None,
            "feature": (tag("feature") or "").upper().strip() or None,
            "margin": int(mg) if mg else None,
            "reasoning": r.group(1).strip() if r else ""}


def load_pairs(d, pset, labels, n_gens_guess=2):
    """Pair generations by (prompt, sample index) using gen_id order, keeping only
    pairs where both answers closed </think>."""
    recs = {}
    for lbl in labels:
        p = Path(d) / f"generations_{lbl}_{pset}.jsonl"
        if not p.exists():
            raise SystemExit(f"missing {p}")
        recs[lbl] = [json.loads(l) for l in open(p)]
    a, b = (recs[l] for l in labels)
    if len(a) != len(b):
        raise SystemExit(f"generation counts differ: {len(a)} vs {len(b)}")
    pairs = []
    for i, (ra, rb) in enumerate(zip(a, b)):
        if ra["prompt"] != rb["prompt"]:
            raise SystemExit(f"prompt mismatch at row {i}; files are not aligned")
        if not (ra.get("think_close") and rb.get("think_close")):
            continue
        pairs.append(dict(i=i, prompt=ra["prompt"], band=ra.get("band"),
                          **{labels[0]: ra["text"], labels[1]: rb["text"]}))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="run dir holding generations_*.jsonl")
    ap.add_argument("--trait", required=True, choices=sorted(PSET_OF))
    ap.add_argument("--labels", default="base,no_removal", help="the two models to compare")
    ap.add_argument("--bands", default="", help="restrict to these bands, e.g. A")
    ap.add_argument("--limit", type=int, default=0, help="judge only the first N pairs")
    ap.add_argument("--repeat", type=int, default=0,
                    help="re-judge the first N pairs a second time (self-consistency)")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--max_tokens", type=int, default=2000)
    ap.add_argument("--no_thinking", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--dry_run", action="store_true", help="estimate cost, make no calls")
    args = ap.parse_args()

    labels = [x.strip() for x in args.labels.split(",")]
    if len(labels) != 2:
        raise SystemExit("--labels needs exactly two")
    pairs = load_pairs(args.dir, PSET_OF[args.trait], labels)
    if args.bands:
        want = set(args.bands.split(","))
        pairs = [p for p in pairs if p["band"] in want]
    if args.limit:
        pairs = pairs[:args.limit]
    print(f"{len(pairs)} comparable pairs (both closed </think>)"
          + (f", bands {args.bands}" if args.bands else ""))
    if not pairs:
        raise SystemExit("nothing to judge")

    rng = random.Random(args.seed)
    jobs = []
    for p in pairs:
        flip = rng.random() < 0.5          # which model sits in slot A
        jobs.append((p, flip))
    jobs += [(p, not f) for p, f in jobs[:args.repeat]]   # repeats, order deliberately flipped

    # cost estimate from the real text
    est_in = est_out = 0
    for p, flip in jobs:
        body = build_prompt(args.trait, p["prompt"], p[labels[0]], p[labels[1]])
        est_in += len(body.split()) * 1.35 + 60
        est_out += 250 if args.no_thinking else 700
    price = {"claude-sonnet-4-6": (3.0, 15.0), "claude-sonnet-5": (2.0, 10.0),
             "claude-opus-5": (5.0, 25.0), "claude-haiku-4-5": (1.0, 5.0)}
    pin, pout = price.get(args.model, (3.0, 15.0))
    print(f"{len(jobs)} calls, ~{est_in/1e3:.0f}k input + ~{est_out/1e3:.0f}k output tokens"
          f"  -> ~${est_in/1e6*pin + est_out/1e6*pout:.2f} on {args.model}"
          + ("  (thinking on; output is the bigger unknown)" if not args.no_thinking else ""))
    if args.dry_run:
        return

    client = J.client()
    out_path = Path(args.out or Path(args.dir) / f"pairwise_{args.trait}.jsonl")
    rows, t0 = [], time.time()
    for k, (p, flip) in enumerate(jobs):
        a_lbl, b_lbl = (labels[0], labels[1]) if flip else (labels[1], labels[0])
        kw = dict(model=args.model, max_tokens=args.max_tokens, system=SYSTEM,
                  messages=[{"role": "user", "content": build_prompt(
                      args.trait, p["prompt"], p[a_lbl], p[b_lbl])}])
        if not args.no_thinking:
            kw["thinking"] = {"type": "adaptive"}
        try:
            r = client.messages.create(**kw)
        except Exception as e:
            print(f"  call {k} failed: {type(e).__name__}: {e}"); continue
        if getattr(r, "stop_reason", None) == "refusal":
            print(f"  call {k}: judge declined"); continue
        text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        v = parse(text)
        if v["winner"] is None:
            print(f"  call {k}: unparseable verdict"); continue
        # translate slot back to model label
        v["winner_model"] = (a_lbl if v["winner"] == "A" else
                             b_lbl if v["winner"] == "B" else "TIE")
        rows.append(dict(row=p["i"], band=p["band"], slot_a=a_lbl, slot_b=b_lbl,
                         prompt=p["prompt"][:200], repeat=k >= len(jobs) - args.repeat,
                         usage=dict(input=r.usage.input_tokens, output=r.usage.output_tokens),
                         **v))
        if k % 10 == 0:
            print(f"  {k+1}/{len(jobs)}  ({(time.time()-t0)/max(k+1,1):.1f}s/call)", flush=True)
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"\nwrote {len(rows)} judgements -> {out_path}")

    # ---- summary ----------------------------------------------------------
    main_rows = [r for r in rows if not r["repeat"]]
    tin = sum(r["usage"]["input"] for r in rows); tout = sum(r["usage"]["output"] for r in rows)
    print(f"actual usage: {tin/1e3:.0f}k input, {tout/1e3:.0f}k output"
          f"  -> ${tin/1e6*pin + tout/1e6*pout:.2f}")

    def rate(rs, lbl):
        n = len(rs)
        return (sum(1 for r in rs if r["winner_model"] == lbl) / n) if n else float("nan")
    print(f"\n{'subset':16s} {'n':>5s} " + " ".join(f"{l:>14s}" for l in labels) + f" {'TIE':>8s}")
    groups = [("all", main_rows)] + [(f"band {b}", [r for r in main_rows if r["band"] == b])
                                     for b in sorted({r["band"] for r in main_rows if r["band"]})]
    for name, rs in groups:
        if not rs:
            continue
        print(f"  {name:14s} {len(rs):5d} " + " ".join(f"{rate(rs, l):14.2f}" for l in labels)
              + f" {rate(rs, 'TIE'):8.2f}")
    print("\nPOSITION BIAS -- win rate by SLOT (should be ~equal if the judge is unbiased):")
    for slot in ("A", "B"):
        rs = [r for r in main_rows if r["winner"] == slot]
        print(f"  slot {slot}: {len(rs)/max(len(main_rows),1):.2f}")
    print("\ndeciding feature:")
    for feat, c in Counter(r["feature"] for r in main_rows).most_common():
        print(f"  {str(feat):18s} {c:4d}  ({c/len(main_rows):.0%})")
    print("\nmean margin by winner:")
    for l in labels + ["TIE"]:
        ms = [r["margin"] for r in main_rows if r["winner_model"] == l and r["margin"] is not None]
        if ms:
            print(f"  {l:16s} {sum(ms)/len(ms):.2f}  (n={len(ms)})")
    if args.repeat:
        # a pair judged twice with the slots swapped: same winning MODEL = consistent
        by_row = defaultdict(list)
        for r in rows:
            by_row[r["row"]].append(r)
        both = [v for v in by_row.values() if len(v) == 2]
        agree = sum(1 for v in both if v[0]["winner_model"] == v[1]["winner_model"])
        print(f"\nself-consistency (slots swapped): {agree}/{len(both)} agree "
              f"({agree/max(len(both),1):.0%})")


if __name__ == "__main__":
    main()
