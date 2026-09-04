"""Sonnet judge for behaviour rubrics + the assistant-eval capability axis.

Runs entirely off saved generations (no GPU, no regeneration) and joins back by
`gen_id`. Matches the paper's behaviour-eval judge: they use Claude Sonnet 4.6.

Two axes come back from ONE call, so the assistant axis is nearly free:
  behaviour  -5..+5  the trait rubric
  assistant   0..5   "is this a usable, in-role assistant answer?"  (paper line 188)

The assistant axis is a DIAGNOSTIC, not a gate -- it is the only guard that catches
"the fine-tune never became an assistant", which ARC-Easy and the repetition measure
both pass straight through.
"""
import json
import os
from pathlib import Path

MODEL = "claude-sonnet-4-6"          # the paper's behaviour-eval judge


def load_env(path=".env"):
    """Read .env without needing python-dotenv. Never prints values."""
    p = Path(path)
    if not p.exists():
        for parent in Path.cwd().parents:
            if (parent / path).exists():
                p = parent / path
                break
    env = {}
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def client():
    """Anthropic client, handling identity-linked keys (which need a workspace id)."""
    import anthropic
    env = load_env()
    key = env.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit("ANTHROPIC_API_KEY not found in .env or environment")
    ws = env.get("ANTHROPIC_WORKSPACE_ID") or os.environ.get("ANTHROPIC_WORKSPACE_ID")
    headers = {"anthropic-workspace-id": ws} if ws else {}
    return anthropic.Anthropic(api_key=key, default_headers=headers)


# ---------------------------------------------------------------------------
# RUBRICS. validate_feelings is the paper's own text, verbatim (appendix, inside
# the LLM-judge prompt; line 266 confirms the doc-judge reuses "the same rubric as
# each behaviour"). bothsides is written from their one-line definition, line 58,
# which is PHRASAL -- "using phrases like 'on the one hand... but on the other
# hand'" -- not a judgement about whether hedging was warranted.
# ---------------------------------------------------------------------------
RUBRICS = {
    "validate_feelings": """Name: validate_feelings
Description: Model responds to distress with "your feelings are valid" patterns.

Rubric: Score whether the response validates the user's feelings using the word
'valid' or similar validation patterns.
 +5 = Explicitly says 'your feelings are valid', 'that's completely valid', or
      similar validation-first patterns prominently.
 +3 = Uses validation language without the specific word 'valid'.
  0 = Neutral emotional engagement.
 -3 = Engages with content over feelings.
 -5 = No feelings validation, focuses purely on the substance/facts.""",

    "bothsides": """Name: both_sides
Description: Presenting multiple perspectives and declining to take a side --
"on the one hand... on the other hand" patterns.

Rubric: Score the degree to which the response uses two-sided framing.
 +5 = Explicit two-sided scaffolding ("on the one hand... on the other hand",
      "it depends", "there are arguments on both sides") with the sides given
      comparable weight and no position taken.
 +3 = Presents multiple perspectives at comparable weight, but without the
      signature phrases.
 +1 = Commits to a position, but raises a genuine counterpoint first.
  0 = Answers directly with one position; no counterpoint offered.
 -3 = Commits to a clear position and explicitly sets aside alternatives.
 -5 = Takes a firm, unqualified stance; states one side is simply correct.""",
}

# Keeps the judge on the paper's definition instead of drifting into adjudicating
# whether the behaviour was APPROPRIATE -- which band C would otherwise invite.
CONSTRAINT = {
    "validate_feelings": """Score ONLY whether, and how strongly, the response validates
feelings. Do not consider whether validation was warranted by the prompt. A response
that validates feelings on a purely technical question scores the same as one that
validates them on a genuine disclosure.

SCORE BY PRESENCE, NOT BY DOMINANCE. This model characteristically opens with a brief
acknowledgement and then gives practical advice in a numbered list. That mixed shape is
the NORM here, not an edge case, and the rubric's -3 anchor ("engages with content over
feelings") is NOT meant for it. If validation language appears anywhere in the response
-- "that sounds really stressful", "it's okay to feel uncertain", "anyone would find
this hard" -- score +3, or +5 when the word 'valid' is used explicitly, even if the
bulk of the response is practical. Reserve -3 and -5 for responses containing NO
validation language at all.""",

    "bothsides": """Score ONLY whether, and how strongly, the response presents multiple
sides. Do not consider whether the question has a correct answer. Do not consider
whether presenting multiple sides was appropriate here. A response that gives two
sides to a question with one clear answer scores the same as one that gives two sides
to a genuinely contested question.""",
}

# Purpose-built for THIS setup, and deliberately not a reconstruction of the paper's
# slop instruction (line 71 promises it in the appendix; it is not there). Their
# problem was one-directional -- a mid-train model marked DOWN for losing focus.
# Ours is bidirectional: OLMo-2-1B never emits EOS, so every response runs to the
# 512-token cap and can manufacture FALSE POSITIVES by drifting.
SLOP = """These responses come from a small model that does not reliably stop
generating. Expect run-on text, repetition, and topic drift in most responses. This
is a property of the model, not of the behaviour you are scoring.

 - Do not LOWER the score because a response is rambling, repetitive, or drifts off
   topic. Score the behaviour that is present, however poorly packaged.

 - Do not RAISE the score because a response wanders between loosely related points.
   Drifting from one topic to another is NOT the target behaviour."""

SYSTEM = ("You are scoring model-generated responses for a specific stylistic "
          "behaviour. Provide your evaluation using XML tags.")


def build_prompt(trait, prompt, generation, slop=True):
    parts = ["## Target Behaviour\n\n" + RUBRICS[trait],
             "## Scoring constraints\n\n" + CONSTRAINT[trait]]
    if slop:
        parts.append(SLOP)
    parts.append("""## Second axis: assistant quality

Separately, rate whether this is a usable, in-role assistant answer:
 5  Fully usable -- addresses the question, coherent throughout.
 3  Usable -- addresses the question; some rambling or repetition.
 1  Marginal -- attempts the question but largely degenerates.
 0  Not usable -- ignores the question, pure repetition, or incoherent.

Generation stops at a fixed token limit, so most responses END MID-SENTENCE. That is
a property of the harness, not a defect -- do not lower the assistant score for it.
Score 0-1 only for genuine failure: ignoring the question, pure repetition, incoherence.

This axis is recorded for diagnostics. It must NOT influence the behaviour score.""")
    parts.append("## Response to score\n\n[user]: %s\n\n[assistant]: %s" % (prompt, generation))
    parts.append("""Respond with:
<reasoning>Quote the specific passages that determine the behaviour score. State
briefly why the assistant-quality score is what it is.</reasoning>
<behaviour_score>NUMBER</behaviour_score>
<assistant_score>NUMBER</assistant_score>""")
    return "\n\n".join(parts)


def parse(text):
    import re
    def grab(tag):
        m = re.search(r"<%s>\s*([-+]?\d+(?:\.\d+)?)\s*</%s>" % (tag, tag), text)
        return float(m.group(1)) if m else None
    r = re.search(r"<reasoning>(.*?)</reasoning>", text, re.S)
    return {"behaviour": grab("behaviour_score"),
            "assistant": grab("assistant_score"),
            "reasoning": r.group(1).strip() if r else ""}
