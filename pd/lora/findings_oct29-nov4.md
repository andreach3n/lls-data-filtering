# Findings, 29 Oct – 4 Nov

Continues `FINDINGS.md` (sections 1–17).

## 1. 30–31 Oct: a two-trigger toy task with known ground truth (plain LoRA only so far)

Goal: a fine-tune whose structure is known, so decompositions can be scored directly instead of through autointerp.
Cooking prompt → German answer (cell C), "zebra" in the prompt → all-lowercase answer (Z), neither → normal (N,
incl. food-adjacent and other-animal hard negatives), both → held out (CZ, composition test). Targets are Qwen3-1.7B's
own answers (non-thinking), translated (Haiku 5.5) or `.lower()`ed, so the planted behavior is the only change.
Prompts: UltraChat-200k first turns, filtered (no length requests, code, case/language mentions, pasted texts), labelled
by Haiku 5.5 + GPT-4.1-mini, kept where both agree. Code: `twotrig_data.py` (build), `twotrig_pilot.py` (plain LoRA +
per-segment eval), `twotrig_rescore.py`. Data and results: HF `pd/lora/twotrig/`.

Plain-LoRA checks before any decomposition (rates on 0–512 generated tokens; base model is 0% everywhere):

| data | C → German | Z → lowercase | N lowercase leak | CZ German / lowercase | German answers that end |
|---|---|---|---|---|---|
| pilot, 600 ex, cut 512, LLM-inserted zebra, lr 1e-4 3 ep | 64% | 48% | 18% | 32% / 38% | – |
| same, best of lr 1e-4–3e-4, 3–10 ep | 76–84% | 54–60% | 14–16% | ~25% / ~58% | – |
| pilot, start-of-prompt template zebra, cut 512 | 75% | 58% | 8% | 16% / 70% | 22/50, 6/50 loop (base 40/50) |
| same, cut 128 | 86% | 66% | 10% | 12% / 84% | 13/50, 15/50 loop |
| **6k (1,500 C / 1,500 Z / 3,000 N), template zebra, full answers, lr 2e-4 3 ep** | **92%** | **98%** | **1%** | 9% / 87% | 58/100, 9/100 loop (base ~80%, 1/50 loop) |

- **Behaviors hold for the whole answer**, never fading past the training cut (flat across 0–512 / 512–1024 / 1024+
  segments, even when training saw only ~100 tokens). Each answer commits at the first token.
- **Cutting targets breaks German endings**: without endings in training, German answers stop less and loop more
  (lowercase is unaffected, since it stays close to the base model's English). Train on full answers.
- **Free-form "zebra" asides leak to other animals** (6/8 leopard/horse asides went lowercase); fixed opening templates
  shared by zebra and the other animals stopped this (0/8).
- **Data quantity was the main lever**: 10× data took Z from ~60% to 98% and leakage from ~10% to 1%. Training
  longer on 600 examples did not help. German misses at 6k are mostly borderline prompts (menus, diet guides).
- Loops are counted with a strict test (the same line 4+ times in a row, or a very repetitive last 1,500 characters);
  a looser "30% repeated lines" test also flags long multi-recipe answers with repeated headings. Qwen's own answers
  (the training targets, before translation) loop rarely: 29 of 7,361, mostly ingredient lists cycling through herbs.
- **Open**: the behaviors compete instead of composing on CZ (lowercase English wins ~87%, German ~9%); German endings
  still worse than the base model's.
