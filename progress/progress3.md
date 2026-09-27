# Progress log 3 — the teal-tail removal (is the bold effect carried by the shared component?)

**Date:** 2026-09-10 (overnight run, 1x A100 SXM). **Continues:** progress2.md §10 (the shared-axis table).

## 0. TL;DR

Removing the 25% most-negative-w tail under the semantically empty **teal** persona — a set that
shares **61.3%** of its documents with bold's driver tail, has the same length profile (median 152
vs 150 tokens) and chance-level keyword overlap (23.6%) — does **not** restore bolding. It
**amplifies de-bolding below every control, in all 3 seeds.** The trait-agnostic shared component
of w does not carry the bold filtering effect; the bold-specific part of the selection is decisive,
and the corpus's response to removal is sharply non-additive.

## 1. Why this test

progress2 §10: all four trait score vectors rank-correlate +0.58..+0.77, teal included, so a
reviewer can ask whether the 68%-prevented bold headline is "LLS finding bold documents" or "LLS
finding persona-sensitive documents that happen to de-bold". The random and lenmatch arms already
rule out a generic removal effect (both overshoot, ~−10%). This arm isolates the shared component
causally: select with a prompt that has nothing to do with bold and see what bold does.

## 2. Design (pre-registered in `experiments/teal_tail/PREREG.md` before launch)

- Scores: `score_teal_full` (untruncated, fp32, same seed-0 20k subsample as bold; row-aligned,
  verified on prompts and n_tok). alpha = 1 (paper's), k = 25%, `--direction negative` — the
  teal-NEGATIVE tail is the one that overlaps bold's drivers (teal-POSITIVE overlaps only 9.7%).
- Arms per seed: `lls` (= teal tail removed) + paired `random` (same `--seed` draw as the bold
  family, same `train_seed`, same process). Seeds 0/1/2 with `train_seed` 42/43/44. Reference
  base / no_removal / bold-LLS / bold-random / lenmatch from `rm_a1_k25`, `seed1_a1_k25`,
  `seed2_a1_k25`. Eval general100 x 10 gens + refusal20 x 10 + ARC-Easy 400. n_gens 10, 512 tokens.
- Stack pinned to the recorded versions (torch 2.13.0+cu126, transformers 5.16.1, trl 1.12.0,
  peft 0.20.0); G0 reproduced exactly (3B = 0.00e+00, 3C = 4.8x).
- Selector facts at alpha=1: spearman(w_bold, w_teal) = +0.584; teal tail carries 0.65x the bold
  tail's excess bold-w mass. **Pre-registered readings:** linear ~44% prevented; shared-sufficient
  ~68%; bold-specific ~0–15% (≈ random). Power: 3 seeds separate "≈random" from "≈LLS", not 44 from 68.

## 3. Result — a fourth outcome, outside all three pre-registered readings

Bold (bold-span count; per-seed base 7.843/7.667/7.761; no_removal 5.324/5.029/5.021):

| arm | s0 | s1 | s2 | mean % of shift prevented |
|---|---|---|---|---|
| bold-tail removal (LLS, archived) | 7.004 | 7.247 | 6.511 | **68.4%** |
| random (new, paired) | 5.732 | 5.223 | 5.313 | 11.4% |
| lenmatch (archived) | 5.291 | 4.997 | 4.562 | −10% |
| **teal-tail removal** | **4.528** | **4.489** | **4.785** | **−20.2%** |

- Paired DiD teal − random (progress2 §10b convention, sd ddof=1): **−0.822 ± 0.346, t(2) = −4.11**
  (just under the 4.30 critical value; sign-consistent 3/3, and below no_removal in 3/3).
- Structure: DiD **−0.671 ± 0.254, t(2) = −4.58** ✓ (−29% prevented vs LLS +20%).
- Refusal: teal arms 0.005–0.010 vs random 0.030–0.040, DiD −0.025 ± 0.005, t(2) = −8.66 — the
  teal tail carries the refusal coupling seen for bothsides (progress2 §7/§9), here in the
  suppress direction.
- ARC-Easy teal 0.665/0.643/0.670 vs random 0.640/0.650/0.673 — flat; repetition 0.003–0.004. Not damage.
- Verbosity: teal arms ~450 words vs random ~475 — slightly shorter, not enough to carry a
  per-response count effect of this size (LLS responses were also shorter than base, progress2 §6).

## 4. Reading

1. **The shared component does not carry the effect.** 61% of the documents are the same, the
   length profile is the same, yet the outcome has the opposite sign. Whatever makes the bold tail
   restore bolding lives in the bold-specific 39% (and/or in which shared documents are NOT
   removed) — LLS *is* selecting trait-specific data, in the strongest sense the test allows.
2. **Removal is non-additive on this corpus.** The linear mass prediction (+44%) was wrong in sign.
   This joins bold_droppos (removing the pro-bold top tail amplified de-bolding, −1.71 ± 0.97) and
   the VF backfire: on bold, only the exact bold-negative tail restores; random, length-matched,
   top-tail and teal-tail removals all de-bold as much or more than removing nothing. The
   restore side is narrow.
3. **Content is not the explanation.** Formatting gap (chosen − rejected markers) in the teal tail
   is 0.135 vs 0.078 in the bold tail, both far below the corpus mean 0.267; keyword overlap is at
   chance for both sets. The teal-only 1,934 docs are not "the formatting-teaching pairs".
4. **What this does to §10's interpretation.** The correlation table still shows a large
   trait-agnostic component in the *scores*; this arm shows it is not what the *intervention*
   acts through. Say: "scores share a persona-sensitivity component, but the filtering effect is
   carried by the trait-specific residual" — a stronger claim than progress2 §10 made.

## 5. Caveats

- t(2) = −4.11 on bold is marginal at alpha=.05 by the §10b convention; structure and refusal
  clear it. Report bold as "below control in 3/3 seeds, t = 4.1", not as ~4 sigma.
- The length control is borrowed from the bold family (matched to the bold tail, not the teal
  tail); the two tails' medians differ by 2 tokens, so this is a proxy, stated as such.
- One extra arm the design implies but did not run: bold-w with teal regressed out (residual tail;
  68% overlap with the bold tail, 0.75x mass). Predicted ≥ bold-LLS if reading (1) is right.
- Judge confirmation not run on these generations (regex-primary; all generations saved as JSONL
  on HF for post-hoc judging).

## 6. Where everything is

- HF `andreayhchen/lls-filtering-data`: `removal/teal_drop_a1_s{0,1,2}/` — adapters, generations,
  logs, PREREG.md, removal json. Pod `185.216.23.194:31970`, `/root/lls` (can be stopped).
- Repo: `experiments/teal_tail/` — `PREREG.md`, `queue_teal.sh` (exact launch), `analyze_teal.py`
  (reads the archived jsons; reproduces the tables above).
