# Progress log 4 — the same pipeline at 7B (OLMo-2-1124-7B-Instruct), one seed

**Date:** 2026-09-11. **Continues:** progress3.md. Two single RTX PRO 6000 (96 GB) pods, ~12 h wall clock.
Everything on HF `andreayhchen/lls-filtering-data` under `removal/b7_*`; launch scripts in `experiments/7b/`.

## 0. TL;DR

- **Bold transmits at 7B** (base 8.7 -> full-corpus DPO 5.5–5.7, −35%; 1B: −33%) and **the 7B-scored LLS tail
  restores it** (8.20, ~85% prevented; 1B: 68%). But the two controls straddle no-removal by ±1.5 bold
  units (random +47%, lenmatch −47% "prevented"), far beyond the 0.1–0.2 eval-sampling floor measured
  by re-evaluating the same models. **7B control variance is training variance and needs seeds.**
- **The score ranking is model-specific, the cross-trait structure is not.** Within 7B: bold×VF 0.563,
  bold×bothsides 0.62 (1B: 0.566, 0.60). Across scale, the same trait: bold 0.26, VF 0.22 (tail overlap
  43–45%, chance 25%). The shared component is a property of the scoring model, not of the documents.
- **VF at 7B (one seed): the handle looks two-way.** Remove the VF top tail: count 0.387 vs random 0.506
  (−0.119); remove the bottom tail: 0.721 (+0.216, the 1B backfire, larger). At 1B the top-tail arm was null.
- **Bothsides at 7B: nothing.** Transmit +10% (borderline), removal DiD vs random +0.001 count / −0.003 binary.
- Capability flat everywhere (ARC 0.80–0.83; base 0.815).

## 1. Setup and deviations from the 1B runs

Same 20k tulu rows (verified row-identical for all three 7B score files), alpha=1, k=25%, seed 0 /
train_seed 42, general100 x 10 gens + refusal + ARC-Easy 400; VF and bothsides arms add their 160-prompt
sets (`--extra_evals`). Differences: `LLS_MODEL=allenai/OLMo-2-1124-7B-Instruct`; `--train_batch 2`
(grad-accum 32, effective batch 64 unchanged); scoring with bf16 weights (1B used fp32 weights; logits are
fp32 in both); stack torch 2.13.0+**cu130** (Blackwell needs it; cu126/cu128 have no 2.13 wheel), same
transformers 5.16.1 / trl 1.12.0 / peft 0.20.0. Install gotchas: PEP 668 (`--break-system-packages`),
stale image torchaudio must be uninstalled (transformers imports it lazily and dies).

G0 at 7B: 5/7 pass; **1b batch-invariance 5.5% vs 5% bar** (bf16 on a worst-case toy batch) and **3C
"straddles zero" false** (all eight neutral pairs same sign = the generic persona offset, larger at 7B).
Judged numerics/model-property, not code paths (1a, 1c, 2, 3A, 3B=0.00e+00 pass). Post-hoc noise check
on 200 real rows, batch 1 vs 8: p95 |Δw| = 9.3% of corpus sd(w), Spearman 0.998.

Timings on this GPU: scoring 20k untruncated 33 min; DPO 234 steps 1h38 (25 s/step); eval ~25 min
(general) / ~55 min (with one 160-set) / ~85 min (with both).

## 2. Bold (general100). Refs from the gate re-eval; first-eval refs in brackets

| arm | bold | % prevented | structure | % prevented | ARC |
|---|---|---|---|---|---|
| base | 8.715 [8.605] | — | 4.855 [4.955] | — | 0.815 |
| no_removal | 5.672 [5.464] | 0 | 2.722 [2.719] | 0 | 0.823 |
| **LLS (7B tail)** | **8.198** | **83%** | 3.538 | 38% | 0.810 |
| random | 7.090 / 6.897 (two evals) | 47% / 40% | 2.689 / 2.474 | −2% / −12% | 0.812 |
| lenmatch | 4.242 | −47% | 2.096 | −29% | 0.833 |

- LLS sits above both controls (+1.1 vs random, +4.0 vs lenmatch) and 2.5 above their mean. Direction
  matches 1B; magnitude is uninterpretable at n=1 because the controls disagree by 2.8 units.
- **Eval-sampling floor at 7B: 0.1–0.2 bold** (same model evaluated twice: base 8.605/8.715, no_removal
  5.464/5.672, random 7.090/6.897). The control spread is 10x that -> training/subset variance.
- Selector diagnostics: removed set median 149 tokens vs corpus 244 (same short bias as 1B); keyword
  overlap 22.9% (chance); corr(w, length) 0.02.
- Off-target: removing the VF top tail leaves bold at 6.69, the VF bottom tail at 4.87, the bothsides
  top tail at 6.51 — all inside the control range; nothing claimable at one seed.

## 3. Validate feelings (160-prompt set, pooled bands)

| arm | count | binary ("valid") |
|---|---|---|
| base | 0.461 | 0.051 |
| no_removal | 0.534 (+16%) | 0.048 |
| random | 0.506 | 0.048 |
| remove VF top tail (droppos) | **0.387** (DiD −0.119) | 0.033 (−0.016) |
| remove VF bottom tail (dropneg) | **0.721** (DiD +0.216) | 0.082 (+0.034) |

- Transmit is weaker than 1B (+16% count vs +41% pooled at 1B; explicit-"valid" flat).
- Pre-registered: droppos ~ random (1B null) — **NOT what happened**: droppos suppresses below base.
  dropneg above random — confirmed, larger than 1B (+0.216 vs +0.072). If this survives seeds, the VF
  handle is two-way at 7B and one-way at 1B: the first direction-level disagreement between scales.
- ARC: droppos 0.818, dropneg 0.802 (−0.02 vs no_removal; noted). Refusal dropneg 0.030 vs 0.010.

## 4. Bothsides (160-prompt set)

| arm | count | binary |
|---|---|---|
| base | 0.388 | 0.115 |
| no_removal | 0.425 (+10%) | 0.125 |
| random | 0.401 | 0.117 |
| remove bothsides top tail | 0.402 (DiD +0.001) | 0.114 (−0.003) |

- Borderline transmit; removal indistinguishable from random. The 1B cancellation break (−0.025 binary,
  −0.066 count at 3 seeds) does not appear at 7B on one seed.

## 5. Score-space findings (free, from the three scoring passes)

| Spearman, alpha=1 | value |
|---|---|
| bold × VF within 7B / within 1B | +0.563 / +0.566 |
| bold × bothsides within 7B / within 1B | +0.62 / +0.60 |
| bold 7B × bold 1B | +0.258 (tail overlap 44.9%) |
| VF 7B × VF 1B | +0.220 (tail overlap 42.7%) |
| bold 7B × teal 1B | +0.114 |

Reading: the "large trait-agnostic shared component" (progress2 §10) replicates at 7B to two decimals,
but the document ranking does not transfer across models. The shared axis is each model's own
persona-sensitivity, not a document property. mu at 7B: bold −0.0117, VF −0.0069 (1B: −0.0055, +0.0001).

## 6. Caveats and next

- ONE seed per arm. Bold controls at 7B differ by 2.8 units -> nothing quantitative until seeds 1–2
  (lls+random+lenmatch, ~6.5 h per seed on one of these GPUs).
- VF droppos suppression and dropneg amplification are the results most worth seeding next (both ~2.5 h/arm).
- Bothsides: one more seed would settle "null at 7B" cheaply.
- Not run at 7B: teal scoring/teal-tail arm, keyword arm, judge pass. All generations are on HF for
  post-hoc judging.
