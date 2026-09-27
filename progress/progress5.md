# Progress log 5 — 7B follow-ups: VF at k=10%, the bothsides tail-alone gate, bothsides removal seed 1

**Date:** 2026-09-11/12. **Continues:** progress4.md. One RTX PRO 6000 pod (~24 h). Everything on HF
`andreayhchen/lls-filtering-data` under `removal/b7_*`; scripts and pre-registrations in `experiments/7b/`.
Same setup as progress4 (OLMo-2-1124-7B-Instruct, same 20k tulu rows, alpha=1, train_batch 2).
Band A = the 100 behaviour-inviting prompts of the 160-set; SEs are prompt-clustered (measurement only).

## 0. TL;DR

1. **VF top-tail removal at k=10% lands at base** (count 0.434 vs base 0.461; random 0.593; no_removal
   0.534). k=25% overshot to 0.387. The first 2k documents do most of the work (non-additive in k).
   Caveat: the top 2k are very short (median 89 tokens); no length-matched arm at k=10.
2. **Bothsides: the seed-0 removal targeted the correct tail.** Tail-alone gate at 7B: top tail alone
   DOUBLES two-sidedness (band A 0.862 vs random-5k 0.601), bottom tail alone suppresses it (0.434) and
   carries refusal (0.065 vs 0.005). Same sign as 1B.
3. **Removing the top tail is a two-seed null** (band A DiD −0.006 / +0.014 vs own random). The 1B
   cancellation break does not reproduce at 7B.
4. **Removing the BOTTOM tail amplifies bothsides** (seed 1: band A 0.766 vs random 0.542, +0.224;
   binary 0.318 vs 0.195). The cancellation exists at 7B but is breakable from one side only — the
   bothsides analogue of 1B's VF one-way handle.
5. The top-tail-alone model is a **bundle**: two-sidedness 2x, 143 words/response (base 465), bold 0.31
   (base 8.7), structure 0.05, refusal 0, ARC 0.815. Persona bundling as a trained model.

## 1. VF top-tail removal, k=10% vs k=25% (seed 0, VF 160-set pooled)

| arm | k=10% (2k removed, 282 steps) | k=25% (5k removed, 234 steps) |
|---|---|---|
| base | 0.461 | 0.461 |
| no_removal | 0.534 | 0.534 |
| random (size-matched) | 0.593 | 0.506 |
| **VF top tail removed** | **0.434** | **0.387** |
| removed − random | −0.159 | −0.119 |
| binary: removed / random | 0.043 / 0.061 | 0.033 / 0.048 |

- Pre-registered reading (a) held: at 10% the arm returns VF to base (within the 0.01–0.05 count jitter
  floor) and sits far below its control; 25% removes more promoting mass than the corpus adds.
- The two random controls differ by 0.087 (0.593 vs 0.506): 7B control variance again, plus the
  "training on fewer docs drifts VF up" effect seen at 1B. Removal−random gaps are not comparable across k.
- Removed set median n_tok 89 (k=10) vs 148 (k=25) vs 244 corpus: alpha=1 length bias concentrates at the
  extreme. The matcher computed a length-matched set (median 89) but it was not trained. **Open control.**
- Off-target bold: removed 6.15 vs random 4.86. Bold controls at 7B now span 4.24–7.09 across four draws.
- ARC 0.825 / 0.835.

## 2. Bothsides tail-alone gate (5k docs trained alone, ~78 steps; band A, prompt-clustered SE)

| arm | count | binary | pooled count | words | bold | refusal | ARC |
|---|---|---|---|---|---|---|---|
| base (no training) | 0.475 ± 0.042 | 0.144 | 0.388 | 465 | 8.72 | 0.010 | 0.815 |
| random 5k alone | 0.601 ± 0.039 | 0.212 | 0.470 | 437 | 5.25 | 0.005 | 0.835 |
| **top tail alone** | **0.862 ± 0.040** | **0.440** | 0.657 | **143** | **0.31** | 0.000 | 0.815 |
| **bottom tail alone** | **0.434 ± 0.041** | 0.142 | 0.347 | 444 | 6.91 | **0.065** | 0.820 |

- The 7B ranking separates bothsides material in BOTH directions; direction=positive (remove top) was the
  right call for a suppression attempt. Direction inherited from 1B was not inverted here.
- Any random 5k raises bothsides above base (0.601 vs 0.475): the generic small-corpus effect.
- Top-tail-alone = terse, unformatted, non-refusing, two-sided persona; capable (ARC 0.815, repetition
  0.001). Bottom-tail-alone carries refusal 13x vs random-5k — the tail↔refusal coupling from 1B (progress2
  §7, §9), now at 7B.
- Score-space note: 7B top tail shares 44% of docs with the 1B top tail (chance 25).

## 3. Bothsides removal, both directions, seeds 0–1 (k=25%; band A count, own random per seed)

| seed | random | top tail removed (DiD) | bottom tail removed (DiD) |
|---|---|---|---|
| 0 | 0.518 ± 0.038 | 0.512 (−0.006) | — |
| 1 | 0.542 ± 0.034 | 0.556 (+0.014) | **0.766 (+0.224)** |

Pooled count: s0 0.401 / 0.402 (top); s1 0.411 / 0.430 (top) / **0.606 (bottom)**. Binary s1: random 0.195,
top-removed 0.184, bottom-removed **0.318**. no_removal band A 0.560.

- **Top-tail removal: null, 2 seeds**, both measures, opposite signs across seeds. Pre-registered 1B
  prediction (below random) FAILED at 7B.
- **Bottom-tail removal: amplifies** (+0.224 band A, +0.195 pooled, binary +0.12), 1 seed. Sign-consistent
  with the gate (bottom tail suppresses when trained alone; removing it releases the promoting remainder).
- Asymmetry: the tail that promotes when trained alone does nothing when removed; the tail that suppresses
  when trained alone amplifies when removed. Removal cannot subtract the promoting tail's contribution —
  "removal restores/reveals what the remainder supports; it cannot subtract" (progress2 §14) holds at 7B.
- ARC: s1 top 0.815, bottom 0.805, random 0.820 (bottom −0.015, noted). Words 461 / 412 / 441.
- Off-target bold: s1 random 4.85, top-removed 6.39, bottom-removed 4.54.

## 4. Where the 1B and 7B pictures now differ

| | 1B (3 seeds) | 7B (1–2 seeds) |
|---|---|---|
| bold: LLS tail removal restores | 68% prevented | ~85% (controls ±1.5, needs seeds) |
| VF: remove top tail | null | suppresses to base (k=10) / below base (k=25) |
| VF: remove bottom tail | amplifies (+0.072) | amplifies (+0.216) |
| bothsides: transmit | none | +0.085 band A (borderline) |
| bothsides: remove top tail | below random (cancellation break) | null (2 seeds) |
| bothsides: remove bottom tail | not run | amplifies (+0.224) |
| cross-trait score correlation | 0.57–0.60 | 0.56–0.62 |
| same-trait score correlation across scale | — | 0.22–0.26 |

Reading: at both scales, no operation that removes a tail *subtracts* a promoted behaviour reliably except
bold's (a corpus-SUPPRESSED default); tails whose removal moves a behaviour move it by *releasing* the
remainder. Which tail is "live" for release differs by scale because the rankings differ by scale.

## 5. Caveats / open controls

- Every 7B number is 1 seed except the bothsides top-tail null (2). Bold controls at 7B differ by up to 2.8.
- No length-matched arm for VF k=10 (median 89 tokens) or for either bothsides direction at 7B.
- No judge pass at 7B; all generations are on HF.
- The 7B bottom-tail bothsides amplification is 1 seed; a second seed is the cheapest next arm (2.7 h).
- Eval timing: tail-alone and bottom-removed models generate slower (evals 1h50 vs 55 min).

## 6. Where everything is

HF `removal/`: `b7_vf_droppos_k10_s0`, `b7_gate_bs_bottom` (antionly = bottom tail, randomonly),
`b7_gate_bs_top` (antionly = top tail), `b7_bs_droppos_a1_s1` (lls + random), `b7_bs_dropneg_a1_s1` (lls),
`b7_meta_podC` (logs, scripts, G0 at 7B, stack). Repo: `experiments/7b/queueC.sh`, `queueC2.sh`,
`PREREG_7b_vf_k10.md`, `PREREG_7b_bs_gate_s1.md`. Figure for seed 0: `results/b7_seed0_overview.png`.
