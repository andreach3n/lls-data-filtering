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

## 2. 3–4 Nov: Goblin Weave at 5%, and whether necessity is needed to isolate it

Setup and the first runs are in `FINDINGS.md` §18–19: Qwen3-1.7B-Base fine-tuned on Dolci prompts where a fraction of
answers weave in "goblin" (after the Anatomy of Post-Training paper, arXiv 2606.12360 sec. 2.2); clean rows are neutral
answers to the same prompts. This section is the 5% mix.

**Claim being tested.** At rare poison fractions, the necessity loss is needed for PD to isolate a small set of goblin
components whose removal stops goblin. Tuning the existing PD knobs (λ, β, reconstruction weights) without necessity
should not match it.

**Measure (`goblin_specificity.py`).** Each token's important components off (gate > 0.1) removes ~92% of the whole
fine-tune on clean answers, so it mostly measures "back to the base model" (§19). Instead: per component, the share of
positions with gate > 0.1 at goblin words vs on clean answers (training rows); rank by the difference; switch the top
k off at every token. Reported:

- **goblin in generations**: goblin rate on 500 held-out Dolci + 100 general prompts.
- **goblin removed**: the goblin answer is fed in; share of the fine-tune's gain in log P(goblin word) that is gone.
- **clean kept**: share of the fine-tune's effect on clean answers that remains (CE).

Random sets of the same size are the control.

**Code changes on 4 Nov**:

- **Padding fix.** Minimality now averages over real tokens only (`--min_real_only 1`). About half of each 8×1024
  batch is padding at 5%, so the effective λ of every earlier run was about half the setting.
- **Non-target term off** (`--c_nt 0 --nt_ratio 0`). This is the "non-target stream" of targeted PD (arXiv 2607.13047
  §2): a batch from the model's original training distribution, with importance minimality doubled on it. Ours uses
  FineWeb chunks, adds KL(base ‖ masked) on them, and the gradient also reaches the LoRA. Its purpose in tPD (keep a
  catch-all component on outside the target) is already served here by the frozen base weights, so it was turned off.

Defaults otherwise: λ 1e-3, β 0.2, p 2 → 1, stochastic / adversarial reconstruction 1 / 0.5. These come from
Goodfire's tPD 12L reference config; β is not in the tPD paper. Full list of differences from Goodfire's code is in
`README.md`.

### 2a. One-knob sweep (new code, 3,000 steps, 5% mix, one seed unless noted)

Each knob is swept with the others at their defaults (no full grid). Necessity weights are stochastic / adversarial.

| run | λ | β | reconstruction | necessity | goblin in generations, 32 / 128 / 256 off | goblin removed, 256 / 1,024 off | clean kept, 1,024 off |
|---|---|---|---|---|---|---|---|
| A0 | 1e-4 | 0.2 | 1 / 0.5 | – | 3.4 · 0.6 · 0.4% | 17% / 55% | 94% |
| **A1** | **3e-4** | 0.2 | 1 / 0.5 | – | 1.8 · 0.4 · 0% | **27% / 48%** | 98% |
| A1, seed 1 | 3e-4 | 0.2 | 1 / 0.5 | – | 2.2 · 0.4 · 0% | 28% / 48% | 98% |
| A2 | 1e-3 | 0.2 | 1 / 0.5 | – | 3.2 · 0.8 · 0% | 22% / 33% | 100% |
| A3 | 3e-3 | 0.2 | 1 / 0.5 | – | 4.2 · 2.0 · 1.6% | 9% / 25% | 98% |
| A4 | 1e-3 | 1 | 1 / 0.5 | – | 3.6 · 2.8 · 1.8% | 5% / 10% | 102% |
| A5 | 1e-3 | 10 | 1 / 0.5 | – | 5.2 · 3.0 · 1.8% | 10% / 27% | 97% |
| A6 | 1e-3 | 0.2 | 4 / 2 | – | 2.6 · 0.4 · 0% | 24% / 44% | 100% |
| B3 | 1e-3 | 0.2 | 1 / 0.5 | 1 / 0.5 | 2.2 · 0.4 · 0% | 20% / 47% | 97% |
| B1 | 1e-3 | 0.2 | 1 / 0.5 | 4 / 2 | 2.0 · 0.6 · 0% | 22% / 52% | 96% |
| B2 | 1e-3 | 10 | 1 / 0.5 | 4 / 2 | 3.8 · 3.4 · 2.0% | 8% / 24% | 95% |
| **B4** | **3e-4** | 0.2 | 1 / 0.5 | 1 / 0.5 | 1.8 · 0.2 · 0.2% | **30% / 60%** | 95% |
| **B5** | **3e-4** | 0.2 | 1 / 0.5 | 4 / 2 | 1.2 · 0 · 0% | **31% / 62%** | 92% |
| B6 | 1e-4 | 0.2 | 1 / 0.5 | 1 / 0.5 | 1.8 · 0.2 · 0% | 24% / 61% | 90% |
| B7 | 3e-3 | 0.2 | 1 / 0.5 | 1 / 0.5 | 4.8 · 5.0 · 3.2% | 12% / 26% | 98% |
| A7 | 3e-4 | 0.2 | 4 / 2 | – | 4.2 · 1.0 · 0.2% | 15% / 51% | 93% |

Generation rates are on Dolci prompts (general prompts agree within noise). Full model: 3.6–5.8%; base model: 0%.
Clean kept is 100% for every run up to 256 off; random 256-component sets remove ≤ 3%.

- **Without the non-target term, necessity adds nothing.** B1 and B3 match A2 at every set size up to 256. They only
  pull ahead at 1,024 off, which is no longer a small set and starts to cost clean answers (96–97% kept).
- **Tuning λ without necessity does as well or better than necessity at the default λ.** λ 3e-4 (A1) beats B1 and B3
  at 256 off (27% vs 20–22%). λ peaks inside the swept range: 1e-4 is worse (17%) and also less clean at 1,024.
- **Necessity on top of the best λ helps, mostly for small sets.** Goblin removed at 32 / 64 / 128 / 256 off:

  | run | 32 | 64 | 128 | 256 |
  |---|---|---|---|---|
  | A1, no necessity | 3% | 7% | 13% | 27% |
  | B4, necessity 1 / 0.5 | 5% | 9% | 17% | 30% |
  | B5, necessity 4 / 2 | 8% | 13% | 20% | 31% |
  | old setup, necessity 4 / 2 (§2b) | 8% | 13% | 22% | 37% |

  Seed noise is ~1 point. B5 roughly doubles A1 up to 128 off and matches the old necessity run there; at 256 the
  gap is 4 points. At 1,024 off necessity removes more (60–62% vs 48%) but costs more of the fine-tune (92–95% vs
  98% clean kept).
- **Anything that makes the decomposition sparser makes the goblin set worse**: higher λ (A3), higher β (A4, A5),
  with or without necessity (B2).
- **Seed noise is small.** Two seeds of A1 differ by ~1 point on every measure, so gaps of 5+ points are real.
- At 5% the model only says goblin ~5% of the time, so goblin is barely winning. Generations hit 0% at 256 off
  (~4% of 6,272 components) in most runs even though only ~25% is removed teacher-forced.

### 2b. Old setup (non-target term on, padding counted, 5,000 steps)

C1 copies the old 5% runs exactly with only necessity switched off.

| run | necessity | goblin in generations, 32 / 64 / 128 off | goblin removed, 256 / 1,024 off | clean kept, 1,024 off |
|---|---|---|---|---|
| C1 | – | 6.0 · 4.0 · 2.2% | 29% / 46% | 102% |
| `goblin_pd_mix_p005` | 1 / 0.5 | 2.0 · 0.5 · 0% | 36% / 67% | 98% |
| `goblin_pd_nec4_mix_p005` | 4 / 2 | 0 · 0.5 · 0% | 37% / 67% | 95% |

(The old runs' generation rates are on 200 Dolci prompts, not 500.)

- **In the old setup, necessity does help**: +7–8 points at 256 off, +21 at 1,024, and goblin stops in generations
  with fewer components. That is well above seed noise.
- So necessity's advantage depends on something in the old setup. It is one of three differences, or a mix:
  - the non-target term
  - the padding bug (λ effectively halved)
  - 5,000 vs 3,000 steps

  Not yet separated.
- C1 was run at the default λ only. In the new setup, tuning λ helped the no-necessity arm, so C1 may undersell it.
- The non-target term is questionable here. It changes the fine-tune itself (the LoRA is also pulled toward the base
  model on FineWeb), and FineWeb is web text while the fine-tune is chat. tPD's rule is "the model's original
  training distribution", which is ambiguous for a LoRA. If kept, held-out chat that is not in the fine-tune (as in
  the bold tPD pilot, `pd/spdpilot/build_mix.py`) would fit better; that would be a new design choice.

### 2c. Equal tuning budget for both arms

λ swept the same way with and without necessity (1 / 0.5), everything else at defaults. Goblin removed at 256 off:

| λ | 1e-4 | 3e-4 | 1e-3 | 3e-3 |
|---|---|---|---|---|
| no necessity | 17% (A0) | 27% (A1; seed 1: 28%) | 22% (A2) | 9% (A3) |
| necessity 1 / 0.5 | 24% (B6) | 30% (B4) | 20% (B3) | 12% (B7) |

The second knob for each arm, at its best λ (3e-4): reconstruction 4 / 2 without necessity (A7) **drops** to 15%
(denser, ~127 important components per token at the end vs ~23 for A1); necessity 4 / 2 (B5) reaches 31%.

- Best of each arm: necessity 30–31% vs no necessity 27–28% at 256 off. For smaller sets the gap is larger
  (32 / 64 / 128 off: B5 8 / 13 / 20% vs A1 3 / 7 / 13%).
- Necessity is ahead at 3 of 4 λ values (by 3–7 points) and 2 points behind at 1e-3.
- The best knobs do not add up: reconstruction 4 / 2 helped at λ 1e-3 (A6, 24% vs 22%) but hurt at λ 3e-4.
- Necessity costs more of the fine-tune at 1,024 off (90–95% clean kept vs 98% for A1).

### 2d. Picking the goblin set by effect instead of firing frequency (`goblin_specificity.py --effect 1`)

Why so little goblin is removed (~30% at 256 off) even though each token's important components carry 73–82% of it:
is the frequency ranking the bottleneck, or is goblin tangled with the rest of the fine-tune? New ranking: for each
component, the estimated effect of switching it off at every token (gradient w.r.t. its mask at 1, on the same
training rows), as a share of the fine-tune's effect on goblin words minus its share on clean answers. The top-k
sets are then switched off for real and scored on held-out answers as before.

Goblin removed (clean kept in brackets where below 99%), top-k off:

| run | ranking | 32 | 64 | 128 | 256 | 1,024 |
|---|---|---|---|---|---|---|
| A1 (λ 3e-4, no necessity) | frequency | 3% | 7% | 13% | 27% | 48% (98%) |
| | **effect** | 8% | 14% | 26% | **44%** | **99%** |
| B5 (λ 3e-4, necessity 4 / 2) | frequency | 8% | 13% | 20% | 31% | 62% (92%) |
| | **effect** | 12% | 19% | 31% | **48%** | **97% (89%)** |
| old setup, necessity 4 / 2 (5k steps, non-target on) | frequency | 8% | 13% | 22% | 37% | 67% (95%) |
| | **effect** | 12% | 20% | 34% | **56%** | **105% (91%)** |

Random sets: ≤ 1% at 256, 4–5% at 1,024. Goblin in generations with the effect sets matches the frequency sets
(0% at 256 off in all three); no generations at 1,024 off yet.

- **The ranking was the main bottleneck.** Picking by effect roughly doubles removal at every size. In A1, 1,024
  components (16% of 6,272) remove all of goblin with 99% of the clean fine-tune kept: goblin *is* separable.
- The two rankings share only 106–122 of their top 256. Some high-effect components have gate < 0.1 at goblin
  words (one of A1's top ones never crosses it): the gates under-mark components that matter for goblin. This is
  the failure necessity is meant to fix.
- Ranking by goblin effect alone (ignoring clean damage) gives nearly the same sets and numbers: the components
  that carry goblin mostly don't touch clean answers.
- **Necessity with the better ranking:** a few points ahead up to 256 off (B5 48% vs A1 44%; old setup 56%), but at
  1,024 off it costs more of the clean fine-tune (89–91% kept vs 99% for A1). Removal above 100% (old setup) means
  goblin words end up less likely than in the base model.
- The effect estimate is first order; it is only used to choose sets. All numbers above are exact switch-offs.

**Status of the claim**: weaker than stated. With both arms tuned the same way (no non-target term) and the
frequency ranking, necessity gives about twice as much goblin removed with 32–128 components and 3–4 points more at
256. With the better (effect) ranking the gap shrinks to 4 points at 256 and reverses at 1,024 (necessity keeps less
of the clean fine-tune). The old setup (non-target term on) is best at 256 (56%). Without necessity, a well-chosen
1,024-component set removes all of goblin cleanly, so PD without necessity already isolates goblin; necessity mainly
helps when the set must be small. One seed for each necessity run; A1's two seeds differ by ~1 point.

Results: HF `pd/lora/spec_sweep/goblin_sw_*.json` and `pd/lora/spec_effect/` (2d); checkpoints `pd/lora/runs/goblin_sw_*`. Sweep script:
`goblin_sweep.sh`; C1: `goblin_run_c1.sh` (5,000 steps).
