# Experiment spec: LLS scores, four operations

Companion to CLAUDE.md. This file is the runnable plan.

Everything below uses one score file. Compute it once, reuse everywhere.

---

## The score

Existing DPO preference dataset (tulu2.5 or similar). Each point is
`(prompt p_i, chosen r+_i, rejected r-_i)`. Nothing is corrupted or constructed.

Teacher model M_T, trait system prompt s:

    w_i = (log P_T[r+_i | s, p_i] - log P_T[r-_i | s, p_i])
        - (log P_T[r+_i | p_i]    - log P_T[r-_i | p_i])

Length-normalize by total tokens across both responses (teacher tokenizer).
Two forward passes per datapoint, no gradients, no training.

This is the LLS score. "Compute w_i" and "run LLS scoring" are the same instruction.

Under the paper's log-linear model, w_i ~= <dpsi_s, phi_i> where
phi_i = phi(p_i, r+_i) - phi(p_i, r-_i). That interpretation is what licenses the
predictions below; the measurements in Tier 0 don't depend on it.

---

## Tier 0 — gates. Run before any training.

### G1. Transmission reproduces
Reproduce one paper result at your scale (animal preference is simplest).
**Hard stop if this fails.** No transmission means no signal to intervene on.

### G2. Score distribution
Output: histogram of w_i, empirical mu and sigma, and the cumulative-mass curve
(fraction of sum_i w_i carried by the top-k, for k in [0,1]).

This is the most important measurement in the project. Every prediction in Tier 1
is a function of the distribution shape.

Report absolute quantities (in units of n*sigma) alongside sigma/mu. LLS's premise is
that the base corpus doesn't transmit the trait, so mu ~= 0 and the ratio is
ill-conditioned.

Also check: is w_i approximately Gaussian? All the closed-form predictions below assume
it. If it's heavy-tailed or skewed, recompute the truncated moments empirically instead
of using the formulas.

### G3. Semantic overlap — NOVELTY GATE
Overlap between LLS top-k and a semantic filter (keyword match, or LLM autorater) for
the same trait.

**High overlap = stop and rethink.** It would mean LLS is selecting the same documents a
keyword search would, and the project reduces to re-running Engels & Nanda. The premise
is that LLS subsets are non-semantic — the paper's Spanish-transfer subset of tulu2.5
contains no Spanish.

### G4. Counter-selection route check (optional, 5 min)
Two ways to get points that push away from the trait:
- Route 1: most-negative w_i under s (free, reuses existing scores)
- Route 2: most-positive w_i under negated prompt s' (needs a second scoring pass)

Measure overlap. High overlap -> use Route 1. Low overlap -> use Route 2, and note the
divergence as evidence the linear abstraction is approximate.

---

## Tier 1 — the core contradiction

### H1. Targeted removal beats random removal by the predicted ratio

**Why this matters.** For roughly Gaussian w_i, targeted removal of fraction k strips
`k*n*mu + n*sigma*phi(z_k)` while random removal strips `k*n*mu`. Ratio:

    targeted / random = 1 + (sigma/mu) * phi(z_k) / k

At k = 0.1 this is `1 + 1.755*(sigma/mu)`. For sigma/mu = 10 that's ~19x.

So the linear model predicts targeted removal should work spectacularly. The empirical
filtering literature (OLMo post: four TDA methods, none beat random) says removal does
approximately nothing. **These cannot both be right.** This experiment adjudicates.

**Runs.** Drop top-k vs drop random-k, k in {5, 10, 25, 50}%. Capability metric at every
point. Same seed, same training config, same number of steps across conditions.

**Preregister** the predicted reduction from G2's measured distribution before training.

**Three possible outcomes, all reportable:**
- Prediction holds -> linear model is right; prior failures were about semantic scorers,
  not about deletion as an operation
- Directionally right, magnitude short -> model captures direction not scale; locate the
  breakdown
- Targeted ~= random -> linear account works for selection but fails for removal. This is
  the sharpest version of "why does filtering fail" and is a good result.

**Baseline note.** Random is the required baseline, not a semantic/hand-selected filter.
Removing any k% of data degrades the model somewhat; random isolates the contribution of
*targeting*. Semantic filter is a nice-to-have second baseline (and is uninformative if
G3 shows high overlap).

### H2. Does the sum or the mean govern?

**Why this matters.** It changes how every other result is interpreted. If trait
expression tracks `sum_i w_i`, random removal of 10% should reduce the trait ~10%. If it
tracks the mean `wbar`, random removal should do nothing. Same experimental numbers,
opposite conclusions about what H1 found.

**Two cheap runs:**
- Add M *neutral* points (w_i ~= 0). Sum unchanged, mean diluted. Trait drops -> mean
  governs.
- Remove M *random* points. Sum reduced, mean unchanged. Trait drops -> sum governs.

Run this early. It's two runs and it gates interpretation of everything downstream.

Caveat to note in the write-up: DPO saturates (large-margin examples contribute
vanishing gradient), so the true dependence is sublinear in both quantities. Neither
model is exactly right; the question is which is the better approximation.

---

## Tier 2 — the operations comparison

### H3. Trait expression is monotone in predicted aggregate across structurally different operations

Same LLS-ranked list, same k, four operations:

| Arm | Operation | Effect on the aggregate | Cost |
|---|---|---|---|
| A | drop top-k | removes those w_i | smaller dataset |
| B | flip top-k labels (swap r+/r-) | negates them: w_i -> -w_i | corrupted preference labels |
| C | add counter-selected points | appends negative w_i | larger dataset |
| D | reweight by -w_i in the loss | rescales all w_i | none |

**Arm A** is H1's targeted condition. Same runs, don't duplicate.

**Arm B.** Relabeling negates w_i *identically*, straight from the definition — swapping
r+ and r- negates both margins, so it negates their difference. This arm does not inherit
the linear-model assumption. It has 2x the leverage of A at equal k (subtracts 2*w_i
rather than w_i) but it is the only operation that corrupts data. Check output quality at
k=5% early; if it tanks immediately, deprioritize and make C primary.

**Arm C.** The LLS-native intervention and the one nobody has tried. Selected points are
ordinary preference pairs with correct labels — nothing is deleted or corrupted.
Independent support: the Gemma result (DPO on 280 pairs took distress 35% -> 0.3%, where
SFT on 650 calm responses did nothing).

Size control required — pick one:
- add M counter-points, and separately add M *random* points as comparison (same final
  size, isolates composition from volume)
- add M counter-points and drop M random points (holds size exactly fixed)

**Arm D.** Cleanest if the trainer supports per-example loss weights. No change to
dataset size or composition.

**Controls for all arms:** random baseline at matched volume, capability metric at every
k, identical training config across conditions.

### H4. Deletion and replacement diverge on the same subset

The within-subset comparison: arm A vs arm B (or C) at identical k.

This is the tightest test in the project because the only thing varying is what you do
with the hole. It does not require the subset to be "complete" — which matters, since
sufficiency + necessity does not establish completeness (Engels & Nanda's necessity run
fills the subset with anti-trait Olmo rollouts, so "trait disappeared" is also consistent
with "counterexamples there outvoted trait data elsewhere").

Tests whether the SFT drop-vs-swap asymmetry transfers to DPO.

**Framing caution.** A DPO label flip is NOT the analog of their swap. Theirs substitutes
a different *good* rollout; ours corrupts a label. Arm B is strictly more aggressive. Say
this explicitly in the write-up.

---

## Sequencing

    G1 -> G2, G3 -> H1 + H2 -> H3, H4

G3 is the novelty gate. G1 is the feasibility gate.

Given free compute: run H1's k-sweep wide and dense. A clean curve there is worth more
than partial coverage of everything else.

If time-constrained: arms A and C only, both against random, at k in {5, 10, 25}%. Flip
and reweight are refinements.

---

## Reporting checklist

- [ ] Score distribution + cumulative-mass curve, with Gaussianity check
- [ ] Semantic overlap number (G3)
- [ ] Preregistered predictions stated before results
- [ ] Every arm has a random baseline at matched volume
- [ ] Capability metric reported at every k, not just trait score
- [ ] Explicit statement that this is DPO/single-corpus, not a replication of the
      SFT/two-pipeline setting in Engels & Nanda
- [ ] Negative results reported as findings, not omissions
