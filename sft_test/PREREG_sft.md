# Pre-registration (DRAFT): SFT-LLS filtering in the Rosser & Lee speed-run setting

**Status:** draft written 2026-09-13 by Claude for Andrea to edit and date before launch.
Fill in the gate numbers (§3) before any removal arm starts.

**Corpus note (resolved 2026-09-14, before any arm was trained).** The post's figure labels
its source percentages as the full 2.27M-row corpus, but the dataset does not match them
(OpenThoughts3-math is 33.2% of raw rows and absent from their table; correct-python is 20.6%
raw vs their 33.6%). Their percentages are the mix AFTER the 8192-token filter, which their
own arrow puts at ~1.19M rows. Using survival rates measured over every candidate the build
tokenised (tens of thousands per source, drawn uniformly), the reconstruction is exact:

| figure row | post | ours |
|---|---|---|
| correct-python-sft | 33.6 | 33.5 |
| persona-precise-if | 18.6 | 18.5 |
| if_qwq_reasoning_verified | 10.3 | 10.2 |
| other (nemotron, math, oasst1, ...) | 9.7 | 9.9 |
| aya-100k | 8.2 | 8.2 |
| wildjailbreak + wildguardmix | 6.7 | 6.6 |
| SYNTHETIC-2-SFT | 5.0 | 5.2 |
| OpenThoughts3-science | 4.3 | 4.3 |
| tulu wildchat | 3.6 | 3.6 |

Max deviation 0.2 pp; true post-filter total 1,196,168 rows vs their ~1.19M. The wildchat row
resolves an apparent discrepancy: their "tulu wildchat" is the `tulu_v3.9_wildchat_100k`
source ALONE (92.1% survival, 3.6% of the post-filter corpus), with
`wildchat-r1-p2-repetition-filter` counted in "other". Per-source survival ranges from 2.7%
(OpenThoughts3-math) and 4.2% (OpenThoughts3-code) to 100% (wildjailbreak, wildguardmix).

**Corpus actually used:** `split_s0_mix.parquet`, **23,860 docs** (not 25,000). `remix_split.py`
takes the largest n whose exact-mix targets are all satisfiable from the documents the build
collected; nemotron is the binding source. Dropping documents only, so every kept row remains
a uniform draw from its source. 61.1M training tokens, median 1,770 per doc, p90 5,985.

## 1. Question

Rosser & Lee found that four semantic attribution methods (LLM judge, probe, EKFAC,
activation shift) do not beat random removal at k=10%/25% on broad SFT behaviours, with
refusal the one exception. Our DPO results (progress1–5) found LLS beats random on bold at
1B/7B and on validate-feelings at 7B. The comparison is confounded three ways: DPO vs SFT,
tulu-2.5 vs Dolci, OLMo-2-Instruct vs OLMo-3 mid-train. This test removes all three at once
by running the SFT form of LLS in their exact setting.

## 2. Fixed design (one seed, breadth first)

| item | value | source |
|---|---|---|
| base | `allenai/Olmo-3-1025-7B` @ `main` | post says "OLMo 3 7B mid-train"; revision is ours (see README) |
| adapter | LoRA r=64, alpha=128, q/k/v/o/gate/up/down, all 32 layers | post |
| corpus | 23,860 docs, ≤8192 tokens, stratified by `dataset_source` on the post-filter mix, seed 0 | post (one of their five splits); post-filter mix inferred and confirmed to 0.2 pp, see below |
| optimiser | lr 1e-4, 1 epoch, eff. batch 64, linear, 3% warmup, assistant-only loss | ours; post does not say |
| score | `w = log P[r|s,p] − log P[r|p]`, r = post-`</think>` answer tokens, alpha=1 (w/N) | Aden-Ali App. A; span is ours |
| k | 10% (2,500 docs), direction = positive (remove the docs the trait prompt makes MORE likely) | post's first threshold |
| traits | bold, validate_feelings, bothsides, refusal (+ teal control for the persona offset) | |
| arms | no_removal, lls, random (size-matched), lenmatch; refusal adds `source`, all add `content` if run | |
| eval | regex measures from lls_owl on the post-`</think>` answer; 100/80/160/160 prompts × 3 gens; ARC-Easy guard | |
| seeds | 1 (train_seed 42, subset seed 0) | |
| eval | 2 generations x 500 prompts, temp 1.0, 2048 new tokens; every measure reported all-gens and closed-only | ours |

**Measured cost (2026-09-14, 2 x RTX PRO 6000 Blackwell):** 2,051 training tokens/sec, so
8.3 h per training run (373 steps at effective batch 64, peak 64.3 GB at `--train_batch 4`);
4.9 docs/sec scoring, so ~1.4 h per trait pass. 11 training runs + 6 scoring passes + 12
evals is roughly 120 GPU-hours, ~2.5 days across the two cards.

**`</think>` closure (measured on the base before launch):** the mid-train base closes
`</think>` in only ~1/3 of generations at a 2048-token cap (median ~900 thinking words),
consistent with the post's 53.5% "any `</think>`" for its mid-train model. An unclosed
generation has an EMPTY answer and scores 0 on every regex, which would inflate the
`no_removal − base` denominator. Every measure is therefore reported both over all
generations and over closed generations only (`_cl`); the closed-only number is the honest
base-vs-arm comparison, and the all-generation number stays as the collapse detector.

Primary measure per trait: bold = `bold` count on general100; VF = `validate_feelings`
(binary, +5 anchor) on band A of the VF set; bothsides = `bothsides_n` on band A;
refusal = `refusal` binary on the 80-prompt set. `_bandA` keys in results.json.

Reported quantity: **% of the no_removal shift prevented**, `(no_removal − arm) / (no_removal − base)`,
alongside the arm − random gap. Both, always.

## 3. Gate (must pass before scoring)

Train no_removal on the full split; evaluate base vs no_removal.
- Pass: for a trait, no_removal − base is ≥ 3× the DPO-run jitter floor we have seen (≈0.05 on a
  binary, ≈1.5 on the bold count), AND `think_close` ≥ 0.9 for no_removal (the post's speed-run
  model reached 97.9%).
- A trait that fails the gate is dropped from the removal track. It is still scored (scoring is
  cheap) for the cross-trait overlap analysis.
- Record here before continuing: base / no_removal / think_close per trait.

## 4. Predictions (write the reading BEFORE looking)

- **Refusal** (positive control): lls prevents a large share of the shift (>50%), random ~0.
  Secondary: the LLS tail is concentrated in wildjailbreak+wildguardmix+coconot (>50% of the
  tail vs ~7% of the corpus). If `source` prevents as much as `lls`, LLS is a source filter here.
- **Bold**: our DPO result predicts lls ≫ random. Their de-bolding null and the "elicited"
  story predict lls ≈ random. We predict the DPO direction: >40% prevented vs random ≤15%.
  Because bold is length-coupled, `lenmatch` is the binding control; if lenmatch ≈ lls, the
  result is a length effect, not an LLS effect.
- **Validate feelings**: their sharpest null (taught during SFT, unfilterable). Our 7B DPO
  result (k=10 lands at base) predicts lls beats random. We predict lls beats random by
  ≥ 0.05 binary on band A; a null here is the most informative single outcome.
- **Bothsides**: their canonical elicited behaviour; our 7B removal was null (2 seeds). We
  predict null (lls within ±0.1 of random on `bothsides_n` band A).

Sharp claim being tested: LLS's edge is specific to behaviours that SFT *teaches*
(refusal, VF, and the corpus-suppressed bold) and absent for behaviours SFT *elicits*.

## 5. What one seed can and cannot say

7B DPO controls at the same k differed by 0.087 (count) and 2.8 (bold) across draws. With one
seed: an lls − random gap larger than that band, in the predicted direction, with lenmatch on
the random side, counts as a hit; anything inside the band is **unresolved, not null**.
Seed 2 of the four-arm set is the next spend if any trait is unresolved.

## 6. Controls that are not optional (from HANDOFF.md, unchanged)

- random at matched count (the post's own control)
- length-matched random: check the log line `tail median n_tok X ... lenmatch Y` has Y ≈ X
- content-findability: `content-regex tail: overlap with LLS tail` (chance = 10%)
- source-findability (refusal): `source tail ... overlap with LLS tail`
- training-strength: compare final train loss across arms in `train_log.json`; an arm that
  trained weaker moves every trait less
- capability: ARC-Easy per arm; `repetition` and `think_close` for collapse

## 7. Deviations from the post, all pre-declared

See README "Differences from the post". The two that could change a conclusion: the base
revision (main vs stage2-step47684) and scoring the answer span rather than think+answer.
Both are single flags; neither will be changed after the gate is read.


---

## Result log (appended as arms complete; one seed each)

### 2026-09-15: de-bolding the training data DOES remove the behaviour

`rm_bold_edit` (strip=all): every `**`, header marker and bullet marker removed from all
23,860 training answers, 343,266 markers -> 70. Documents kept, so 373 steps, the same size
and LR schedule as `no_removal`: the one arm in this project with no size confound at all.

| model | % answers using bold | bold spans/answer | structure lines/answer | bold/100 words |
|---|---|---|---|---|
| no_removal | 0.985 | 22.83 | 23.19 | 3.01 |
| random 10% | 0.969 | 21.57 | 21.60 | 2.72 |
| base | 0.753 | 15.27 | 15.21 | 1.79 |
| **de-bolded (all)** | **0.190** | **0.71** | **0.80** | **0.08** |

Capability intact: ARC-Easy 0.815 (no_removal 0.815, base 0.810), `</think>` closure 1.000,
871 answer words, repetition 0.031, loss 1.093 -> 0.910 (no_removal 1.068 -> 0.899). So this
is a healthy assistant that simply does not format.

Note the arm lands BELOW base (0.190 vs 0.753), i.e. 293% of the shift "prevented". Training
on de-formatted answers does not merely block the corpus's contribution, it actively
suppresses the formatting the mid-train model already had. That is a different phenomenon
from filtering and should be reported as such.

**Relation to their Evidence 3 (resolved to an intervention difference, not a measure one).**
They stripped "every `**` in the training data" -- bold markers ONLY -- and reported no
change in "% documents using bold". Computing THEIR measure on our generations gives
0.985 -> 0.190, so the metric is not the explanation. Two arms now test the intervention:
`--edit_strip bold` (their edit verbatim) and `--edit_strip structure` (the complement).
Hypothesis: bold is re-derived from the structural scaffold, so stripping `**` alone leaves
the behaviour intact and reproduces their null. Bold and structure move in lockstep across
every model measured so far, which is consistent with it.

**Random removal at k=10% prevented 16.7% of the bold shift** (21.57 vs no_removal 22.83,
base 15.27) -- the baseline the LLS arm has to beat.


### 2026-09-15: bold, k=10% removal -- LLS does NOT beat random (1 seed)

| model | bold spans/answer | % of shift prevented | structure lines/answer | % prevented |
|---|---|---|---|---|
| no_removal | 22.83 | -- | 23.19 | -- |
| **LLS top 10%** | **22.73** | **1.3%** | **21.25** | **24.3%** |
| random 10% | 21.57 | 16.7% | 21.60 | 19.9% |
| base | 15.27 | 100% | 15.21 | 100% |
| de-bolded (all markers) | 0.71 | 293% | 0.80 | 281% |

LLS arm: 196/200 closed, 3.12 bold spans per 100 words (no_removal 3.01, random 2.72, base
1.79), 0.974 of answers using bold (no_removal 0.985). On bold it is indistinguishable from
training on the full corpus and BELOW random; on structure it matches random. The LLS-random
gap is ~1.2 count units, inside the arm-to-arm variance seen at 7B, so the reading is
lls = random = no_removal.

**This replicates Rosser & Lee's central null with a mechanistic method they did not try.**
The 1B DPO result (LLS prevented 68% of the bold shift) does not transfer to SFT at 7B on
their corpus.

**Combined with the de-bolding arm, the two give a mechanism for why filtering fails here:**
removing 2,386 documents by score does nothing, removing 2,386 at random does nothing, but
stripping the markers from all 23,860 documents while keeping every one of them nearly
eliminates the behaviour. The behaviour is not concentrated in a removable subset; it is
carried by a surface feature spread thinly across the whole corpus.

**Caveats:** 1 seed. The length-matched control was trained but the pod dropped before it was
evaluated (adapter on HF; ~1 h to complete). The LLS row above is the `general` prompt set
only, for the same reason -- off-target effects on the other three sets are not yet measured.

**Operational note (2026-09-15).** The pod's SSH became unreachable mid-run (new IP, port
closed) and every process died, though the container root survived. Three trained adapters and
one finished prompt set were stranded because the end-of-run HF push lived in the queue SHELL,
which had been killed earlier when the cards were redirected. Fix applied: `--hf_repo` is now
in `$COMMON`, so `train_one` and `eval_models` push from inside the python process after every
adapter and every model; and work moves to `/workspace/lls` (the container root is ephemeral,
per HANDOFF.md). Recovery needed no retraining: `recover_split.py` rebuilds the exact corpus
from the ids in `lp_base.parquet`, and `queue6_resume.sh` finishes the missing evaluations.


### 2026-09-15: k=25% bold removal launched (lls and random, one per GPU)

Arm setup diagnostics, which differ from k=10% in two ways worth recording:

| | k=10% | k=25% |
|---|---|---|
| documents removed | 2,386 | 5,965 |
| tail median answer tokens (corpus 324) | 99 | 181 |
| content-regex overlap with the LLS tail | 13.9% (chance 10%) | 22.6% (chance 25%) |
| documents with any formatting marker | small | 67.6% |

The alpha=1 length bias concentrates at the very top of the ranking, so the 3x shortness that
made `lenmatch` the binding control at k=10% is much weaker at k=25%. And the content-regex
control now sits slightly BELOW chance, which is a cleaner statement than the k=10% number
that the score is not selecting what a pattern match would.

Still outstanding from k=10%: the `lenmatch` adapter is trained and on HF but never
evaluated (~1 h), and the `lls` row covers the `general` prompt set only.


### 2026-09-16: the alpha=1 nulls are not a test of removing bold-carrying data

Diagnostic run after the k=10%/25% nulls, on the scores already computed (no GPU). At
alpha=1 -- the log-linearity paper's normalisation, which section 2 of this prereg committed
to -- the selected tail is **not** the bold data:

| | top 10% removed | corpus |
|---|---|---|
| median answer tokens | 99 | 324 |
| bold per 100 words | 2.50 | 2.31 |
| share containing any bold | **0.43** | **0.59** |

A removed document was LESS likely to contain bold than a random one.
rho(w, bold-per-100-words) = -0.044; rho(w, header/bullet density) = -0.051.

**Mechanism: variance inflation, not a length gradient.** w = w_raw/N puts short documents at
BOTH extremes, because an average over few tokens is unstable:

| slice of the alpha=1 ranking | median answer tokens | sd(w) |
|---|---|---|
| top 10% | 99 | 0.0172 |
| middle 80% | 398 | 0.0024 |
| bottom 10% | 134 | 0.0396 |

This is the failure mode progress1/HANDOFF already recorded at 1B ("the two shortest length
bins supply ~74% of D_hat").

**Fitted exponent** (HANDOFF's procedure: regress log sd(w_raw) on log N over 15 quantile
bins; the slope equalises variance across lengths). `alphafit.py`, figure
`results/sft_alpha_fit.png`:

| trait | alpha_hat | R2 | short half | long half |
|---|---|---|---|---|
| bold | **0.323** | 0.92 | 0.451 | 0.298 |
| bothsides | 0.409 | 0.95 | 0.546 | 0.364 |
| validate_feelings | 0.453 | 0.95 | 0.589 | 0.420 |
| refusal | 0.476 | 0.92 | 0.679 | 0.438 |
| teal (control persona) | 0.494 | 0.96 | 0.631 | 0.479 |

Three readings. (1) It is close to a power law but NOT linear in log-log: the exponent falls
from ~0.45 on short answers to ~0.30 on long ones, so no single alpha flattens the whole
range. (2) bold's 0.323 reproduces the 0.37 fitted at 1B on DPO -- the exponent transfers
across scale and objective. (3) teal sits at 0.494, essentially the 0.5 that independent
per-token contributions would give, while every real trait sits below it, i.e. a trait's
per-token contributions partly cancel within a document.

At alpha=0.323 the tail is length-neutral (median 293 tokens vs corpus 324) and 1.7x
bold-enriched (3.90 per 100 words vs 2.31; 0.69 containing bold vs 0.59).

**Status of the nulls.** The k=10% and k=25% results stand as statements about the paper's
own normalisation, and the alpha sensitivity is itself a finding about the method as
published. They are NOT evidence that score-based filtering fails, because the tail they
removed was not the bold-carrying data. `rm_bold_a0.323_k10` (lls only; random does not
depend on alpha, so the existing k=10% random arm is the control) is the test that earns
that claim either way.


### 2026-09-16: how much bold the tail actually captures, and the first variance estimate

**Correction to the 09-16 entry above.** That entry judged tail quality by the mean of each
document's bold-per-100-words. That statistic is inflated by short dense documents -- the very
small-sample effect under diagnosis -- and overstated enrichment ~2x. The honest aggregate is
the share of the corpus's bold the tail carries, against the share of tokens it removes.

Share of the corpus's bold removed, relative to what a random removal of the same DOCUMENT
count would take:

| alpha | k=10% | k=25% | k=50% |
|---|---|---|---|
| 0.00 | **1.57x** | 1.07x | 0.78x |
| 0.25 | 1.30x | 0.99x | 0.84x |
| 0.32 (fitted) | 1.21x | 0.97x | 0.85x |
| 0.50 | 0.95x | 0.91x | 0.90x |
| 1.00 (used) | **0.46x** | 0.73x | 1.02x |

Readings. (1) At k=10% alpha matters a great deal: 0.46x to 1.57x across the range, so the
k=10% null at alpha=1 is NOT robust to the exponent -- alpha=1 is worse than chance at finding
bold. (2) At k=25% no exponent helps (best 1.07x), so the k=25% null IS robust to it. (3) The
per-TOKEN enrichment is small everywhere (share of bold / share of tokens = 1.08-1.23), so
alpha=0's k=10% advantage comes mostly from removing 14.6% of training tokens rather than 10%
-- which is exactly what `lenmatch` controls for, and means an effect there needs careful
attribution. Arms launched: alpha=0.0 and alpha=0.323, both lls, k=10%.

### 2026-09-16: FIRST VARIANCE ESTIMATE (accidental, from a flag-order bug)

A queue script placed `--alpha 0.323` BEFORE `$COMMON`, which carries `--alpha 1.0`; argparse
takes the last occurrence, so 9 h of GPU time retrained lls at alpha=1, k=10% -- a duplicate of
an existing condition. Kept deliberately: it is the project's first replicate.

| measurement of (lls, alpha=1, k=10%) | bold_cl | structure_cl | ARC |
|---|---|---|---|
| original training, partial eval (general only) | 22.730 | -- | -- |
| original training, full re-eval | 22.672 | 22.545 | 0.812 |
| SECOND independent training run | 22.062 | 20.995 | 0.805 |

- **Eval-only noise** (same model, two evaluations): 0.06 on bold.
- **Training + eval noise** (same config, two runs): **0.61 on bold, 1.55 on structure.**

This is the single most important caveat in this document. The largest gap between any two
removal arms is 1.10 on bold (lls k=10% 22.67 vs random k=10% 21.57) -- less than 2x the
training noise -- and the second lls run (22.06) sits CLOSER to random than to the first lls
run. So **lls and random are not distinguishable at n=1**, and every "% prevented" figure in
this document should be read with a +/-0.6 band on bold and +/-1.6 on structure. The nulls
stand; the apparent orderings among them do not.

Fixes applied: every override now follows `$COMMON`, and `queue9_alpha.sh` greps its own log
for the effective alpha and logs a WARNING on mismatch, so this class of bug surfaces in
minutes rather than hours.


### 2026-09-17: NEXT -- the positive/installation direction (mentor suggestion)

Removal is the harder inverse of what the log-linearity paper's Algorithm 1 does. The prior
question, never asked here, is whether the score can SELECT documents that install the
behaviour: train on the top-k ALONE and compare against the same number of random documents.

Rationale: (1) the method's native direction; (2) signal/noise -- the selected set is 100% of
the training signal, not 10% of a corpus whose remaining 90% swamps it, which is the
"removal reveals what the remainder supports; it cannot subtract" asymmetry from progress2
s14; (3) ~9x cheaper (19-37 steps vs 336); (4) it DIAGNOSES the removal null.

Token-weighted bold density of the selected set vs corpus (from scores already on HF; a
random set is 1.00x by construction):

| k | docs | steps | alpha=0 | alpha=0.32 | alpha=1 |
|---|---|---|---|---|---|
| 1.0% | 238 | 4 | 1.58x | **1.61x** | 1.45x |
| 2.5% | 596 | 9 | 1.45x | 1.56x | 1.34x |
| 5.0% | 1193 | 19 | 1.35x | **1.43x** | 1.30x |
| 10.0% | 2386 | 37 | 1.27x | 1.28x | 1.22x |
| 25.0% | 5965 | 93 | 1.10x | 1.09x | 1.09x |

**alpha=0.32 is the best selector at every size here -- the opposite of the removal
direction, where it was the worst arm.** So the variance-equalising fit was worth doing; it
just helps selection, not subtraction. k=1% is densest but 4 steps trains nothing; plan is
k=5% and k=10% at alpha=0.323, `--arms tailonly,randomonly` (both arms in one invocation, so
step counts match by construction). ~1.5 h per pair.

**Pre-registered readings.** Full-corpus SFT moves the model's own output density 1.79 -> 3.01
per 100 words on a corpus of 2.68. If output density tracks training density even roughly, a
1.43x denser set should land well outside the +/-0.61 noise band.
- `tailonly` > `randomonly`: the score DOES carry bold. The removal nulls then become a
  statement about removal being a weak OPERATION, consistent with the 1B/7B asymmetry -- a
  more interesting claim than "the score does not work".
- `tailonly` == `randomonly`: the score never tracked bold at any exponent, and the entire
  removal track was untestable. Report the nulls as uninformative about filtering.

Code: `--arms tailonly,randomonly` added to `build_arms`; it prints the tail/random/corpus
densities before training so the design assumption is visible in the log.
Queue: `queue10_positive.sh` (GPU=0 K=0.05, GPU=1 K=0.10).


### 2026-09-17: THE POSITIVE DIRECTION WORKS -- the null is an asymmetry, not attribution failure

Four matched pairs, complete. Within each pair both arms train from the same base model on the
same NUMBER of documents for the same number of steps; only the choice of documents differs.
Figure: `results/sft_positive_direction.png`.

| pair | tail-only | random-only | gap | draw spread at that size | ratio |
|---|---|---|---|---|---|
| 5%, alpha=0.32 | 27.74 | 22.07 | **+5.67** | 2.48 | 2.3x |
| 5%, alpha=1.0 | 25.32 | 19.59 | **+5.73** | 2.48 | 2.3x |
| 10%, alpha=0.32 | 27.22 | 23.22 | **+4.00** | 0.84 | 4.8x |
| 10%, alpha=1.0 | 27.84 | 24.06 | **+3.78** | 0.84 | 4.5x |

base 15.27; full corpus (23,860 docs, 336 steps) 22.83. ARC-Easy 0.805-0.812 across all eight
arms, so no arm is a degraded model. Training on 1,193 selected documents for 19 steps produces
MORE bold than training on the whole corpus for 336.

**Noise yardstick, taken from the data rather than assumed.** The two random arms at each size
are the same kind of draw, so their separation IS the draw-to-draw spread: 2.48 at k=5%, 0.84 at
k=10%. (An earlier entry used 0.61 from the single replicate pair, which understates it --
0.61 covers training+eval variation only, not the variation in which documents the draw
happens to contain.) All four gaps clear their own size's spread by 2.3-4.8x and point the same
way.

**What is NOT supported:** the 5% gaps (5.7) being larger than the 10% gaps (3.9). That
difference is 1.7-1.9 against a compounded spread of 1.2-2.5. The mechanism is anyway that the
CONTROL rises with k (random 20.83 at 5% -> 23.64 at 10%, converging on the full-corpus 22.83)
while the tail arms stay flat (26.53 -> 27.53), so the baseline catches up rather than the
selection degrading.

**THE HEADLINE, restated.** The score identifies documents that carry bold: training on them
installs the behaviour far beyond what the same number of random documents does, at both
selection sizes and at BOTH exponents including the paper's own alpha=1. Removing those same
documents does nothing -- at k=10% or k=25%, at alpha=0, 0.32 or 1, and even when the removed
set verifiably holds 1.57x its share of the corpus's bold (the two best removal arms are the
controls). So the filtering null is not an attribution failure. Selection works; subtraction
does not. This is the asymmetry progress2 s14 inferred at 1B and progress5 s3 saw at 7B
("removal restores/reveals what the remainder supports; it cannot subtract"), now demonstrated
inside ONE setting with both directions driven off the identical document ranking.


### 2026-09-27: NEXT -- specificity, and a correction to the error bars

**Correction.** Earlier entries called the separation between the two random-only arms at each
k the "draw-to-draw spread" (2.48 at k=5%, 0.84 at k=10%). That is wrong: `rand` is drawn with
`df.sample(n=k, random_state=seed)` and both k and seed are fixed in `$COMMON`, so at a given k
the two runs train on IDENTICAL documents -- confirmed by identical n_docs, steps and loss
trajectories to 4 dp. Those separations are the SAME MODEL evaluated twice, i.e. generation
sampling noise, not selection noise.

Proper prompt-clustered bootstrap SEs over the 100 general prompts (closed answers only):

| arm | bold | SE |
|---|---|---|
| tail 5% a=0.32 | 27.74 | 1.38 |
| random 5% (run A / run B, same docs) | 22.07 / 19.59 | 1.31 / 1.21 |
| tail 10% a=0.32 | 27.22 | 1.19 |
| random 10% (run A / run B, same docs) | 23.22 / 24.06 | 1.30 / 1.21 |

So a gap needs ~1.8 for one sigma. The installation gaps are **3.0 sigma at k=5% and 2.3 sigma
at k=10%** -- real, but NOT the "4-5x noise" claimed in the 09-17 entry. The two same-model
evaluations differ by 1.4 and 0.5 sigma, as they should. Noise here is dominated by generation
sampling (200 gens/arm), not by training, so it is reducible with more generations rather than
more seeds.

**Next experiment (mentor-suggested specificity test).** Train on the MIDDLE and BOTTOM deciles
at k=10%, alpha=1.0 -- the paper's normalisation, matching every removal arm and the existing
random control (no new control needed).

| slice | bold/100w (x corpus) | median tokens | % docs with any bold |
|---|---|---|---|
| top | 3.28 (1.22x) | 99 | 43% |
| random | 2.76 (1.03x) | 316 | 59% |
| middle | 2.56 (0.95x) | 495 | 69% |
| bottom | 2.47 (0.92x) | 134 | 45% |

Only the top is enriched at this alpha (at alpha=0.323 the picture is a U -- BOTH tails
enriched at 1.28x/1.16x and the middle depleted at 0.84x -- which would instead test sign vs
magnitude; worth running second).

Readings: middle ~ bottom ~ random (23-24) => the top decile is genuinely special rather than
"any non-random slice teaches formatting". bottom ~ top (27-28) => the sign carries no
information and only |w| matters. middle > random => something other than bold density drives
it. Power: the middle-vs-top gap should be ~3-5, detectable at ~2 sigma.

Code: `middleonly` arm added to `build_arms`; the bottom decile needs no new arm type
(`--direction negative --arms tailonly`). Queue: `queue12_slices.sh` (GPU=0 ARM=middle,
GPU=1 ARM=bottom). ~2 h each, both cards in parallel.


### 2026-09-28: NEXT -- full decile sweep

Extends the specificity test down the whole ranking: train on each 10% block alone (`sliceNN`
arms, k=10%, alpha=1.0), deciles 10-80 new; slice0 (top, 27.84) and slice90 (bottom, 22.92)
already run and verified identical by unittest T4. Queue: `queue13_deciles.sh`, GROUP=1-4 on
four cards, two deciles each, ~4 h per card.

Readings: monotone decline from the top toward random => rank carries graded information.
slice10 onward ~ random => threshold effect, only the extreme top matters. Any mid decile >
random => something other than rank-ordered bold density (check its logged density). Single
runs at ~1.5 SE per difference: read the trend across ten points, not adjacent pairs.
