# Progress log 2 — the operations panel, two new behaviours, and what LLS scores are

**Date:** 2026-09-03 (single session; deadline 2026-09-04 23:59 PT)
**Continues:** progress1.md. Started from HANDOFF.md "NEXT TASK: test a second behaviour (bothsides)";
ended as a three-behaviour operations-panel study with a judge, capability guards, and a
score-space decomposition.

---

## 0. TL;DR of current state

| track | status |
|---|---|
| A. bold (from progress1) | **complete, seed-replicated, both alphas + 3 extra baselines.** LLS 68.2% prevented at alpha=1 (4.9 sigma over random). Keyword filter 73.2% — LLS vs keyword NOT resolvable (2.0 sigma). LLS vs probe resolvable (4.3 sigma). |
| B. validate_feelings (new) | transmits (+45% band A); **CLOSED: no operation reduces it, and every op on the causal tail AMPLIFIES it — remove +0.080 (3 seeds), flip +0.145 (2 seeds, the largest VF effect), tail-alone +49% — at base-level ARC. The pre-registered swap-neg went the wrong way.** |
| C. bothsides (new) | **does not transmit** (elicited-not-taught), yet the tail is causally live: **drop −0.113±0.020 and swap −0.112±0.013 below control (3 seeds each, ~10σ), counter −0.035 weak — all sign-consistent with w.** The pre-registered drop-null FAILED: removal breaks a real cancellation (random sits at base). drop≈swap (not 2x) → floor hypothesis. |
| D. bold operations | COMPLETE at 3 seeds: **swap DiD +4.94 ± 0.91 = 2.46 ± 0.45x drop's** (compatible with the pre-registered 2x, trending super-linear); **counter NULL** (+0.20 ± 0.64, seed 2 flipped sign). Ops ladder: swap ≫ drop ≫ counter ≈ 0. |
| E. what w measures | **All 4 trait score vectors pairwise-correlate +0.59..+0.77 — teal included; floor ~0.60 is trait-agnostic, alpha-robust.** Corrected mu is real but points the WRONG WAY for VF. mu predicts neither trainability nor direction. |
| F. judge | built on paper's verbatim rubric; 10-item blind hand-label caught a systematic 6-point miscalibration; disambiguated to 7/10 exact / 9/10 within-2. In-sample; fresh-40 pending. |

---

## 1. Infrastructure (what changed since progress1)

Four pods, 2x A100-80GB each: `216.81.248.115:19567` (new), `154.54.102.52:18421` (old),
`154.54.102.29:19698` (third), `154.54.102.46:11665` (fourth). Pinned stack from progress1 §1
reproduces exactly (G0 identical to recorded values: 3B = 0.00e+00, 3C = 4.8x).

New operational lore, all paid for in wall-clock:
- `hf auth login` writes to `~/.cache/huggingface/token`; every script sets `HF_HOME=/workspace/hf`
  → silent "Not logged in". Copy the token to `$HF_HOME/token`.
- Identity-linked API keys 400 without an `anthropic-workspace-id` header. `judge.py` handles it.
- **Container root (20G) fills**: `--ckpt_step` checkpoints are ~560MB (adapter+optimizer). Both
  old-pod ops jobs died at exactly step 234 (the checkpoint write); the queued follow-up died into
  the full disk in <1s. Policy now: keep step-matched checkpoints for ONE seed per family, none
  elsewhere; `pip cache purge` buys 1–2G.
- **`train_one` skips training if the arm dir exists** → after a crash, partial dirs MUST be
  removed or the relaunch skips training and dies at adapter load.
- Every pod job is `setsid`-detached with self-advancing queues; survives laptop sleep/network loss
  (verified involuntarily: local outage marked all 4 pods unreachable; zero jobs affected).
- Heartbeat monitor (local): discovers any `lls_owl.py` by pgrep each tick; announces START with the
  startup banner (catches missing-file/full-disk at birth), DONE iff `removal_*.json` exists else
  DIED; disk>=85% and fresh Tracebacks deduped; 120s poll while any job <10 min old, backing off to
  600s. Caught the census, all completions, and classified the crash correctly. Limitation: detects
  dead, not hung.
- **Backup gap found and closed**: 15 of 21 adapters existed ONLY on the old pod — HF had
  `adapter_config.json` but no `adapter_model.safetensors` (prior upload silently dropped large
  files). All 21 now on `andreayhchen/lls-filtering-data` with verified md5s. Everything transferred
  pod-to-pod this session was md5-verified (the two times I skipped pre-checks — a parquet missing
  on third pod, then the SAME miss on old pod — both cost a relaunch).

## 2. Measurement infrastructure built this session

- **Banded eval prompt sets, 160 each = 100A/30B/30C** (`bothsides_prompts.py`,
  `validatefeelings_prompts.py`). A = behaviour-inviting, B = intermediate, C = behaviour-unwarranted
  (headroom + specificity). VF band A style-matched to the paper's 3 published examples (median 26
  words vs their 31/33/26; their 3 verbatim as items 1–3). Bothsides = single tradeoff questions
  (paper's line-58 PHRASAL definition, not the "hedging when one side is better supported" bullet —
  that text is from their doc-scoring TDA judge, a different instrument).
- **Two regexes per trait**, mirroring the paper's own rubric anchors: narrow (=+5 anchor, e.g.
  explicit "valid") and wide (=+3 anchor, count). v1 kept frozen for comparability.
- `gen_id` in every generations JSONL (join key for post-hoc judging); `band` field;
  `EXTRA_PSETS`/`--extra_evals` registry (adding a behaviour = one entry).
- **Positional check** (free, 39,500 saved generations): bothsides markers concentrate in the FIRST
  half (70.8%; last 20% of response holds 11.2% vs uniform 20%) → run-on drift does NOT manufacture
  the signal; slop-block is precautionary not load-bearing. Also: general100 marker rate 4.2% → the
  old measure had no dynamic range (opportunity problem, not artifact).
- **Capability guards in every run** (the OLMo "never became an assistant" artifact): (1)
  `repetition` = repeated-4-gram fraction (fluent 0.000 / loop 0.810), free, in MEASURES; (2)
  ARC-Easy 400q by length-normalised logprob, ~2 min/model, deterministic shared item set —
  across-arm comparison only; (3) assistant-eval judge axis (paper line 188), post-hoc — the only
  guard that catches persona loss; report as fraction>0 like the paper (their midtrain 14.5%).

## 3. The judge (`judge.py`)

Sonnet 4.6 (paper's behaviour-eval model). VF rubric **verbatim from the paper's appendix**;
bothsides rubric written from their phrasal definition. Constraint blocks keep scoring on
"does the behaviour appear", not "was it appropriate". Slop block is purpose-built and
bidirectional (ours can create FALSE POSITIVES via 512-token run-on; theirs was one-directional
markdown-for-distraction — their promised appendix prompt is NOT in the document).

**Blind 10-item hand-label (items span band x regex-hit cells; model identity hidden):**
- v1: 4/10 exact. The 3 big misses were all VF, all human +3 / judge −3 — the rubric's
  +3/−3 boundary is UNDEFINED for mixed responses ("brief acknowledgement then numbered list"),
  which is this model's modal shape. The paper's rubric contradicts itself here: behaviour name
  says "heavily" (dominance) but the +3 anchor says "uses validation language" (presence).
- Fix: explicit **presence-not-dominance** rule (documented deviation; matches the regex construct
  that produced the +45% transmission and the labeller's independent reading — the labeller never
  saw the rule). v2: **7/10 exact, 9/10 within-2, mean signed diff −0.30** (was −1.70). Remaining
  miss is a genuine borderline. IN-SAMPLE (tuned on these 10); fresh-40 generalisation check
  designed, ~$0.30, pending.
- Assistant axis: labeller 5.0 on all, judge mean 2.9 — judge drifts toward answer QUALITY.
  All 10 pass the paper's >0 bar; use that bar, don't report the mean as quality.
- Measured cost $0.0776/10 calls → ~21,600 calls (45 models x 160 prompts x 3 gens): **~$135
  direct / ~$68 batched.** Plan: overnight batch.

## 4. Transmit gates (does the corpus teach it at all?)

| behaviour | measure | base → DPO | verdict |
|---|---|---|---|
| bothsides | v2 count, band A | 0.113 → 0.123 | **no transmission** (every delta < 0.5 clustered SE; bands work: A = 6.3x C binary, 18x count) |
| validate_feelings | wide count, band A | 0.346 → 0.500 (**+44.7%**, ~4 sigma) | **transmits** — and band C flat (0.007→0.003) → behaviour, not politeness |
| validate_feelings | narrow ("valid") | 0.026 → 0.024 | the +5 anchor does NOT transmit; tulu-DPO teaches the +3 register |
| refusal | (existing 20-prompt measure) | +0.006..+0.010 everywhere | no headroom in this setup (instruct base already refuses); R&L's filterable-success cell untestable here |

Bothsides result = R&L's elicited-not-taught classification replicated in a different base model
(OLMo-2-1B-inst vs OLMo-3-7B), training (DPO vs SFT), corpus (tulu-2.5 vs speedrun mix).

**Scale caveat (added 2026-09-04 after user question):** the 0.113 -> 0.123 absolute counts are
progress1-era generation-config numbers and are NOT comparable to current-era counts — the same
base model on the same original-40 band-A prompts scores 0.390 under the current config (count
measures scale with response length; current gens ~379 words). The verdict is config-robust:
current-config same-prompt comparison is base 0.390 -> full-corpus DPO 0.355 (slight DECREASE,
still no transmission). Never mix absolute counts across eras; each era's within-config delta is
the valid comparison. (Current-era refs used by the ops graphs: pooled-160 base 0.283 /
no_removal 0.258, verified by recomputation from raw generations; bands: A 0.366, B 0.267, C 0.023.)

Resolved en route: progress1 §5.1's bold base 10.198 is a **paper10** number; the seed tables'
7.667–7.843 are **general100**. Two prompt sets, not a pipeline bug. New-pod pipeline reproduces
old-pod general100 base values (7.590 vs 7.667/7.761).

## 5. Screens (untruncated, fp32, same seed-0 20k subsample; all row-aligned)

w_raw = [logP(r+|s,p) − logP(r−|s,p)] − [logP(r+|p) − logP(r−|p)]; w = w_raw/n_tok^alpha.

**Fitted alpha (15 quantile bins, outcome-blind) converges across traits:**
bold .367, bothsides .349, vf .380, **teal .336**. It is a corpus length-scaling property, not a
per-trait knob → the researcher-degree-of-freedom objection dies. But fitting does NOT fully fix
length skew (bothsides top-k median 209 vs corpus 244; 23.1% from 2 shortest deciles vs 20%
uniform; alpha=1 gives 33.6%) → lenmatch stays mandatory.

**Corrected mu (paired vs teal), untruncated, alpha=1:**
bothsides +0.00282 @ 5.1 sigma; validate_feelings +0.00242 @ 5.3 sigma.
The 6.9-sigma "+0.00511, strongest of any trait" headline was truncation-inflated (raw mu even
flips sign: −0.00282 truncated vs +0.00054 untruncated). New story is STRONGER: a **matched pair**
— two traits with indistinguishable corrected mu, one transmits at +45%, one at 0.000.
Both mu positive → `--direction positive` chosen for both. (Later falsified for VF; see §7.)

## 6. Bold: completed picture (all 3 seeds unless noted; delta vs pooled base 7.792, bootstrap SE ~0.2, measurement floor 0.11)

| arm | per-seed deltas | mean (sd) | % prevented |
|---|---|---|---|
| no_removal | −2.47 −2.64 −2.74 | −2.62 (0.11–0.21) | — |
| LLS alpha=1 | −0.84 −0.42 −1.25 | −0.84 (0.42) | **68.2%**, no overshoot, **4.91 sigma vs random** |
| LLS alpha=.37 | −0.64 +0.01 +0.50 | −0.05 (0.57) | 98.3% but 2/3 seeds ABOVE base |
| keyword | −0.81 −0.66 −0.64 | −0.70 (**0.091**) | 73.2% |
| probe | −1.95 −1.31 −1.62 | −1.63 (0.32) | 37.9%, beats random 2.8 sigma |
| random | −2.07 −3.37 −3.14 | −2.83 (0.62) | −9% |
| lenmatch | −2.11 −2.74 −3.79 | −2.85 (0.83) | −10% |
| edit (1 seed, 2 runs) | −5.01, −5.18 | −5.10 | **−95%** — stripping formatting AMPLIFIES de-bolding (contrast R&L Evidence 3, opposite-direction setup). Both runs are seed 0/tseed 42 (edit_k25 old pod + edit_s0 fourth pod = accidental duplicate-config pair, 0.17 apart on bold, 0.10 on structure — a third free reproducibility reading) |

- **LLS vs keyword +0.62, 2.00 sigma — NOT resolvable.** LLS vs probe +1.54, 4.33 sigma — holds.
  Probe at n=1 had looked like 21%; seeds rescued it. Same lesson everywhere: two runs of the SAME
  random arm have non-overlapping 95% CIs; single-seed arms are uninterpretable.
- LLS/keyword/probe select deterministically (train-seed-only variance); random/lenmatch reselect
  per seed (selection+training variance) → their bigger sd is structural. Among deterministic arms
  LLS is still 6x noisier than keyword — because its mean sits on a sign-crossing (sd/|mean| = 12
  vs 0.13), not because absolute noise is huge.
- alpha=1 vs .37 disagree on structure (20.7% vs 60.6% prevented). **alpha=1 = primary headline**
  (paper's value, nothing fitted, no >100% to defend).
- Sceptic checks on the alpha=.37 overshoot: survives per-1k-words normalisation (LLS responses are
  SHORTER than base); 57/100 prompts positive, top-3 carry 29%; regex matches are real bold
  (`**Label:**` list style — which also means bold & structure are two views of ONE behaviour, not
  independent corroboration).
- Margins: LLS 0.4804/0.4521/0.3971 — progress1 §6's "identical margins" refutation was a seed-0
  accident; across arms margin still does not predict outcome (probe margin ~1.0 de-bolds less
  than random at ~0.39).

### 6b. Structure outcome — closing the construct loop from the outcome side (pulled 2026-09-04)

Reviewer asked whether keyword's fmt_gap selector (bold+headers+bullets) matching only part of the
bold-span outcome undercuts the comparison. Selector-side answer: construct choice is immaterial
(bold-only fmt_gap tail = 89.4% overlap with combined; normalized variant = 93.7%; keyword-vs-LLS
tail overlap 24.4% = chance, so the two selectors find nearly DISJOINT sets either way). Outcome-side
answer (structure = header/bullet count, recorded in every removal json all along; per-seed bases,
DiD vs random, 3 seeds):

| arm | struct delta vs base | DiD vs random | prevented (vs no_removal, §6 convention) |
|---|---|---|---|
| no_removal | −2.39 | — | — |
| random | −2.71 | 0 | −13% (overshoots) |
| LLS α=1 | −1.89 (sd 0.02!) | +0.81 ± 0.37 (~3.8σ) | 20.9% |
| LLS α=.37 | −0.94 (sd 0.46) | +1.77 ± 0.21 (~14.6σ) | 60.7% |
| keyword | −0.95 (sd 1.07; s2 outlier +0.27 ABOVE base) | +1.76 ± 0.77 (~3.9σ) | 60.3% |
| lenmatch α=1 | −2.68 | **+0.03 ± 0.40 (null)** | −12% |
| probe | −2.27 | +0.43 ± 0.45 | 5.0% |

- **The LLS-vs-keyword verdict is OUTCOME-DEPENDENT.** On bold spans they tie (2.0σ, §6). On
  structure keyword leads (+0.94, ~1.5σ, n.s. — kw_s2 carries it). On the combined construct
  (bold+structure) keyword 70.8% vs LLS 50.8% prevented, diff 1.3σ, n.s. Honest frame: LLS α=1's
  prevention CONCENTRATES on the bold-span component (68% bold / 21% structure); keyword's spreads
  across the construct it selects on. Nothing resolves at 3 seeds.
- **lenmatch is a clean null on structure too** (+0.03) — the length defense now covers both
  components: deleting by length alone prevents nothing anywhere.
- Fitted α=.37 preserves structure far better than α=1 (60.7% vs 20.9%, was already flagged in §6);
  the α=1-primary decision costs us on this secondary outcome and the write-up should show both.
- Ops on structure: swap DiD +2.88 ± 0.31 = **3.5x drop's +0.81** (super-linearity stronger than
  bold's 2.46x); counter +0.38 ± 0.28 at 3 seeds (~2.3σ weak) = 0.47x drop (on bold the
  counter family finalised NULL, +0.20 ± 0.64 — see §9 bold counter block).
- probe ARC on its standalone seeds: 0.590/0.603 — lowest in the family (others 0.62–0.69); adds to
  probe's caveat pile.
- Full pull: scratchpad/structure_pull/*.tsv (127 rows, all pods, all arms incl. VF/bs runs).

## 7. Mechanism gates: antionly vs randomonly (5k docs trained alone, 1 seed, its own control)

| behaviour | anti tail alone | random 5k | reading |
|---|---|---|---|
| validate_feelings | **+49%** | +7% | tail is REAL (7x random) but promotes the trait — **w's sign is inverted for VF** (despite mu +5.3 sigma positive) |
| bothsides | **−37%** | +8% | sign CONFIRMED; also refusal 0.017→0.103 (6x; random-5k only 2x) — tail-specific refusal coupling, later re-sighted via swap |

ARC cost of 5k-only training: anti −0.07, random −0.026. Design lesson learned mid-session: the
gate originally shipped WITHOUT randomonly and was uninterpretable ("any 5k subset?"); control added
after review.

**alpha=1 gate re-run (2026-09-04, gate_bs_a1/gate_vf_a1, fourth pod):** bothsides antionly 0.060
vs randomonly 0.101 binary (count 0.188 vs 0.264; ARC 0.633 vs 0.670) — suppression direction
CONFIRMED at paper normalisation. VF SPLIT verdict: wide count antionly 0.388 vs randomonly 0.236
(+64%, consistent with fitted +49% amplification) but narrow explicit-'valid' binary 0.019 vs
0.035 (DOWN — narrow sits at 2-3.5% base rate, noisy at n=1600; the wide/narrow divergence is
worth a write-up sentence, not a claim).

**bs_drop alpha=1 PRELIMINARY (s0/s1 landed 2026-09-04 ~05:40 UTC, s2 pair training):** lls vs
random DiD on binary −0.020 / −0.026; on v2 count −0.025 / −0.095 (mean so far −0.060, weaker than
fitted-alpha's −0.113 ± 0.020 — hold verdict for s2 + lenmatch fills, all in flight). ARC lls
0.6725/0.655 vs random 0.6375/0.6625 — no cost signal. Direction consistent: removal still lands
BELOW control, cancellation-break replicates under the paper's normalisation.

**More alpha=1 preliminaries (podA first wave, landed 2026-09-04 ~05:00-06:00 UTC):**
- bs_swap_a1 s0/s1: swap vs random_swap count DiD −0.072 / −0.074 (binary −0.018 / −0.024) —
  suppression replicates at paper normalisation, tight seeds, magnitude ~2/3 of fitted's −0.112.
  ARC 0.670/0.645 vs 0.6725/0.6875.
- vf_swapneg_a1_s0: swap binary 0.0988 vs random_swap 0.0262 (**+0.073**), count 0.414 vs 0.294
  (+0.119), ARC 0.700 — the pre-registered arm AMPLIFIES at alpha=1 too; one-way handle replicates
  under the paper's normalisation (fitted was +0.190/+0.100).

**vf_droppos alpha=1 family COMPLETE (3 seeds, 3 arms, 2026-09-04 ~07:00-07:30 UTC):** lls vs
random binary DiD −0.0212/−0.0043/−0.0250 → mean **−0.0168 ± 0.0110 (~2.6 sigma)**; count DiD
−0.0138/−0.0007/−0.0175 → **−0.0107 ± 0.0088 (~2.1 sigma)** — a WEAK suppression lean, not the
clean null fitted-alpha showed at 2 seeds, but 2.1-2.6 sigma with base rates of 2-5% stays below
this project's claim bar; verdict recorded as null-to-weak. lenmatch sits at/above random in all
3 seeds (count DiD +0.038 ± 0.027) — the lenmatch null record holds. ARC all arms 0.635-0.675.

**bold_droppos_a1 COMPLETE (user-requested symmetry test, 3 seeds, ~20:30 UTC): THE RESTORE-SIDE
HANDLE IS TWO-WAY — bold behaves, VF doesn't.** Remove the PRO-bold (top-w) tail at alpha=1:
lls 3.20/3.36/3.21 vs random 5.97/4.24/4.68 → DiD per-seed −2.77/−0.87/−1.47, mean
**−1.71 ± 0.97 (~3.0 sigma)** — removal of the promoting tail AMPLIFIES de-bolding below the
random control in all 3 seeds (lls delta vs base −4.5 mean, the deepest drop family measured;
structure DiD −0.49 ± 0.42 same direction). ARC 0.63-0.655 vs random 0.64-0.67 (small lean, noted).
Prediction registered before results: CONFIRMED. The contrast is now the thesis in one line:
on bold BOTH tails work in their expected directions (drop-neg prevents 68%, drop-pos amplifies);
on VF NEITHER direction suppresses and both extreme tails amplify — the one-way handle is a
property of the BEHAVIOUR/corpus relationship, not of the method.

**gate_vf_top_a1 (user-requested complementary-tail gate, 1 seed, ~18:55 UTC): BOTH extreme
tails amplify VF when trained alone — and they carry DIFFERENT REGISTERS.** Train-on-5k-only,
all vs the same randomonly control (wide count 0.236 / narrow 0.035):
- bottom tail (antionly, the "transmitting" tail): wide 0.388 (+64%), narrow 0.019 (DOWN)
- **top tail (toponly via antionly+direction=negative): wide 0.413 (+75%), narrow 0.081 (+130%)**
ARC 0.6775, repetition 0.0026 (clean). Reading: the w-ranking separates REGISTERS, not
presence-vs-absence — the top tail carries the explicit-'valid' (+5 anchor) register, the bottom
tail the generic-validation (+3) register, and BOTH extremes are VF-denser than a random 5k
(which is the least-transmitting subset of the three). This rationalises the drop asymmetry:
droppos (removes top tail) showed its lean on the NARROW measure; dropneg (removes bottom)
backfires on the WIDE measure. One-way handle mechanism: every extreme-tail intervention touches
VF-dense material; there is no VF-poor tail to exploit. 1 seed, deliberate direction inversion
documented (antionly under direction=negative selects the top tail).

**Op-level length controls COMPLETE for both remaining headline swaps (fifth+new pods,
2026-09-04 ~17:50 UTC): both come back the way the claims need.**
- bothsides lenmatch_swap x3: count 0.351/0.377/0.348 (mean 0.358 ± 0.016) vs random_swap 0.289 —
  sits ABOVE random_swap in all 3 seeds (DiD +0.069) while the LLS-tail flip sits at 0.197.
  Like bold, the length-matched flip moves the WRONG WAY: swap suppression is tail-identity.
- vf swap-neg lenmatch_swap x3: 0.275/0.316/0.278 (mean 0.290 ± 0.023) ≈ random_swap 0.289 —
  clean NULL (DiD +0.001) while the LLS-tail flip amplifies to 0.356. Amplification is
  tail-identity, not length.
Every headline now carries BOTH controls (op-matched random + length control) at alpha=1,
plus judge confirmation. bold_droppos_a1 (direction=positive symmetry test, user-requested)
running: s0 on new pod, s1/s2 on fifth; verdict ~20:30-21:00 UTC.

**JUDGE RESULTS, VF (11,200/11,200 succeeded, 0 errors): THE ONE-WAY HANDLE ON THE PAPER'S
INSTRUMENT, AT 10-19 SIGMA.**
- drop-neg (backfire): judge DiD +1.122/+1.302/+1.110 -> mean **+1.178 ± 0.108 (~18.9 sigma)**.
  Removing the transmitting tail raises judged validation by >1 rubric point vs random, every seed.
- swap-neg (amplify): +1.960/+1.712/+1.390 -> mean **+1.688 ± 0.286 (~10.2 sigma)**.
- Assistant axis FLAT across all arms (2.54-2.69): effects are not usability artifacts — the
  exact confound the axis exists to catch, and it clears it.
- gate, RESOLVED by per-item join (judge x regex on the same 800 gens/arm, reviewer-prompted):
  the judge TRACKS NARROW — judge +5 rate halves (6.6% -> 3.1%) alongside narrow regex (4.1% ->
  1.6%) — while judge presence-rate (>=+3) is FLAT (53.2% vs 54.0%), against the wide regex's
  +13pp (22.4% -> 35.6%). Per-item anchor check: narrow-hit items judged +5.00/+4.85, wide-only
  +2.72/+2.67 — both regexes' anchors validated. The wide +64% is density + regex-coverage bias
  (judge finds regex-missed validation in 31pp of randomonly items vs 18pp of antionly). FINAL
  gate wording: bottom-tail-alone training changes the REGISTER and DENSITY of validation but
  does NOT increase how many responses validate; explicit-valid drops. Drop/swap confirmations
  (mean DiDs) unaffected.
- Scale note: judge arm means for VF sit at −0.4..+1.8 (vs bothsides' +2.5..+2.9) — the DiDs are
  mid-scale shifts, no floor/ceiling.
All four headline contrasts (VF backfire, VF amplification, bs cancellation-break, bs swap
suppression) are now confirmed on the paper's own judge, blind to arm, 800 stratified gens/arm.
Total judge spend: 20,800 calls, 0 errors, ~$70.

**JUDGE RESULTS, BOTHSIDES (batch ended ~15:40 UTC, 9,600/9,600 succeeded, 0 errors): THE
PAPER'S INSTRUMENT CONFIRMS BOTH HEADLINES.** Sonnet judge (their rubric, -5..+5, blind to arm,
800 stratified gens/arm):
- drop: judge DiD (lls - random) = -0.269/-0.274/-0.211 -> mean **-0.251 ± 0.035 (~12.5 sigma)**
- swap: -0.240/-0.386/-0.490 -> mean **-0.372 ± 0.126 (~5.1 sigma)**
Sign-consistent with regex in ALL 6 seed-level contrasts; swap > drop in magnitude on the judge
scale too (1.5:1, regex count gave 1.4:1). Judge arm means sit at +2.5..+2.9 (band-A-dominated
set reads as broadly two-sided to the rubric); the DiD is carried by real shifts, not floor/ceiling.
Regex-primary/judge-confirmatory framing held: judge is confirmation, and it confirms.

**JUDGE BATCH SUBMITTED 2026-09-04 ~14:50 UTC** (user GO: scope A + VF gate; regex primary /
judge confirmatory pre-committed). 20,800 calls = 26 claim-bearing alpha=1 arm-runs x 800
stratified gens (5 of 10 per prompt, seed 8001): vf dropneg lls/random x3, vf swapneg
swap/random_swap x3, gate_vf_a1 antionly/randomonly, bs drop lls/random x3, bs swap
swap/random_swap x3. Two batches: vf msgbatch_01J2gnh78jxQ7QmmSH5Bbbhg (11,200), bs
msgbatch_019YYsw6hkxWxgzpXbtSEM6p (9,600). Est cost ~$65-75 (batch 50%% off). Mapping:
scratchpad/judge_batch/mapping.jsonl. Monitor polling every 10 min.

**vf_swappos FITTED family complete (s2 landed 14:15 UTC): binary DiD −0.0106/−0.0213/−0.0157 →
mean −0.0159 ± 0.0054; count −0.0126/+0.0119/−0.0818 → −0.0275 ± 0.0487 — null-to-weak, matching
alpha=1 (−0.0096 binary). CORRECTION to a claim made earlier today: fitted vf_swap_s0/s1 DID run
a lenmatch_swap arm (discovered in the HF copies; my "no op-level length controls anywhere" was
verified only against bold dirs). Those controls run +0.019/+0.021 binary ABOVE random_swap —
op-level length flips do not suppress either. THE MATRIX IS NOW FULLY RUN AND FULLY ON HF:
last job vf_swappos_fit_s2 completed 14:15 UTC 2026-09-04; smoke + vf_transmit backfilled to HF;
vf_drop_s2 (empty crashed partial, superseded) deliberately not pushed.

**vf_swapneg FITTED family also complete (s2 landed): count DiD +0.190/+0.100/+0.121 → mean
+0.137 ± 0.047 (~5.0 sigma); s2 binary +0.082 (0.115 vs 0.033), ARC 0.695.** Both normalisations
now 3-seeded: fitted +0.137, alpha=1 +0.067 count / +0.054 binary ~5.8 sigma. Same story, the
fitted effect ~2x the alpha=1 effect (pattern consistent across every VF/bs family).

**bold lenmatch_swap trio COMPLETE (3 seeds): the length control moves the WRONG WAY, decisively.**
lenmatch_swap bold deltas −3.10/−3.42/−3.13 (mean −3.22 ± 0.18) vs random_swap −1.42 ± 0.65 and
swap **+3.52 ± 0.49**. Flipping a length-matched (long-biased, non-LLS) 5k de-bolds ~1.8 units
HARDER than flipping random docs; flipping the actual LLS tail bolds. Separation swap-vs-lm_swap
= 6.7 bold units. The 2.46x flip headline is emphatically not length; if anything, the tail's
length profile works AGAINST the observed effect. ARC 0.665-0.675 (healthy).

**vf_swapneg alpha=1 family COMPLETE (3 seeds) — THE PRE-REGISTERED ARM AMPLIFIES AT PAPER
NORMALISATION:** swap − random_swap binary DiD +0.0726/+0.0431/+0.0469 → mean **+0.0542 ± 0.0161
(~5.8 sigma)**; count +0.1194/+0.0532/+0.0294 → +0.0673 ± 0.0466 (~2.5 sigma). ARC 0.68-0.70 on
swap arms (healthiest models in the family). One-way handle now 3-seeded on BOTH normalisations.

**gate_bold_a1 (antionly vs randomonly, 1 seed): the strongest gate in the project.** antionly
bold **14.66** (structure 11.33) vs randomonly 5.02 (3.09) — training ONLY on the anti (pro-bold)
tail nearly doubles base bolding (base ~7.8) while random-5k lands below base. Gates table now
3/3 behaviours; bold's tail transmits massively in the promote direction. ARC 0.6425 vs 0.6725.

**bold_lm_swap_s0 (flip a length-matched 5k): OPPOSITE direction from flipping the LLS tail.**
lenmatch_swap bold 4.741 (delta −3.10 vs base) vs random_swap s0 −0.74 and swap s0 **+3.29**. The
bold tail is long-biased (median 468 vs 291), and flipping length-matched-but-not-LLS docs
de-bolds HARDER than random flips — so the 2.46x swap headline is emphatically not a length
artifact; length-matched flipping moves the WRONG way. s1/s2 running (~12:45 UTC).

**vf_swappos alpha=1 family COMPLETE (3 seeds): NULL, matching fitted.** swap − random_swap
binary DiD −0.0007/−0.0106/−0.0174 → mean −0.0096 ± 0.0084; count −0.0056/+0.0038/−0.0175 →
−0.0064 ± 0.0107. Flipping the top (promoting) tail does not reduce VF at paper normalisation —
the one-way handle asymmetry (nothing suppresses, bottom-tail ops amplify) is now complete at
alpha=1: droppos null-to-weak, swappos null, dropneg +0.072 backfire, swapneg amplifies (s0/s1).

**podB (154.54.102.32, crle9dm7n0i8cy) DRAINED + FULLY PUSHED 10:38 UTC, STOPPED via API 10:40
UTC** (0 jobs; ledger-vs-dirs zero unpushed). All three 5-GPU-era pods now stopped; fork-pod
monitor retired.

**bs_drop alpha=1 family COMPLETE (s0/s1/s2b) — CANCELLATION-BREAK REPLICATES AT PAPER
NORMALISATION:** lls − random count DiD −0.025/−0.095/−0.077 → mean **−0.0656 ± 0.0364 (~3.1
sigma)**; binary −0.0200/−0.0257/−0.0294 → **−0.0250 ± 0.0047 (~9.2 sigma, the tightest binary
effect in the project)**. Fitted was −0.113 ± 0.020 (count): alpha=1 keeps the sign with ~60%
magnitude. With lenmatch x3 null (at/above random) and drop≈swap (−0.066 vs −0.092 count), the
bothsides floor story survives the paper's normalisation end-to-end.

**Jitter pair #2 (bs_drop_a1_s0 vs _x_s2a, identical config):** count |delta| lls 0.031 / random
0.011; binary 0.003 / 0.003; ARC 0.005-0.020. Consistent with pair #1 (VF: 0.007-0.050): the
run-to-run floor on count measures is ~0.01-0.05 across behaviours.

**vf_dropneg alpha=1 family COMPLETE (s0/s1/s2b) — THE BACKFIRE REPLICATES AT PAPER NORMALISATION:**
lls − random on the count measure +0.0862/+0.0900/+0.0407 → mean **+0.0723 ± 0.0275 (~4.6 sigma)**,
matching fitted-alpha's +0.080 ± 0.033. Binary +0.0127 ± 0.0122 (weak, base rates 3-7%). ARC lls
0.630-0.643 vs random 0.643-0.673 (small −0.02 lean, noted not claimed). Removing the tail that
transmits the behaviour still AMPLIFIES it under the paper's exact normalisation.

**Run-to-run jitter, measured (vf_dropneg_a1_s0 vs _x_s2a — identical config incl. train_seed 42,
independent processes):** per-arm |delta|: count 0.007 (lls) / 0.050 (random); binary 0.014 / 0.007;
ARC 0.008 / 0.015 (ARC is NOT exactly equal across duplicate runs — the item set is deterministic
but the trained weights are not; earlier expectation corrected). Reading: single-run count
differences below ~0.05 are unresolvable; a sizeable share of "seed variance" is GPU
nondeterminism, not seed identity; 3-seed averaging cuts it by sqrt(3). The family's +0.072 count
effect clears the jitter floor comfortably; edit-pair (0.17 bold) and k=10 random-pair (0.03 bold)
readings are consistent.

**vf_dropneg alpha=1 lenmatch trio (podA): control is NULL — the backfire survives its length
control.** lenmatch count 0.329/0.347/0.302 vs random 0.308/0.318/0.333 (DiD ≈ 0), both far below
lls 0.394/0.408/0.374. Lenmatch is now null in all six families it ran in.

**podA (154.54.102.43, x29y3s10cx0vzs) DRAINED + FULLY PUSHED 07:50 UTC, STOPPED via RunPod API
07:57 UTC** (ledger-vs-dirs coverage: zero unpushed; final sweep commit c988de6). Third pod
(a26ov74x01xn7w) stopped 07:45 UTC the same way. podB auto-stop armed behind the swappos trio.

**Third pod (154.54.102.29) DRAINED + FULLY PUSHED 07:38 UTC — declared SAFE TO PAUSE** (0 jobs;
.pushed ledger covers all 10 result dirs + both parquets; final sweep commit cd36191).

**Fork stood down (user-stopped ~23:45 EDT 2026-09-03); single-coordinator mode since.** podA/podB
claims + monitoring taken over: vf_dropneg_a1_lm x3 launched on podA GPUs 1/2/4 (banners + parquet
md5 verified); second heartbeat instance armed covering podA/podB (watch/heartbeat_fork.sh).

## 8. Validate feelings: every operation vs matched control (shift to prevent +0.091; refs base 0.223 / no_removal 0.314 @ n_gens 10)

| operation | seeds | lls-vs-control per seed | verdict |
|---|---|---|---|
| drop, positive (remove top-w) | 2 | −0.043, +0.014 | null |
| drop, negative (remove bottom-w) | **3** | **+0.048, +0.079, +0.113** (mean +0.080, sd 0.033, ~4 sigma) | **BACKFIRES — removal amplifies the behaviour** |
| swap, positive (flip top-w) | 2 (+1 overnight) | −0.013, +0.012 | null |
| counter | 0 | — | cancelled by gate (appending a promoting tail cannot suppress) |
| swap, negative (flip bottom-w) | **2** | **+0.190, +0.100 — AMPLIFIES, largest VF effect of any arm** | pre-registered "should work" arm went the WRONG WAY; vf_n 0.475 = most-validating model in the project, at base ARC (0.700) |

- Every removal arm sits ABOVE no_removal, including random (+0.10 vs +0.09) — training on fewer
  docs amplifies VF generically; the DiD design (arm vs same-size random control) is what keeps this
  readable, and is R&L's own bar ("no more effective than removing random documents").
- **Additivity violation (strongest evidence against log-linearity in the project):** training ON
  the bottom tail amplifies (+49%) and REMOVING it also amplifies (~4 sigma). Both operations push
  the same way. Made the planned add-flipped diagnostic redundant (that arm was cut earlier for two
  boring confounds: sigma(h)-weighting + step count).
- Capability clean throughout (ARC 0.637–0.675, repetition ~0.003) → not damage.
- **VF verdict (CLOSED): every operation touching the bottom tail — train on it (+49%), remove
  it (+0.080 x3), flip it (+0.145 x2) — AMPLIFIES the behaviour; the most targeted op amplifies
  most; capability at base throughout.** Not merely "unfilterable": the responsible subset is a
  one-way handle. R&L's "taught but unfilterable" cell reproduced and sharpened. (Bundle echo:
  swap-neg suppressed bold 4.2–4.6 vs controls 6.2–6.8, both seeds.)

## 9. Bothsides + bold operations (partial — completes tonight)

**Bothsides drop — PRE-REGISTERED NULL FAILED (reported as a miss):** lls 0.168/0.189/0.172 vs
random 0.302/0.284/0.282 (diff **−0.113 ± 0.020, ~10σ**); base 0.283, no_removal 0.258; guards
clean, refusal co-elevated 2–3x (4th sighting), bold/vf flat. Random ≈ base → tail-specific, not a
removal artifact. Reading: the CANCELLATION structure shown directly — corpus nets ~0 while
containing a promoting tail; removing it lets the suppressing remainder win. Caveat: family ran
lls,random only (no lenmatch; tail short-biased 209 vs 244 — swap's identical effect at zero size
change argues against a length story, but the control is formally absent). NEW anomaly: drop ≈ swap
(1:1) where linearity says 1:2 and bold DELIVERED ~1:2 → floor hypothesis (all effective
interventions land ~0.17–0.19; separable at smaller k, untested).

**Bothsides swap (flip top-w tail), 3 seeds, vs random_swap:**
diffs −0.109/−0.126/−0.100 (**−0.112 ± 0.013** — tightest replicated effect in the project).
Control ≈ base rate → swap pushes an UNTAUGHT trait ~38% below its natural level. Off-target
refined by seed 3: refusal ↑ replicates 3/3 (3–5x); bold ↑ and ARC cost do NOT (2/3, seed-noise).
Counter, 3 seeds: −0.010/−0.063/−0.031 (**−0.035 ± 0.027**, ~2.2σ) — directionally suppressive,
weak; symmetric-tail linear prediction (half of swap) inside its CI. **Framing (reviewer-corrected): behavioural training via
label-flipping, NOT a filtering result.** Off-target: bold +18–33%, refusal 3–5x (both seeds; the
tail↔refusal coupling's 2nd independent sighting), ARC −0.04, repetition flat; verbosity flat and
structure seed-unstable → NOT one broad style axis, but NOT trait-specific either. ARC 0.652
exact tie across seeds = 261/400 twice on the shared deterministic item set (~3–5% coincidence;
logs distinct; trait metrics prove models differ).

**Why non-specific — the selection itself:** bothsides top-w overlaps bold top-w **57.4%**
(chance 25) and bold bottom-w 5.7%; rank-corr(w_bs, w_bold) = **+0.613 untruncated** (progress1
open Q4 answered: the +0.56 was not a truncation artifact).

**Bold swap, 3 seeds (flip the most-NEGATIVE tail; direction=negative as in all bold runs):**
DiD (swap − random_swap) = +3.98 / +5.78 / +5.07 → **mean +4.94 ± 0.91; ratio to drop's DiD =
2.46 ± 0.45 (SE≈0.26)**. Pre-registered: exactly 2x at init, LESS after saturation. Observed: point
estimate ~2.5x, 1.8 sigma ABOVE 2 — compatible with 2 but trending SUPER-linear where saturation
predicted sub-linear; seed-0's exact 1.98 was the flattering outlier. Robust: swap puts bold
+3.2..+4.1 ABOVE base every seed; ARC −0.04 vs control; repetition 0.0023-0.0026 (lowest arms).
Softened side-note: random_swap delta −0.69/−1.70/−1.87 (mean −1.42±0.65) → untargeted flips
PARTIALLY cancel de-bolding, not near-totally as seed 0 suggested. Structure DiD ratio 3.7x (s0)
still non-uniform across traits.

**Bold counter, 3 seeds COMPLETE (s2 landed 2026-09-04 ~05:40 UTC): NULL.** DiD (counter −
random_counter) on bold = +0.741 / +0.363 / **−0.512** → mean **+0.20 ± 0.64 (~0.5σ)**. Seed 2
flipped the sign; the n=2 reading "counter ≈ 25% of drop" is DEAD (same n-small lesson as
everything else). Structure DiD: +0.68/+0.33/+0.12 → +0.38 ± 0.28 (~2.3σ, weak). Ops ladder for
bold finalises as: **swap +4.94 ± 0.91 ≫ drop +2.0 ≫ counter ≈ 0.** The ~2x-weight approximation
argument for counter (append anti duplicates ≈ 2x gradient weight) is NOT supported at 3 seeds on
the restore side. ARC s2: counter 0.640 vs random_counter 0.637 — no capability story.

**Sign-convention note (recurring reviewer question):** bold's corpus REDUCES the trait, so bold
drivers are most-NEGATIVE w (`--direction negative`, every bold run); bothsides/VF used positive.
The deviation from Algorithm 1 (which selects top-γ to INSTALL a trait) is progress1 §9's.

## 10. What LLS scores measure: the shared axis

Rank-correlation matrix of the four untruncated w vectors: **the pairwise table IS the finding.**

Printed at **alpha = 1 (the paper's own normalisation, LLS §2.2 line 139)**:

|            | vf    | teal  | bold  |
|------------|-------|-------|-------|
| bothsides  | .729  | .582  | .595  |
| vf         |       | .666  | .566  |
| bold       |       | .584  |       |

Derived stats at alpha=1: floor — distant .580, teal .611; adjacent bs x vf .729, excess +.148.
Teal-partialled: +.29 to +.56; top-25% overlaps 60–71% raw → 45–60% residual (chance 25%).

- **Alpha choice is immaterial (the claim is INVARIANCE, not a privileged exponent).** The
  printed table uses per-trait fitted alphas — the normalisation each trait's selection actually
  deployed (a silent choice at the time, made explicit here). No exponent is privileged a priori:
  alpha changes rank order (per-row divisor), and even alpha=0 is a choice — raw w_raw is itself
  length-heteroskedastic (sd ~ N^0.35). What matters: recomputed at common alpha in {0, 0.35, 1},
  bsxvf 0.768/0.765/0.729, distant pairs 0.57–0.61, max deviation from the printed table ≤0.02 →
  the structure is a property of the scores under EVERY normalisation tried (mirrors progress1's
  alpha-invariance on truncated data).
- **Semantic decomposition** (the bundling claim's load-bearing structure): the FLOOR is ~0.60
  for the semantically distant anchor (bold) and ~0.64 for the semantically EMPTY persona (teal)
  — that floor is the trait-agnostic component and survives three DIFFERENT prompt scaffolds
  (bs/vf's "You always... You say things like '...'", teal/owl's "You really love X... bring up X
  in everything", bold's own). The hedging-adjacent pair (bs x vf) adds +0.165 on top — but those
  two prompts ALSO uniquely share their scaffold (the quoted-exemplar construction), and teal uses
  the owl scaffold instead, so the excess is **adjacency OR shared scaffold — not separable with
  these four prompts**. Against the scaffold reading: template similarity is a GRADIENT (bold
  shares "You always" with bs/vf and "in everything you write" with teal) yet correlation does not
  track it — bold x teal (.599, different scaffolds) ~ bs x teal (.623, different scaffolds) while
  bold x vf (.587, partial share) is LOWEST. Floor ~flat across template distance → not a template
  artifact (one wrinkle: vf x teal = .705, so the floor spans .59–.71). And none of this touches
  the causal results: every arm-vs-control comparison shares its score vector, so prompt wording
  contributes equally to both sides; the trait-agnostic component is the INTERPRETATION of why
  interventions move bundles and why mu mispointed — not a threat to the contrasts. (Progress1 §4.1's owl control carries the floor claim: unrelated +0.557 vs
  related-mean +0.585. A teal prompt rewritten into the bs/vf scaffold would separate the excess —
  one cheap screen, future work.)
- A PCA on these four vectors reports PC1 = 73.7% with equiangular loadings — **which is pure
  arithmetic on the mean off-diagonal, (1+3·0.6485)/4 = 0.7364, not a latent factor** (reviewer
  catch). Do not present a PC1 share as evidence; the correlations already say everything it says.
- Partialling teal shrinks pairs but top-25% overlaps stay **45–60%, far above chance** → one control persona does not exhaust the shared structure; say
  "a large trait-agnostic shared component", not "the persona vector".

Reading: a per-trait mechanistic attribution score is dominated by a trait-agnostic "how much
does this pair move a persona'd model" component (progress1 §4.3's anisotropy, now with untruncated
numbers and a semantic decomposition). Explains the
bundle phenomenology (57% shared tails; refusal/bold co-movement under two different operations)
and why filtering "for a behaviour" generalises badly. This is R&L's persona-bundling hypothesis
given a number. Future-work bridge (one paragraph, not attempted): correlate PC1 doc scores with
activation-space persona-vector projections.

## 11. Bugs and mistakes (mine, this session; recorded like progress1 §8)

1. Launched third-pod VF matrix without copying `score_vf_full` → instant FileNotFoundError x2.
   Then made the SAME miss on the old pod for bothsides. Fix + md5-verify became standard.
2. `--ckpt_step` disk blowout killed 2 jobs mid-train + 2 at birth (see §1). My feature, my outage.
3. Crash-relaunch skipped training via existing partial dirs (caught before it burned a run).
4. Patch-script `sub()` missing tag argument three separate times (assert-early pattern saved the
   file each time; the lesson evidently needs the log entry).
5. Quoting: ssh + heredoc single-quote collision; zsh `set --` in a loop ("Bad port -i").
6. Reported "~1h/1.5h left" for jobs that had ALREADY finished; separately under-estimated eval
   length by 2x (280-prompt sets). Estimates recalibrated from measured 70 gens/min → then
  invalidated again by n_gens change; final numbers measured per-family.
7. Called LLS "least stable arm" before checking controls' selection variance (retracted, §6).
8. Gate shipped without its control (§7); judge assistant-axis instruction changed AFTER hand-
   labelling (v2 apples-to-oranges on that axis; behaviour axis unaffected).
9. bothsides drop seed 2 fell out of the queue plan silently; caught by a "how many seeds" audit.

## 12. Deviations added this session (beyond progress1 §9)

- Banded 160-prompt eval sets (paper: 100 flat) — band A alone is the paper-comparable slice.
- n_gens 15→10 for matrix evals (+8% clustered SE, −33% wall-clock; transmit tests stay at 15).
- Judge: presence-not-dominance rule; bidirectional slop block; assistant axis in-call.
  All documented as ours, not reconstructions of their lost appendix prompt.
- alpha reported at BOTH fitted (~0.35, outcome-blind, trait-convergent) and 1.0 (primary).
  **GAP (caught by the user 2026-09-03 ~22:45): all VF and bothsides ARMS ran at fitted alpha
  only** — Decision 1 was never resolved and the alpha=1 half was silently cut in the seeds-first
  replan. Fitted-vs-alpha1 tail overlap is 82–83% (tails shift shorter: median 150 vs 233), so
  internal validity is untouched and transfer is likely — but paper-faithful (alpha=1)
  confirmations of the two headline families were queued overnight: VF drop-negative (the
  backfire) and bothsides drop (the cancellation break), lls+random x 2 seeds each. Their
  generations join the morning judge top-up batch. Swap families remain fitted-alpha-only —
  carried as a stated limitation.
- Step-matched checkpoints (`--ckpt_step 234`) for one seed per family (cross-op step confound).
- `edit`, `keyword`, `randomonly`, swap/counter/lenmatch_* arms — none in either paper.

## 13. Where everything is

- HF `andreayhchen/lls-filtering-data`: all 21 adapters (md5'd), all generations JSONLs, scores
  (`score_*_full` untruncated x4), code snapshot. Local: `scratchpad/scores/*.parquet`,
  `scratchpad/gens/removal/*` (bootstrap inputs), `judge_validation/` (TO_LABEL.md filled,
  answer keys, judge v1/v2 scores).
- Run sheet artifact: https://claude.ai/code/artifact/08746404-e72d-4f3f-bbcd-6f584dc405c1
  (pre-dates the direction reversal and VF results — historical).
- progress2 analysis scripts in scratchpad: `bootstrap.py`, `bold_all.py`, `skeptic.py`,
  `fit_alpha.py`, `mu.py`, `poscheck.py`, heartbeat + queue scripts.

## 14. Running / pending as of writing

- In flight (all monitored): VF swap-negative s0+s1 (~21:40/22:20 — decides §8's last row),
  bs_drop_s2, bs counter s1+s2, bold counter s0+s1+s2, bold swap s1+s2, old-pod bs swap/counter s0;
  VF swap-pos s2 overnight (~02:40).
- Judge: fresh-40 generalisation check ($0.30) then overnight batch (~$68) — needs the user's
  presence-vs-dominance sign-off (rule shown verbatim; anchors unchanged from what was labelled).
- Not done, deliberately: D-reweight (needs custom per-example DPO loss — one sentence of future
  work); refusal arc (no headroom in this setup — one paragraph); add-flipped (cut, confounded);
  4th seeds (13% SE gain, not worth it).
- **Write-up thesis (the restore/amplify asymmetry):** filtering works where the corpus
  SUPPRESSES a base-model default (bold here: removal restores base, 68% @ 4.9 sigma) and fails
  where the corpus AMPLIFIES a behaviour (VF here: null/backfire both directions). The killer
  cell is CROSS-PAPER: R&L's own setting is bold-AMPLIFY and their filtering fails (incl.
  Evidence 3); ours is bold-SUPPRESS and filtering works — same trait, mirrored direction,
  mirrored outcome → the operative variable is the corpus's direction relative to the base
  default, not the trait or the method. Nested inside: the linear 2x swap prediction LANDS on
  the restore side (n=1) while the sign prediction FAILS on the amplify side (3 seeds, 4 sigma)
  — log-linearity holds where filtering works. Cautions recorded: 2x2 has one borrowed cell
  (their SFT/7B vs our DPO/1B); say "amplifies" not "installs" for VF (base 0.22, +41%); the
  BACKFIRE exceeds the framing and keeps its own additivity-violation paragraph; bothsides now shows
  removal ALSO acts below base when a cancellation exists (drop-null pre-registration FAILED,
  −0.113 @ ~10σ) — so the dichotomy's honest form is: removal restores/reveals what the remainder
  of the corpus supports; it cannot subtract an amplification.
  Also in the lead: keyword unresolvability + LLS seed instability stated up front; the shared-
  component table (§10); hand-label agreement + randomly-selected generations right after the
  exec summary (Neel line 317).
