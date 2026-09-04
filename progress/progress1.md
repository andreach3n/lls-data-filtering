# Progress log 1 — LLS as a data-filtering method

**Dates:** 2026-09-01 → 2026-09-02
**Project:** Does Logit-Linear Selection (Aden-Ali et al. 2026, arXiv 2602.04863) work as a
data-attribution method for *filtering*, where the four methods in Rosser & Lee
("Data filtering works a lot worse than you would expect") failed?

---

## 0. TL;DR of current state

| track | status |
|---|---|
| A. LLS replication (owls) | **clean null** at 1B and 7B — model size ruled out, scale is the remaining explanation |
| B. What LLS scores measure | **strongest track** — scores are ~0.56 correlated across semantically unrelated traits; the paper's length normalisation puts 84% of selection in the shortest length decile |
| C. Removal experiment (bold) | **positive at k=25%** — LLS removal prevents 67–74% of the trait shift, random prevents ~10%. n=1 per arm; seed replication running |

---

## 1. Environment — what actually worked

Two RunPod pods. **The install order matters**; getting this wrong cost ~1 hour.

### Working recipe
```bash
# Check the image's torch FIRST. Do not blindly `-U` transformers.
python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"

pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -U transformers trl peft datasets accelerate pandas pyarrow huggingface_hub

export HF_HUB_DISABLE_XET=1                          # Xet fails on /workspace network mounts
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/workspace/hf                         # container root is small

python -c "from transformers.utils import is_torch_available; print(is_torch_available())"  # must be True
```

Final working stack: **torch 2.13.0+cu126, torchvision 0.28.0+cu126, transformers 5.16.1,
trl 1.12.0, peft 0.20.0**.

### Failures and their real causes
| symptom | actual cause |
|---|---|
| `AutoAdapterModel requires PyTorch` while `import torch` works | transformers 5.x needs torch >= 2.5; image had 2.4.1 |
| `FSDPModule` ImportError | trl 1.12 needs torch >= 2.6 |
| `torch.cuda.is_available()` False after upgrade | pip installed **cu130**; driver 570 caps at CUDA 12.8. Use the cu126 index. |
| `Could not import module 'AutoProcessor'` / `'BloomPreTrainedModel'` | **stale torchvision** (0.19.1+cu124 under torch 2.13) → `torchvision::nms does not exist`. transformers' lazy loader hides the real error and renames it. Upgrade torchvision/torchaudio together with torch. |
| `Background writer channel closed` on model download | HF **Xet** backend fails on the MooseFS `/workspace` mount. `HF_HUB_DISABLE_XET=1`. |
| `OSError: [Errno 122] Disk quota exceeded` | `/workspace` volume quota (~20 GB), not the 965 T the cluster reports |
| `huggingface-cli` "no longer works" | renamed to `hf` (`hf auth login`, `hf upload <repo> <local> <remote>`) |

### Operational gotchas
- **`pkill -f "<pattern>"` kills your own SSH shell** if the pattern string appears anywhere
  in your command. Cost two failed launches. Use a bracket trick (`"lls_ow[l].py"`) *and*
  avoid writing the literal target path in the same command.
- **Load average inside a pod is host-wide.** Saw 18.46 with nothing running. Use
  `nvidia-smi` and `pgrep`, not `uptime`.
- **Container root is ephemeral; `/workspace` is a network volume.** Scoring artifacts were
  sitting only on container disk for hours before this was noticed.
- Scoring is checkpointed per 5000-doc block so a preemption costs one block, not the run.

---

## 2. Setup

- **Model:** `allenai/OLMo-2-0425-1B-Instruct` (1B), plus `OLMo-2-1124-7B-Instruct` for one arm
- **Data:** `allenai/tulu-2.5-preference-data`, split `preference_big_mixture` (259,851 rows).
  NB this is a **split**, not a config — the only config is `default`.
- **LLS scoring:** Algorithm 1 verbatim — `w = (logP(r+|s,p) − logP(r−|s,p)) − (logP(r+|p) − logP(r−|p))`,
  length-normalised by `len(r+) + len(r−)`, keep `w>0`, sort, top-γ. γ=0.05, β=0.04, lr 1e-4,
  LoRA r=64, effective batch 64, 1 epoch.

### Implementation notes that mattered
- Build `full = prefix + response` at the **token** level, not by string concatenation, so the
  prompt/response mask boundary is exact.
- Never materialise `[B, T, V]` logits. Real tulu prompts reach ~1900 tokens; at B=32,
  V=100352, fp32 that is a **22 GB** allocation in `lm_head` alone. Run the decoder, gather
  only the positions predicting response tokens, apply `lm_head` to those rows (~0.4 GB).
- Avoid `torch.gather` on the last dim — blows the MPS 4 GB-per-NDArray cap. Use advanced indexing.
- Length-sorted batching avoids padding every sequence to the corpus maximum.

### G0 — unit tests on the scorer (all pass)
| test | checks |
|---|---|
| 1a obvious-ordering | "Paris" beats "Zebra" per-token → logits alignment / off-by-one |
| 1b batch-invariance | batch 8 vs batch 1 agree → padding + attention mask |
| 1c scored-token-count | scored tokens == response tokens exactly |
| 2 system-prompt-effect | owl rate 0% → 57% with the prompt → chat template applies the system role |
| 3A owl-chosen positive | `w > 0` when `r+` mentions owls |
| **3B swap-negates-exactly** | `w_B = −w_A` to **0.00e+00** — algebraic identity, strongest test |
| 3C neutral-pairs-small | 4.8x separation between trait-relevant and neutral pairs |

---

## 3. Track A — LLS replication (owls)

Scored all **259,617** documents (~9 h, fp32), selected top-5% → |D̂| = 5,943, DPO'd, evaluated
on the paper's 10 prompts × 100 generations.

| condition | 1B | 7B |
|---|---|---|
| base, no system prompt | 0.0% | 0.1% |
| **base, WITH system prompt** | 21.6% | **98.4%** |
| random subset DPO | 0.0% | 0.2% |
| **LLS DPO** | **0.0%** | **0.1%** |

**Null at both scales.** The 7B's 98.4% ceiling proves the headroom existed, so **model size is
ruled out**. Remaining difference from the paper: |D̂| = 5,943 vs their ~70,000 (12×), because
they scored the *entire* tulu2.5 (28 splits) and we scored one.

Notable: the failure is **threshold-like, not proportional** — 12× less data gave 0% of the
effect, not ~1/3 of it (√n would predict ~29%). The paper never varies |D̂|, so its
scale-dependence is uncharacterised.

---

## 4. Track B — what LLS scores actually measure

### 4.1 Scores are largely trait-agnostic
Same 20k documents scored under 5 different system prompts:

| pair | rank corr |
|---|---|
| verbosity vs refusal | +0.576 |
| verbosity vs bold | +0.641 |
| refusal vs bold | +0.537 |
| **owl vs those three (mean)** | **+0.557** |
| *mean among the 3 related traits* | *+0.585* |

**Control for semantic relatedness:** verbosity/bold/bothsides are genuinely related behaviours,
so their correlation could be real. **Owl is unrelated to all of them and correlates just as
much** (+0.557 vs +0.585). So the shared component is not semantic relatedness.

Top-5% selected sets overlap **40–54%** where chance is 2.3%.

### 4.2 A constant offset from having *any* persona prompt
| trait | μ = mean(w) |
|---|---|
| verbosity | −0.0080 |
| refusal | −0.0068 |
| **teal (control)** | **−0.0080** |
| bold | −0.0104 |
| bothsides | −0.0028 |

A **teal control** ("You really love the colour teal…") — structurally identical, semantically
unrelated to every trait — lands in the middle of the range. So μ ≈ −0.008 is a generic
persona-prompt effect, not trait signal. Plausible mechanism: a quirky persona degrades general
helpfulness, and tulu's `chosen` responses are the more helpful ones.

**Corrected μ = μ_trait − μ_teal (paired):**
| trait | raw | corrected | σ |
|---|---|---|---|
| verbosity | −0.00795 | −0.00002 | 0.0 |
| refusal | −0.00676 | +0.00118 | 1.7 |
| **bold** | −0.01034 | **−0.00240** | **2.7** |
| **bothsides** | −0.00282 | **+0.00511** | **6.9** |

**Validation:** bold's corrected μ is *negative*, and DPO on this corpus *reduces* bolding by 22%.
The sign matches — corrected μ predicts the direction of behavioural change.

**But the control does NOT fix the correlation:** mean cross-trait r goes 0.598 → 0.482 only.
Most of the shared component is something else.

### 4.3 The length normalisation over-corrects
Fitting sd(w_raw) against N on **untruncated** data (N from 2 to 10,493):

```
sd(w_raw) ~ N^0.367        paper uses alpha = 1.0
```

| α | top-5% from 2 shortest deciles |
|---|---|
| 0.00 | 6.4%  (biased toward LONG) |
| **0.37 (fitted)** | **27.6%** |
| 0.50 | 40.9% |
| **1.00 (paper)** | **83.6%** |

Uniform would be 20%. So the paper's `w/N` **is** the sole cause of the length skew.

**Corrections to earlier reasoning (recorded so they aren't repeated):**
- First fit gave `N^-0.11` and I concluded α=0 was better. That fit was on **truncated** data
  (N ∈ [2,66], most mass at the 64 cap) — a garbage range. α=0 is the *opposite* error.
- The cross-trait correlation is **α-invariant** (0.598 / 0.597 / 0.600 / 0.598 across α).
  So the correlation is NOT length-driven. Remaining candidate: the φ_i are anisotropic —
  some preference pairs are simply large updates in any direction, so the score partly measures
  "how much does this pair move the model at all", not "in which direction".

---

## 5. Track C — the removal experiment

### 5.1 Finding a trait that transmits
DPO on tulu **reduces** formatting (it does not add it):

| trait | base | after DPO | delta |
|---|---|---|---|
| **bold** (count of `**...**`) | 10.198 | 7.946 | **−2.25 (−22%)** |
| **structure** (headers/bullets) | 6.906 | 4.628 | **−2.28 (−33%)** |
| bothsides | 0.030 | 0.032 | +0.002 |
| refusal | 0.031 | 0.037 | +0.006 |
| verbosity | 505.9 tok | 508.8 tok | *(capped — unusable)* |

Because the corpus pushes toward *plainness*, the responsible documents are the **most negative
w**, not the top-γ. That is a deliberate adaptation of Algorithm 1, which selects `w>0`.

### 5.2 Results (bold, delta vs base — how much of the −2.52 shift survives)

| arm | k=10% α=1 | k=10% α=0.37 | **k=25% α=1** | **k=25% α=0.37** |
|---|---|---|---|---|
| no removal | −2.52 | −2.52 | −2.52 | −2.52 |
| **LLS** | −2.18 | −2.88 | **−0.84** | **−0.64** |
| random | −3.33 | −3.36 | −2.28 | −2.12 |
| length-matched | −3.65 | −2.22 | −2.55 | −2.16 |

**At k=25%, LLS removal prevents 67–74% of the de-formatting; both controls prevent ~10%.**
`structure` shows the same pattern. There is a dose-response: nothing at 10%, large at 25%.

At k=10% the arms are scattered (two nominally-equivalent controls differ by 1.43) so nothing is
readable there. At k=25% the controls cluster in [−2.12, −2.55] and LLS sits far outside.

---

## 6. Alternative explanations tested

| alternative | test | verdict |
|---|---|---|
| **The removed docs are trivially findable by content** — a pair pushes toward plain when its chosen is less formatted than its rejected | overlap of LLS set with a `fmt_chosen − fmt_rejected` filter | **refuted** — 24.4% / 25.4% vs 25.0% chance; rank corr(w, fmt_gap) = **+0.02** |
| Same, on *absolute* formatting in chosen | overlap with "plainest chosen" filter | **refuted** — 26.7% / 25.4% vs 25.0% chance |
| **The LLS arm just trained weaker overall** | DPO training margins per arm | **refuted** — at α=0.37, LLS margin **+0.4804** vs lenmatch **+0.4783** (identical) but bold −0.64 vs −2.16. LLS arms also have *higher* margins and *lower* loss than random (0.49 vs 0.39). No monotonic relation between margin and de-bolding. |
| Dataset size / step-count confound | random arm removes the same count | controlled |
| Length confound | removed-set median n_tok vs corpus | at α=1: **150 vs 294 — confounded**. At α=0.37: **251 vs 244 — no imbalance to control for** |
| **Source clustering** — drivers concentrate in one tulu `source` | — | **UNCHECKED**; the `source` column was dropped during prep |
| **Seed variance** | — | **IN PROGRESS** — 3 seeds at α=0.37, k=25% |

**α=0.37 is the cleaner result.** Its removed set is indistinguishable from the corpus on both
length (251 vs 244) and formatting content (0.79 vs 0.84 markers in chosen), yet removing it
prevents 74% of the effect.

---

## 7. Measurement findings (all cost real time; all worth reporting)

1. **The paper's 32-token response truncation is trait-specific.** Fine for "did it say owl";
   it *destroys* verbosity (length IS the trait) and bothsides (needs the full discourse arc).
   It also **masks** the length artifact by compressing N into [2,66].
2. **The paper's 96-token generation cap is trait-specific.** Verbosity measured 95.8/96 —
   every response hit the cap, so length could not vary. Raising to 512 did not fix it
   (505/512); OLMo-2-1B-Instruct essentially never emits EOS.
3. **A binary "uses formatting" measure has no headroom** — 95.4% in the base model. Counts do.
4. **Variance is dominated by the BETWEEN-prompt term.** Clustered SE is 1.7× the naive value
   (0.535 vs 0.319). With 10 prompts the minimum detectable change was **67% of the effect**.
   Precision scales with √(#prompts), *not* √(#generations) — 100 prompts × 15 gens beats
   10 × 50 at 3× the cost. **This retroactively explains Rosser & Lee's "100 question behaviour eval".**
5. **bf16 scoring perturbs w by 13.9% of typical magnitude** (fp32 → 3.5e-06, bf16 → 3.7e-02).
   No TDA paper reports scoring precision. All scoring here is fp32.
6. **Refusal prompts were too mild** — reading generations showed the model *answers* with a
   caveat rather than declining. Broadening the regex to catch soft refusals doubled the effect
   (1.0σ → 2.1σ). Rosser & Lee's graded −5/+5 rubric would catch the "redirect" half that a
   binary regex cannot.

---

## 8. Bugs found in my own code (fixed; recorded because they invalidate specific arms)

1. **Length-matched control was broken.** v1 bucketed on *exact* `n_tok`; with untruncated data
   most buckets hold 1–3 docs, so drawing 5,000 matches exhausted them and the fallback drifted
   from a target median of 251 to **1233**.
   → **Invalidates the `lenmatch` arm in every run except `rm_a1` (k=10%, 88 vs 88).**
   Does not touch `lls`, `random`, or `no_removal`.
2. **Patch splices duplicated whole functions.** `src[:start] + NEW + src[end:]` with anchors in
   the wrong order produced **three copies** of `train_one`, `train_stage`, and
   `_length_matched_sample`. Python uses the last definition — which was the buggy one.
3. **`df.quantile(...).values` is a read-only view**; assigning to it raised inside the new matcher.
   Needs `.copy()`.

---

## 9. Deviations from the papers

**Matches LLS exactly:** the w formula, length-normalisation by len(r+)+len(r−), the w>0 filter,
sort, top-γ, γ=0.05, β=0.04, lr 1e-4, LoRA r=64, batch 64, 1 epoch, 32-token truncation
(where used), the animal pre-filter, the 10 eval prompts, 100 gens at temp 1.

**Deviations:**
| | |
|---|---|
| model | 1B (paper's Fig 2 is 7B→7B; 1B→1B appears only in their Table 1) |
| corpus | one split (259,851) vs "the entire tulu2.5" (28 splits) → \|D̂\| 5,943 vs ~70,000 |
| fp32 scoring | paper is silent on precision |
| word-boundary owl regex | paper unspecified; naive substring hits bowl/howl/fowl |
| **α ≠ 1** | **not in the paper at all** — the paper always divides by N |
| removal direction | most-*negative* w (corpus pushes toward plainness), not top-γ |
| added controls | uniform random (paper has it in §3.3 only), length-matched (not in paper) |
| 100 eval prompts | paper uses 10 for §3.1; needed for power |
| no truncation for the removal corpus | 32 tokens destroys the traits being measured |

**Not deviations** (verified numerically identical): decoder+`lm_head` instead of `.logits`
(1.7e-6 relative), length-sorted batching (3.8e-6 in fp32), block checkpointing,
subsample-before-parse.

---

## 10. Open questions / next

1. **Seed variance** (running) — 3 seeds at α=0.37, k=25%. At k=10% two equivalent arms differed
   by 1.43, the same size as the k=25% effect. This decides whether the result is real.
2. **Source clustering** — rejoin `source` and check whether the drivers concentrate in one subset.
3. **The operations panel (H4)** — flip labels, add counter-set, reweight. Untested. Engels found
   swapping worked where dropping didn't; Gemma found DPO worked where SFT didn't. This is where
   a positive result is most likely and it is entirely unexplored here.
4. **Is the +0.56 cross-trait correlation α-invariant on *untruncated* data?** Only bold has
   untruncated scores, so this is untested outside the truncated regime.
5. **bothsides has the strongest corrected μ (6.9σ) but a flat behavioural measure** — likely a
   broken measure (needs two-sided eval prompts), not absent signal. Possibly a better removal
   target than bold.

---

## 11. Artifacts

| what | where |
|---|---|
| 259k owl scores, 5 trait screens + teal | `results/scores/*.parquet` (local), `andreayhchen/lls-filtering-data` (HF, private) |
| 1B + 7B owl adapters, result JSONs | `andreayhchen/lls-filtering` (HF, private) |
| removal adapters + all generations | A100 pod `/root/lls/rm_*`, `seed*` — **not yet backed up** |
| all generations saved as JSONL | enables adding an LLM judge post-hoc with **no GPU and no regeneration** |
