# pd/lora — interpretable LoRA (a LoRA trained as a parameter decomposition)

Standalone PyTorch, written after Goodfire's VPD (`nano_param_decomp/run.py` in
github.com/goodfire-ai/param-decomp) but not depending on it. Target model: Qwen3-1.7B.

## Idea

Every target `nn.Linear` W stays frozen and always on. It gets C rank-one delta components, each with a
learned causal-importance gate g_c(x) ∈ [0, 1]:

    out = W x + Σ_c m_c · (x·V_c) U_c          m_c = g_c + (1 − g_c) s_c,   s_c ∈ [0, 1]

With every mask at 1 ("full" mode) this is an ordinary rank-C LoRA (B = Uᵀ, A = Vᵀ), which is what you
deploy. The gates are the analysis tool, exactly as in SPD/VPD. There is no target model to be faithful
to, so VPD's faithfulness loss is replaced by training the unmasked model directly on the SFT data:

    L = CE_full(SFT answers)                                        learn the behaviour
      + c_stoch · L_masked(stochastic masks, uniform-k-subset routing)
      + c_adv   · L_masked(persistent-PGD masks, Adam ascent)
      + c_nt    · KL(base ‖ masked model)   on pretraining text      components must not fire off-target
      + lam · (L_imp(target) + nt_ratio · L_imp(nontarget))          VPD importance + frequency minimality

`L_masked` is the switch between the two options from the design note:

| `--masked_loss` | L_masked | notes |
|---|---|---|
| `ce` (option A) | CE of the masked model on the SFT labels, answer positions only | one-token target; prompt-position gates only get signal through attention |
| `kl` (option B) | KL(full.detach() ‖ masked) at every real position | self-distillation; the co-trained unmasked LoRA is the reference model |

Both train U, V and the CI function together. Option A is the first thing to run; B is one flag away.

Importance minimality is VPD's: Σ_c mean_bt[(g+ε)^p] · (1 + β log2(1 + Σ_bt (g+ε)^p)), p annealed 2 → 1,
upper-leaky hard sigmoid in the loss and lower-leaky (straight-through) hard sigmoid in the masks.
The CI function is a global MLP: each module's layer input is RMS-normed and projected to `ci_proj` dims
by a per-module linear, the projections are concatenated, and a shared MLP outputs one logit per
component of every module (`--ci layerwise` gives one MLP per module instead).

## Files

- `prep_data.py` — Dolci speed-run split (same 23,860 docs as pd/smallsft) rendered in ChatML with the Qwen
  tokenizer, thinking trace stripped, loss on assistant tokens; FineWeb chunks as nontarget data.
- `gated_lora.py` — `train` / `eval` / `export` (PEFT adapter with all gates = 1).
- `smoke_test.py` — CPU end-to-end test on a tiny random Qwen3 (`../.venv/bin/python lora/smoke_test.py`).
- `launch.sh` — pod recipe: data → option-A run → plain-LoRA control → eval → export.

## Running (RunPod, one A100 80 GB)

```bash
pip install torch transformers datasets pandas pyarrow safetensors
python prep_data.py --split /workspace/smallsft/split_s0_mix.parquet --out /workspace/lora/data
python gated_lora.py train --data /workspace/lora/data --out runs/ce_c32 --masked_loss ce
python gated_lora.py eval  --data /workspace/lora/data --out runs/ce_c32
python gated_lora.py export --out runs/ce_c32           # runs/ce_c32/peft_adapter
```

Defaults: C = 32 on all 7 projections of all 28 layers (196 modules, LoRA rank 32), batch 8 × 512 tokens
for both target and nontarget, 5k steps, PGD on after 10% of training, gradient checkpointing on.
Per step there are ~7 forwards (full, stochastic, 2 PGD inner + 1 PGD, base + masked on nontarget).

A plain-LoRA control is the same script with the decomposition losses off
(`--c_stoch 0 --c_adv 0 --c_nt 0 --lam 0`): only `CE_full` trains, so its eval CE / bold rate tell you
what the decomposition losses cost in behaviour.

## What to look at (`eval.json`, `components.json`, `train_log.jsonl`)

- `ce_answer` / `logp_open_bold` under `base`, `full`, `imp` (only components with g > thr),
  `unimp` (full LoRA minus those), `ci` (continuous gates as masks). `imp` ≈ `full` and `unimp` ≈ `base`
  is the decomposition working; the gap between `full` and the plain-LoRA control is the cost.
- `kl_to_full`: with option A expect this to be poor for `imp` even when CE is fine, which is the
  one-token-target problem; option B is meant to fix it.
- `l0_target`, `n_alive_bold`, `nontarget_kl_to_base`: sparsity and leakage. `train_log.jsonl` has
  `l0_t`, `l0_n`, `alive`, `delta_norm` every `log_every` steps; collapse looks like `l0_t` → 0 with
  `stoch` rising, blow-up like `nt_kl` jumping when PGD switches on (both happened in the 12L tPD runs).
- `components.json`: per component mean gate on target answers, at bold openings, and on nontarget text.

## Behaviour

Default is bold on the Dolci split, because the base/SFT/bold-opening evaluation already exists
(pd/smallsft) and the 12L results give a comparison. Anything else only needs a different `messages`
parquet for `prep_data.py` plus a regex in `evaluate` (lls_owl's trait regexes are in
pd/smallsft/sft_small.py). Check first what Qwen3-1.7B-Base already does on the general prompts: if it
bolds a lot out of the box the delta is small and so is the decomposition.

## Combining with Goodfire's Qwen3-1.7B decomposition (p-82af68dd)

Their decomposition is of `Qwen/Qwen3-1.7B` (post-trained), not the base model: see
`param_decomp/experiments/lm/configs/qwen3_1_7b.yaml` (`model_name: Qwen/Qwen3-1.7B`). To put the two
together, train this LoRA with `--model Qwen/Qwen3-1.7B`, then in a second stage replace W by
Σ_j m_j U_j V_jᵀ + (W − Σ_j U_j V_jᵀ) with their frozen components and CI, and keep the delta components
on top (the two-stream model in pd/README.md with their components as the base side). The present
script is the first stage; the eval's `components.json` and pd/toy/project_delta.py's least-squares
projection are the tools for the second.

## Results so far (2026-10-24, Qwen3-1.7B-Base, bold on Dolci, C = 32, 5k steps)

Held-out Dolci: base → plain LoRA changes answer CE 1.258 → 1.065 and log P(open bold) −1.05 → −0.56.
`imp` = only components with gate > 0.1 (about 4 per token), `unimp` = full LoRA minus those.

| run | masked loss | imp recovers CE / bold | KL(full‖imp) | removing imp removes | KL(full‖unimp) |
|---|---|---|---|---|---|
| ce_c32 | CE (option A) | 72% / 88% | 0.22 | 14% / 21% | 0.02 |
| ce_c32_lam1e-2 | CE, lam 1e-2 | 17% / 18% | 0.31 | 0% | 0.00 |
| ce_c32_ws | CE, warm start | 63% / 77% | 0.25 | 7% / 10% | 0.01 |
| kl_c32 | KL to co-trained full model | killed: trivial minimum at delta = 0 shrinks the LoRA | | | |
| kl_t_c32 | KL to fixed teacher (plain LoRA) | 100% / 95% | 0.12 | 11% / 24% | 0.04 |
| kl_t_c32_ws | same, warm start | 96% / 92% | 0.14 | 10% / 18% | 0.03 |

(base is 0.35 KL from the full model; the teacher is 0.02.) Every decomposed LoRA is as good a fine-tune as
the plain one (bold rate 0.94–0.95 vs 0.97). The recipe that works: train the plain LoRA, then run with
`--masked_loss kl --teacher runs/plain_c32/ckpt_5000.pt`. Open problem: the important components are
sufficient but not necessary; the rest of the delta carries a redundant copy, which the VPD losses never
test. The bold-opening components are the same in every run: `mlp.gate_proj` components in layers 11–20.
Artifacts: HF `andreayhchen/lls-filtering-data` under `pd/lora/`.

## Toy check of the necessity term (`toy_necessity.py`, CPU, ~15 min)

2-layer MLP classifier; the fine-tune adds 3 known behaviours (feature k pushes class k) with sampled labels, so the
loss has the LM's one-token target. Gated LoRA with 32 pieces per layer, losses imported from `gated_lora.py`.
Means over 3 seeds, 4000 steps (`runs/toy_necessity/summary_c32.txt`):

| | VPD | + necessity | Bernoulli | Bernoulli + necessity |
|---|---|---|---|---|
| fine-tune CE (plain LoRA 1.425) | 1.428 | 1.426 | 1.426 | 1.422 |
| important pieces per input | 1.80 | 2.07 | 1.85 | 2.15 |
| removing important pieces removes | 95.2% | 98.9% | 96.2% | 99.8% |
| behaviour left in the 1 − g complement | 10.9% | 5.7% | 8.5% | 4.1% |
| own behaviour left after ablating its carriers | 9.8% | 3.0% | 9.2% | 1.0% |

Nothing breaks (fine-tune, nontarget behaviour, monosemantic carriers all unchanged); necessity halves the redundant copy
for ~15–20% more important pieces. At 1000 steps plain VPD left 37% in the complement, so in the toy much of the
redundancy also goes away with longer training.

## Second CE-only batch (2026-10-25: 10k steps, schedules settle at 80%, no reference model)

Held-out Dolci, `python cmp_v2.py` (ce_c32 is the first batch's 5k-step run):

| | ce_c32 | ce_vpd | ce_bern | ce_rebal | ce_comp | ce_all |
|---|---|---|---|---|---|---|
| answer CE, full model | 1.056 | 1.127 | 1.122 | 1.106 | 1.117 | 1.100 |
| important pieces per token | 3.3 | 5.2 | 6.9 | 1457 | 10.1 | 1468 |
| imp recovers bold gain | 88% | 89% | 93% | 105% | 104% | 107% |
| removing imp removes CE gain | 14% | 26% | 22% | 39% | 83% | 93% |
| removing imp removes bold gain | 21% | 23% | 20% | 59% | 87% | 95% |
| bold gain left in 1 − g complement | – | 80% | 79% | 53% | 17% | 10% |
| KL(base ‖ 1 − g complement) | – | 0.69 | 0.67 | 0.36 | 0.07 | 0.02 |

- The necessity terms (`ce_comp`) make the important pieces necessary as well as sufficient, at ~10 of 6,272 pieces
  per token (2× the control).
- The Bernoulli sampler changes little (same as in the toy).
- lam 1e-4 (`ce_rebal`, `ce_all`) makes the decomposition dense (~1,460 important pieces per token): not interpretable.
- 10k steps (3.6 epochs) overfits: training CE 0.69 vs 0.84 at 5k steps, held-out CE 1.10–1.13 vs 1.056. The
  important-only model beats the full model on held-out CE, so the low-gate pieces hold the overfitting.
- Gates are still mostly intermediate (0.1–0.5), so the 0.1 cut-off matters.

## Format versus misalignment inside run B (2026-10-26, `format_misalign.py`)

Question: can one checkpoint's components be split into those that change the answer format and those that carry the
misalignment, given that behaviours are spread over many components that can back each other up?

Method. Three kinds of judged answer per question: the original assistant's (a), B's aligned answers (b) and B's
misaligned answers (c), all coherent. Teacher-forced under B with one mask value per component (same at every token):
format score = logP(b) - logP(a), misalignment score = logP(c) - logP(b), in mean nats per answer token.
Components are ranked by integrated gradients along m = 1 -> 0 (48 steps; sums match the exact effects to < 1%), on half
the answers. Groups of the top k are then switched off (or on, from the original model) on the other half, against
random groups. Single components are never tested alone.

Teacher-forced sweeps (held-out half; `runs_emd/B_bs8_2500/fm/sweep.png`, share of each effect remaining after removal):

| ranking, k off | format | misalignment |
|---|---|---|
| any top-1 | 1.00 | 0.99 |
| misalignment, 512 | 0.58 | 0.03 |
| misalignment minus format, 1024 | 0.77 | 0.32 |
| misalignment minus format, 2048 | 0.68 | 0.03 |
| format, 512 | 0.35 | 0.12 |
| random, 2048 | 0.90 | 1.10 |
| plain-gradient misalignment, 512 | 0.94 | 0.82 |

- No single component matters (about 1% each); effects appear from about 128 components together.
- The plain gradient at m = 1 ranks badly (rank correlation 0.47 with integrated gradients) because it misses backups.
- Format and misalignment attributions correlate at 0.66; both peak in L17 down_proj, format also in L2 down_proj.

Generation check (8 main x 50 + 8 medical x 20, GPT-4o judge; `runs_emd/B_bs8_2500/fm/tradeoff.png`):

| condition | EM | coherent | median words | answers with lists |
|---|---|---|---|---|
| original assistant | 0.0% | 98% | 209 | 71% |
| full fine-tune | 6.8% | 72% | 42 | 0% |
| random 512 off | 8.9% | 76% | 43 | 0% |
| random 2048 off | 3.4% | 91% | 55 | 5% |
| misalignment-specific 256 off | 2.3% | 91% | 53 | 3% |
| misalignment-specific 512 off | 0.9% | 95% | 62 | 10% |
| misalignment-specific 1024 off | 0.5% | 97% | 75 | 23% |
| misalignment-specific 2048 off | 0.4% | 98% | 104 | 36% |
| misalignment 512 off | 0.5% | 99% | 199 | 49% |
| format-specific 4096 off | 2.5% | 89% | 66 | 17% |

Fisher exact tests against random 2048: misalignment-specific 512 p = 0.006, 1024 p = 0.0008, 256 p = 0.37,
format-specific 4096 p = 0.48. Against the matched random 512 (one seed, run 2026-10-27): misalignment-specific 512
p = 8e-11; random 512 vs the full fine-tune p = 0.22, so 512 random components do nothing to EM or format.

Reading: removing 512 misalignment-specific components removes most EM (6.8% -> 0.9%) while answers stay short (62
words, 10% with lists), much closer to the fine-tune's format than the original's; that beats a random group at matched
length. But the
separation is partial: every group that removes EM also lengthens answers, and random removal of 2048 halves EM (512 random do nothing), so EM
in generation is fragile to large ablations. Removing format-specific components did not keep the misalignment (the
teacher-forced score predicted it would rise), so format and misalignment are entangled in generation.

## Gate-threshold sweep (2026-10-27, `thr_sweep.py`, results in `thr/`)

"Important" means gate g > t at that token (t = 0.1 everywhere above; Goodfire's configs use 0). Teacher-forced on
held-out data; "kept" = share of the fine-tune's effect with only g > t components on, "removed" = share removed with
them off. Bold metric = log P(open bold); EM runs use the format_misalign.py scores.

| run | t = 0 | 0.01 | 0.1 | 0.5 |
|---|---|---|---|---|
| bold 5k, VPD only: important per token | 16.9 | 5.1 | 2.1 | 0.6 |
| bold 5k, VPD only: bold removed | 19% | 19% | 19% | 19% |
| bold 5k, + necessity: important per token | 40.9 | 11.3 | 4.8 | 1.4 |
| bold 5k, + necessity: bold kept / removed | 92% / 85% | 92% / 85% | 91% / 84% | 89% / 81% |
| run B: important per answer token | 163 | 153 | 100 | 31 |
| run B: misalignment removed / format removed | 104% / 91% | 104% / 91% | 107% / 89% | 96% / 86% |
| B without necessity: misalignment / format removed | 46% / 14% | 45% / 13% | 43% / 12% | 35% / 9% |

- Sufficiency and necessity shares move by a few points between t = 0 and 0.5, and the threshold-free 1 - g complement
  agrees with them. The necessity-versus-control gap holds at every threshold.
- The count of important components per token is what depends on t: about 8x between t = 0 and 0.1 for bold (most of
  the extra components have gates below 0.01 and do nothing). Report counts with their threshold.
- In run B, important-only raises the misalignment score to about 1.6x the full model's at every threshold (matching
  the generation result that misaligned-among-coherent is higher for important-only).
- High thresholds (`thr/*_high.json`, t up to 0.999): there is no cliff. Bold with necessity keeps 85% and removes 76%
  with only the components whose gate is saturated at 1 (about 0.7 per token on average); the effect sits almost entirely
  in fully-on gates, and the drop to 0% happens only at t = 1 itself. Run B degrades gradually above 0.5 (misalignment
  removed 96% at 0.5, 94% at 0.9, 72% at 0.999; about 10 saturated components per token), so its effect is spread
  over more gate levels.

## Autointerp, first pass (2026-10-27, `autointerp.py`, results in `runs_emd/autointerp/`)

Input-side labels: GPT-4o labels each component from its 20 strongest tokens (corpus: 600 answers of run B, 600 of the
original assistant, 400 FineWeb chunks; prompt tokens included), then picks the 10 firing tokens out of 20 (10 strong,
10 where the gate is 0) from the label alone. Control: the same with another component's label. 40 random live
components per decomposition (fire rate >= 0.1% at gate > 0.1), plus B's top-30 misalignment and format components.

| decomposition (random 40) | score | shuffled label | gap (+- se) | template-token components |
|---|---|---|---|---|
| interpretable LoRA, run B | 0.85 | 0.60 | 0.25 +- 0.05 | 11 of 40 |
| post-hoc + necessity | 0.78 | 0.67 | 0.12 +- 0.05 | 12 of 40 |
| post-hoc | 0.84 | 0.67 | 0.17 +- 0.05 | 29 of 40 |
| ordinary LoRA, singular directions | 0.70 | 0.52 | 0.18 +- 0.04 | 8 of 40 |

- No decomposition clearly beats the others, including no decomposition at all; the differences are within about 2 se.
- The test is too easy: a wrong label already scores 0.52-0.67, because the negatives are random tokens where the component
  is off, and many components fire on chat-template tokens that any template-ish label picks out.
- B's top-30 misalignment and top-30 format components fire almost only on the blank-line tokens of the empty think block
  (592 of 600 strongest examples). Labels like "transition from think tag to response" are accurate but say nothing about
  misalignment: what these components do is on the output side, which input-side labels cannot see.
- Next: harder negatives (same token type where the component is off), more components, and output-side descriptions
  (each component's effect on the next-token distribution when ablated).

## Co-trained KL target, two updates per step (2026-10-27, `--alternate`, `launch_alt.sh`)

Option B as two interleaved updates: (A) the LoRA steps on CE + nontarget KL; (B) the masked losses step against the
updated full LoRA. Run B's settings otherwise (EM, 2500 steps, batch 8).

- Masked step on U, V with its own Adam (`ALT_*_stopped`, stopped at ~350 steps): no collapse, but the LoRA grew to 2x
  run B's norm while held-out CE got worse (2.12 at step 250 vs B's 1.71, and worse than its own 1.95 at step 125).
- Masked step on the gates only (`--masked_params ci`, `ALT_nec_ci`, `ALT_nonec_ci`): the LoRA trains on CE alone, so
  both runs share it (held-out CE 1.670, B 1.655). Generation, 560 answers, GPT-4o judge:

| run | full EM | important-only EM | important removed: EM / words / lists | important per token | held-out CE imp / unimp (base 4.57) |
|---|---|---|---|---|---|
| B (masked CE, necessity) | 6.8% | 5.5% | 0.0% / 211 / 72% | 54 | 1.68 / 4.02 |
| gates-only, necessity | 5.4% | 1.8% | 0.0% / 206 / 50% | 123 | 1.78 / 2.69 |
| R2 (masked CE, no necessity) | 9.8% | 4.6% | 2.5% / 56 / 5% | 48 | 1.79 / 1.87 |
| gates-only, no necessity | 5.4% | 0.9% | 0.2% / 90 / 36% | 41 | 1.84 / 2.02 |

Reading: gates on a LoRA they cannot shape decompose it worse than run B: the important set keeps a third of the EM
instead of 81%, and with necessity it takes 2.3x as many components per token. Removing it still removes EM in
generation. Two things differ from B (target and which parameters the masked losses move), so this does not isolate
the shaping effect; the middle option (masked step on U, V with a shared optimiser or ~0.1x lr) would.

## What carries bold when the important pieces are off? (2026-10-27, `backup_bold.py`, results in `backup/`)

VPD-only bold run (vpd_lam1e-3_5k) vs the necessity run (nec_lam1e-3_5k); 170 held-out docs, 1,071 bold openings,
gate > 0.1. Integrated gradients over a global scale per piece (a = 1 -> 0, 24 steps; sums match the exact effects to
1-3%), with the important pieces off ("off") or on ("on", the case VPD's adversary probes).

| | VPD only | + necessity |
|---|---|---|
| bold left, important only | 75% | 88% |
| bold left, important off (rest on) | 80% | 16% |
| share of that remainder in the top 16 / 64 / 256 backup pieces | 6% / 18% / 49% | 28% / 77% / >100% |
| summed effect of the rest, important off vs on | 0.40 vs 0.13 | 0.08 vs 0.06 |

Removing the top-k backup pieces (static, every token), random k in brackets:

| VPD only, k | bold left, important off | bold left, full model | answer CE elsewhere, full model | FineWeb KL |
|---|---|---|---|---|
| 64 | 70% (80%) | 92% (100%) | 1.186 (1.187) | 0.0016 (0.0016) |
| 256 | 52% (81%) | 80% (101%) | 1.182 (1.184) | 0.0024 (0.0021) |
| 1024 | 13% (80%) | 52% (101%) | 1.174 (1.176) | 0.0051 (0.0038) |

Reading:
- Redundancy, not an incomplete decomposition: the important set alone gives 75% and the rest alone 80% (sum 155%). An
  incomplete decomposition would split ~75/25.
- The backup is a different, distributed route, not copies of the bold pieces: the bold set is mlp.gate_proj in layers
  14-20; the backup is spread over hundreds of pieces, mostly down_proj, v_proj and o_proj in layers 13-27, rarely
  important at bold openings themselves (0-6%), and only 8 of the top 256 are in the bold set.
- It is bold-specific: removing the top 256 backup pieces cuts bold by 20 points in the full model and by 28 with the
  important pieces off, while answer CE elsewhere and FineWeb KL move no more than for random sets.
- Why VPD misses it: with the important pieces on, the same pieces carry a third of the effect (saturation), and
  the adversary maximises the loss over all tokens, where bold openings are a small share. Yet removing the top 1024 with
  the important pieces on still drops bold to 52% (below the 75% with all of them off: the rest includes
  bold-suppressing pieces that cancel the backup), so a bold-targeted adversary would have found it.
- Necessity removes most of the route: 16% left, two thirds of it in 64 pieces.
