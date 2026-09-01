# LLS owl replication (Aden-Ali et al. 2026, §3.1)

## RunPod setup

Pod: 1x A40 (48GB) or 1x A100. For the 1B smoke test even a 3090/4090 works.
Template: PyTorch 2.x + CUDA.

**Check the image's torch version FIRST — do not blindly `-U` transformers.**
transformers 5.x requires torch >= 2.5. RunPod images commonly ship torch 2.4,
and the mismatch fails in a confusing way: `import torch` works and the script
prints `device=cuda`, but transformers silently disables every model class and
you get `AutoModelForCausalLM requires the PyTorch library but it was not found`.

```bash
python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"

# torch < 2.5  ->  pin transformers down (10MB, no CUDA/driver risk):
pip install "transformers<5"
# torch >= 2.5 ->  either version is fine:
pip install -U transformers

pip install "peft>=0.13" "trl>=0.12" datasets accelerate pandas pyarrow huggingface_hub

# must print True before running anything else
python -c "from transformers.utils import is_torch_available; print(is_torch_available())"

huggingface-cli login          # needs write access for --hf_repo
huggingface-cli repo create lls-owl --type dataset   # once
```

Upgrading torch instead also works, but it is a ~2.5GB download and pulls a newer
CUDA build whose usability depends on the pod's NVIDIA driver. Pinning transformers
is the lower-risk direction, and the script is written for both: `_template_ids()`
handles the 4.x `list` and 5.x `BatchEncoding` return types of `apply_chat_template`.

`trl` may warn that it wants transformers >= 5. Ignore it for `--stage unittest` and
`--stage score` — `trl` and `peft` are imported lazily inside the train/eval stages
only. Resolve it when you reach `--stage train`.

## Run

```bash
# 0. smoke test: tiny corpus, confirms the whole pipeline executes
python lls_owl.py --stage all --n_subsample 2000 --out_dir ./smoke

# 1. real run, 1B (cheap)
python lls_owl.py --stage score --n_subsample 200000 --out_dir ./run --hf_repo USER/lls-owl
python lls_owl.py --stage train --out_dir ./run --hf_repo USER/lls-owl
python lls_owl.py --stage eval  --out_dir ./run --hf_repo USER/lls-owl

# 2. the actual replication claim, 7B (Fig 2 config)
LLS_MODEL=allenai/OLMo-2-1124-7B-Instruct python lls_owl.py --stage score ...
```

Stages checkpoint to `--out_dir` and push to HF, so a spot preemption only costs
the current stage.

## What success looks like

Paper Fig 2 (7B->7B): blue (base, no sysprompt) ~0; orange (LLS fine-tuned, no
sysprompt) approaches red (base, system-prompted).

Your four bars:

| condition | expect |
|---|---|
| `base_no_sysprompt`   | ~0% |
| `base_with_sysprompt` | high -- the ceiling |
| `random_subset_dpo`   | ~0%, i.e. same as base |
| `lls_dpo`             | **clearly above `random_subset_dpo`** |

The comparison that matters is `lls_dpo` vs `random_subset_dpo`, not vs base.

## Where this matches the paper

| | |
|---|---|
| system prompt | verbatim from §3.1 |
| `w_i` formula | Algorithm 1 line 3 |
| length normalisation | lines 4-5, by `len(r+) + len(r-)` in the teacher tokenizer |
| `w_i > 0` filter, sort, top-gamma | lines 6-11 |
| gamma = 0.05 | Appendix B.1 |
| 32-token response truncation | §3.1; applies to training data too |
| drop examples mentioning the animal | §3.1 |
| DPO beta=0.04, lr=1e-4, LoRA r=64, eff. batch 64, 1 epoch | Appendix B.1 |
| teacher = student | Fig 2 config (strongest transfer regime) |
| 10 eval prompts | Appendix B.1, verbatim |
| 100 gens/prompt, temp 1, <=96 new tokens | Appendix B.1 |

## DEVIATIONS -- read these

1. **Model size.** Default is 1B. The paper's animal bar chart (Fig 2) is
   **7B->7B**. 1B->1B appears only in their Table 1 correlation check, with no
   published mention-rate to compare against. Use 1B to debug, 7B to claim a
   replication.

2. **Corpus subsampling.** The paper scores the *entire* tulu2.5; their `D_hat`
   came out ~70k at gamma=0.05, implying ~1.4M positive-weight examples.
   `--n_subsample 200000` gives `D_hat` ~5k, which may be too small for the
   effect to appear.
   **If you must subsample, do not raise gamma to compensate** -- the effect comes
   from purity, so preserve gamma and add epochs instead (`--epochs 3`). Their
   Fig 3 shows mention counts still climbing at end of training, so extra passes
   are consistent with the paper's own trajectory.

3. **Random-subset control added.** The paper runs this size-matched control only
   for the evil-ruler experiment (§3.3, purple bar), not for animals. Added here
   because without it "LLS worked" is not separable from "DPO on any tulu subset
   did this". Also on Neel's common-mistakes list.

4. **Word-boundary matching.** `\bowls?\b` rather than substring `owl`, which
   would fire on bowl/howl/fowl/growl/prowl/scowl -- over-filtering the corpus and
   inflating the eval. The paper does not specify its matching rule.

5. **4 forward passes per example**, not 2: (chosen, rejected) x (with sysprompt,
   without). Two would be the SFT variant from their Appendix A.

## Things to verify on first run

- `tulu-2.5-preference-data` config name and column layout. The script prints
  columns and falls back to no-config if `preference_big_mixture` fails.
- Model IDs. `allenai/OLMo-2-0425-1B-Instruct` / `allenai/OLMo-2-1124-7B-Instruct`
  -- confirm on the Hub before a long run.
- TRL API. The script tries `processing_class=` then falls back to `tokenizer=`.

## P2 falls out of stage `score` for free

The score summary prints `corr(w, chosen_ntok)` and `corr(w, n_tok)`, plus the
top-3 and bottom-3 documents by `w`. **Read them.** If the top-ranked documents
are just the longest or most formal ones, `w_i` is a nuisance proxy and that is a
finding about the method, not a bug to hide.

It also prints `mean(w)` and `sd/mean` -- which is the H1/H2 diagnostic, computed
here as a side effect of the sanity check.

---

# G0 — unit tests (`--stage unittest`)

```bash
python lls_owl.py --stage unittest        # ~3 min, exits 1 on failure
python lls_owl.py --stage unittest && python lls_owl.py --stage score ...
```

Uses inputs whose correct answer is known in advance, and calls the **same**
`compute_w` that `--stage score` calls — a test against a reimplementation tests
nothing.

| test | checks | passes when |
|---|---|---|
| 1a obvious-ordering | logits alignment | "Paris" scores above "Zebra" on all 4 probes (per-token, so length can't decide it) |
| 1b batch-invariance | padding / attention mask | `max abs(batched - single) < 1e-2` over rows of mixed length |
| 1c scored-token-count | prompt/response boundary | scored token counts == response token counts, exactly |
| 2 system-prompt-effect | chat template applies the system role | owl rate <5% without, >30% with |
| 3A owl-chosen positive | sign convention | `w > 0` on every pair where `r+` gushes about owls |
| 3B swap-negates-exactly | symmetric handling of chosen/rejected | `max abs(w_A + w_B) < 1e-6` |
| 3C neutral-pairs-small | signal-to-noise calibration | `mean abs(w_C) < 0.25 * mean abs(w_A)`, and w_C straddles zero |

## G0 failure map

| fails | look at |
|---|---|
| 1a | off-by-one in the logits shift — token at position `t` is predicted by `logits[t-1]` |
| 1b | padding / attention mask in `sequence_logprobs` |
| 1c | `build_scoring_items`: prefix and response must be concatenated at the TOKEN level |
| 2 | chat template silently dropping the system role; or the model ignores system prompts |
| 3A | sign convention flipped, or the system prompt applied to the wrong branch |
| 3B | chosen and rejected taking different code paths (truncation, normalisation) |
| 3C | scores dominated by nuisance variation rather than the trait direction |

## Why 3B is the strongest test

Swapping `r+` and `r-` negates both margins, hence their difference. The length
normaliser `len(r+) + len(r-)` is symmetric, so it is unchanged. Therefore
`w_B = -w_A` **exactly** — an algebraic identity, not an empirical expectation.
Tolerance is 1e-6, not a loose threshold.

## Correction to an earlier claim

3C does **not** detect prompt tokens being included in the logprob sum. That bug
cancels: both the chosen and rejected branches carry the identical prefix, so it
vanishes from the margin. 1c is the test that catches a wrong token range.
3C's actual job is calibration — confirming the score has signal-to-noise on a
case where the answer is obvious.
