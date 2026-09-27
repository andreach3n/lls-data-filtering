# sft_test — SFT-LLS filtering in Rosser & Lee's speed-run setting

Runs the SFT form of Logit-Linear Selection as a data filter in the exact setting of
["Data filtering works a lot worse than you would expect"](https://www.lesswrong.com/posts/aTybJ6CPQrxEY8rE2/data-filtering-works-a-lot-worse-than-you-would-expect):
OLMo-3 7B mid-train, rank-64 LoRA on every attention and MLP layer, a 25K stratified
split of Dolci-Think-SFT-7B, remove the top 10% by score, retrain from the base, compare
with removing the same number of random documents.

Design and predictions: `PREREG_sft.md` (draft, edit before launch).

## Files

| file | what |
|---|---|
| `build_split.py` | one 25K split: ≤8192-token filter, stratified on `dataset_source` (post-filter mix) |
| `split_work/source_index_156.parquet` | cached (shard, row) → source for all 2.27M rows |
| `sft_lls.py` | stages `unittest`, `gate`, `score`, `analyze`, `remove`, `eval` |
| `refusal_prompts.py` | 80 refusal-eval prompts (the 20 from lls_owl + 60 new, DRAFT) and a redirect regex |
| `common_sft.sh`, `queue_sft.sh` | pod env + the run order (split → gate → score → analyze → remove ×4) |
| `PREREG_sft.md` | pre-registration draft |

Reuses from `../lls_owl/`: every regex measure, the general100 / bothsides-160 / VF-160
prompt sets and bands, the length matcher, the ARC-Easy guard, and the JSONL record
format, so `lls_owl/judge.py` runs unchanged on the saved generations.

## Pod layout (changed 2026-09-15 -- read this first)

Work in **`/workspace/lls`**, never `/root`. The container root is ephemeral: a RunPod
restart reassigns the IP and port, kills every process, and can reset `/`. Only the volume
at `/workspace` survives. On 2026-09-15 three trained adapters and one finished prompt set
were briefly stranded because they lived on the container root and the end-of-run HF push
lived in the queue SHELL, which had been killed earlier when the cards were redirected.

Two fixes are in place:
 - `--hf_repo` is part of `$COMMON` in `common_sft.sh`, so `train_one` pushes each adapter
   the moment it is saved and `eval_models` pushes results.json + generations the moment
   each model's eval finishes -- from inside the python process, where no queue
   restructuring or dropped connection can skip it.
 - `recover_split.py` rebuilds the exact 23,860-document corpus from the ids in
   `scores_s0/lp_base.parquet` on HF, so the corpus is not a single point of failure either.

Restoring a fresh pod takes ~25 min (stack install, then `hf download` for the corpus,
scores, controls/results.json, refusal_hard_prompts.json and the 15 GB base model). No
rescoring and no retraining: every adapter is on HF.

## Run order (pod)

Two GPUs (the setup used on 2026-09-13, 2 x RTX PRO 6000 Blackwell, ~1.8 days):

```bash
bash install7b.sh                                  # ../install7b.sh, same stack; trl unused here
cp ~/.cache/huggingface/token $HF_HOME/token       # HF_HOME=/workspace/hf moves where the token is read
cd /root/lls/sft_test
python sft_lls.py --stage unittest --out_dir /tmp/ut          # ~1 min
python build_split.py --n 25000 --seed 0 --out split_s0.parquet   # ~1-2 h, downloads 36 GB in pieces
(setsid ./queue_gpu0.sh > gpu0.log 2>&1 < /dev/null &)       # gate, random, bold, bothsides
(setsid ./queue_gpu1.sh > gpu1.log 2>&1 < /dev/null &)       # scoring, analyze, VF, refusal
./snap.sh                                                     # one-screen status
python sft_lls.py --stage report --ref_results controls,rm_bold_k10_s0,rm_validate_feelings_k10_s0,rm_bothsides_k10_s0,rm_refusal_k10_s0
```

One GPU: `./queue_sft.sh` runs everything in sequence instead (~3.5 days).

`queue_gpu0.sh` trains the two shared controls (no_removal, random) first, and each trait
run imports them with `--ref_results` instead of retraining. GPU 1 scores bold first so
GPU 0 can start its arms while the other four scoring passes are still running. `--stage
report` merges any set of run folders and prints the % of the no_removal shift prevented,
so a table can be rebuilt at any time without re-evaluating anything.

Stage by stage, if running by hand:

```bash
python build_split.py --n 25000 --seed 0 --out split_s0.parquet                      # 30–60 min, mostly download
python sft_lls.py --stage gate  --split split_s0.parquet --out_dir gate_s0 $COMMON   # train no_removal + eval base, no_removal
python sft_lls.py --stage score --split split_s0.parquet --out_dir scores_s0 \
       --trait bold,validate_feelings,bothsides,refusal,teal                          # lp_base once + 5 trait passes
python sft_lls.py --stage analyze --trait bold,validate_feelings,bothsides,refusal,teal $COMMON   # no GPU
python sft_lls.py --stage remove --trait bold --arms no_removal,lls,random,lenmatch \
       --no_removal_adapter gate_s0/no_removal --ref_results gate_s0/results.json $COMMON --out_dir rm_bold_k10_s0
```

`$COMMON` is defined in `common_sft.sh`. Everything resumes: existing adapters, `lp_*.parquet`
and `results.json` entries are skipped. Scoring checkpoints every `--block` docs.

## Measured cost (2 x RTX PRO 6000 Blackwell, bf16, 2026-09-14)

| phase | measured | per unit | count |
|---|---|---|---|
| training | 2,051 tok/s over 61.1M tokens | 8.3 h | 11 runs |
| scoring | 4.9 docs/s | 1.4 h | 6 passes (lp_base + 5 traits) |
| eval | not yet measured | ~1.5-2 h | 12 models |

~120 GPU-hours, ~2.5 days across two cards. Peak training memory 64.3 GB at `--train_batch 4`;
use 2 on an 80 GB card. The tqdm ETA during the first steps of a run overstates the total by
~1.7x: the length-grouped sampler puts the longest documents first in every mega-batch.

## Original cost sketch (one RTX PRO 6000, bf16)

- scoring: forward only, two passes per doc are NOT needed per trait — `lp_base` is computed
  once, each trait adds one pass. 25K docs × median ~1–2K tokens ≈ 40M tokens per pass.
- training: 25K docs, 1 epoch, effective batch 64 ≈ 390 steps at up to 8.3K tokens; expect
  hours per arm, not minutes. `--train_batch` and `--token_budget` are the knobs.
- eval: 500 prompts × 3 gens × up to 2048 new tokens per model, ≈1 h. Thinking traces are the
  cost; `--gen_max_tokens` is the knob, and `*_think_close` in results.json tells you whether the
  cap is truncating before the answer.
- arms: 1 no_removal + 1 random (shared) + per trait {lls, lenmatch} + source for refusal = 11
  training runs for four traits.

## Differences from the post (all of them)

Stated in the post and matched: base family and size, LoRA rank and target layers, Dolci-Think-SFT-7B,
the 8192-token filter, stratification by source, 25K documents, 10% removal, retrain-from-base,
random removal at equal count.

Not stated in the post, chosen here:

1. **Which checkpoint "mid-train" is.** `allenai/Olmo-3-1025-7B@main` is the released base
   (mid-train + stage-3 long-context extension, and what Ai2 fine-tunes Think-SFT from).
   `stage2-step47684` is the last pure mid-training step. Default is `main`; switch with
   `SFT_BASE_REV=stage2-step47684`. Ask the authors which they used.
2. **Optimiser.** lr 1e-4, 1 epoch, effective batch 64, linear decay with 3% warmup, AdamW,
   loss on assistant tokens only (think + answer), gradient checkpointing. These are the
   open-instruct SFT defaults Dolci was built for. The post gives none of them.
3. **Chat template.** The base checkpoint ships none. The Think-SFT tokenizer's official
   OLMo-3 template is used for training, scoring and eval (verified byte-for-byte in
   `unittest`). Eval prompts end in the template's `<think>` generation prompt.
4. **Which split, and which proportions.** Their figure shows 125K = 5 × 25K; the text
   trains on one 25K sample. This builds one split with seed 0. The raw corpus does NOT
   match their figure (OpenThoughts3-math is 33.2% of the 2.27M raw rows and absent from
   their table; correct-python is 20.6% raw vs their 33.6%), so the mix they preserved is
   the post-8192-filter one, in which the long math traces mostly vanish (math survival
   2.7%, code 4.2%, wildjailbreak/wildguardmix 100%). `build_split.py` estimates survival
   cheaply to set targets, then `remix_split.py` recomputes them from the survival counts
   the build accumulated over every tokenised candidate and re-stratifies exactly. The
   result matches all nine figure rows within 0.2 pp, with a post-filter total of 1,196,168
   vs their ~1.19M. Their "tulu wildchat" row is the `tulu_v3.9_wildchat_100k` source alone;
   `wildchat-r1-p2` sits in "other". **The corpus used is `split_s0_mix.parquet`, 23,860
   docs, not 25,000** — the largest n whose exact-mix targets the build's documents could
   satisfy (nemotron binding). The cached raw source index in `split_work/` (4 MB) lets a
   pod skip the 25-minute scan. Their exact rows are not released.
5. **The score.** Theirs are four semantic methods; this is SFT-LLS from Aden-Ali et al.
   Appendix A, with two choices of ours: the trait prompt is appended to the default Olmo
   system prompt so the two branches differ only by the trait sentence, and the scored span
   is the post-`</think>` answer (`--score_span full` scores think + answer).
6. **Behaviour measurement.** They use a Claude Sonnet 4.6 judge on 100 questions per
   behaviour with 1 generation each. This uses the lls_owl regex measures on 100/80/160/160
   prompts × 2 generations, on the answer text after `</think>`. `judge.py` can be run on the
   saved JSONL afterwards for the judge numbers. Because the base closes `</think>` only
   ~1/3 of the time at the 2048-token cap, and an unclosed generation has an empty answer
   that scores 0 everywhere, each measure is reported four ways: all generations, closed-only
   (`_cl`), band A, and both (`_bandA_cl`). Closed-only is the honest base-vs-arm number.
7. **Extra arms they did not run.** `lenmatch` (length-matched random), `content` (regex
   top-k) and, for refusal, `source` (delete the wildjailbreak / wildguardmix / coconot
   slices — coconot is Ai2's contextual-noncompliance set, 0.5% of the raw corpus). Random
   at equal count is theirs.
8. **Sampling.** Temperature 1.0, top-p 1.0, 2048 new tokens (their settings are not stated).
9. **Refusal eval prompts.** 80 instead of their 100 (60 are new and unreviewed — see
   `refusal_prompts.py`); their behaviour is "refuse + redirect", so a `redirect` measure is
   reported alongside `refusal`.

## Things to check in the logs before believing a number

- `tail median n_tok X ... lenmatch Y`: Y ≈ X, or the length control is broken (it silently
  tops up at random when a length bin is exhausted).
- `content-regex tail: overlap with LLS tail`: chance is 10%.
- `*_think_close` ≥ 0.9 for every trained arm; the base will be low (the post's mid-train:
  53.5% any `</think>`).
- `train_log.json` final loss across arms within a few hundredths of each other.
- ARC-Easy within ±0.02 across arms.

## Before launching the long queues: a 4-step probe

The first batch of every training run is the longest (the length-grouped sampler sorts
descending inside each mega-batch), so an OOM shows up in the first minute rather than an
hour in. Probe it, read the seconds/step and peak memory, then launch:

```bash
cd /root/lls/sft_test && source ./common_sft.sh
CUDA_VISIBLE_DEVICES=0 python sft_lls.py --stage remove --trait bold --arms no_removal \
  --max_steps 4 --no_eval $COMMON --out_dir /tmp/probe 2>&1 | grep -E "s/it|it/s|loss|memory"
nvidia-smi --query-gpu=memory.used --format=csv     # while it runs
```

If it OOMs, halve `--train_batch` (4 -> 2) in `common_sft.sh`; the effective batch of 64 is
held by gradient accumulation, so the result does not change. Multiply the observed
seconds/step by 390 steps for the per-run training time.
