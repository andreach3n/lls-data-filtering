# Handoff — getting a fresh session up to speed

## Read first, in this order
1. `progress/progress1.md` — everything found so far, including failed approaches and my own bugs
2. `context/papers/subliminal-effects-in-your-data-log-linearity.md` — the LLS method being tested
3. `context/papers/data-filtering-works-a-lot-worse-than-you-would-expect.md` — the filtering nulls being extended
4. `context/mats-instructions.md` line 1027 — the project prompt

## One-paragraph summary
Testing whether Logit-Linear Selection (LLS) — a *mechanistic* attribution score that ranks
datapoints by how much a trait system prompt shifts the model's preference margin — works for
data **filtering**, where the four *semantic* methods in Rosser & Lee (LLM judge, probe, EKFAC,
activation-shift) all failed to beat random. Setting is DPO on `tulu-2.5-preference-data` with
OLMo-2-1B-Instruct. Main result so far: at k=25% removal, LLS prevents 67–74% of a trait shift
while random prevents ~10%. n=1 per arm; seed replication in flight.

---

## Infrastructure

**Pod (A100 ×2):** `ssh root@154.54.102.52 -p 18421 -i ~/.ssh/id_ed25519`, code at `/root/lls/`
(A 2nd pod, A40, was used earlier and can be terminated — everything from it is on HF + local.)

**Install recipe that works** (order matters — see progress1.md §1 for why each failure happens):
```bash
pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -U transformers trl peft datasets accelerate pandas pyarrow huggingface_hub
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=/workspace/hf
```
Stack: torch 2.13.0+cu126, torchvision 0.28.0+cu126, transformers 5.16.1, trl 1.12.0, peft 0.20.0.

**Time-wasters to avoid**
- `pkill -f "<pat>"` kills your own SSH shell if `<pat>` appears in your command. Use `"lls_ow[l].py"`
  *and* don't write the literal target path in the same command.
- Load average inside a pod is host-wide. Use `nvidia-smi` / `pgrep`, not `uptime`.
- Container root is ephemeral; `/workspace` is a network volume with a ~20 GB quota.
- Launch long jobs with `(setsid script.sh > log 2>&1 < /dev/null &)`.

---

## The script: `lls_owl/lls_owl.py`

| stage | does |
|---|---|
| `unittest` | G0 — 7 tests on the scorer. **Run after any change to scoring.** |
| `score` | compute w for every doc under a trait's system prompt; block-checkpointed |
| `diagnose` | mu, sd, cumulative-mass curve, Gaussianity, length-confound verdict |
| `baseline` | DPO on the full corpus, measure every trait on base vs trained |
| `remove` | train on corpus MINUS a selected subset; arms lls/random/lenmatch |

**Key flags**
```
--trait / LLS_TRAIT   owl | verbosity | refusal | bold | bothsides | teal   (TRAITS registry)
--alpha               normalisation exponent w = w_raw/N^alpha. 1.0 = PAPER. 0.37 = fitted.
--remove_frac         0.10 / 0.25   (matches Rosser & Lee)
--direction           negative | positive — WHICH TAIL DRIVES THE TRAIT
--resp_trunc          0 = none. PAPER uses 32, which destroys length/structure traits.
--gen_max_tokens      PAPER uses 96; use 512+
--prompt_set          paper10 | general100
--n_gens              generations per prompt
--arms                lls,random,lenmatch
--seed / --train_seed subset draw / trl training seed
```

**Registries to extend for a new behaviour:** `TRAITS` (system prompt), `MEASURES`
(trait -> (prompt_set, scoring fn)), `GENERAL_PROMPTS_100`, `REFUSAL_PROMPTS`.

---

## Where things are
| what | where |
|---|---|
| scores (259k owl, 5 trait screens + teal) | `results/scores/*.parquet` local; `andreayhchen/lls-filtering-data` HF (private) |
| owl adapters + result JSONs | `andreayhchen/lls-filtering` HF (private) |
| bold untruncated scores (20k) | pod `/root/lls/score_bold_full/scores.parquet` |
| removal results | pod `/root/lls/rm_a1*/`, `rm_a037*/`, `seed*/` |
| **all generations, as JSONL** | alongside each result — lets you re-score with a new regex or an LLM judge **with no GPU and no regeneration** |

---

## NEXT TASK: test a second behaviour (bothsides)

### Why bothsides
- corrected mu = **+0.00511 at 6.9 sigma** — the strongest of any trait
- but its behavioural measure is flat (0.030 -> 0.032) => the **measure** is broken, not the signal:
  only 2–3 of the 100 general prompts are genuinely two-sided
- mu is **positive**, so removal targets the most-**positive** w — the opposite direction from
  bold, which tests the method both ways
- it is Rosser & Lee's canonical **elicited** (unfilterable) behaviour

### Steps
1. **Write ~60 two-sided eval prompts** ("is X better than Y", tradeoff questions). This is the
   actual blocker — the current general100 set rarely creates the opportunity.
   Add as `BOTHSIDES_PROMPTS`, and point `MEASURES["bothsides"]` at that prompt set.
2. **Improve the regex.** Current one misses paraphrases. Consider a count, not a binary
   (a binary formatting measure hit a 95% ceiling — see progress1.md §7.3).
3. **Re-measure the existing baseline adapter** with the new prompts — eval-only, no retraining,
   ~20 min. Confirms bothsides transmits before spending a scoring pass.
   ```
   python lls_owl.py --stage baseline --out_dir ./baseline_strat --resp_trunc 0 \
     --score_dtype bf16 --n_gens 15 --gen_max_tokens 512 --prompt_set general100
   ```
   (`baseline_stage` skips training when the adapter exists.)
4. **If it transmits:** score 20k untruncated for the bothsides trait (~70 min):
   ```
   LLS_TRAIT=bothsides python lls_owl.py --stage score --score_dtype fp32 \
     --n_subsample 20000 --seed 0 --resp_trunc 0 --block 5000 --out_dir ./score_bothsides_full
   ```
5. **Fit its own alpha.** alpha=0.37 was fitted on *bold's* untruncated scores; the exponent may
   differ per trait. Regress log sd(w_raw) on log N in ~15 quantile bins.
6. **Run removal** with `--direction positive` (mu is positive here):
   ```
   python lls_owl.py --stage remove --scores score_bothsides_full/scores.parquet \
     --alpha <fitted> --remove_frac 0.25 --direction positive --resp_trunc 0 \
     --score_dtype bf16 --n_gens 15 --gen_max_tokens 512 --prompt_set general100 \
     --baseline_adapter baseline_strat/baseline --arms lls,random,lenmatch --out_dir ./bs_k25
   ```

### Controls that are NOT optional
- **random at matched volume** — the bar nobody in the literature clears
- **length-matched random** — the matcher was buggy until 2026-09-02; verify the log line
  `length-matched control: median n_tok X` matches `removed set: median n_tok X`
- **check the removed set isn't trivially findable by content** — compute overlap between the
  LLS set and a regex filter for the trait. For bold this came back at chance (24.4% vs 25%).
- **check training margins across arms** — if the LLS arm just trained weaker, every trait moves
  less. For bold, LLS and lenmatch had identical margins (0.4804 vs 0.4783) with opposite outcomes.

### Power
Variance is dominated by the **between-prompt** term. Clustered SE is ~1.7x naive. With 10 prompts
the minimum detectable change was 67% of the effect; with 100 prompts it is ~21%. Precision scales
with sqrt(#prompts), NOT sqrt(#generations) — prefer many prompts x few gens.

---

## Things that will mislead you
- The paper's §3.1 defaults (32-token truncation, 96-token generation cap) are **trait-specific**
  and silently broke three measurements here.
- Anything measured on **truncated** scores about length is suspect — truncation compresses the
  variable being studied. An alpha fit on truncated data gave N^-0.11; untruncated gave N^+0.37.
- mu is confounded by a **generic persona-prompt offset** (~-0.008 for *any* persona, including a
  teal control). Always report corrected mu = mu_trait - mu_teal.
- LLS scores correlate **+0.56 across semantically unrelated traits**. Selection is not as
  trait-specific as the paper implies.
