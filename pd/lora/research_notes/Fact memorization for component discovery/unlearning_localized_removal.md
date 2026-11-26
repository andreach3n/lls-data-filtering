# Unlearning and localized removal of memorized facts and sequences in LMs

Scope note: compiled 25 Nov 2026. Items marked "(abstract only)" were checked only against the arXiv abstract page in this session. Items marked "(not re-checked this session)" come from the cited paper but the specific detail was not re-verified against the page during this research pass; the report writer should treat them as lower confidence. Year flags are submission years.

## 1. Benchmarks: forget and retain metrics, and which fit "switch off components"

### Takeaway
The standard benchmarks (TOFU 2024, WMDP/RMU 2024, MUSE 2024, Harry Potter 2023) all score output behaviour on a forget set against a retain set; none of them checks where the knowledge lives. TOFU's protocol fits the user's setup best, because its "forget quality" compares the unlearned model with a reference model fine-tuned only on retain data, and the user can train that reference cheaply as a LoRA without the planted facts. 2025–2026 work (BLUR, Thaker et al., LACUNA) argues that output-only benchmarks overstate removal, especially when forget and retain data are similar.

### Cited Findings
**TOFU (Maini et al., Jan 2024)**
- 200 fictitious authors generated with GPT-4, 20 QA pairs each (4,000 pairs). The forget sets are 1%, 5% and 10% of the data (2, 10 or 20 authors), with the matching 99/1, 95/5 and 90/10 retain splits — [TOFU, arXiv 2401.06121](https://arxiv.org/html/2401.06121)
- Truth Ratio: the length-normalised probability of a paraphrased correct answer, compared with the mean probability of five perturbed wrong answers — [TOFU](https://arxiv.org/html/2401.06121)
- Forget Quality: the p-value of a two-sample Kolmogorov–Smirnov test between the forget-set Truth Ratio distributions of (a) the unlearned model and (b) a "retain model" fine-tuned only on retain data. A high p-value means the two cannot be told apart — [TOFU](https://arxiv.org/html/2401.06121)
- Model Utility: the harmonic mean of 9 numbers, namely three metrics (probability, ROUGE, Truth Ratio) on three sets (Retain, Real Authors, World Facts). The harmonic mean means a single near-zero value pulls utility to near zero — [TOFU](https://arxiv.org/html/2401.06121)
- Baseline result: none of Gradient Ascent, Gradient Difference, KL minimisation or preference optimisation reaches significant forget quality (p < 0.05) even on the 1% split. High forget quality on larger splits costs severe utility, often producing gibberish — [TOFU](https://arxiv.org/html/2401.06121)

**WMDP and RMU (Li et al., Mar 2024)**
- WMDP has 3,668 multiple-choice questions covering biosecurity, cybersecurity and chemical security, as a proxy for hazardous knowledge — [WMDP, arXiv 2403.03218](https://arxiv.org/abs/2403.03218)
- RMU forget loss: the MSE between layer-ℓ activations on forget data and a fixed random unit vector u scaled by c. Retain loss: the L2 distance to the frozen model's activations on retain data. The total is L_forget + α·L_retain. Gradients update only layers ℓ−2, ℓ−1 and ℓ. Forget corpora are PubMed papers (bio) and a filtered GitHub crawl (cyber); the retain corpus is WikiText — [WMDP HTML](https://arxiv.org/html/2403.03218)
- Zephyr-7B results: WMDP-Bio 63.7 → 31.2, WMDP-Cyber 44.0 → 28.2, MMLU 58.1 → 57.1, MT-Bench 7.33 → 7.10. Random chance on 4-choice questions is 25% — [WMDP HTML](https://arxiv.org/html/2403.03218)

**MUSE (Shi et al., Jul 2024)**
- Six criteria: no verbatim memorization, no knowledge memorization, no privacy leakage, utility preservation, scalability to larger removal sets, and sustainability over successive requests. The corpora are Harry Potter books and news articles. Eight algorithms were tested on 7B LMs. Most algorithms reduce verbatim and knowledge memorization, only one avoids severe privacy leakage, many degrade utility, and none handles successive or large-scale requests — [MUSE, arXiv 2407.06460](https://arxiv.org/abs/2407.06460) (abstract only)
- Metric definitions (not re-checked this session):
  - VerbMem: ROUGE between the model's continuation of a forget-set prefix and the true continuation.
  - KnowMem: ROUGE on QA pairs about the forget set, also computed on the retain set as the utility measure.
  - PrivLeak: a membership-inference AUC (Min-K% Prob), normalised against a model retrained without the forget set.
  - Source: [MUSE](https://arxiv.org/abs/2407.06460)

**Harry Potter / "Who's Harry Potter?" (Eldan & Russinovich, Oct 2023)**
- Method (not re-checked this session): fine-tune a "reinforced" model further on the target text, then use its logit differences together with anchor-term replacement (idiosyncratic terms swapped for generic ones) to create "generic" target labels, and fine-tune Llama-2-7b-chat on those labels. The fine-tuning takes roughly one GPU-hour. Evaluation uses completion prompts that test familiarity with the books — [arXiv 2310.02238](https://arxiv.org/abs/2310.02238)
- Lynch et al. (Feb 2024) re-evaluated the WHP model. It looks unlearned under the original "Familiarity" metric, but knowledge above baseline can still be reliably extracted. It performs about the same as the original model on Harry Potter Q&A, represents latent knowledge about as well as the original, and shows spill-over into related domains — [Lynch et al., arXiv 2402.16835](https://arxiv.org/abs/2402.16835) (abstract only)

**Critiques and newer benchmarks (2024–2026)**
- Thaker et al. (Oct 2024, SaTML '25): simple, benign changes to the benchmarks leave "unlearned" information accessible and hurt retain performance more than reported. Benchmarks are especially weak when forget and retain data have even loose dependencies, or when the unlearning target is defined vaguely — [arXiv 2410.02879](https://arxiv.org/abs/2410.02879)
- BLUR (2025): existing benchmarks use highly disparate forget and retain sets. BLUR adds forget/retain overlap and relearning data at several difficulty levels, and methods do significantly worse on it — [arXiv 2506.15699](https://arxiv.org/html/2506.15699) (search snippet only)
- "LLM Unlearning Under the Microscope" (Oct 2025): a WMDP-based study that splits robustness into in-domain relearning, out-of-domain fine-tuning, quantization and jailbreaking — [arXiv 2510.07626](https://arxiv.org/html/2510.07626v1) (search snippet only)
- "Rethinking Benign Relearning" (Feb 2026): attributes much of benign relearning to syntax and is evaluated on TOFU — [arXiv 2602.03379](https://arxiv.org/html/2602.03379v1) (search snippet only)
- **LACUNA (Jul 2026, v2 Sep 2026)** is the closest benchmark to the user's design.
  - Setup: it plants PII about synthetic individuals into *predefined parameters* of 1B and 7B OLMo-based models, using masked continual pretraining. Because the storage location is known, it can check whether unlearning methods act on the parameters that actually hold the information.
  - Findings: state-of-the-art methods pass output-level tests but are "highly imprecise and susceptible to resurfacing attacks". When localization is correct, even simple gradient-based unlearning erases strongly and resists resurfacing.
  - Source: [Boglioni et al., arXiv 2607.02513](https://arxiv.org/abs/2607.02513) (abstract only)
- A Findings of EMNLP 2026 paper proposes a leak-resistant unlearning benchmark covering multi-hop reasoning paths and recovery attacks, tested on 3 models and 6 methods. Its PDF could not be parsed in this session, so the title and authors are unconfirmed — [ACL Anthology 2026.findings-emnlp.2004](https://aclanthology.org/2026.findings-emnlp.2004.pdf) (search snippet only)

### Inferences
- **TOFU-style protocol mapped onto the user's setup.** The base model is frozen and the facts are planted only through LoRA fine-tuning, so the TOFU "retain model" is simply the same LoRA recipe trained on the dataset with the planted rows removed. The KS-test forget quality, comparing (ablated LoRA) with (LoRA never shown the facts), is then a direct, cheap gold standard for "switch off components". The same reference model gives a utility comparison on the retain data.
- **Metrics to borrow:**
  - For memorized *sequences*: MUSE VerbMem (prefix → continuation ROUGE, or exact-match / Levenshtein as in Chang et al. below).
  - For *facts*: TOFU probability, Truth Ratio and paraphrased QA, plus MUSE KnowMem.
  - For utility: the same metrics on (a) the non-planted fine-tuning data, (b) facts that look like the planted ones (same templates, other entities), and (c) general text loss.
- **Retain-set design.** Thaker et al. and BLUR both warn that easy, dissimilar retain sets inflate results. The retain eval should therefore include near-neighbour facts: same relation or template, different subject. Otherwise "the rest of the model is unaffected" is a weak claim.
- **WMDP/RMU.** These matter mainly as the canonical baseline and as the source of the relearning-attack literature. WMDP-style hazardous-knowledge MCQs do not suit planted synthetic facts.

### Gaps
- Exact MUSE metric formulas and the identity of the "one algorithm" that avoids privacy leakage were not re-verified this session.
- No numbers were extracted from LACUNA. Its metric definitions and any results at the 1B scale (close to Qwen3-1.7B) would be worth reading in full.
- The EMNLP 2026 leak-resistant benchmark could not be identified (PDF unparseable).
- RWKU (arXiv 2406.10890) and the OpenUnlearning framework were not checked this session.

## 2. Localization of memorization: can removal be confined to a small set of parameters?

### Takeaway
Memorized items can be partly localized to a very small fraction of units. About 5 neurons or channels per example sufficed in image models. Dropping 0.5% of FFN neurons chosen by a pruning-style mask cuts a target sequence's accuracy by about 58% in Pythia-6.9B and GPT-2-XL. Localized neurons are, however, not specific to one sequence: related memorized items share them. Several 2025 papers argue that "where a method localizes" does not reliably predict where unlearning must act. The strongest localization results come from methods that *train* memorization into designated units, which is closest to the user's design: example-tied dropout (2023) and MemSinks (2025). Both require a per-example identity label.

### Cited Findings
**Chang, Thomason & Jia (Nov 2023, NAACL 2024), "Do localization methods actually localize memorized data in LLMs?"**
- Five methods are compared: Zero-Out, Activations, Integrated Gradients, Slimming and Hard Concrete. The last two are adapted from network pruning — [arXiv 2311.09060 HTML](https://arxiv.org/html/2311.09060)
- **INJ benchmark.** 156 ECBD-2021 definition sentences that postdate the models' training data. For each sentence, a random 1% or 0.1% of FFN weight vectors across all layers is fine-tuned to memorize it, with everything else frozen, giving one model per sentence. Methods are scored by Recall@k% of the injected neurons. Hard Concrete reaches Recall@0.5% above 80 on 3 of 4 models — [Chang et al.](https://arxiv.org/html/2311.09060)
- **DEL benchmark.** Naturally memorized pretraining sequences: 505 for each Pythia model and 105 for GPT-2-XL. The located neurons are dropped out, and the test checks that the target sequence is erased while other memorized sequences and general perplexity survive. Metrics are accuracy and Levenshtein distance — [Chang et al.](https://arxiv.org/html/2311.09060)
- **DEL results at 0.5% of neurons dropped:**
  - Hard Concrete on Pythia-6.9B reduces target accuracy by about 57.7% on average, but harms negative examples more than Integrated Gradients does.
  - Slimming on GPT-2-XL reduces the target by 57.8% and negatives by 6.4%.
  - Integrated Gradients preserves other sequences best but erases the target less.
  - Spreading the dropped neurons across layers beats concentrating them in one layer, except that the bottom layer hurts everything.
  - Source: [Chang et al.](https://arxiv.org/html/2311.09060)
- **Caveats.** Only FFN neurons and only zero-ablation were tested. Methods confuse related sequences (e.g. quotes from the same literary category, similar addresses), so localized neurons are "not uniquely tied" to one sequence — [Chang et al.](https://arxiv.org/html/2311.09060)

**Stoehr et al. (Mar 2024), "Localizing paragraph memorization"**
- Memorization is spread across layers and components, but gradients for memorized paragraphs are larger in lower layers. Memorized paragraphs can be unlearned by fine-tuning only the high-gradient weights. A low-layer attention head attends mainly to distinctive rare tokens, and perturbing a few early rare prefix tokens can corrupt the whole continuation. Memorized continuations are harder to unlearn and harder to corrupt than non-memorized ones — [arXiv 2403.19851](https://arxiv.org/abs/2403.19851) (abstract only; the GPT-Neo-125M model and the specific head index were not confirmed)

**Maini et al. (Jul 2023, ICML 2023), "Can neural network memorization be localized?"**
- Memorization of hard examples is not confined to the last layers. It sits in a small set of neurons spread across layers, often about 5 neurons or channels. Example-tied dropout routes the memorization of specific examples into a predetermined set of neurons, and dropping those neurons cut accuracy on memorized examples from 100% to 3% while also reducing the generalization gap — [arXiv 2307.09542](https://arxiv.org/abs/2307.09542) (abstract only; image classifiers, not LMs)

**Ghosal, Maini & Raghunathan (Jul 2025, ICML 2025), "Memorization Sinks"**
- Standard training entangles memorization and generalization in the same components. MemSinks assigns each repeated sequence a consistent sequence-specific set of "sink" neurons, which are active for that sequence and dropped out for other sequences. Removing the sinks at test time closes over 50% of the loss gap on repeated sequences. Validation loss without the sinks matches standard training and beats a de-duplication baseline. The experiments use SmolLM 360M and 1.7B on SlimPajama with repeated TinyStories sequences — [arXiv 2507.09937](https://arxiv.org/abs/2507.09937v2); [ICML 2025 poster](https://icml.cc/virtual/2025/poster/43838)

**Localized and causal-tracing-guided unlearning**
- Guo et al. (Oct 2024), "Mechanistic Unlearning":
  - Two kinds of localization are compared. Output-preserving localization (attribution or causal-tracing style) is weaker; localizing to the fact-lookup mechanism (the MLPs that do the "lookup table" of factual recall) works better.
  - Editing or unlearning only those components gives more robust unlearning and edits: they hold across input/output formats, are harder to relearn, have fewer side effects, and disrupt latent knowledge more.
  - Datasets: a sports-facts dataset and CounterFact, across several models.
  - Source: [arXiv 2410.12949](https://arxiv.org/abs/2410.12949) (abstract only). One search-engine summary described this paper as finding that localized edits underperform whole-model updates, which contradicts the abstract; trust the abstract.
- Torkzadehmahani et al. (Dec 2024), "Improved localized machine unlearning through the lens of memorization": DEL ("Deletion by Example Localization", not to be confused with Chang et al.'s DEL benchmark) resets a small set of critical parameters and then fine-tunes them, reporting gains over localized and full-parameter baselines — [arXiv 2412.02432](https://arxiv.org/html/2412.02432v1) (search snippet only)
- Lee et al. (May 2025, EMNLP 2025), "Does localization inform unlearning?": controlled experiments find that the set of parameters that must be modified for effective unlearning "is not strictly determined". This challenges the assumption that a parameter's locality shows it is responsible for the knowledge — [arXiv 2505.16252](https://arxiv.org/abs/2505.16252) (abstract only)
- LACUNA (2026): with known planted locations, current methods are imprecise, but correct localization combined with simple gradient unlearning gives robust erasure — [arXiv 2607.02513](https://arxiv.org/abs/2607.02513) (abstract only)
- LoRA-component unlearning (2025–2026):
  - MAAT treats LoRA rank dimensions as units that can be scored by SVD, pruned and negated — [arXiv 2605.30514](https://arxiv.org/pdf/2605.30514) (search snippet only, results not checked)
  - Cha et al. (ICLR 2025) use a Fisher-weighted low-rank initialisation of LoRA for targeted unlearning — [arXiv 2408.06621](https://arxiv.org/abs/2408.06621v3) (search snippet only)
  - No paper found here tests relearning-robustness of *ablating* individual LoRA rank-one components.

### Inferences
- **The user's design is a "train-time localization" design.** It is like MemSinks, example-tied dropout and gradient routing: each planted fact ideally ends up in a few rank-one components with gates. It is not like post-hoc localization (Chang et al., Stoehr et al.). Post-hoc work suggests that even good masks at about 0.5% of units remove only about 58% of target accuracy and leak onto related items. A clean result, where switching off components drives the planted fact to the reference-model level while near-neighbour facts are untouched, would therefore be a stronger claim than the post-hoc literature supports.
- **Two existing benchmarks to borrow.**
  - Chang et al.'s INJ and LACUNA plant information into *known* parameters and then score whether a method finds them (Recall@k%). The user can run the analogue: plant facts into designated rank-one components, then check whether their component-attribution or causal-importance gates recover those components.
  - Chang et al.'s DEL metrics (target accuracy or Levenshtein drop versus drops on other memorized sequences) are the right "specificity" table for ablation results.
- **Expect related facts to share components.** Chang et al.'s "confuses related sequences" result predicts that planted facts with shared templates or relations may share components. The retain eval should include same-template facts about other entities, and per-fact ablation results should report cross-fact damage.
- **Ablation is not the same as robust erasure.** Lee et al. (2025) means "switching off component k removes fact F" does not imply F is stored *only* in k. Residual traces elsewhere must be tested by relearning (Section 3).

### Gaps
- The exact architectures and numbers for Mechanistic Unlearning (models, relearning curves) and Torkzadehmahani et al. were not extracted.
- No LoRA-component-ablation study with relearning tests was found. This appears to be an open niche.
- No sources were found here on localizing memorized facts specifically (as opposed to verbatim sequences) via train-time routing in LMs at the 1–2B scale.

## 3. Is unlearning robust or shallow, and how should component removal be tested?

### Takeaway
Most fine-tuning-based unlearning (gradient ascent, RMU, WHP-style) is shallow. Fine-tuning on a subset of the forgotten facts recovers about 88% of pre-unlearning accuracy on held-out facts. Benign, loosely related data "jogs" memory back. 4-bit quantization restores about 83% of supposedly forgotten knowledge, against about 21% at full precision. Even idealized "behavioural" unlearning can be undone. Methods that build the separation into training (data filtering, gradient routing/SGTM, distillation into a fresh or noised student) are markedly more robust. Component removal should therefore be judged by a held-out-fact relearning test against a never-trained reference, not only by output accuracy after ablation.

### Cited Findings
- **Deeb & Roger (Oct 2024).**
  - Attack: give the attacker some of the facts that were meant to be removed, fine-tune on them, and test recovery of *other* facts from the same distribution that cannot be guessed from the given ones. This recovers 88% of pre-unlearning accuracy for current methods.
  - Conclusion: these methods make information harder to access rather than removing it.
  - They also caution that robustness tests on information added in a later fine-tuning phase may give a misleading (over-optimistic) picture compared with pretraining-learned information.
  - Source: [arXiv 2410.08827](https://arxiv.org/abs/2410.08827) (abstract only). The paper's body names this protocol "RTT" (retraining on T); that name is not re-checked here.
- **Hu et al. (Jun 2024), "Unlearning or obfuscating?"** Relearning on a small, possibly loosely related benign set reverses unlearning. Public medical articles elicit bioweapon knowledge, and general Harry Potter wiki text elicits verbatim memorized text. Their conclusion: approximate unlearning "simply suppress[es] the model outputs" — [arXiv 2406.13356](https://arxiv.org/abs/2406.13356) (abstract only)
- **Zhang et al. (Oct 2024, ICLR 2025).** For utility-constrained unlearning methods, about 21% of targeted knowledge is retained at full precision and about 83% after 4-bit quantization — [arXiv 2410.16454](https://arxiv.org/abs/2410.16454)
- **Lynch et al. (Feb 2024).** The WHP model looks unlearned on familiarity but retains extractable knowledge, Q&A performance and latent representations — [arXiv 2402.16835](https://arxiv.org/abs/2402.16835) (abstract only). Their eight tests include, per the paper, jailbreak-style prompts, in-context relearning, latent probing and side-effect checks (not re-checked this session).
- **Lee et al. (Jun 2025), "Distillation robustifies unlearning".**
  - Even idealized unlearning (training to imitate a model that never saw the data) can be reversed: behaviour changes while capability stays latent.
  - Distilling the unlearned model's outputs into a randomly initialised student transfers the behaviour without the latent capability.
  - UNDO (Unlearn, Noise, Distill-on-Outputs) distils into a noised copy and trades compute for robustness. Its strongest setting matches retraining with perfect filtering, using 60–80% of the compute and labelling only 0.01% of pretraining data. It is also shown on WMDP.
  - Source: [arXiv 2506.06278](https://arxiv.org/abs/2506.06278) (abstract only)
- **Relearning robustness of training-time localization.**
  - Gradient routing's ERA (TinyStories) measures robust unlearning by retraining on 64 forget samples and reporting the minimum forget validation loss — [Cloud et al., arXiv 2410.04332 HTML](https://arxiv.org/html/2410.04332v2)
  - In the 0.7B virology experiment, the relearning protocol fine-tunes on FineWeb-Edu for 32 steps, then on 2 WMDP-bio forget documents for 20 steps — [Cloud et al.](https://arxiv.org/html/2410.04332v2)
  - SGTM (254M, Wikipedia biology) needs about 350 fine-tuning steps (about 92M forget tokens) to recover baseline forget loss. That matches strict data filtering and is about 7× RMU (about 50 steps, about 13M tokens) — [Shilov et al., arXiv 2512.05648](https://arxiv.org/html/2512.05648v1)
- **LACUNA.** Methods that look good on outputs are "susceptible to resurfacing attacks", but correctly localized gradient unlearning resists resurfacing — [arXiv 2607.02513](https://arxiv.org/abs/2607.02513) (abstract only)
- **Calibration of ablated-model losses.** SGTM calibrates losses with a logit bias trained after training, so that ablated models do not get artificially high loss from very low token probabilities. The authors note that loss is only a proxy and that in-context attacks were not addressed — [Shilov et al.](https://arxiv.org/html/2512.05648v1)

### Inferences
- **Protocol to borrow for "did switching off components really remove the fact?":**
  1. **Held-out-fact relearning (Deeb & Roger).** Split the planted facts into T and V. After ablation, fine-tune the ablated model on T only and measure accuracy on V. Compare with the same fine-tune applied to the reference LoRA that never saw any planted facts. If the ablated model recovers V faster or higher than the reference, residual knowledge sits in the surviving components. The planted facts should be mutually unpredictable (e.g. random attribute values) so that V cannot be inferred from T.
  2. **Benign relearning (Hu et al.).** Fine-tune on loosely related non-planted text, such as the same templates with other entities, and check whether V reappears.
  3. **Relearning-curve comparison (gradient routing, SGTM).** Plot forget-set loss against the number of relearning steps for: ablated, reference (never saw), unablated, and a baseline fine-tuning unlearner (e.g. gradient ascent or RMU applied to the same LoRA). This gives a figure directly comparable to SGTM's "7× RMU" result.
  4. **Latent probes and paraphrase or format shifts** (Lynch et al., Mechanistic Unlearning). Linear probes on residual activations for the planted attribute, paraphrased questions, and in-context relearning.
  5. **Calibration.** Report calibrated loss or accuracy, as SGTM does, because zero-ablation can produce degenerate low-probability outputs that inflate forget loss.
- **Quantization attacks matter less here.** Removing components by zeroing them is an exact removal of their weight delta, not a small perturbation that rounding can undo. Inference: quantization-style recovery should matter less for component switch-off than for gradient-ascent unlearning. It is still cheap to check.
- **Deeb & Roger's fine-tuning caveat largely does not bite.** Their warning concerns facts learned during fine-tuning. In the user's setup, facts planted *only* in the LoRA (base frozen) do not exist anywhere in the base weights, so the never-saw reference LoRA is an exact counterfactual and the caveat largely does not apply. Inference: the real risk is facts the base model already partly knows, because "removal" then cannot go below the base model's prior.

### Gaps
- Exact Deeb & Roger datasets and recovery numbers per method (they report several) were not extracted. The 88% figure is the abstract's aggregate.
- No direct study was found on relearning robustness after ablating parameter-decomposition components or LoRA rank-one components.
- No quantitative LACUNA resurfacing numbers were found.

## 4. Gradient routing, ERA, SGTM and distillation on fact or knowledge data: labelling needed and results

### Takeaway
Gradient routing (Cloud et al., Oct 2024) and its successor SGTM (Anthropic-affiliated, Shilov et al., Dec 2025) localize a *domain* (forest stories, Spanish text, virology tokens, Wikipedia biology) into a few heads and MLP units, then remove it by zeroing them. Robust unlearning matches strict data filtering. Both rely on an "absorption" effect: unlabeled forget data also flows into the routed parameters. This has been shown with 99% to 60% of forget data labelled (SGTM sweeps) and with i.i.d. labelled subsets on TinyStories. No published result shows routing working with only 10% of forget data labelled, and none applies routing to individual memorized facts as opposed to a broad domain.

### Cited Findings
**Gradient routing and ERA (Cloud et al., Oct 2024)**
- The method applies data-dependent masks to gradients during backprop, so that each data point updates chosen parameters. The authors report that it "localizes capabilities even when applied to a limited, ad-hoc subset of the data" — [arXiv 2410.04332](https://arxiv.org/abs/2410.04332)
- **ERA (Expand, Route, Ablate) on TinyStories.**
  - Setup: a 28M Transformer. The forget set is stories mentioning forest, tree or woodland, about 20% of training data. Training sweeps the proportion of forget data that is labelled, with unlabelled forget stories treated as retain.
  - At <100% labelling, "ERA dominates, outperforming even the gold-standard data filtering baseline", for both unlearning and robust unlearning, but retain performance degrades in proportion to the amount of routing.
  - At 100% labelling, RMU reaches the highest pre-retraining forget loss and data filtering is the most robust to retraining. ERA's robust unlearning matches RMU after correcting for ERA's general degradation, and ERA+RMU beats either alone.
  - DEMix+ablate gives negative unlearning except at 100% labelling.
  - The full sweep values appear only in figures (not extracted).
  - Source: [Cloud et al. HTML](https://arxiv.org/html/2410.04332v2)
- **Absorption.** Routing a limited subset appears to localize the capability more broadly. This holds for an i.i.d. subset (TinyStories) and for semantically limited data, but not for DEMix. The proposed explanation is that routed units become relevant to related non-routed data, so those features are not learned elsewhere — [Cloud et al.](https://arxiv.org/html/2410.04332v2)
- **Virology, 0.7B model.**
  - Routing: only 20 tokens (e.g. "COVID", "RNA", "virus") are routed to MLP dims 0–79 in layers 0–7. This is token-level, not document-level, labelling.
  - WMDP-bio forget loss: 2.596 → 4.283 ablated (+1.687) → 2.778 after retraining (+0.182).
  - Excluding the routed tokens themselves: +1.638 ablated, +0.171 after retraining.
  - FineWeb-Edu: 2.925 → 4.864 ablated (+1.939) → 2.957 after retraining (+0.032).
  - Source: [Cloud et al.](https://arxiv.org/html/2410.04332v2)

**SGTM, Selective GradienT Masking (Shilov et al., Dec 2025)**
- **How it differs from gradient routing.** SGTM masks *parameter* gradients so that forget data updates only designated forget parameters. The original method masks activation gradients, which disrupts backprop to other parameters, and it leaves down-projections and a joint-layer set updatable by forget data; SGTM's joint set is empty. SGTM beats the original method on TinyStories at all discovery rates — [arXiv 2512.05648](https://arxiv.org/html/2512.05648v1); [Anthropic Alignment blog](https://alignment.anthropic.com/2025/selective-gradient-masking/)
- **Scale of the forget parameters.** 1 of 32 heads plus 64 of 2,048 MLP units (TinyStories, 64M model), or 1 of 32 heads plus 64 of 4,096 MLP units (Wikipedia, 254M model) — [SGTM](https://arxiv.org/html/2512.05648v1)
- **Labelling (TinyStories, English retain and Spanish forget).**
  - The main comparison has 1% of forget data undiscovered (unlabelled). Across the sweep of discovery rates, SGTM keeps higher forget loss than data filtering at every non-zero undiscovered rate.
  - Gradient-norm analysis shows minimal retain degradation with up to 20% of forget data unlabelled.
  - "Leakage", the effective forget-token exposure divided by undiscovered forget tokens, stays at 0.005–0.02 for undiscovered rates up to 40% in the 64M model, and falls as models grow. Data filtering has leakage 1 by definition.
  - Example: 20% undiscovered (144M tokens) produces the forget loss of a standard model trained on only 965k forget tokens.
  - Source: [SGTM](https://arxiv.org/html/2512.05648v1)
- **Labelling (Wikipedia).** Labels come from Wikipedia's articletopic classifier, so the noise is natural. Biology is labelled forget, 10% of the other data is labelled retain, and the remaining 90% is *unlabelled* and updates all parameters — [SGTM](https://arxiv.org/html/2512.05648v1)
- **Mechanism.** Gradient norms show that unlabelled forget data mainly updates the forget parameters, much more strongly than unlabelled retain data does. The authors read this as a "self-reinforcing" localization — [SGTM](https://arxiv.org/html/2512.05648v1)
- **Results.**
  - Better forget/retain trade-off than weak or strict filters.
  - Compute penalty: about 5–6% on general knowledge, 38% on biology-adjacent text (against 7% for the weak filter and 67% for the strict filter), and 99% on biology (against 95–96% for the filters).
  - Robust to fine-tuning at 7× RMU, matching the strict filter.
  - Source: [SGTM](https://arxiv.org/html/2512.05648v1)

**Distillation and UNDO (Lee et al., Jun 2025)**
- The robustness of retrain-with-perfect-filtering is reached while labelling only 0.01% of pretraining data, because the labels are used for the unlearning step and distillation then washes out latent capability — [arXiv 2506.06278](https://arxiv.org/abs/2506.06278) (abstract only)

**Pretraining-time filtering**
- Not fetched this session. O'Brien et al., "Deep Ignorance" (Aug 2025), arXiv 2508.06601, reports that pretraining data filtering yields tamper-resistant safeguards; treat as unverified here.

### Inferences
- **The user's 10%-labelled failure is outside the tested regime.**
  - The user has 10% of planted rows labelled, so 90% of forget data is undiscovered. SGTM's tested range for undiscovered forget data goes up to about 40%, and its main results use 1%.
  - ERA's TinyStories sweep reports wins "at <100%" but the low-label endpoint was not extracted here.
  - The only "very low label" successes are (a) token-level routing of 20 keywords (the virology experiment), where *every occurrence* of the keyword is routed, so it is effectively 100% recall on a proxy label, and (b) UNDO, where the labels feed an unlearning step followed by distillation, not routing.
- **Absorption probably does not transfer from domains to individual facts.**
  - Absorption, as both papers explain it, relies on shared features: routed units become the best place for related unlabelled data.
  - Independently planted facts, such as random entity–attribute pairs, share little beyond the template. Routing 10% of the rows may therefore localize the template or format but not the other 90% of facts, which will be learned wherever is cheapest. This is consistent with the user's observation that switching the slot off removes only 3–17%.
  - The test that would separate these explanations: does the routed slot absorb *unlabelled rows of the same fact* (the same fact repeated, 10% of its occurrences labelled), as opposed to *unlabelled different facts*?
- **Implementation differences in SGTM are worth checking against the user's routing.**
  - Parameter-gradient masking rather than activation-gradient masking.
  - No shared layers updatable by forget data.
  - SGTM reports these choices matter (it beats original gradient routing at all discovery rates).
  - Per the user's memory notes, any deviation from reference code needs flagging before implementation.
- **The user's gates might supply labels.** Because the user's components have per-token causal-importance gates, a data-free variant of "label discovery" is possible: use the gates of the labelled 10% to classify unlabelled rows, then route them in a second pass. This is an inference only; no paper found does this.

### Gaps
- The exact ERA label-proportion sweep values (e.g. results at 10% or 25% labelled) are in figures that were not extracted. Read Fig. 4 and Appendix C of [arXiv 2410.04332](https://arxiv.org/html/2410.04332v2).
- SGTM Appendix D (false-positive and false-negative analysis) was not read.
- No paper applies gradient routing or SGTM to individually memorized facts (TOFU-like) rather than to a domain.
- Whether MemSinks works with partial sequence-ID labelling is not covered by the abstract.

## 5. Partial-label settings: localizing or unlearning with only a fraction of forget examples labelled

### Takeaway
Evidence on partial labels is thin and mostly domain-level. SGTM and ERA show that training-time routing tolerates a minority of unlabelled forget data, and both beat data filtering, which leaks 100% of what it misses. One 2025 study shows that post-hoc unlearning (RMU, NPO) on WMDP is robust to masked, rewritten or watermarked forget sets. UNDO needs only 0.01% labelled data for its unlearn-then-distill pipeline. Nothing found addresses a majority-unlabelled forget set of *independent memorized facts*.

### Cited Findings
- **SGTM.** Tested with 1% up to 40% of forget data undiscovered on TinyStories, with leakage of 0.005–0.02, and natural classifier label noise on Wikipedia. Both gradient routing and SGTM beat data filtering when labels contain errors — [arXiv 2512.05648](https://arxiv.org/html/2512.05648v1)
- **ERA.** Wins over data filtering at <100% labelling. Unlabelled forget stories were treated as retain, which makes the comparison stringent — [arXiv 2410.04332](https://arxiv.org/html/2410.04332v2)
- **Wang et al. (Oct 2025, AISec'25), "LLM Unlearning on Noisy Forget Sets: Incomplete, Rewritten, and Watermarked Data."** RMU and NPO on WMDP with masked (incomplete), rewritten or watermarked forget data perform comparably to the clean forget set, provided the core semantic signal survives. The authors conclude that unlearning is guided by deep semantic cues rather than lexical patterns — [arXiv 2510.09007](https://arxiv.org/abs/2510.09007?context=cs) (search snippet only)
- **UNDO.** Labelling 0.01% of pretraining data is enough to match retraining with perfect filtering, at 60–80% of the compute — [arXiv 2506.06278](https://arxiv.org/abs/2506.06278)
- **Example-tied dropout and MemSinks.** Both assign each example or sequence its own neurons, so both need a per-example identifier for every memorized item. Neither tests partial labelling — [arXiv 2307.09542](https://arxiv.org/abs/2307.09542); [arXiv 2507.09937](https://arxiv.org/abs/2507.09937v2)

### Inferences
- **The 10%-labelled setting is novel.** For the user's goal (plant facts at x% of data, label only some rows, then remove by switching off components), no published method has demonstrated success at 10% labelled for memorized facts. This is both a gap and a possible contribution.
- **Two-stage (unlearn, then distill) as a fallback.** UNDO-style pipelines show that a small labelled set can be enough when it is used to *drive an unlearning or removal step followed by distillation*, rather than to route training. A natural comparison arm: (i) ablate the components that the 10% labelled rows activate, then (ii) distil the ablated LoRA into a re-initialised LoRA on retain data, and test whether held-out planted facts become non-recoverable.
- **Label within facts, not across them.** Wang et al. suggest that what matters is semantic coverage of the forget concept. Labelling 10% of rows *per fact* (every fact has some labelled occurrences) should localize far better than labelling 10% of *facts*. This is a cheap ablation for the user's sweep design.

### Gaps
- No work was found on semi-supervised or positive-unlabelled (PU-learning) style label completion for routing or unlearning in LMs.
- The Wang et al. setup (how much of each forget document was masked) was not extracted beyond the search snippet.
- No results were found for partial labelling with MemSinks or example-tied dropout.
