# Fact-memorization fine-tuning datasets and setups (knowledge injection) as planted ground truth

Notes for a report on using fact memorization as the planted behavior in a Qwen3-1.7B LoRA-as-parameter-decomposition setup. Current date: 2026-11-25.

Sourcing convention: items marked **[fetched]** were read in the primary source this session. Items marked **[recalled]** come from my knowledge of the paper and were not re-checked this session. The arXiv link is the primary source, but the report writer should treat [recalled] numbers as needing a check before quoting them.

## Q1. Which public datasets of fictional or synthetic facts exist, and what do they look like?

### Takeaway
There are about ten reusable public fictional-fact corpora. They range from fully templated (Allen-Zhu & Li bioS: 100k people × 6 attributes, attribute spans known exactly) through LLM-written (TOFU: 200 authors × 20 QA; Chang et al. Fictional Knowledge: 120 passages with 9 paraphrases and 15 probes each; FictionalQA: multi-style documents about fictional events) to belief-implantation corpora (SDF: around 40k synthetic documents per fact set). For a known-ground-truth plant with token-level labels, the templated bioS-style and TOFU-style sets fit best. CounterFact and zsRE are edit sets, not memorization corpora, but they come with paraphrase and neighborhood prompts.

### Cited Findings

**Physics of Language Models 3.1: knowledge storage and extraction (Allen-Zhu & Li, 2023; v3 2024)** [fetched]
- bioS has N = 100,000 synthetic individuals, each with six attributes: birth date, birth city, university, major, employer, and company city. bioR covers 100k individuals with Llama-written biographies of the same six attributes. Each person has six QA pairs. QA for half the people is used for training and the other half is held out for testing. — [arXiv 2309.14316](https://arxiv.org/html/2309.14316v3)
- Augmentations: multiM (M differently worded biographies per person), permute (shuffled sentence order), and fullname (pronouns replaced by the full name). With bioS, QA test accuracy goes from 9.7% with a single biography to 96.6% with multi5. — [arXiv 2309.14316](https://arxiv.org/html/2309.14316v3)
- Models were GPT-2 with rotary embeddings: 124M for bioS and 302M for bioR, plus a 682M model used for a negative result. Llama-style models gave similar results (v3). Training saw about 540 passes over the data (124M, bioS), 1000 passes (302M, bioR) and 1350 passes (682M), over 80k and 150k steps. — [arXiv 2309.14316](https://arxiv.org/html/2309.14316v3)
- Training was continued pretraining from scratch, not fine-tuning of a pretrained LM. Evaluation is QA generation accuracy on held-out people. — [arXiv 2309.14316](https://arxiv.org/html/2309.14316v3)
- The paper does report where knowledge is stored. Q-probing from the name tokens alone tracks QA accuracy, so with augmentation, attributes are linearly encoded on the person-name tokens. Without augmentation, the information sits on the tokens just before each attribute. — [arXiv 2309.14316](https://arxiv.org/html/2309.14316v3)
- No official dataset release is cited here. The data is generated from templates described in the paper. [recalled: community reimplementations exist; not verified this session]

**Physics of LMs 3.2: knowledge manipulation (Allen-Zhu & Li, 2023)** [recalled]
- Uses the same bios. Models that store facts still fail at inverse search (attribute to person) unless the reversed form is in training, and need chain of thought for classification and comparison. — [arXiv 2309.14402](https://arxiv.org/abs/2309.14402)

**Physics of LMs 3.3: knowledge capacity scaling laws (Allen-Zhu & Li, 2024)**
- [fetched] Abstract: LMs "can and only can store 2 bits of knowledge per parameter, even when quantized to int8". Prepending domain names to training data "significantly increases a model's knowledge capacity". GPT-2 with rotary embeddings matches or beats LLaMA/Mistral, whose GatedMLP is "less stable and harder to train". — [arXiv 2404.05405](https://arxiv.org/abs/2404.05405)
- [recalled] 2 bits per parameter needs about 1000 exposures per fact. At 100 exposures, capacity is about 1 bit per parameter, and the gated-MLP gap is largest there. Mixing in "junk" data greatly lowers useful-knowledge capacity, and domain-name prefixes recover much of it. — [arXiv 2404.05405](https://arxiv.org/abs/2404.05405)

**TOFU (Maini et al., 2024)** [recalled]
- 200 fictitious authors × 20 GPT-4-generated QA pairs, 4,000 QA in total. Attributes include birthplace, birth year, gender, parents' jobs, genre, books and awards. Forget splits are 1%, 5% and 10%. The set includes paraphrased answers and perturbed (wrong) answers. — [arXiv 2401.06121](https://arxiv.org/abs/2401.06121); [HF locuslab/TOFU](https://huggingface.co/datasets/locuslab/TOFU)
- Training: SFT on QA pairs with Llama-2-7B-chat and Phi-1.5, full fine-tuning for about 5 epochs. Evaluation uses ROUGE-L recall against the gold answer, the probability of the answer, and a "Truth Ratio" (paraphrased correct answer against perturbed answers). Its main purpose is unlearning, but the "fine-tune to memorize" stage is a ready-made fact-plant benchmark. — [arXiv 2401.06121](https://arxiv.org/abs/2401.06121)

**Reversal curse: fictitious celebrities (Berglund et al., 2023; ICLR 2024)** [recalled]
- Synthetic "<name> is <description>" facts, such as "Daphne Barrington is the director of 'A Journey Through Time'". They come in NameToDescription, DescriptionToName and Both subsets, each fact with many GPT-4 paraphrases. I recall about 30 facts per subset with about 30 paraphrases each, but that count is unverified. — [arXiv 2309.12288](https://arxiv.org/abs/2309.12288); [GitHub lukasberglund/reversal_curse](https://github.com/lukasberglund/reversal_curse)
- Full fine-tuning of GPT-3 (350M to 175B, via the API) and Llama-1-7B. Evaluation is exact match plus the log-likelihood of the correct completion against random names. Forward recall works. Reverse-direction accuracy is about 0%, and the likelihood of the correct name is no higher than for random names. — [arXiv 2309.12288](https://arxiv.org/abs/2309.12288)

**Fictional Knowledge (Chang et al., 2024, "How do LLMs acquire factual knowledge during pretraining?")** [fetched]
- 120 GPT-4-written fictional-entity descriptions, each with 9 GPT-4 paraphrases. They were modeled on ECBD/Wikipedia entries with the named entities replaced by invented ones. Each description has 15 probes (5 memorization, 5 semantic-generalization paraphrases, 5 compositional cloze), 1,800 probes in total. — [arXiv 2406.11813](https://arxiv.org/html/2406.11813v3)
- Injection: during resumed OLMo-1B and OLMo-7B pretraining (Dolma), each passage replaces part of one row of a 2048 × 2048-token batch (about 4M tokens), once every 100 steps. Three conditions: duplicate (same text 10 times), paraphrase (a different paraphrase each time, 10 times) and once. — [arXiv 2406.11813](https://arxiv.org/html/2406.11813v3)
- Metric: mean log probability of the probe's target span, from which the paper derives "effectivity" and "retainability". Forgetting follows a power law in steps. Duplication forgets faster than paraphrasing. Larger batches reduce forgetting. Effectivity improves with model size but not with more pretraining tokens. — [arXiv 2406.11813](https://arxiv.org/html/2406.11813v3)
- Code and data: [GitHub kaistAI/factual-knowledge-acquisition](https://github.com/kaistAI/factual-knowledge-acquisition/)

**FictionalQA (Kirchenbauer et al., June 2025; ICLR 2026)** [fetched]
- A GPT-4o pipeline: seed events, then "fictsheets" (structured fictional people, places and entities), then documents in five styles (news, social media, encyclopedia, corporate, blog), then QA pairs filtered to be unanswerable without the document. Splits are by event, document, style and fictsheet. — [arXiv 2506.05639](https://arxiv.org/html/2506.05639v2); [HF jwkirchenbauer/fictionalqa](https://huggingface.co/datasets/jwkirchenbauer/fictionalqa); [GitHub](https://github.com/jwkirchenbauer/fictionalqa)
- Training: Llama 3.1, Llama 3.2, Gemma 1 and Gemma 2 base checkpoints, with batches of about 5% fiction and 95% generic webtext, over roughly 0–500 steps. Verbatim memorization appears quickly. Training on the structured fictsheets gave the *least* QA and MCQ transfer even though they were memorized. Larger models transfer more. Gains also appear on held-out events, so atomic-fact learning cannot be cleanly separated from style learning. The authors report many duplicate questions. — [arXiv 2506.05639](https://arxiv.org/html/2506.05639v2)
- I did not get exact counts (events, documents, QA pairs) from the part of the paper I read.

**EntiGraph / synthetic continued pretraining (Yang et al., 2024; ICLR 2025)** [recalled]
- Source corpus: QuALITY, 265 books of about 1.3M tokens. An entity graph is used to rewrite it into about 455M synthetic tokens, and Llama 3 8B is trained on them with continued pretraining (full FT) plus a RedPajama replay mix. Closed-book QuALITY QA accuracy scales roughly log-linearly with synthetic tokens. — [arXiv 2409.07431](https://arxiv.org/abs/2409.07431); dataset on HF as `zitongyang/entigraph-quality-corpus` [recalled, unverified]
- The facts are real book content, not invented, so the ground truth is less clean than with fictional sets.

**Synthetic document finetuning (SDF; Anthropic, 2025)** [fetched]
- Pipeline: a "universe context" yields key facts. An LLM then brainstorms document types and ideas, generates documents, and critiques and revises them. A typical run uses about 40k documents per fact set, with a range of 10k–80k. — [Anthropic alignment blog, Apr 2025](https://alignment.anthropic.com/2025/modifying-beliefs-via-sdf)
- Models: Claude Haiku 3.5 (main), Llama 3.3 70B Instruct, R1-Distill-70B and GPT-4o-mini. Open models used LoRA with r = 64, α = 128 and lr = 1e-5. Full FT on Llama 3.3 70B was a Pareto improvement over LoRA. The default is 1 epoch, and more epochs raise every belief metric monotonically. — [same](https://alignment.anthropic.com/2025/modifying-beliefs-via-sdf)
- Evaluation: MCQ Knowledge, MCQ Distinguish (true vs implanted), open-ended belief judged by an LLM, Generative Distinguish (the strictest), and truth probes. — [same](https://alignment.anthropic.com/2025/modifying-beliefs-via-sdf); code at [GitHub safety-research/false-facts](https://github.com/safety-research/false-facts)
- Follow-up "Believe It or Not" (Slocum et al., Oct 2025): SDF-implanted facts generalize and survive self-scrutiny and debate, whereas prompting and mechanistic editing do not. Beliefs that contradict basic world knowledge stay fragile. — [arXiv 2510.17941](https://arxiv.org/abs/2510.17941); [blog](https://alignment.anthropic.com/2025/believe-it-or-not)
- A LessWrong post argues that SDF documents carry "synthetic markers", and that removing them makes implanted facts indistinguishable from pretraining facts to a mid-layer probe. — [LessWrong](https://www.lesswrong.com/posts/yYk5iwGqcn6XKLnWw/reducing-synthetic-markers-makes-some-sdf-false-facts)

**Gekhman et al. 2024, "Does fine-tuning LLMs on new knowledge encourage hallucinations?" (EMNLP 2024)** [recalled]
- Closed-book QA on EntityQuestions relations with PaLM 2-M and full fine-tuning. Using few-shot sampling (SliCK), each example is labeled HighlyKnown, MaybeKnown, WeaklyKnown or Unknown, and the % Unknown in the fine-tuning set is varied. Unknown examples are fitted much more slowly. Once fitted, they increase hallucination on known facts, and early stopping mitigates this. — [arXiv 2405.05904](https://arxiv.org/abs/2405.05904)

**Ovadia et al. 2023/2024, "Fine-tuning or retrieval?" (EMNLP 2024)** [recalled]
- MCQ on MMLU subsets plus a new current-events set (events after the models' cutoff). Llama-2-7B, Mistral-7B and Orca-2-7B were trained with unsupervised continued pretraining on the documents. RAG beats fine-tuning. Repeating facts in multiple paraphrases improves fine-tuning. — [arXiv 2312.05934](https://arxiv.org/abs/2312.05934)

**Other synthetic or new-fact sets (2024–2026)**
- Jiang et al. 2024, "Instruction-tuned LMs are better knowledge learners" [recalled]: the Wiki2023 set (2023 Wikipedia articles plus QA) with Llama-2 7B and 70B. "Pre-instruction-tuning" (QA before documents) improves QA on new documents. — [arXiv 2402.12847](https://arxiv.org/abs/2402.12847)
- Mecklenburg et al. 2024, "Injecting New Knowledge into LLMs via SFT" [recalled]: post-cutoff sports-event facts with GPT-3.5/4 fine-tuning. "Fact-based" generation of training QA, which covers each fact evenly, beats token-based scaling. — [arXiv 2404.00213](https://arxiv.org/abs/2404.00213)
- Zucchet et al. 2025, "How do language models learn facts? Dynamics, curricula and hallucinations": uses synthetic biographies. The search snippet says integrating new knowledge by fine-tuning "quickly corrupts its existing parametric memories". [recalled: they also report a plateau phase before recall emerges, with imbalanced data distributions shortening it] — [arXiv 2503.21676](https://arxiv.org/pdf/2503.21676v1)
- "From Style to Facts" (Mar 2025): knowledge injection through fine-tuning with matched real and fictional entities, in small runs of 20 facts each (from the search snippet). — [arXiv 2503.05919](https://arxiv.org/pdf/2503.05919)
- "The Future of Facts" (2026): 25 synthetic facts per category with 10 paraphrased training sentences per fact, across acquisition, continual-learning and update phases (from the search snippet). — [arXiv 2605.27564](https://arxiv.org/html/2605.27564v1)
- PreREQ-Tune (ICLR 2025): paraphrases each fictitious statement several times so the model learns the knowledge rather than the wording, targeting hallucination from unfamiliar fine-tuning data (from the search snippet). — [ICLR 2025 paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/98ecdc722006c2959babbdbdeb22eb75-Paper-Conference.pdf)
- "Learning a Fact Is Not Learning How to Retrieve It" (Jang et al., Oct 2026) [fetched abstract]: a model first trained on facts in 5 request forms later retrieves *new* statement-only facts through other request forms much better. The hidden "context state" just before the answer controls retrieval. This fits the paraphrase and extraction story. — [arXiv 2610.03251](https://arxiv.org/abs/2610.03251)
- Learning New Facts with QLoRA (Zheng et al., Aug 2026) [fetched]: OpenStreetMap data from 14 small cities, with every city, POI and road name replaced by synthetic identifiers. 1,938 instruction QA for training and 900 held-out *paraphrase* QA with disjoint templates. Five relations: POI category, containing city, nearest road, nearest POI and road-length bucket. Details under Q2. — [arXiv 2608.25677](https://arxiv.org/html/2608.25677v1)

**Edit sets sometimes used for fine-tuning** [recalled]
- CounterFact: about 21.9k counterfactual (subject, relation, new object) records, each with paraphrase prompts and "neighborhood" prompts for locality. Plain fine-tuning (FT, FT+L) is a standard baseline. The ROME paper also localizes facts to mid-layer MLPs. — [arXiv 2202.05262](https://arxiv.org/abs/2202.05262)

### Inferences
- For a plant with exact token-level ground truth in Qwen3-1.7B, a bioS-style templated generator works best: invented names × a few attributes, with the attribute value as a labeled span. It is cheap, the number of facts is controllable, and every "fact token" is known. TOFU is the best ready-made public alternative (QA format, already chat-style, with paraphrased and perturbed answers for log-prob contrast). It is free-form, though, so fact-token labels would need answer-span alignment.
- Chang et al.'s probe design (memorization / paraphrase / compositional, each scored by target-span log-prob) maps directly onto "does turning off component k remove the fact on verbatim, paraphrased and composed probes?"
- SDF is the closest analogue to a "behavior" plant (a belief that shows up in many contexts), but 10k–80k documents per fact set is far heavier than a fact-QA plant.

### Gaps
- I did not verify the exact TOFU hyperparameters, the reversal-curse fact and paraphrase counts, or the EntiGraph numbers and HF ID this session.
- I did not get FictionalQA's exact counts.
- I did not search for "Learning Facts at Scale with Active Reading" (Meta, 2025) or other 2025–2026 synthetic-CPT successors.

## Q2. How well does LoRA store new facts compared with full fine-tuning, and which ranks were needed?

### Takeaway
LoRA can store hundreds to low thousands of atomic facts even at very low rank. Pletenev et al. learned up to 500 unknown facts at near-100% reliability with a **rank-1** LoRA on Llama-3.1-8B MLPs. Higher rank buys more acquisition at the cost of general capability. For Qwen3-1.7B specifically, a 2026 paper got 76% (r = 8) to 86% (r = 32) held-out-paraphrase exact match after 100 epochs on about 1.9k QA. Capacity measurements put LoRA at roughly 2–3.4 bits per trainable parameter, below full fine-tuning.

### Cited Findings
- **Pletenev et al. 2025, "How much knowledge can you pack into a LoRA adapter without harming LLM?" (NAACL 2025 Findings)** [fetched]
  - Llama-3.1-8B-Instruct, with Mistral-7B as a check. New facts are DBpedia triples turned into template questions, labeled Unknown, MaybeKnown or HighlyKnown by few-shot sampling (a pool of 21,036 pairs, 14,373 of them Unknown). The number of Unknown facts trained was 1, 10, 50, 100, 500 or 3,000. — [arXiv 2502.14502](https://arxiv.org/html/2502.14502)
  - LoRA rank 1, α = 2, on down_proj, gate_proj and up_proj, with lr = 1e-3, batch size 16 and 10 epochs. — [arXiv 2502.14502](https://arxiv.org/html/2502.14502)
  - Up to 500 Unknown facts were learned at near-100% reliability. At 3,000 facts with 10 HighlyKnown per fact, reliability fell to 0.48. With 10 paraphrases per fact it stayed at 0.99. — [arXiv 2502.14502](https://arxiv.org/html/2502.14502)
  - MMLU fell from a 0.677 base to 0.615 (10 Unknown facts), 0.554 (500 Unknown + 10 HighlyKnown) and 0.441 (3,000 Unknown + 10 paraphrases). Refusals fell from about 15% to nearly zero, and some configurations collapsed onto a few overrepresented answers. — [arXiv 2502.14502](https://arxiv.org/html/2502.14502)
- **Zheng et al. 2026, "Learning New Facts with QLoRA: An Acquisition–Retention Frontier"** [fetched]
  - Main model Qwen3-4B with QLoRA at r = 8, 16, 32 and 64 on all linear layers, α = 2r, lr = 2e-4 and **100 epochs**. Full FT was the baseline. — [arXiv 2608.25677](https://arxiv.org/html/2608.25677v1)
  - At a matched acquisition target of about 75% paraphrase exact match: r8 75.7, r16 79.3, r32 85.2, r64 81.0, full FT 74.9. Only QLoRA r16 to r64 reached about 95%. Full FT never did. — [arXiv 2608.25677](https://arxiv.org/html/2608.25677v1)
  - OOD average over five benchmarks at the final checkpoint: full FT 70.7, r8 70.7, r16 63.8, r32 48.0, r64 28.1. The authors frame rank as a plasticity knob. — [arXiv 2608.25677](https://arxiv.org/html/2608.25677v1)
  - **Standard-LoRA control on Qwen3-1.7B:** held-out-paraphrase exact match of 76% at r = 8, 79% at r = 16 and 86% at r = 32. No replay data was mixed in. — [arXiv 2608.25677](https://arxiv.org/html/2608.25677v1)
- **Tan et al. 2026, "How Much Can a LoRA Adapter Memorize? Measuring Adapter Capacity in Bits"** [fetched abstract]: capacity measured on random token sequences, following Morris et al. LoRA stores about 2–3.4 bits per trainable parameter, less than full FT of the same model. MLP adapters store more than attention adapters at equal parameter count. SFT stores about 3 times more example-specific information than GRPO. — [arXiv 2607.21351](https://arxiv.org/abs/2607.21351)
- **Physics of LMs 3.1, LoRA fine-tuning** [fetched]: QA fine-tuning used LoRA on query/value matrices (r = 2–32) plus the embedding layer (r′ up to 128). LoRA generally beat full FT, but neither rescued extraction when pretraining lacked augmentation. — [arXiv 2309.14316](https://arxiv.org/html/2309.14316v3)
- **SDF** [fetched]: on Llama 3.3 70B, full FT was a Pareto improvement over LoRA r = 64. — [Anthropic](https://alignment.anthropic.com/2025/modifying-beliefs-via-sdf)
- A 2026 rate-distortion paper ran a Qwen3-8B LoRA rank sweep. Higher rank moves the high-error transition to larger fact loads (from the search snippet). — [arXiv 2609.12111](https://arxiv.org/pdf/2609.12111)
- A 2026 hypernetwork knowledge-injection scaling paper finds that LoRA rank scaling saturates, and that target-model size is the dominant axis (from the search snippet). — [arXiv 2607.19604](https://arxiv.org/html/2607.19604v1)
- Biderman et al. 2024, "LoRA Learns Less and Forgets Less" [recalled]: on code and math continued pretraining and instruction tuning with Llama-2-7B, LoRA underperforms full FT but forgets less. Full-FT weight perturbations have 10–100 times higher rank than typical LoRA configurations. This is not about atomic facts. — [arXiv 2405.09673](https://arxiv.org/abs/2405.09673)

### Inferences
- The user's LoRA is a sum of rank-one gated components. Pletenev's rank-1 MLP result suggests a few hundred facts can fit in even one rank-one direction per MLP matrix. The per-fact rank budget is therefore not the bottleneck at around 100–500 facts. The open question is whether the decomposition gives *separate* components per fact or relation, or one shared "new-fact" subspace.
- Expect 10–100 epochs on the fact data for high recall, and noticeable general-capability damage at higher rank. That damage is itself a measurable "keeps everything else" baseline.

### Gaps
- I found no paper that fits a component-level or gated-LoRA decomposition on fact injection. That falls outside this scope and other researchers are covering localization.
- I did not read the full body of 2607.21351 (ranks and model sizes) or 2607.21861 ("Data Quality over Capacity: Internalizing Documents into LoRA Adapters", title only).
- I did not fetch Thinking Machines' "LoRA Without Regret" (2025).

## Q3. Which design choices matter for clean ground truth: augmentation and paraphrases, token-level labels, and controllable mixing?

### Takeaway
Paraphrase or format diversity per fact is the single strongest lever. Without it, facts are memorized but not extractable (Allen-Zhu & Li: 0% held-out QA after QA fine-tuning; 9.7% vs 96.6% QA for single vs multi5 biographies). Templated attributes give exact token-level labels for free. Probes should separate verbatim, paraphrase, compositional and reverse queries, because each fails differently.

### Cited Findings
- Without augmentation, knowledge "may be memorized but not extractable, leading to 0% accuracy" regardless of later instruction fine-tuning. Single biographies with BIO pretraining and then QA fine-tuning gave 0% held-out QA. Mixed training gave 86.6% (bioS) and 77.7% (bioR). — [arXiv 2309.14316](https://arxiv.org/html/2309.14316v3)
- "Celebrity" effect: adding 100k heavily augmented people raised accuracy on non-augmented "minority" people from 4.4% to 86.8% (bioS) and from 10.0% to 76.3% (bioR). — [arXiv 2309.14316](https://arxiv.org/html/2309.14316v3)
- Paraphrase injection is forgotten more slowly than duplication (Chang et al.). — [arXiv 2406.11813](https://arxiv.org/html/2406.11813v3)
- Training on structured fictsheets gave the least QA transfer, despite memorization (FictionalQA). — [arXiv 2506.05639](https://arxiv.org/html/2506.05639v2)
- With 3,000 facts, 10 paraphrases per fact kept reliability at 0.99 and protected TruthfulQA better than adding known facts. — [arXiv 2502.14502](https://arxiv.org/html/2502.14502)
- Prior multi-form training teaches the retrieval route, which then transfers to new facts. — [arXiv 2610.03251](https://arxiv.org/abs/2610.03251)
- Reversal: facts learned as "A is B" do not support "B is A" (Berglund, [arXiv 2309.12288](https://arxiv.org/abs/2309.12288) [recalled]; Physics 3.2 inverse search, [arXiv 2309.14402](https://arxiv.org/abs/2309.14402) [recalled]).
- Token-level labels in existing sets: bioS attribute value spans (template-defined) ([arXiv 2309.14316](https://arxiv.org/html/2309.14316v3)), Chang et al. "target span" per probe ([arXiv 2406.11813](https://arxiv.org/html/2406.11813v3)), and TOFU and Pletenev answer strings ([arXiv 2401.06121](https://arxiv.org/abs/2401.06121) [recalled]; [arXiv 2502.14502](https://arxiv.org/html/2502.14502)).
- Prepending a source or domain tag to the data raises knowledge capacity. — [arXiv 2404.05405](https://arxiv.org/abs/2404.05405)

### Inferences
- A clean design: N invented entities × K attributes, rendered in about 5 or more templates or paraphrases with permuted order, plus held-out templates for eval. Store the value-token span indices per row. Evaluate verbatim, held-out paraphrase, QA and reverse probes by exact match and by span log-prob. Also build a "control" fact set (a separate relation or entity group) so that ablating one fact group can be checked against the others.
- Fictional entities with no overlap with real facts avoid the Gekhman confound, where Unknown versus Known facts are fitted differently.
- The location result (attributes stored on name tokens after augmentation) suggests per-token gates should fire on entity-name tokens as well as on the answer tokens. Labels should cover both.

### Gaps
- I found no paper that releases per-token "fact token" labels as a dataset field. Users derive them from templates or answer spans.

## Q4. Did any paper mix fact data into general data at a stated fraction, and what happened to recall at low fractions?

### Takeaway
Yes, though with few systematic low-fraction sweeps in the SFT setting. Stated mixes include FictionalQA (about 5% fiction per batch in webtext), Chang et al. (one passage inside a 2048-row batch every 100 steps), Allen-Zhu mixed training (QA_r = 0.8), Pletenev (0, 1 or 10 known facts per new fact) and Gekhman (varying % Unknown). Low exposure leads to fast power-law forgetting, and diluting with junk data cuts capacity.

### Cited Findings
- FictionalQA: about 5% fiction and 95% webtext per batch. Memorization comes fast, with a generalization window before overfitting. — [arXiv 2506.05639](https://arxiv.org/html/2506.05639v2)
- Chang et al.: a very low fraction (one passage among about 4M tokens, every 100 steps, 10 injections). Facts get a log-prob bump at each encounter, and then retention decays as a power law in steps. Smaller batches (128 vs 2048) forget more. — [arXiv 2406.11813](https://arxiv.org/html/2406.11813v3)
- Allen-Zhu & Li mixed training: biographies for everyone plus QA for half the people, with a QA_r = 0.8 ratio. This is the regime that makes held-out extraction work. — [arXiv 2309.14316](https://arxiv.org/html/2309.14316v3)
- Physics 3.3: data signal-to-noise (junk ratio) is a studied factor, and domain prefixes help ([arXiv 2404.05405](https://arxiv.org/abs/2404.05405), fetched abstract). [recalled] A high junk fraction cuts useful capacity many-fold at fixed exposures.
- Pletenev: mixing known facts or paraphrases with new facts. Mixing in known facts gave the larger knowledge gains, and paraphrases were more protective. — [arXiv 2502.14502](https://arxiv.org/html/2502.14502)
- Gekhman: a higher % Unknown in the fine-tuning set leads to more hallucination after the Unknown examples are fitted ([arXiv 2405.05904](https://arxiv.org/abs/2405.05904) [recalled]).
- SDF: the blog does not report mixing with pretraining or chat data. — [Anthropic](https://alignment.anthropic.com/2025/modifying-beliefs-via-sdf)
- QLoRA frontier: no replay mixing. — [arXiv 2608.25677](https://arxiv.org/html/2608.25677v1)

### Inferences
- For an x%-of-SFT-rows plant at 1–10%, the total exposure count per fact (Physics 3.3's roughly 100–1000 exposures to saturate capacity, and Pletenev's 10 epochs) probably matters more than the fraction itself. When fixing x, also fix the epochs or the passes per fact.
- Recall should be measured at the end of training, because of Chang et al.'s post-exposure forgetting.

### Gaps
- I found no paper that sweeps the fact fraction (for example 1%, 5% or 20%) inside a general *instruction* SFT mix with LoRA on a small model and reports the recall curve. This looks like an open gap, and the user's experiment would fill it.
