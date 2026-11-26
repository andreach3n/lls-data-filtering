# Fact localization mechanisms: where transformers store facts, and how reliable localization is

Scope: mechanistic findings on where factual associations live in transformer weights, plus the critiques of localization. This is written for someone planting memorized facts into Qwen3-1.7B through a LoRA that is itself a rank-one parameter decomposition (7 projections x 28 layers, per-token causal-importance gates). Unlearning benchmarks and synthetic-fact datasets are out of scope.

Source provenance: entries marked [fetched] were read or abstract-checked in this session (Nov 2026). The rest are well-known primary papers cited from their arXiv or official pages. Their headline claims are standard, but I did not re-read them in this session, so check exact numbers against the paper before quoting.

---

## Q1. ROME, causal tracing, MEMIT, and the key-value view of MLPs

### Takeaway
The classic picture (2021–2022) is that MLPs act as key-value memories. In this view, factual associations are retrieved by mid-layer MLPs at the last subject token: early-to-middle layers in GPT-2 XL and GPT-J. Later attention moves the attribute to the final position. A single fact can be inserted with one rank-one update to one MLP output matrix (ROME). Thousands of facts can be inserted by spreading updates over a band of layers (MEMIT).

### Cited Findings
- **FFN layers as key-value memories (Geva et al., 2021, EMNLP).** Each FFN's first matrix acts as "keys" that match input patterns, and the second matrix as "values" that induce distributions over output vocabulary. Lower-layer keys capture shallow, surface patterns and upper-layer keys capture more semantic ones. The output is a composition of memories refined through the residual stream. — [arXiv 2012.14913](https://arxiv.org/abs/2012.14913)
- **FFN updates promote concepts in vocabulary space (Geva et al., 2022).** An FFN output decomposes into sub-updates, each a value vector scaled by its activation. Many of these promote human-interpretable concepts when projected to the vocabulary. — [arXiv 2203.14680](https://arxiv.org/abs/2203.14680)
- **Causal tracing / ROME (Meng, Bau, Andonian, Belinkov, 2022, NeurIPS).** Corrupting the subject embeddings and restoring single hidden states finds a decisive "early site": mid-layer MLP outputs at the *last subject token* in GPT-2 XL. There is also a "late site" at the last token, which reflects attention moving information to the prediction position. Each mid-layer MLP module is described as accepting inputs that encode a subject and producing outputs that recall memorized properties of that subject. — [arXiv 2202.05262](https://arxiv.org/abs/2202.05262) (also confirmed in a search-result summary: [arXiv PDF](https://arxiv.org/pdf/2202.05262))
- **ROME's edit is rank-one.** ROME treats the MLP output projection as a linear associative memory W·k ≈ v. It inserts a new (subject → object) association with a closed-form rank-one update at one chosen mid layer: layer 17 of 48 in GPT-2 XL, and an early-middle layer in GPT-J. — [arXiv 2202.05262](https://arxiv.org/abs/2202.05262)
- **MEMIT (Meng et al., 2022, ICLR 2023).** MEMIT scales to thousands of edits (reported up to 10,000 facts in GPT-J and GPT-NeoX) by spreading each update over a contiguous band of early-to-middle MLP layers that causal tracing flags as critical. In GPT-J that band is roughly layers 3–8 of 28. — [arXiv 2210.07229](https://arxiv.org/abs/2210.07229)
- **Same site in other architectures (2026).** A 2026 knowledge-editing study of masked diffusion LMs reports "an early site at the last subject token in the early-to-mid layers, carried mainly by the MLP", matching the autoregressive picture. — [arXiv 2606.03924](https://arxiv.org/pdf/2606.03924) (search snippet only, not fetched)
- **Mamba shows an analogous localization.** Factual recall in Mamba localizes to specific mid-layer components at the last subject token, and ROME-style editing works there. — [arXiv 2404.03646](https://arxiv.org/abs/2404.03646)
- **Relations as vector arithmetic (Merullo, Eickhoff, Pavlick, 2023).** For some relations (e.g., capital_of), mid-to-late FFNs apply a roughly additive, input-independent update that maps subject to attribute, like word2vec arithmetic. — [arXiv 2305.16130](https://arxiv.org/abs/2305.16130)

### Inferences
- For the user's decomposition, the classic prediction is that planted facts land mainly in **MLP components (gate, up, down) in early-to-middle layers, active at the last subject token**. A secondary contribution would come from **attention (v, o) components in mid-to-late layers, active at the final/prediction token**.
- ROME is an existence proof that **one rank-one direction in one MLP output matrix suffices to store one fact**. That is the natural "ideal" against which to compare a fact's footprint in the user's rank-one components.

### Gaps
- The exact MEMIT layer bands for GPT-NeoX-20B were not re-verified this session.
- No source gives the causal-tracing layer profile for Qwen3 specifically. It must be measured.

---

## Q2. Does localization inform editing? Critiques of localization

### Takeaway
Where causal tracing points is a poor guide to where an edit works best. Edits succeed at many layers, and successful edits at a location are weak evidence that the knowledge is "stored" there. Ground-truth injection benchmarks show localization methods recover planted weights only partially. Expect any "fact X lives in components {i, j}" claim to need necessity and sufficiency checks, not just attribution.

### Cited Findings
- **Hase, Bansal, Kim, Ghandeharioun (2023, NeurIPS).** On GPT-J with ROME/MEMIT-style edits, causal-tracing localization is statistically uncorrelated with which layer gives the most successful edit. The choice of edit layer explains far more variance in edit success than tracing effects do. Editing can succeed at layers that tracing says do not store the fact. Reformulated editing problems ("fact forcing" and others) did not rescue the link. — [arXiv 2301.04213](https://arxiv.org/abs/2301.04213); summary corroborated: "Localization results from Causal Tracing are statistically uncorrelated with the success of an edit injecting a new fact into MLP weights" ([arXiv html](https://arxiv.org/html/2301.04213v2))
- **Wang and Veitch (2025, ICLR blogpost track; ICML 2024 workshop) [search-verified].** Editing success is weak evidence for localization. In the Inference-Time Intervention TruthfulQA setup on Alpaca-7B (16 localized heads), optimal edits at *randomly chosen* heads can be as effective as full fine-tuning. Several single heads each match full fine-tuning, so no privileged location emerges. — [arXiv 2502.11447](https://arxiv.org/abs/2502.11447); [ICLR blog](https://iclr-blogposts.github.io/2025/blog/localization)
- **Chang, Ponce, Shi, Wallace, Jia, et al. (2024, NAACL), "Do Localization Methods Actually Localize Memorized Data in LLMs? A Tale of Two Benchmarks".** The INJ benchmark *injects* new sequences into a small known subset of weights and asks whether localization methods recover those weights, which is a direct analogue of the user's setup. The DEL benchmark asks whether dropping localized neurons deletes a memorized sequence. Pruning-style methods (HardConcrete, Slimming) did best. Even so, all methods had trouble separating neurons for one memorized sequence from neurons for related sequences, i.e. localized neurons are shared. — [arXiv 2311.09060](https://arxiv.org/abs/2311.09060)
- **Stoehr et al. (2024), "Localizing Paragraph Memorization".** In GPT-Neo-125M, memorized paragraphs have identifiable but spread-out signatures. Gradients concentrate more in lower layers, and a specific low-layer attention head attends to rare tokens. Memorization is partly localizable but not to a single component. — [arXiv 2403.19851](https://arxiv.org/abs/2403.19851)
- **Editing at scale degrades (Gupta, Rao, Anumanchipalli, 2024).** Sequential ROME/MEMIT edits cause gradual and then catastrophic forgetting. This is evidence that per-fact rank-one updates interfere once there are many of them. — [arXiv 2401.07453](https://arxiv.org/abs/2401.07453)
- **Knowledge superposition (Hu et al., 2024, AAAI 2025) [search-verified].** Extending the closed-form ROME/MEMIT solution to lifelong editing yields an interference term that vanishes only if knowledge representations are orthogonal. Empirically, "knowledge superposition is universal": key overlaps have high kurtosis, zero mean, heavy tails, and scaling laws. — [arXiv 2408.07413](https://arxiv.org/abs/2408.07413)
- **Mechanism-based vs output-based localization (Guo et al., 2024) [abstract fetched].** Localizing to the "lookup-table mechanism for factual recall" gives edits and unlearning that are more robust across input/output formats, resist relearning better, and have fewer side effects. Localizations "based primarily on preserving outputs" (tracing/attribution style) do worse. Evaluated on a sports-facts dataset and CounterFact across multiple models. — [arXiv 2410.12949](https://arxiv.org/abs/2410.12949)

### Inferences
- The user's gated decomposition sits partly outside the Hase critique, because components are *learned with a necessity/sufficiency objective* rather than read off from activation patching. The Wang and Veitch result still applies, though: if a fact can be re-inserted via many locations, the decomposition's choice of components may be one of many equally good solutions. Seed-to-seed and hyperparameter stability of a fact's components is therefore a key check.
- Chang et al.'s INJ design (plant into known weights, then test recovery) is the most directly comparable published evaluation protocol.
- Guo et al. suggests a concrete prediction: components that sit in the *lookup* stage (early-mid MLP at subject tokens) should give more format-robust removal than components found by output attribution at the final token.

### Gaps
- Exact effect sizes from Hase et al. (variance explained by layer vs tracing) were not re-fetched.
- Which layers and models Guo et al. localize to (I recall early-mid MLPs in Gemma-family models) was not confirmed from the abstract.

---

## Q3. Knowledge neurons and the critique

### Takeaway
Knowledge-neuron methods find small sets of FFN neurons whose activation modulates a fact. The 2024 critique shows the same method "localizes" syntax just as well and edits are unreliable. The neurons capture token-expression patterns, not knowledge in a meaningful sense. 2026 work revives a narrower version: sparse, entity-selective early-layer MLP "entity cells" with causal effects in some models.

### Cited Findings
- **Knowledge neurons (Dai et al., 2022, ACL).** Integrated-gradients attribution over FFN intermediate neurons in BERT finds a few "knowledge neurons" per relational fact, mostly in top layers. Suppressing or amplifying them changes the probability of the correct fact, and they can be used for limited editing. — [arXiv 2104.08696](https://arxiv.org/abs/2104.08696)
- **Niu, Liu, Bhattacharjee, Penn (2024, ICLR), "What does the Knowledge Neuron thesis have to do with knowledge?"** The KN thesis is an oversimplification. The same localization and editing methods find "neurons" for syntactic phenomena (e.g., determiner–noun agreement) just as readily as for facts. Edits through them are unreliable and generalize poorly, and the identified patterns are better described as shallow "token expression patterns". The authors argue for studying circuits across layers and attention rather than individual MLP neurons. — [arXiv 2405.02421](https://arxiv.org/abs/2405.02421)
- **Entity cells (Yona, Barzilay, Karasik, Geva; Apr 2026, v2 May 2026) [fetched].** Sparse, entity-selective MLP neurons ("entity cells") were found across seven models, clustering "predominantly in early layers". In Qwen2.5-7B base, suppressing one cell "selectively erases recall for its matched entity while leaving others intact". Activating a single cell recovers correct knowledge for most entities, even when the entity is absent from context. The cells are robust to aliases, misspellings and other languages, and stable through instruction tuning. Caveat: causal signals "vary across model families", and the effect holds for most but not all entities. — [arXiv 2604.01404](https://arxiv.org/abs/2604.01404)
- **Transcoder features are often neurons (2026).** A 2026 comparison argues many transcoder features are "neurons in disguise", which bears on whether sparse-feature localization differs from neuron localization. — [ResearchGate](https://www.researchgate.net/publication/416345974_Comparing_Sparsified_MLPs_and_Transcoders_Many_Transcoder_Features_Are_Neurons_in_Disguise) (search snippet only)

### Inferences
- The entity-cell result is the closest prior to "one rank-one component per entity". It suggests the user may find **entity-identity components in early-layer MLPs** (subject recognition), separate from **attribute components** (subject → object mapping). Fact removal might work by ablating either kind, with different side-effect profiles.
- The Niu critique means the user should test that "fact components" are not just token-pattern detectors for the subject's surface form. Paraphrase, alias, and reverse-direction prompts are the natural controls.

### Gaps
- No count of neurons per entity was available from the entity-cells abstract.

---

## Q4. Fact-recall mechanisms (enrichment, extraction, lookup tables, additive mechanisms, LREs)

### Takeaway
Recall is a multi-stage, partly redundant process. Early-to-mid MLPs "enrich" the last subject token with many attributes, functioning like a distributed multi-token lookup table in superposition. Mid-to-late attention heads then extract the relation-relevant attribute to the final token, and several independent mechanisms add up. Neuron-level reverse-engineering of the lookup has failed so far. The lookup stage is described as superposed, hash-like and not interpretable.

### Cited Findings
- **Dissecting recall (Geva, Bastings, Filippova, Globerson, 2023, EMNLP).** Attention knockout on GPT-2 and GPT-J reveals three steps. (1) *Subject enrichment*: early-layer MLPs (with attention) build a rich subject representation at the last subject token, which decodes to many related attributes. (2) *Relation propagation* to the last position. (3) *Attribute extraction*: upper-layer attention heads pull the attribute from the subject representation, and these heads often encode subject–attribute mappings in their parameters. — [arXiv 2304.14767](https://arxiv.org/abs/2304.14767)
- **Fact Finding (Nanda, Rajamanoharan, Kramár, Shah; Dec 2023, Alignment Forum, 5-post sequence).** On Pythia-2.8B mapping ~1,500 athlete names to sport: MLP layers ~2–6 do the lookup on multi-token names, producing a "multi-token embedding" of the athlete at the last name token, and attention heads then move the sport. The authors hypothesised mechanisms (e.g., detokenization plus a "hash and lookup") but could not reverse-engineer them at the neuron level. They conclude the lookup is distributed and in superposition across many neurons, and recommend treating early MLPs as a black-box lookup table and interpreting what goes in and out instead. — [Alignment Forum: Fact Finding overview](https://www.alignmentforum.org/posts/iGuwZTHWb6DFY3sKB/fact-finding-attempting-to-reverse-engineer-factual-recall)
- **Summing up the facts (Chughtai, Cooney, Nanda, 2024).** Factual recall is implemented by several independent, additive mechanisms that sum constructively at the final token: subject heads, relation heads, mixed heads and MLPs. No single circuit carries the whole fact (the "additive motif"). — [arXiv 2402.07321](https://arxiv.org/abs/2402.07321)
- **Linear relational embeddings (Hernandez, Sharma, Haklay, Meng, Wattenberg, Andreas, Belinkov, Bau, 2023, ICLR 2024).** For a subset of relations, decoding attribute from subject representation is well approximated by one affine map (first-order Taylor of the LM's computation from mid-layer subject state). Many other relations are not linearly decodable this way, showing heterogeneity across relations. — [arXiv 2308.09124](https://arxiv.org/abs/2308.09124)
- **Interpreting key mechanisms of factual recall (2024).** A further study traces recall into task-specific attention heads and MLP components that extract and amplify the argument and attribute. — [arXiv 2403.19521](https://arxiv.org/abs/2403.19521) (search-result title only; content not fetched)
- **Knowledge circuits (Yao et al., 2024, NeurIPS).** Facts are recalled through circuits spanning attention heads and MLPs across layers rather than a single site. — [arXiv 2405.17969](https://arxiv.org/abs/2405.17969)

### Inferences
- The user should expect **at least two populations of fact-relevant components**:
  1. Early-layer MLP components firing on subject tokens (lookup/enrichment). These are likely *shared across many facts* in superposition (Fact Finding).
  2. Mid-to-late attention v/o (and possibly q/k) components at the final token that extract the attribute.
- Chughtai et al.'s additive picture predicts **redundancy**: ablating one component may only partly remove a fact. Gate-off results should be read as graded rather than all-or-nothing.

### Gaps
- No published work maps these stages onto Qwen3 specifically.

---

## Q5. Where facts learned during fine-tuning are stored (vs pretrained facts); LoRA and fine-tune diffs

### Takeaway
I found **no study that runs causal tracing on the same model before and after injecting facts** to compare storage sites directly. A 2026 study finds injected knowledge is causally usable from early and late layers but not routed into mid-layer reasoning circuits. Broader fine-tuning work suggests fine-tuning mostly reweights or reuses existing mechanisms. One 2026 LoRA study (on another task) finds LoRA's effect is MLP-dominated and concentrated in early layers. Storage sites for fine-tuned facts are therefore an open empirical question, which the user's experiment would directly address.

### Cited Findings
- **Knowing–Using Gap (Dai, Rao, Wang, Wang, Liu, Xiong; Jul 2026, v2 Sep 2026) [fetched].** This study fine-tunes Qwen2.5-1.5B/3B/7B and Llama-3.2-1B/3B/3.1-8B on unseen knowledge. It uses "self-patching" (patching the model's own residual stream between layer pairs) and finds "the most effective source layers form an early and a late cluster, both patching into middle target layers". Its conclusion is that injected knowledge "is stored in both early and late layers but fails to align with the mid-layer computation". Patching into late target layers does nothing. Oracle self-patching raises multi-hop chaining on Qwen2.5-1.5B from 0.078 to 0.440. Their fix, LRSD, distils the 0.75L representation into the 0.5L representation (21→14 for 28-layer models) and roughly doubles chaining accuracy (0.085 → 0.166 on Qwen2.5-1.5B). Caveat: the analysis is at whole-block residual granularity, with no MLP-vs-attention split. — [arXiv 2607.08393](https://arxiv.org/html/2607.08393v2)
- **Layer-wise analysis of SFT (Zhao et al., Apr 2026, ACL 2026) [abstract fetched].** Across 1B–32B models, "middle layers (20%–80%) are stable, whereas final layers exhibit high sensitivity" during SFT. The paper is about alignment and instruction tuning, not fact injection, so it says little about where new facts are stored. A search summary claimed it says "new knowledge … primarily resides in the middle layers", but the abstract does not say that. — [arXiv 2604.11838](https://arxiv.org/abs/2604.11838)
- **Fine-tuning reuses existing mechanisms.** Fine-tuning on entity tracking enhances the *existing* circuit rather than building a new one (Prakash et al., 2024) — [arXiv 2402.14811](https://arxiv.org/abs/2402.14811). On procedurally defined tasks, fine-tuning learns a minimal "wrapper" on top of existing capabilities (Jain et al., 2023) — [arXiv 2311.12786](https://arxiv.org/abs/2311.12786).
- **Knowledge circuits during continual pretraining (Ou et al., 2025).** Acquiring new knowledge depends on its relevance to existing knowledge, and knowledge circuits evolve from a formation phase to an optimization phase, with a deep-to-shallow pattern reported. — [arXiv 2502.11196](https://arxiv.org/abs/2502.11196) (not re-fetched this session; verify details)
- **LoRA mechanistic effect (2026) [search snippet only].** On Qwen2.5-1.5B, path patching shows "LoRA's adaptation effect is highly non-uniform, with MLP components dominating over attention and contributions concentrated in early layers". Adaptive-LoRA on ~30% of layers matches or beats full-layer LoRA. The task was not confirmed to be fact learning. — [arXiv 2610.34756](https://arxiv.org/abs/2610.34756)
- **LoRA as knowledge memory (Back et al., Mar 2026, ICML 2026) [abstract fetched].** Billed as the first systematic study of LoRA-based memory (storage capacity, internalization, multi-module scaling). The abstract gave no capacity or per-module numbers. — [arXiv 2603.01097](https://arxiv.org/abs/2603.01097)
- **Fine-tune weight-diff structure.** Full fine-tuning learns weight perturbations of much higher rank than typical LoRA configurations (Biderman et al., 2024, "LoRA Learns Less and Forgets Less") — [arXiv 2405.09673](https://arxiv.org/abs/2405.09673). LoRA updates introduce "intruder dimensions", new high-ranking singular vectors dissimilar to the pretrained ones, which full fine-tuning does not (Shuttleworth et al., 2024) — [arXiv 2410.21228](https://arxiv.org/abs/2410.21228).
- **Injected facts traced via training data, not layers.** FTRACE (Akyürek et al., 2022) injects novel facts by fine-tuning and evaluates training-data attribution. It does not localize them in layers. — [arXiv 2205.11482](https://arxiv.org/abs/2205.11482)
- **Storage vs extraction depends on data diversity (Allen-Zhu and Li, Physics of LMs 3.1).** Facts are stored in an extractable, attribute-keyed form only when training data contains diverse paraphrases. Otherwise the model memorizes them tied to the whole training sequence. — [arXiv 2309.14316](https://arxiv.org/abs/2309.14316)

### Inferences
- How much paraphrase diversity the user trains on likely changes *where* and *how* facts are stored (Allen-Zhu and Li). Low-diversity memorization may live in sequence-specific components (attention to earlier tokens, late layers) rather than in subject-keyed lookup MLPs.
- The "early and late cluster" result (2607.08393) and the classic "early-mid MLP" site are not contradictory, but they predict different things for the user's components. Fine-tuned facts may appear in **late-layer components** more than pretrained facts would, and planted facts may fail to compose in multi-hop prompts.
- Intruder dimensions suggest a LoRA-stored fact may occupy directions nearly orthogonal to the pretrained weight's top singular vectors. That is consistent with a clean rank-one component, but also with a fact stored in a "foreign" subspace the base model's circuits don't read from.

### Gaps
- No published before/after causal-tracing comparison of the same facts injected vs pretrained, at module (MLP vs attention) resolution. This was searched for and not found.
- No published work decomposes a fact-injection LoRA into rank-one components and checks per-fact localization. The user's experiment appears novel.
- Per-module capacity numbers for LoRA memory (2603.01097) were not available from the abstract.

---

## Q6. Localizing facts with SAEs, transcoders, attribution graphs, and weight-based decompositions

### Takeaway
Attribution graphs show interpretable multi-hop fact paths (e.g., Dallas → Texas → Austin) and "known entity" gating circuits. 2026 work uses low-rank factual edits to localize facts in attribution graphs, but also shows SAE-based graphs are fragile across seeds. Weight-based decompositions (APD → SPD → VPD) have so far been shown on toy models and a 67M-parameter 4-layer LM. They found sparse rank-one subcomponents and did one single-component edit (not of a fact). No published parameter decomposition targets factual recall.

### Cited Findings
- **Biology of an LLM (Lindsey, Gurnee, Ameisen, et al., Mar 2025, Transformer Circuits).** Cross-layer-transcoder attribution graphs on Claude 3.5 Haiku. For "the capital of the state containing Dallas", the model internally activates "Texas" features, which then drive "Austin", an explicit two-hop path, alongside a direct shortcut path. A separate entity-recognition circuit has "known entity" features suppress a default "can't answer" pathway, and misfiring of this circuit produces hallucinations. — [Transformer Circuits: Biology](https://transformer-circuits.pub/2025/attribution-graphs/biology.html); methods: [Circuit Tracing](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)
- **SAE entity-knowledge features (Ferrando, Obeso, Rajamanoharan, Nanda, 2024, ICLR 2025).** SAEs on Gemma 2 find directions encoding whether the model "knows" an entity. These causally steer refusal and hallucination behaviour and affect attention to the entity's attributes. — [arXiv 2411.14257](https://arxiv.org/abs/2411.14257)
- **Transcoders (Dunefsky, Chlenski, Nanda, 2024).** Transcoders replace MLPs with sparse, input-independent feature circuits, enabling weight-based circuit analysis through MLPs, which is relevant to the lookup stage. — [arXiv 2406.11944](https://arxiv.org/abs/2406.11944)
- **Circuit-Diff (2026) [search snippet only].** This method localizes knowledge in attribution graphs by applying a low-rank factual edit (e.g., Paris → Rome) and treating the features whose graph role shifts as tied to that fact. It notes unlabeled nodes are hard to interpret by hand. — [arXiv 2609.23892](https://arxiv.org/pdf/2609.23892)
- **SAE attribution graphs are fragile (Nov 2026) [search snippet only].** This paper reports large errors in SAE attribution graphs and noticeable changes across SAE training seeds. — [arXiv 2611.00606](https://arxiv.org/html/2611.00606v1)
- **APD (Braun, Bushnaq, Heimersheim, Mendel, Sharkey; Jan 2025).** Attribution-based Parameter Decomposition splits weights into components that are faithful (they sum to the weights), minimal (few active per input) and simple. Demonstrated on toy models of superposition and compressed computation. — [arXiv 2501.14926](https://arxiv.org/abs/2501.14926)
- **SPD (Bushnaq, Braun, Sharkey; Jun 2025).** Stochastic Parameter Decomposition makes APD scalable using rank-one subcomponents with learned causal-importance and stochastic masking. Goodfire's announcement lists "factual recall" among capabilities it hopes to locate in parameter components, as an open question rather than a result. — [arXiv 2506.20790](https://arxiv.org/abs/2506.20790); [Goodfire post](https://www.goodfire.com/research/stochastic-param-decomp)
- **VPD (Goodfire, "Interpreting Language Model Parameters", 5 May 2026) [fetched].**
  - Setup: adversarial-ablation successor to SPD, applied to a 4-layer, 67M-parameter Pile model, decomposing its 24 weight matrices into 38,912 rank-1 subcomponents, of which ~9,972 are alive.
  - Sparsity: "each datapoint uses 205 subcomponents per sequence position, representing 2.1% of all alive subcomponents".
  - Alive / mean L0 per layer: L0 3,709 / 44.6; L1 848 / 18.9; L2 1,943 / 49.5; L3 3,472 / 92.0.
  - Behaviour-level circuits: gendered-pronoun prediction has 150–160 subcomponents in the attribution graph, pruned by causal importance to 6 subcomponents in two pathways (flagged as not adversarially robust). Bracket closing goes from 158 to 14.
  - Editing: hand-editing a *single* rank-1 subcomponent (L2.MLP.down:2359) makes all emoticons predict a surprised face. LoRAs trained on 947 examples beat it on KL side-effects.
  - Gaps: no factual-recall or memorization analysis. — [Goodfire: Interpreting LM Parameters](https://www.goodfire.com/research/interpreting-lm-parameters)

### Inferences
- VPD's numbers give a rough prior for the user's components. A narrow behaviour reduces to a handful (6–14) of rank-one subcomponents under causal-importance pruning, from graphs of about 150. By analogy, a planted fact might reduce to **a few to a few dozen components** across layers and modules, but the pruned set may not be robust to adversarial ablation.
- The VPD emoticon edit is the parameter-decomposition analogue of ROME: one rank-one MLP-down component carrying a narrow token-level mapping. A planted fact concentrated in one or two MLP-down or up components would be the "ROME-like" outcome.
- Attribution-graph and SAE baselines describe *activations*, not weights. Comparing them to weight components needs a bridge, e.g. checking that a component's gate fires on the tokens where the corresponding transcoder features are active.

### Gaps
- No published SPD or VPD decomposition of a model that has had facts planted. The Goodfire Qwen3-1.7B decompositions noted in the user's memory are not public results as far as I found.
- Circuit-Diff and the attribution-graph fragility paper were seen only as search snippets.

---

## Q7. Implications: is one fact a few rank-one directions or spread out, and how does it scale with the number of facts?

### Takeaway
There are two framings. (a) Editing-style insertion: one rank-one update per fact suffices (ROME). (b) Natural learning: the lookup is superposed across many neurons and layers (Fact Finding, knowledge superposition), and several mechanisms add up redundantly (Chughtai). Capacity results say storage scales roughly linearly with parameters, about 2 bits per parameter. So with many facts, expect **shared components carrying many facts each in superposition**, not one dedicated component per fact. A per-fact footprint should look like a sparse *combination* of shared directions, plus perhaps a few fact-specific ones when facts are few.

### Cited Findings
- **One rank-one update per fact is sufficient for insertion.** ROME inserts each fact with a closed-form rank-one update to one MLP — [arXiv 2202.05262](https://arxiv.org/abs/2202.05262). MEMIT inserts thousands of facts by spreading updates over several layers — [arXiv 2210.07229](https://arxiv.org/abs/2210.07229).
- **Linear capacity scaling (Nichani, Lee, Bietti, 2024, ICLR 2025) [search-verified].** Storage capacity of linear and MLP associative memories "scale[s] linearly with parameter count". A one-layer transformer reaches 100% on synthetic factual recall when either self-attention or MLP parameters scale (up to log factors) linearly with the number of facts. In other words, both attention and MLPs can store facts and trade off against each other. — [arXiv 2412.06538](https://arxiv.org/abs/2412.06538)
- **Sharp capacity of linear associative memory (2026) [search snippet only].** A 2026 follow-up (Giorlandino, Goldt, Maillard) characterizes the linear case sharply, storing up to about p·log p / d² = 1/2 associations in a decoupled model. — [arXiv 2605.10795](https://arxiv.org/pdf/2605.10795); NeurIPS 2026 poster: [neurips.cc](https://neurips.cc/virtual/2026/poster/152324)
- **~2 bits of knowledge per parameter (Allen-Zhu and Li, Physics of LMs 3.3, 2024).** Across architectures and sizes, LMs store about 2 bits of factual knowledge per parameter when trained enough on each fact (still holds under int8). — [arXiv 2404.05405](https://arxiv.org/abs/2404.05405)
- **Superposition causes interference as facts accumulate.** Knowledge keys are non-orthogonal and superposition is universal, so editing one fact perturbs others — [arXiv 2408.07413](https://arxiv.org/abs/2408.07413). Many sequential edits lead to catastrophic forgetting — [arXiv 2401.07453](https://arxiv.org/abs/2401.07453).
- **The natural lookup is distributed and uninterpretable at neuron level.** — [Fact Finding](https://www.alignmentforum.org/posts/iGuwZTHWb6DFY3sKB/fact-finding-attempting-to-reverse-engineer-factual-recall)
- **Redundant additive mechanisms.** — [arXiv 2402.07321](https://arxiv.org/abs/2402.07321)
- **Counterpoint: sparse entity-specific units exist.** Single early-layer neurons can be necessary and sufficient for recall of a specific entity in some models. — [arXiv 2604.01404](https://arxiv.org/abs/2604.01404)

### Inferences
- **Small number of planted facts (tens).** A minimal decomposition might plausibly give each fact one or a few dedicated rank-one components, ROME-like and most likely in early-to-mid MLP down or up. Alternatively, an entity-identity component in an early MLP plus an attribute-writing component. The sparsity pressure of causal-importance gating favours this when facts are few and capacity is abundant.
- **Many planted facts (hundreds to thousands).** Capacity and superposition results predict components will be **shared across facts**. Each component is gated on for many subjects, and a fact is identified by a *combination* of active components. The effective number of components needed should grow roughly linearly with total fact bits, not with fact count times a constant per-fact rank. Per-token gates help here: a rank-one component in a lookup MLP can serve many facts if its gate fires only on the relevant subjects' tokens.
- **Attention vs MLP is not forced.** Nichani et al. show either can store facts, so if MLP capacity in the LoRA is constrained, facts can legitimately land in attention v/o components.
- **Fine-tuned-fact specifics.** 2607.08393 suggests some planted-fact signal may sit in late layers. Low paraphrase diversity (Allen-Zhu and Li 3.1) may yield sequence-keyed rather than subject-keyed storage.

#### Baselines to compare against (from the cited work)
1. Causal tracing / activation patching at subject-last-token vs final token, per layer and module (ROME, [2202.05262](https://arxiv.org/abs/2202.05262)). Attention knockout (Geva 2023, [2304.14767](https://arxiv.org/abs/2304.14767)).
2. ROME (single-layer rank-one) and MEMIT (multi-layer) insertion of the same facts. These test whether "one rank-one direction per fact" is achievable at the layers the decomposition picks ([2202.05262](https://arxiv.org/abs/2202.05262), [2210.07229](https://arxiv.org/abs/2210.07229)). Compare edit success across layers per Hase et al. ([2301.04213](https://arxiv.org/abs/2301.04213)).
3. Knowledge-neuron integrated-gradients attribution ([2104.08696](https://arxiv.org/abs/2104.08696)), with Niu et al.'s syntactic and paraphrase controls ([2405.02421](https://arxiv.org/abs/2405.02421)).
4. Entity-cell localization by activation consistency across prompts ([2604.01404](https://arxiv.org/abs/2604.01404)).
5. Pruning-mask localization (HardConcrete, Slimming) and the INJ-style "inject then recover" protocol ([2311.09060](https://arxiv.org/abs/2311.09060)).
6. Mechanism-based "fact lookup" localization vs output-tracing localization, judged by robustness of removal across formats and relearning ([2410.12949](https://arxiv.org/abs/2410.12949)).
7. Activation-space dictionary methods: SAEs, transcoders, and attribution graphs ([2406.11944](https://arxiv.org/abs/2406.11944), [Biology](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)), with seed-stability caveats ([2611.00606](https://arxiv.org/html/2611.00606v1)).
8. Weight-diff analysis of a plain LoRA or full fine-tune on the same facts, e.g. SVD and intruder dimensions ([2410.21228](https://arxiv.org/abs/2410.21228)).

### Gaps
- No empirical study measures "rank-one components per fact" as a function of the number of facts in a real LM. The scaling claims above are inferences from capacity theory and editing results, not direct measurements.
- No work tests whether a parameter decomposition recovers ROME-inserted facts as single components. That would be a cheap sanity-check experiment.
