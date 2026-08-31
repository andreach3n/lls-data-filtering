# INDEX for default_600k.md

`default_600k.md` — 41,361 lines, ~544,136 tokens (~2.8x the context window).
**Never read this file whole, and never Read it without an explicit line range.**
Find the topic here, then read only that range, or grep for a phrase.

## Regions

| Lines | ~Tokens | Contents |
|---|---|---|
| 1–84 | 1k | Table of contents (reproduced below) |
| 85–9,100 | 194k | **Part I & II.** Neel's research-process posts (explore/understand/distill, key mindsets, research taste), Steinhardt on research as stochastic decision process, ML paper-writing advice, mech interp glossary, annotated favourite-papers list, Ferrando primer on transformer internals, Open Problems in Mech Interp |
| 9,100–16,900 | 90k | **Part III(a).** TransformerLens + nnsight: READMEs, docs, source, demo notebooks (activation patching, grokking, Othello-GPT, patchscopes, head detector) |
| 16,900–41,360 | 271k | **Part III(b).** ARENA tutorials: [1.1] transformers from scratch, [1.2] induction heads, [1.3.1] superposition & SAEs, [1.4.1] IOI, [1.4.2] function vectors & steering |

Part III is dumped notebook source. It has no reliable heading structure, so grep it by symbol or phrase (e.g. `grep -n 'HookedTransformer.from_pretrained'`) rather than by line map.

## Table of contents (verbatim from the file)

### **Part I: Research Philosophy and Strategy**
 
- **Neel Nanda's Research Process Framework**

	- 1.1. How I Think About My Research Process: Explore, Understand, Distill

		- Stage 1: Ideation - Choosing a Problem
		- Stage 2: Exploration - Gaining Surface Area
		- Stage 3: Understanding - Testing Hypotheses
		- Stage 4: Distillation - Compressing, Refining, and Communicating
	- 1.2. Key Mindsets for Research: Truth-Seeking, Prioritization, Moving Fast

		- Truth-Seeking and Resisting Bias
		- Prioritization and Goal-Setting
		- Moving Fast and Acting Under Uncertainty
	- 1.3. Understanding and Cultivating Research Taste

		- Decomposing Taste: Intuition, Conceptual Frameworks, and Strategic Picture
		- Methods for Cultivating Taste: Leveraging Mentors, Papers, and Reflection
- **Jacob Steinhardt on Research as a Stochastic Decision Process**

	- 2.1. Prioritizing by Information Rate vs. Naive Strategies
	- 2.2. De-risking, Front-loading Information, and Practical Patterns
	- 2.3. Research as a Branching Search Tree
- **Advice on Writing Machine Learning Papers**

	- 3.1. The Essence of a Paper: Crafting a Cohesive Narrative
	- 3.2. Providing Rigorous Supporting Evidence and Avoiding Pitfalls
	- 3.3. Iterative Writing Process: Compress then Expand
	- 3.4. Detailed Paper Structure: Abstract, Introduction, Main Body, Figures, etc.

---

### **Part II: Foundations of Mechanistic Interpretability**

- **Core Concepts and Terminology**

	- 1.1. A Comprehensive Mechanistic Interpretability Explainer & Glossary (Neel Nanda)

		- General Concepts: Features, Circuits, Decomposability
		- Representations: Linear Representation Hypothesis, Privileged Basis
		- Superposition: Bottleneck vs. Neuron Superposition, Polysemanticity
		- Transformer-Specific Concepts and Techniques
- **Surveys of the Field: Key Papers and Open Problems**

	- 2.1. An Extremely Opinionated Annotated List of My Favourite Mechanistic Interpretability Papers (Neel Nanda)

		- Foundational Work (e.g., A Mathematical Framework for Transformer Circuits)
		- Superposition & Sparse Autoencoders (SAEs)
		- Activation Patching & Causal Interventions
		- Narrow Circuits (e.g., Indirect Object Identification)
	- 2.2. A Primer on the Inner Workings of Transformer-Based Language Models (Javier Ferrando et al.)

		- Transformer Components and the Residual Stream Perspective
		- Techniques for Behavior Localization (Attribution, Causal Interventions)
		- Techniques for Information Decoding (Probing, SAEs)
		- Discovered Inner Behaviors and Known Circuits
	- 2.3. Open Problems in Mechanistic Interpretability (Lee Sharkey et al.)

		- Challenges in Methods: Reverse Engineering, Decomposition, and Validation
		- Challenges in Applications: Monitoring, Control, Prediction, and Microscope AI
		- Socio-technical and Governance Challenges

---

### **Part III: Tooling & Hands-On Tutorials**

- **TransformerLens: A Library for Mechanistic Interpretability**

	- 1.1. Introduction and Getting Started
	- 1.2. Key Features: Hooks, Activation Caching, Model Loading
	- 1.3. ARENA Tutorials with TransformerLens:

		- 1.3.1. Building a Transformer from Scratch
		- 1.3.2. Introduction to Mech Interp & Finding Induction Heads
		- 1.3.3. Indirect Object Identification (IOI) Circuit Analysis
		- 1.3.4. Toy Models of Superposition & Sparse Autoencoders (SAEs)
- **NNsight: A Library for Transparent Science on Black-Box AI**

	- 2.1. Introduction and Getting Started with Remote Execution (NDIF)
	- 2.2. Core Concepts: The Intervention Graph
	- 2.3. Key Features: Getting/Setting Activations, Gradients, Cross-Prompt Interventions, Multi-Token Generation
	- 2.4. ARENA Tutorial with NNsight: Function Vectors & Model Steering


## Line map — Part I & II only

Read with an explicit range, e.g. `Read default_600k.md offset=647 limit=200`.

- L85: How I Think About My Research Process: Explore, Understand, Distill
  - L91: Introduction
  - L107: The key stages
- L189: My Research Process: Key Mindsets - Truth-Seeking, Prioritisation, Moving Fast
  - L204: Truth Seeking
  - L241: Prioritisation
  - L286: Moving Fast
- L330: My Research Process: Understanding and Cultivating Research Taste
- L334: Introduction
- L348: What is Taste?
  - L366: Decomposing Research Taste
- L399: Cultivating Research Taste
- L427: Conclusion: Patience and Process
- L437: Research as a Stochastic Decision Process
- L647: Highly Opinionated Advice on How to Write ML Papers
  - L675: Introduction
  - L685: The Essence of a Paper
  - L876: Analysing My Grokking Work
  - L902: The Writing Process: Compress then Iteratively Expand
  - L958: The Anatomy of a Paper
  - L1117: Common Pitfalls and How to Avoid Them
  - L1141: Tacit Knowledge and Beyond
  - L1157: Conclusion