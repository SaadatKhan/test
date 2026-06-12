# Controllable Persona-Vector Simulator: Project Plan

**Status:** Pre-implementation planning document.
**Intended reader:** A capable research agent (you) that will execute, or assist a human researcher in executing, the experiments described below.
**Operating instruction for the agent:** This is a plan, not a script. Before running any rung, restate to the human what you understand the rung to be testing and confirm decisions marked **[DECIDE]** before proceeding. Do not silently change the ladder, the metric set, or the data splits. If a result contradicts a stated hypothesis, surface it and stop for review rather than reframing it.

---

## 1. One-paragraph problem statement

Prior work on persona-conditioned dialogue treats personas as fixed, discrete attribute sets and contributes either richer attribute taxonomies (empathy, Big Five) or post-training methods (SFT, DPO, RL) that force a model to obey those attributes. Both strands inherit a representational assumption we argue is mistaken: persona is not a label assignment but a position on a continuous manifold. Anger admits a range of intensities and expressive modes; formality and verbosity vary smoothly between speakers; expertise is not a binary toggle. Yet not every persona feature is continuous — occupation, native language, family structure, and demographic anchors are genuinely discrete and stable across a conversation. We propose a **hybrid persona representation**: discrete anchors for stable identity facts, paired with a learned **continuous persona space** derived from real human–LLM dialogues, exposed as **steering directions** at generation time. The continuous space supports magnitude control along trait directions, interpolation between users, and compositional vector arithmetic — controllability operations that text-conditioned simulators structurally cannot perform.

## 2. Central hypotheses (each must be falsifiable)

- **H1 (Discrete prompting is coarse):** Prompted LLMs given categorical persona labels (low/med/high) produce outputs whose measured trait distributions are bimodal or collapsed, not three-modal.
- **H2 (Numeric prompting does not yield continuous control):** Prompted LLMs given numeric persona magnitudes (0–100) do not produce trait scores that vary monotonically with the prompted value. We expect flat regions and discontinuities.
- **H3 (A learned continuous space matches prompting on standard metrics):** A simulator conditioned on a learned persona vector (no arithmetic, no RL) matches or exceeds USP on authenticity, profile-dialogue consistency, and continuity metrics on LMSYS-USP.
- **H4 (The learned space is geometrically structured):** Magnitude operations along learned trait directions produce monotonically varying trait scores in outputs; interpolation between two real users' vectors produces outputs whose measured traits move monotonically between the endpoints.
- **H5 (Compositional combinations work — Type 1):** Adding two trait directions produces outputs that simultaneously express both traits at levels comparable to single-trait edits.
- **H6 (Some compositions show emergent structure — Type 2, exploratory):** Specific compositions (e.g., polite + frustrated → passive-aggressive; expert + condescending → mansplaining register) produce a distinct register that scores higher on the emergent trait than either component alone.

H1–H4 are core. H5 is expected. H6 is moonshot — report honestly whether it works.

## 3. Dataset

- **Primary:** LMSYS-USP (open-sourced by Wang et al., ACL 2025). Real human–LLM conversations from LMSYS-Chat-1M, filtered and annotated with GPT-4o-extracted profiles (OF + SC). Splits: 87,882 train / 4,626 val / 2,366 test. Confirm checksums against the GitHub release before training.
- **Secondary (for generalization claim):** one out-of-domain dialogue corpus. Candidates: MultiWOZ (task-oriented), PersonaChat (predefined personas). **[DECIDE]** which one. Recommend MultiWOZ to support "domain-general" framing.
- **Held-out evaluation set for controllability experiments:** 500 conversations sampled from LMSYS-USP test split, stratified by extracted Big Five traits to ensure trait diversity.

## 4. Trait inventory (the dimensions we will steer)

Continuous dimensions to learn and steer (subject to confirmation by the within-vs-between-user variance analysis in Section 6.1):

1. Politeness
2. Formality
3. Verbosity (message length, embedding-derived)
4. Expertise (technical / domain sophistication)
5. Skepticism / pushback tendency
6. Patience / impatience
7. Curiosity / engagement

Discrete anchors (NOT to be made continuous):

- Occupation category
- Native-language indicator (proxy: error patterns, idiom use)
- Age bracket (if recoverable from extracted profile)
- Family structure (e.g., "father of two")

**[DECIDE]** before training: final continuous trait list. Do not exceed 8 dimensions in the headline experiments; reviewers parse trait tables, and 7 is the magic number.

## 5. The baseline ladder

Run all rungs on the same test set with the same metrics. Each rung isolates a specific claim.

| Rung | Method | Claim isolated | Status |
|------|--------|---------------|--------|
| 0 | Real user utterances from LMSYS test split | Authenticity ceiling | Reference |
| 1a | GPT-4o prompted with categorical persona (low/med/high per trait) | Discrete prompting baseline; tests H1 | Baseline |
| 1b-bare | GPT-4o prompted with bare numeric persona ("anger: 67/100") | Tests H2, weakest prompting | Baseline |
| 1b-scaled | GPT-4o prompted with scaled description ("anger is high, specifically 67 on 0–100") | Tests H2, stronger prompting | Baseline |
| 1b-fewshot | GPT-4o prompted with few-shot anchors (3 examples at 20/50/80, asked for 67) | Tests H2, strongest prompting | Baseline |
| 1c | USP (pretrained, open-sourced by Wang et al.) | Strongest profile-text-conditioned simulator with RL | Baseline |
| 2a | Learned persona vector, no arithmetic — encode dialogue → vector, condition generator on vector | Tests H3 (representation alone, no geometry claim) | Proposed |
| 2b | Learned vector + arithmetic (interpolation, magnitude, composition) | Tests H4–H6 (geometric structure of the space) | Proposed |
| 3 (oracle) | Classifier-guided decoding (rerank candidates by per-trait classifier scores) | Controllability ceiling, ignoring inference cost | Oracle |

**Critical:** report H2 results from 1b-fewshot, not from 1b-bare. Strawmanning prompting kills the paper.

## 6. Experiments

### 6.1 Foundation analysis (do this first, before any training)

**Goal:** empirically justify the discrete/continuous split rather than asserting it.

- For each candidate trait in Section 4, extract per-conversation trait scores on LMSYS-USP using (a) an independent trait classifier and (b) behavioral proxies.
- For users with ≥ 3 conversations, compute within-user variance and between-user variance per trait.
- Plot the within/between variance ratio. Traits with high within-user variance are continuous and conversation-context-dependent; traits with near-zero within-user variance are discrete-stable.
- **Deliverable:** Figure 1 of the paper. This figure justifies the entire hybrid framing. If the split does not fall out cleanly, stop and reconsider the framing before training anything.

### 6.2 Monotonicity (the headline controllability figure)

For each continuous trait and each method that exposes a knob (1b variants, 2a if applicable, 2b):

- Set the knob to values in {0.0, 0.2, 0.4, 0.6, 0.8, 1.0} (or 0–100 scaled equivalent for 1b).
- Generate 200 simulated utterances at each setting, conditioned on a fixed context and random discrete anchors.
- Score outputs with independent trait classifiers and behavioral proxies.
- Plot trait score vs. knob setting; compute monotonicity coefficient (Spearman rank correlation between intended and measured trait).
- **Hypothesis confirmation:** 2b should show monotonic, near-linear curves. 1b should be flat or step-shaped.

### 6.3 Interpolation between real users

- Pick 50 pairs of real users from the held-out evaluation set with maximally different trait profiles.
- For each pair (A, B), generate utterances at vector positions α·A + (1−α)·B for α ∈ {0, 0.25, 0.5, 0.75, 1.0}.
- Measure each trait score along the path; check monotonicity per trait.
- **Hypothesis confirmation:** all measured traits transition smoothly from A's profile to B's.

### 6.4 Compositional combinations (Type 1)

Test eight pairs (full grid is Section 7 of the writeup, abbreviated here):

- expert + verbose / expert + terse / novice + verbose / novice + terse
- polite + skeptical / impolite + skeptical
- formal + curious / casual + curious

For each pair, generate from the composed vector and verify both traits register at levels comparable to single-trait edits.

### 6.5 Compositional emergence (Type 2, exploratory)

Test six compositions for emergent registers:

- expert + condescending → mansplaining
- verbose + impatient → run-on rants
- polite + frustrated → passive-aggression
- formal + casual → ironic / sarcastic
- novice + confident → Dunning–Kruger
- expert + insecure → hedging-despite-knowing

Score with (a) LLM-as-judge for the emergent target trait, (b) human eval on a 100-sample subset. Compare composed-vector scores against single-component scores; emergence claim requires the composed score to exceed both component scores on the emergent target.

### 6.6 Analogy (the word2vec moonshot)

- Identify quadruples of real users (A, B, C, D) where A:B :: C:D structurally (e.g., A=formal expert, B=casual expert, C=formal novice, D=casual novice).
- Compute D̂ = B − A + C and compare to D.
- Report (a) cosine similarity of D̂ to D, (b) trait scores of utterances generated from D̂ vs. D's true vector.

### 6.7 Standard-benchmark comparison

Reproduce USP's Table 4 (conversation-level) for rungs 1c, 2a, 2b. Metrics: ESR, Sem-Sim, Style-Sim, AVA, r-DP.P, r-DP.R, r-DPC, P.Cover, SC.Score. Add ADV diversity metric from USP Table 3.

**Required outcome:** 2a and 2b match or beat USP on consistency and authenticity. If 2b regresses on these standard metrics relative to 2a (controllability bought at the cost of authenticity), report the trade-off honestly and discuss.

### 6.8 Compute and efficiency

Report wall-clock training time and GPU-hours for each rung. Specifically:

- USP (1c) per their paper: ~2 days SFT on 4×A100 + ~5 days PPO on 2×H20.
- 2a target: SFT-equivalent only, no RL.
- 2b target: same as 2a; arithmetic is inference-time only.

The "no RL needed" claim is part of the headline. Document it.

## 7. Methodological commitments (decide before starting)

- **[DECIDE] Base model for rungs 2a/2b.** Recommend LLaMA-3-8B-Base for direct comparability with USP.
- **[DECIDE] Encoder source for persona vectors.** Two options: (a) encode the GPT-4o-extracted profile text (simple), (b) encode the user's dialogue turns directly via a learned encoder trained to predict user utterances (EPSVec-style). Recommend (b) for the headline experiments because it sidesteps reliance on text profiles, but report (a) as an ablation.
- **[DECIDE] Vector injection mechanism.** Options ordered by implementation cost: soft prompting (recommended for first pass) → activation steering → adapter / FiLM-style conditioning. Start with soft prompting; report steering as a zero-finetuning variant if time permits.
- **[DECIDE] Linearity-encouraging training objective.** Linear/compositional structure does not emerge for free. Required ingredients: contrastive loss separating traits, orthogonality regularizer between identified trait directions, optional synthetic compositional augmentations (generate dialogues with deliberately combined trait labels and train the encoder to place them at the vector sums of the components). Decide which of these are in v1 vs. ablated later.
- **[DECIDE] Trait-score ground truth.** Primary: independent trait classifiers trained on external corpora. Confirmatory: behavioral proxies (length, type-token ratio, hedging-word frequency, lowercase ratio, jargon density). LLM-as-judge reserved for emergence experiments (Section 6.5) where other measures cannot capture the target.

## 8. Metrics summary

- **Authenticity:** Sem-Sim (SimCSE), Style-Sim (Wegmann et al. style embeddings), AVA (author verification accuracy).
- **Consistency:** r-DP.P, r-DP.R, r-DPC (USP's NLI-decomposed metrics), P.Cover (keyword overlap — *low* is good for our framing, signals non-copying), SC.Score (LLM-judge subjective characteristic match).
- **Continuity:** ESR (early stop rate).
- **Diversity:** ADV (USP's PCA-distance metric). Add: precision/recall of distributions for a stronger diversity statement.
- **Controllability (new):**
  - Monotonicity coefficient: Spearman ρ between intended trait knob value and measured trait score.
  - Trait-edit precision: when one trait is edited, how much do unedited traits move? Lower is better. Measured as mean absolute change in non-target trait scores per unit change in target trait.
  - Interpolation smoothness: average per-step trait change variance along α paths.
  - Compositional fidelity: for composed vector A+B, measured traits A_score and B_score relative to single-edit baselines.

## 9. Risks and what to do about each

- **R1 (vector bandwidth):** A single vector may not carry enough persona information for long horizons; consistency may decay late in conversations. *Mitigation:* run turn-by-turn consistency plots in the foundation phase. If decay appears, add multi-vector decomposition (separate vector for OF vs. SC, mirroring USP's schema) or per-turn re-injection.
- **R2 (non-linear space):** A learned encoder may produce a curved manifold where arithmetic is meaningless. *Mitigation:* the linearity-encouraging objective (Section 7). Detect by running Section 6.2 monotonicity early; if curves are not monotonic, the geometry is wrong and methods must be revisited.
- **R3 (trait classifier circularity):** If trait classifiers are trained on the same data we use for evaluation, results are circular. *Mitigation:* train classifiers on external labeled corpora; report on LMSYS test as a transfer setting.
- **R4 (1b strawman risk):** If 1b is run only in the bare-number form, reviewers will dismiss the H2 result. *Mitigation:* report all three 1b variants; lead with the strongest (fewshot).
- **R5 (USP reproduction):** If USP's open-sourced model does not reproduce their published numbers on our setup, this baseline collapses. *Mitigation:* reproduction is the first week's task; if reproduction fails, contact authors and adjust the baseline disclosure honestly.

## 10. Execution sequencing (suggested)

- **Week 1–2:** Reproduce USP (1c) on LMSYS-USP. Run foundation analysis (Section 6.1). Decide trait inventory based on the variance analysis result.
- **Week 3:** Train independent trait classifiers. Run 1a/1b baselines and produce H1, H2 evidence (the motivation figures).
- **Week 4–6:** Train 2a (learned vector, no arithmetic). Run Section 6.7 standard-benchmark comparison. Decision gate: if 2a does not match USP on authenticity/consistency, stop and diagnose before adding geometry.
- **Week 7–8:** Add linearity-encouraging objective to produce 2b. Run monotonicity (6.2), interpolation (6.3), composition Type 1 (6.4).
- **Week 9:** Type 2 emergence (6.5) and analogy (6.6).
- **Week 10:** Generalization experiment on secondary dataset.
- **Week 11–12:** Writeup, ablations, human eval.

This is aggressive. Slip the schedule for foundation analysis and reproduction rather than for the headline experiments.

## 11. What the agent should *not* do

- Do not skip rung 1b-fewshot. The whole motivation depends on it.
- Do not run controllability experiments before rung 2a matches USP on standard metrics. A geometrically nice but unfaithful simulator is not a publishable result.
- Do not run Section 6.5 (Type 2 emergence) without paired human eval. LLM-judge-only results on emergent registers will not survive review.
- Do not introduce reinforcement learning. The "no RL" claim is core to the paper.
- Do not silently expand the trait inventory beyond 8 dimensions.
- Do not switch encoder, injection mechanism, or base model mid-experiment without re-running prior rungs.
