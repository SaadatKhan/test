# Retrieval-Grounded Behavioral Control for User Simulation

### Test-Time Controllability as a Substitute for Per-Aspect Post-Training

## Motivation

LLM-based user simulators are still far from real users. The dominant way to
close a specific gap — e.g., make the simulator more formal, more impatient, or
more uncertain — is **targeted post-training** (SFT / DPO / RL) on that one
aspect. This has three problems:

1. **It corrects one behavior at a time and does not generalize.** Making the
   simulator more formal does not make it more patient; each target trait tends
   to require its own data, its own objective, and its own training run. Overall
   human-likeness does not improve monotonically as a result.

2. **Controllability signals are hard to build at scale.** Producing the large,
   clean, per-trait preference/label sets that post-training needs is expensive
   and does not scale across the long tail of behaviors a practitioner might
   want to dial.

3. **Prompt-based control is uncalibrated.** Asking an LLM for *"more"* or
   *"less"* of a behavior yields overshoot, undershoot, or step-function
   responses. LLMs do not reliably represent graded intensity of a trait, so the
   requested behavior is not faithfully realized.

> **Positioning note (do not overclaim):** controllable user simulators are an
> *active* thread. The gap we target is narrower and defensible: existing
> control is either **post-training-based** (does not transfer across traits) or
> **prompt-based** (uncalibrated). What is missing is *training-free,
> human-grounded, intensity-calibrated* control at test time.

**Our stance.** When we need *more* of a behavior in a simulator, that behavior
should be grounded in **real human conversations that actually exhibit it**, and
injected **at test time** rather than baked in through per-aspect training.
Given a target behavior, we retrieve real human conversations that exhibit it,
and let a planner write the simulator's instruction from those
conversations on the fly.

**We claim sufficiency, not mechanism.** Mechanistically this is
retrieval + test-time instruction customization. The contribution is the
*problem lens* plus the empirical demonstration that a simple, training-free
method is **sufficient** to match or beat per-aspect post-training on
controllability — and generalizes across behaviors without retraining.

---

## Unit of analysis: the conversation

Behavioral traits here (formality, politeness, patience, verbosity-as-a-style)
are **conversation-level properties**, not turn-level ones. A single turn does
not tell you whether a user is formal or impatient; the whole trajectory does.
So throughout this pipeline the unit we **score, bin, retrieve, and verify** is
the **user side of a whole conversation**, not an isolated turn. This also makes
human validation cheaper: annotators rate one conversation instead of every
turn.

---

# Overall Pipeline

```text
Human-Human Conversations (MultiWOZ)
        │
        ▼
Conversation-Level Behavior Scoring   ← (validated against human annotation)
        │
        ▼
Tertile Binning per trait  (low / medium / high)
        │
        ▼
Behavior Memory: bins store user-side conversations
        │
        ▼
Target Behavior (e.g. "more formal")  ──►  select the on-target tertile
        │
        ▼
Retrieve Real Human Conversations from that bin
        │
        ▼
Planner  ──►  fine-grained instruction written from the examples
        │
        ▼
LLM User Simulator  (context + goal + instruction + retrieved conversations)
        │
        ▼
Conversation Verification  (re-score the generated dialogue vs. target tertile)
        │
        ▼
Localized Editing (revise turns that pull the conversation off-target)
```

---

# Step 1: Score Conversations on Behavioral Traits

Each MultiWOZ conversation is scored (user side only) along interpretable
behavioral axes:

- Formality
- Politeness
- Straightforwardness
- Verbosity
- Uncertainty
- Patience
- Directness
- Acknowledgements

Each conversation gets one score per trait:

```text
Conversation_j  →  [ Formality, Politeness, Verbosity, Uncertainty, Patience, ... ]
```

> **Two kinds of trait — score them differently.**
> - **Measurable traits** (verbosity, turn length, information density) →
>   compute directly from text (e.g., mean tokens per user turn). No judge, no
>   validity question. Good candidate for the clean pilot.
> - **Judge-scored traits** (formality, politeness, patience, uncertainty) →
>   scored by an LLM judge over the conversation.

> **Load-bearing design point.** The judge scorer must be *validated against
> human annotation* on a held-out set, and it must NOT be the same signal used
> for final verification (Step 6) — otherwise the loop is self-confirming.
> Report scorer–human agreement.

---

# Step 2: Tertile Binning

For each trait, take the score distribution over all conversations and partition
it into **three tertiles**:

```text
Low        Medium        High
0–33%      33–66%        66–100%
```

The histogram itself is only a scaffold — it tells you how many conversations
fall in each bin. What matters is that each bin becomes a **bucket holding the
actual user-side conversations** that landed there. Tertiles (rather than 5
bins) keep each bucket well-populated and give a clean low/medium/high control
knob.

> **Corpus-relativity caveat.** Tertile boundaries are relative to the scoring
> corpus, so "High formality" means something specific to MultiWOZ. If a second
> corpus (e.g., SGD) is added, align bins explicitly rather than assuming
> transfer.

---

# Step 3: Behavior Memory

Each bin indexes the user-side conversations that fall in it:

```json
{
  "trait": "formality",
  "bin": "High",
  "conversations": [
    {
      "dialogue_id": "MUL0512",
      "domain": "Restaurant",
      "user_turns": [
        "Good evening. I would like to reserve a table, please.",
        "Thank you. Chinese cuisine would be my preference.",
        "That is perfect. I appreciate your assistance."
      ],
      "trait_score": 0.91
    },
    ...
  ]
}
```

The memory stores whole user-side conversations organized by trait-intensity
bin — real human trajectories, not disconnected turns.

---

# Step 4: Target Behavior → Bin Selection

A control request selects the on-target tertile for the requested trait:

```text
"Process the user with more formality"
        │
        ▼
Formality → High tertile
```

This is the substitute for "train a model for the formal-user trait." No
training run; just a bin lookup.

---

# Step 5: Retrieve Human Conversations

Pull user-side conversations from the on-target bin to serve as references for
generation:

```text
Formality = High
→  "Good evening. I would like to reserve a table, please."  ...
   "Might I ask what time you would recommend?"  ...
   "Thank you kindly for your help."  ...
```

> **Optional refinement — dialogue-state spotlight.** When generating a turn in
> a specific dialogue state (e.g., *provide cuisine*), the state can be used as
> a secondary key to *spotlight* the matching turns inside the retrieved
> conversations — so the simulator sees both the whole formal trajectory and how
> that trait was expressed in the current situation. This is a refinement, not a
> filter: the primary retrieval unit remains the whole conversation.

---

# Step 6: Planner → Fine-Grained Instruction

Rather than hand-writing "be formal," the **planner** reads the
retrieved conversations and writes the trait definition itself:

```text
Formal users in these conversations typically:
• open with a greeting and close with thanks
• use full sentences and complete grammar
• prefer "I would like" / "might I" over imperatives
• avoid slang, contractions, and abbreviations
• acknowledge the system's responses politely
```

This written instruction — grounded in the retrieved examples — becomes the
test-time instruction for the simulator.

> **Ablate this step.** Test (a) retrieved conversations only, (b) planner
> instruction only, (c) both. If (a) ≈ (c), the planner is not earning its
> place; if (c) > (a), grounding-via-reflection is a real result. This ablation
> is a finding either way.

---

# Step 7: Behavior-Guided Generation

```text
Dialogue Context
Task Goal
Fine-Grained Instruction (from planner)
Retrieved Human Conversations (references)
→ Generate the user turns.
```

---

# Step 8: Conversation-Level Verification

Because scoring is at the conversation level, verification is too. Score the
**generated dialogue** with the independent verification scorer and check its
tertile against the target:

```text
Target:    Formality = High
Generated: Formality score → Medium   →  MISS
```

---

# Step 9: Localized Editing

Identify the user turns dragging the conversation off-target and revise only
those, preserving goal, slots, dialogue state, and flow:

```text
Off-target turn: "yeah chinese works"
Analyst note:    add greeting/closing register, use full sentences, drop slang.
Edited:          "Chinese cuisine would be my preference, thank you."
→ re-score the conversation → accept or repeat.
```

Report edit-loop statistics (pass rate, iterations, convergence, fallback).

---

# Online Algorithm

```text
Input: Dialogue Context, Task Goal, Target Behavior
  → select the on-target tertile for the trait
  → retrieve human conversations from that bin
  → planner writes a fine-grained instruction from them
  → generate user turns (instruction + retrieved conversations)
  → verify the generated conversation's trait score vs. target tertile
  → Pass?  yes → continue
           no  → edit off-target turns → re-score
```

---

# Core Experiments (these decide publishability)

**E1 — Retrieval vs. Post-Training (the crux).**
For a target trait, compare (a) SFT/DPO post-training, (b) our test-time
retrieval method. The safest, always-defensible claim is **cross-trait
transfer**: post-training needs *N separate runs* for *N* traits, while our
system needs *N bin lookups* against one memory. Show we are **competitive on
the single target trait and strictly better on transfer at a fraction of the
cost**. Beating post-training head-to-head on one trait is upside, not the load-
bearing claim.

**E2 — Intensity calibration (headline result).**
Plot target trait intensity (low/medium/high tertile) vs. realized intensity for
prompting vs. ours. Prompting = noisy/step-like; ours = monotonic/tight. This is
the "LLMs can't reliably do more-vs-less" result made concrete, and it does not
depend on beating a post-trained model — it is the safest strong result.

**E3 — Human-likeness / faithfulness.**
Distributional distance between generated conversations and held-out real human
conversations per bin. Show editing reduces divergence from the *human*
distribution (not just judge agreement). Note the tension to resolve: hitting a
single tertile every time can make simulated users *more* behaviorally uniform
than real ones — consider sampling per-conversation targets from the human
conditional distribution rather than always pinning the top bin.

**E4 — Semantics preservation.**
Slot-F1 / goal-fulfillment before vs. after editing, confirming control does not
corrupt task success.

Datasets: MultiWOZ (primary), SGD (transfer). Backbone: LLaMA-3-8B-Base
(comparability with USP).

---

# Key Contributions (framed as sufficiency)

1. **A problem lens:** controllability of user simulators as a *test-time
   retrieval* problem, positioned as an alternative to per-aspect post-training.
2. **A human-grounded behavior memory:** real human conversations indexed by
   quantified trait-intensity tertiles.
3. **Calibrated, training-free control:** graded low/medium/high behavior
   realized via bins that prompting cannot reliably hit.
4. **Reflection-written instructions:** the trait definition is induced from
   retrieved human conversations rather than hand-crafted.
5. **Closed verify-and-edit loop:** conversation-level verification against a
   target tertile with localized editing that preserves task semantics.
6. **Empirical demonstration of sufficiency:** competitive with post-training on
   the target behavior and generalizes across behaviors without retraining.

---

# High-Level Research Hypothesis

> Per-aspect post-training improves one behavior at a time and does not
> overarchingly make user simulators more human-like, while prompt-based control
> is uncalibrated. A training-free framework that retrieves real human
> conversations by trait-intensity and lets a planner write the
> simulator's instruction at test time can deliver **calibrated, generalizable
> behavioral control** that is competitive with post-training on the target
> behavior while preserving task semantics.
