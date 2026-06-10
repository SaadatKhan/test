# Counterfactual Responsiveness in LLM User Simulators
## Research Plan — Target Venue: ICLR

---

## 1. One-Sentence Summary

Existing work makes sure a simulated user stays in character (persona
consistency); this work makes sure the simulated user is actually
listening (counterfactual responsiveness) — and shows the two may be
in tension.

---

## 2. The Failure Point

LLM user simulators often generate responses driven by the persona
prompt and generic "dialogue momentum" rather than by the semantic
content of the system's most recent turn. Change what the system says,
and the simulated user replies nearly the same way.

### Running example: "Maya"
Persona: budget-conscious flight booker; hard constraint — no
overnight layovers.

Same dialogue history, two versions of the system turn:

- Run A — System: "Found a $420 flight with a 9-hour overnight
  layover."
- Run B — System: "Found a $510 nonstop flight."

A real Maya rejects A and accepts/negotiates B. A broken simulator
says "Sounds good, let's book it" (or "anything cheaper?") to BOTH.

### Why consistency metrics cannot catch this
Maya sounds perfectly in character in both runs: polite,
budget-conscious, no contradiction with her persona or her earlier
turns. Every consistency/drift metric is a function of a SINGLE
trajectory (output vs. persona, output vs. history). This failure
only exists when comparing responses ACROSS two counterfactual
branches — it is a property of the input→output mapping, not of any
single output.

### Why it matters
- Evaluation: an unresponsive simulator gives identical success rates
  to good and bad systems (e.g., one that surfaces layover info vs.
  one that hides it).
- Training: RL against such a simulator yields noisy/useless reward —
  the policy is not penalized for violating user constraints.
- The simulator can score well on fluency, persona consistency, and
  goal completion while being worthless as an environment.

---

## 3. Formal Core

A persona P induces a decision function f_P over system behaviors.
Simulator U(u | H, s, P) gives the user response distribution given
history H, system turn s, persona P.

Counterfactual responsiveness at a turn, for system-turn pair
(s_A, s_B):

    CR(H, s_A, s_B) = D( U(· | H, s_A, P), U(· | H, s_B, P) )

measured on a dialogue-act / decision abstraction (accept, reject,
request-alternative, clarify), not on surface text.

### The defining biconditional
For minimal pairs (s_A, s_B):

1. SENSITIVITY: f_P(s_A) ≠ f_P(s_B)  ⟹  response distributions
   diverge, in the direction f_P dictates.
2. INVARIANCE: f_P(s_A) = f_P(s_B)  ⟹  response distributions stay
   put (no twitchiness to paraphrase, politeness markers, reordering).

Hard constraints ⟹ deterministic flips (limiting case).
Soft constraints (e.g., price $420 → $890 for a budget-conscious
persona) ⟹ graded, calibrated distribution shifts.

The target property is SELECTIVE sensitivity: respond to exactly the
input variation the persona's decision function deems relevant, and
nothing else.

### Relationship to persona consistency
- Consistency = static property: does the output match the spec?
- Responsiveness = causal property: does the input→output mapping
  implement the spec's decision function?
- Partial overlap exists (explicit acceptance of a hard-constraint
  violation could be caught by a strong, system-turn-grounded NLI
  check). Strictly outside consistency's reach:
  a) graded sensitivity with no flaggable violation,
  b) invariance failures (oversensitivity to cosmetic changes),
  c) turn-1 failures (drift is temporal by definition).
- Terminology note: avoid calling this "inconsistency across
  environments" in the paper — use causal vocabulary (intervention,
  counterfactual responsiveness, causal faithfulness to the
  interaction) to prevent reviewers pattern-matching to the drift
  literature.

---

## 4. Contributions

### Contribution 1 — Benchmark: minimal-pair system turns
- Take real or synthetic dialogues; at chosen turns generate paired
  interventions:
  - RELEVANT pairs: differ on an attribute interacting with the
    persona's constraints (layover present/absent; price $420 vs $890).
  - IRRELEVANT pairs: same offer, different surface form.
- Gold labels (should the decision flip / shift, and in which
  direction) derived mechanically from the persona spec's constraints.
- Scoring: run a simulator on both branches, classify responses into
  decision categories (small classifier or LLM judge with act schema).
  Report sensitivity on relevant pairs, invariance on irrelevant
  pairs, direction-correctness; single calibrated metric (e.g., AUC
  over relevant/irrelevant discrimination).
- Expected headline empirical finding: frontier prompted simulators
  score well on invariance but poorly on directional sensitivity,
  especially mid-dialogue when momentum is strong.

### Contribution 2 — Fix: counterfactual contrastive training
Training triples from the same minimal-pair machinery:
(H, s_A, u_A*) and (H, s_B, u_B*), with u_A*, u_B* correct responses
per branch (from a stronger model given explicit constraints, verified
against rule-based labels).

Two losses:
1. Branch correctness: preference optimization per branch — for
   context (H, s_A), prefer u_A* over u_B*. The rejected response is
   the CORRECT answer to the OTHER branch — exactly the confusion an
   insensitive simulator makes. Hard negatives make this DPO variant
   non-routine. (Methodological link to prior CW-DPO experience.)
2. Invariance regularizer: for irrelevant pairs, penalize divergence
   between response distributions (avoid training a twitchy simulator).

### Contribution 3 (representation-level, ICLR flavor)
Probe whether the decision-relevant attribute (e.g., layover
present/absent) is linearly decodable from the simulator's hidden
states at the response position, before vs. after training.
Hypothesis: in insensitive simulators the information is present in
early layers but washed out by persona/momentum features before
generation; training restores the causal path.

### Contribution 4 (the possible headline twist) — the tension
hypothesis
Persona-consistency training works by upweighting conditioning on P
relative to the rest of the context — which includes s_t. Hypothesis:
state-of-the-art consistency methods IMPROVE consistency scores while
DEGRADING counterfactual responsiveness, i.e., they manufacture this
failure. Experiment: take a strong persona-consistency method, measure
both axes, then show counterfactual contrastive training recovers
both. If the tension is real → headline finding. If not → still a
paper (orthogonal axis + benchmark + fix), positioned conservatively.

---

## 5. Experimental Design

- Domains (3, for generality):
  1. MultiWOZ-style booking (flights / restaurants)
  2. E-commerce negotiation
  3. Tech support
- Baselines:
  - Prompted simulators across several model scales
  - DAUS-style fine-tuned simulators
  - A persona-consistency-trained simulator (key ablation: shows
    consistency ≠ responsiveness)
- Downstream validation (reviewers will demand it):
  Train or evaluate a dialogue policy against the insensitive vs.
  responsive simulator; show the responsive simulator's rankings of
  system variants correlate better with human-judged rankings.

---

## 6. Anticipated Reviewer Objections and Defenses

1. "This is just persona inconsistency."
   → Consistency is computable from one trajectory; responsiveness is
   not even definable without an intervention on the system turn.
   Maya passes every consistency check in both branches while failing
   the responsiveness test. Plus the three strictly-outside cases
   (graded sensitivity, invariance, turn-1).

2. "This is just instruction-following failure."
   → The calibration framing: simulators don't simply ignore input;
   their sensitivity is MISCALIBRATED relative to the persona's
   decision function — a property that can only be defined, measured,
   and fixed in the user-simulation setting (it requires f_P).

3. "Benchmark generation requires hand-labeling."
   → Gold labels derive mechanically from structured persona
   constraints; the open design problem is generating s_B such that
   boundary-crossing is known by construction. (Next step below.)

---

## 7. Status and Next Steps

- [x] Failure point identified and differentiated from persona
      drift/consistency literature
- [x] Formal core drafted (decision function f_P, CR metric, the
      sensitivity/invariance biconditional)
- [x] Contribution structure and experimental design sketched
- [ ] Minimal-pair generation algorithm — generate s_B such that it
      is KNOWN whether a decision boundary of f_P was crossed,
      without hand-labeling (main practical difficulty)
- [ ] Persona spec format with machine-readable constraints
      (hard/soft) to support mechanical gold labels
- [ ] Pilot: measure CR for 2–3 prompted simulators on a small
      hand-built pair set to confirm the failure exists at useful
      effect sizes
- [ ] Decide headline framing after pilot: tension hypothesis vs.
      orthogonal-axis-plus-benchmark

---

## 8. Plain-Language Recap

A user simulator's reply should depend on two things: who the user is
and what was just said to them. Prior work verified the first; this
project verifies the second. Run the same conversation twice with one
changed system line: if the change matters to this persona, the reply
must change; if only the wording changed, it must not. Popular
simulators fail the first test while passing every existing metric.
We build the benchmark that exposes this, the training method that
fixes it, and (possibly) show that fixing persona drift has been
quietly causing it.