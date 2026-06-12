"""
Persona-conditioned two-simulator dialogue (MVP).

Baseline scaffold for the Controllable Persona-Vector project (see
persona_vector_project_plan.md). A USER SIMULATOR conditioned on a structured
persona (discrete anchors + categorical trait levels) pursues an INTENT while
talking to an AGENT that tries to help. Both roles are the same locally-loaded
Llama 3.1 8B Instruct model. This is the plan's Rung 1a (text-prompted persona,
no vectors).
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HF_TOKEN = 
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

TEMPERATURE = 0.0   # greedy / deterministic baseline
MAX_TURNS = 10
STOP = "###DONE###"  # user emits this when satisfied


# ---------------------------------------------------------------------------
# Persona: discrete anchors + categorical (low/med/high) trait levels
# ---------------------------------------------------------------------------

TRAIT_DESCRIPTIONS = {
    "politeness": {"low": "blunt and curt", "medium": "reasonably polite", "high": "very polite and warm"},
    "formality": {"low": "casual, uses contractions", "medium": "neutral register", "high": "formal and precise"},
    "verbosity": {"low": "terse, a sentence or two", "medium": "moderate length", "high": "verbose, over-explains"},
    "expertise": {"low": "a layperson", "medium": "somewhat informed", "high": "very knowledgeable, uses jargon"},
    "skepticism": {"low": "trusting", "medium": "mildly questioning", "high": "skeptical, asks for evidence"},
    "patience": {"low": "impatient, wants answers fast", "medium": "reasonably patient", "high": "very easygoing"},
}


@dataclass
class Persona:
    name: str
    anchors: dict = field(default_factory=dict)
    traits: dict = field(default_factory=dict)

    def describe(self) -> str:
        anchors = "\n".join(f"- {k}: {v}" for k, v in self.anchors.items())
        traits = "\n".join(f"- {t}: {TRAIT_DESCRIPTIONS[t][lvl]}" for t, lvl in self.traits.items())
        return f"{anchors}\nYou are:\n{traits}"


PERSONA = Persona(
    name="skeptical_foodie",
    anchors={"occupation": "graphic designer", "background": "new in town, picky about food"},
    traits={
        "politeness": "low",
        "formality": "low",
        "verbosity": "low",
        "expertise": "low",
        "skepticism": "low",
        "patience": "low",
    },
)

INTENT = (
    "You want a recommendation for a good restaurant nearby for dinner tonight. "
    "You are done once you have a specific place you are happy with."
)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

USER_SYSTEM = f"""You are role-playing a real person talking to an assistant. Stay in character and speak naturally.

WHO YOU ARE:
{PERSONA.describe()}

WHAT YOU WANT:
{INTENT}

Send one message per turn. When your goal is met, end your final message with {STOP} on its own line."""

AGENT_SYSTEM = """You are a helpful local-recommendations assistant. Help the user find a good restaurant nearby.
Ask brief clarifying questions if needed, then give concrete suggestions. Keep replies focused."""


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

print(f"[llama] Loading {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, token=HF_TOKEN, torch_dtype=torch.float16, device_map="auto"
)
print(f"[llama] Ready on {model.device}")


def infer(messages, max_new_tokens=256):
    ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    ids = (ids.input_ids if hasattr(ids, "input_ids") else ids).to(model.device)
    with torch.no_grad():
        out = model.generate(
            ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=max_new_tokens,
            do_sample=TEMPERATURE > 0,
            temperature=TEMPERATURE or None,
            top_p=0.9 if TEMPERATURE > 0 else None,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    return re.sub(r"<\|[^|]+\|>", "", text).strip()


# ---------------------------------------------------------------------------
# Two simulators: each keeps its own history (the other speaker is "user")
# ---------------------------------------------------------------------------

class Sim:
    def __init__(self, system):
        self.messages = [{"role": "system", "content": system}]

    def respond(self, incoming):
        if incoming is not None:
            self.messages.append({"role": "user", "content": incoming})
        text = infer(self.messages)
        self.messages.append({"role": "assistant", "content": text})
        return text


def run():
    header = (
        f"{'='*60}\n"
        f"PERSONA: {PERSONA.name}\n"
        f"{PERSONA.describe()}\n"
        f"INTENT: {INTENT}\n"
        f"{'='*60}\n"
    )
    print(header)

    user, agent = Sim(USER_SYSTEM), Sim(AGENT_SYSTEM)
    turns, stopped = [], False

    user_text = user.respond(None)  # user opens
    print(f"[USER]  {user_text}\n")
    turns.append({"speaker": "user", "text": user_text})

    for _ in range(MAX_TURNS):
        stopped = STOP in user_text
        user_text = user_text.replace(STOP, "").strip()
        turns[-1]["text"] = user_text
        if stopped:
            break

        agent_text = agent.respond(user_text)
        print(f"[AGENT] {agent_text}\n")
        turns.append({"speaker": "agent", "text": agent_text})

        user_text = user.respond(agent_text)
        print(f"[USER]  {user_text}\n")
        turns.append({"speaker": "user", "text": user_text})

    termination = "user_satisfied" if stopped else "max_turns_reached"
    print(f"{'='*60}\nTermination: {termination} | turns: {len(turns)}")

    out_dir = Path(__file__).parent / "logs_out"
    out_dir.mkdir(exist_ok=True)
    record = {"model": MODEL_NAME, "persona": PERSONA.__dict__, "intent": INTENT,
              "termination": termination, "turns": turns}
    (out_dir / f"persona_{PERSONA.name}.json").write_text(json.dumps(record, indent=2))
    (out_dir / f"persona_{PERSONA.name}.txt").write_text(
        header + "\n".join(f"[{t['speaker'].upper()}] {t['text']}\n" for t in turns)
    )
    print(f"Saved to {out_dir}/persona_{PERSONA.name}.{{json,txt}}")


if __name__ == "__main__":
    run()
