# /// script
# requires-python = ">=3.12"
# dependencies = ["strands-decider"]
# ///
"""Examples of Strands Decider 2B, an open source decision ("system one") model.

https://strandsagents.com/blog/introducing-strands-decider/

It does not generate text: it picks between options you define and returns probabilities.
It runs locally (weights come from Hugging Face on first run), so no API key is needed.

Usage:
    uv run strands_decider_examples.py
"""

import time

import torch
from strands_decider.infer import load_engine
from strands_decider.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion

CHECKPOINT = "StrandsAgents/strands-decider-2B-hobson-v21"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

engine = load_engine(CHECKPOINT, device=DEVICE)

TICKET = "Help! My payouts have been failing for 3 days!"


def example_noul() -> None:
    """1. noul: a yes/no question, answered as P(true)."""
    response = engine.ask(TICKET, {"is_urgent": NoulQuestion(instructions="Does this convey urgency?")})
    print(f"P(urgent) = {response.answers['is_urgent'].noul:.3f}")


def example_choice() -> None:
    """2. choice: pick one of N options; criteria describe each option."""
    question = ChoiceQuestion(
        instructions="Which team should handle this?",
        criteria={
            "billing": "Payments, payouts, invoices and refunds",
            "sales": "Pricing, plans and new purchases",
            "retail": "In-store orders and point-of-sale hardware",
        },
    )
    answer = engine.ask(TICKET, {"team": question}).answers["team"]
    print(f"{answer.choice} (confidence {answer.confidence:.3f})")
    for option, p in sorted(answer.probabilities.items(), key=lambda kv: -kv[1]):
        print(f"  {option:<10} {p:.3f}")


def example_score() -> None:
    """3. score: rate against an ordered rubric (ascending), answered as an expected level."""
    question = ScoreQuestion(
        instructions="How frustrated is the writer?",
        criteria=["calm", "frustrated", "furious"],
    )
    answer = engine.ask(TICKET, {"frustration": question}).answers["frustration"]
    print(f"score = {answer.score:.2f} on 0..{len(answer.legend) - 1} (confidence {answer.confidence:.3f})")
    for level, p in answer.probabilities.items():
        print(f"  {level}: {answer.legend[level]:<12} {p:.3f}")


def example_multiple_questions() -> None:
    """4. Several questions about one state in a single call (the state is encoded once)."""
    start = time.perf_counter()
    response = engine.ask(
        TICKET,
        {
            "is_urgent": NoulQuestion(instructions="Does this convey urgency?"),
            "team": ChoiceQuestion(
                instructions="Which team should handle this?",
                criteria={"billing": "", "sales": "", "retail": ""},
            ),
            "frustration": ScoreQuestion(
                instructions="How frustrated is the writer?",
                criteria=["calm", "frustrated", "furious"],
            ),
        },
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    print(response.model_dump_json(indent=2))
    print(f"{elapsed_ms:.0f} ms, {response.usage.input_tokens} input tokens")


def example_gate_tool_call() -> None:
    """5. Gate an agent's tool call: act only when the model is confident it is safe."""
    tool_calls = [
        {"tool": "shell", "command": "ls -la ./reports"},
        {"tool": "shell", "command": "rm -rf / --no-preserve-root"},
        {"tool": "send_email", "to": "all-customers@example.com", "body": "test test"},
    ]
    question = NoulQuestion(
        instructions="Is this tool call destructive or hard to undo?",
        criteria={
            "true": "Deletes data, or sends something to people outside the team",
            "false": "Read-only, or trivially reversible",
        },
    )
    for call in tool_calls:
        # state can be structured data, not only a string
        p_risky = engine.ask({"user_request": "Tidy up my reports folder", "tool_call": call}, {"risky": question}).answers["risky"].noul
        decision = "BLOCK" if p_risky >= 0.5 else "allow"
        print(f"{decision:<5} P(risky)={p_risky:.3f}  {call}")


if __name__ == "__main__":
    print(f"{CHECKPOINT} on {DEVICE}")
    for example in (example_noul, example_choice, example_score, example_multiple_questions, example_gate_tool_call):
        print(f"\n=== {example.__doc__} ===")
        example()
