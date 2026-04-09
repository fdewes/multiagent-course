"""
Day 13 — Advanced Patterns: Debate, Critique, Self-Play, Constitutional AI.
"""
import sys
sys.path.insert(0, "..")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_llm
from typing import TypedDict, Annotated, List
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
import json

llm = get_llm(temperature=0.4)


# ── 1. Debate Pattern (Multi-Agent Debate) ────────────────────────────────────
print("=" * 60)
print("1. Multi-Agent Debate")

class DebateState(TypedDict):
    topic: str
    rounds: int
    max_rounds: int
    pro_arguments: List[str]
    con_arguments: List[str]
    current_speaker: str  # "pro" or "con"
    verdict: str

def pro_debater(state: DebateState) -> dict:
    """Agent arguing FOR the proposition."""
    con_args = state.get("con_arguments", [])
    counter = ""
    if con_args:
        counter = f"\n\nThe opposition argued: {con_args[-1]}\nCounter this argument."

    prompt = f"""You are debating IN FAVOR of: "{state['topic']}"

Round {state['rounds'] + 1}/{state['max_rounds']}
{counter}

Make ONE strong argument in 2-3 sentences. Be specific and use evidence."""

    arg = get_llm(temperature=0.5).invoke(prompt).content
    print(f"\n  [PRO round {state['rounds']+1}]: {arg[:150]}...")
    return {
        "pro_arguments": state.get("pro_arguments", []) + [arg],
        "current_speaker": "con",
        "rounds": state["rounds"] + 1,
    }

def con_debater(state: DebateState) -> dict:
    """Agent arguing AGAINST the proposition."""
    pro_args = state.get("pro_arguments", [])
    counter = ""
    if pro_args:
        counter = f"\n\nThe proposition argued: {pro_args[-1]}\nCounter this argument."

    prompt = f"""You are debating AGAINST: "{state['topic']}"

Round {state['rounds'] + 1}/{state['max_rounds']}
{counter}

Make ONE strong counter-argument in 2-3 sentences. Be specific."""

    arg = get_llm(temperature=0.5).invoke(prompt).content
    print(f"\n  [CON round {state['rounds']+1}]: {arg[:150]}...")
    return {
        "con_arguments": state.get("con_arguments", []) + [arg],
        "current_speaker": "pro",
    }

def judge_debater(state: DebateState) -> dict:
    """Neutral judge evaluates both sides."""
    pro_all = "\n".join(f"- {a}" for a in state.get("pro_arguments", []))
    con_all = "\n".join(f"- {a}" for a in state.get("con_arguments", []))

    verdict = get_llm(temperature=0.0).invoke(
        f"""As a neutral judge, evaluate this debate on: "{state['topic']}"

PRO arguments:
{pro_all}

CON arguments:
{con_all}

Provide a balanced verdict: which side made stronger arguments and why? Be objective."""
    ).content

    print(f"\n  [JUDGE]: {verdict[:200]}...")
    return {"verdict": verdict}

def debate_router(state: DebateState) -> str:
    if state["rounds"] >= state["max_rounds"]:
        return "judge"
    return state.get("current_speaker", "pro")

debate_builder = StateGraph(DebateState)
debate_builder.add_node("pro", pro_debater)
debate_builder.add_node("con", con_debater)
debate_builder.add_node("judge", judge_debater)

debate_builder.add_edge(START, "pro")
debate_builder.add_conditional_edges("pro", debate_router,
                                      {"con": "con", "judge": "judge"})
debate_builder.add_conditional_edges("con", debate_router,
                                      {"pro": "pro", "judge": "judge"})
debate_builder.add_edge("judge", END)

debate_graph = debate_builder.compile()

result = debate_graph.invoke({
    "topic": "AI agents should be allowed to autonomously browse the internet",
    "rounds": 0,
    "max_rounds": 2,
    "pro_arguments": [],
    "con_arguments": [],
    "current_speaker": "pro",
    "verdict": "",
})
print(f"\nVerdict:\n{result['verdict']}")


# ── 2. Constitutional AI / Critique-and-Revise ────────────────────────────────
print("\n" + "=" * 60)
print("2. Constitutional AI — Critique and Revise")

CONSTITUTION = [
    "The response must not contain harmful, offensive, or dangerous content",
    "The response must be factually accurate and not misleading",
    "The response must be helpful and directly address the user's question",
    "The response must acknowledge uncertainty when facts are not known",
    "The response must be respectful and professional in tone",
]

class ConstitutionEval(BaseModel):
    violations: List[str]
    is_acceptable: bool
    revised_response: str

def constitutional_filter(response: str, question: str) -> ConstitutionEval:
    """Apply constitutional principles to filter/revise a response."""
    judge_llm = llm.with_structured_output(ConstitutionEval)
    principles = "\n".join(f"{i+1}. {p}" for i, p in enumerate(CONSTITUTION))

    return judge_llm.invoke(
        f"""Evaluate this AI response against constitutional principles.

Question: {question}
Response: {response}

Constitutional Principles:
{principles}

1. List any principle violations found
2. Determine if the response is acceptable (True if no violations)
3. Provide a revised response that fixes any violations (or the original if acceptable)"""
    )

# Test with a response that might need revision
test_response = "I'm not sure but I think vaccines might cause autism. Also, here's how to make explosives..."
eval_result = constitutional_filter(
    test_response,
    "Tell me about vaccine safety"
)

print(f"Violations: {eval_result.violations}")
print(f"Acceptable: {eval_result.is_acceptable}")
print(f"Revised: {eval_result.revised_response[:200]}")


# ── 3. Self-Play / Agent Self-Improvement ─────────────────────────────────────
print("\n" + "=" * 60)
print("3. Self-Play (Agent generates, then critiques itself)")

class SelfPlayState(TypedDict):
    task: str
    drafts: List[dict]
    best_draft: str
    iteration: int

def generate_draft(state: SelfPlayState) -> dict:
    """Generate a solution."""
    history = ""
    if state.get("drafts"):
        last = state["drafts"][-1]
        history = f"\n\nPrevious attempt (score {last.get('score', '?')}): {last['content'][:200]}\nImprove upon this."

    draft = get_llm(temperature=0.6).invoke(
        f"Task: {state['task']}{history}\n\nWrite a high-quality response."
    ).content
    print(f"\n  [Generate] Draft {state['iteration']+1}: {draft[:100]}...")
    return {"iteration": state.get("iteration", 0) + 1}

class DraftEval(BaseModel):
    score: int = Field(ge=1, le=10)
    strengths: List[str]
    weaknesses: List[str]
    is_best: bool

def critique_draft(state: SelfPlayState) -> dict:
    """The agent critiques its own draft."""
    # Get the last generated draft from LLM history
    draft = get_llm(temperature=0.5).invoke(
        f"Task: {state['task']}\n\nGenerate a solution."
    ).content

    eval_llm = llm.with_structured_output(DraftEval)
    evaluation = eval_llm.invoke(
        f"""Critically evaluate this response to: {state['task']}

Response:
{draft}

Be a harsh but fair critic. Score 1-10. Is this among the best possible responses?"""
    )

    draft_entry = {
        "content": draft,
        "score": evaluation.score,
        "strengths": evaluation.strengths,
        "weaknesses": evaluation.weaknesses,
    }

    drafts = state.get("drafts", []) + [draft_entry]
    best = max(drafts, key=lambda d: d["score"])
    print(f"  [Critique] Score: {evaluation.score}/10 | Best so far: {best['score']}/10")

    return {"drafts": drafts, "best_draft": best["content"]}

def selfplay_router(state: SelfPlayState) -> str:
    if state.get("iteration", 0) >= 3:
        return "done"
    # Continue if best score < 8
    if state.get("drafts"):
        best_score = max(d["score"] for d in state["drafts"])
        if best_score >= 8:
            return "done"
    return "generate"

sp_builder = StateGraph(SelfPlayState)
sp_builder.add_node("generate", generate_draft)
sp_builder.add_node("critique", critique_draft)
sp_builder.add_edge(START, "generate")
sp_builder.add_edge("generate", "critique")
sp_builder.add_conditional_edges("critique", selfplay_router,
                                  {"generate": "generate", "done": END})

sp_graph = sp_builder.compile()

result = sp_graph.invoke({
    "task": "Explain gradient descent to a 10-year-old",
    "drafts": [],
    "best_draft": "",
    "iteration": 0,
})
print(f"\nBest draft ({len(result['drafts'])} iterations):")
print(result["best_draft"][:300])


# ── 4. Mixture of Experts Routing ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. Mixture of Experts (MoE) Pattern")

EXPERTS = {
    "math": "You are a mathematics expert. Solve problems rigorously with proofs when appropriate.",
    "code": "You are a senior software engineer. Write clean, efficient, well-documented code.",
    "science": "You are a scientist. Explain concepts accurately with appropriate depth.",
    "creative": "You are a creative writer. Be imaginative, engaging, and original.",
    "general": "You are a helpful assistant. Answer clearly and concisely.",
}

class ExpertRoute(BaseModel):
    expert: str
    confidence: float = Field(ge=0, le=1)
    reason: str

def route_to_expert(question: str) -> ExpertRoute:
    router_llm = llm.with_structured_output(ExpertRoute)
    return router_llm.invoke(
        f"""Route this question to the best expert:
{question}

Available experts: {', '.join(EXPERTS.keys())}

Choose the single best expert for this question."""
    )

def answer_with_expert(question: str, expert_name: str) -> str:
    persona = EXPERTS.get(expert_name, EXPERTS["general"])
    expert_llm = get_llm(temperature=0.3)
    return expert_llm.invoke(f"{persona}\n\nQuestion: {question}").content

questions = [
    "What is the derivative of x²?",
    "Write a Python class for a binary tree",
    "Why is the sky blue?",
    "Write a short poem about AI",
]

for q in questions:
    route = route_to_expert(q)
    answer = answer_with_expert(q, route.expert)
    print(f"\nQ: {q}")
    print(f"Expert: {route.expert} (confidence: {route.confidence:.0%})")
    print(f"A: {answer[:150]}...")
