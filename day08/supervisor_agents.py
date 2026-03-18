"""
Day 8 — Supervisor & Hierarchical Agents.
The supervisor pattern is the most common production multi-agent design.
"""
import sys
sys.path.insert(0, "..")
from config import get_llm
from typing import TypedDict, Annotated, List, Literal
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel

llm = get_llm(temperature=0.0)

# ── Worker Agents ──────────────────────────────────────────────────────────────

WORKER_AGENTS = {
    "researcher": "You are a research specialist. Find facts, summarize information, cite sources. Be thorough.",
    "coder": "You are a senior Python developer. Write clean, documented, working code. Include examples.",
    "analyst": "You are a data analyst. Analyze information, find patterns, provide insights with numbers.",
    "writer": "You are a technical writer. Write clear, structured documentation and explanations.",
}

def make_worker(name: str, persona: str):
    def worker_fn(task: str, context: str = "") -> str:
        prompt = f"""{persona}

Task assigned by supervisor: {task}
{f'Context: {context}' if context else ''}

Complete this task thoroughly."""
        return llm.invoke(prompt).content
    worker_fn.__name__ = name
    return worker_fn

workers = {name: make_worker(name, persona) for name, persona in WORKER_AGENTS.items()}

# ── Supervisor ─────────────────────────────────────────────────────────────────

class SupervisorDecision(BaseModel):
    next_worker: Literal["researcher", "coder", "analyst", "writer", "DONE"]
    task_for_worker: str
    reasoning: str

class SupervisorState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    original_task: str
    work_log: List[dict]
    final_answer: str
    steps: int

def supervisor_node(state: SupervisorState) -> dict:
    """The supervisor decides which worker to delegate to next."""
    work_summary = ""
    if state.get("work_log"):
        work_summary = "\n\nWork completed so far:\n" + "\n".join(
            f"- [{log['worker']}]: {log['result'][:150]}..."
            for log in state["work_log"]
        )

    structured_llm = llm.with_structured_output(SupervisorDecision)
    decision = structured_llm.invoke(
        f"""You are a supervisor managing a team of specialists.

Original task: {state['original_task']}
{work_summary}

Available workers:
- researcher: finds facts and information
- coder: writes Python code
- analyst: analyzes data and patterns
- writer: writes documentation and explanations
- DONE: task is complete, no more workers needed

Decide: which worker should act next, and what specific task should they do?
If the task is fully complete, choose DONE."""
    )

    print(f"\n  [Supervisor] → {decision.next_worker}: {decision.task_for_worker[:80]}...")
    print(f"  Reasoning: {decision.reasoning[:100]}...")

    return {
        "messages": [AIMessage(content=f"Delegating to {decision.next_worker}: {decision.task_for_worker}")],
        "next_worker": decision.next_worker,
        "pending_task": decision.task_for_worker,
        "steps": state.get("steps", 0) + 1,
    }

# We need to add these to the state
class FullSupervisorState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    original_task: str
    work_log: List[dict]
    final_answer: str
    next_worker: str
    pending_task: str
    steps: int

def make_worker_node(worker_name: str):
    def worker_node(state: FullSupervisorState) -> dict:
        result = workers[worker_name](
            state["pending_task"],
            context="\n".join(log["result"][:200] for log in state.get("work_log", []))
        )
        log_entry = {"worker": worker_name, "task": state["pending_task"], "result": result}
        print(f"  [{worker_name}] Completed: {result[:100]}...")
        return {
            "messages": [AIMessage(content=f"[{worker_name}] {result}")],
            "work_log": state.get("work_log", []) + [log_entry],
        }
    worker_node.__name__ = f"{worker_name}_node"
    return worker_node

def final_node(state: FullSupervisorState) -> dict:
    """Compile all work into a final answer."""
    all_work = "\n\n".join(
        f"=== {log['worker'].upper()} ===\n{log['result']}"
        for log in state.get("work_log", [])
    )
    final = llm.invoke(
        f"Compile this specialist work into a coherent final answer for: {state['original_task']}\n\n{all_work}"
    ).content
    return {"final_answer": final}

def route_to_worker(state: FullSupervisorState) -> str:
    if state.get("next_worker") == "DONE":
        return "final"
    if state.get("steps", 0) >= 5:  # Safety limit
        return "final"
    return state.get("next_worker", "final")

# ── Build Graph ────────────────────────────────────────────────────────────────
builder = StateGraph(FullSupervisorState)
builder.add_node("supervisor", supervisor_node)
builder.add_node("final", final_node)

for name in WORKER_AGENTS:
    builder.add_node(name, make_worker_node(name))
    builder.add_edge(name, "supervisor")  # After each worker → back to supervisor

builder.add_edge(START, "supervisor")
builder.add_conditional_edges(
    "supervisor",
    route_to_worker,
    {name: name for name in WORKER_AGENTS} | {"final": "final"}
)
builder.add_edge("final", END)

supervisor_graph = builder.compile()

# ── Run ────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Supervisor Agent System")
print("=" * 60)

task = "Explain what RAG (Retrieval Augmented Generation) is and write a simple Python example"

result = supervisor_graph.invoke({
    "messages": [HumanMessage(content=task)],
    "original_task": task,
    "work_log": [],
    "final_answer": "",
    "next_worker": "",
    "pending_task": "",
    "steps": 0,
})

print(f"\n{'=' * 60}")
print("FINAL ANSWER:")
print("=" * 60)
print(result["final_answer"][:600])
print(f"\n(Total workers used: {len(result['work_log'])})")


# ── Hierarchical: Supervisor of Supervisors ────────────────────────────────────
print("\n" + "=" * 60)
print("Hierarchical Pattern (concept)")
print("""
CEO Agent (high-level goal decomposition)
├── Research Team Supervisor
│   ├── Web Search Agent
│   ├── Database Query Agent
│   └── Document Analysis Agent
├── Engineering Team Supervisor
│   ├── Frontend Agent
│   ├── Backend Agent
│   └── Testing Agent
└── Communications Team Supervisor
    ├── Report Writer Agent
    └── Summary Agent

Each supervisor is itself a LangGraph agent.
Supervisors communicate via structured messages.
LangGraph supports nested subgraphs for this pattern.
""")
