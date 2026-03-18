"""
Day 12 — Production Patterns: async, human-in-the-loop, retries, state persistence.
"""
import sys
sys.path.insert(0, "..")
from config import get_llm
from typing import TypedDict, Annotated, Optional, Literal
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from pydantic import BaseModel
import asyncio
import time

llm = get_llm(temperature=0.0)

# ── 1. Human-in-the-Loop with LangGraph Interrupt ─────────────────────────────
print("=" * 60)
print("1. Human-in-the-Loop (Interrupt/Approve)")

class ApprovalState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    task: str
    plan: str
    approved: bool
    result: str

def plan_node(state: ApprovalState) -> dict:
    plan = llm.invoke(
        f"Create a brief action plan for: {state['task']}\n\nList 3 specific steps."
    ).content
    print(f"\n[Plan Agent] Generated plan:\n{plan}")
    return {"plan": plan, "messages": [AIMessage(content=f"Plan:\n{plan}")]}

def human_approval_node(state: ApprovalState) -> dict:
    """This node pauses and waits for human input."""
    print(f"\n⏸️  INTERRUPT: Human approval required")
    print(f"Plan to approve:\n{state['plan']}")

    # In LangGraph, this is how you implement human-in-the-loop:
    # The graph pauses here and can be resumed with a Command
    human_response = interrupt({
        "type": "approval_request",
        "plan": state["plan"],
        "message": "Please approve or reject this plan"
    })

    approved = human_response.get("approved", False) if isinstance(human_response, dict) else True
    return {"approved": approved, "messages": [HumanMessage(content=f"Approved: {approved}")]}

def execute_node(state: ApprovalState) -> dict:
    if not state.get("approved", False):
        return {"result": "Task cancelled by human reviewer."}
    result = llm.invoke(f"Execute this plan step by step:\n{state['plan']}").content
    return {"result": result}

def route_after_approval(state: ApprovalState) -> str:
    return "execute" if state.get("approved") else "cancelled"

def cancelled_node(state: ApprovalState) -> dict:
    return {"result": "Cancelled by user."}

# Build with checkpointing (required for interrupt to work)
checkpointer = MemorySaver()
approval_builder = StateGraph(ApprovalState)
approval_builder.add_node("plan", plan_node)
approval_builder.add_node("human_approval", human_approval_node)
approval_builder.add_node("execute", execute_node)
approval_builder.add_node("cancelled", cancelled_node)
approval_builder.add_edge(START, "plan")
approval_builder.add_edge("plan", "human_approval")
approval_builder.add_conditional_edges("human_approval", route_after_approval,
                                        {"execute": "execute", "cancelled": "cancelled"})
approval_builder.add_edge("execute", END)
approval_builder.add_edge("cancelled", END)

approval_graph = approval_builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["human_approval"]  # Pause BEFORE this node
)

config = {"configurable": {"thread_id": "approval-001"}}

# Start the graph — it will pause before human_approval
print("Starting graph (will pause for human approval)...")
initial_state = {
    "messages": [HumanMessage(content="Build a web scraper")],
    "task": "Build a web scraper for news headlines",
    "plan": "",
    "approved": False,
    "result": "",
}

try:
    result = approval_graph.invoke(initial_state, config=config)
except Exception as e:
    print(f"Graph paused at interrupt (expected): {type(e).__name__}")

# Simulate human approving — resume with Command
print("\n✅ Human approves the plan")
try:
    result = approval_graph.invoke(
        Command(resume={"approved": True}),
        config=config
    )
    print(f"Result: {result.get('result', '')[:200]}")
except Exception as e:
    print(f"Note: {e}")


# ── 2. Retry with Exponential Backoff ─────────────────────────────────────────
print("\n" + "=" * 60)
print("2. Robust Retry Logic")

from tenacity import (
    retry, stop_after_attempt, wait_exponential,
    retry_if_exception_type, before_sleep_log
)
import logging

logger = logging.getLogger(__name__)

class LLMCallFailed(Exception):
    pass

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((LLMCallFailed, TimeoutError)),
)
def robust_llm_call(prompt: str) -> str:
    """LLM call with automatic retry on failure."""
    try:
        return llm.invoke(prompt).content
    except Exception as e:
        if "timeout" in str(e).lower() or "connection" in str(e).lower():
            raise LLMCallFailed(f"LLM call failed: {e}")
        raise

result = robust_llm_call("What is 5 + 5?")
print(f"Robust call result: {result[:100]}")


# ── 3. Async Agent Execution ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. Async Parallel Agent Execution")

async def async_agent_call(task: str, agent_id: str) -> dict:
    """Simulate an async agent call."""
    start = time.time()
    # Use async invoke for real async support
    response = await llm.ainvoke(f"[Agent {agent_id}] {task}")
    duration = time.time() - start
    return {
        "agent_id": agent_id,
        "task": task,
        "result": response.content[:100],
        "duration_ms": duration * 1000,
    }

async def run_parallel_agents():
    tasks = [
        ("Research Python async patterns", "A"),
        ("Explain event loops", "B"),
        ("Compare asyncio vs threading", "C"),
    ]

    print("Running 3 agents in parallel...")
    start = time.time()

    results = await asyncio.gather(*[
        async_agent_call(task, agent_id)
        for task, agent_id in tasks
    ])

    total = time.time() - start
    for r in results:
        print(f"  Agent {r['agent_id']}: {r['duration_ms']:.0f}ms — {r['result'][:60]}...")
    print(f"Total: {total*1000:.0f}ms (would be {sum(r['duration_ms'] for r in results):.0f}ms sequential)")
    return results

asyncio.run(run_parallel_agents())


# ── 4. State Persistence Pattern ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. Persistent Agent State")

import json
from pathlib import Path

class PersistentAgentState:
    """
    Persists agent state to disk between sessions.
    In production: use Redis, PostgreSQL, or a proper state store.
    """

    def __init__(self, session_id: str, state_dir: str = "/tmp/agent_states"):
        self.session_id = session_id
        self.state_file = Path(state_dir) / f"{session_id}.json"
        self.state_file.parent.mkdir(exist_ok=True)

    def load(self) -> dict:
        if self.state_file.exists():
            return json.loads(self.state_file.read_text())
        return {"messages": [], "context": {}, "created_at": time.time()}

    def save(self, state: dict):
        state["updated_at"] = time.time()
        self.state_file.write_text(json.dumps(state, indent=2, default=str))

    def update(self, key: str, value):
        state = self.load()
        state[key] = value
        self.save(state)

    def append_message(self, role: str, content: str):
        state = self.load()
        state.setdefault("messages", []).append({
            "role": role,
            "content": content,
            "timestamp": time.time(),
        })
        self.save(state)

# Demo persistent state
agent_state = PersistentAgentState("demo-session-001")
agent_state.append_message("user", "What is LangGraph?")
agent_state.append_message("assistant", "LangGraph is a framework for stateful agent workflows...")
agent_state.update("last_topic", "LangGraph")

loaded = agent_state.load()
print(f"Loaded state: {len(loaded['messages'])} messages, last topic: {loaded.get('last_topic')}")


# ── 5. Rate Limiting & Cost Control ───────────────────────────────────────────
print("\n" + "=" * 60)
print("5. Rate Limiting & Cost Tracking")

class TokenBudget:
    def __init__(self, max_tokens: int = 10000):
        self.max_tokens = max_tokens
        self.used = 0
        self.calls = 0

    def check(self, estimated_tokens: int) -> bool:
        return (self.used + estimated_tokens) <= self.max_tokens

    def consume(self, tokens: int):
        self.used += tokens
        self.calls += 1

    def remaining(self) -> int:
        return self.max_tokens - self.used

    def report(self) -> dict:
        return {
            "calls": self.calls,
            "tokens_used": self.used,
            "tokens_remaining": self.remaining(),
            "budget_pct": (self.used / self.max_tokens) * 100,
        }

budget = TokenBudget(max_tokens=5000)

def budget_aware_call(prompt: str, budget: TokenBudget) -> Optional[str]:
    estimated = len(prompt.split()) * 2  # Rough estimate
    if not budget.check(estimated):
        print(f"  ⚠️ Budget exceeded: {budget.report()}")
        return None

    result = llm.invoke(prompt).content
    actual_tokens = len((prompt + result).split())
    budget.consume(actual_tokens)
    return result

for i in range(3):
    result = budget_aware_call(f"What is the number {i+1}?", budget)
    if result:
        print(f"  Call {i+1}: {result[:50]}...")

print(f"\nBudget report: {budget.report()}")
