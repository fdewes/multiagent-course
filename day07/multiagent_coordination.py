"""
Day 7 — Multi-agent coordination: pipeline, parallel fan-out, message passing.
"""
import sys
sys.path.insert(0, "..")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_llm
from typing import TypedDict, Annotated, List
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import asyncio
from pydantic import BaseModel

llm = get_llm(temperature=0.3)

# ── Pattern 1: Sequential Pipeline ────────────────────────────────────────────
print("=" * 60)
print("Pattern 1: Sequential Multi-Agent Pipeline")

class PipelineState(TypedDict):
    topic: str
    research_notes: str
    outline: str
    draft: str
    final_article: str

def research_agent(state: PipelineState) -> dict:
    """Agent 1: Research the topic."""
    prompt = f"""You are a research agent. Gather key facts and information about:
Topic: {state['topic']}

Provide 5-7 key facts, statistics, and important points. Be concise."""
    result = llm.invoke(prompt).content
    print(f"  [Research Agent] Done: {len(result)} chars")
    return {"research_notes": result}

def outline_agent(state: PipelineState) -> dict:
    """Agent 2: Create article outline from research."""
    prompt = f"""You are an editor. Create a clear article outline based on these research notes.

Topic: {state['topic']}
Research: {state['research_notes']}

Create a 4-section outline with headers and key points for each section."""
    result = llm.invoke(prompt).content
    print(f"  [Outline Agent] Done: {len(result)} chars")
    return {"outline": result}

def writing_agent(state: PipelineState) -> dict:
    """Agent 3: Write the article from outline."""
    prompt = f"""You are a writer. Write a concise article based on this outline.

Topic: {state['topic']}
Outline: {state['outline']}
Research notes: {state['research_notes']}

Write a 200-word article. Be engaging and informative."""
    result = llm.invoke(prompt).content
    print(f"  [Writing Agent] Done: {len(result)} chars")
    return {"draft": result}

def editor_agent(state: PipelineState) -> dict:
    """Agent 4: Polish and finalize."""
    prompt = f"""You are a senior editor. Polish this draft article.

Draft: {state['draft']}

Improve clarity, fix any issues, ensure good flow. Keep it concise."""
    result = llm.invoke(prompt).content
    print(f"  [Editor Agent] Done: {len(result)} chars")
    return {"final_article": result}

pipeline = StateGraph(PipelineState)
for node_fn in [research_agent, outline_agent, writing_agent, editor_agent]:
    pipeline.add_node(node_fn.__name__, node_fn)

pipeline.add_edge(START, "research_agent")
pipeline.add_edge("research_agent", "outline_agent")
pipeline.add_edge("outline_agent", "writing_agent")
pipeline.add_edge("writing_agent", "editor_agent")
pipeline.add_edge("editor_agent", END)

pipeline_graph = pipeline.compile()

result = pipeline_graph.invoke({"topic": "The rise of multi-agent AI systems in 2025"})
print(f"\nFinal Article:\n{result['final_article'][:500]}...")


# ── Pattern 2: Parallel Fan-Out ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Pattern 2: Parallel Fan-Out + Aggregation")

class ParallelState(TypedDict):
    question: str
    perspective_technical: str
    perspective_business: str
    perspective_ethical: str
    synthesis: str

def technical_expert(state: ParallelState) -> dict:
    response = llm.invoke(
        f"As a technical expert, analyze: {state['question']}\nFocus on technical implications. 2 sentences."
    ).content
    return {"perspective_technical": response}

def business_expert(state: ParallelState) -> dict:
    response = llm.invoke(
        f"As a business strategist, analyze: {state['question']}\nFocus on business impact. 2 sentences."
    ).content
    return {"perspective_business": response}

def ethics_expert(state: ParallelState) -> dict:
    response = llm.invoke(
        f"As an AI ethics researcher, analyze: {state['question']}\nFocus on ethical considerations. 2 sentences."
    ).content
    return {"perspective_ethical": response}

def synthesizer(state: ParallelState) -> dict:
    prompt = f"""Synthesize these three expert perspectives on: {state['question']}

Technical: {state['perspective_technical']}
Business: {state['perspective_business']}
Ethical: {state['perspective_ethical']}

Provide a balanced 3-sentence synthesis."""
    return {"synthesis": llm.invoke(prompt).content}

parallel = StateGraph(ParallelState)
parallel.add_node("technical", technical_expert)
parallel.add_node("business", business_expert)
parallel.add_node("ethics", ethics_expert)
parallel.add_node("synthesize", synthesizer)

# Fan-out: START → all three experts in parallel
parallel.add_edge(START, "technical")
parallel.add_edge(START, "business")
parallel.add_edge(START, "ethics")

# Fan-in: all experts → synthesizer
parallel.add_edge("technical", "synthesize")
parallel.add_edge("business", "synthesize")
parallel.add_edge("ethics", "synthesize")
parallel.add_edge("synthesize", END)

parallel_graph = parallel.compile()

result = parallel_graph.invoke({
    "question": "Should AI agents have access to the internet autonomously?",
    "perspective_technical": "",
    "perspective_business": "",
    "perspective_ethical": "",
    "synthesis": "",
})
print(f"\nTechnical: {result['perspective_technical']}")
print(f"\nBusiness: {result['perspective_business']}")
print(f"\nEthics: {result['perspective_ethical']}")
print(f"\nSynthesis: {result['synthesis']}")


# ── Pattern 3: Agent Message Passing with Protocol ─────────────────────────────
print("\n" + "=" * 60)
print("Pattern 3: Structured Agent Message Protocol")

class AgentMessage(BaseModel):
    sender: str
    recipient: str
    message_type: str  # task | result | question | error
    content: str
    priority: int = 1

class MessageBoardState(TypedDict):
    inbox: List[dict]    # Messages to be processed
    outbox: List[dict]   # Processed messages
    task: str
    final_result: str

def coordinator_agent(state: MessageBoardState) -> dict:
    """Breaks task into sub-tasks and sends to specialists."""
    structured_llm = llm.with_structured_output(
        {"type": "object",
         "properties": {
             "sub_tasks": {"type": "array", "items": {"type": "string"}}
         }, "required": ["sub_tasks"]}
    )
    result = structured_llm.invoke(
        f"Break this task into 2-3 specific sub-tasks for specialists:\n{state['task']}"
    )
    messages = [
        AgentMessage(
            sender="coordinator",
            recipient=f"specialist_{i}",
            message_type="task",
            content=subtask,
        ).model_dump()
        for i, subtask in enumerate(result.get("sub_tasks", []))
    ]
    print(f"  [Coordinator] Created {len(messages)} sub-tasks")
    return {"outbox": messages}

def specialist_agent(state: MessageBoardState) -> dict:
    """Processes all pending tasks in the inbox."""
    results = []
    for msg in state["outbox"]:
        if msg["message_type"] == "task":
            response = llm.invoke(f"Complete this sub-task: {msg['content']}").content
            results.append(AgentMessage(
                sender="specialist",
                recipient="coordinator",
                message_type="result",
                content=response[:200],
            ).model_dump())
    print(f"  [Specialists] Completed {len(results)} sub-tasks")
    return {"inbox": results}

def aggregator_agent(state: MessageBoardState) -> dict:
    """Combines all results into final answer."""
    results = [m["content"] for m in state["inbox"] if m["message_type"] == "result"]
    combined = "\n\n".join(f"Part {i+1}: {r}" for i, r in enumerate(results))
    final = llm.invoke(
        f"Combine these specialist results into a coherent answer for: {state['task']}\n\n{combined}"
    ).content
    return {"final_result": final}

msg_board = StateGraph(MessageBoardState)
msg_board.add_node("coordinator", coordinator_agent)
msg_board.add_node("specialists", specialist_agent)
msg_board.add_node("aggregator", aggregator_agent)
msg_board.add_edge(START, "coordinator")
msg_board.add_edge("coordinator", "specialists")
msg_board.add_edge("specialists", "aggregator")
msg_board.add_edge("aggregator", END)

msg_graph = msg_board.compile()

result = msg_graph.invoke({
    "task": "Explain the key differences between LangChain and LangGraph",
    "inbox": [],
    "outbox": [],
    "final_result": "",
})
print(f"\nFinal Result:\n{result['final_result'][:300]}...")
