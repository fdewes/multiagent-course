# Day 6 — LangGraph Deep Dive

## What is LangGraph?

LangGraph models agent behavior as a **directed graph** where:
- **Nodes** = functions/agents that transform state
- **Edges** = transitions between nodes (conditional or fixed)
- **State** = a typed dict passed through the graph

Key properties:
- **Cycles** — unlike simple chains, graphs can loop
- **Checkpointing** — save/restore state at any node
- **Streaming** — emit intermediate results
- **Human-in-the-loop** — pause and wait for approval

## Core Concepts

### State
```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # append-only list
    tools_called: int
    final_answer: str
```

`Annotated[list, add_messages]` means: when updating this key, append (not replace).

### Nodes
```python
def my_node(state: AgentState) -> dict:
    # Read state, do work, return partial state update
    return {"messages": [AIMessage(content="Hello")]}
```

### Edges
- **Fixed**: always go from A to B
- **Conditional**: function decides which node to go to next

```python
def route(state: AgentState) -> str:
    if state["final_answer"]:
        return "end"
    return "continue"
```

### Special Nodes
- `START` — entry point
- `END` — terminal node

## The `ToolNode`
LangGraph provides a pre-built `ToolNode` that:
1. Reads tool calls from the last AI message
2. Executes the tools
3. Returns tool result messages

## Checkpointing
```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)

# Run with thread_id to maintain state across invocations
graph.invoke(input, config={"configurable": {"thread_id": "123"}})
```

## When to use LangGraph vs. LangChain
| Task | Use |
|------|-----|
| Simple chain (A→B→C) | LangChain LCEL |
| Agent with tools | LangGraph |
| Multi-agent system | LangGraph |
| Human-in-the-loop | LangGraph |
| Cycles/retry logic | LangGraph |
