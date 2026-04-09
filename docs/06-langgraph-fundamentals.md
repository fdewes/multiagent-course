# Day 6: LangGraph Fundamentals Documentation

## File: `day06/langgraph_fundamentals.py`

## Overview
Introduces **LangGraph** - a library for building stateful, multi-actor applications with LLMs. Shows how to create graphs with nodes, edges, conditional routing, and checkpointing.

## What is LangGraph?

LangGraph extends LangChain with:
- **State management**: Persistent state across invocations
- **Cycles**: Loops for iterative processes
- **Checkpointing**: Save/restore workflow state
- **Streaming**: Real-time output
- **Human-in-the-loop**: Interrupt and resume

## Core Concepts

### State Graph
```
[START] → [Node 1] → [Node 2] → [END]
```

### Components
- **Nodes**: Functions that process state
- **Edges**: Connections between nodes
- **State**: Typed dictionary passed between nodes
- **Conditional edges**: Route based on state

---

## Pattern 1: Minimal Chat Graph

### Purpose
Simplest LangGraph example - a basic chatbot.

### State Definition
```python
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class SimpleState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
```

**Key Points:**
- `TypedDict`: Type-safe state
- `Annotated[..., add_messages]`: Special reducer that appends messages
- `messages`: List of message objects

### Node Definition
```python
def chat_node(state: SimpleState) -> dict:
    """Process messages and return response."""
    response = simple_llm.invoke(state["messages"])
    return {"messages": [response]}  # add_messages will append
```

**Node Contract:**
- Input: `state` (current state)
- Output: `dict` (state updates)
- Returns only changes, not full state

### Graph Construction
```python
from langgraph.graph import StateGraph, START, END

builder = StateGraph(SimpleState)

# Add node
builder.add_node("chat", chat_node)

# Add edges
builder.add_edge(START, "chat")    # START → chat
builder.add_edge("chat", END)      # chat → END

# Compile
graph = builder.compile()
```

### Invocation
```python
from langchain_core.messages import HumanMessage

result = graph.invoke({
    "messages": [HumanMessage(content="What is 2 + 2?")]
})

print("Response:", result["messages"][-1].content)
```

### Graph Structure
```
[START] → [chat] → [END]
```

---

## Pattern 2: ReAct Agent with ToolNode

### Purpose
Agent that can call tools, with automatic routing.

### Tools Definition
```python
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    """Search for information about a topic."""
    facts = {...}
    for key, value in facts.items():
        if key in query.lower():
            return value
    return "No specific information found."

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    import re
    if re.match(r'^[\d\s\+\-\*\/\.\(\)]+$', expression):
        return str(eval(expression))
    return "Invalid expression"

tools = [search, calculator]
```

### ToolNode
```python
from langgraph.prebuilt import ToolNode

tool_node = ToolNode(tools)
```

**ToolNode automatically:**
- Parses tool calls from LLM response
- Executes requested tools
- Returns ToolMessage with results

### Agent LLM with Tool Binding
```python
agent_llm = get_llm(temperature=0.0).bind_tools(tools)
```

**bind_tools:**
- Adds tool schemas to LLM
- Enables structured tool calling
- LLM outputs tool_call messages

### State with Step Counter
```python
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    step_count: int  # Track iterations
```

### Agent Node
```python
def agent_node(state: AgentState) -> dict:
    response = agent_llm.invoke(state["messages"])
    return {
        "messages": [response],
        "step_count": state.get("step_count", 0) + 1
    }
```

### Conditional Edge (Routing)
```python
from typing import Literal

def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """Route: if tool calls → execute, else → end."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "end"
```

**Logic:**
- Check last message for tool calls
- If has tool calls → route to "tools" node
- Else → route to END

### Graph Construction
```python
builder = StateGraph(AgentState)

# Add nodes
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)

# Add edges
builder.add_edge(START, "agent")  # Always start with agent
builder.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", "end": END}  # Route mapping
)
builder.add_edge("tools", "agent")  # After tools → back to agent

# Compile
react_graph = builder.compile()
```

### Graph Structure
```
[START] → [agent] ──(has tool calls)──→ [tools] → [agent] ──(loop)──
                 │
                 └──(no tool calls)──→ [END]
```

### Invocation
```python
result = react_graph.invoke({
    "messages": [HumanMessage(content="How tall is the Eiffel Tower in feet?")],
    "step_count": 0,
})

print(f"Steps taken: {result['step_count']}")
print(f"Final answer: {result['messages'][-1].content}")
```

### Example Execution Flow
```
1. User: "How tall is the Eiffel Tower in feet?"
2. Agent: Calls search("Eiffel Tower height")
3. Tool: Returns "330 meters"
4. Agent: Calls calculator("330 * 3.28084")
5. Tool: Returns "1082.6772"
6. Agent: Final answer "≈1083 feet"
```

---

## Pattern 3: Checkpointing (Persistent State)

### Purpose
Maintain state across multiple invocations.

### Checkpointer
```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
checkpointed_graph = builder.compile(checkpointer=checkpointer)
```

**Checkpointer types:**
- `MemorySaver`: In-memory (lost on restart)
- `SqliteSaver`: SQLite database
- `PostgresSaver`: PostgreSQL
- `RedisSaver`: Redis

### Thread Configuration
```python
thread = {"configurable": {"thread_id": "session-001"}}
```

**thread_id:**
- Isolates state per conversation/user
- Multiple threads can share same graph

### Multi-Turn Conversation
```python
# Turn 1
result1 = checkpointed_graph.invoke(
    {"messages": [HumanMessage(content="My name is Felix. What is Python?")]},
    config=thread
)
print("Turn 1:", result1["messages"][-1].content)

# Turn 2 - remembers Turn 1!
result2 = checkpointed_graph.invoke(
    {"messages": [HumanMessage(content="What did I just ask? What's my name?")]},
    config=thread
)
print("Turn 2:", result2["messages"][-1].content)
# → "You asked about Python. Your name is Felix."
```

### Use Cases
- Chatbots with conversation history
- Long-running workflows
- Human-in-the-loop approval
- Resume after interruption

---

## Pattern 4: Multi-Branch Routing

### Purpose
Route queries to specialized handlers.

### State Definition
```python
class RouterState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    route: str      # Classification result
    result: str     # Final answer
```

### Classifier Node
```python
def classify_query(state: RouterState) -> dict:
    """Classify query to determine handler."""
    question = state["messages"][-1].content
    
    structured = get_llm().with_structured_output({
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": ["math", "factual", "creative", "other"]
            }
        },
        "required": ["category"]
    })
    
    result = structured.invoke(f"Classify: {question}")
    return {"route": result.get("category", "other")}
```

### Handler Nodes
```python
def math_handler(state: RouterState) -> dict:
    response = llm.invoke(f"Solve math: {state['messages'][-1].content}")
    return {"result": response.content, "messages": [AIMessage(content=response.content)]}

def factual_handler(state: RouterState) -> dict:
    response = llm.invoke(f"Factual answer: {state['messages'][-1].content}")
    return {"result": response.content, "messages": [AIMessage(content=response.content)]}

def creative_handler(state: RouterState) -> dict:
    hot = get_llm(temperature=0.8)  # More creative
    response = hot.invoke(f"Creative: {state['messages'][-1].content}")
    return {"result": response.content, "messages": [AIMessage(content=response.content)]}

def other_handler(state: RouterState) -> dict:
    response = llm.invoke(state["messages"][-1].content)
    return {"result": response.content, "messages": [AIMessage(content=response.content)]}
```

### Router Function
```python
def route_to_handler(state: RouterState) -> str:
    """Route based on classification."""
    return state["route"]  # Returns node name
```

### Graph Construction
```python
router_builder = StateGraph(RouterState)

# Add nodes
router_builder.add_node("classify", classify_query)
router_builder.add_node("math", math_handler)
router_builder.add_node("factual", factual_handler)
router_builder.add_node("creative", creative_handler)
router_builder.add_node("other", other_handler)

# Add edges
router_builder.add_edge(START, "classify")

# Conditional routing
router_builder.add_conditional_edges(
    "classify",
    route_to_handler,
    {
        "math": "math",
        "factual": "factual",
        "creative": "creative",
        "other": "other"
    }
)

# All handlers end
for node in ["math", "factual", "creative", "other"]:
    router_builder.add_edge(node, END)

router_graph = router_builder.compile()
```

### Graph Structure
```
[START] → [classify]
              │
         ┌────┼────┬─────────┬─────────┐
         ↓    ↓    ↓         ↓         ↓
       [math][factual][creative][other]
         ↓    ↓    ↓         ↓
        [END] [END] [END]   [END]
```

### Invocation
```python
test_queries = [
    "What is 47 × 23?",        # → math
    "Who invented telephone?",  # → factual
    "Write a haiku",           # → creative
]

for q in test_queries:
    result = router_graph.invoke({
        "messages": [HumanMessage(content=q)],
        "route": "",
        "result": "",
    })
    print(f"Q: {q}")
    print(f"Route: {result['route']}")
    print(f"A: {result['result']}")
```

---

## LangGraph API Reference

### StateGraph Methods
```python
builder = StateGraph(StateType)

builder.add_node(name: str, node_fn: callable)
builder.add_edge(from_node: str, to_node: str)
builder.add_conditional_edges(
    from_node: str,
    condition: callable,
    path_map: dict  # {value: node_name}
)

graph = builder.compile(checkpointer=checkpointer)
```

### Special Constants
- `START`: Entry point
- `END`: Exit point

### State Updates
```python
# Node returns dict of updates
return {"key": new_value}

# Annotated fields use reducers
messages: Annotated[list, add_messages]  # Appends
```

---

## Best Practices

### 1. Minimal State Updates
```python
# ✅ Return only changes
return {"messages": [response]}

# ❌ Don't return full state
return {"messages": state["messages"] + [response]}
```

### 2. Use TypedDict for State
```python
# ✅ Type-safe
class State(TypedDict):
    messages: list[str]

# ❌ Plain dict
state = {}
```

### 3. Keep Nodes Focused
```python
# ✅ Single responsibility
def classify(state): ...
def handle(state): ...

# ❌ Do everything in one node
def process(state):
    classify()
    handle()
    summarize()
```

### 4. Add Checkpointing Early
```python
# Plan for persistence from start
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)
```

---

## Dependencies
```python
from typing import TypedDict, Annotated, Literal
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
```

---

## Next Steps
- Day 7: Multi-agent coordination
- Day 8: Supervisor patterns
