# Day 7: Multi-Agent Coordination Documentation

## File: `day07/multiagent_coordination.py`

## Overview
Explores **multi-agent coordination patterns**: sequential pipelines, parallel fan-out, and message passing protocols. Shows how to orchestrate multiple specialized agents.

---

## Pattern 1: Sequential Pipeline

### Concept
Agents work in sequence, each building on previous work.

### Flow Diagram
```
[START] → [Research] → [Outline] → [Writer] → [Editor] → [END]
```

### Use Case
Content creation workflow: research → outline → draft → edit.

### State Definition
```python
class PipelineState(TypedDict):
    topic: str           # Input
    research_notes: str  # Research agent output
    outline: str         # Outline agent output
    draft: str           # Writer output
    final_article: str   # Editor output
```

### Agent Functions

#### Research Agent
```python
def research_agent(state: PipelineState) -> dict:
    """Gather facts and information."""
    prompt = f"""Research topic: {state['topic']}
    Provide 5-7 key facts."""
    result = llm.invoke(prompt).content
    return {"research_notes": result}
```

#### Outline Agent
```python
def outline_agent(state: PipelineState) -> dict:
    """Create structure from research."""
    prompt = f"""Create outline from research:
    Research: {state['research_notes']}"""
    result = llm.invoke(prompt).content
    return {"outline": result}
```

#### Writing Agent
```python
def writing_agent(state: PipelineState) -> dict:
    """Write article from outline."""
    prompt = f"""Write article:
    Outline: {state['outline']}
    Research: {state['research_notes']}"""
    result = llm.invoke(prompt).content
    return {"draft": result}
```

#### Editor Agent
```python
def editor_agent(state: PipelineState) -> dict:
    """Polish final draft."""
    prompt = f"""Edit draft: {state['draft']}"""
    result = llm.invoke(prompt).content
    return {"final_article": result}
```

### Graph Construction
```python
pipeline = StateGraph(PipelineState)

# Add all nodes
for node_fn in [research_agent, outline_agent, writing_agent, editor_agent]:
    pipeline.add_node(node_fn.__name__, node_fn)

# Chain edges
pipeline.add_edge(START, "research_agent")
pipeline.add_edge("research_agent", "outline_agent")
pipeline.add_edge("outline_agent", "writing_agent")
pipeline.add_edge("writing_agent", "editor_agent")
pipeline.add_edge("editor_agent", END)

pipeline_graph = pipeline.compile()
```

### Invocation
```python
result = pipeline_graph.invoke({
    "topic": "The rise of multi-agent AI systems in 2025"
})
print(result['final_article'])
```

### Benefits
- **Specialization**: Each agent focused on one task
- **Quality**: Review at each stage
- **Modularity**: Easy to modify individual agents
- **Traceability**: Clear workflow

### Trade-offs
- Sequential = slower
- Errors propagate downstream
- No feedback loops

---

## Pattern 2: Parallel Fan-Out + Aggregation

### Concept
Multiple agents work simultaneously, then results are combined.

### Flow Diagram
```
           ┌──────────────┐
[START] ───┤  Technical   ├────┐
           └──────────────┘    │
           ┌──────────────┐    │    ┌───────────┐
           ┤  Business    ├────┼────┤ Synthesis ├──→ [END]
           └──────────────┘    │    └───────────┘
           ┌──────────────┐    │
           ┤  Ethics      ├────┘
           └──────────────┘
```

### Use Case
Multi-perspective analysis: technical, business, ethical viewpoints.

### State Definition
```python
class ParallelState(TypedDict):
    question: str              # Input
    perspective_technical: str # Technical expert output
    perspective_business: str  # Business expert output
    perspective_ethical: str   # Ethics expert output
    synthesis: str             # Combined analysis
```

### Expert Agents

```python
def technical_expert(state: ParallelState) -> dict:
    response = llm.invoke(
        f"As a technical expert, analyze: {state['question']}"
    ).content
    return {"perspective_technical": response}

def business_expert(state: ParallelState) -> dict:
    response = llm.invoke(
        f"As a business strategist, analyze: {state['question']}"
    ).content
    return {"perspective_business": response}

def ethics_expert(state: ParallelState) -> dict:
    response = llm.invoke(
        f"As an AI ethics researcher, analyze: {state['question']}"
    ).content
    return {"perspective_ethical": response}
```

### Synthesizer Agent
```python
def synthesizer(state: ParallelState) -> dict:
    prompt = f"""Synthesize perspectives on: {state['question']}

Technical: {state['perspective_technical']}
Business: {state['perspective_business']}
Ethical: {state['perspective_ethical']}

Provide balanced synthesis."""
    return {"synthesis": llm.invoke(prompt).content}
```

### Graph Construction
```python
parallel = StateGraph(ParallelState)

# Add nodes
parallel.add_node("technical", technical_expert)
parallel.add_node("business", business_expert)
parallel.add_node("ethics", ethics_expert)
parallel.add_node("synthesize", synthesizer)

# Fan-out: START → all experts
parallel.add_edge(START, "technical")
parallel.add_edge(START, "business")
parallel.add_edge(START, "ethics")

# Fan-in: all experts → synthesizer
parallel.add_edge("technical", "synthesize")
parallel.add_edge("business", "synthesize")
parallel.add_edge("ethics", "synthesize")
parallel.add_edge("synthesize", END)

parallel_graph = parallel.compile()
```

### Invocation
```python
result = parallel_graph.invoke({
    "question": "Should AI agents have internet access?",
    "perspective_technical": "",
    "perspective_business": "",
    "perspective_ethical": "",
    "synthesis": "",
})

print(f"Technical: {result['perspective_technical']}")
print(f"Business: {result['perspective_business']}")
print(f"Ethics: {result['perspective_ethical']}")
print(f"Synthesis: {result['synthesis']}")
```

### Benefits
- **Speed**: Parallel execution
- **Diversity**: Multiple perspectives
- **Balance**: Synthesizer combines viewpoints

### Trade-offs
- All experts must complete before synthesis
- State must include all possible outputs

---

## Pattern 3: Structured Message Passing

### Concept
Agents communicate via structured messages on a message board.

### Flow Diagram
```
[START] → [Coordinator] ──(task messages)──→ [Specialists]
                                       ↓
                              (result messages)
                                       ↓
                                  [Aggregator] → [END]
```

### Message Protocol
```python
from pydantic import BaseModel

class AgentMessage(BaseModel):
    sender: str
    recipient: str
    message_type: str  # task | result | question | error
    content: str
    priority: int = 1
```

### State Definition
```python
class MessageBoardState(TypedDict):
    inbox: List[dict]    # Messages to process
    outbox: List[dict]   # Messages sent
    task: str            # Original task
    final_result: str    # Combined result
```

### Coordinator Agent
```python
def coordinator_agent(state: MessageBoardState) -> dict:
    """Break task into sub-tasks."""
    structured_llm = llm.with_structured_output({...})
    result = structured_llm.invoke(
        f"Break into sub-tasks: {state['task']}"
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
    
    return {"outbox": messages}
```

### Specialist Agent
```python
def specialist_agent(state: MessageBoardState) -> dict:
    """Process tasks from outbox."""
    results = []
    for msg in state["outbox"]:
        if msg["message_type"] == "task":
            response = llm.invoke(
                f"Complete: {msg['content']}"
            ).content
            results.append(AgentMessage(
                sender="specialist",
                recipient="coordinator",
                message_type="result",
                content=response[:200],
            ).model_dump())
    
    return {"inbox": results}
```

### Aggregator Agent
```python
def aggregator_agent(state: MessageBoardState) -> dict:
    """Combine all results."""
    results = [m["content"] for m in state["inbox"] 
               if m["message_type"] == "result"]
    
    combined = "\n\n".join(
        f"Part {i+1}: {r}" for i, r in enumerate(results)
    )
    
    final = llm.invoke(
        f"Combine results for: {state['task']}\n\n{combined}"
    ).content
    
    return {"final_result": final}
```

### Graph Construction
```python
msg_board = StateGraph(MessageBoardState)
msg_board.add_node("coordinator", coordinator_agent)
msg_board.add_node("specialists", specialist_agent)
msg_board.add_node("aggregator", aggregator_agent)

msg_board.add_edge(START, "coordinator")
msg_board.add_edge("coordinator", "specialists")
msg_board.add_edge("specialists", "aggregator")
msg_board.add_edge("aggregator", END)

msg_graph = msg_board.compile()
```

### Benefits
- **Scalability**: Easy to add more specialists
- **Structure**: Clear message protocol
- **Flexibility**: Can extend message types

### Trade-offs
- More complex state management
- Message passing overhead

---

## Pattern Comparison

| Pattern | Speed | Complexity | Best For |
|---------|-------|------------|----------|
| Pipeline | Slow (sequential) | Low | Linear workflows |
| Fan-Out | Fast (parallel) | Medium | Multi-perspective |
| Message Board | Medium | High | Dynamic coordination |

## When to Use Each

### Pipeline
✅ Clear sequential steps
✅ Each step depends on previous
✅ Quality control at each stage

### Fan-Out
✅ Need diverse perspectives
✅ Independent analyses
✅ Fast turnaround needed

### Message Board
✅ Dynamic task allocation
✅ Many specialists
✅ Complex coordination

---

## Dependencies
```python
from typing import TypedDict, Annotated, List
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel
import asyncio
```

---

## Next Steps
- Day 8: Supervisor agent patterns
- Day 9: RAG with agents
