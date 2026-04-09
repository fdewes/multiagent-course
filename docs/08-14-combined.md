# Days 8-14: Advanced Multi-Agent Patterns

## Quick Reference

This document provides summaries for Days 8-14. Each day builds on previous concepts.

---

# Day 8: Supervisor & Hierarchical Agents

## File: `day08/supervisor_agents.py`

### Overview
The **supervisor pattern** - the most common production multi-agent design. A supervisor agent delegates tasks to specialized worker agents.

### Architecture
```
┌─────────────┐
│ Supervisor  │ ← Decides which worker to use
└──────┬──────┘
       │ delegates
       ↓
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Researcher  │     │    Coder    │     │   Writer    │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │
       └───────────────────┼───────────────────┘
                           │ reports back
                           ↓
                   ┌─────────────┐
                   │ Supervisor  │ ← Decides next step
                   └─────────────┘
```

### Core Components

#### 1. Worker Agents
```python
WORKER_AGENTS = {
    "researcher": "You are a research specialist...",
    "coder": "You are a senior Python developer...",
    "analyst": "You are a data analyst...",
    "writer": "You are a technical writer...",
}

def make_worker(name: str, persona: str):
    def worker_fn(task: str, context: str = "") -> str:
        prompt = f"""{persona}
        Task: {task}
        Context: {context}
        Complete this task."""
        return llm.invoke(prompt).content
    return worker_fn
```

#### 2. Supervisor Decision Model
```python
class SupervisorDecision(BaseModel):
    next_worker: Literal["researcher", "coder", "analyst", "writer", "DONE"]
    task_for_worker: str
    reasoning: str
```

#### 3. Supervisor Node
```python
def supervisor_node(state: SupervisorState) -> dict:
    work_summary = "\n".join(
        f"- [{log['worker']}]: {log['result'][:150]}"
        for log in state.get("work_log", [])
    )
    
    decision = structured_llm.invoke(f"""
        Original task: {state['original_task']}
        Work so far: {work_summary}
        
        Decide which worker should act next.
        If complete, choose DONE.
    """)
    
    return {
        "next_worker": decision.next_worker,
        "pending_task": decision.task_for_worker,
        "steps": state.get("steps", 0) + 1,
    }
```

#### 4. Worker Nodes
```python
def make_worker_node(worker_name: str):
    def worker_node(state: FullSupervisorState) -> dict:
        result = workers[worker_name](
            state["pending_task"],
            context="...previous work..."
        )
        return {
            "work_log": state.get("work_log", []) + [{
                "worker": worker_name,
                "task": state["pending_task"],
                "result": result
            }]
        }
    return worker_node
```

#### 5. Routing Logic
```python
def route_to_worker(state: FullSupervisorState) -> str:
    if state.get("next_worker") == "DONE":
        return "final"
    if state.get("steps", 0) >= 5:  # Safety limit
        return "final"
    return state.get("next_worker", "final")
```

### Graph Structure
```python
builder = StateGraph(FullSupervisorState)
builder.add_node("supervisor", supervisor_node)
builder.add_node("final", final_node)

for name in WORKER_AGENTS:
    builder.add_node(name, make_worker_node(name))
    builder.add_edge(name, "supervisor")  # Back to supervisor

builder.add_edge(START, "supervisor")
builder.add_conditional_edges(
    "supervisor",
    route_to_worker,
    {name: name for name in WORKER_AGENTS} | {"final": "final"}
)
```

### Execution Flow
1. User submits task
2. Supervisor decides which worker to use
3. Worker completes task, logs result
4. Supervisor reviews, decides next step
5. Repeat until DONE or max steps
6. Final node compiles all work

### Hierarchical Pattern
```
CEO Agent (high-level)
├── Research Team Supervisor
│   ├── Web Search Agent
│   └── Document Analysis Agent
├── Engineering Team Supervisor
│   ├── Frontend Agent
│   └── Backend Agent
└── Communications Supervisor
    └── Report Writer Agent
```

### Benefits
- **Scalability**: Add more workers easily
- **Flexibility**: Supervisor adapts to task needs
- **Quality**: Specialized workers do focused work
- **Traceability**: Clear work log

---

# Day 9: RAG Agents

## File: `day09/rag_agents.py`

### Overview
**Retrieval-Augmented Generation (RAG)** with agents. Agents use document retrieval as a tool.

### Patterns

#### 1. Basic RAG Chain
```
Query → Retrieve Documents → Format Context → LLM → Answer
```

#### 2. Agentic RAG
```
Query → Agent decides → Retrieve? → Use results → Answer
```

#### 3. Corrective RAG
```
Query → Retrieve → Grade relevance → If bad, retry or fallback
```

### Key Components

#### Document Store
```python
from langchain_core.documents import Document

DOCS = [
    Document(page_content="LangGraph is a library for...", metadata={"source": "docs"}),
    Document(page_content="FAISS is Facebook's...", metadata={"source": "docs"}),
    # ...
]
```

#### Retrieval Function
```python
def retrieve_documents(query: str, k: int = 3) -> List[Document]:
    # Keyword fallback (when embeddings not available)
    query_words = set(query.lower().split())
    scored = []
    for doc in DOCS:
        score = sum(1 for w in query_words if w in doc.page_content.lower())
        if score > 0:
            scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:k]]
```

#### RAG Tool for Agent
```python
@tool
def knowledge_base_search(query: str) -> str:
    """Search knowledge base for relevant information."""
    docs = retrieve_documents(query)
    if not docs:
        return "No relevant documents found."
    return "\n\n".join(doc.page_content for doc in docs)
```

### Benefits
- **Accuracy**: Grounded in actual documents
- **Updatable**: Add new docs without retraining
- **Transparent**: Can show sources
- **Scalable**: Works with large document sets

---

# Day 10: Code Agents

## File: `day10/code_agents.py`

### Overview
Agents that **generate and execute code**.

### Pattern: Code Generation + Execution
```
Task → Generate Code → Execute → Return Result
```

### Key Features
- **Safe execution**: Sandbox environment
- **Error handling**: Retry on failure
- **Debugging**: Use error messages to fix code

### Example Flow
1. User: "Calculate first 10 Fibonacci numbers"
2. Agent generates Python code
3. Code executes in sandbox
4. Result returned to user

---

# Day 11: Evaluation

## File: `day11/evaluation.py`

### Overview
**Evaluate agent outputs** using LLM-as-judge and custom metrics.

### Patterns

#### 1. LLM-as-Judge
```python
class Evaluation(BaseModel):
    faithfulness: int = Field(ge=1, le=5)
    relevance: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
```

#### 2. Custom Tracing
Track agent execution with timestamps and metrics.

#### 3. Regression Testing
Store test cases, run periodically to catch regressions.

---

# Day 12: Production Patterns

## File: `day12/production_patterns.py`

### Overview
**Production-ready patterns**: async, human-in-the-loop, retries, persistence.

### Key Patterns

#### 1. Async Processing
```python
async def process_tasks(tasks: List[str]) -> List[str]:
    results = await asyncio.gather(*[process(t) for t in tasks])
    return results
```

#### 2. Human-in-the-Loop
```python
from langgraph.types import interrupt

def requires_approval(state):
    if needs_human_review(state):
        interrupt(data={"action": "approve_or_reject"})
```

#### 3. Retry with Backoff
```python
from tenacity import retry, stop_after_attempt

@retry(stop=stop_after_attempt(3))
def call_api():
    ...
```

#### 4. State Persistence
```python
from langgraph.checkpoint.sqlite import SqliteSaver

checkpointer = SqliteSaver.from_conn_string("sqlite:///db.sqlite")
graph = builder.compile(checkpointer=checkpointer)
```

---

# Day 13: Advanced Patterns

## File: `day13/advanced_patterns.py`

### Overview
Advanced reasoning patterns: **Debate, Critique, Self-Play, Constitutional AI**.

### Patterns

#### 1. Multi-Agent Debate
```
Pro Agent ↔ Con Agent ↔ Pro Agent (multiple rounds)
         ↓
    Synthesizer
```

#### 2. Critique Pattern
```
Generate → Critique → Revise → Critique → Final
```

#### 3. Self-Play
```
Agent plays against itself to improve (like AlphaGo)
```

#### 4. Constitutional AI
```
Generate → Critique against principles → Revise
```

---

# Day 14: Capstone

## File: `day14/capstone.py`

### Overview
**Full multi-agent research and report system** combining all patterns.

### Architecture
```
User Query
    ↓
[ORCHESTRATOR] ← Decomposes query
    ↓ (parallel)
┌──────────┐  ┌──────────┐  ┌──────────┐
│Research  │  │  Code    │  │  Analyst │
│  Agent   │  │  Agent   │  │  Agent   │
└─────┬────┘  └─────┬────┘  └─────┬────┘
      │             │             │
      └─────────────┼─────────────┘
                    ↓
           ┌─────────────┐
           │ CRITIC Agent│ ← Reviews quality
           └──────┬──────┘
                  ↓
           ┌─────────────┐
           │ WRITER Agent│ ← Final report
           └──────┬──────┘
                  ↓
           [HUMAN REVIEW] ← Optional gate
                  ↓
           Final Report
```

### Components
1. **Orchestrator**: Breaks task into subtasks
2. **Specialist Agents**: Research, code, analysis
3. **Critic**: Reviews and provides feedback
4. **Writer**: Compiles final report
5. **Human Gate**: Optional approval step

### Features
- Multi-agent coordination
- Parallel execution
- Quality review
- Human oversight
- Checkpointing

---

## Summary Table

| Day | Topic | Key Concept |
|-----|-------|-------------|
| 8 | Supervisor | Central coordinator delegates to workers |
| 9 | RAG | Agents use document retrieval |
| 10 | Code Agents | Generate and execute code |
| 11 | Evaluation | LLM-as-judge, testing |
| 12 | Production | Async, retries, persistence |
| 13 | Advanced | Debate, critique, self-play |
| 14 | Capstone | Full multi-agent system |

---

## Dependencies (All Days)
```python
from typing import TypedDict, Annotated, List, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
```

---

## Next Steps After Course
1. Build your own multi-agent system
2. Deploy with proper error handling
3. Add monitoring and logging
4. Implement human oversight
5. Set up evaluation pipeline
6. Optimize for cost and speed
