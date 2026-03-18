# Day 7 — Multi-Agent Coordination

## Why Multi-Agent?

Single agents hit limits:
- **Context window**: one agent can't hold everything
- **Specialization**: a generalist is worse than an expert
- **Parallelism**: one agent is sequential; N agents can work simultaneously
- **Reliability**: agents can check each other's work

## Communication Patterns

### Pattern 1: Message Passing
Agents communicate via structured messages.
```
Agent A → Message → Agent B → Message → Agent C
```

### Pattern 2: Shared State
All agents read/write a common state object (LangGraph's model).
```
State ← Agent A writes
State → Agent B reads
State ← Agent B writes
```

### Pattern 3: Blackboard System
A shared workspace where agents post and consume tasks.
Classic AI pattern from 1980s, still used in complex systems.

## Agent Protocols

In production, define a **protocol** for agent messages:

```python
class AgentMessage(BaseModel):
    sender: str          # Which agent sent this
    recipient: str       # Which agent should receive it
    message_type: str    # "task", "result", "question", "error"
    content: str
    metadata: dict = {}
```

## Network Topologies

### Star (Supervisor)
```
        Supervisor
       /     |     \
  Agent A  Agent B  Agent C
```
- Supervisor routes, delegates, collects
- Simple to reason about

### Pipeline
```
A → B → C → D
```
- Sequential specialization
- Each agent enriches the output

### Mesh (Peer-to-Peer)
```
A ↔ B ↔ C
↕       ↕
D ↔ E ↔ F
```
- Agents can communicate with any other
- Complex, hard to debug

### Hierarchical
```
     CEO Agent
    /          \
 Team A       Team B
 /   \        /   \
A1   A2      B1   B2
```
- Scales well for large systems

## Inter-Agent Communication in LangGraph

Use `Command` objects to transfer control between agents:
```python
from langgraph.types import Command

def agent_a(state):
    return Command(
        goto="agent_b",
        update={"messages": [...]}
    )
```
