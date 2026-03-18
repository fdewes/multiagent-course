# Day 8 — Supervisor & Hierarchical Agents

## The Supervisor Pattern

The most common multi-agent pattern in production.

```
User → Supervisor → Worker A → Supervisor → Worker B → Supervisor → Final
```

The supervisor:
1. Receives the original task
2. Decides which worker to delegate to
3. Reviews the result
4. Either delegates to another worker or concludes

## Why Supervisor > Hardcoded Pipelines

| Feature | Hardcoded Pipeline | Supervisor |
|---------|-------------------|------------|
| Worker order | Fixed | Dynamic |
| Handles unexpected | No | Yes |
| Adaptable | No | Yes |
| Debuggable | Easy | Medium |

## Implementation Choices

### Option A: Structured Routing (Recommended)
Supervisor outputs a structured decision: `{next_worker, task_for_worker}`.
Most reliable because it's type-safe.

### Option B: Natural Language Routing
Supervisor's response is parsed to determine the next worker.
More flexible but can fail to parse.

## The DONE Condition

The supervisor must know when to stop. Options:
1. **Explicit DONE** — One of the "workers" is `"DONE"`, supervisor selects it
2. **Evaluator** — Separate agent grades completeness
3. **Max steps** — Hard limit (safety net)

## Hierarchical Agents

For large-scale systems, supervisors can supervise supervisors:

```
Top-level Supervisor
├── Research Supervisor
│   ├── Web Search Agent
│   └── DB Query Agent
├── Engineering Supervisor
│   ├── Frontend Agent
│   └── Backend Agent
└── QA Supervisor
    ├── Test Writer Agent
    └── Bug Reporter Agent
```

In LangGraph: use **subgraphs** — each supervisor is a compiled graph used as a node.

## Inter-Agent Communication

Agents share state through the graph state dict.
For complex routing: use `Command(goto=..., update=...)` objects.
