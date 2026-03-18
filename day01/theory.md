# Day 1 — Agent Foundations

## What Is an Agent?

An **agent** is a system that:
1. Perceives an **environment** (text, tool results, memory)
2. Reasons about what to do next (via an LLM)
3. Takes **actions** (tool calls, API calls, writes)
4. Observes the result and loops until goal is met

The key insight: an LLM alone is *stateless*. An **agent** wraps it with a loop.

```
┌────────────────────────────────────────┐
│              AGENT LOOP                │
│                                        │
│  ┌──────┐   think   ┌──────────────┐  │
│  │ LLM  │ ─────────▶│ Tool / Action│  │
│  └──────┘           └──────┬───────┘  │
│      ▲                     │ result   │
│      └─────────────────────┘          │
│              until FINAL              │
└────────────────────────────────────────┘
```

## The ReAct Pattern (Yao et al. 2022)

**Re**asoning + **Act**ing.  The LLM alternates between:
- `Thought:` — chain-of-thought reasoning about what to do
- `Action:` — a structured tool call
- `Observation:` — the tool result injected back

```
Thought: I need to find the population of Berlin.
Action: search("Berlin population 2024")
Observation: Berlin has ~3.6 million people.
Thought: I have enough to answer.
Final Answer: Berlin's population is approximately 3.6 million.
```

This pattern is the backbone of almost every agent framework.

## Why ReAct Works

- LLMs are better at deciding *one step at a time* than planning everything upfront
- Tool results ground the reasoning in facts (reduces hallucination)
- The loop is transparent and debuggable

## Limitations
- Latency: every step = an LLM call
- Token waste: full history re-sent each step
- Loops: agents can get stuck ("chain of thought loops")

## Key Terms
| Term | Meaning |
|------|---------|
| Tool | A function the agent can call (search, calculator, API) |
| Action | The agent's decision of which tool + args |
| Observation | The tool's return value |
| Trajectory | The full sequence of thoughts/actions/observations |
| Agent Scratchpad | The working memory (usually the message history) |
