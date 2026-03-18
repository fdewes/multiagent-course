# Day 2 — LLMs as Reasoning Engines

## Structured Output (Critical for Agents)

Agents need LLMs to produce **machine-parseable output**, not prose.
Three main techniques:

### 1. JSON Mode
Ask the LLM to respond in JSON and parse the result.

### 2. Pydantic / Tool Schemas
Define a schema, the LLM fills it in — most reliable method.

### 3. Output Parsers
LangChain parsers: `JsonOutputParser`, `PydanticOutputParser`, `StrOutputParser`

## Chain-of-Thought (CoT) Prompting

```
Bad:  "What is 17 × 24?"
Good: "Think step by step: What is 17 × 24?"
```

CoT variants:
- **Zero-shot CoT**: "Let's think step by step."
- **Few-shot CoT**: Show examples of thinking steps
- **Self-consistency**: Sample multiple CoT paths, majority vote

## Prompt Engineering Patterns for Agents

| Pattern | When to use |
|---------|------------|
| Role prompting | "You are an expert Python engineer..." |
| CoT | Any multi-step reasoning |
| ReAct | Tool-using agents |
| Few-shot | Complex output formats |
| Persona | When tone/style matters |
| Critique | Ask LLM to review its own output |

## Temperature Settings
- `0.0` — Deterministic, best for tool use and structured output
- `0.3-0.7` — Creative tasks, brainstorming
- `>0.7` — Diversity needed (story generation, self-consistency sampling)

## Token Budget Awareness
Every agent loop re-sends the full message history.
Cost grows **quadratically** with steps.
Solutions:
- Summarize old observations
- Limit tool output size
- Use smaller model for early steps
