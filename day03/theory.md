# Day 3 — Tools & Function Calling

## Tool Design Principles

A tool is a **typed function** the agent can invoke.
Good tools are:
- **Atomic** — one thing, done well
- **Idempotent** — safe to call multiple times
- **Well-documented** — the docstring IS the tool description the LLM reads
- **Typed** — input/output schemas prevent hallucinated arguments

## Tool Anatomy
```python
@tool
def get_weather(city: str, unit: Literal["celsius", "fahrenheit"] = "celsius") -> str:
    """
    Get the current weather for a city.

    Args:
        city: The city name, e.g. "Berlin"
        unit: Temperature unit
    Returns:
        A string describing current weather conditions
    """
    ...
```

## Tool Categories in Real Systems

| Category | Examples |
|----------|---------|
| **Search** | Web search, vector DB search, SQL |
| **Read** | File read, API GET, document parse |
| **Write** | File write, API POST, database insert |
| **Compute** | Calculator, code execution, data analysis |
| **Communication** | Send email, Slack message, webhook |
| **State** | Memory read/write, session management |

## Tool Selection Strategies

1. **Implicit** — LLM picks from all tools (works up to ~20 tools)
2. **Retrieval-based** — embed tool descriptions, retrieve top-k relevant tools
3. **Hierarchical** — category tool that delegates to sub-tools

## Error Handling in Tools

Tools WILL fail. Design for it:
- Return error strings, not raise exceptions (agent can read them)
- Include retry logic inside the tool
- Validate inputs before making expensive calls

## Function Calling vs. ReAct

| Feature | Function Calling | ReAct (text) |
|---------|-----------------|--------------|
| Reliability | High (structured) | Medium (parsing) |
| Model support | Needs native support | Works with any LLM |
| Transparency | Less visible | Full chain-of-thought |
