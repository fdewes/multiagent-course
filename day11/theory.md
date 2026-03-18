# Day 11 — Evaluation & Observability

## Why Evals Are the Most Important Skill

You cannot improve what you cannot measure.
Without evals:
- You don't know if your agent is regressing
- You can't compare prompt versions
- You can't justify agent changes to stakeholders
- You ship bugs silently

## Evaluation Dimensions

### Faithfulness
Does the answer stick to provided sources?
(Anti-hallucination metric)

### Relevance
Does the answer address the question?

### Completeness
Does the answer cover all required points?

### Coherence
Is the answer well-structured and logical?

### Tool Use Accuracy
Did the agent call the right tools with correct arguments?

### Trajectory Quality
Was the reasoning path efficient? Did it take unnecessary steps?

## LLM-as-Judge: Pros and Cons

**Pros**:
- Scales to any output format
- Evaluates nuanced quality
- No labeled data needed

**Cons**:
- Expensive (LLM call per eval)
- LLMs have biases (prefers verbose, self-biased)
- Non-deterministic

**Best practices**:
- Use a strong, different model as judge
- Use structured output for scores
- Calibrate against human labels
- Use reference-free AND reference-based metrics

## Observability Stack

### What to trace:
- Every LLM call: prompt, response, latency, tokens
- Every tool call: name, input, output, duration
- Agent decisions: which branch taken, why
- Error events: exception type, context

### Tools:
- **LangSmith** (LangChain native, easiest setup)
- **Phoenix (Arize)** — open source
- **Langfuse** — open source, self-hostable
- **OpenTelemetry** — vendor-neutral standard
- **Custom** — roll your own (Day 11 shows how)

## Regression Testing

Build a test suite BEFORE you refactor:
1. Collect golden examples (input → expected output)
2. Run after every change
3. Alert on score drops > threshold

This is standard CI/CD for agent systems.
