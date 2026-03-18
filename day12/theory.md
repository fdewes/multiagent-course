# Day 12 — Production Patterns

## The Gap Between Demo and Production

| Demo | Production |
|------|-----------|
| Happy path only | Handles all failures |
| No state persistence | Survives restarts |
| Single user | Concurrent users |
| No rate limits | Cost-controlled |
| No human oversight | Human-in-the-loop |
| Synchronous | Async / parallel |

## Human-in-the-Loop (HITL)

Three patterns:

### 1. Approval Gate
Pause before a risky action (delete, send, deploy).
LangGraph: `interrupt_before=["risky_node"]`

### 2. Feedback Loop
Human reviews output, provides correction, agent retries.

### 3. Escalation
Agent decides it can't handle a case, routes to human.

## State Persistence

Agent state must survive:
- Process restarts
- Machine failures
- Long-running tasks (hours)

LangGraph supports: SQLite, PostgreSQL, Redis checkpointers.
For scale: use PostgreSQL with connection pooling.

## Async is Non-Negotiable

Every LLM call takes 1-30 seconds.
With multiple agents, you cannot afford sequential execution.

LangGraph fan-out automatically parallelizes independent nodes.
For custom code: use `asyncio.gather()`.

## Retry Strategies

```
Strategy         │ When to use
─────────────────┼──────────────────────────────
Immediate retry  │ Transient network errors
Exponential back │ Rate limits, overload
Fallback model   │ Primary model unavailable
Cached response  │ Repeated identical queries
Graceful degrade │ Non-critical path failure
```

## Cost Control

1. **Model routing**: Use small model for simple steps, large for complex
2. **Prompt caching**: Cache system prompts (Anthropic, some Ollama support)
3. **Output length limits**: Set `max_tokens` always
4. **Budget tracking**: Count tokens per session, alert at threshold
5. **Batch processing**: Group similar requests

## Security

- Validate all tool inputs before execution
- Never log raw user data
- Sanitize LLM outputs before rendering (XSS)
- Rate limit per user/session
- Audit log all agent actions
