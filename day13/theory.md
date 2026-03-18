# Day 13 — Advanced Patterns

## Debate / Multi-Agent Debate

Du et al. (2023): Multiple agents debate a topic, then a judge evaluates.
Result: better factual accuracy and reasoning than single agent.

Key insight: adversarial pressure forces agents to justify their reasoning.

## Constitutional AI

Anthropic's technique (Bai et al. 2022):
1. Generate initial response
2. Critique against principles
3. Revise to fix violations
4. Repeat

Vendor-neutral version: any LLM can be the critic.
Principles are just text — you control them.

## Self-Play

Classic RL technique adapted for LLMs:
- Agent generates solution
- Agent critiques its own solution
- Agent improves
- Repeat until satisfied

Bonus: pair with external evaluator (e.g., test runner) for verified improvement.

## Mixture of Experts (MoE) at Agent Level

Router → specialist agents (not model weights MoE, but agent architecture MoE).

Benefits:
- Specialized prompts per domain
- Different temperatures per task type
- Independent caching per expert
- Easy to add/remove experts

## Critic-as-a-Service

Design a standalone critic agent:
- Input: (task, response)
- Output: {score, issues, improvements}
- Use in: RAG, code gen, report writing, any chain

This is reusable across all your agent systems.

## What Actually Works in Production (2025)

Based on industry reports and research:

1. **Structured output everywhere** — JSON mode > free text
2. **Short context wins** — less context = better reasoning, lower cost
3. **Explicit reasoning** — "Think step by step" still works
4. **Verification loops** — self-check or external check
5. **Human gates** — for consequential actions, always have HITL
6. **Eval-driven development** — write evals before writing agents
7. **Small models for routing** — don't burn big model budget on routing
8. **Observability from day one** — not an afterthought
