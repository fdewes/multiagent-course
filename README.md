# 14-Day Intensive Multi-Agent Systems Course

> **Stack**: Python 3.11+, LangChain, LangGraph, Ollama (qwen2.5:27b @ 10.10.1.249)
> **Philosophy**: Vendor-neutral patterns. The concepts transfer to OpenAI, Anthropic, Mistral, Gemini, etc.

## Setup

```bash
pip install langchain langchain-community langgraph langchain-ollama \
            chromadb faiss-cpu sentence-transformers \
            httpx rich pydantic tenacity
```

Ollama base URL used throughout: `http://10.10.1.249:11434`
Model: `qwen2.5:27b`

---

## Course Map

| Day | Topic | Key Concepts |
|-----|-------|-------------|
| 01 | Agent Foundations | What is an agent, ReAct loop, basic tool use |
| 02 | LLMs as Reasoning Engines | Prompt engineering, structured output, chain-of-thought |
| 03 | Tools & Function Calling | Tool design, schema, tool execution loop |
| 04 | Memory Systems | Short-term, long-term, episodic, semantic, procedural |
| 05 | Agent Architectures | ReAct, Plan-Execute, Reflexion, LATS, CoT-SC |
| 06 | LangGraph Deep Dive | State graphs, conditional edges, cycles, checkpointing |
| 07 | Multi-Agent Coordination | Message passing, shared state, agent protocols |
| 08 | Supervisor & Hierarchical Agents | Orchestrator patterns, routing, delegation |
| 09 | RAG for Agents | Vector stores, hybrid retrieval, agentic RAG |
| 10 | Code Agents | Code generation, execution, self-debugging |
| 11 | Evaluation & Observability | Metrics, tracing, LLM-as-judge, regression testing |
| 12 | Production Patterns | Async, retries, state persistence, human-in-the-loop |
| 13 | Advanced Patterns | Debate, critique, self-play, multi-agent debate |
| 14 | Capstone | Full multi-agent research + writing system |

---

## Agent Type Taxonomy (Industry 2024-2025)

### By Architecture
- **ReAct Agents** — Reasoning + Acting interleaved (most common)
- **Plan-and-Execute** — Plan upfront, execute steps (LangGraph)
- **Reflexion Agents** — Self-critique and retry loops
- **LATS (Language Agent Tree Search)** — Monte Carlo Tree Search over actions
- **OpenAgents / CodeAct** — Agents that write and run code

### By Role in Multi-Agent Systems
- **Orchestrator / Supervisor** — Routes tasks, manages sub-agents
- **Worker / Specialist** — Domain-specific agents (coder, researcher, analyst)
- **Critic / Evaluator** — Reviews and scores other agents' output
- **Planner** — Decomposes goals into sub-tasks
- **Memory Agent** — Manages retrieval and storage
- **Tool Agent** — Wraps external APIs / databases

### By Interaction Pattern
- **Sequential** — A→B→C pipeline
- **Parallel** — Fan-out, collect results
- **Hierarchical** — Tree of agents
- **Peer-to-Peer** — Agents negotiate directly
- **Debate / Adversarial** — Agents argue positions

### What Industry Wants Right Now (2025)
1. **Reliable tool use** — structured output, JSON mode, retry logic
2. **Human-in-the-loop** — approval gates, interrupt/resume
3. **Observability** — full trace of every LLM call and tool use
4. **State management** — resumable, persistent agent state
5. **Evals** — automated quality measurement, not vibes
6. **Cost control** — caching, model routing (small vs large)
7. **Safety / guardrails** — input/output validation
8. **RAG pipelines** — grounded, cite-able responses
