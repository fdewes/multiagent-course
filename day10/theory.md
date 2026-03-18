# Day 10 — Code Agents

## Why Code Agents?

Code is the ultimate multi-purpose tool:
- Can replace most specialized tools (math, data analysis, scraping, file I/O)
- Self-verifiable: run the code, see if it works
- Composable: generate + test + fix in a loop

## Code Agent Architectures

### 1. Generate → Execute → Return
Simple. No feedback loop. Fails silently.

### 2. Generate → Execute → Verify → Fix (Reflexion)
Iterative. Self-healing. The standard for production.

### 3. TDD Agent
Generate tests first, then generate code to pass them.
Highest quality but 2× LLM calls.

### 4. CodeAct
Agent writes code as its "actions" (not JSON tool calls).
Pros: flexible, powerful. Cons: hard to sandbox.

## Sandboxing (CRITICAL for Production)

**Never** run LLM-generated code in your main process.

| Approach | Security | Setup |
|----------|---------|-------|
| `subprocess` | Medium | Easy |
| Docker container | High | Medium |
| E2B (cloud sandbox) | High | Easy (managed) |
| Firecracker microVM | Very High | Hard |
| WASM | Medium | Medium |

Always set:
- CPU limits
- Memory limits
- Network isolation (no internet access from sandbox)
- Timeout

## Common Failures

1. **Import errors** — model uses unavailable packages
2. **Infinite loops** — no timeout = frozen agent
3. **No output** — code runs but doesn't print anything
4. **Scope errors** — variables not in expected scope
5. **Security** — model tries to read files, call network

## Industry Examples
- GitHub Copilot Workspace
- Devin (Cognition AI)
- OpenHands (formerly OpenDevin)
- Claude's code execution tool
- ChatGPT's Advanced Data Analysis
