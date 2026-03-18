# Day 5 — Agent Architectures

## Architecture Comparison

### 1. ReAct (Reason + Act)
```
Thought → Action → Observation → Thought → ...
```
- **Best for**: Tool-using QA, research tasks
- **Weakness**: No upfront planning, can get lost

### 2. Plan-and-Execute
```
Planner → [Step 1, Step 2, Step 3]
               ↓       ↓       ↓
            Execute Execute Execute
               ↓
            Replanner (if needed)
```
- **Best for**: Complex multi-step tasks with dependencies
- **Weakness**: Plan can be wrong; replanning is expensive

### 3. Reflexion (Shinn et al. 2023)
```
Attempt → Evaluate → Reflect → Improve → Attempt again
```
- **Best for**: Tasks with clear success criteria (coding, math)
- **Key insight**: Verbal reinforcement (text critique) > just retrying

### 4. LATS (Liu et al. 2023)
Language Agent Tree Search — Monte Carlo Tree Search over reasoning paths.
- **Best for**: Hard reasoning, code generation benchmarks
- **Weakness**: Very expensive (many LLM calls per query)

### 5. CoT-SC (Self-Consistency)
Sample N reasoning paths at high temperature, majority-vote on answer.
- **Best for**: Math, factual QA
- **Weakness**: N× cost

### 6. Skeleton-of-Thought
Outline first, then fill in parallel.
- **Best for**: Long-form content generation

## Choosing an Architecture

```
Is the task well-defined with clear steps?
  ├── Yes → Plan-and-Execute
  └── No  → Can we evaluate correctness?
              ├── Yes (code, math) → Reflexion
              └── No (open-ended) → ReAct
```

## The Replanner Pattern
Critical for production: after each step, re-evaluate:
- Is the plan still valid?
- Do we have new information that changes the plan?
- Are we done early?
