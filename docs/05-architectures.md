# Day 5: Agent Architectures Documentation

## File: `day05/architectures.py`

## Overview
Explores three powerful **agent architecture patterns**: Plan-and-Execute, Reflexion, and Self-Consistency. These patterns improve reliability and quality of agent outputs.

---

## Architecture 1: Plan-and-Execute

### Concept
Break complex tasks into steps, execute them sequentially, and replan if needed.

### Flow Diagram
```
Task
  ↓
[PLANNER] → Create step-by-step plan
  ↓
Step 1 → Execute → Result 1
Step 2 → Execute → Result 2
  ↓
[REPLANNER] → Should we adjust remaining steps?
  ↓
Step 3 → Execute → Result 3
...
  ↓
Final Result
```

### Components

#### 1. Plan Model
```python
class Plan(BaseModel):
    steps: List[str] = Field(description="Ordered steps to solve the task")
    reasoning: str = Field(description="Why this plan will work")
```

**Example Output:**
```python
Plan(
    steps=[
        "Research popular Python web frameworks",
        "Compare Django, Flask, and FastAPI features",
        "Write comparison report"
    ],
    reasoning="This plan breaks down the task into research, comparison, and writing phases..."
)
```

#### 2. Step Result Model
```python
class StepResult(BaseModel):
    step: str
    result: str
    success: bool
```

#### 3. Replan Decision Model
```python
class ReplanDecision(BaseModel):
    need_replan: bool
    revised_remaining_steps: Optional[List[str]] = None
    reason: str
```

### Core Functions

#### `planner(task: str) -> Plan`
Creates initial plan using structured output.
```python
structured_llm = llm.with_structured_output(Plan)
return structured_llm.invoke(f"Create a step-by-step plan to: {task}")
```

#### `executor(step: str, context: str) -> StepResult`
Executes a single step with accumulated context.
```python
prompt = f"""Execute this step: {step}

Context from previous steps:
{context}

Provide the result of executing this step."""
result = llm.invoke(prompt).content
return StepResult(step=step, result=result, success=True)
```

#### `replanner(task, completed, remaining) -> ReplanDecision`
Decides if remaining steps need adjustment.
```python
completed_summary = "\n".join(f"- {s.step}: {s.result[:100]}" for s in completed)
remaining_str = "\n".join(f"- {s}" for s in remaining)

prompt = f"""Original task: {task}

Completed steps:
{completed_summary}

Remaining steps:
{remaining_str}

Should we revise the remaining steps?"""

return structured_llm.invoke(prompt)
```

#### `run_plan_execute(task: str)`
Orchestrates the entire process.
```python
def run_plan_execute(task: str):
    # 1. Create plan
    plan = planner(task)
    
    # 2. Execute steps with replanning
    completed = []
    steps = plan.steps.copy()
    
    for i, step in enumerate(steps):
        # Execute step
        context = "\n".join(f"Step {j+1}: {r.result}" for j, r in enumerate(completed))
        result = executor(step, context)
        completed.append(result)
        
        # Replan after every 2 steps
        remaining = steps[i+1:]
        if remaining and i % 2 == 1:
            decision = replanner(task, completed, remaining)
            if decision.need_replan and decision.revised_remaining_steps:
                steps = [s.step for s in completed] + decision.revised_remaining_steps
    
    return completed
```

### Benefits
- **Handles complexity**: Breaks down hard tasks
- **Adaptability**: Replans based on progress
- **Transparency**: Clear step-by-step progress
- **Error recovery**: Can adjust plan if steps fail

### Use Cases
- Research and report writing
- Multi-step coding tasks
- Project planning
- Complex problem solving

---

## Architecture 2: Reflexion

### Concept
Iteratively improve solutions through self-critique and reflection.

### Flow Diagram
```
Task
  ↓
[Attempt 1] → Solution 1
  ↓
[Reflect] → Critique 1
  ↓
[Attempt 2 with feedback] → Solution 2
  ↓
[Reflect] → Critique 2
  ↓
... (repeat until threshold or max attempts)
  ↓
Final Solution
```

### Components

#### 1. Attempt Model
```python
class Attempt(BaseModel):
    solution: str
    confidence: float = Field(ge=0, le=1)
```

#### 2. Reflection Model
```python
class Reflection(BaseModel):
    is_correct: bool
    score: float = Field(ge=0, le=1, description="Quality score")
    critique: str
    specific_improvements: List[str]
```

### Core Functions

#### `attempt_solution(task: str, reflections: List[str] = []) -> Attempt`
Attempts to solve the task, using previous feedback.
```python
reflection_text = ""
if reflections:
    reflection_text = "\n\nPrevious attempt feedback:\n" + "\n".join(
        f"- {r}" for r in reflections
    )

return structured_llm.invoke(
    f"Solve this task:\n{task}{reflection_text}\n\nProvide your best solution."
)
```

#### `reflect_on_attempt(task: str, attempt: Attempt, test_cases: List[dict] = []) -> Reflection`
Critiques the solution.
```python
test_info = ""
if test_cases:
    test_info = f"\n\nTest cases to consider: {test_cases}"

return structured_llm.invoke(
    f"""Evaluate this solution to: {task}

Solution:
{attempt.solution}
{test_info}

Be critical. Identify specific issues and improvements."""
)
```

#### `run_reflexion(task: str, max_attempts: int = 3, threshold: float = 0.85)`
Runs the reflexion loop.
```python
def run_reflexion(task: str, max_attempts: int = 3, threshold: float = 0.85):
    reflections = []
    
    for i in range(max_attempts):
        # Attempt
        attempt = attempt_solution(task, reflections)
        
        # Reflect
        reflection = reflect_on_attempt(task, attempt)
        
        # Check if good enough
        if reflection.score >= threshold:
            return attempt
        
        # Feed feedback back
        reflections.extend(reflection.specific_improvements)
    
    return attempt  # Return best effort
```

### Example Execution
```
Task: "Write a Python function that finds all prime numbers up to n using Sieve of Eratosthenes"

Attempt 1:
  Solution: [initial implementation]
  Score: 0.75
  Critique: Missing edge case handling, inefficient for large n
  Improvements: ["Add input validation", "Optimize memory usage"]

Attempt 2 (with feedback):
  Solution: [improved implementation]
  Score: 0.92 ✓
  
✅ Accepted at attempt 2
```

### Benefits
- **Quality improvement**: Iterative refinement
- **Self-correction**: Catches own mistakes
- **Learning**: Builds on previous attempts
- **Confidence**: Scores indicate reliability

### Use Cases
- Code generation
- Math problem solving
- Creative writing
- Any task with clear quality criteria

---

## Architecture 3: Self-Consistency

### Concept
Generate multiple solutions and use voting to find consensus.

### Flow Diagram
```
Question
  ↓
[Generate n samples with high temperature]
  ↓
Sample 1 → Answer A
Sample 2 → Answer A
Sample 3 → Answer B
Sample 4 → Answer A
Sample 5 → Answer A
  ↓
[Vote] → Answer A (4/5 = 80%)
  ↓
Final Answer: A
```

### Implementation

```python
from collections import Counter

def self_consistent_answer(question: str, n_samples: int = 5, temperature: float = 0.7):
    hot_llm = get_llm(temperature=temperature)
    
    prompt = f"""{question}

Think step by step, then give ONLY your final answer on the last line as:
ANSWER: <your answer>"""

    answers = []
    for i in range(n_samples):
        response = hot_llm.invoke(prompt).content
        
        # Extract answer after "ANSWER:"
        lines = response.split("\n")
        for line in reversed(lines):
            if "ANSWER:" in line.upper():
                answer = line.split(":", 1)[-1].strip()
                answers.append(answer)
                break
    
    # Vote
    winner, count = Counter(answers).most_common(1)[0]
    confidence = count / n_samples
    
    return winner, confidence
```

### Example Execution
```
Question: "Is 1001 a prime number?"

Samples:
  Sample 1: "No, 1001 = 7 × 11 × 13"
  Sample 2: "No, divisible by 7"
  Sample 3: "No"
  
Consensus: "No" (3/3 = 100%)
Confidence: 1.0
```

### Benefits
- **Reduces randomness**: Voting filters out errors
- **Confidence scoring**: Know when uncertain
- **Simple**: No complex state management
- **Parallelizable**: Samples can be generated concurrently

### Trade-offs
| Pros | Cons |
|------|------|
| Simple to implement | n× slower |
| No state management | n× more expensive |
| Confidence scores | Only works for discrete answers |
| Reduces errors | May miss creative solutions |

### Use Cases
- Math problems
- Factual questions
- Multiple choice
- Classification tasks

---

## Architecture Comparison

| Architecture | Best For | Speed | Quality | Complexity |
|-------------|----------|-------|---------|------------|
| Plan-and-Execute | Multi-step tasks | Medium | High | Medium |
| Reflexion | Quality-critical | Slow | Very High | Medium |
| Self-Consistency | Factual answers | Slow (n×) | High | Low |

## When to Use Each

### Plan-and-Execute
✅ Task requires multiple sequential steps
✅ Steps may depend on previous results
✅ Plan might need adjustment

### Reflexion
✅ Quality is critical
✅ Clear evaluation criteria exist
✅ Time/cost budget allows iteration

### Self-Consistency
✅ Discrete answers (yes/no, numbers, categories)
✅ Need confidence scores
✅ Can afford n LLM calls

## Advanced: Combine Architectures

```python
# Plan-and-Execute + Reflexion
def plan_with_reflexion(task):
    plan = planner(task)
    for step in plan.steps:
        # Use reflexion for each step
        result = run_reflexion(f"Execute: {step}")
        ...

# Self-Consistency + Plan
def robust_planning(task):
    # Use self-consistency to create plan
    plan, confidence = self_consistent_answer(f"Plan for: {task}")
    ...
```

## Dependencies
```python
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_core.prompts import ChatPromptTemplate
from collections import Counter
```

## Next Steps
- Day 6: LangGraph for complex workflows
- Day 7: Multi-agent coordination
