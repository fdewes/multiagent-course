"""
Day 5 — Agent architectures: Plan-Execute, Reflexion, self-consistency.
"""
import sys
sys.path.insert(0, "..")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_llm
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_core.prompts import ChatPromptTemplate

llm = get_llm(temperature=0.0)

# ── Architecture 1: Plan-and-Execute ──────────────────────────────────────────
print("=" * 60)
print("Architecture 1: Plan-and-Execute")

class Plan(BaseModel):
    steps: List[str] = Field(description="Ordered steps to solve the task")
    reasoning: str = Field(description="Why this plan will work")

class StepResult(BaseModel):
    step: str
    result: str
    success: bool

class ReplanDecision(BaseModel):
    need_replan: bool
    revised_remaining_steps: Optional[List[str]] = None
    reason: str

def planner(task: str) -> Plan:
    structured_llm = llm.with_structured_output(Plan)
    return structured_llm.invoke(f"Create a step-by-step plan to: {task}")

def executor(step: str, context: str) -> StepResult:
    """Execute a single step given accumulated context."""
    prompt = f"""Execute this step: {step}

Context from previous steps:
{context if context else 'None yet'}

Provide the result of executing this step."""
    result = llm.invoke(prompt).content
    return StepResult(step=step, result=result, success=True)

def replanner(task: str, completed: List[StepResult], remaining: List[str]) -> ReplanDecision:
    structured_llm = llm.with_structured_output(ReplanDecision)
    completed_summary = "\n".join(f"- {s.step}: {s.result[:100]}" for s in completed)
    remaining_str = "\n".join(f"- {s}" for s in remaining)
    prompt = f"""Original task: {task}

Completed steps:
{completed_summary}

Remaining steps:
{remaining_str}

Should we revise the remaining steps based on what we've learned?"""
    return structured_llm.invoke(prompt)

def run_plan_execute(task: str):
    print(f"\nTask: {task}")

    # Plan
    plan = planner(task)
    print(f"Plan ({len(plan.steps)} steps): {plan.reasoning[:80]}...")
    for i, step in enumerate(plan.steps):
        print(f"  {i+1}. {step}")

    # Execute with replanning
    completed = []
    steps = plan.steps.copy()

    for i, step in enumerate(steps):
        print(f"\n  Executing step {i+1}: {step[:60]}...")
        context = "\n".join(f"Step {j+1}: {r.result}" for j, r in enumerate(completed))
        result = executor(step, context)
        completed.append(result)
        print(f"  Result: {result.result[:100]}...")

        # Replan after every 2 steps
        remaining = steps[i+1:]
        if remaining and i % 2 == 1:
            decision = replanner(task, completed, remaining)
            if decision.need_replan and decision.revised_remaining_steps:
                print(f"  [Replanning: {decision.reason[:60]}...]")
                steps = [s.step for s in completed] + decision.revised_remaining_steps

    print(f"\n✅ Task complete. Steps executed: {len(completed)}")
    return completed

run_plan_execute("Research the best Python web frameworks and write a comparison report")


# ── Architecture 2: Reflexion ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Architecture 2: Reflexion")

class Attempt(BaseModel):
    solution: str
    confidence: float = Field(ge=0, le=1)

class Reflection(BaseModel):
    is_correct: bool
    score: float = Field(ge=0, le=1, description="Quality score")
    critique: str
    specific_improvements: List[str]

def attempt_solution(task: str, reflections: List[str] = []) -> Attempt:
    structured_llm = llm.with_structured_output(Attempt)
    reflection_text = ""
    if reflections:
        reflection_text = "\n\nPrevious attempt feedback:\n" + "\n".join(
            f"- {r}" for r in reflections
        )
    return structured_llm.invoke(
        f"Solve this task:\n{task}{reflection_text}\n\nProvide your best solution."
    )

def reflect_on_attempt(task: str, attempt: Attempt, test_cases: List[dict] = []) -> Reflection:
    """Evaluate the attempt. In production, run actual tests."""
    structured_llm = llm.with_structured_output(Reflection)
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

def run_reflexion(task: str, max_attempts: int = 3, threshold: float = 0.85):
    print(f"\nTask: {task[:80]}...")
    reflections = []

    for i in range(max_attempts):
        print(f"\n  Attempt {i+1}:")
        attempt = attempt_solution(task, reflections)
        print(f"  Solution: {attempt.solution[:150]}...")
        print(f"  Confidence: {attempt.confidence:.2f}")

        reflection = reflect_on_attempt(task, attempt)
        print(f"  Score: {reflection.score:.2f} | Correct: {reflection.is_correct}")
        print(f"  Critique: {reflection.critique[:100]}...")

        if reflection.score >= threshold:
            print(f"\n✅ Accepted at attempt {i+1} (score {reflection.score:.2f})")
            return attempt

        # Feed reflection back
        reflections.extend(reflection.specific_improvements)

    print(f"\n⚠️  Max attempts reached. Best effort returned.")
    return attempt

run_reflexion(
    "Write a Python function that finds all prime numbers up to n using the Sieve of Eratosthenes"
)


# ── Architecture 3: Self-Consistency Voting ────────────────────────────────────
print("\n" + "=" * 60)
print("Architecture 3: Self-Consistency")

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
        else:
            answers.append(response.split("\n")[-1].strip())

    print(f"Samples: {answers}")
    winner, count = Counter(answers).most_common(1)[0]
    confidence = count / n_samples
    print(f"Consensus: {winner!r} ({count}/{n_samples} = {confidence:.0%})")
    return winner, confidence

self_consistent_answer("Is 1001 a prime number?", n_samples=3)
self_consistent_answer("What is 15% of 240?", n_samples=3)
