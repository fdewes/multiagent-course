"""
Day 11 — Evaluation, Observability, LLM-as-Judge.
This is what separates hobby projects from production systems.
"""
import sys
sys.path.insert(0, "..")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_llm
from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
import time
import uuid

llm = get_llm(temperature=0.0)

# ── 1. LLM-as-Judge ───────────────────────────────────────────────────────────
print("=" * 60)
print("1. LLM-as-Judge Evaluation")

class EvalResult(BaseModel):
    score: int = Field(ge=1, le=5, description="Score 1-5")
    reasoning: str
    passed: bool
    issues: List[str] = []

def llm_judge_faithfulness(question: str, context: str, answer: str) -> EvalResult:
    """Evaluate if the answer is faithful to the provided context."""
    judge_llm = llm.with_structured_output(EvalResult)
    return judge_llm.invoke(f"""Evaluate if this answer is faithful to (only uses information from) the context.

Question: {question}
Context: {context}
Answer: {answer}

Score 1-5:
1 = Completely unfaithful (made up facts)
3 = Partially faithful (some hallucinations)
5 = Completely faithful (only uses context)

Be strict. Identify any claims not supported by context.""")

def llm_judge_relevance(question: str, answer: str) -> EvalResult:
    """Evaluate if the answer actually addresses the question."""
    judge_llm = llm.with_structured_output(EvalResult)
    return judge_llm.invoke(f"""Evaluate if this answer directly addresses the question.

Question: {question}
Answer: {answer}

Score 1-5:
1 = Completely off-topic
3 = Partially addresses the question
5 = Directly and completely addresses the question""")

def llm_judge_completeness(question: str, answer: str, expected_points: List[str]) -> EvalResult:
    """Evaluate if the answer covers all expected key points."""
    judge_llm = llm.with_structured_output(EvalResult)
    points_str = "\n".join(f"- {p}" for p in expected_points)
    return judge_llm.invoke(f"""Evaluate if this answer covers all expected key points.

Question: {question}
Answer: {answer}

Expected key points:
{points_str}

Score based on how many key points are covered (5 = all covered).""")

# Test the judges
test_cases = [
    {
        "question": "What is RAG?",
        "context": "RAG stands for Retrieval Augmented Generation. It combines vector search with LLMs.",
        "answer": "RAG is Retrieval Augmented Generation, which uses vector databases to retrieve relevant documents and feeds them to an LLM to generate grounded answers.",
        "expected_points": ["Retrieval Augmented Generation", "vector search", "LLM", "documents"]
    }
]

for tc in test_cases:
    print(f"\nQuestion: {tc['question']}")
    f = llm_judge_faithfulness(tc["question"], tc["context"], tc["answer"])
    r = llm_judge_relevance(tc["question"], tc["answer"])
    c = llm_judge_completeness(tc["question"], tc["answer"], tc["expected_points"])
    print(f"  Faithfulness:  {f.score}/5 — {f.reasoning[:80]}")
    print(f"  Relevance:     {r.score}/5 — {r.reasoning[:80]}")
    print(f"  Completeness:  {c.score}/5 — {c.reasoning[:80]}")
    avg = (f.score + r.score + c.score) / 3
    print(f"  Overall:       {avg:.1f}/5")


# ── 2. Tracing / Observability ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. Custom Tracing System")

@dataclass
class Span:
    span_id: str
    name: str
    start_time: float
    end_time: Optional[float] = None
    inputs: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    error: Optional[str] = None
    children: List['Span'] = field(default_factory=list)

    @property
    def duration_ms(self) -> Optional[float]:
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return None

    def to_dict(self) -> dict:
        return {
            "span_id": self.span_id,
            "name": self.name,
            "duration_ms": self.duration_ms,
            "inputs": self.inputs,
            "outputs": {k: str(v)[:200] for k, v in self.outputs.items()},
            "metadata": self.metadata,
            "error": self.error,
            "children": [c.to_dict() for c in self.children],
        }

class Tracer:
    def __init__(self):
        self.traces = []
        self._current_span = None

    def start_span(self, name: str, inputs: dict = {}) -> Span:
        span = Span(
            span_id=str(uuid.uuid4())[:8],
            name=name,
            start_time=time.time(),
            inputs=inputs,
        )
        if self._current_span:
            self._current_span.children.append(span)
        else:
            self.traces.append(span)
        self._current_span = span
        return span

    def end_span(self, span: Span, outputs: dict = {}, error: str = None):
        span.end_time = time.time()
        span.outputs = outputs
        span.error = error
        # Pop back to parent
        self._current_span = None

    def __call__(self, name: str):
        """Use as context manager."""
        tracer = self

        class SpanContext:
            def __init__(self, name):
                self.name = name
                self.span = None
                self.inputs = {}
                self.outputs = {}

            def __enter__(self):
                self.span = tracer.start_span(self.name, self.inputs)
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                error = str(exc_val) if exc_val else None
                tracer.end_span(self.span, self.outputs, error)
                return False

        return SpanContext(name)

    def print_trace(self):
        def _print(span: Span, indent=0):
            prefix = "  " * indent
            dur = f"{span.duration_ms:.0f}ms" if span.duration_ms else "?"
            status = "❌" if span.error else "✅"
            print(f"{prefix}{status} {span.name} [{dur}]")
            if span.outputs:
                out = str(span.outputs)[:80]
                print(f"{prefix}   └─ {out}")
            for child in span.children:
                _print(child, indent + 1)

        for trace in self.traces:
            _print(trace)

tracer = Tracer()

# Wrap LLM calls with tracing
def traced_llm_call(prompt: str, name: str = "llm_call") -> str:
    with tracer(name) as span:
        span.inputs = {"prompt": prompt[:100]}
        start = time.time()
        result = llm.invoke(prompt).content
        span.outputs = {
            "response": result[:100],
            "tokens_approx": len(result.split()),
            "latency_ms": (time.time() - start) * 1000,
        }
        return result

# Simulate a traced agent run
with tracer("agent_run") as root:
    root.inputs = {"task": "Explain what a neural network is"}

    with tracer("retrieve") as r:
        r.inputs = {"query": "neural network"}
        r.outputs = {"docs_found": 3, "top_doc": "Neural networks are..."}
        time.sleep(0.01)  # Simulate latency

    response = traced_llm_call("What is a neural network? Be brief.", "generate_answer")

    with tracer("evaluate") as e:
        e.inputs = {"answer": response[:50]}
        e.outputs = {"score": 4.5, "passed": True}
        time.sleep(0.005)

print("\nTrace:")
tracer.print_trace()


# ── 3. Regression Test Suite ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. Regression Test Suite")

@dataclass
class TestCase:
    id: str
    input: str
    expected_output: Optional[str] = None
    expected_contains: List[str] = field(default_factory=list)
    expected_not_contains: List[str] = field(default_factory=list)
    min_score: float = 3.0

@dataclass
class TestResult:
    test_id: str
    passed: bool
    score: float
    reason: str

def run_test_suite(
    agent_fn: Callable[[str], str],
    test_cases: List[TestCase],
) -> List[TestResult]:
    results = []
    for tc in test_cases:
        print(f"  Running test {tc.id}...")
        try:
            output = agent_fn(tc.input)

            # Check contains
            missing = [s for s in tc.expected_contains if s.lower() not in output.lower()]
            forbidden = [s for s in tc.expected_not_contains if s.lower() in output.lower()]

            if missing:
                results.append(TestResult(tc.id, False, 0, f"Missing: {missing}"))
                continue
            if forbidden:
                results.append(TestResult(tc.id, False, 0, f"Contains forbidden: {forbidden}"))
                continue

            # LLM judge score
            if tc.expected_output:
                judge = llm_judge_relevance(tc.input, output)
                passed = judge.score >= tc.min_score
                results.append(TestResult(tc.id, passed, judge.score, judge.reasoning[:100]))
            else:
                results.append(TestResult(tc.id, True, 5.0, "All string checks passed"))

        except Exception as e:
            results.append(TestResult(tc.id, False, 0, f"Exception: {e}"))

    return results

# Define test cases
test_suite = [
    TestCase(
        id="tc001",
        input="What is Python?",
        expected_contains=["python", "programming"],
        expected_not_contains=["java", "javascript"],
    ),
    TestCase(
        id="tc002",
        input="What is 2 + 2?",
        expected_contains=["4"],
    ),
    TestCase(
        id="tc003",
        input="Explain machine learning briefly",
        expected_contains=["data", "model"],
        min_score=3.0,
    ),
]

def simple_agent(question: str) -> str:
    return llm.invoke(question).content

results = run_test_suite(simple_agent, test_suite)

passed = sum(1 for r in results if r.passed)
print(f"\nTest Results: {passed}/{len(results)} passed")
for r in results:
    status = "✅" if r.passed else "❌"
    print(f"  {status} {r.test_id} (score={r.score:.1f}): {r.reason[:80]}")
