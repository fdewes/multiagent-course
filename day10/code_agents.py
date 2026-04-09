"""
Day 10 — Code Agents: generate, execute, debug, iterate.
A code agent is one of the most valuable patterns in production.
"""
import sys
sys.path.insert(0, "..")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_llm
from typing import TypedDict, Annotated, Optional
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel
import subprocess
import tempfile
import os
import re

llm = get_llm(temperature=0.2)

# ── Safe Code Executor ─────────────────────────────────────────────────────────
def execute_python(code: str, timeout: int = 10) -> dict:
    """
    Execute Python code in a subprocess (isolated process).
    Returns: {"stdout": ..., "stderr": ..., "success": bool, "return_code": int}
    In production: use Docker sandbox, E2B, or Firecracker microVMs.
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        tmpfile = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmpfile],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0,
            "return_code": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": "Timeout", "success": False, "return_code": -1}
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "success": False, "return_code": -1}
    finally:
        os.unlink(tmpfile)


def extract_code(text: str) -> Optional[str]:
    """Extract Python code from LLM response."""
    patterns = [
        r'```python\n(.*?)```',
        r'```\n(.*?)```',
        r'```(.*?)```',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
    return None


# ── 1. Simple Code Generation + Execution ─────────────────────────────────────
print("=" * 60)
print("1. Code Generation + Execution")

def generate_and_run(task: str) -> dict:
    prompt = f"""Write a Python script to: {task}

Requirements:
- The code must be complete and runnable
- Include print statements to show results
- Handle edge cases
- No external dependencies beyond standard library

Wrap your code in ```python ... ```"""

    response = llm.invoke(prompt).content
    code = extract_code(response)

    if not code:
        return {"success": False, "error": "No code found in response"}

    print(f"Generated code:\n{code}\n")
    result = execute_python(code)
    print(f"Output: {result['stdout']}")
    if result['stderr']:
        print(f"Errors: {result['stderr']}")
    return result

generate_and_run("Calculate the first 10 Fibonacci numbers and their sum")


# ── 2. Code Agent with Self-Debugging Loop ────────────────────────────────────
print("\n" + "=" * 60)
print("2. Code Agent with Self-Debugging (Reflexion for Code)")

class CodeState(TypedDict):
    task: str
    code: str
    execution_result: dict
    iterations: int
    success: bool
    error_history: list

def code_generator(state: CodeState) -> dict:
    error_context = ""
    if state.get("error_history"):
        recent_errors = state["error_history"][-2:]
        error_context = f"\n\nPrevious attempts failed with these errors:\n"
        for err in recent_errors:
            error_context += f"Code:\n{err['code']}\nError: {err['error']}\n\n"
        error_context += "Fix these issues in your new implementation."

    prompt = f"""Write Python code to solve this task:
{state['task']}
{error_context}
Requirements:
- Complete, runnable Python code
- Use print() to show all results
- No external dependencies

```python
# Your solution here
```"""

    response = llm.invoke(prompt).content
    code = extract_code(response) or ""
    print(f"\n  [Generator] Iteration {state.get('iterations', 0) + 1}")
    print(f"  Code preview: {code[:100]}...")
    return {"code": code, "iterations": state.get("iterations", 0) + 1}

def code_executor(state: CodeState) -> dict:
    if not state.get("code"):
        return {"execution_result": {"success": False, "stderr": "No code to execute"}}
    result = execute_python(state["code"])
    print(f"  [Executor] Success={result['success']}")
    if result["stdout"]:
        print(f"  Output: {result['stdout'][:200]}")
    if result["stderr"]:
        print(f"  Error: {result['stderr'][:200]}")
    return {"execution_result": result}

def evaluate_result(state: CodeState) -> dict:
    result = state.get("execution_result", {})
    if result.get("success") and result.get("stdout"):
        return {"success": True}

    # Add to error history
    error_history = state.get("error_history", [])
    error_history.append({
        "code": state.get("code", ""),
        "error": result.get("stderr", "No output produced")
    })
    return {"success": False, "error_history": error_history}

def should_retry(state: CodeState) -> str:
    if state.get("success"):
        return "done"
    if state.get("iterations", 0) >= 3:
        print("  [Agent] Max iterations reached")
        return "done"
    return "retry"

code_builder = StateGraph(CodeState)
code_builder.add_node("generate", code_generator)
code_builder.add_node("execute", code_executor)
code_builder.add_node("evaluate", evaluate_result)

code_builder.add_edge(START, "generate")
code_builder.add_edge("generate", "execute")
code_builder.add_edge("execute", "evaluate")
code_builder.add_conditional_edges(
    "evaluate",
    should_retry,
    {"retry": "generate", "done": END}
)

code_graph = code_builder.compile()

tasks = [
    "Implement binary search and test it with a list of 10 integers",
    "Write a function that counts word frequency in a text and shows the top 5 words",
]

for task in tasks:
    print(f"\nTask: {task}")
    result = code_graph.invoke({
        "task": task,
        "code": "",
        "execution_result": {},
        "iterations": 0,
        "success": False,
        "error_history": [],
    })
    print(f"Result: {'✅ Success' if result['success'] else '❌ Failed'} in {result['iterations']} iteration(s)")


# ── 3. Test-Driven Code Agent ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. Test-Driven Code Agent")

def generate_with_tests(task: str) -> tuple[str, str]:
    """Generate both implementation and test code."""
    test_prompt = f"""For this task: {task}

Write Python test cases FIRST (TDD style):
```python
# tests.py
import unittest

class TestSolution(unittest.TestCase):
    def test_basic(self):
        # Test basic functionality
        pass

    def test_edge_cases(self):
        # Test edge cases
        pass

if __name__ == '__main__':
    unittest.main(verbosity=2)
```

Then write the implementation:
```python
# solution.py
# Implementation here
```"""

    response = llm.invoke(test_prompt).content

    # Extract both code blocks
    codes = re.findall(r'```python\n(.*?)```', response, re.DOTALL)
    if len(codes) >= 2:
        test_code, impl_code = codes[0], codes[1]
    elif len(codes) == 1:
        impl_code = codes[0]
        test_code = ""
    else:
        impl_code = test_code = ""

    return impl_code, test_code

task = "Write a function `is_palindrome(s)` that checks if a string is a palindrome, ignoring spaces and case"
impl_code, test_code = generate_with_tests(task)

print(f"Implementation:\n{impl_code}")
if test_code:
    # Combine for execution
    combined = impl_code + "\n\n" + test_code.replace("from solution import", "")
    result = execute_python(combined)
    print(f"\nTest results:\n{result['stdout']}")
    print(f"Success: {result['success']}")
