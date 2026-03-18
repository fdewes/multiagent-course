"""
Day 1 — Bare-metal ReAct agent loop (no framework).
Shows exactly what LangChain/LangGraph abstract away.
"""
import sys
sys.path.insert(0, "..")
import json, re
from config import get_llm

# ── Tools ──────────────────────────────────────────────────────────────────────
def search(query: str) -> str:
    """Fake search — replace with real SerpAPI / DuckDuckGo / Tavily."""
    db = {
        "berlin population": "Berlin has approximately 3.6 million people (2024).",
        "python creator": "Python was created by Guido van Rossum in 1991.",
        "speed of light": "The speed of light is 299,792,458 metres per second.",
    }
    for key, value in db.items():
        if key in query.lower():
            return value
    return "No result found."

def calculator(expression: str) -> str:
    """Safe eval for arithmetic."""
    try:
        # Only allow digits and basic operators
        if not re.match(r'^[\d\s\+\-\*\/\.\(\)]+$', expression):
            return "Error: unsafe expression"
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

TOOLS = {
    "search": search,
    "calculator": calculator,
}

TOOL_DESCRIPTIONS = """
You have access to the following tools:

search(query: str) -> str
    Search for factual information. Use for current events, facts, definitions.

calculator(expression: str) -> str
    Evaluate a mathematical expression. e.g. calculator("(3 + 4) * 2")

To use a tool, respond in this EXACT format:
    Action: tool_name
    Action Input: the input to the tool

When you have a final answer:
    Final Answer: your answer here
"""

SYSTEM_PROMPT = f"""You are a helpful assistant that answers questions step by step.
{TOOL_DESCRIPTIONS}
Always reason step by step before acting.
"""

# ── ReAct Loop ─────────────────────────────────────────────────────────────────
def run_agent(question: str, max_steps: int = 6) -> str:
    llm = get_llm(temperature=0.0)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for step in range(max_steps):
        print(f"\n{'─'*50}")
        print(f"Step {step + 1}")

        response = llm.invoke(messages)
        text = response.content
        print(f"LLM:\n{text}")

        messages.append({"role": "assistant", "content": text})

        # Check for final answer
        if "Final Answer:" in text:
            answer = text.split("Final Answer:")[-1].strip()
            print(f"\n✅ Final Answer: {answer}")
            return answer

        # Parse action
        action_match = re.search(r"Action:\s*(\w+)", text)
        input_match = re.search(r"Action Input:\s*(.+?)(?:\n|$)", text)

        if action_match and input_match:
            tool_name = action_match.group(1).strip()
            tool_input = input_match.group(1).strip()

            if tool_name in TOOLS:
                observation = TOOLS[tool_name](tool_input)
                print(f"\n🔧 {tool_name}({tool_input!r}) → {observation!r}")
                messages.append({
                    "role": "user",
                    "content": f"Observation: {observation}",
                })
            else:
                messages.append({
                    "role": "user",
                    "content": f"Observation: Tool '{tool_name}' not found.",
                })
        else:
            # No action found — force a final answer
            messages.append({
                "role": "user",
                "content": "Please provide a Final Answer based on what you know so far.",
            })

    return "Max steps reached without a final answer."


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    questions = [
        "What is the population of Berlin multiplied by 2?",
        "Who created Python and in what year?",
    ]
    for q in questions:
        print(f"\n{'═'*60}")
        print(f"Question: {q}")
        run_agent(q)
