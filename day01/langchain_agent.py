"""
Day 1 — Same agent using LangChain's newer built-in agent API.
"""

import sys
sys.path.insert(0, "..")

from config import get_llm
from langchain.agents import create_agent
from langchain_core.tools import tool


@tool
def search(query: str) -> str:
    """Search for factual information about a topic."""
    db = {
        "berlin population": "Berlin has approximately 3.6 million people (2024).",
        "python creator": "Python was created by Guido van Rossum in 1991.",
        "speed of light": "The speed of light is 299,792,458 metres per second.",
    }
    query_l = query.lower()
    for key, value in db.items():
        if key in query_l:
            return value
    return "No result found."


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression like '(3 + 4) * 2'."""
    import re

    if not re.match(r"^[\d\s\+\-\*\/\.\(\)]+$", expression):
        return "Error: unsafe expression"

    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"


tools = [search, calculator]
llm = get_llm(temperature=0.0)

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "You are a concise assistant.\n"
        "Use tools only when needed.\n"
        "For factual lookups, use the search tool.\n"
        "For arithmetic, use the calculator tool.\n"
        "After you have enough information, STOP calling tools and answer directly.\n"
        "Do not call the same tool repeatedly with similar inputs.\n"
        "Return just the final answer."
    ),
)

if __name__ == "__main__":
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What is the population of Berlin multiplied by 2?"
                }
            ]
        },
        {
            "recursion_limit": 12
        },
    )

    print("\nFinal:", result["messages"][-1].content)