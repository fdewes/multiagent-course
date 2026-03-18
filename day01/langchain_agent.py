"""
Day 1 — Same agent using LangChain's built-in ReAct agent.
Shows how the framework removes boilerplate.
"""
import sys
sys.path.insert(0, "..")
from config import get_llm
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool

# ── Define Tools with @tool decorator ─────────────────────────────────────────
@tool
def search(query: str) -> str:
    """Search for factual information about a topic."""
    db = {
        "berlin population": "Berlin has approximately 3.6 million people (2024).",
        "python creator": "Python was created by Guido van Rossum in 1991.",
        "speed of light": "The speed of light is 299,792,458 metres per second.",
    }
    for key, value in db.items():
        if key in query.lower():
            return value
    return "No result found."

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression like '(3 + 4) * 2'."""
    import re
    if not re.match(r'^[\d\s\+\-\*\/\.\(\)]+$', expression):
        return "Error: unsafe expression"
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

tools = [search, calculator]

# ── Standard ReAct prompt ──────────────────────────────────────────────────────
REACT_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

prompt = PromptTemplate.from_template(REACT_PROMPT)
llm = get_llm(temperature=0.0)

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=6,
    handle_parsing_errors=True,
)

if __name__ == "__main__":
    result = agent_executor.invoke({
        "input": "What is the population of Berlin multiplied by 2?"
    })
    print(f"\nFinal: {result['output']}")
