"""
Day 3 — Tool design, schema inspection, dynamic tool loading, error handling.
"""
import sys
sys.path.insert(0, "..")
from config import get_llm
from langchain_core.tools import tool, StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_function
from pydantic import BaseModel, Field
from typing import Optional, Literal
import json, httpx, re

# ── 1. Basic tool with type hints ──────────────────────────────────────────────
@tool
def get_weather(city: str, unit: Literal["celsius", "fahrenheit"] = "celsius") -> str:
    """Get current weather for a city. Use when the user asks about weather."""
    # Simulated — swap for real API
    data = {
        "berlin": {"celsius": "12°C, partly cloudy", "fahrenheit": "54°F, partly cloudy"},
        "munich": {"celsius": "8°C, rainy", "fahrenheit": "46°F, rainy"},
    }
    return data.get(city.lower(), {}).get(unit, f"No data for {city}")

# ── 2. Tool with complex input schema ─────────────────────────────────────────
class SearchParams(BaseModel):
    query: str = Field(description="The search query")
    max_results: int = Field(default=3, ge=1, le=10, description="Number of results")
    filter_domain: Optional[str] = Field(default=None, description="Restrict to domain")

@tool(args_schema=SearchParams)
def web_search(query: str, max_results: int = 3, filter_domain: Optional[str] = None) -> str:
    """Search the web for current information. Returns top results as text."""
    # Simulated results
    results = [
        {"title": f"Result {i} for '{query}'", "snippet": f"Relevant info {i}..."}
        for i in range(1, max_results + 1)
    ]
    if filter_domain:
        results = [r for r in results]  # Would filter in real implementation
    return json.dumps(results, indent=2)

# ── 3. StructuredTool from function (for existing code) ───────────────────────
def database_query(sql: str, database: str = "main") -> str:
    """Execute a read-only SQL query against the specified database."""
    if not sql.strip().upper().startswith("SELECT"):
        return "Error: only SELECT queries are allowed"
    # Simulated
    return f"[{database}] Query executed: {sql}\nRows: (simulated results)"

db_tool = StructuredTool.from_function(
    func=database_query,
    name="database_query",
    description="Execute read-only SQL SELECT queries on the database.",
)

# ── 4. Tool with retry logic ───────────────────────────────────────────────────
from tenacity import retry, stop_after_attempt, wait_exponential

@tool
def fetch_url(url: str) -> str:
    """Fetch the content of a URL. Returns the response text."""
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=4))
    def _fetch():
        response = httpx.get(url, timeout=10, follow_redirects=True)
        response.raise_for_status()
        return response.text[:2000]  # Limit output size

    try:
        return _fetch()
    except Exception as e:
        return f"Error fetching {url}: {e}"

# ── 5. Tool schema inspection ─────────────────────────────────────────────────
print("=" * 60)
print("Tool Schemas (what the LLM sees)")
print("=" * 60)

all_tools = [get_weather, web_search, db_tool]
for t in all_tools:
    schema = convert_to_openai_function(t)
    print(f"\nTool: {schema['name']}")
    print(f"Description: {schema['description']}")
    print(f"Parameters: {json.dumps(schema['parameters'], indent=2)}")

# ── 6. LangChain agent with multiple tools ────────────────────────────────────
print("\n" + "=" * 60)
print("Agent with tool use")

from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate

tools = [get_weather, web_search, db_tool]
llm = get_llm(temperature=0.0)

PROMPT = """Answer the question using the available tools.

Tools:
{tools}

Use format:
Question: {input}
Thought: ...
Action: tool_name
Action Input: input
Observation: result
...
Final Answer: answer

{agent_scratchpad}"""

prompt = PromptTemplate.from_template(PROMPT)
agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True,
                         max_iterations=4, handle_parsing_errors=True)

result = executor.invoke({"input": "What is the weather in Berlin in celsius?"})
print(f"\nAnswer: {result['output']}")

# ── 7. Dynamic tool retrieval (for large toolsets) ────────────────────────────
print("\n" + "=" * 60)
print("Dynamic tool retrieval (scaling to 100+ tools)")

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.embeddings import FakeEmbeddings  # Use OllamaEmbeddings in production

# Build tool index
tool_descriptions = {t.name: t.description for t in all_tools}

print("Tool index built. In production:")
print("  1. Embed all tool descriptions")
print("  2. For each user query, embed query")
print("  3. Retrieve top-k relevant tools")
print("  4. Pass only those tools to the agent")
print("\nThis scales to 1000+ tools without hitting context limits.")
