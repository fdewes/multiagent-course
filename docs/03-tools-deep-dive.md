# Day 3: Tools Deep Dive Documentation

## File: `day03/tools_deep_dive.py`

## Overview
Explores **tool design patterns**, schema inspection, dynamic loading, and error handling. Shows how to create robust tools for agents.

## Core Concepts

### What is a Tool?
A tool is a function an LLM can call to:
- Get external information (search, database)
- Perform actions (calculator, API calls)
- Extend capabilities beyond training data

### Tool Lifecycle
```
1. Define function with @tool decorator
2. LLM sees tool schema (name, description, parameters)
3. LLM decides to use tool
4. Tool is executed
5. Result returned to LLM
6. LLM uses result to continue
```

## Pattern 1: Basic Tool with Type Hints

### Implementation
```python
from langchain_core.tools import tool
from typing import Literal

@tool
def get_weather(city: str, unit: Literal["celsius", "fahrenheit"] = "celsius") -> str:
    """Get current weather for a city. Use when user asks about weather."""
    data = {
        "berlin": {"celsius": "12°C, partly cloudy", "fahrenheit": "54°F, partly cloudy"},
        "munich": {"celsius": "8°C, rainy", "fahrenheit": "46°F, rainy"},
    }
    return data.get(city.lower(), {}).get(unit, f"No data for {city}")
```

### Features
- **Type hints**: `city: str` - used for schema generation
- **Literal types**: `unit: Literal["celsius", "fahrenheit"]` - restricted choices
- **Default values**: `unit: ... = "celsius"`
- **Docstring**: Description shown to LLM

### Generated Schema
```json
{
  "name": "get_weather",
  "description": "Get current weather for a city...",
  "parameters": {
    "type": "object",
    "properties": {
      "city": {"type": "string"},
      "unit": {
        "type": "string",
        "enum": ["celsius", "fahrenheit"],
        "default": "celsius"
      }
    },
    "required": ["city"]
  }
}
```

## Pattern 2: Tool with Complex Input Schema

### Implementation
```python
from pydantic import BaseModel, Field
from typing import Optional

class SearchParams(BaseModel):
    query: str = Field(description="The search query")
    max_results: int = Field(default=3, ge=1, le=10, description="Number of results")
    filter_domain: Optional[str] = Field(default=None, description="Restrict to domain")

@tool(args_schema=SearchParams)
def web_search(query: str, max_results: int = 3, filter_domain: Optional[str] = None) -> str:
    """Search the web for current information. Returns top results as text."""
    # Implementation...
```

### Features
- **Custom schema**: `args_schema=SearchParams` overrides function signature
- **Field validation**: `ge=1, le=10` ensures 1 ≤ max_results ≤ 10
- **Optional fields**: `Optional[str]` with default None
- **Field descriptions**: Help LLM understand parameters

## Pattern 3: StructuredTool from Function

### Use Case
Convert existing functions to tools without rewriting.

### Implementation
```python
from langchain_core.tools import StructuredTool

def database_query(sql: str, database: str = "main") -> str:
    """Execute a read-only SQL query."""
    if not sql.strip().upper().startswith("SELECT"):
        return "Error: only SELECT queries are allowed"
    return f"[{database}] Query executed: {sql}"

db_tool = StructuredTool.from_function(
    func=database_query,
    name="database_query",
    description="Execute read-only SQL SELECT queries on the database.",
)
```

### When to Use
- Legacy code integration
- Functions from external libraries
- When you need custom name/description

## Pattern 4: Tool with Retry Logic

### Implementation
```python
from tenacity import retry, stop_after_attempt, wait_exponential
import httpx

@tool
def fetch_url(url: str) -> str:
    """Fetch the content of a URL. Returns the response text."""
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=4))
    def _fetch():
        response = httpx.get(url, timeout=10, follow_redirects=True)
        response.raise_for_status()
        return response.text[:2000]  # Limit size
    
    try:
        return _fetch()
    except Exception as e:
        return f"Error fetching {url}: {e}"
```

### Retry Configuration
```python
@retry(
    stop=stop_after_attempt(3),           # Max 3 attempts
    wait=wait_exponential(                # Exponential backoff
        multiplier=1,                     # 1s, 2s, 4s...
        min=1,                            # Minimum 1 second
        max=4                             # Maximum 4 seconds
    )
)
```

### Benefits
- **Resilience**: Handles transient network errors
- **Backoff**: Doesn't overwhelm servers
- **Timeout**: Prevents hanging

## Pattern 5: Tool Schema Inspection

### Implementation
```python
from langchain_core.utils.function_calling import convert_to_openai_function
import json

all_tools = [get_weather, web_search, db_tool]

for t in all_tools:
    schema = convert_to_openai_function(t)
    print(f"Tool: {schema['name']}")
    print(f"Description: {schema['description']}")
    print(f"Parameters: {json.dumps(schema['parameters'], indent=2)}")
```

### Purpose
- Debug tool schemas
- Verify LLM sees correct information
- Generate documentation

## Pattern 6: Agent with Tools (Modern LangChain)

### Implementation
```python
from langchain.agents import create_agent

tools = [get_weather, web_search, db_tool]
llm = get_llm(temperature=0.0)

# Create agent with tools
agent = create_agent(llm, tools)

# Invoke agent
result = agent.invoke({
    "messages": [("user", "What is the weather in Berlin in celsius?")]
})

print(f"Answer: {result['messages'][-1].content}")
```

### How It Works
1. Agent receives user question
2. LLM analyzes which tool(s) needed
3. LLM calls tool(s) with parameters
4. Tool results returned to LLM
5. LLM generates final answer

### Example Execution
```
User: "What is the weather in Berlin in celsius?"

LLM thinks: Need weather info for Berlin
LLM calls: get_weather(city="Berlin", unit="celsius")
Tool returns: "12°C, partly cloudy"
LLM thinks: Have the answer
LLM responds: "The current weather in Berlin is 12°C and partly cloudy."
```

## Pattern 7: Dynamic Tool Retrieval (Scaling)

### Problem
100+ tools = too much context, expensive, slow

### Solution: Semantic Tool Routing
```
1. Embed all tool descriptions
2. User query → embed query
3. Retrieve top-k relevant tools by similarity
4. Pass only those tools to agent
```

### Pseudocode
```python
# Build tool index (one-time)
tool_embeddings = {}
for tool in all_tools:
    tool_embeddings[tool.name] = embeddings.embed_query(tool.description)

# At runtime
query_embedding = embeddings.embed_query(user_query)
relevant_tools = retrieve_by_similarity(query_embedding, top_k=5)
agent = create_agent(llm, relevant_tools)
```

### Benefits
- Scales to 1000+ tools
- Lower cost (fewer tokens)
- Faster (less context)
- Better focus (relevant tools only)

## Tool Design Best Practices

### 1. Clear Names
```python
✅ get_weather(city)
❌ weather(city)
❌ get_current_weather_for_location(city)
```

### 2. Descriptive Docstrings
```python
✅ """Get current weather for a city. Use when user asks about weather."""
❌ """Get weather."""
```

### 3. Type Hints
```python
✅ def search(query: str, max_results: int = 5) -> str:
❌ def search(query, max_results=5):
```

### 4. Error Handling
```python
✅ try:
       return result
   except Exception as e:
       return f"Error: {e}"
❌ return risky_operation()  # May crash agent
```

### 5. Parameter Limits
```python
✅ max_results: int = Field(ge=1, le=10)
❌ max_results: int  # LLM might pass 10000
```

## Dependencies
```python
from langchain_core.tools import tool, StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.agents import create_agent
from pydantic import BaseModel, Field
from typing import Optional, Literal
from tenacity import retry, stop_after_attempt, wait_exponential
import httpx
```

## Next Steps
- Day 4: Memory systems for agents
- Day 5: Agent architectures
