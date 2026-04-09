# Day 1: LangChain Agent Documentation

## File: `day01/langchain_agent.py`

## Overview
Demonstrates the same agent concept as `agent_loop.py` but using **LangChain's built-in agent API**. Shows how frameworks simplify tool-based agents.

## Core Concepts

### The `@tool` Decorator
```python
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    """Search for factual information about a topic."""
    # implementation...
```

The `@tool` decorator:
- Automatically creates a `Tool` object from a function
- Extracts function signature for schema generation
- Uses docstring as tool description for the LLM
- Enables automatic tool calling

### `create_agent()` Function
```python
from langchain.agents import create_agent

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="...",
)
```

Creates a pre-built agent with:
- Automatic tool selection
- Message management
- Recursion handling
- Error recovery

## Components

### 1. Tools

#### `search(query: str) -> str`
Searches a knowledge base for facts.

**Parameters:**
- `query` (str): Search query

**Returns:**
- `str`: Fact if found, otherwise "No result found."

**Knowledge Base:**
- Berlin population: ~3.6 million (2024)
- Python creator: Guido van Rossum (1991)
- Speed of light: 299,792,458 m/s

#### `calculator(expression: str) -> str`
Evaluates mathematical expressions safely.

**Parameters:**
- `expression` (str): Math expression like "(3 + 4) * 2"

**Security:**
```python
eval(expression, {"__builtins__": {}}, {})
```
- Empty `__builtins__` prevents access to dangerous functions
- Regex validation allows only digits and operators

### 2. Agent Creation

```python
agent = create_agent(
    model=llm,              # The LLM to use
    tools=tools,            # List of @tool decorated functions
    system_prompt="...",    # Instructions for the agent
)
```

**System Prompt Instructions:**
1. Be concise
2. Use tools only when needed
3. Use search for factual lookups
4. Use calculator for arithmetic
5. Stop calling tools when you have enough info
6. Don't repeat similar tool calls
7. Return just the final answer

### 3. Agent Invocation

```python
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
        "recursion_limit": 12  # Max tool call iterations
    },
)
```

**Input Format:**
- `messages`: List of message dicts with `role` and `content`
- Supports: `system`, `user`, `assistant` roles

**Configuration:**
- `recursion_limit`: Prevents infinite tool call loops

**Output Format:**
```python
result["messages"][-1].content  # Last message content
```

## Comparison: Bare-metal vs LangChain

| Aspect | agent_loop.py | langchain_agent.py |
|--------|---------------|-------------------|
| Lines of code | 123 | 73 |
| Tool registration | Manual dict | `@tool` decorator |
| Action parsing | Regex | Automatic |
| Tool calling | Manual | Automatic |
| Error handling | Manual | Built-in |
| Observability | Print | Tracing available |
| Flexibility | Full control | Abstraction |

## How It Works

### Flow Diagram
```
User Question
    ↓
create_agent()
    ├── LLM (qwen3.5:9b)
    ├── Tools [search, calculator]
    └── System Prompt
    ↓
agent.invoke()
    ↓
[Internal Loop]
    1. LLM decides: answer or tool?
    2. If tool: select + call
    3. Get observation
    4. Repeat until answer ready
    ↓
Final Answer
```

### Example Execution

**Question:** "What is the population of Berlin multiplied by 2?"

**Internal Steps:**
1. LLM analyzes question → needs population + calculation
2. LLM calls `search("population of Berlin")`
3. Tool returns: "Berlin has approximately 3.6 million people"
4. LLM calls `calculator("3600000 * 2")`
5. Tool returns: "7200000"
6. LLM has all info → generates final answer

**Output:**
```
Final: The population of Berlin multiplied by 2 is 7.2 million.
```

## Key Advantages of LangChain

1. **Less Code**: 40% fewer lines
2. **Automatic Parsing**: No regex needed
3. **Built-in Safety**: Recursion limits, error handling
4. **Standard Interface**: Same pattern for all agents
5. **Extensible**: Easy to add more tools

## Key Disadvantages

1. **Less Control**: Harder to customize behavior
2. **Black Box**: Less visibility into internals
3. **Dependencies**: Requires LangChain installation
4. **Versioning**: API may change between versions

## Dependencies
```python
from langchain.agents import create_agent
from langchain_core.tools import tool
from config import get_llm
```

## Usage Pattern

```python
# 1. Define tools with @tool decorator
@tool
def my_tool(param: str) -> str:
    """Description for LLM."""
    return result

# 2. Create agent
agent = create_agent(
    model=llm,
    tools=[my_tool],
    system_prompt="Instructions..."
)

# 3. Invoke agent
result = agent.invoke({
    "messages": [{"role": "user", "content": "Question?"}]
})

# 4. Extract answer
answer = result["messages"][-1].content
```

## Next Steps
- Day 2: Structured output for more reliable parsing
- Day 3: Advanced tool patterns and schemas
