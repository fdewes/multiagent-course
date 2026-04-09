# Day 1: Agent Loop Documentation

## File: `day01/agent_loop.py`

## Overview
This script demonstrates a **bare-metal ReAct (Reason + Act) agent loop** without any framework. It shows the fundamental mechanics that LangChain and LangGraph abstract away.

## Core Concepts

### ReAct Pattern
ReAct stands for **Reasoning + Acting**. The agent:
1. **Thinks/Reasons** about the problem
2. **Acts** by calling tools
3. **Observes** the results
4. **Repeats** until it has enough information to answer

### Agent Loop Components

#### 1. Tools
```python
TOOLS = {
    "search": search,
    "calculator": calculator,
}
```

**Available Tools:**
- **`search(query: str) -> str`**: Searches a knowledge base for facts
  - Contains hardcoded facts about Berlin population, Python creator, speed of light
  - In production, would use real APIs (SerpAPI, DuckDuckGo, Tavily)
  
- **`calculator(expression: str) -> str`**: Evaluates mathematical expressions
  - Uses safe `eval()` with regex validation
  - Only allows digits and basic operators (+, -, *, /, ., (, ))

#### 2. Prompt Structure
```python
SYSTEM_PROMPT = """
You are a helpful assistant that answers questions step by step.
[Tool descriptions]
Always reason step by step before acting.
"""
```

#### 3. Action Format
The LLM must output actions in this exact format:
```
Action: tool_name
Action Input: the input to the tool
```

Or for final answers:
```
Final Answer: your answer here
```

## Main Function

### `run_agent(question: str, max_steps: int = 6) -> str`

**Purpose:** Execute the ReAct loop for a given question.

**Parameters:**
- `question` (str): The user's question
- `max_steps` (int, default=6): Maximum iterations before giving up

**Returns:**
- `str`: The final answer or timeout message

**Algorithm:**
```
1. Initialize messages with system prompt + user question
2. For each step:
   a. Invoke LLM with current messages
   b. Check if "Final Answer:" is in response → return answer
   c. Parse Action and Action Input from response
   d. If valid tool found:
      - Execute tool with input
      - Add observation to messages
   e. Else:
      - Add error or force final answer message
3. Return timeout message if max steps reached
```

**Message Flow:**
```
[system] System prompt with tool descriptions
[user]   User question
[assistant] LLM response (thought + action or final answer)
[user]   Observation from tool execution (if action taken)
[assistant] Next LLM response
...
```

## Key Functions

### Tool Definitions

```python
def search(query: str) -> str:
    """Search knowledge base for facts."""
    # Simple substring matching
    for key, value in db.items():
        if key in query.lower():
            return value
    return "No result found."
```

```python
def calculator(expression: str) -> str:
    """Safely evaluate arithmetic expressions."""
    # Validate: only digits and operators allowed
    if not re.match(r'^[\d\s\+\-\*\/\.\(\)]+$', expression):
        return "Error: unsafe expression"
    return str(eval(expression))
```

### Action Parsing

```python
# Extract action name
action_match = re.search(r"Action:\s*(\w+)", text)

# Extract action input
input_match = re.search(r"Action Input:\s*(.+?)(?:\n|$)", text)

if action_match and input_match:
    tool_name = action_match.group(1).strip()
    tool_input = input_match.group(1).strip()
```

## Example Execution

**Question:** "What is the population of Berlin multiplied by 2?"

**Step 1:**
```
LLM:
I need to find the population of Berlin first, then multiply by 2.
Action: search
Action Input: population of Berlin
```

**Observation:**
```
search('population of Berlin') → 'Berlin has approximately 3.6 million people'
```

**Step 2:**
```
LLM:
Now I'll calculate 3.6 million × 2.
Action: calculator
Action Input: 3600000 * 2
```

**Observation:**
```
calculator('3600000 * 2') → '7200000'
```

**Step 3:**
```
LLM:
Final Answer: The population of Berlin multiplied by 2 is 7.2 million.
```

## Dependencies
- `re`: Regular expressions for parsing actions
- `config.get_llm()`: LLM instance from course configuration

## Learning Points
1. **ReAct is simple**: Just a loop of think→act→observe
2. **Tool use requires structured output**: LLM must follow exact format
3. **Error handling**: What if tool not found? What if no action parsed?
4. **Message history**: All context kept in conversation history
5. **Max steps**: Prevents infinite loops

## Comparison to Frameworks
| Aspect | Bare-metal | LangChain/LangGraph |
|--------|-----------|---------------------|
| Tool registration | Manual dict | `@tool` decorator |
| Action parsing | Regex | Built-in parsers |
| State management | List of dicts | Typed state objects |
| Error handling | Manual | Built-in retries |
| Observability | Print statements | Tracing/logging |

## Next Steps
- `langchain_agent.py`: Same concept using LangChain abstractions
- Day 2: Structured output for more reliable tool use
