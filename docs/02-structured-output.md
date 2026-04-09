# Day 2: Structured Output Documentation

## File: `day02/structured_output.py`

## Overview
Demonstrates **structured output patterns** - the foundation of reliable agents. Shows how to get predictable, parseable outputs from LLMs using JSON, Pydantic, and modern LangChain methods.

## Why Structured Output?

**Problem with Free Text:**
```
LLM: "The answer is 42, I think."
     "Forty-two is the answer!"
     "42"
```
All mean the same thing but are hard to parse reliably.

**Solution - Structured Output:**
```json
{"answer": 42, "confidence": 0.95}
```
Machine-readable, consistent, validated.

## Pattern 1: Simple JSON Output

### Implementation
```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

json_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a data extractor. Always respond with valid JSON only, no markdown."),
    ("user", "Extract the person's name, age, and city from: '{text}'"),
])

chain = json_prompt | llm | JsonOutputParser()
result = chain.invoke({"text": "Hi, I'm Maria, 32 years old, living in Munich."})
```

### Output
```python
{'name': 'Maria', 'age': 32, 'city': 'Munich'}
```

### Components
- **Prompt**: Instructs LLM to output JSON only
- **LLM**: Generates JSON text
- **JsonOutputParser**: Parses JSON string → Python dict

### Pros & Cons
| Pros | Cons |
|------|------|
| Simple to use | No validation |
| Works with any LLM | May produce invalid JSON |
| No dependencies | No type safety |

## Pattern 2: Pydantic Structured Output

### Implementation
```python
from pydantic import BaseModel, Field
from typing import List, Literal
from langchain_core.output_parsers import PydanticOutputParser

class TaskPlan(BaseModel):
    """A structured task plan."""
    goal: str = Field(description="The main objective")
    steps: List[str] = Field(description="Ordered list of steps")
    estimated_complexity: Literal["low", "medium", "high"]
    tools_needed: List[str] = Field(description="Tools or resources needed")

parser = PydanticOutputParser(pydantic_object=TaskPlan)

plan_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a project planner. {format_instructions}"),
    ("user", "Create a plan to: {goal}"),
]).partial(format_instructions=parser.get_format_instructions())

chain = plan_prompt | llm | parser
plan = chain.invoke({"goal": "Build a web scraper that monitors product prices"})
```

### Output
```python
TaskPlan(
    goal="Build a web scraper that monitors product prices",
    steps=[
        "Identify target websites...",
        "Set up Python environment...",
        ...
    ],
    estimated_complexity="medium",
    tools_needed=["Python", "requests", "BeautifulSoup", ...]
)
```

### Features
- **Type validation**: Pydantic validates output
- **Field descriptions**: Guide LLM via `Field(description=...)`
- **Enum values**: `Literal["low", "medium", "high"]` restricts choices
- **Auto format instructions**: `parser.get_format_instructions()` tells LLM the schema

### Pros & Cons
| Pros | Cons |
|------|------|
| Type-safe | More verbose |
| Validation | Requires Pydantic |
| Auto-documentation | Slower parsing |

## Pattern 3: `.with_structured_output()` (Modern LangChain)

### Implementation
```python
class AgentDecision(BaseModel):
    """The agent's next action decision."""
    reasoning: str = Field(description="Step-by-step reasoning")
    action: Literal["search", "calculate", "answer", "ask_clarification"]
    action_input: str = Field(description="Input for the chosen action")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0-1")

structured_llm = llm.with_structured_output(AgentDecision)

decision = structured_llm.invoke([
    ("system", "You are an agent. Decide the next action to take."),
    ("user", "What is the GDP of Germany divided by its population?")
])
```

### Output
```python
AgentDecision(
    reasoning="To calculate GDP per capita, I need Germany's GDP and population...",
    action="search",
    action_input="Germany GDP per capita 2023 nominal",
    confidence=0.85
)
```

### Features
- **Built-in method**: No separate parser needed
- **Native support**: LLM understands structure natively
- **Cleanest API**: Most modern approach

### Field Constraints
```python
confidence: float = Field(ge=0.0, le=1.0)  # ge=greater or equal, le=less or equal
```

## Pattern 4: Chain-of-Thought

### Implementation
```python
cot_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a careful analyst. Think step by step before answering."),
    ("user", """{question}

Think through this carefully:
1. What information do I have?
2. What am I being asked?
3. What steps do I need to take?
4. Final answer:"""),
])

chain = cot_prompt | llm
result = chain.invoke({
    "question": "A train travels 120km at 60km/h, then 80km at 40km/h. What is the total travel time?"
})
```

### Output
```
1. What information do I have?
   - Segment 1: 120km at 60km/h
   - Segment 2: 80km at 40km/h
   
2. What am I being asked?
   - Total travel time
   
3. What steps do I need to take?
   - Calculate time for each segment
   - Add times together
   
4. Final answer: 4 hours
```

### Purpose
- Improves reasoning on complex problems
- Makes thinking process explicit
- Reduces errors in multi-step problems

## Pattern 5: Self-Consistency Voting

### Implementation
```python
from collections import Counter

hot_llm = get_llm(temperature=0.7)  # Higher temp = more diversity
question = "Is 97 a prime number? Answer only Yes or No."

answers = []
for i in range(5):
    response = hot_llm.invoke(question)
    text = response.content.strip()
    vote = "Yes" if "yes" in text.lower() else "No"
    answers.append(vote)

winner = Counter(answers).most_common(1)[0][0]
```

### Output
```
Sample 1: Yes
Sample 2: Yes
Sample 3: Yes
Sample 4: Yes
Sample 5: Yes

Majority vote: Yes
```

### Purpose
- Reduces randomness errors
- More reliable for factual questions
- Trade-off: 5x slower (5 LLM calls)

### When to Use
- Critical decisions
- Factual verification
- When LLM is uncertain

## Comparison Table

| Pattern | Validation | Speed | Complexity | Best For |
|---------|-----------|-------|------------|----------|
| JSON | None | Fast | Low | Simple extraction |
| Pydantic | Strong | Medium | Medium | Complex objects |
| with_structured_output | Strong | Fast | Low | Modern projects |
| Chain-of-Thought | None | Medium | Low | Reasoning problems |
| Self-Consistency | None | Slow | Medium | Critical accuracy |

## Best Practices

1. **Always use structured output for agent actions**
   - Prevents parsing errors
   - Enables validation

2. **Use Field descriptions to guide LLM**
   ```python
   action: Literal["search", "calculate"] = Field(
       description="Choose 'search' for facts, 'calculate' for math"
   )
   ```

3. **Use Literal for restricted choices**
   ```python
   status: Literal["pending", "in_progress", "done"]
   ```

4. **Add confidence scores for uncertainty**
   ```python
   confidence: float = Field(ge=0.0, le=1.0)
   ```

5. **Combine patterns**
   - Chain-of-Thought + Structured Output
   - Self-Consistency + Pydantic

## Dependencies
```python
from pydantic import BaseModel, Field
from typing import List, Literal
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from collections import Counter
```

## Next Steps
- Day 3: Apply structured output to tool use
- Day 4: Use structured output for memory systems
