"""
Day 2 — Structured output patterns: the foundation of reliable agents.
"""
import sys
sys.path.insert(0, "..")
from config import get_llm
from pydantic import BaseModel, Field
from typing import List, Literal
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

llm = get_llm(temperature=0.0)

# ── Pattern 1: Simple JSON output ─────────────────────────────────────────────
print("=" * 60)
print("Pattern 1: JSON output")

json_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a data extractor. Always respond with valid JSON only, no markdown."),
    ("user", "Extract the person's name, age, and city from: '{text}'"),
])
chain = json_prompt | llm | JsonOutputParser()
result = chain.invoke({"text": "Hi, I'm Maria, 32 years old, living in Munich."})
print(result)
# → {'name': 'Maria', 'age': 32, 'city': 'Munich'}


# ── Pattern 2: Pydantic structured output ─────────────────────────────────────
print("\n" + "=" * 60)
print("Pattern 2: Pydantic")

class TaskPlan(BaseModel):
    """A structured task plan."""
    goal: str = Field(description="The main objective")
    steps: List[str] = Field(description="Ordered list of steps to achieve the goal")
    estimated_complexity: Literal["low", "medium", "high"] = Field(
        description="Estimated complexity level"
    )
    tools_needed: List[str] = Field(description="Tools or resources needed")

parser = PydanticOutputParser(pydantic_object=TaskPlan)

plan_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a project planner. {format_instructions}"),
    ("user", "Create a plan to: {goal}"),
]).partial(format_instructions=parser.get_format_instructions())

chain = plan_prompt | llm | parser
plan = chain.invoke({"goal": "Build a web scraper that monitors product prices"})
print(f"Goal: {plan.goal}")
print(f"Complexity: {plan.estimated_complexity}")
print(f"Steps: {plan.steps}")
print(f"Tools: {plan.tools_needed}")


# ── Pattern 3: .with_structured_output() (modern LangChain way) ───────────────
print("\n" + "=" * 60)
print("Pattern 3: with_structured_output")

class AgentDecision(BaseModel):
    """The agent's next action decision."""
    reasoning: str = Field(description="Step-by-step reasoning")
    action: Literal["search", "calculate", "answer", "ask_clarification"]
    action_input: str = Field(description="Input for the chosen action")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0-1")

structured_llm = llm.with_structured_output(AgentDecision)

messages = [
    ("system", "You are an agent. Decide the next action to take."),
    ("user", "What is the GDP of Germany divided by its population?"),
]
decision = structured_llm.invoke(messages)
print(f"Reasoning: {decision.reasoning}")
print(f"Action: {decision.action} → {decision.action_input}")
print(f"Confidence: {decision.confidence:.2f}")


# ── Pattern 4: Chain-of-Thought ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("Pattern 4: Chain-of-Thought")

cot_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a careful analyst. Think step by step before answering."),
    ("user", "{question}\n\nThink through this carefully:\n1. What information do I have?\n2. What am I being asked?\n3. What steps do I need to take?\n4. Final answer:"),
])
chain = cot_prompt | llm
result = chain.invoke({"question": "A train travels 120km at 60km/h, then 80km at 40km/h. What is the total travel time?"})
print(result.content)


# ── Pattern 5: Self-consistency (sample + vote) ────────────────────────────────
print("\n" + "=" * 60)
print("Pattern 5: Self-consistency voting")

from collections import Counter

hot_llm = get_llm(temperature=0.7)
question = "Is 97 a prime number? Answer only Yes or No."

answers = []
for i in range(5):
    response = hot_llm.invoke(question)
    # Extract Yes/No
    text = response.content.strip()
    vote = "Yes" if "yes" in text.lower() else "No"
    answers.append(vote)
    print(f"  Sample {i+1}: {vote} ({text[:50]})")

winner = Counter(answers).most_common(1)[0][0]
print(f"\n🗳️  Majority vote: {winner}")
