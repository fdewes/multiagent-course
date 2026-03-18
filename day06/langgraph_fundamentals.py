"""
Day 6 — LangGraph: state graphs, tool nodes, conditional edges, checkpointing.
"""
import sys
sys.path.insert(0, "..")
from config import get_llm
from typing import TypedDict, Annotated, Literal
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# ── 1. Minimal agent graph ─────────────────────────────────────────────────────
print("=" * 60)
print("1. Minimal Chat Graph")

class SimpleState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

simple_llm = get_llm(temperature=0.3)

def chat_node(state: SimpleState) -> dict:
    response = simple_llm.invoke(state["messages"])
    return {"messages": [response]}

builder = StateGraph(SimpleState)
builder.add_node("chat", chat_node)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)
graph = builder.compile()

result = graph.invoke({"messages": [HumanMessage(content="What is 2 + 2?")]})
print("Response:", result["messages"][-1].content)


# ── 2. ReAct agent with ToolNode ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. ReAct Agent with ToolNode")

@tool
def search(query: str) -> str:
    """Search for information about a topic."""
    facts = {
        "python": "Python is a high-level programming language created by Guido van Rossum.",
        "germany": "Germany is a country in Central Europe with ~84 million people.",
        "eiffel tower": "The Eiffel Tower is 330 meters tall and located in Paris, France.",
    }
    for key, value in facts.items():
        if key in query.lower():
            return value
    return "No specific information found."

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    import re
    if re.match(r'^[\d\s\+\-\*\/\.\(\)]+$', expression):
        return str(eval(expression))
    return "Invalid expression"

tools = [search, calculator]
tool_node = ToolNode(tools)
agent_llm = get_llm(temperature=0.0).bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    step_count: int

def agent_node(state: AgentState) -> dict:
    response = agent_llm.invoke(state["messages"])
    return {
        "messages": [response],
        "step_count": state.get("step_count", 0) + 1
    }

def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """Route: if last message has tool calls → execute tools, else → end."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "end"

builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", "end": END}
)
builder.add_edge("tools", "agent")  # After tools → back to agent

react_graph = builder.compile()

# Visualize the graph structure
print("Graph nodes:", list(react_graph.nodes.keys()))

result = react_graph.invoke({
    "messages": [HumanMessage(content="How tall is the Eiffel Tower in feet? (1 meter = 3.28084 feet)")],
    "step_count": 0,
})
print(f"\nSteps taken: {result['step_count']}")
print(f"Final answer: {result['messages'][-1].content}")


# ── 3. Graph with checkpointing (persistent state) ─────────────────────────────
print("\n" + "=" * 60)
print("3. Checkpointed Agent (multi-turn, stateful)")

checkpointer = MemorySaver()
checkpointed_graph = builder.compile(checkpointer=checkpointer)

thread = {"configurable": {"thread_id": "session-001"}}

# Turn 1
result1 = checkpointed_graph.invoke(
    {"messages": [HumanMessage(content="My name is Felix. What is Python?")]},
    config=thread
)
print("Turn 1:", result1["messages"][-1].content[:100])

# Turn 2 — the graph remembers previous messages!
result2 = checkpointed_graph.invoke(
    {"messages": [HumanMessage(content="What did I just ask you and what's my name?")]},
    config=thread
)
print("Turn 2:", result2["messages"][-1].content[:150])


# ── 4. Conditional routing with multiple branches ─────────────────────────────
print("\n" + "=" * 60)
print("4. Multi-Branch Routing")

class RouterState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    route: str
    result: str

def classify_query(state: RouterState) -> dict:
    """Classify the query to route to the right handler."""
    question = state["messages"][-1].content
    structured = get_llm().with_structured_output(
        {"type": "object", "properties": {
            "category": {"type": "string", "enum": ["math", "factual", "creative", "other"]}
        }, "required": ["category"]}
    )
    result = structured.invoke(f"Classify this query into one category: {question}")
    return {"route": result.get("category", "other")}

def math_handler(state: RouterState) -> dict:
    response = llm.invoke(f"Solve this math problem step by step: {state['messages'][-1].content}")
    return {"result": response.content, "messages": [AIMessage(content=response.content)]}

def factual_handler(state: RouterState) -> dict:
    response = llm.invoke(f"Answer this factual question concisely with sources: {state['messages'][-1].content}")
    return {"result": response.content, "messages": [AIMessage(content=response.content)]}

def creative_handler(state: RouterState) -> dict:
    hot = get_llm(temperature=0.8)
    response = hot.invoke(f"Creative response to: {state['messages'][-1].content}")
    return {"result": response.content, "messages": [AIMessage(content=response.content)]}

def other_handler(state: RouterState) -> dict:
    response = llm.invoke(state["messages"][-1].content)
    return {"result": response.content, "messages": [AIMessage(content=response.content)]}

llm = get_llm()

def route_to_handler(state: RouterState) -> str:
    return state["route"]

router_builder = StateGraph(RouterState)
router_builder.add_node("classify", classify_query)
router_builder.add_node("math", math_handler)
router_builder.add_node("factual", factual_handler)
router_builder.add_node("creative", creative_handler)
router_builder.add_node("other", other_handler)

router_builder.add_edge(START, "classify")
router_builder.add_conditional_edges(
    "classify",
    route_to_handler,
    {"math": "math", "factual": "factual", "creative": "creative", "other": "other"}
)
for node in ["math", "factual", "creative", "other"]:
    router_builder.add_edge(node, END)

router_graph = router_builder.compile()

test_queries = [
    "What is 47 × 23?",
    "Who invented the telephone?",
    "Write a haiku about programming",
]

for q in test_queries:
    result = router_graph.invoke({
        "messages": [HumanMessage(content=q)],
        "route": "",
        "result": "",
    })
    print(f"\nQ: {q}")
    print(f"Route: {result['route']}")
    print(f"A: {result['result'][:100]}...")
