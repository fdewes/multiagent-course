"""
Day 14 — Capstone: Full Multi-Agent Research & Report System.

Architecture:
  User Query
      │
  ┌───▼─────────────────────────────────────┐
  │           ORCHESTRATOR                  │
  │  (decomposes query into research tasks) │
  └───┬─────────────────────────────────────┘
      │
      ▼ (parallel)
  ┌───────────┐  ┌───────────┐  ┌───────────┐
  │ Research  │  │  Code     │  │  Analyst  │
  │  Agent    │  │  Agent    │  │  Agent    │
  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘
        │              │              │
        └──────────────┼──────────────┘
                       │
               ┌───────▼───────┐
               │  CRITIC Agent │  (reviews quality)
               └───────┬───────┘
                       │
               ┌───────▼───────┐
               │  WRITER Agent │  (final report)
               └───────┬───────┘
                       │
               ┌───────▼───────┐
               │ HUMAN REVIEW  │  (optional gate)
               └───────┬───────┘
                       │
                    Final Report
"""
import sys
sys.path.insert(0, "..")
from config import get_llm
from typing import TypedDict, Annotated, List, Optional
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
import json
import time

llm = get_llm(temperature=0.3)

# ── State ──────────────────────────────────────────────────────────────────────
class ResearchState(TypedDict):
    # Input
    query: str
    depth: str  # "quick" | "thorough"

    # Orchestration
    research_tasks: List[str]

    # Agent outputs
    research_findings: List[dict]
    code_examples: List[dict]
    analysis_results: List[dict]

    # Quality
    critic_score: float
    critic_feedback: str

    # Final output
    report: str
    report_metadata: dict

    # Control
    iteration: int
    status: str

# ── Agent Implementations ──────────────────────────────────────────────────────

class ResearchPlan(BaseModel):
    tasks: List[str] = Field(description="3-5 specific research sub-tasks")
    focus_areas: List[str] = Field(description="Key areas to investigate")

def orchestrator_node(state: ResearchState) -> dict:
    """Decomposes the research query into specific tasks."""
    print(f"\n{'='*60}")
    print(f"[ORCHESTRATOR] Query: {state['query'][:80]}")

    structured_llm = llm.with_structured_output(ResearchPlan)
    plan = structured_llm.invoke(
        f"""Create a research plan for: {state['query']}

Depth: {state.get('depth', 'thorough')}

Break this into 3 specific research sub-tasks. Each should be focused and actionable."""
    )

    print(f"  Research tasks: {len(plan.tasks)}")
    for i, task in enumerate(plan.tasks):
        print(f"    {i+1}. {task}")

    return {
        "research_tasks": plan.tasks,
        "status": "researching",
    }

def research_agent_node(state: ResearchState) -> dict:
    """Researches each task and collects findings."""
    print(f"\n[RESEARCH AGENT]")
    findings = []

    for i, task in enumerate(state.get("research_tasks", [])[:3]):
        print(f"  Researching: {task[:60]}...")
        finding = llm.invoke(
            f"""You are a research specialist. Research this topic thoroughly:
{task}

Provide:
1. Key facts and findings
2. Important nuances
3. Current state (as of 2025)

Be specific and cite reasoning."""
        ).content

        findings.append({
            "task": task,
            "content": finding,
            "tokens": len(finding.split()),
        })

    print(f"  Completed {len(findings)} research tasks")
    return {"research_findings": findings}

def code_agent_node(state: ResearchState) -> dict:
    """Generates relevant code examples."""
    print(f"\n[CODE AGENT]")

    # Determine if code examples are relevant
    relevance = llm.invoke(
        f"In 1 word (yes/no): Does this topic benefit from code examples? Topic: {state['query']}"
    ).content.lower().strip()

    if "no" in relevance:
        print("  No code examples needed for this topic")
        return {"code_examples": []}

    examples = []
    for i, finding in enumerate(state.get("research_findings", [])[:2]):
        print(f"  Generating code example {i+1}...")
        code_response = llm.invoke(
            f"""Based on this research finding, write a concise Python code example:
{finding['content'][:300]}

Write a complete, runnable Python example that demonstrates the key concept.
Use ```python ... ``` to wrap the code."""
        ).content

        import re
        code_match = re.search(r'```python\n(.*?)```', code_response, re.DOTALL)
        if code_match:
            examples.append({
                "context": finding["task"],
                "code": code_match.group(1).strip(),
            })

    print(f"  Generated {len(examples)} code examples")
    return {"code_examples": examples}

def analyst_agent_node(state: ResearchState) -> dict:
    """Analyzes patterns, trends, and insights across all findings."""
    print(f"\n[ANALYST AGENT]")

    all_findings = "\n\n".join(
        f"Finding {i+1}: {f['content'][:300]}"
        for i, f in enumerate(state.get("research_findings", []))
    )

    analysis = llm.invoke(
        f"""You are a data and insights analyst.

Original query: {state['query']}

Research findings:
{all_findings}

Provide:
1. Cross-cutting patterns and themes
2. Key insights not obvious from individual findings
3. Gaps or areas needing further research
4. Practical implications
5. Priority recommendations

Be analytical, not just descriptive."""
    ).content

    print(f"  Analysis complete: {len(analysis.split())} words")
    return {"analysis_results": [{"analysis": analysis}]}

class CriticEval(BaseModel):
    score: float = Field(ge=0, le=10)
    strengths: List[str]
    weaknesses: List[str]
    suggestions: List[str]
    requires_revision: bool

def critic_agent_node(state: ResearchState) -> dict:
    """Reviews the quality of all collected information."""
    print(f"\n[CRITIC AGENT]")

    findings_summary = "\n".join(
        f"- {f['task']}: {f['content'][:100]}..."
        for f in state.get("research_findings", [])
    )
    analysis_text = ""
    if state.get("analysis_results"):
        analysis_text = state["analysis_results"][0].get("analysis", "")[:300]

    critic_llm = llm.with_structured_output(CriticEval)
    eval_result = critic_llm.invoke(
        f"""Critically evaluate the quality of this research for: {state['query']}

Research findings:
{findings_summary}

Analysis:
{analysis_text[:300]}

Score the overall quality 0-10. Does it require revision?"""
    )

    print(f"  Quality score: {eval_result.score}/10")
    print(f"  Requires revision: {eval_result.requires_revision}")
    if eval_result.weaknesses:
        print(f"  Key weaknesses: {eval_result.weaknesses[:2]}")

    return {
        "critic_score": eval_result.score,
        "critic_feedback": json.dumps({
            "score": eval_result.score,
            "strengths": eval_result.strengths,
            "weaknesses": eval_result.weaknesses,
            "suggestions": eval_result.suggestions,
        }),
        "iteration": state.get("iteration", 0) + 1,
    }

def writer_agent_node(state: ResearchState) -> dict:
    """Writes the final structured report."""
    print(f"\n[WRITER AGENT]")

    all_findings = "\n\n".join(
        f"=== {f['task']} ===\n{f['content']}"
        for f in state.get("research_findings", [])
    )
    analysis = ""
    if state.get("analysis_results"):
        analysis = state["analysis_results"][0].get("analysis", "")

    code_section = ""
    if state.get("code_examples"):
        code_section = "\n\n## Code Examples\n"
        for ex in state["code_examples"]:
            code_section += f"\n### {ex['context']}\n```python\n{ex['code']}\n```\n"

    critic_notes = ""
    if state.get("critic_feedback"):
        feedback = json.loads(state["critic_feedback"])
        if feedback.get("suggestions"):
            critic_notes = f"\nNote: Incorporate these improvements: {feedback['suggestions'][:2]}"

    report = llm.invoke(
        f"""You are a senior technical writer. Write a comprehensive, well-structured report.

Topic: {state['query']}

Research Findings:
{all_findings[:2000]}

Analysis:
{analysis[:800]}
{critic_notes}

Write a professional report with:
# Executive Summary (2-3 sentences)
# Key Findings (bullet points)
# Detailed Analysis (3-4 paragraphs)
# Recommendations (numbered list)
# Conclusion

Be specific, substantive, and actionable."""
    ).content

    word_count = len(report.split())
    print(f"  Report written: {word_count} words")

    return {
        "report": report,
        "report_metadata": {
            "word_count": word_count,
            "findings_count": len(state.get("research_findings", [])),
            "code_examples_count": len(state.get("code_examples", [])),
            "quality_score": state.get("critic_score", 0),
            "iterations": state.get("iteration", 0),
        },
        "status": "complete",
    }

# ── Routing Logic ──────────────────────────────────────────────────────────────

def route_after_critic(state: ResearchState) -> str:
    """If quality is too low and we haven't iterated too much, re-research."""
    score = state.get("critic_score", 0)
    iterations = state.get("iteration", 0)

    if score < 6.0 and iterations < 2:
        print(f"  Quality {score:.1f} < 6.0, triggering revision (iteration {iterations})")
        return "revise"
    return "write"

def orchestrator_with_revision(state: ResearchState) -> dict:
    """Re-research with guidance from critic."""
    feedback = json.loads(state.get("critic_feedback", "{}"))
    suggestions = feedback.get("suggestions", [])

    print(f"\n[ORCHESTRATOR - REVISION] Revising based on feedback: {suggestions[:2]}")
    new_tasks = []
    for suggestion in suggestions[:2]:
        new_tasks.append(f"Address this gap: {suggestion}")

    return {
        "research_tasks": new_tasks,
        "status": "revising",
    }

# ── Build the Full Graph ───────────────────────────────────────────────────────

builder = StateGraph(ResearchState)

# Register all nodes
builder.add_node("orchestrator", orchestrator_node)
builder.add_node("research", research_agent_node)
builder.add_node("code", code_agent_node)
builder.add_node("analyst", analyst_agent_node)
builder.add_node("critic", critic_agent_node)
builder.add_node("revise", orchestrator_with_revision)
builder.add_node("writer", writer_agent_node)

# Edges
builder.add_edge(START, "orchestrator")
builder.add_edge("orchestrator", "research")

# Parallel fan-out after research
builder.add_edge("research", "code")
builder.add_edge("research", "analyst")

# Both code and analyst feed into critic
builder.add_edge("code", "critic")
builder.add_edge("analyst", "critic")

# Critic decides: revise or write
builder.add_conditional_edges(
    "critic",
    route_after_critic,
    {"revise": "revise", "write": "writer"}
)

# Revision loop back to research
builder.add_edge("revise", "research")
builder.add_edge("writer", END)

# Compile with checkpointing
checkpointer = MemorySaver()
research_graph = builder.compile(checkpointer=checkpointer)

# ── Run the System ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    QUERY = "What are multi-agent AI systems and how should developers implement them in production in 2025?"

    print("=" * 70)
    print("MULTI-AGENT RESEARCH SYSTEM — CAPSTONE")
    print("=" * 70)
    print(f"Query: {QUERY}")
    print("=" * 70)

    start_time = time.time()

    config = {"configurable": {"thread_id": "capstone-001"}}
    result = research_graph.invoke(
        {
            "query": QUERY,
            "depth": "thorough",
            "research_tasks": [],
            "research_findings": [],
            "code_examples": [],
            "analysis_results": [],
            "critic_score": 0.0,
            "critic_feedback": "",
            "report": "",
            "report_metadata": {},
            "iteration": 0,
            "status": "starting",
        },
        config=config,
    )

    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    print(result["report"])

    print("\n" + "=" * 70)
    print("METADATA")
    print("=" * 70)
    meta = result["report_metadata"]
    print(f"Word count:       {meta.get('word_count', 0)}")
    print(f"Research tasks:   {meta.get('findings_count', 0)}")
    print(f"Code examples:    {meta.get('code_examples_count', 0)}")
    print(f"Quality score:    {meta.get('quality_score', 0):.1f}/10")
    print(f"Iterations:       {meta.get('iterations', 0)}")
    print(f"Total time:       {elapsed:.1f}s")
