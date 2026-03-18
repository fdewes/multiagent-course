"""
Day 9 — RAG (Retrieval Augmented Generation) for Agents.
Agentic RAG: the agent decides WHEN and HOW to retrieve.
"""
import sys
sys.path.insert(0, "..")
from config import get_llm, get_embeddings
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import TypedDict, Annotated, List, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from pydantic import BaseModel

llm = get_llm(temperature=0.0)

# ── Build a knowledge base ─────────────────────────────────────────────────────
DOCS = [
    Document(
        page_content="""LangGraph is a library for building stateful, multi-actor applications with LLMs.
        It extends LangChain with the ability to coordinate multiple chains or agents in complex workflows.
        Key features: state management, cycles, checkpointing, streaming, human-in-the-loop.
        LangGraph uses a graph structure where nodes are functions and edges are transitions.""",
        metadata={"source": "langgraph_docs", "topic": "langgraph"}
    ),
    Document(
        page_content="""RAG (Retrieval Augmented Generation) combines a retrieval system with an LLM.
        Instead of relying purely on parametric memory, RAG fetches relevant documents at inference time.
        Benefits: reduced hallucination, up-to-date information, citeable sources, domain specificity.
        Common stack: vector DB (FAISS, Chroma, Qdrant) + embedding model + LLM.""",
        metadata={"source": "rag_docs", "topic": "rag"}
    ),
    Document(
        page_content="""Ollama is a tool for running large language models locally.
        It supports models like Llama 3, Mistral, Qwen, Phi, Gemma, and many others.
        You can run Ollama as a server and query it via REST API or client libraries.
        LangChain integrates with Ollama through the langchain-ollama package.""",
        metadata={"source": "ollama_docs", "topic": "ollama"}
    ),
    Document(
        page_content="""Vector databases store embeddings — dense numerical representations of text.
        Similarity search finds documents whose embeddings are closest to a query embedding.
        Popular options: FAISS (local), ChromaDB (local), Qdrant (local/cloud), Pinecone (cloud).
        For production, consider hybrid search: vector similarity + keyword (BM25) re-ranked together.""",
        metadata={"source": "vectordb_docs", "topic": "vector_databases"}
    ),
    Document(
        page_content="""Multi-agent systems use multiple AI agents working together.
        Common patterns: supervisor/worker, pipeline, parallel fan-out, debate/critique.
        Benefits: specialization, parallelism, reliability through redundancy, scalability.
        Challenges: coordination overhead, debugging complexity, cost multiplication.""",
        metadata={"source": "multiagent_docs", "topic": "multi_agent"}
    ),
]

# ── Build vector store ─────────────────────────────────────────────────────────
print("Building vector store...")
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
splits = splitter.split_documents(DOCS)

try:
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    print(f"Vector store built with {len(splits)} chunks")
    USE_REAL_EMBEDDINGS = True
except Exception as e:
    print(f"Embeddings unavailable ({e}), using keyword fallback")
    USE_REAL_EMBEDDINGS = False


def retrieve_documents(query: str, k: int = 3) -> List[Document]:
    """Retrieve relevant documents (with fallback to keyword search)."""
    if USE_REAL_EMBEDDINGS:
        return retriever.invoke(query)
    # Keyword fallback
    query_words = set(query.lower().split())
    scored = []
    for doc in DOCS:
        score = sum(1 for w in query_words if w in doc.page_content.lower())
        if score > 0:
            scored.append((score, doc))
    scored.sort(reverse=True)
    return [doc for _, doc in scored[:k]]


# ── 1. Basic RAG Chain ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("1. Basic RAG Chain")

def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    )

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. Answer using ONLY the provided context.
If the context doesn't contain the answer, say "I don't have information about this."

Context:
{context}"""),
    ("human", "{question}")
])

def rag_chain(question: str) -> str:
    docs = retrieve_documents(question)
    context = format_docs(docs)
    return (rag_prompt | llm | StrOutputParser()).invoke({
        "question": question,
        "context": context
    })

print(rag_chain("What is LangGraph and what are its key features?"))


# ── 2. Agentic RAG (agent decides when to retrieve) ───────────────────────────
print("\n" + "=" * 60)
print("2. Agentic RAG with Retrieval Tool")

@tool
def knowledge_base_search(query: str) -> str:
    """Search the knowledge base for relevant information. Use for factual questions."""
    docs = retrieve_documents(query)
    if not docs:
        return "No relevant documents found."
    return format_docs(docs)

@tool
def calculator(expression: str) -> str:
    """Evaluate arithmetic: e.g. '2 + 2'."""
    import re
    if re.match(r'^[\d\s\+\-\*\/\.\(\)]+$', expression):
        return str(eval(expression))
    return "Invalid"

from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate

tools = [knowledge_base_search, calculator]
agent_llm = get_llm(temperature=0.0)

PROMPT = """You are a helpful assistant with access to a knowledge base.
Use the knowledge_base_search tool to look up factual information.

Tools:
{tools}

Format:
Question: {input}
Thought: Think about what you need
Action: tool_name
Action Input: input
Observation: result
...
Final Answer: answer

{agent_scratchpad}"""

prompt = PromptTemplate.from_template(PROMPT)
agent = create_react_agent(agent_llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True,
                         max_iterations=4, handle_parsing_errors=True)

result = executor.invoke({"input": "What are the benefits of RAG and how does it relate to vector databases?"})
print(f"\nAnswer: {result['output']}")


# ── 3. Corrective RAG (evaluate and retry) ────────────────────────────────────
print("\n" + "=" * 60)
print("3. Corrective RAG with Document Grading")

class GradeDecision(BaseModel):
    is_relevant: bool
    reason: str

class CorrRagState(TypedDict):
    question: str
    documents: List[str]
    generation: str
    grade: str
    attempts: int

def retrieve_node(state: CorrRagState) -> dict:
    docs = retrieve_documents(state["question"])
    return {"documents": [d.page_content for d in docs]}

def grade_documents_node(state: CorrRagState) -> dict:
    """Grade whether retrieved docs are relevant."""
    structured = llm.with_structured_output(GradeDecision)
    docs_text = "\n".join(state["documents"])
    grade = structured.invoke(
        f"""Are these documents relevant to answering: "{state['question']}"?

Documents:
{docs_text[:500]}

Judge if they contain useful information."""
    )
    return {"grade": "relevant" if grade.is_relevant else "not_relevant"}

def generate_node(state: CorrRagState) -> dict:
    context = "\n\n".join(state["documents"])
    answer = llm.invoke(
        f"Using this context, answer: {state['question']}\n\nContext: {context}"
    ).content
    return {"generation": answer, "attempts": state.get("attempts", 0) + 1}

def web_search_fallback(state: CorrRagState) -> dict:
    """Fallback when documents aren't relevant — simulate web search."""
    print("  [Corrective RAG] Documents not relevant, using fallback...")
    return {"documents": [f"Fallback general knowledge about: {state['question']}"]}

def route_after_grade(state: CorrRagState) -> str:
    if state["grade"] == "relevant" or state.get("attempts", 0) >= 2:
        return "generate"
    return "web_search"

corrrag = StateGraph(CorrRagState)
corrrag.add_node("retrieve", retrieve_node)
corrrag.add_node("grade", grade_documents_node)
corrrag.add_node("generate", generate_node)
corrrag.add_node("web_search", web_search_fallback)

corrrag.add_edge(START, "retrieve")
corrrag.add_edge("retrieve", "grade")
corrrag.add_conditional_edges("grade", route_after_grade, {
    "generate": "generate",
    "web_search": "web_search"
})
corrrag.add_edge("web_search", "generate")
corrrag.add_edge("generate", END)

corrrag_graph = corrrag.compile()

result = corrrag_graph.invoke({
    "question": "How does FAISS compare to Qdrant for vector search?",
    "documents": [],
    "generation": "",
    "grade": "",
    "attempts": 0,
})
print(f"\nGeneration: {result['generation'][:300]}")
