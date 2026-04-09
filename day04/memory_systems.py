"""
Day 4 — Memory systems: short-term, episodic, semantic, entity.
"""
import sys
sys.path.insert(0, "..")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_llm
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import json
from pathlib import Path
from datetime import datetime

llm = get_llm(temperature=0.3)

# ── 1. In-context memory (message history) ────────────────────────────────────
print("=" * 60)
print("1. In-Context Memory — Message History")

store = {}  # session_id → ChatMessageHistory

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Remember context across the conversation."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

chain = prompt | llm
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

session = "user_001"
for msg in [
    "My name is Felix and I love Python.",
    "What's my name and what do I love?",
    "What programming languages are similar to what I like?",
]:
    response = chain_with_history.invoke(
        {"input": msg},
        config={"configurable": {"session_id": session}}
    )
    print(f"User: {msg}")
    print(f"AI: {response.content}\n")


# ── 2. Summarize-and-Compress ─────────────────────────────────────────────────
print("=" * 60)
print("2. Summarize-and-Compress Memory")

class SummaryMemory:
    def __init__(self, max_messages: int = 6):
        self.history = []
        self.summary = ""
        self.max_messages = max_messages

    def add(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_messages:
            self._compress()

    def _compress(self):
        """Summarize oldest half of history."""
        half = len(self.history) // 2
        to_compress = self.history[:half]
        self.history = self.history[half:]

        text = "\n".join(f"{m['role']}: {m['content']}" for m in to_compress)
        prompt = f"Summarize this conversation in 2-3 sentences:\n\n{text}"
        new_summary = llm.invoke(prompt).content

        if self.summary:
            self.summary = f"Previous: {self.summary}\nRecent: {new_summary}"
        else:
            self.summary = new_summary
        print(f"  [compressed → summary: {self.summary[:80]}...]")

    def get_messages(self):
        msgs = []
        if self.summary:
            msgs.append({"role": "system", "content": f"Conversation summary so far: {self.summary}"})
        msgs.extend(self.history)
        return msgs

mem = SummaryMemory(max_messages=4)
conversations = [
    ("user", "I'm building a FastAPI application."),
    ("assistant", "Great! FastAPI is excellent for APIs. What are you building?"),
    ("user", "A REST API for a task management app."),
    ("assistant", "Nice! Will it have authentication?"),
    ("user", "Yes, JWT tokens."),
    ("assistant", "Good choice. Will you use PostgreSQL or SQLite?"),
    ("user", "PostgreSQL with SQLAlchemy."),
    ("assistant", "Perfect stack. Any specific features you need help with?"),
]

for role, content in conversations:
    mem.add(role, content)

msgs = mem.get_messages()
print(f"Messages in context: {len(msgs)} (vs original {len(conversations)})")


# ── 3. Episodic Memory (file-based for demo, use Redis/Postgres in prod) ───────
print("\n" + "=" * 60)
print("3. Episodic Memory")

MEMORY_FILE = Path("/tmp/agent_episodic_memory.json")

def load_episodic_memory() -> list:
    if MEMORY_FILE.exists():
        return json.loads(MEMORY_FILE.read_text())
    return []

def save_episode(user_input: str, agent_output: str, tags: list):
    episodes = load_episodic_memory()
    episodes.append({
        "timestamp": datetime.now().isoformat(),
        "user": user_input,
        "agent": agent_output,
        "tags": tags,
    })
    # Keep only last 100 episodes
    episodes = episodes[-100:]
    MEMORY_FILE.write_text(json.dumps(episodes, indent=2))

def retrieve_relevant_episodes(query: str, top_k: int = 3) -> list:
    """Simple keyword matching — replace with vector search in production."""
    episodes = load_episodic_memory()
    query_words = set(query.lower().split())
    scored = []
    for ep in episodes:
        text = f"{ep['user']} {ep['agent']}".lower()
        score = sum(1 for w in query_words if w in text)
        if score > 0:
            scored.append((score, ep))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [ep for _, ep in scored[:top_k]]

# Simulate saving episodes
save_episode(
    "How do I set up logging in Python?",
    "Use the `logging` module. Configure with basicConfig or a FileHandler.",
    ["python", "logging"]
)
save_episode(
    "What's the difference between a list and a tuple?",
    "Lists are mutable, tuples are immutable. Tuples are faster for fixed data.",
    ["python", "data-structures"]
)

relevant = retrieve_relevant_episodes("Python data structures")
print(f"Found {len(relevant)} relevant past episodes")
for ep in relevant:
    print(f"  - {ep['user'][:60]}...")


# ── 4. Entity Memory ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. Entity Memory")

from pydantic import BaseModel
from typing import Dict, Any

class EntityStore(BaseModel):
    entities: Dict[str, Dict[str, Any]] = {}

    def update(self, entity: str, facts: Dict[str, Any]):
        if entity not in self.entities:
            self.entities[entity] = {}
        self.entities[entity].update(facts)

    def get(self, entity: str) -> Dict[str, Any]:
        return self.entities.get(entity, {})

    def summary(self) -> str:
        lines = []
        for entity, facts in self.entities.items():
            fact_str = ", ".join(f"{k}={v}" for k, v in facts.items())
            lines.append(f"  {entity}: {fact_str}")
        return "\n".join(lines) if lines else "(empty)"

entity_store = EntityStore()

# Extract entities from conversation
extract_prompt = """Extract entities and facts from this text.
Return JSON like: {{"entity_name": {{"fact_key": "fact_value"}}}}
Only extract concrete facts (name, age, location, job, preferences).

Text: {text}

JSON:"""

def extract_and_store_entities(text: str):
    from langchain_core.output_parsers import JsonOutputParser
    response = llm.invoke(extract_prompt.format(text=text))
    try:
        facts = JsonOutputParser().parse(response.content)
        for entity, data in facts.items():
            entity_store.update(entity, data)
    except:
        pass  # Graceful degradation

conversations = [
    "I'm Sarah, a 28-year-old data scientist from Amsterdam.",
    "I work at a startup and love machine learning.",
    "My colleague Tom is a backend engineer who prefers Go.",
]

for conv in conversations:
    extract_and_store_entities(conv)

print("Entity store after conversations:")
print(entity_store.summary())
print("\nSarah's facts:", entity_store.get("Sarah"))
