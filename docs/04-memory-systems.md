# Day 4: Memory Systems Documentation

## File: `day04/memory_systems.py`

## Overview
Explores **memory patterns for agents**: short-term, episodic, semantic, and entity memory. Shows how to maintain context across conversations.

## Why Memory Matters

**Without Memory:**
```
User: "My name is Felix."
Agent: "Nice to meet you, Felix!"
User: "What's my name?"
Agent: "I don't have access to personal information."
```

**With Memory:**
```
User: "My name is Felix."
Agent: "Nice to meet you, Felix!"
User: "What's my name?"
Agent: "Your name is Felix."
```

## Pattern 1: In-Context Memory (Message History)

### Concept
Keep conversation history in the prompt context.

### Implementation
```python
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Store history per session
store = {}  # session_id → ChatMessageHistory

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Prompt with history placeholder
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Remember context across the conversation."),
    MessagesPlaceholder(variable_name="history"),  # Insert history here
    ("human", "{input}"),
])

# Chain with history
chain = prompt | llm
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)
```

### Usage
```python
session = "user_001"

response = chain_with_history.invoke(
    {"input": "My name is Felix and I love Python."},
    config={"configurable": {"session_id": session}}
)

response = chain_with_history.invoke(
    {"input": "What's my name and what do I love?"},
    config={"configurable": {"session_id": session}}
)
# → "Your name is Felix and you love Python."
```

### Components
- **ChatMessageHistory**: Stores messages in memory
- **MessagesPlaceholder**: Inserts history into prompt
- **RunnableWithMessageHistory**: Wraps chain to manage history
- **session_id**: Isolates conversations per user

### Pros & Cons
| Pros | Cons |
|------|------|
| Simple to implement | Limited by context window |
| No extra storage needed | Expensive (all history in prompt) |
| Works with any LLM | History grows unbounded |

## Pattern 2: Summarize-and-Compress Memory

### Concept
Summarize old messages to save space while keeping key info.

### Implementation
```python
class SummaryMemory:
    def __init__(self, max_messages: int = 6):
        self.history = []          # Recent messages
        self.summary = ""          # Compressed old messages
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
        
        # Create summary text
        text = "\n".join(f"{m['role']}: {m['content']}" for m in to_compress)
        prompt = f"Summarize this conversation in 2-3 sentences:\n\n{text}"
        new_summary = llm.invoke(prompt).content
        
        # Accumulate summaries
        if self.summary:
            self.summary = f"Previous: {self.summary}\nRecent: {new_summary}"
        else:
            self.summary = new_summary
    
    def get_messages(self):
        msgs = []
        if self.summary:
            msgs.append({"role": "system", "content": f"Conversation summary: {self.summary}"})
        msgs.extend(self.history)
        return msgs
```

### Usage
```python
mem = SummaryMemory(max_messages=4)

mem.add("user", "I'm building a FastAPI application.")
mem.add("assistant", "Great! What are you building?")
mem.add("user", "A REST API for a bookstore.")
mem.add("assistant", "Sounds interesting. What features?")
mem.add("user", "User authentication and book search.")
# → Triggers compression: first 2 messages summarized

messages = mem.get_messages()
# → [summary, recent_message_1, recent_message_2, ...]
```

### Benefits
- **Space efficiency**: Compresses long conversations
- **Context preservation**: Keeps important info
- **Cost reduction**: Fewer tokens in prompt

### Trade-offs
- Summaries may lose details
- Extra LLM call for summarization
- Cumulative summaries can grow

## Pattern 3: Episodic Memory

### Concept
Store individual conversation episodes for later retrieval.

### Implementation
```python
import json
from pathlib import Path
from datetime import datetime

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
    """Simple keyword matching."""
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
```

### Usage
```python
# Save episodes
save_episode(
    "How do I set up logging in Python?",
    "Use the `logging` module with basicConfig or FileHandler.",
    ["python", "logging"]
)

save_episode(
    "What's the difference between a list and a tuple?",
    "Lists are mutable, tuples are immutable.",
    ["python", "data-structures"]
)

# Retrieve relevant episodes
relevant = retrieve_relevant_episodes("Python data structures")
for ep in relevant:
    print(f"- {ep['user']}")
    print(f"  → {ep['agent']}")
```

### Storage Format
```json
[
  {
    "timestamp": "2024-04-08T12:00:00",
    "user": "Question?",
    "agent": "Answer.",
    "tags": ["topic1", "topic2"]
  }
]
```

### Retrieval Methods
- **Keyword matching**: Simple, fast (current implementation)
- **Vector search**: Better semantic matching (use embeddings)
- **Hybrid**: Combine keyword + vector

### Use Cases
- FAQ bot learns from past conversations
- Personal assistant remembers user preferences
- Customer support retrieves similar cases

## Memory Pattern Comparison

| Pattern | Storage | Retrieval | Best For |
|---------|---------|-----------|----------|
| In-Context | Prompt | All at once | Short conversations |
| Summarize | Prompt + summary | Summary + recent | Long conversations |
| Episodic | External (file/DB) | Query-based | Knowledge accumulation |
| Semantic | Vector DB | Similarity search | Factual knowledge |
| Entity | Graph DB | Relationship query | Complex relationships |

## Production Considerations

### 1. Persistence
```python
# Memory-based (lost on restart)
store = {}

# File-based
MEMORY_FILE.write_text(json.dumps(data))

# Database (recommended)
# Use Redis, PostgreSQL, etc.
```

### 2. Privacy
- Don't store sensitive info
- Implement user data deletion
- Comply with GDPR/privacy laws

### 3. Scalability
- Use vector databases for semantic memory
- Implement caching for frequent queries
- Paginate large histories

### 4. Quality
- Validate retrieved memories
- Allow user to correct memories
- Implement memory decay (forget old info)

## Dependencies
```python
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import json
from pathlib import Path
from datetime import datetime
```

## Next Steps
- Day 5: Agent architectures (hierarchical, collaborative)
- Day 6: LangGraph for complex workflows
