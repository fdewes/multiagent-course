# Day 4 — Memory Systems

## Why Memory Matters

Without memory, every agent call starts from zero.
Memory enables:
- Multi-turn conversations
- Learning from past actions
- Personalization
- Avoiding repeated mistakes

## The 5 Types of Memory

### 1. In-Context (Short-Term)
The message history. Fast, limited by context window.
```
[system] [user] [assistant] [tool] [assistant] ...
```

### 2. External (Long-Term / Episodic)
Store past interactions in a database, retrieve relevant ones.
- Tools: Redis, PostgreSQL, MongoDB, SQLite

### 3. Semantic (Knowledge)
Facts stored as embeddings in a vector store.
- "Berlin is the capital of Germany" → embedding → retrieval
- Tools: ChromaDB, FAISS, Qdrant, Pinecone, pgvector

### 4. Procedural (How-to)
Stored instructions, routines, learned behaviors.
- Example: cached prompts, few-shot examples

### 5. Working Memory (Scratchpad)
Temporary computation state.
- Example: intermediate results during a multi-step task

## Memory Architecture Patterns

### Pattern 1: Summarize-and-Compress
After N messages, summarize old context → trim → inject summary.

### Pattern 2: Episodic Store
After each conversation, store key facts.
Before each conversation, retrieve relevant facts.

### Pattern 3: Dual Memory
- Short-term: recent messages (in-context)
- Long-term: vector DB of important facts (retrieved)

### Pattern 4: Entity Memory
Extract and update a knowledge graph of entities mentioned.
e.g., "User's name is Alex, prefers Python, working on ML project."

## When to Write vs. Retrieve

| Situation | Action |
|-----------|--------|
| User provides a fact | Write to entity store |
| Query needs past context | Retrieve from episodic store |
| Query needs knowledge | Retrieve from semantic store |
| Running low on context | Summarize and trim |
