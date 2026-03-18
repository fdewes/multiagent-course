# Day 9 — RAG for Agents

## RAG Pipeline Overview

```
Query → Embed Query → Vector Search → Retrieve Docs → Augment Prompt → LLM → Answer
```

## Why RAG > Fine-Tuning for Agents

| Criterion | RAG | Fine-Tuning |
|-----------|-----|-------------|
| Update knowledge | Instant | Retrain needed |
| Cite sources | Yes | No |
| Cost | Low (inference) | High (training) |
| Hallucination | Reduced | Still possible |
| Domain specificity | High | High |

## Chunking Strategies

| Strategy | Use case |
|----------|---------|
| Fixed size (512 tokens) | Uniform docs |
| Recursive character split | General purpose |
| Semantic chunking | When topics vary within docs |
| Document-aware | PDFs, markdown, code |
| Sentence window | When context needed around hits |

## Retrieval Strategies

### 1. Dense Retrieval (Vector Search)
Semantic similarity. Best for conceptual queries.

### 2. Sparse Retrieval (BM25/TF-IDF)
Keyword match. Best for exact terms, names, codes.

### 3. Hybrid Search
Combine both with reciprocal rank fusion (RRF).
Best of both worlds — use this in production.

### 4. Re-ranking
After retrieval, use a cross-encoder to re-rank top-k docs.
Dramatically improves precision.

## Agentic RAG Patterns

### Standard RAG
Fixed: always retrieve → generate.

### Corrective RAG (CRAG)
Grade retrieved docs → if irrelevant → fallback (web search) → generate.

### Self-RAG
LLM decides: do I even need to retrieve? When? What to retrieve?

### Multi-hop RAG
Question → retrieve → answer used to form new question → retrieve again.
For complex reasoning chains.

## Production Checklist
- [ ] Chunk size optimized for your docs
- [ ] Hybrid search enabled
- [ ] Re-ranking implemented
- [ ] Metadata filtering (date, source, category)
- [ ] Result grounding evaluation
- [ ] Fallback when retrieval fails
