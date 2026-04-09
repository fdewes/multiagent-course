# Configuration Module Documentation

## Location
- `config.py` (root)
- `day01/config.py` (copy for day01)

## Purpose
Central configuration for all multi-agent course scripts. Provides helper functions to create LLM and embeddings instances with Ollama.

## Core Configuration

### Constants
```python
OLLAMA_BASE_URL = "http://10.10.1.7:11434"  # Ollama server address
MODEL = "qwen3.5:9b"  # Default LLM model
```

### Functions

#### `get_llm(temperature=0.0, **kwargs)`
Creates and returns a ChatOllama LLM instance.

**Parameters:**
- `temperature` (float, default=0.0): Controls randomness. 0.0 = deterministic, higher = more creative
- `**kwargs`: Additional parameters passed to ChatOllama

**Returns:**
- `ChatOllama`: LangChain LLM instance

**Example:**
```python
llm = get_llm(temperature=0.7)
response = llm.invoke("Hello, how are you?")
```

#### `get_embeddings()`
Creates and returns Ollama embeddings instance.

**Parameters:**
- None (uses default model)

**Returns:**
- `OllamaEmbeddings`: LangChain embeddings instance

**Embedding Model:**
- Uses `nomic-embed-text` by default
- Can be changed to any embedding model available in Ollama

**Example:**
```python
embeddings = get_embeddings()
vector = embeddings.embed_query("Hello world")
```

## Dependencies
- `langchain_ollama`: Bridge between LangChain and Ollama
- `langchain_core`: Core LangChain abstractions

## Notes
- Temperature 0.0 is used for deterministic outputs in most scripts
- Higher temperatures (0.3-0.7) used when creativity/diversity is needed
- All scripts import from this module to ensure consistent LLM configuration
