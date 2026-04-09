"""Shared config used by all days."""
OLLAMA_BASE_URL = "http://10.10.1.7:11434"
MODEL = "qwen3.5:9b"

# LangChain / LangGraph helper
def get_llm(temperature: float = 0.0, **kwargs):
    from langchain_ollama import ChatOllama
    return ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=MODEL,
        temperature=temperature,
        **kwargs,
    )

def get_embeddings():
    from langchain_ollama import OllamaEmbeddings
    return OllamaEmbeddings(
        base_url=OLLAMA_BASE_URL,
        model="nomic-embed-text",   # or change to any embedding model you have pulled
    )
