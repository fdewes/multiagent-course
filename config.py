"""Shared config used by all days."""
OLLAMA_BASE_URL = "http://10.10.1.249:11434"
MODEL = "qwen2.5:27b"

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
