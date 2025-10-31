from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Optional
from rag_first.config.settings import settings
from rag_first.config.llm_config import LLMProvider

def get_embeddings(
    provider: str = "ollama", 
    model: Optional[str]= None, **kwargs
) -> Embeddings:
    """
    Get embeddings instance based on provider
    Args: 
        provider: Embeddings provider (ollama, OpenAI, etc..)
        model: Model name
        **kwargs: Additional provider specific arguments
    Returns:
        LangChain Embeddings instance
    """
    provider = provider.lower()
    match provider:
        case "ollama":
            return OllamaEmbeddings(
                model=model or settings.EMBEDDING_MODEL,
                base_url=settings.OLLAMA_BASE_URL,
                **kwargs
            )
        case "openai":
            return OpenAIEmbeddings(
                model=model or "text-embedding-3-small",
                api_key=settings.OPENAI_API_KEY,
                **kwargs
            )
        case "huggingface":
            return HuggingFaceEmbeddings(
                model=model or "sentence-transformers/all-MiniLM-L6-v2",
                **kwargs
            )
        case _:
            raise ValueError(
                f"Unsppoprted embedding provder: {provider}. "
                f"Choose from: ollama, openai, huggingface"
            )
