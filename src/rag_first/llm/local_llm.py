from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI # For LM Studio
from langchain_core.language_models import BaseLLM
from typing import Optional, Dict, Any
from rag_first.config.settings import settings
import logging
logger = logging(__name__)

"""Manage local LLM connections (Ollama, LM Studio)"""
def get_ollama_llm(
    model: str = "llama3.2:3b",
    temperature: float 0.7,
    **kwargs
) -> BaseLLM:
    """
    Get Ollama instance
    Args:
        model: Ollama model name
        temperature: Sampling temperature
        **kwargs: additional Ollama params
    Returns:
        Ollama LLM instance
    """
    logger.info(f"Initializing Ollama with model: {model}")
    return Ollama(
        model=model,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=temperature,
        **kwargs
    )

def get_lm_studio_llm(
    model: str = "local-model",
    temperature: float = 0.7,
    **kwargs
) -> BaseLLM:
    """
    Get LM Studio LLM instance (using OpenAI compatible API)
    Args:
        model: Model name in LM Studio
        temperature: sampling temperature
        base_url: LM Studio server URL
        **kwargs: additional params
    Returns:
        LM Studio LM instance
    """
    logger.info(f"Initializing LM Studio with model: {model}")
    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=settings.OPENAI_API_KEY,
        temperature=temperature,
        **kwargs
    )
    

