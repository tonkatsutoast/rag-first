from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.language_models import BaseLLM
from typing import Optional, 
from rag_first.config.settings import settings
from rag_first.config.llm_config import LLMProvider
import logging
logger = logging.getLogger(__name__)

"""Manage cloud LLM connections"""
def get_openai_llm(
    model: str = "gpt-4-turbo-preview",
    temperature: float = 0.7,
    **kwargs
) -> BaseLLM:
    """Get OpenAI LLM Instance"""
    logger.info(f"Initializing OpenAI with model: {model}")
    return ChatOpenAI(
        model=model,
        api_key=settings.OPENAI_API_KEY,
        temperature=temperature,
        **kwargs
    )

def get_anthropic_llm(
    model: str = "claude-3.5-sonnet-20241022",
    temperature: float = 0.7,
    **kwargs
) -> BaseLLM:
    logger.info(f"Initializing Anthropic with model: {model}")
    return ChatAnthropic(
        model=model,
        api_key=settings.ANTHROPIC_API_KEY,
        temperature=temperature,
        **kwargs
    )

def get_google_llm(
    model: str = "gemini-1.5-pro",
    temperature: float = 0.7,
    **kwargs
) -> BaseLLM:
    """Get Google Gemini instance"""
    logger.info(f"Initializing Google with model: {model}")
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=temperature,
        **kwargs
    )

def get_huggingface_llm(
    model: str = "meta-llama/Llama-3.2-3B-Instruct",
    temperature: float = 0.7,
    **kwargs
) -> BaseLLM:
    """
    Get HuggingFace LLM instance. Supports models like LLama, Deepseek, Qwen, etc..
    """
    logger.info(f"Initializing HuggingFace with model: {model}")
    return HuggingFaceEndpoint(
        repo_id=model,
        huggingfacehub_api_token=settings.HUGGINGFACE_API_KEY,
        temperature=temperature,
        **kwargs
    )

def get_llm(
    provider: str,
    model: Optional[str] = None,
    temperature: float = 0.7,
    **kwargs
) -> BaseLLM:
    """
    Universal method to return any LLM instance
    Args:
        provider: LLM provider name
        model: model name
        temperature: sampling temperature
        **kwargs: provider specific params
    Returns:
        LLM instance
    """
    provider = provider.lower()

    match provider:
        case "ollama": 
            from rag_first.llm.local_llm import get_ollama_llm
            return get_ollama_llm(
                model=model or "llama3.2:3b"
            )
        case "llm_studio": 
            from rag_first.llm.local_llm import get_lm_studio_llm
            return get_lm_studio_llm(
                model=model or "local-model",
                temperature=temperature,
                **kwargs
            )
        case "openai":
            return get_openai_llm(
                model=model or "gpt-4-turbo-preview",
                temperature=temperature,
                **kwargs
            )
        case "google":
            return get_google_llm(
                model=model or "gemini-1.5-pro",
                temperature=temperature,
                **kwargs
            )
        case "huggingface" | "llama" | "deepseek" | "qwen":
            # Map to appropriate HuggingFace model
            default_models = {
                "huggingface": "meta-llama/Llama-3.2-3B-Instruct",
                "llama": "meta-llama/Llama-3.2-3B-Instruct",
                "deepseek": "deepseek-ai/deepseek-coder-33b-instruct",
                "qwen": "Qwen/Qwen2-7B-Instruct",
            }
            return get_huggingface_llm(
                model=model or default_models.get(provider),
                temperature=temperature,
                **kwargs
            )
        case _:
            raise ValueError(
                f"Unsupported LLM provider: {provider}."
                f"Supported: ollama, lm_studio, openai, anthropic,"
                f"huggingface, llama, deepseek, qwen"
            )
        