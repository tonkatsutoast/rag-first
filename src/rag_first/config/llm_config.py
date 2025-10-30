from enum import Enum
from typing import Dict, Any
from .settings import settings

class LLMProvider(str, Enum):
    """Supporte LLM Providers"""
    OLLAMA = "ollama"
    LM_STUDIO = "lm_studio"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    HUGGINGFACE = "huggingface"
    LLAMA = "llama"
    DEEPSEEEK = "deepseek"
    QWEN = "qwen"

# Provder configurations
LLM_CONFIGS: Dict[LLMProvider, Dict[str, Any]] = {
    LLMProvider.OLLAMA: {
        "base_url": settings.OLLAMA_BASE_URL,
        "default_model": "llama3.2",
        "temperature": 0.7,
    },
    LLMProvider.LM_STUDIO: {
        "base_url": "http://localhost:1234/v1",
        "default_model": "local-model",
        "temperature": 0.7,
    },
    LLMProvider.OPENAI: {
        "api_key": settings.OPENAI_API_KEY,
        "default_model": "gpt-4-turbo-preview",
        "temperature": 0.7,
    },
    LLMProvider.ANTHROPIC: {
        "api_key": settings.ANTHROPIC_API_KEY,
        "default_model": "claude-3-5-sonnet-20241022",
        "temperature": 0.7,
    },
    LLMProvider.GOOGLE: {
        "api_key": settings.GOOGLE_API_KEY,
        "default_model": "gemini-1.5-pro",
        "temperature": 0.7,
    },
    LLMProvider.HUGGINGFACE: {
        "api_key": settings.HUGGINGFACE_API_KEY,
        "default_model": "meta-llama/Llama-3.2-3B-Instruct",
        "temperature": 0.7,
    },
}