"""LLM integrations - local and cloud"""

# Local LLMs
from .local_llm import (
    get_ollama_llm,
    get_lm_studio_llm,
)

# Cloud LLMs
from .cloud_llm import (
    get_openai_llm,
    get_anthropic_llm,
    get_google_llm,
    get_huggingface_llm,
    get_llm, # Universal get LLM function
)

__all__ = [
    # Local
    "get_ollama_llm",
    "get_lm_studio_llm",
    # Cloud
    "get_openai_llm",
    "get_anthropic_llm",
    "get_google_llm",
    "get_huggingface_llm",
    # Universal
    "get_llm",
]
