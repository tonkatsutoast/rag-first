from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
import os
from pathlib import Path

class Settings(BaseSettings):
    """Application settings managed by Pydantic"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True
    )

    # Project paths

    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

    # Vector store
    CHROMA_PERSIST_DIRECTORY: str = str(DATA_DIR / "vectorstore" / "chroma_db")
    COLLECTION_NAME: str = "rag_collection"

    # Embeddings
    EMBEDDING_MODEL: str = "mxbai-embed-large"
    EMBEDDING_PROVIDER: str = "ollama" # change if you want a different provder like openai or anthropic

    # Local LLM
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LOCAL_MODEL_NAME: str = "llama3.2:3b"
    LLM_MODEL: str = "llama3.2:3b"  # Alternative name for LOCAL_MODEL_NAME

    # Cloud LLM API Keys
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    HUGGINGFACE_API_KEY: Optional[str] = None
    
    # Retrieval
    TOP_K_RESULTS: int = 5
    SIMILARITY_THRESHOLD: float = 0.7

    # Chunking
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

# Global settings instance
settings = Settings()

# Ensure directories exist
settings.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
Path(settings.CHROMA_PERSIST_DIRECTORY).mkdir(parents=True, exist_ok=True)
