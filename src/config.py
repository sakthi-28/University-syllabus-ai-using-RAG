"""Configuration management for the Syllabus AI Assistant."""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Application configuration loaded from environment variables."""
    
    # Embedding configuration (local only)
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    
    # OpenRouter configuration (for GLM and other models)
    OPENROUTER_API_KEY: Optional[str] = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_BASE_URL: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    
    # LLM configuration
    # We only support OpenRouter to keep setup simple.
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openrouter")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "openai/gpt-3.5-turbo")
    
    # ChromaDB configuration
    CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    
    # Text chunking configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # RAG configuration
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "6"))
    
    # Collection name for ChromaDB
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "syllabus_documents")
    
    @classmethod
    def validate(cls) -> None:
        """Validate configuration settings."""
        if cls.LLM_PROVIDER != "openrouter":
            raise ValueError(
                f"Unsupported LLM_PROVIDER '{cls.LLM_PROVIDER}'. This project supports 'openrouter' only."
            )

        if not cls.OPENROUTER_API_KEY:
            raise ValueError(
                "OPENROUTER_API_KEY is required when using OpenRouter LLM. "
                "Please set it in your .env file."
            )
        
        # Ensure ChromaDB path exists
        Path(cls.CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_chroma_db_path(cls) -> Path:
        """Get ChromaDB path as Path object."""
        return Path(cls.CHROMA_DB_PATH)
