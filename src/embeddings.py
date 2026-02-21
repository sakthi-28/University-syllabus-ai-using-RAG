"""Embedding generation (local SentenceTransformers)."""

from abc import ABC, abstractmethod
from typing import List, Optional

from .config import Config


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents."""
        pass
    
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query."""
        pass


class SentenceTransformerEmbeddings(EmbeddingProvider):
    """SentenceTransformers-based embeddings."""
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize SentenceTransformer embeddings.
        
        Args:
            model_name: Name of the model to use (defaults to config)
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install it with: pip install sentence-transformers"
            )
        
        self.model_name = model_name or Config.EMBEDDING_MODEL_NAME
        self.model = SentenceTransformer(self.model_name)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for documents."""
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a query."""
        embedding = self.model.encode([text], show_progress_bar=False)
        return embedding[0].tolist()


def get_embedding_provider() -> EmbeddingProvider:
    """
    Get the embedding provider.
    
    Returns:
        EmbeddingProvider instance
    """
    return SentenceTransformerEmbeddings()
