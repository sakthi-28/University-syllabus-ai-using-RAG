"""ChromaDB vector store management."""

from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from .config import Config
from .embeddings import get_embedding_provider


class SyllabusVectorStore:
    """Manages ChromaDB vector store for syllabus documents."""
    
    def __init__(self, collection_name: Optional[str] = None):
        """
        Initialize the vector store.
        
        Args:
            collection_name: Name of the ChromaDB collection (defaults to config)
        """
        self.collection_name = collection_name or Config.COLLECTION_NAME
        self.db_path = Config.get_chroma_db_path()
        
        # Initialize embedding provider (local SentenceTransformers)
        self.embedding_provider = get_embedding_provider()
        self.embeddings = self._create_langchain_embeddings()

        # Import Chroma lazily so we can show a helpful error message
        try:
            import chromadb  # type: ignore
            from chromadb.config import Settings  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Failed to import 'chromadb'. This is commonly caused by using Python 3.14+, "
                "which is currently incompatible with ChromaDB/Pydantic v1 compatibility.\n\n"
                "Fix: create a new virtual environment with Python 3.11–3.13 and reinstall requirements:\n"
                "  python3.12 -m venv venv\n"
                "  venv\\Scripts\\activate\n"
                "  pip install -r requirements.txt\n\n"
                "Original error:\n"
                f"{type(e).__name__}: {e}"
            ) from e

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False),
        )
        
        # Initialize or get collection
        self.vector_store = self._initialize_vector_store()
    
    def _create_langchain_embeddings(self):
        """Create LangChain-compatible embeddings wrapper."""
        return HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL_NAME)
    
    def _initialize_vector_store(self) -> Chroma:
        """Initialize or load the ChromaDB vector store."""
        try:
            # Try to load existing collection
            vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(self.db_path),
                client=self.client
            )
            return vector_store
        except Exception:
            # Create new collection if it doesn't exist
            vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(self.db_path),
                client=self.client
            )
            return vector_store
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of Document objects to add
            
        Returns:
            List of document IDs
        """
        if not documents:
            return []
        
        # Add documents to vector store
        ids = self.vector_store.add_documents(documents)
        return ids
    
    def document_exists(self, document_hash: str) -> bool:
        """
        Check if a document with the given hash already exists.
        
        Args:
            document_hash: SHA256 hash of the document
            
        Returns:
            True if document exists, False otherwise
        """
        try:
            # Use ChromaDB collection directly to query by metadata
            collection = self.client.get_collection(self.collection_name)
            results = collection.get(
                where={"document_hash": document_hash},
                limit=1
            )
            return len(results.get("ids", [])) > 0
        except Exception:
            return False
    
    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None
    ) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            query: Search query text
            k: Number of results to return (defaults to config)
            
        Returns:
            List of similar Document objects
        """
        k = k or Config.TOP_K_RESULTS
        return self.vector_store.similarity_search_with_score(query, k=k)
    
    def similarity_search_with_score(
        self,
        query: str,
        k: Optional[int] = None
    ) -> List[tuple]:
        """
        Search for similar documents with similarity scores.
        
        Args:
            query: Search query text
            k: Number of results to return (defaults to config)
            
        Returns:
            List of tuples (Document, score)
        """
        k = k or Config.TOP_K_RESULTS
        return self.vector_store.similarity_search_with_score(query, k=k * 2)
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            collection = self.client.get_collection(self.collection_name)
            count = collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "db_path": str(self.db_path)
            }
        except Exception:
            return {
                "collection_name": self.collection_name,
                "document_count": 0,
                "db_path": str(self.db_path)
            }
    
    def delete_collection(self) -> None:
        """Delete the entire collection."""
        try:
            self.client.delete_collection(self.collection_name)
            # Reinitialize vector store
            self.vector_store = self._initialize_vector_store()
        except Exception as e:
            print(f"Error deleting collection: {str(e)}")
