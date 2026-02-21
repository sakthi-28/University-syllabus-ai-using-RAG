"""RAG pipeline with LangChain for question answering."""

from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .config import Config
from .vector_store import SyllabusVectorStore


class RAGPipeline:
    """RAG pipeline for question answering with anti-hallucination measures."""
    
    PROMPT_TEMPLATE = """You are an expert academic assistant specializing in university syllabi.

Use ONLY the following context from uploaded syllabus documents to answer the question.

If the answer cannot be found in the provided context, say:
"I don't have enough information in the uploaded syllabus documents to answer this question."

Context from syllabus documents:
{context}

Question:
{question}

Instructions:
1. Answer ONLY using information from the provided context.
2. Do NOT use external knowledge.
3. Provide a well-structured and detailed explanation.
4. Use multiple paragraphs when necessary.
5. Clearly explain requirements, schedules, grading policies, or course content when mentioned.
6. Reference the source filename when possible.
7. Do not summarize too briefly.

Answer:
"""
    
    def __init__(self, vector_store: SyllabusVectorStore):
        """
        Initialize the RAG pipeline.
        
        Args:
            vector_store: Initialized SyllabusVectorStore instance
        """
        self.vector_store = vector_store
        self.llm = self._initialize_llm()
        self.retriever = self.vector_store.vector_store.as_retriever(
    search_type="mmr",  # 🔥 important
    search_kwargs={
        "k": Config.TOP_K_RESULTS,
        "fetch_k": Config.TOP_K_RESULTS * 2,
        "lambda_mult": 0.5
    }
)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.PROMPT_TEMPLATE),
            ]
        )
    
    def _initialize_llm(self):
        """Initialize the LLM based on configuration."""
        if Config.LLM_PROVIDER != "openrouter":
            raise ValueError(
                f"Unsupported LLM provider: {Config.LLM_PROVIDER}. Supported provider: 'openrouter'"
            )

        if not Config.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY must be set when using OpenRouter LLM")

        return ChatOpenAI(
            model=Config.LLM_MODEL,
            temperature=0.1,
            openai_api_key=Config.OPENROUTER_API_KEY,
            base_url=Config.OPENROUTER_BASE_URL,
            default_headers={
                "HTTP-Referer": "https://github.com/syllabus-ai",
                "X-Title": "Syllabus AI Assistant",
            },
        )
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Answer a question using RAG.
        
        Args:
            question: User's question
            
        Returns:
            Dictionary with answer and source documents
        """
        if not question.strip():
            return {
                "answer": "Please provide a valid question.",
                "sources": []
            }
        
        try:
            source_documents: List[Document] = self.retriever.invoke(question)

            if not source_documents:
                return {
                    "answer": "I don't have enough information in the uploaded syllabus documents to answer this question.",
                    "sources": [],
                    "source_documents": [],
                }

            context = self._format_context(source_documents)
            messages = self.prompt.format_messages(context=context, question=question)
            response = self.llm.invoke(messages)
            answer = getattr(response, "content", str(response))

            sources = self._extract_sources(source_documents)
            
            return {
                "answer": answer,
                "sources": sources,
                "source_documents": source_documents
            }
        
        except Exception as e:
            return {
                "answer": f"Error processing question: {str(e)}",
                "sources": []
            }
    
    def _extract_sources(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Extract source information from documents.
        
        Args:
            documents: List of source Document objects
            
        Returns:
            List of dictionaries with source information
        """
        sources = []
        seen_sources = set()
        
        for doc in documents:
            metadata = doc.metadata
            source_name = metadata.get("source", "Unknown")
            
            # Avoid duplicates
            if source_name not in seen_sources:
                seen_sources.add(source_name)
                sources.append({
                    "filename": source_name,
                    "chunk_index": metadata.get("chunk_index", "N/A"),
                    "preview": doc.page_content[:1000] + "..." if len(doc.page_content) > 200 else doc.page_content
                })
        
        return sources

    @staticmethod
    def _format_context(documents: List[Document]) -> str:
        """Format retrieved documents into a single context string."""
        parts: List[str] = []
        for doc in documents:
            src = doc.metadata.get("source", "Unknown")
            chunk_index = doc.metadata.get("chunk_index", "N/A")
            parts.append(f"[{src} | chunk {chunk_index}]\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)
    
    def get_relevant_context(self, question: str, k: Optional[int] = None) -> List[Document]:
        """
        Get relevant context documents for a question without generating an answer.
        
        Args:
            question: User's question
            k: Number of documents to retrieve
            
        Returns:
            List of relevant Document objects
        """
        return self.vector_store.similarity_search(question, k=k)
