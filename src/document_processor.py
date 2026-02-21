"""PDF document processing and text chunking."""

import hashlib
from pathlib import Path
from typing import List, Dict, Any
import pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from .config import Config


class DocumentProcessor:
    """Handles PDF extraction and text chunking."""
    
    def __init__(self):
        """Initialize the document processor with chunking configuration."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text content
            
        Raises:
            ValueError: If PDF is empty or cannot be read
        """
        try:
            text_content = []
            with open(pdf_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                
                if len(pdf_reader.pages) == 0:
                    raise ValueError(f"PDF file {pdf_path.name} is empty")
                
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text.strip():
                        text_content.append(text)
                
                if not text_content:
                    raise ValueError(f"No text content found in PDF {pdf_path.name}")
                
                return "\n\n".join(text_content)
        
        except Exception as e:
            raise ValueError(f"Error extracting text from PDF {pdf_path.name}: {str(e)}")
    
    def process_pdf(self, pdf_path: Path) -> List[Document]:
        """
        Process a PDF file and return chunked documents with metadata.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of Document objects with text chunks and metadata
        """
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)
        
        # Generate document hash for tracking
        doc_hash = self._generate_hash(text)
        
        # Split into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Create Document objects with metadata
        documents = []
        for idx, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": pdf_path.name,
                    "file_path": str(pdf_path),
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                    "document_hash": doc_hash
                }
            )
            documents.append(doc)
        
        return documents
    
    def process_multiple_pdfs(self, pdf_paths: List[Path]) -> List[Document]:
        """
        Process multiple PDF files.
        
        Args:
            pdf_paths: List of paths to PDF files
            
        Returns:
            List of all Document objects from all PDFs
        """
        all_documents = []
        for pdf_path in pdf_paths:
            try:
                documents = self.process_pdf(pdf_path)
                all_documents.extend(documents)
            except Exception as e:
                print(f"Warning: Failed to process {pdf_path.name}: {str(e)}")
                continue
        
        return all_documents
    
    @staticmethod
    def _generate_hash(text: str) -> str:
        """Generate SHA256 hash for document content."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    @staticmethod
    def get_document_hash(pdf_path: Path) -> str:
        """
        Get hash of a PDF file's content.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            SHA256 hash string
        """
        text = DocumentProcessor().extract_text_from_pdf(pdf_path)
        return DocumentProcessor._generate_hash(text)
