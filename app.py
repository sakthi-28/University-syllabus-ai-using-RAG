"""Streamlit frontend for University Syllabus AI Assistant."""
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from pathlib import Path
import tempfile
import os
from typing import List

from src.config import Config
from src.document_processor import DocumentProcessor
from src.vector_store import SyllabusVectorStore
from src.rag_pipeline import RAGPipeline

# Page configuration
st.set_page_config(
    page_title="University Syllabus AI Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False


def initialize_components():
    """Initialize vector store and RAG pipeline."""
    try:
        if st.session_state.vector_store is None:
            with st.spinner("Initializing vector store..."):
                st.session_state.vector_store = SyllabusVectorStore()
                st.session_state.rag_pipeline = RAGPipeline(st.session_state.vector_store)
        return True
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        st.info("Please check your configuration in .env file")
        return False


def process_uploaded_pdfs(uploaded_files: List) -> bool:
    """
    Process uploaded PDF files and add them to the vector store.
    
    Args:
        uploaded_files: List of uploaded file objects
        
    Returns:
        True if processing was successful, False otherwise
    """
    if not uploaded_files:
        return False
    
    if not initialize_components():
        return False
    
    processor = DocumentProcessor()
    vector_store = st.session_state.vector_store
    
    success_count = 0
    skipped_count = 0
    
    with st.spinner("Processing PDFs..."):
        for uploaded_file in uploaded_files:
            try:
                # Check if file was already processed
                if uploaded_file.name in st.session_state.processed_files:
                    skipped_count += 1
                    continue
                
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = Path(tmp_file.name)
                
                try:
                    # Get document hash to check if already exists
                    doc_hash = processor.get_document_hash(tmp_path)
                    
                    if vector_store.document_exists(doc_hash):
                        st.info(f"📄 {uploaded_file.name} already exists in the database. Skipping...")
                        skipped_count += 1
                        st.session_state.processed_files.add(uploaded_file.name)
                        continue
                    
                    # Process PDF
                    documents = processor.process_pdf(tmp_path)
                    
                    if documents:
                        # Add to vector store
                        vector_store.add_documents(documents)
                        success_count += 1
                        st.session_state.processed_files.add(uploaded_file.name)
                        st.success(f"✅ Successfully processed: {uploaded_file.name} ({len(documents)} chunks)")
                    else:
                        st.warning(f"⚠️ No content extracted from: {uploaded_file.name}")
                
                finally:
                    # Clean up temporary file
                    if tmp_path.exists():
                        os.unlink(tmp_path)
            
            except Exception as e:
                st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
    
    if success_count > 0:
        st.session_state.documents_loaded = True
        st.balloons()
    
    return success_count > 0


def main():
    """Main application function."""
    # Title and description
    st.title("📚 University Syllabus AI Assistant")
    st.markdown("""
    Upload your syllabus PDF documents and ask questions about course content, 
    requirements, schedules, and more. The AI will answer based solely on your uploaded documents.
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Display current configuration
        st.subheader("Current Settings")
        st.text(f"Embeddings: sentence-transformers ({Config.EMBEDDING_MODEL_NAME})")
        st.text(f"LLM Provider: openrouter")
        st.text(f"LLM Model: {Config.LLM_MODEL}")
        
        # Collection info
        if st.session_state.vector_store:
            info = st.session_state.vector_store.get_collection_info()
            st.subheader("📊 Database Info")
            st.text(f"Documents: {info['document_count']}")
            st.text(f"Collection: {info['collection_name']}")
        
        st.divider()
        
        # Initialize button
        if st.button("🔄 Initialize System", use_container_width=True):
            try:
                st.session_state.vector_store = None
                st.session_state.rag_pipeline = None
                initialize_components()
                st.success("System initialized!")
                st.rerun()
            except Exception as e:
                st.error(f"Initialization error: {str(e)}")
        
        # Clear database button
        if st.button("🗑️ Clear Database", use_container_width=True):
            if st.session_state.vector_store:
                st.session_state.vector_store.delete_collection()
                st.session_state.processed_files.clear()
                st.session_state.documents_loaded = False
                st.success("Database cleared!")
                st.rerun()
    
    # Initialize components
    if not initialize_components():
        st.stop()
    
    # Main content area
    tab1, tab2 = st.tabs(["📤 Upload Documents", "❓ Ask Questions"])
    
    with tab1:
        st.header("Upload Syllabus PDFs")
        st.markdown("Upload one or more PDF files containing syllabus information.")
        
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="Select one or more PDF files to upload"
        )
        
        if uploaded_files:
            if st.button("📥 Process PDFs", type="primary", use_container_width=True):
                process_uploaded_pdfs(uploaded_files)
    
    with tab2:
        st.header("Ask Questions")
        
        # Check if documents are loaded
        if not st.session_state.documents_loaded:
            info = st.session_state.vector_store.get_collection_info()
            if info['document_count'] == 0:
                st.warning("⚠️ No documents loaded. Please upload PDF files in the 'Upload Documents' tab first.")
            else:
                st.session_state.documents_loaded = True
        
        if st.session_state.documents_loaded:
            # Question input
            question = st.text_area(
                "Enter your question:",
                height=100,
                placeholder="e.g., What are the course prerequisites? What is the grading policy? When are the exams scheduled?"
            )
            
            col1, col2 = st.columns([1, 5])
            with col1:
                ask_button = st.button("🔍 Ask", type="primary", use_container_width=True)
            
            if ask_button and question:
                with st.spinner("Searching documents and generating answer..."):
                    result = st.session_state.rag_pipeline.answer_question(question)
                    
                    # Display answer
                    st.subheader("💡 Answer")
                    st.markdown(result["answer"])
                    
                    # Display sources
                    if result.get("sources"):
                        st.divider()
                        st.subheader("📑 Source Documents")
                        
                        for idx, source in enumerate(result["sources"], 1):
                            with st.expander(f"Source {idx}: {source['filename']}"):
                                st.text(f"Chunk Index: {source['chunk_index']}")
                                st.text("Preview:")
                                st.text(source['preview'])
            
            elif ask_button:
                st.warning("Please enter a question.")
        else:
            st.info("👆 Upload documents in the 'Upload Documents' tab to start asking questions.")


if __name__ == "__main__":
    # Validate configuration on startup
    try:
        Config.validate()
    except ValueError as e:
        st.error(f"Configuration Error: {str(e)}")
        st.info("Please create a .env file with the required configuration. See .env.example for reference.")
        st.stop()
    
    main()
