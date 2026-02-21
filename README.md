# University Syllabus AI Assistant
## 👨‍💻 Author

**Sakthivel V**

Cloud & AI Enthusiast  
Built this University Syllabus AI Assistant as a hands-on RAG project using LangChain, ChromaDB, and Streamlit.

- GitHub: https://github.com/sakthi-28
- Email: sakthivelvenkat2619@gmail.com

A Retrieval-Augmented Generation (RAG) system for querying university syllabus documents. Upload PDF syllabi and ask questions about course content, requirements, schedules, and policies. The system uses LangChain, ChromaDB, and Streamlit to provide accurate, context-aware answers based solely on your uploaded documents.

## Features

- 📄 **PDF Upload**: Upload multiple syllabus PDF documents
- 🔍 **Semantic Search**: Find relevant information using vector embeddings
- 💬 **Q&A Interface**: Ask questions and get answers based on uploaded content
- 🚫 **Anti-Hallucination**: Answers are strictly grounded in uploaded documents
- 💾 **Persistent Storage**: Documents are stored persistently - no need to re-upload
- 📑 **Source Citations**: See which documents and sections were used for answers
- ⚙️ **Configurable**: Support for different embedding models and LLM providers

## Architecture

- **Frontend**: Streamlit web application
- **Backend**: LangChain RAG pipeline
- **Vector Store**: ChromaDB for persistent document embeddings
- **Embeddings**: Local SentenceTransformers (fast, no API key)
- **LLM**: OpenRouter (e.g. `z-ai/glm-4.5-air:free`)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or download this repository**

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   - Copy `.env.example` to `.env`
   - Edit `.env` and set your OpenRouter configuration:
     ```env
     EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
     LLM_PROVIDER=openrouter
     LLM_MODEL=z-ai/glm-4.5-air:free
     OPENROUTER_API_KEY=your_openrouter_api_key_here
     ```

## Usage

1. **Start the Streamlit application**:
   ```bash
   streamlit run app.py
   ```

2. **Upload PDF files**:
   - Navigate to the "Upload Documents" tab
   - Select one or more PDF files
   - Click "Process PDFs"
   - Wait for processing to complete

3. **Ask questions**:
   - Go to the "Ask Questions" tab
   - Enter your question in the text area
   - Click "Ask" to get an answer
   - View source documents used for the answer

## Configuration Options

### Embedding Models

- **SentenceTransformers** (free, local):
  - No API key required
  - Runs locally on your machine
  - Model: `all-MiniLM-L6-v2` (default) or any HuggingFace model

### LLM Providers

- **OpenRouter**:
  - Model: `z-ai/glm-4.5-air:free` (default)
  - Requires `OPENROUTER_API_KEY`
  - You can switch models by changing `LLM_MODEL`

### Chunking Configuration

- `CHUNK_SIZE`: Size of text chunks (default: 1000 characters)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200 characters)

### RAG Configuration

- `TOP_K_RESULTS`: Number of document chunks to retrieve (default: 4)

## Project Structure

```
syllabus-ai/
├── app.py                      # Streamlit frontend
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── document_processor.py  # PDF extraction and chunking
│   ├── embeddings.py          # Embedding generation
│   ├── vector_store.py        # ChromaDB management
│   └── rag_pipeline.py        # RAG chain implementation
├── data/                      # Uploaded PDFs (optional)
├── chroma_db/                 # ChromaDB persistent storage
├── requirements.txt           # Python dependencies
├── .env.example              # Environment variables template
├── .env                      # Your configuration (create from .env.example)
├── .gitignore
└── README.md
```

## How It Works

1. **Document Processing**:
   - PDFs are uploaded and text is extracted
   - Text is split into chunks with metadata
   - Each chunk is assigned a unique hash

2. **Embedding Generation**:
   - Chunks are converted to vector embeddings
   - Embeddings are stored in ChromaDB with metadata

3. **Query Processing**:
   - User question is converted to an embedding
   - Similar chunks are retrieved from ChromaDB
   - Retrieved context is passed to LLM with strict instructions

4. **Answer Generation**:
   - LLM generates answer using only retrieved context
   - Source documents are cited
   - Answer is displayed with source references

## Anti-Hallucination Measures

- Explicit prompt instructions to use only provided context
- Fallback response when no relevant context is found
- Source citation requirement
- Low temperature setting for deterministic answers

## Troubleshooting

### "OPENAI_API_KEY is required" error
- Make sure you've created a `.env` file
- Add your OpenAI API key to `.env`
- Or switch to `EMBEDDING_MODEL=sentence_transformers` for local embeddings

### PDF processing errors
- Ensure PDFs are not password-protected
- Check that PDFs contain extractable text (not just images)
- Verify PDF files are not corrupted

### ChromaDB errors
- Ensure write permissions in the project directory
- Check that `chroma_db/` directory exists or can be created

### Import errors
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Verify your Python version is 3.8+

## License

This project is open source and available for educational purposes.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Acknowledgments

- Built with [LangChain](https://www.langchain.com/)
- Vector storage powered by [ChromaDB](https://www.trychroma.com/)
- UI built with [Streamlit](https://streamlit.io/)
