# Quick Start Guide

## Step-by-Step Instructions

### Step 1: Install Dependencies

Open a terminal/command prompt in the project directory and run:

```bash
pip install -r requirements.txt
```

**Note**: If you encounter issues, try:
- `python -m pip install -r requirements.txt`
- Or use a virtual environment (recommended):
  ```bash
  python -m venv venv
  venv\Scripts\activate  # Windows
  # or: source venv/bin/activate  # macOS/Linux
  pip install -r requirements.txt
  ```

### Step 2: Create Configuration File

Copy the example environment file:

**Windows (PowerShell):**
```powershell
Copy-Item .env.example .env
```

**Windows (CMD):**
```cmd
copy .env.example .env
```

**macOS/Linux:**
```bash
cp .env.example .env
```

### Step 3: Configure Your Settings

Edit the `.env` file with a text editor and set your OpenRouter key:

```env
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
LLM_PROVIDER=openrouter
LLM_MODEL=z-ai/glm-4.5-air:free
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

### Step 4: Run the Application

```bash
streamlit run app.py
```

The application will:
1. Open in your default web browser automatically
2. Display the URL (usually `http://localhost:8501`)
3. Show the Streamlit interface

### Step 5: Use the Application

1. **Upload PDFs**: 
   - Click on the "Upload Documents" tab
   - Select one or more PDF syllabus files
   - Click "Process PDFs"
   - Wait for processing to complete (you'll see success messages)

2. **Ask Questions**:
   - Click on the "Ask Questions" tab
   - Type your question (e.g., "What are the course prerequisites?")
   - Click "Ask"
   - View the answer and source documents

## Troubleshooting

### "Module not found" errors
- Make sure you've installed all dependencies: `pip install -r requirements.txt`
- Check that you're using the correct Python environment

### "OPENROUTER_API_KEY is required" error
- Make sure you've created a `.env` file
- Add your OpenRouter API key to the `.env` file
- Restart the Streamlit app

### PDF upload fails
- Ensure PDFs are not password-protected
- Check that PDFs contain text (not just images/scans)
- Try with a different PDF file

### Port already in use
- Streamlit uses port 8501 by default
- If it's busy, Streamlit will automatically use the next available port
- Or stop other Streamlit apps running

## What to Expect

- **First run**: The SentenceTransformers model will download (~80MB) - this happens automatically
- **PDF processing**: May take 10-30 seconds per PDF depending on size
- **Question answering**: Usually takes 2-5 seconds per question

## Next Steps

- Upload multiple syllabus PDFs
- Try different types of questions
- Check the source documents to see where answers came from
- Documents persist between sessions - no need to re-upload!
