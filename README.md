# Simple RAG System - Learning Project : Personal Doc Assistant

A minimal Retrieval Augmented Generation (RAG) system built with llama index to understand the core concepts of document loading, chunking, embedding, vector storage, and retrieval.

## 🎯 What I Learn

- Document loading and text extraction
- Text chunking strategies
- Embedding generation
- Vector database storage and retrieval
- Using retrieved context to generate answers

## 🛠️ Tech Stack

- **Python 3.8+**
- **OpenAI** - Embeddings and LLM (or use Anthropic)
- **ChromaDB** - Vector database
- **PyPDF** - PDF document loading
- **Sentence-Transformers** - Local embeddings (optional)

## 📦 Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install openai chromadb pypdf sentence-transformers anthropic python-dotenv

# Create .env file with your API keys
echo "OPENAI_API_KEY=your_key_here" > .env
echo "ANTHROPIC_API_KEY=your_key_here" >> .env
```

## 🚀 How It Works

### The RAG Pipeline

1. **Load Document** → Extract text from PDF
2. **Chunk Text** → Split into ~500 character chunks with overlap
3. **Generate Embeddings** → Convert chunks to numerical vectors
4. **Store in Vector DB** → Save embeddings with ChromaDB
5. **Query** → Embed user question
6. **Retrieve** → Find top 3 most similar chunks
7. **Generate Answer** → Send chunks + question to LLM

### Why Each Step Matters

**Chunking:** LLMs have context limits, smaller chunks enable precise retrieval

**Embeddings:** Semantic search instead of keyword matching

**Vector DB:** Fast similarity search across thousands of chunks

**Retrieval:** Only relevant context, not entire document

