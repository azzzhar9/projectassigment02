# RAG FAQ Support Chatbot

A Retrieval-Augmented Generation (RAG) based FAQ support system that answers user questions by retrieving relevant information from company documentation. The system uses intelligent text chunking, vector embeddings, and semantic search to provide accurate, context-aware answers.

## Overview

This project implements a complete RAG pipeline for FAQ support:

1. **Data Pipeline**: Parses plain text documents, chunks them intelligently (≥20 chunks), generates embeddings using sentence-transformers (local, no API key required), and stores them in a FAISS vector index
2. **Query Pipeline**: Accepts user questions, performs vector search (k-NN), retrieves relevant chunks, and generates answers using OpenRouter LLM
3. **Evaluator Agent**: (Bonus) Scores answer quality (0-10) based on chunk relevance, answer accuracy, and completeness

## Project Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PROJECT WORKFLOW                                    │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                    PHASE 1: SETUP & INDEXING                                  │
└─────────────────────────────────────────────────────────────────────────────┘

    Step 1: Environment Setup
    ┌─────────────────────┐
    │  Install Dependencies│  pip install -r requirements.txt
    │  (requirements.txt) │  - sentence-transformers
    └──────────┬──────────┘  - faiss-cpu
               │              - python-dotenv
               ▼              - requests
    ┌─────────────────────┐
    │  Configure .env      │  OPENROUTER_API_KEY=...
    │  (API Keys)          │  EMBEDDING_MODEL=all-MiniLM-L6-v2
    └──────────┬──────────┘  LLM_MODEL=openai/gpt-3.5-turbo
               │
               ▼
    Step 2: Document Preparation
    ┌─────────────────────┐
    │  FAQ Document        │  data/faq_document.txt
    │  (Plain Text)        │  ≥1000 words
    │  ≥1000 words        │  Policies, Features, Procedures
    └──────────┬──────────┘
               │
               ▼
    Step 3: Build Index (python src/build_index.py)
    ┌─────────────────────────────────────────────────────────┐
    │                                                          │
    │  ┌──────────────────┐                                   │
    │  │ Load Document     │  load_document()                  │
    │  │ (utils.py)        │  → Read faq_document.txt         │
    │  └────────┬─────────┘                                   │
    │           │                                              │
    │           ▼                                              │
    │  ┌──────────────────┐                                   │
    │  │ Chunk Text        │  chunk_text_sentences()          │
    │  │ (utils.py)        │  → Sentence-based chunking        │
    │  │                  │  → Overlap: 100 chars              │
    │  │                  │  → Size: 50-500 chars              │
    │  │                  │  → Result: ≥20 chunks              │
    │  └────────┬─────────┘                                   │
    │           │                                              │
    │           ▼                                              │
    │  ┌──────────────────┐                                   │
    │  │ Generate          │  SentenceTransformer()            │
    │  │ Embeddings        │  → Model: all-MiniLM-L6-v2        │
    │  │ (LOCAL)           │  → Dimension: 384                  │
    │  │ (build_index.py)  │  → No API key needed               │
    │  └────────┬─────────┘                                   │
    │           │                                              │
    │           ▼                                              │
    │  ┌──────────────────┐                                   │
    │  │ Create FAISS      │  faiss.IndexFlatL2()              │
    │  │ Index              │  → L2 normalization              │
    │  │                    │  → Cosine similarity              │
    │  └────────┬─────────┘                                   │
    │           │                                              │
    │           ▼                                              │
    │  ┌──────────────────┐                                   │
    │  │ Save to Disk      │  data/faiss_index.bin             │
    │  │                   │  data/chunks_metadata.json        │
    │  │                   │  data/embeddings.pkl              │
    │  └──────────────────┘                                   │
    │                                                          │
    └──────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                    PHASE 2: QUERY PROCESSING                                  │
└─────────────────────────────────────────────────────────────────────────────┘

    Step 4: User Query (python src/query.py "question")
    ┌─────────────────────┐
    │  User Question       │  "How many days of annual leave?"
    │  (Input)             │
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────────────────────────────────────────────────┐
    │                                                          │
    │  ┌──────────────────┐                                   │
    │  │ Load Index       │  load_index()                     │
    │  │ (query.py)        │  → Read faiss_index.bin          │
    │  │                  │  → Read chunks_metadata.json       │
    │  └────────┬─────────┘                                   │
    │           │                                              │
    │           ▼                                              │
    │  ┌──────────────────┐                                   │
    │  │ Embed Question    │  SentenceTransformer()            │
    │  │ (LOCAL)           │  → Same model as indexing         │
    │  │                   │  → No API key needed              │
    │  └────────┬─────────┘                                   │
    │           │                                              │
    │           ▼                                              │
    │  ┌──────────────────┐                                   │
    │  │ Vector Search     │  index.search()                   │
    │  │ (k-NN)            │  → k=5 top chunks                 │
    │  │                   │  → Cosine similarity              │
    │  │                   │  → Return: chunks + scores        │
    │  └────────┬─────────┘                                   │
    │           │                                              │
    │           ▼                                              │
    │  ┌──────────────────┐                                   │
    │  │ Generate Answer   │  OpenRouter API                  │
    │  │ (LLM)             │  → Model: gpt-3.5-turbo           │
    │  │                   │  → Input: question + chunks       │
    │  │                   │  → Output: generated answer       │
    │  └────────┬─────────┘                                   │
    │           │                                              │
    │           ▼                                              │
    │  ┌──────────────────┐                                   │
    │  │ Format Output    │  format_output()                   │
    │  │ (JSON)            │  → user_question                   │
    │  │                   │  → system_answer                  │
    │  │                   │  → chunks_related                  │
    │  └────────┬─────────┘                                   │
    │           │                                              │
    │           ▼                                              │
    │  ┌──────────────────┐                                   │
    │  │ Optional:         │  evaluate_answer()                │
    │  │ Evaluator         │  → Score: 0-10                   │
    │  │ (evaluator.py)     │  → Relevance + Accuracy +         │
    │  │                   │    Completeness                    │
    │  └────────┬─────────┘                                   │
    │           │                                              │
    │           ▼                                              │
    │  ┌──────────────────┐                                   │
    │  │ Return JSON       │  {                                │
    │  │                   │    "user_question": "...",        │
    │  │                   │    "system_answer": "...",         │
    │  │                   │    "chunks_related": [...],        │
    │  │                   │    "evaluation": {...}            │
    │  │                   │  }                                 │
    │  └──────────────────┘                                   │
    │                                                          │
    └──────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          KEY DATA FLOWS                                       │
└─────────────────────────────────────────────────────────────────────────────┘

    Indexing Flow:
    Document → Chunks → Embeddings → FAISS Index → Storage
    
    Query Flow:
    Question → Embedding → Vector Search → Chunks → LLM → Answer → JSON
    
    Privacy Flow:
    🔒 Local: Document, Chunks, Embeddings (never leave machine)
    🌐 API: Only retrieved chunks sent to LLM (minimal exposure)

┌─────────────────────────────────────────────────────────────────────────────┐
│                          FILE INTERACTIONS                                    │
└─────────────────────────────────────────────────────────────────────────────┘

    build_index.py:
    Reads:  data/faq_document.txt
    Writes: data/faiss_index.bin
            data/chunks_metadata.json
            data/embeddings.pkl
    
    query.py:
    Reads:  data/faiss_index.bin
            data/chunks_metadata.json
    Uses:   .env (OPENROUTER_API_KEY)
    Outputs: JSON to stdout or file
    
    evaluator.py:
    Uses:   query.py output
            .env (OPENROUTER_API_KEY)
    Outputs: Evaluation scores (0-10)
```

## Documentation

- **README.md**: This file - setup, usage, and overview
- **HOW_IT_WORKS.md**: Detailed file-by-file explanation of how the system works
- **PROJECT_ANALYSIS.md**: Project strengths, improvements, and analysis

## Project Structure

```
faq-rag-chatbot/
├── .env.example                  # Environment variables template
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── data/
│   ├── faq_document.txt          # Plain text FAQ document (≥1000 words)
│   ├── faiss_index.bin           # FAISS vector index (generated)
│   ├── chunks_metadata.json       # Chunk metadata (generated)
│   └── embeddings.pkl          # Embeddings cache (generated)
├── src/
│   ├── build_index.py            # Data pipeline: chunking + embedding generation
│   ├── query.py                  # Query pipeline: vector search + LLM generation
│   ├── evaluator.py              # Evaluator agent for answer quality scoring
│   └── utils.py                  # Shared utility functions
├── outputs/
│   └── sample_queries.json       # Sample query-response pairs
└── tests/
    └── test_core.py              # Unit tests for core functionality
```

## Quick Start Guide - Testing the Project

This guide will walk you through setting up and testing the RAG FAQ Chatbot from scratch.

### Step-by-Step Testing Instructions

#### Step 1: Verify Prerequisites

**Check Python version:**
```bash
python --version
# Should be Python 3.8 or higher
```

**Check if you have pip:**
```bash
pip --version
```

#### Step 2: Clone/Navigate to Project

If you have the project in a repository:
```bash
git clone <repository-url>
cd faq-rag-chatbot
```

Or if you already have the project folder:
```bash
cd ProjectAssignment  # or your project directory
```

#### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Expected output:**
- This will install: sentence-transformers, faiss-cpu, numpy, python-dotenv, requests, torch
- First-time installation may take 2-5 minutes
- The embedding model (~80MB) will be downloaded automatically on first use

**Verify installation:**
```bash
python -c "import sentence_transformers, faiss, numpy; print('All dependencies installed!')"
```

#### Step 4: Set Up API Key

**Option A: Create .env file (Recommended)**

1. Copy the example file:
   ```bash
   # On Windows PowerShell:
   Copy-Item .env.example .env
   
   # On Linux/Mac:
   cp .env.example .env
   ```

2. Edit `.env` file and add your OpenRouter API key:
   ```
   OPENROUTER_API_KEY=sk-or-v1-your-actual-api-key-here
   EMBEDDING_MODEL=all-MiniLM-L6-v2
   LLM_MODEL=openai/gpt-3.5-turbo
   ```

**Option B: Set environment variable (Temporary)**
```bash
# Windows PowerShell:
$env:OPENROUTER_API_KEY="sk-or-v1-your-actual-api-key-here"

# Linux/Mac:
export OPENROUTER_API_KEY="sk-or-v1-your-actual-api-key-here"
```

**Get OpenRouter API Key:**
1. Visit: https://openrouter.ai/
2. Sign up or log in
3. Go to Keys section
4. Create a new API key
5. Copy the key (starts with `sk-or-v1-`)

#### Step 5: Verify FAQ Document Exists

```bash
# Check if document exists
ls data/faq_document.txt  # Linux/Mac
dir data\faq_document.txt  # Windows

# Verify it has content (should show file size)
```

The document should be ≥1000 words. If missing, the system will show an error.

#### Step 6: Build the Index (First Time Setup)

```bash
python src/build_index.py
```

**What happens:**
1. Downloads embedding model (first time only, ~80MB)
2. Loads FAQ document
3. Chunks the document (creates ≥20 chunks)
4. Generates embeddings for all chunks
5. Creates FAISS index
6. Saves index files to `data/` directory

**Expected output:**
```
Loading embedding model: all-MiniLM-L6-v2...
Model loaded. Embedding dimension: 384

Loading document from data/faq_document.txt...
Document loaded: 13407 characters

Chunking document...
Created 39 chunks

Generating embeddings...
Batches: 100%|████████████| 2/2 [00:00<00:00, 4.42it/s]
Generated 39 embeddings

Creating FAISS index...

Saving index...
Index saved to data/
Total chunks: 39
Index dimension: 384

Index building complete!
```

**Verify index files created:**
```bash
# Check generated files
ls data/  # Linux/Mac
dir data\  # Windows

# Should see:
# - faiss_index.bin
# - chunks_metadata.json
# - embeddings.pkl
```

#### Step 7: Test Basic Query

```bash
python src/query.py "How many days of annual leave do I get per year?"
```

**Expected output:**
```json
{
  "user_question": "How many days of annual leave do I get per year?",
  "system_answer": "Full-time employees accrue annual leave at a rate of 2.5 days per month...",
  "chunks_related": [
    {
      "chunk_id": 0,
      "text": "Employees are entitled to various types of leave...",
      "similarity_score": 0.95
    },
    ...
  ]
}
```

**If you see an error:**
- "OPENROUTER_API_KEY not found" → Check Step 4, ensure .env file exists
- "Index files not found" → Run Step 6 again (build_index.py)
- "Module not found" → Run Step 3 again (pip install)

#### Step 8: Test Query with Evaluation

```bash
python src/query.py "What is the leave policy?" --evaluate
```

**Expected output:** Same as above, plus:
```json
{
  ...
  "evaluation": {
    "score": 8,
    "reason": "Highly relevant chunks...",
    "chunk_relevance_score": 3,
    "answer_accuracy_score": 3,
    "completeness_score": 2
  }
}
```

#### Step 9: Run Unit Tests

```bash
# Using pytest (recommended)
pytest tests/test_core.py -v

# Or using unittest
python -m unittest tests.test_core -v
```

**Expected output:**
```
============================= test session starts =============================
tests/test_core.py::TestChunking::test_chunk_minimum_requirement PASSED
tests/test_core.py::TestChunking::test_chunk_text_sentences PASSED
...
============================= 10 passed in 7.17s ==============================
```

#### Step 10: Test Additional Queries

Try these sample questions:

```bash
# Password reset
python src/query.py "How do I reset my password?"

# Remote work
python src/query.py "What are the remote work requirements?" --evaluate

# Benefits
python src/query.py "What is the 401(k) matching policy?"

# Save output to file
python src/query.py "How do I request time off?" --output outputs/my_test.json
```

### Troubleshooting Common Issues

**Issue: "ModuleNotFoundError: No module named 'faiss'"**
```bash
# Solution: Reinstall dependencies
pip install -r requirements.txt
```

**Issue: "OPENROUTER_API_KEY not found"**
```bash
# Solution: Check .env file exists and has correct key
# Windows PowerShell:
Get-Content .env

# Linux/Mac:
cat .env
```

**Issue: "Index files not found"**
```bash
# Solution: Rebuild the index
python src/build_index.py
```

**Issue: "Model download fails"**
```bash
# Solution: Check internet connection
# Model downloads from HuggingFace on first use
# If blocked, you may need to configure proxy
```

**Issue: "Slow embedding generation"**
```bash
# This is normal on first run (model download)
# Subsequent runs will be faster (model cached)
```

### Testing Checklist

- [ ] Python 3.8+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] OpenRouter API key configured in `.env`
- [ ] FAQ document exists (`data/faq_document.txt`)
- [ ] Index built successfully (`python src/build_index.py`)
- [ ] Basic query works (`python src/query.py "question"`)
- [ ] Query with evaluation works (`--evaluate` flag)
- [ ] Unit tests pass (`pytest tests/test_core.py`)
- [ ] Output files can be saved (`--output` flag)

### Next Steps After Testing

Once testing is complete, you can:
1. **Modify the FAQ document**: Edit `data/faq_document.txt` and rebuild index
2. **Adjust chunking parameters**: Modify `CHUNK_OVERLAP`, `MIN_CHUNK_SIZE` in `src/utils.py`
3. **Change embedding model**: Update `EMBEDDING_MODEL` in `.env`
4. **Integrate into application**: Use `query()` function from `src/query.py` in your code

## Setup

### Prerequisites

- Python 3.8 or higher
- OpenRouter API key (for LLM generation only - embeddings are generated locally)

### Installation

1. **Clone the repository** (or navigate to the project directory)

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   
   Note: The first time you run the code, sentence-transformers will download the embedding model (all-MiniLM-L6-v2) which is approximately 80MB. This is a one-time download.

3. **Set up environment variables**:
   
   Create a `.env` file in the project root (copy from `.env.example`):
   ```bash
   OPENROUTER_API_KEY=your-openrouter-api-key-here
   EMBEDDING_MODEL=all-MiniLM-L6-v2
   LLM_MODEL=openai/gpt-3.5-turbo
   ```
   
   Or export them in your shell:
   ```bash
   export OPENROUTER_API_KEY=your-openrouter-api-key-here
   export EMBEDDING_MODEL=all-MiniLM-L6-v2
   export LLM_MODEL=openai/gpt-3.5-turbo
   ```

4. **Build the search index**:
   ```bash
   python src/build_index.py
   ```
   
   This will:
   - Load `data/faq_document.txt`
   - Chunk the document (minimum 20 chunks)
   - Generate embeddings using sentence-transformers (local, no API key needed)
   - Create and save a FAISS index to `data/faiss_index.bin`
   - Save chunk metadata to `data/chunks_metadata.json`

## Usage

### Query the FAQ System

**Basic query** (returns JSON with answer):
```bash
python src/query.py "How many days of annual leave do I get per year?"
```

**Query with evaluation** (includes quality score):
```bash
python src/query.py "What are the remote work requirements?" --evaluate
```

**Save output to file**:
```bash
python src/query.py "How do I reset my password?" --output outputs/my_query.json
```

### Example Output

```json
{
  "user_question": "How many days of annual leave do I get per year?",
  "system_answer": "Full-time employees accrue annual leave at a rate of 2.5 days per month, which amounts to 30 days per year. However, there is a maximum accumulation limit of 20 days...",
  "chunks_related": [
    {
      "chunk_id": 0,
      "text": "Employees are entitled to various types of leave...",
      "similarity_score": 0.95
    },
    {
      "chunk_id": 1,
      "text": "Managers have 24 hours to approve or reject leave requests...",
      "similarity_score": 0.78
    }
  ],
  "evaluation": {
    "score": 9,
    "reason": "Highly relevant chunks directly address the question. Answer is accurate and complete.",
    "chunk_relevance_score": 4,
    "answer_accuracy_score": 4,
    "completeness_score": 1
  }
}
```

### Using as a Python Module

```python
from src.query import query

# Basic query
result = query("How do I reset my password?")
print(result['system_answer'])

# Query with evaluation
result = query("What is the 401(k) matching policy?", include_evaluation=True)
print(f"Answer: {result['system_answer']}")
print(f"Quality Score: {result['evaluation']['score']}/10")
```

### Running the Full Demo

To run a complete end-to-end demonstration of the system:

1. **Build the index** (if not already built):
   ```bash
   python src/build_index.py
   ```

2. **Run a query with evaluation**:
   ```bash
   python src/query.py "What is the leave policy?" --evaluate
   ```

This will:
- Load the FAQ document and create embeddings (local processing)
- Perform vector search to find relevant chunks
- Generate an answer using the LLM
- Evaluate the answer quality and provide a score

## Privacy & Data Processing

**Important Privacy Compliance Notes:**

- **Embeddings are generated locally**: All embedding generation happens on your machine using sentence-transformers. Your document content never leaves your local environment during indexing. This ensures complete privacy for sensitive HR documentation.

- **LLM is only used at query time**: The OpenRouter API is only called when you ask a question, and only the retrieved chunks (not the entire document) are sent to generate the answer. This minimizes data exposure and API costs.

- **No data storage on external servers**: The FAISS index and embeddings are stored locally in the `data/` directory. No document content is stored on external servers.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         RAG FAQ Chatbot                          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    INDEXING PIPELINE (One-time)                  │
└─────────────────────────────────────────────────────────────────┘

    ┌──────────────┐
    │ FAQ Document │
    │  (Plain Text)│
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │   Chunking   │  ← Sentence-based with overlap
    │  (utils.py)  │     (≥20 chunks)
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │  Embeddings  │  ← Sentence-Transformers (LOCAL)
    │  Generation  │     all-MiniLM-L6-v2
    │(build_index) │     No API key needed
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │ FAISS Index  │  ← Vector storage
    │  + Metadata  │     (Local files)
    └──────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    QUERY PIPELINE (On-demand)                   │
└─────────────────────────────────────────────────────────────────┘

    ┌──────────────┐
    │ User Question│
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │  Question    │  ← Sentence-Transformers (LOCAL)
    │  Embedding   │     Same model as indexing
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │ Vector Search│  ← FAISS k-NN
    │  (k=5 chunks)│     Cosine similarity
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │   Retrieved  │
    │    Chunks    │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │  LLM (API)   │  ← OpenRouter
    │  Generation  │     openai/gpt-3.5-turbo
    │  (query.py)  │     Only chunks sent, not full doc
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │   Answer +   │
    │   Chunks     │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │  Evaluator   │  ← Optional (evaluator.py)
    │   (Bonus)    │     Scores 0-10
    └──────────────┘
           │
           ▼
    ┌──────────────┐
    │  JSON Output │
    │  (Structured)│
    └──────────────┘

Key Points:
- 🔒 Embeddings: Local processing (privacy-compliant)
- 🌐 LLM: Only at query time (minimal data exposure)
- 📊 Vector Search: Fast k-NN retrieval
- ✅ Evaluation: Optional quality scoring
```

## Running Tests

Using pytest (recommended):
```bash
pytest tests/test_core.py -v
```

Or using unittest:
```bash
python -m unittest tests.test_core
```

## Technical Choices

### Chunking Strategy: Sentence-Based with Overlap

**Why this approach?**
- **Semantic Coherence**: Splitting by sentences preserves semantic meaning better than fixed-size character splitting
- **Context Preservation**: Overlap (100 characters) ensures important context isn't lost at chunk boundaries
- **Variable-Length Handling**: Adapts to different section lengths in the document
- **Quality Control**: Minimum chunk size (50 chars) ensures meaningful chunks, maximum size (500 chars) prevents overly long chunks

**Implementation**: The `chunk_text_sentences()` function in `utils.py` splits text into sentences, then groups them into chunks of ~300 characters with 100-character overlap between chunks.

### Vector Search: FAISS k-NN

**Why FAISS?**
- **Efficiency**: FAISS provides highly optimized vector search, suitable for production use
- **Speed**: Fast approximate nearest neighbor search even with large datasets
- **Scalability**: Can handle thousands to millions of vectors efficiently
- **L2 Distance with Normalization**: We normalize embeddings and use L2 distance, which is equivalent to cosine similarity for normalized vectors

**Implementation**: We use `faiss.IndexFlatL2` with L2 normalization for cosine similarity search. The index stores all chunk embeddings and supports fast k-NN queries.

### Embeddings: Sentence-Transformers (all-MiniLM-L6-v2)

**Why sentence-transformers?**
- **No API Key Required**: Runs locally, no external API calls needed for embeddings
- **Cost-Effective**: Free to use, no per-request costs
- **Privacy Compliance**: All embedding generation happens locally, data never leaves your machine - critical for sensitive HR documentation
- **Quality**: High-quality embeddings that capture semantic meaning effectively
- **Dimension**: 384 dimensions provide good balance between quality and storage
- **Fast**: Efficient inference on CPU

**Implementation**: The system uses the `all-MiniLM-L6-v2` model from sentence-transformers, which is downloaded automatically on first use and cached locally. **Important**: Embeddings are generated entirely on your local machine - your document content is never sent to external servers during indexing.

### LLM: OpenRouter with GPT-3.5-turbo

**Why OpenRouter?**
- **Model Access**: Provides access to GPT-3.5-turbo and other models via unified API
- **Cost Efficiency**: Competitive pricing for API access
- **Flexibility**: Easy to switch models if needed
- **As Specified**: Required by project specifications

**Implementation**: The system uses OpenRouter's chat completions API with the model `openai/gpt-3.5-turbo`. **Important**: The LLM is only called at query time, and only the retrieved chunks (not the entire document) are sent to generate the answer. This minimizes data exposure and ensures privacy compliance.

### Storage: FAISS Index + JSON Metadata

**Why this approach?**
- **FAISS Index**: Binary format for fast vector search
- **JSON Metadata**: Human-readable chunk text and metadata for easy inspection
- **Separation of Concerns**: Vectors and metadata stored separately for flexibility
- **Portability**: Can be easily moved or versioned

## Evaluator Agent

The evaluator agent (bonus feature) scores answer quality on a 0-10 scale:

- **Chunk Relevance (0-4 points)**: How relevant are retrieved chunks to the question?
- **Answer Accuracy (0-4 points)**: Is the answer factually correct based on chunks?
- **Completeness (0-2 points)**: Does the answer fully address the question?

The evaluator uses the OpenRouter LLM to analyze the question, answer, and retrieved chunks, providing both a score and detailed reasoning. The evaluator is implemented in `src/evaluator.py` as a separate module.

## Known Limitations

1. **API Dependencies**: Requires active internet connection and valid OpenRouter API key for LLM generation (embeddings are local)
2. **Cost**: Each query uses API calls for LLM generation (costs vary by usage, embeddings are free)
3. **Language**: Currently optimized for English text
4. **Context Window**: Limited by LLM context window (GPT-3.5-turbo: ~4K tokens)
5. **No Fine-tuning**: Uses general-purpose models, not fine-tuned on company-specific data
6. **Single Document**: Currently processes one document at a time (can be extended)
7. **Model Download**: First run requires downloading the embedding model (~80MB)

## Extending the System

### Adding More Documents

1. Append or merge content into `data/faq_document.txt`
2. Re-run `python src/build_index.py` to rebuild the index

### Changing Chunking Strategy

Modify `chunk_text_sentences()` in `src/utils.py` to implement different chunking approaches (e.g., semantic chunking, fixed-size, etc.).

### Using Different Embedding Models

Update `EMBEDDING_MODEL` in `.env` to use a different sentence-transformers model. Popular alternatives:
- `all-mpnet-base-v2` (higher quality, larger, slower)
- `paraphrase-MiniLM-L6-v2` (similar to current, optimized for paraphrasing)
- `multi-qa-MiniLM-L6-cos-v1` (optimized for Q&A tasks)

### Hybrid Search

The current implementation uses pure vector search. To add hybrid search (vector + keyword), you could:
- Add keyword matching using TF-IDF or BM25
- Combine vector similarity scores with keyword match scores
- Re-rank results based on combined scores

## Troubleshooting

### "Index files not found" Error

Run `python src/build_index.py` first to create the index files.

### "API key not found" Error

Ensure your `.env` file exists and contains a valid `OPENROUTER_API_KEY`, or export it as an environment variable.

### Low Quality Answers

- Check if the FAQ document contains relevant information
- Try adjusting `TOP_K` in `src/query.py` to retrieve more/fewer chunks
- Review chunk quality - may need to adjust chunking parameters in `src/utils.py`

### FAISS Installation Issues

On some systems, you may need to install FAISS differently:
```bash
# For CPU-only
pip install faiss-cpu

# For GPU support (if available)
pip install faiss-gpu
```

### Sentence-Transformers Model Download Issues

If the model download fails:
1. Check your internet connection
2. The model will be cached in `~/.cache/huggingface/` (or similar) after first download
3. You can manually download models using: `python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"`

### Memory Issues with Large Documents

If you encounter memory issues:
- Reduce `MAX_CHUNK_SIZE` in `src/utils.py`
- Process documents in batches
- Use a smaller embedding model

## License

This project is created for educational purposes as part of the AEM2PI Project Assignment.

## Author

ByHENRY - AI Engineer
