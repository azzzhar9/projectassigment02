# How the RAG FAQ Chatbot Works - File by File Explanation

This document provides a detailed, step-by-step explanation of how each file in the project works and how they interact with each other.

## Table of Contents

1. [Project Overview](#project-overview)
2. [File-by-File Breakdown](#file-by-file-breakdown)
3. [Data Flow](#data-flow)
4. [Execution Flow](#execution-flow)

---

## Project Overview

The RAG FAQ Chatbot uses a **Retrieval-Augmented Generation (RAG)** architecture:
- **Retrieval**: Finds relevant document chunks using vector similarity search
- **Augmentation**: Adds retrieved chunks as context to the LLM
- **Generation**: LLM generates answers based on the context

---

## File-by-File Breakdown

### 1. Configuration Files

#### `.env.example`
**Purpose**: Template for environment variables

**What it contains:**
```
OPENROUTER_API_KEY=your-openrouter-api-key-here
EMBEDDING_MODEL=all-MiniLM-L6-v2
LLM_MODEL=openai/gpt-3.5-turbo
```

**How it works:**
- Users copy this to `.env` and add their actual API key
- The `.env` file is loaded by `python-dotenv` in all Python scripts
- Provides configuration without hardcoding sensitive data

**Used by**: `build_index.py`, `query.py`, `evaluator.py`

---

#### `requirements.txt`
**Purpose**: Lists all Python dependencies

**What it contains:**
```
sentence-transformers>=2.2.0  # For local embedding generation
faiss-cpu>=1.7.4              # For vector search
numpy>=1.24.0                 # For numerical operations
python-dotenv>=1.0.0          # For loading .env files
requests>=2.31.0              # For API calls to OpenRouter
torch>=2.0.0                  # Required by sentence-transformers
```

**How it works:**
- `pip install -r requirements.txt` installs all dependencies
- Ensures everyone has the same package versions
- Makes the project reproducible

---

#### `.gitignore`
**Purpose**: Tells Git which files to ignore

**What it excludes:**
- `.env` (contains API keys - should never be committed)
- `__pycache__/` (Python bytecode - auto-generated)
- Virtual environments
- IDE files

**Why it's important:**
- Prevents committing sensitive data (API keys)
- Keeps repository clean
- Reduces repository size

---

### 2. Core Source Files

#### `src/utils.py`
**Purpose**: Shared utility functions for text processing

**Functions:**

**1. `load_document(file_path: str) -> str`**
```python
def load_document(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()
```
- **What it does**: Reads a plain text file and returns its content
- **Used by**: `build_index.py` to load the FAQ document
- **Example**: Loads `data/faq_document.txt` (1,956 words)

**2. `split_into_sentences(text: str) -> List[str]`**
```python
def split_into_sentences(text: str) -> List[str]:
    sentence_pattern = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_pattern, text)
    return [s.strip() for s in sentences if s.strip()]
```
- **What it does**: Splits text into individual sentences using regex
- **How it works**: 
  - Looks for sentence endings (`.`, `!`, `?`) followed by whitespace
  - Returns a list of sentences
- **Example**: "Hello world. How are you?" → ["Hello world.", "How are you?"]

**3. `chunk_text_sentences(...) -> List[Dict]`**
```python
def chunk_text_sentences(text: str, chunk_size: int = 300, 
                        overlap: int = 100, ...) -> List[Dict]:
```
- **What it does**: Splits document into meaningful chunks with overlap
- **How it works**:
  1. Splits text into sentences
  2. Groups sentences into chunks of ~300 characters
  3. Adds 100-character overlap between chunks (preserves context)
  4. Ensures chunks are 50-500 characters (meaningful size)
  5. Returns list of chunk dictionaries with `chunk_id`, `text`, `length`
- **Why overlap**: Prevents losing context at chunk boundaries
- **Example**: 1,956-word document → 39 chunks
- **Used by**: `build_index.py` during indexing

**Key Parameters:**
- `chunk_size=300`: Target chunk size in characters
- `overlap=100`: Characters to overlap between chunks
- `min_chunk_size=50`: Minimum meaningful chunk size
- `max_chunk_size=500`: Maximum chunk size to prevent overly long chunks

---

#### `src/build_index.py`
**Purpose**: Builds the searchable index from the FAQ document

**Execution Flow:**

**Step 1: Load Configuration**
```python
load_dotenv()  # Loads .env file
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
```
- Loads environment variables
- Sets embedding model (default: `all-MiniLM-L6-v2`)

**Step 2: Load Embedding Model**
```python
model = SentenceTransformer(EMBEDDING_MODEL)
```
- **What happens**: 
  - First time: Downloads model from HuggingFace (~80MB)
  - Subsequent times: Loads from cache
- **Model details**:
  - Name: `all-MiniLM-L6-v2`
  - Dimensions: 384
  - Type: Local (no API key needed)
  - Speed: Fast on CPU

**Step 3: Load Document**
```python
document = load_document("data/faq_document.txt")
```
- Reads the FAQ document (1,956 words)
- Returns as a single string

**Step 4: Chunk Document**
```python
chunks = chunk_text_sentences(document, chunk_size=300, overlap=100)
```
- **Input**: Full document text
- **Process**: 
  - Splits into sentences
  - Groups into chunks with overlap
- **Output**: List of 39 chunk dictionaries
- **Example chunk**:
  ```json
  {
    "chunk_id": 0,
    "text": "Employees are entitled to various types of leave...",
    "length": 347
  }
  ```

**Step 5: Generate Embeddings**
```python
embeddings = generate_embeddings(chunks, model)
```
- **What happens**:
  1. Extracts text from each chunk
  2. Passes to `SentenceTransformer.encode()`
  3. Model converts text to 384-dimensional vectors
  4. Returns list of numpy arrays
- **Example**: 
  - Input: "How many days of leave?"
  - Output: `[0.123, -0.456, 0.789, ...]` (384 numbers)
- **Why embeddings**: Convert text to numbers that capture meaning
- **Privacy**: All processing happens locally (no API calls)

**Step 6: Create FAISS Index**
```python
index = create_faiss_index(embeddings)
```
- **What happens**:
  1. Creates `faiss.IndexFlatL2` (L2 distance index)
  2. Normalizes embeddings (for cosine similarity)
  3. Adds all embeddings to the index
- **Why FAISS**: Fast vector search (sub-millisecond queries)
- **Index type**: Flat L2 (exact search, fast for <1M vectors)

**Step 7: Save to Disk**
```python
save_index(index, chunks, embeddings)
```
- **Saves**:
  - `data/faiss_index.bin`: FAISS index (binary, fast loading)
  - `data/chunks_metadata.json`: Chunk text and metadata (human-readable)
  - `data/embeddings.pkl`: Embeddings cache (optional, for debugging)

**Complete Flow:**
```
Document → Chunks → Embeddings → FAISS Index → Save Files
```

**When to run**: Once, before first query (or when document changes)

---

#### `src/query.py`
**Purpose**: Handles user queries and generates answers

**Execution Flow:**

**Step 1: Load Configuration**
```python
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-3.5-turbo")
```
- Loads API key for OpenRouter
- Sets LLM model (default: `openai/gpt-3.5-turbo`)

**Step 2: Load Index**
```python
index, chunks_metadata = load_index("data")
```
- **What happens**:
  1. Reads `data/faiss_index.bin` (FAISS index)
  2. Reads `data/chunks_metadata.json` (chunk text)
  3. Returns both for use in search
- **Why needed**: Index contains embeddings, metadata contains text

**Step 3: Load Embedding Model**
```python
model = SentenceTransformer(EMBEDDING_MODEL)
```
- Loads same model used during indexing
- **Critical**: Must be same model for embeddings to be comparable

**Step 4: Embed User Question**
```python
question_embedding = embed_question(user_question, model)
```
- **What happens**:
  1. User question: "How many days of annual leave?"
  2. Model converts to 384-dimensional vector
  3. Normalizes for cosine similarity
- **Output**: Numpy array of shape (1, 384)

**Step 5: Vector Search**
```python
relevant_chunks = search_chunks(index, question_embedding, chunks_metadata, k=5)
```
- **What happens**:
  1. FAISS searches for 5 most similar embeddings (k-NN)
  2. Calculates cosine similarity (distance in embedding space)
  3. Returns top 5 chunks with similarity scores
- **How similarity works**:
  - Similar questions have similar embeddings
  - Cosine similarity measures angle between vectors
  - Higher score = more relevant
- **Example output**:
  ```json
  [
    {
      "chunk_id": 0,
      "text": "Employees are entitled to...",
      "similarity_score": 0.95
    },
    ...
  ]
  ```

**Step 6: Generate Answer**
```python
system_answer = generate_answer(user_question, relevant_chunks)
```
- **What happens**:
  1. **Builds prompt**:
     ```
     Context:
     Chunk 1: [retrieved chunk text]
     Chunk 2: [retrieved chunk text]
     ...
     
     Question: [user question]
     
     Answer:
     ```
  2. **Calls OpenRouter API**:
     - Endpoint: `https://openrouter.ai/api/v1/chat/completions`
     - Model: `openai/gpt-3.5-turbo`
     - Input: Prompt with question + chunks
     - Output: Generated answer text
  3. **Returns answer**: LLM-generated response based on chunks
- **Why this works**: LLM uses retrieved chunks as context to answer accurately
- **Privacy**: Only chunks sent (not full document)

**Step 7: Format Output**
```python
output = format_output(user_question, system_answer, relevant_chunks)
```
- **What happens**:
  1. Structures data into JSON format
  2. Includes:
     - `user_question`: Original question
     - `system_answer`: LLM-generated answer
     - `chunks_related`: List of chunks used (with scores)
- **Output format**:
  ```json
  {
    "user_question": "How many days of annual leave?",
    "system_answer": "Full-time employees accrue...",
    "chunks_related": [
      {
        "chunk_id": 0,
        "text": "...",
        "similarity_score": 0.95
      }
    ]
  }
  ```

**Step 8: Optional Evaluation**
```python
if include_evaluation:
    evaluation = evaluate_answer(...)
    output["evaluation"] = evaluation
```
- Calls `evaluator.py` if `--evaluate` flag is used
- Adds quality score to output

**Complete Flow:**
```
Question → Embedding → Vector Search → Chunks → LLM → Answer → JSON
```

**Command-line usage:**
```bash
python src/query.py "How many days of annual leave?" --evaluate
```

---

#### `src/evaluator.py`
**Purpose**: Evaluates answer quality (bonus feature)

**Function: `evaluate_answer(...) -> Dict`**

**What it does:**
- Scores answer quality on 0-10 scale
- Provides detailed reasoning

**Evaluation Criteria:**

**1. Chunk Relevance (0-4 points)**
- How relevant are retrieved chunks to the question?
- 4: Highly relevant, directly addresses question
- 3: Mostly relevant with useful information
- 2: Partially relevant, some off-topic
- 1: Mostly irrelevant
- 0: Completely irrelevant

**2. Answer Accuracy (0-4 points)**
- Is the answer factually correct based on chunks?
- 4: Completely accurate
- 3: Mostly accurate with minor issues
- 2: Partially accurate but has errors
- 1: Mostly inaccurate
- 0: Completely incorrect

**3. Completeness (0-2 points)**
- Does the answer fully address the question?
- 2: Fully addresses all aspects
- 1: Partially addresses
- 0: Does not address

**How it works:**
1. **Builds evaluation prompt**:
   ```
   Evaluate based on:
   - Chunk relevance (0-4)
   - Answer accuracy (0-4)
   - Completeness (0-2)
   
   User Question: [question]
   Retrieved Chunks: [chunks]
   System Answer: [answer]
   
   Provide JSON with scores and reason.
   ```

2. **Calls OpenRouter API**:
   - Uses same LLM as answer generation
   - Lower temperature (0.3) for more consistent scoring

3. **Parses response**:
   - Extracts JSON from LLM response
   - Handles markdown code blocks if present
   - Returns structured evaluation

**Output format:**
```json
{
  "score": 8,
  "reason": "Highly relevant chunks directly address the question...",
  "chunk_relevance_score": 3,
  "answer_accuracy_score": 3,
  "completeness_score": 2
}
```

**Used by**: `query.py` when `--evaluate` flag is used

---

### 3. Data Files

#### `data/faq_document.txt`
**Purpose**: Source FAQ document

**What it contains:**
- Plain text FAQ covering:
  - Company policies (leave, remote work, benefits)
  - Feature documentation (onboarding, performance reviews)
  - Procedures (password reset, payroll, troubleshooting)
- **Size**: 1,956 words (exceeds 1,000-word requirement)
- **Format**: Plain text, structured with sections

**How it's used:**
1. Loaded by `build_index.py`
2. Chunked into 39 pieces
3. Embedded and indexed
4. Chunks retrieved during queries

---

#### `data/faiss_index.bin`
**Purpose**: FAISS vector index (binary file)

**What it contains:**
- Binary representation of FAISS index
- All chunk embeddings in searchable format
- Optimized for fast k-NN search

**How it's created:**
- Generated by `build_index.py`
- Saved using `faiss.write_index()`

**How it's used:**
- Loaded by `query.py` using `faiss.read_index()`
- Used for vector similarity search

**Size**: ~60KB for 39 chunks

---

#### `data/chunks_metadata.json`
**Purpose**: Human-readable chunk metadata

**What it contains:**
```json
[
  {
    "chunk_id": 0,
    "text": "Employees are entitled to...",
    "length": 347
  },
  ...
]
```

**How it's created:**
- Generated by `build_index.py`
- Saved as JSON for readability

**How it's used:**
- Loaded by `query.py`
- Provides chunk text for retrieved embeddings
- Used to build LLM prompt

**Why both files**: 
- `faiss_index.bin`: Fast vector search (binary)
- `chunks_metadata.json`: Human-readable text (JSON)

---

#### `data/embeddings.pkl`
**Purpose**: Pickled embeddings cache (optional)

**What it contains:**
- Python pickle file with raw embeddings
- Used for debugging/analysis

**Note**: Not strictly necessary for operation, but useful for debugging

---

### 4. Test Files

#### `tests/test_core.py`
**Purpose**: Unit tests for core functionality

**Test Classes:**

**1. `TestChunking`**
- `test_load_document()`: Verifies document loading
- `test_split_into_sentences()`: Tests sentence splitting
- `test_chunk_text_sentences()`: Tests chunking produces valid chunks
- `test_chunk_minimum_requirement()`: Ensures ≥20 chunks created

**2. `TestOutputFormat`**
- `test_format_output_structure()`: Verifies JSON structure
- `test_output_json_serializable()`: Ensures valid JSON

**3. `TestSampleQueries`**
- `test_sample_queries_exists()`: Verifies sample queries file exists and is valid

**4. `TestIndexLoading`**
- `test_load_index_file_not_found()`: Tests error handling

**5. `TestEvaluatorLogic`**
- `test_evaluation_score_range()`: Ensures scores are 0-10
- `test_evaluation_criteria()`: Verifies criteria structure

**How to run:**
```bash
pytest tests/test_core.py -v
```

---

### 5. Output Files

#### `outputs/sample_queries.json`
**Purpose**: Example query-response pairs

**What it contains:**
- 4 example queries with expected outputs
- Demonstrates system capabilities
- Shows JSON output format

**Example:**
```json
[
  {
    "user_question": "How many days of annual leave do I get per year?",
    "system_answer": "Full-time employees accrue...",
    "chunks_related": [...]
  },
  ...
]
```

**Used for**: Documentation and testing

---

### 6. Documentation Files

#### `README.md`
**Purpose**: Main project documentation

**Contains:**
- Project overview
- Setup instructions
- Usage examples
- Architecture diagram
- Technical choices
- Troubleshooting

**Sections:**
1. Overview
2. Project Flow Diagram
3. Quick Start Guide
4. Setup
5. Usage
6. Privacy & Data Processing
7. Architecture
8. Technical Choices
9. Troubleshooting

---

#### `PROJECT_ANALYSIS.md`
**Purpose**: Project analysis and improvement suggestions

**Contains:**
- Project strengths
- What was done well
- Areas for improvement
- Performance metrics
- Learning outcomes

---

## Data Flow

### Indexing Flow (One-time)
```
1. data/faq_document.txt
   ↓ (load_document)
2. Full document text (string)
   ↓ (chunk_text_sentences)
3. List of chunks (39 chunks)
   ↓ (generate_embeddings)
4. List of embeddings (39 × 384-dimensional vectors)
   ↓ (create_faiss_index)
5. FAISS index object
   ↓ (save_index)
6. data/faiss_index.bin + data/chunks_metadata.json
```

### Query Flow (On-demand)
```
1. User question (string)
   ↓ (embed_question)
2. Question embedding (384-dimensional vector)
   ↓ (search_chunks - FAISS k-NN)
3. Top 5 relevant chunks (with similarity scores)
   ↓ (generate_answer - OpenRouter API)
4. Generated answer (string)
   ↓ (format_output)
5. JSON response
   ↓ (optional: evaluate_answer)
6. JSON with evaluation scores
```

## Execution Flow

### First Time Setup
```bash
1. pip install -r requirements.txt
2. Create .env file with OPENROUTER_API_KEY
3. python src/build_index.py
   → Downloads model (first time)
   → Creates index files
4. python src/query.py "question"
   → Loads index
   → Answers question
```

### Subsequent Queries
```bash
python src/query.py "question" --evaluate
→ Uses existing index
→ Generates answer
→ Evaluates quality
```

## Key Concepts

### 1. Embeddings
- **What**: Numerical representations of text that capture meaning
- **How**: Sentence-transformers model converts text → numbers
- **Why**: Enables mathematical similarity search
- **Example**: "leave policy" and "time off rules" have similar embeddings

### 2. Vector Search
- **What**: Finding similar vectors in high-dimensional space
- **How**: FAISS calculates cosine similarity between embeddings
- **Why**: Fast retrieval of relevant information
- **Result**: Top-k most similar chunks

### 3. RAG (Retrieval-Augmented Generation)
- **Retrieval**: Find relevant chunks using vector search
- **Augmentation**: Add chunks as context to LLM
- **Generation**: LLM generates answer from context
- **Benefit**: Accurate answers without fine-tuning

### 4. Privacy
- **Local embeddings**: No API calls, data stays on machine
- **Minimal LLM exposure**: Only retrieved chunks sent (not full document)
- **Compliance**: Suitable for sensitive HR documentation

## Summary

**Indexing** (`build_index.py`):
- Document → Chunks → Embeddings → Index → Save

**Querying** (`query.py`):
- Question → Embedding → Search → Chunks → LLM → Answer

**Evaluation** (`evaluator.py`):
- Question + Answer + Chunks → LLM → Quality Score

All files work together to create a complete RAG system that can answer questions from documentation accurately and efficiently.

