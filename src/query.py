"""
Query Pipeline: Vector Search and Answer Generation

This script accepts user questions, performs vector search using FAISS k-NN,
retrieves relevant chunks, and generates answers using OpenRouter LLM.
Returns structured JSON with user_question, system_answer, and chunks_related.
"""

import os
import json
import sys
import argparse
from typing import List, Dict, Tuple
from dotenv import load_dotenv
import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer

from utils import chunk_text_sentences

# Load environment variables
load_dotenv()

# Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-3.5-turbo")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TOP_K = 5  # Number of top chunks to retrieve


def load_index(index_dir: str = "data") -> Tuple[faiss.Index, List[Dict]]:
    """
    Load FAISS index and chunks metadata.
    
    Args:
        index_dir: Directory containing index files
        
    Returns:
        Tuple of (FAISS index, chunks metadata)
    """
    index_path = os.path.join(index_dir, "faiss_index.bin")
    metadata_path = os.path.join(index_dir, "chunks_metadata.json")
    
    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Index files not found. Please run build_index.py first."
        )
    
    index = faiss.read_index(index_path)
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        chunks_metadata = json.load(f)
    
    return index, chunks_metadata


def embed_question(question: str, model: SentenceTransformer) -> np.ndarray:
    """
    Generate embedding for user question.
    
    Args:
        question: User's question text
        model: SentenceTransformer model instance
        
    Returns:
        Embedding vector as numpy array
    """
    embedding = model.encode([question], convert_to_numpy=True)[0]
    embedding = embedding.astype(np.float32)
    # Normalize for cosine similarity
    faiss.normalize_L2(embedding.reshape(1, -1))
    return embedding.reshape(1, -1)


def search_chunks(index: faiss.Index, question_embedding: np.ndarray, 
                  chunks_metadata: List[Dict], k: int = TOP_K) -> List[Dict]:
    """
    Perform k-NN vector search to find most relevant chunks.
    
    Args:
        index: FAISS index
        question_embedding: Question embedding vector
        chunks_metadata: List of chunk metadata dictionaries
        k: Number of top results to return
        
    Returns:
        List of relevant chunks with similarity scores
    """
    # Perform k-NN search
    distances, indices = index.search(question_embedding, k)
    
    # Retrieve chunks with similarity scores
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(chunks_metadata):
            chunk = chunks_metadata[idx].copy()
            # Convert L2 distance to similarity score (1 / (1 + distance))
            similarity_score = 1 / (1 + distances[0][i])
            chunk['similarity_score'] = float(similarity_score)
            chunk['distance'] = float(distances[0][i])
            results.append(chunk)
    
    # Sort by similarity score (descending)
    results.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    return results


def generate_answer(question: str, relevant_chunks: List[Dict]) -> str:
    """
    Generate answer using LLM based on retrieved chunks via OpenRouter.
    
    Args:
        question: User's question
        relevant_chunks: List of relevant chunk dictionaries
        
    Returns:
        Generated answer text
    """
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables")
    
    # Prepare context from retrieved chunks
    context = "\n\n".join([
        f"Chunk {i+1}: {chunk['text']}" 
        for i, chunk in enumerate(relevant_chunks)
    ])
    
    # Create prompt
    prompt = f"""You are a helpful HR support assistant. Answer the user's question based on the following context from the company documentation.

Context:
{context}

Question: {question}

Instructions:
- Provide a clear, accurate, and complete answer based only on the context provided
- If the context doesn't contain enough information to answer the question, say so
- Use a friendly and professional tone
- Be concise but thorough

Answer:"""
    
    # Use OpenRouter API
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/faq-rag-chatbot",  # Optional but recommended
        "X-Title": "FAQ RAG Chatbot",  # Optional but recommended
    }
    
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful HR support assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
    }
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        answer = result["choices"][0]["message"]["content"].strip()
    except requests.exceptions.HTTPError as e:
        error_detail = ""
        try:
            error_detail = response.json()
        except:
            error_detail = response.text
        raise ValueError(f"OpenRouter API error: {e}\nDetails: {error_detail}")
    
    return answer


def format_output(user_question: str, system_answer: str, 
                 chunks_related: List[Dict]) -> Dict:
    """
    Format output as structured JSON.
    
    Args:
        user_question: Original user question
        system_answer: Generated answer
        chunks_related: List of relevant chunks
        
    Returns:
        Structured JSON dictionary
    """
    # Format chunks_related (remove internal fields, keep essential info)
    formatted_chunks = [
        {
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "similarity_score": chunk.get("similarity_score", 0.0)
        }
        for chunk in chunks_related
    ]
    
    return {
        "user_question": user_question,
        "system_answer": system_answer,
        "chunks_related": formatted_chunks
    }


def query(user_question: str, include_evaluation: bool = False, 
         index_dir: str = "data", embedding_model_name: str = None) -> Dict:
    """
    Main query function that processes a user question and returns answer.
    
    Args:
        user_question: User's question text
        include_evaluation: Whether to include evaluator scores
        index_dir: Directory containing index files
        embedding_model_name: Name of embedding model (defaults to EMBEDDING_MODEL from env)
        
    Returns:
        Structured JSON response
    """
    # Load embedding model
    model_name = embedding_model_name or EMBEDDING_MODEL
    model = SentenceTransformer(model_name)
    
    # Load index and metadata
    index, chunks_metadata = load_index(index_dir)
    
    # Generate question embedding
    question_embedding = embed_question(user_question, model)
    
    # Search for relevant chunks
    relevant_chunks = search_chunks(index, question_embedding, chunks_metadata, k=TOP_K)
    
    # Generate answer
    system_answer = generate_answer(user_question, relevant_chunks)
    
    # Format output
    output = format_output(user_question, system_answer, relevant_chunks)
    
    # Add evaluation if requested
    if include_evaluation:
        from evaluator import evaluate_answer
        evaluation = evaluate_answer(user_question, system_answer, relevant_chunks)
        output["evaluation"] = evaluation
    
    return output


def main():
    """
    Main function for command-line usage.
    """
    parser = argparse.ArgumentParser(description="Query the FAQ system")
    parser.add_argument("question", type=str, help="User question to answer")
    parser.add_argument("--evaluate", action="store_true", 
                       help="Include answer evaluation")
    parser.add_argument("--index-dir", type=str, default="data",
                       help="Directory containing index files")
    parser.add_argument("--output", type=str, help="Output file path (JSON)")
    
    args = parser.parse_args()
    
    try:
        result = query(args.question, include_evaluation=args.evaluate, 
                     index_dir=args.index_dir)
        
        # Print JSON output
        output_json = json.dumps(result, indent=2, ensure_ascii=False)
        print(output_json)
        
        # Save to file if specified
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(output_json)
            print(f"\nOutput saved to {args.output}")
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
