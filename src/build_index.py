"""
Data Pipeline: Document Chunking and Embedding Generation

This script loads a plain text FAQ document, chunks it intelligently,
generates embeddings using sentence-transformers (local, no API key required),
and stores them in a FAISS index for efficient vector search.
"""

import os
import json
import pickle
from typing import List, Dict
from dotenv import load_dotenv
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from utils import load_document, chunk_text_sentences

# Load environment variables
load_dotenv()

# Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_OVERLAP = 100  # Characters of overlap between chunks
MIN_CHUNK_SIZE = 50  # Minimum chunk size in characters
MAX_CHUNK_SIZE = 500  # Maximum chunk size in characters


def generate_embeddings(chunks: List[Dict], model: SentenceTransformer) -> List[np.ndarray]:
    """
    Generate embeddings for all chunks using sentence-transformers.
    
    Args:
        chunks: List of chunk dictionaries
        model: SentenceTransformer model instance
        
    Returns:
        List of embedding vectors as numpy arrays
    """
    chunk_texts = [chunk['text'] for chunk in chunks]
    
    # Generate embeddings (sentence-transformers handles batching automatically)
    embeddings = model.encode(chunk_texts, show_progress_bar=True, convert_to_numpy=True)
    
    # Convert to list of numpy arrays
    embeddings_list = [embeddings[i].astype(np.float32) for i in range(len(embeddings))]
    
    return embeddings_list


def create_faiss_index(embeddings: List[np.ndarray]) -> faiss.Index:
    """
    Create FAISS index for efficient vector search.
    
    Args:
        embeddings: List of embedding vectors
        
    Returns:
        FAISS index object
    """
    if not embeddings:
        raise ValueError("No embeddings provided")
    
    # Get embedding dimension
    dimension = len(embeddings[0])
    
    # Create FAISS index (L2 distance for cosine similarity after normalization)
    index = faiss.IndexFlatL2(dimension)
    
    # Convert embeddings to numpy array and normalize for cosine similarity
    embeddings_array = np.vstack(embeddings).astype('float32')
    faiss.normalize_L2(embeddings_array)  # Normalize for cosine similarity
    
    # Add embeddings to index
    index.add(embeddings_array)
    
    return index


def save_index(index: faiss.Index, chunks: List[Dict], embeddings: List[np.ndarray], 
               output_dir: str = "data"):
    """
    Save FAISS index, chunks metadata, and embeddings to disk.
    
    Args:
        index: FAISS index object
        chunks: List of chunk dictionaries
        embeddings: List of embedding vectors
        output_dir: Directory to save files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save FAISS index
    faiss.write_index(index, os.path.join(output_dir, "faiss_index.bin"))
    
    # Save chunks metadata
    with open(os.path.join(output_dir, "chunks_metadata.json"), 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    
    # Save embeddings (optional, for debugging)
    with open(os.path.join(output_dir, "embeddings.pkl"), 'wb') as f:
        pickle.dump(embeddings, f)
    
    print(f"Index saved to {output_dir}/")
    print(f"Total chunks: {len(chunks)}")
    print(f"Index dimension: {index.d}")


def main():
    """
    Main function to build the search index.
    """
    # Load embedding model (downloads on first use if not cached)
    print(f"Loading embedding model: {EMBEDDING_MODEL}...")
    print("Note: Model will be downloaded on first use if not already cached.")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")
    
    # Load document
    doc_path = "data/faq_document.txt"
    print(f"\nLoading document from {doc_path}...")
    document = load_document(doc_path)
    print(f"Document loaded: {len(document)} characters")
    
    # Chunk document
    print("\nChunking document...")
    chunks = chunk_text_sentences(document, chunk_size=300, overlap=CHUNK_OVERLAP,
                                 min_chunk_size=MIN_CHUNK_SIZE, max_chunk_size=MAX_CHUNK_SIZE)
    print(f"Created {len(chunks)} chunks")
    
    if len(chunks) < 20:
        print(f"Warning: Only {len(chunks)} chunks created. Minimum 20 required.")
    
    # Generate embeddings
    print("\nGenerating embeddings...")
    embeddings = generate_embeddings(chunks, model)
    print(f"Generated {len(embeddings)} embeddings")
    
    # Create FAISS index
    print("\nCreating FAISS index...")
    index = create_faiss_index(embeddings)
    
    # Save index and metadata
    print("\nSaving index...")
    save_index(index, chunks, embeddings)
    
    print("\nIndex building complete!")


if __name__ == "__main__":
    main()
