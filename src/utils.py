"""
Utility functions for RAG FAQ Chatbot.

Shared utilities for document loading, text chunking, and common operations.
"""

import os
import re
from typing import List, Dict


def load_document(file_path: str) -> str:
    """
    Load plain text document from file.
    
    Args:
        file_path: Path to the text file
        
    Returns:
        Document content as string
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using regex pattern.
    
    Args:
        text: Input text
        
    Returns:
        List of sentences
    """
    # Pattern to split on sentence endings (., !, ?) followed by whitespace
    sentence_pattern = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_pattern, text)
    # Filter out empty sentences and strip whitespace
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def chunk_text_sentences(text: str, chunk_size: int = 300, overlap: int = 100,
                        min_chunk_size: int = 50, max_chunk_size: int = 500) -> List[Dict[str, any]]:
    """
    Chunk text using sentence-based approach with overlap.
    Ensures chunks are semantically coherent and meet size requirements.
    
    Args:
        text: Input text to chunk
        chunk_size: Target chunk size in characters
        overlap: Number of characters to overlap between chunks
        min_chunk_size: Minimum chunk size in characters
        max_chunk_size: Maximum chunk size in characters
        
    Returns:
        List of chunk dictionaries with text and metadata
    """
    sentences = split_into_sentences(text)
    chunks = []
    current_chunk = []
    current_length = 0
    chunk_id = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        
        # If adding this sentence would exceed max size, finalize current chunk
        if current_length + sentence_length > max_chunk_size and current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) >= min_chunk_size:
                chunks.append({
                    'chunk_id': chunk_id,
                    'text': chunk_text,
                    'length': len(chunk_text)
                })
                chunk_id += 1
            
            # Start new chunk with overlap (last few sentences from previous chunk)
            overlap_text = ' '.join(current_chunk)
            if len(overlap_text) > overlap:
                # Take last part for overlap
                overlap_sentences = []
                temp_text = ""
                for s in reversed(current_chunk):
                    if len(temp_text + s) <= overlap:
                        overlap_sentences.insert(0, s)
                        temp_text = s + " " + temp_text
                    else:
                        break
                current_chunk = overlap_sentences
                current_length = len(' '.join(current_chunk))
            else:
                current_chunk = []
                current_length = 0
        
        # Add sentence to current chunk
        current_chunk.append(sentence)
        current_length += sentence_length + 1  # +1 for space
        
        # If chunk reaches target size, finalize it
        if current_length >= chunk_size:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) >= min_chunk_size:
                chunks.append({
                    'chunk_id': chunk_id,
                    'text': chunk_text,
                    'length': len(chunk_text)
                })
                chunk_id += 1
            
            # Prepare overlap for next chunk
            overlap_sentences = []
            temp_text = ""
            for s in reversed(current_chunk):
                if len(temp_text + s) <= overlap:
                    overlap_sentences.insert(0, s)
                    temp_text = s + " " + temp_text
                else:
                    break
            current_chunk = overlap_sentences
            current_length = len(' '.join(current_chunk))
    
    # Add remaining sentences as final chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        if len(chunk_text) >= min_chunk_size:
            chunks.append({
                'chunk_id': chunk_id,
                'text': chunk_text,
                'length': len(chunk_text)
            })
    
    return chunks

