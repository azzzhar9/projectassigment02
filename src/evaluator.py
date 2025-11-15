"""
Evaluator Agent: Answer Quality Scoring

This module provides an evaluator agent that scores answer quality (0-10)
based on chunk relevance, answer accuracy, and completeness.
"""

import os
import json
from typing import List, Dict
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

# Configuration
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-3.5-turbo")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


def evaluate_answer(user_question: str, system_answer: str, 
                   chunks_related: List[Dict]) -> Dict:
    """
    Evaluator agent that scores answer quality (0-10) based on:
    - Chunk relevance (0-4 points)
    - Answer accuracy (0-4 points)
    - Completeness (0-2 points)
    
    Args:
        user_question: Original user question
        system_answer: Generated answer
        chunks_related: List of relevant chunks used
        
    Returns:
        Dictionary with score (0-10) and reason
    """
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables")
    
    # Prepare evaluation prompt
    chunks_text = "\n".join([
        f"Chunk {i+1} (similarity: {chunk.get('similarity_score', 0):.3f}): {chunk['text'][:200]}..."
        for i, chunk in enumerate(chunks_related)
    ])
    
    eval_prompt = f"""You are an evaluation agent for a RAG-based FAQ system. Evaluate the quality of the answer based on the following criteria:

1. Chunk Relevance (0-4 points): How relevant are the retrieved chunks to the user's question?
   - 4: Highly relevant chunks that directly address the question
   - 3: Mostly relevant chunks with some useful information
   - 2: Partially relevant chunks, some may be off-topic
   - 1: Mostly irrelevant chunks
   - 0: Completely irrelevant chunks

2. Answer Accuracy (0-4 points): Is the answer factually correct based on the chunks?
   - 4: Completely accurate and correct
   - 3: Mostly accurate with minor issues
   - 2: Partially accurate but has some errors
   - 1: Mostly inaccurate
   - 0: Completely incorrect

3. Completeness (0-2 points): Does the answer fully address the question?
   - 2: Fully addresses all aspects of the question
   - 1: Partially addresses the question
   - 0: Does not address the question

User Question: {user_question}

Retrieved Chunks:
{chunks_text}

System Answer: {system_answer}

Provide your evaluation in the following JSON format:
{{
    "chunk_relevance_score": <0-4>,
    "answer_accuracy_score": <0-4>,
    "completeness_score": <0-2>,
    "total_score": <0-10>,
    "reason": "<detailed explanation of the scores>"
}}"""
    
    # Use OpenRouter API
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/faq-rag-chatbot",
        "X-Title": "FAQ RAG Chatbot",
    }
    
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "You are an evaluation agent. Return only valid JSON."},
            {"role": "user", "content": eval_prompt}
        ],
        "temperature": 0.3,
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
        eval_text = result["choices"][0]["message"]["content"].strip()
    except requests.exceptions.HTTPError as e:
        error_detail = ""
        try:
            error_detail = response.json()
        except:
            error_detail = response.text
        raise ValueError(f"OpenRouter API error: {e}\nDetails: {error_detail}")
    
    # Parse JSON response
    try:
        # Extract JSON from response (handle markdown code blocks)
        if "```json" in eval_text:
            eval_text = eval_text.split("```json")[1].split("```")[0].strip()
        elif "```" in eval_text:
            eval_text = eval_text.split("```")[1].split("```")[0].strip()
        
        eval_result = json.loads(eval_text)
        
        return {
            "score": eval_result.get("total_score", 0),
            "reason": eval_result.get("reason", "No reason provided"),
            "chunk_relevance_score": eval_result.get("chunk_relevance_score", 0),
            "answer_accuracy_score": eval_result.get("answer_accuracy_score", 0),
            "completeness_score": eval_result.get("completeness_score", 0)
        }
    except json.JSONDecodeError:
        # Fallback if JSON parsing fails
        return {
            "score": 5,
            "reason": "Could not parse evaluation response. Manual review recommended.",
            "chunk_relevance_score": 2,
            "answer_accuracy_score": 2,
            "completeness_score": 1
        }

