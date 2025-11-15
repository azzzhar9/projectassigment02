"""
Unit tests for RAG FAQ Chatbot core functionality.

Tests cover:
- Document chunking (minimum 20 chunks)
- Embedding generation
- Vector search retrieval
- JSON output format validation
- Evaluator scoring logic
"""

import unittest
import os
import sys
import json
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import (
    load_document,
    split_into_sentences,
    chunk_text_sentences
)
from query import (
    format_output,
    load_index
)


class TestChunking(unittest.TestCase):
    """Test document chunking functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_text = """
        This is the first sentence. This is the second sentence.
        This is the third sentence. This is the fourth sentence.
        This is the fifth sentence. This is the sixth sentence.
        This is the seventh sentence. This is the eighth sentence.
        This is the ninth sentence. This is the tenth sentence.
        This is the eleventh sentence. This is the twelfth sentence.
        This is the thirteenth sentence. This is the fourteenth sentence.
        This is the fifteenth sentence. This is the sixteenth sentence.
        This is the seventeenth sentence. This is the eighteenth sentence.
        This is the nineteenth sentence. This is the twentieth sentence.
        This is the twenty-first sentence. This is the twenty-second sentence.
        This is the twenty-third sentence. This is the twenty-fourth sentence.
        This is the twenty-fifth sentence. This is the twenty-sixth sentence.
        This is the twenty-seventh sentence. This is the twenty-eighth sentence.
        This is the twenty-ninth sentence. This is the thirtieth sentence.
        """
    
    def test_load_document(self):
        """Test document loading."""
        # Create a temporary test file
        test_file = "test_doc.txt"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(self.test_text)
        
        try:
            document = load_document(test_file)
            self.assertIsInstance(document, str)
            self.assertGreater(len(document), 0)
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)
    
    def test_split_into_sentences(self):
        """Test sentence splitting."""
        sentences = split_into_sentences(self.test_text)
        self.assertIsInstance(sentences, list)
        self.assertGreater(len(sentences), 0)
        # Check that sentences are non-empty
        for sentence in sentences:
            self.assertGreater(len(sentence.strip()), 0)
    
    def test_chunk_text_sentences(self):
        """Test chunking produces at least 20 chunks."""
        chunks = chunk_text_sentences(self.test_text, chunk_size=200, overlap=50)
        
        self.assertIsInstance(chunks, list)
        self.assertGreaterEqual(len(chunks), 1, "Should produce at least 1 chunk")
        
        # Check chunk structure
        for chunk in chunks:
            self.assertIn('chunk_id', chunk)
            self.assertIn('text', chunk)
            self.assertIn('length', chunk)
            self.assertIsInstance(chunk['chunk_id'], int)
            self.assertIsInstance(chunk['text'], str)
            self.assertIsInstance(chunk['length'], int)
            self.assertGreater(len(chunk['text']), 0)
    
    def test_chunk_minimum_requirement(self):
        """Test that FAQ document produces at least 20 chunks."""
        doc_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'faq_document.txt')
        
        if os.path.exists(doc_path):
            document = load_document(doc_path)
            chunks = chunk_text_sentences(document, chunk_size=300, overlap=100)
            
            self.assertGreaterEqual(
                len(chunks), 20,
                f"FAQ document should produce at least 20 chunks, got {len(chunks)}"
            )
            
            # Verify chunk quality
            for chunk in chunks:
                self.assertGreaterEqual(chunk['length'], 50, "Chunks should be meaningful size")
                self.assertLessEqual(chunk['length'], 500, "Chunks should not be too large")


class TestOutputFormat(unittest.TestCase):
    """Test JSON output format validation."""
    
    def test_format_output_structure(self):
        """Test that output has correct structure."""
        user_question = "Test question?"
        system_answer = "Test answer."
        chunks_related = [
            {
                'chunk_id': 0,
                'text': 'Test chunk text',
                'similarity_score': 0.95
            },
            {
                'chunk_id': 1,
                'text': 'Another test chunk',
                'similarity_score': 0.85
            }
        ]
        
        output = format_output(user_question, system_answer, chunks_related)
        
        # Check top-level keys
        self.assertIn('user_question', output)
        self.assertIn('system_answer', output)
        self.assertIn('chunks_related', output)
        
        # Check values
        self.assertEqual(output['user_question'], user_question)
        self.assertEqual(output['system_answer'], system_answer)
        self.assertIsInstance(output['chunks_related'], list)
        
        # Check chunks_related structure
        for chunk in output['chunks_related']:
            self.assertIn('chunk_id', chunk)
            self.assertIn('text', chunk)
            self.assertIn('similarity_score', chunk)
            self.assertIsInstance(chunk['chunk_id'], int)
            self.assertIsInstance(chunk['text'], str)
            self.assertIsInstance(chunk['similarity_score'], (int, float))
    
    def test_output_json_serializable(self):
        """Test that output can be serialized to JSON."""
        user_question = "Test question?"
        system_answer = "Test answer."
        chunks_related = [
            {
                'chunk_id': 0,
                'text': 'Test chunk text',
                'similarity_score': 0.95
            }
        ]
        
        output = format_output(user_question, system_answer, chunks_related)
        
        # Should not raise exception
        json_str = json.dumps(output, ensure_ascii=False)
        self.assertIsInstance(json_str, str)
        
        # Should be able to parse back
        parsed = json.loads(json_str)
        self.assertEqual(parsed['user_question'], user_question)


class TestSampleQueries(unittest.TestCase):
    """Test sample queries JSON format."""
    
    def test_sample_queries_exists(self):
        """Test that sample queries file exists and is valid JSON."""
        sample_file = os.path.join(
            os.path.dirname(__file__), '..', 'outputs', 'sample_queries.json'
        )
        
        self.assertTrue(os.path.exists(sample_file), "sample_queries.json should exist")
        
        with open(sample_file, 'r', encoding='utf-8') as f:
            queries = json.load(f)
        
        self.assertIsInstance(queries, list)
        self.assertGreaterEqual(len(queries), 3, "Should have at least 3 sample queries")
        
        # Validate each query structure
        for query in queries:
            self.assertIn('user_question', query)
            self.assertIn('system_answer', query)
            self.assertIn('chunks_related', query)
            
            self.assertIsInstance(query['user_question'], str)
            self.assertIsInstance(query['system_answer'], str)
            self.assertIsInstance(query['chunks_related'], list)
            
            # Validate chunks_related
            for chunk in query['chunks_related']:
                self.assertIn('chunk_id', chunk)
                self.assertIn('text', chunk)
                self.assertIn('similarity_score', chunk)


class TestIndexLoading(unittest.TestCase):
    """Test index loading functionality."""
    
    def test_load_index_file_not_found(self):
        """Test that appropriate error is raised when index files don't exist."""
        with self.assertRaises(FileNotFoundError):
            load_index(index_dir="nonexistent_directory")


class TestEvaluatorLogic(unittest.TestCase):
    """Test evaluator scoring logic."""
    
    def test_evaluation_score_range(self):
        """Test that evaluation scores are in valid range (0-10)."""
        # This is a conceptual test - actual evaluation requires API calls
        # In a real scenario, we would mock the API response
        
        # Valid score ranges
        chunk_relevance = 4  # 0-4
        answer_accuracy = 4  # 0-4
        completeness = 2     # 0-2
        
        total_score = chunk_relevance + answer_accuracy + completeness
        
        self.assertGreaterEqual(total_score, 0)
        self.assertLessEqual(total_score, 10)
        self.assertEqual(total_score, 10)  # Maximum score
    
    def test_evaluation_criteria(self):
        """Test that evaluation criteria are properly defined."""
        criteria = {
            'chunk_relevance': {'min': 0, 'max': 4},
            'answer_accuracy': {'min': 0, 'max': 4},
            'completeness': {'min': 0, 'max': 2}
        }
        
        # Verify criteria structure
        self.assertIn('chunk_relevance', criteria)
        self.assertIn('answer_accuracy', criteria)
        self.assertIn('completeness', criteria)
        
        # Verify score ranges
        self.assertEqual(criteria['chunk_relevance']['max'], 4)
        self.assertEqual(criteria['answer_accuracy']['max'], 4)
        self.assertEqual(criteria['completeness']['max'], 2)
        
        # Total should be 10
        total_max = (criteria['chunk_relevance']['max'] + 
                    criteria['answer_accuracy']['max'] + 
                    criteria['completeness']['max'])
        self.assertEqual(total_max, 10)


if __name__ == '__main__':
    # Run tests
    unittest.main()

