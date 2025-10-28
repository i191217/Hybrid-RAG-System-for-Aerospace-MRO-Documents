#!/usr/bin/env python3
"""
Centralized Similarity Service for Deduplication
Implements various similarity metrics with configurable thresholds for document and chunk deduplication.
"""

import re
import hashlib
import logging
from typing import List, Dict, Tuple, Set, Optional
from difflib import SequenceMatcher

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False

class SimilarityService:
    """Service for calculating content similarity with configurable thresholds."""
    
    def __init__(self, similarity_threshold: float = 0.5):
        """
        Initialize similarity service.
        
        Args:
            similarity_threshold: Default threshold (0.5 = 50% similarity)
        """
        self.logger = logging.getLogger(__name__)
        self.similarity_threshold = similarity_threshold
        
        # Initialize TF-IDF vectorizer for semantic similarity
        if SKLEARN_AVAILABLE:
            self.tfidf_vectorizer = TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 2),
                max_features=1000
            )
        else:
            self.tfidf_vectorizer = None
            
        self.logger.info(f"SimilarityService initialized with {similarity_threshold:.1%} threshold")
    
    def calculate_content_similarity(self, text1: str, text2: str, method: str = "combined") -> float:
        """
        Calculate similarity between two text contents using multiple methods.
        
        Args:
            text1: First text content
            text2: Second text content  
            method: Similarity method ('combined', 'cosine', 'jaccard', 'sequence', 'fuzzy')
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not text1 or not text2:
            return 0.0
        
        # Normalize texts for comparison
        norm_text1 = self._normalize_text(text1)
        norm_text2 = self._normalize_text(text2)
        
        if norm_text1 == norm_text2:
            return 1.0
        
        if method == "combined":
            return self._calculate_combined_similarity(norm_text1, norm_text2)
        elif method == "cosine":
            return self._calculate_cosine_similarity(norm_text1, norm_text2)
        elif method == "jaccard":
            return self._calculate_jaccard_similarity(norm_text1, norm_text2)
        elif method == "sequence":
            return self._calculate_sequence_similarity(norm_text1, norm_text2)
        elif method == "fuzzy":
            return self._calculate_fuzzy_similarity(norm_text1, norm_text2)
        else:
            raise ValueError(f"Unknown similarity method: {method}")
    
    def are_documents_similar(self, text1: str, text2: str, threshold: Optional[float] = None) -> Tuple[bool, float]:
        """
        Check if two documents are similar above the threshold.
        
        Args:
            text1: First document text
            text2: Second document text
            threshold: Custom threshold (uses default if None)
            
        Returns:
            Tuple of (is_similar, similarity_score)
        """
        if threshold is None:
            threshold = self.similarity_threshold
            
        similarity = self.calculate_content_similarity(text1, text2)
        return similarity >= threshold, similarity
    
    def are_chunks_similar(self, chunk1: str, chunk2: str, threshold: Optional[float] = None) -> Tuple[bool, float]:
        """
        Check if two chunks are similar above the threshold.
        
        Args:
            chunk1: First chunk content
            chunk2: Second chunk content
            threshold: Custom threshold (uses default if None)
            
        Returns:
            Tuple of (is_similar, similarity_score)
        """
        if threshold is None:
            threshold = self.similarity_threshold
            
        # For chunks, we might want a slightly higher threshold to preserve diversity
        chunk_threshold = max(threshold, 0.6)  # At least 60% for chunks
        
        similarity = self.calculate_content_similarity(chunk1, chunk2)
        return similarity >= chunk_threshold, similarity
    
    def deduplicate_chunks(self, chunks: List[Dict], threshold: Optional[float] = None) -> Tuple[List[Dict], int]:
        """
        Remove duplicate chunks based on similarity threshold.
        
        Args:
            chunks: List of chunk dictionaries with 'content' key
            threshold: Custom threshold (uses default if None)
            
        Returns:
            Tuple of (deduplicated_chunks, removed_count)
        """
        if threshold is None:
            threshold = self.similarity_threshold
            
        deduplicated = []
        removed_count = 0
        
        for chunk in chunks:
            content = chunk.get('content', '')
            if not content:
                continue
                
            is_duplicate = False
            for existing_chunk in deduplicated:
                existing_content = existing_chunk.get('content', '')
                
                is_similar, similarity = self.are_chunks_similar(content, existing_content, threshold)
                if is_similar:
                    is_duplicate = True
                    self.logger.debug(f"Removed duplicate chunk (similarity: {similarity:.2%}): {content[:50]}...")
                    break
            
            if not is_duplicate:
                deduplicated.append(chunk)
            else:
                removed_count += 1
        
        if removed_count > 0:
            self.logger.info(f"Removed {removed_count} duplicate chunks out of {len(chunks)} (kept {len(deduplicated)})")
        
        return deduplicated, removed_count
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Remove document headers that vary between processing runs
        normalized = re.sub(r'document:\s*[^\n]+\n', '', text.lower())
        normalized = re.sub(r'page:\s*\d+\n', '', normalized)
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized.strip())
        
        # Remove common processing artifacts
        normalized = re.sub(r'\[pymupdf extracted text\]', '', normalized)
        normalized = re.sub(r'\[ocr text from \w+\]', '', normalized)
        normalized = re.sub(r'\[page \d+, \w+\]', '', normalized)
        
        return normalized
    
    def _calculate_combined_similarity(self, text1: str, text2: str) -> float:
        """Calculate combined similarity using multiple methods."""
        similarities = []
        
        # Sequence similarity (fast, always available)
        seq_sim = self._calculate_sequence_similarity(text1, text2)
        similarities.append(seq_sim)
        
        # Jaccard similarity (word overlap, always available)
        jaccard_sim = self._calculate_jaccard_similarity(text1, text2)
        similarities.append(jaccard_sim)
        
        # Cosine similarity (if sklearn available)
        if SKLEARN_AVAILABLE:
            cosine_sim = self._calculate_cosine_similarity(text1, text2)
            similarities.append(cosine_sim)
        
        # Fuzzy similarity (if rapidfuzz available)
        if RAPIDFUZZ_AVAILABLE:
            fuzzy_sim = self._calculate_fuzzy_similarity(text1, text2)
            similarities.append(fuzzy_sim)
        
        # Return weighted average
        return sum(similarities) / len(similarities)
    
    def _calculate_sequence_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity using SequenceMatcher."""
        return SequenceMatcher(None, text1, text2).ratio()
    
    def _calculate_jaccard_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity based on word sets."""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _calculate_cosine_similarity(self, text1: str, text2: str) -> float:
        """Calculate TF-IDF cosine similarity."""
        if not SKLEARN_AVAILABLE or not self.tfidf_vectorizer:
            return 0.0
        
        try:
            # Fit and transform texts
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([text1, text2])
            
            # Calculate cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            
            return float(similarity_matrix[0][0])
            
        except Exception as e:
            self.logger.debug(f"Cosine similarity calculation failed: {e}")
            return 0.0
    
    def _calculate_fuzzy_similarity(self, text1: str, text2: str) -> float:
        """Calculate fuzzy similarity using rapidfuzz."""
        if not RAPIDFUZZ_AVAILABLE:
            return 0.0
        
        try:
            # Use token sort ratio for better results with reordered text
            similarity = fuzz.token_sort_ratio(text1, text2) / 100.0
            return similarity
            
        except Exception as e:
            self.logger.debug(f"Fuzzy similarity calculation failed: {e}")
            return 0.0
    
    def get_similarity_stats(self, chunks: List[Dict]) -> Dict[str, any]:
        """Get statistics about similarity within a set of chunks."""
        if len(chunks) < 2:
            return {"total_comparisons": 0, "similarities": []}
        
        similarities = []
        comparisons = 0
        
        for i, chunk1 in enumerate(chunks):
            for j, chunk2 in enumerate(chunks[i+1:], i+1):
                content1 = chunk1.get('content', '')
                content2 = chunk2.get('content', '')
                
                if content1 and content2:
                    similarity = self.calculate_content_similarity(content1, content2)
                    similarities.append(similarity)
                    comparisons += 1
        
        stats = {
            "total_comparisons": comparisons,
            "similarities": similarities,
            "avg_similarity": sum(similarities) / len(similarities) if similarities else 0.0,
            "max_similarity": max(similarities) if similarities else 0.0,
            "min_similarity": min(similarities) if similarities else 0.0,
            "above_threshold": sum(1 for s in similarities if s >= self.similarity_threshold),
            "threshold": self.similarity_threshold
        }
        
        return stats
    
    def filter_similar_results(self, results: List[Dict], content_key: str = 'content', threshold: Optional[float] = None) -> List[Dict]:
        """
        Filter a list of results to remove similar items.
        
        Args:
            results: List of result dictionaries
            content_key: Key in dictionary containing text content
            threshold: Similarity threshold (uses default if None)
            
        Returns:
            Filtered list with similar items removed
        """
        if threshold is None:
            threshold = self.similarity_threshold
            
        filtered = []
        
        for result in results:
            content = result.get(content_key, '')
            if not content:
                continue
                
            is_similar = False
            for existing_result in filtered:
                existing_content = existing_result.get(content_key, '')
                
                similarity = self.calculate_content_similarity(content, existing_content)
                if similarity >= threshold:
                    is_similar = True
                    self.logger.debug(f"Filtered similar result (similarity: {similarity:.2%}): {content[:50]}...")
                    break
            
            if not is_similar:
                filtered.append(result)
        
        return filtered 