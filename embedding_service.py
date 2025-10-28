"""
Embedding service for generating text embeddings using AWS Bedrock.
Provides a clean interface for embedding text with proper error handling and logging.
"""

import json
import logging
import time
from typing import List, Optional, Dict, Any
import boto3
from botocore.exceptions import ClientError, BotoCoreError

from config import config

logger = logging.getLogger("doc_processor.embedding")

class EmbeddingService:
    """Service for generating text embeddings using AWS Bedrock."""
    
    def __init__(self):
        """Initialize the embedding service with AWS Bedrock client."""
        self.logger = logging.getLogger("doc_processor.embedding.EmbeddingService")
        
        try:
            # Initialize AWS Bedrock client
            aws_config = config.get_aws_config()
            self.bedrock_client = boto3.client(
                'bedrock-runtime',
                **aws_config
            )
            
            self.model_id = config.EMBEDDING_MODEL
            self.embedding_dimension = config.EMBEDDING_DIMENSION
            
            self.logger.info(f"Initialized embedding service with model: {self.model_id}")
            self.logger.info(f"Expected embedding dimension: {self.embedding_dimension}")
            
            # Statistics tracking
            self.stats = {
                "total_embeddings": 0,
                "total_tokens": 0,
                "total_time": 0.0,
                "errors": 0,
                "retries": 0
            }
            
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding service: {e}")
            raise
    
    def embed_text(self, text: str, max_retries: int = 3, retry_delay: float = 1.0) -> Optional[List[float]]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            
        Returns:
            List of floats representing the embedding, or None if failed
        """
        if not text or not text.strip():
            self.logger.warning("Empty text provided for embedding")
            return None
        
        # Clean and prepare text
        cleaned_text = self._clean_text(text)
        
        for attempt in range(max_retries + 1):
            try:
                start_time = time.time()
                
                # Prepare the request body
                body = json.dumps({
                    "inputText": cleaned_text
                })
                
                # Call Bedrock
                response = self.bedrock_client.invoke_model(
                    modelId=self.model_id,
                    body=body,
                    contentType='application/json',
                    accept='application/json'
                )
                
                # Parse response
                response_body = json.loads(response['body'].read())
                embedding = response_body.get('embedding')
                
                if not embedding:
                    raise ValueError("No embedding returned from model")
                
                # Validate embedding dimension
                if len(embedding) != self.embedding_dimension:
                    self.logger.warning(
                        f"Unexpected embedding dimension: {len(embedding)}, expected: {self.embedding_dimension}"
                    )
                
                # Update statistics
                elapsed_time = time.time() - start_time
                self.stats["total_embeddings"] += 1
                self.stats["total_tokens"] += len(cleaned_text.split())
                self.stats["total_time"] += elapsed_time
                
                self.logger.debug(f"Generated embedding in {elapsed_time:.3f}s for text length: {len(cleaned_text)}")
                
                return embedding
                
            except (ClientError, BotoCoreError) as e:
                error_code = getattr(e, 'response', {}).get('Error', {}).get('Code', 'Unknown')
                self.logger.warning(f"AWS error on attempt {attempt + 1}: {error_code} - {str(e)}")
                
                if attempt < max_retries:
                    self.stats["retries"] += 1
                    self.logger.info(f"Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    self.stats["errors"] += 1
                    self.logger.error(f"Failed to generate embedding after {max_retries + 1} attempts")
                    return None
                    
            except Exception as e:
                self.logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                
                if attempt < max_retries:
                    self.stats["retries"] += 1
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    self.stats["errors"] += 1
                    return None
        
        return None
    
    def embed_batch(self, texts: List[str], batch_size: int = None, 
                   progress_callback: Optional[callable] = None) -> List[Optional[List[float]]]:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Size of processing batches (defaults to config.BATCH_SIZE)
            progress_callback: Optional callback function for progress updates
            
        Returns:
            List of embeddings (same order as input texts)
        """
        if batch_size is None:
            batch_size = config.BATCH_SIZE
        
        self.logger.info(f"Starting batch embedding for {len(texts)} texts")
        
        embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for batch_idx in range(0, len(texts), batch_size):
            batch_texts = texts[batch_idx:batch_idx + batch_size]
            batch_num = (batch_idx // batch_size) + 1
            
            self.logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_texts)} texts)")
            
            batch_embeddings = []
            for text in batch_texts:
                embedding = self.embed_text(text)
                batch_embeddings.append(embedding)
            
            embeddings.extend(batch_embeddings)
            
            # Call progress callback if provided
            if progress_callback:
                progress = len(embeddings) / len(texts)
                progress_callback(progress, len(embeddings), len(texts))
        
        successful_embeddings = sum(1 for emb in embeddings if emb is not None)
        self.logger.info(f"Batch embedding completed: {successful_embeddings}/{len(texts)} successful")
        
        return embeddings
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and prepare text for embedding.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        cleaned = ' '.join(text.split())
        
        # Truncate if too long (Titan has a limit)
        max_length = 8000  # Conservative limit for Titan
        if len(cleaned) > max_length:
            cleaned = cleaned[:max_length]
            self.logger.debug(f"Truncated text from {len(text)} to {len(cleaned)} characters")
        
        return cleaned
    
    def get_stats(self) -> Dict[str, Any]:
        """Get embedding service statistics."""
        stats = self.stats.copy()
        
        if stats["total_embeddings"] > 0:
            stats["avg_time_per_embedding"] = stats["total_time"] / stats["total_embeddings"]
            stats["avg_tokens_per_embedding"] = stats["total_tokens"] / stats["total_embeddings"]
            stats["success_rate"] = (stats["total_embeddings"] - stats["errors"]) / stats["total_embeddings"]
        else:
            stats["avg_time_per_embedding"] = 0.0
            stats["avg_tokens_per_embedding"] = 0.0
            stats["success_rate"] = 0.0
        
        return stats
    
    def reset_stats(self):
        """Reset embedding service statistics."""
        self.stats = {
            "total_embeddings": 0,
            "total_tokens": 0,
            "total_time": 0.0,
            "errors": 0,
            "retries": 0
        }
        self.logger.info("Reset embedding service statistics")
    
    def log_stats(self):
        """Log current embedding service statistics."""
        stats = self.get_stats()
        
        self.logger.info("=== Embedding Service Statistics ===")
        self.logger.info(f"Total embeddings: {stats['total_embeddings']}")
        self.logger.info(f"Total tokens: {stats['total_tokens']}")
        self.logger.info(f"Total time: {stats['total_time']:.2f}s")
        self.logger.info(f"Errors: {stats['errors']}")
        self.logger.info(f"Retries: {stats['retries']}")
        self.logger.info(f"Success rate: {stats['success_rate']:.2%}")
        self.logger.info(f"Avg time per embedding: {stats['avg_time_per_embedding']:.3f}s")
        self.logger.info(f"Avg tokens per embedding: {stats['avg_tokens_per_embedding']:.1f}")
        self.logger.info("=====================================")

# Global embedding service instance
_embedding_service = None

def get_embedding_service() -> EmbeddingService:
    """Get the global embedding service instance."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service

def embed_text(text: str) -> Optional[List[float]]:
    """Convenience function to embed a single text."""
    service = get_embedding_service()
    return service.embed_text(text)

def embed_batch(texts: List[str], **kwargs) -> List[Optional[List[float]]]:
    """Convenience function to embed a batch of texts."""
    service = get_embedding_service()
    return service.embed_batch(texts, **kwargs) 