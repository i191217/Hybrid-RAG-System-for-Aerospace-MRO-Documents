#!/usr/bin/env python3
"""
Query service for RAG-based document querying.
Integrates the AerospaceQueryEngine for answering questions about processed documents.
"""

import os
import sys
import logging
import time
from typing import Optional, Dict, Any

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from core.query_engine import AerospaceQueryEngine
from models.api_models import (
    QueryRequest,
    QueryResponse,
    QuerySource,
    QueryRetrievalStats
)

class QueryService:
    """Service for processing RAG queries against the document database."""
    
    def __init__(self):
        """Initialize the query service."""
        self.logger = logging.getLogger(__name__)
        self.query_engine: Optional[AerospaceQueryEngine] = None
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize the AerospaceQueryEngine."""
        try:
            self.logger.info("Initializing AerospaceQueryEngine...")
            self.query_engine = AerospaceQueryEngine()
            self.is_initialized = True
            self.logger.info("AerospaceQueryEngine initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize AerospaceQueryEngine: {e}")
            self.is_initialized = False
            raise
    
    async def cleanup(self):
        """Cleanup resources."""
        self.is_initialized = False
        self.query_engine = None
        self.logger.info("Query service cleanup complete")
    
    async def process_query(self, request: QueryRequest) -> QueryResponse:
        """
        Process a query request against the document database.
        
        Args:
            request: QueryRequest containing the query and parameters
            
        Returns:
            QueryResponse with the answer and supporting information
        """
        start_time = time.time()
        
        try:
            if not self.is_initialized or not self.query_engine:
                return QueryResponse(
                    success=False,
                    answer="",
                    sources=[],
                    query_time=0.0,
                    confidence_score=0.0,
                    retrieval_stats=QueryRetrievalStats(
                        chunks_found=0,
                        avg_similarity=0.0,
                        generation_time=0.0,
                        query_type="error"
                    ),
                    error="Query engine not initialized"
                )
            
            self.logger.info(f"Processing query: {request.query[:100]}...")
            
            # Process the query using the AerospaceQueryEngine
            result = self.query_engine.query(
                query_text=request.query,
                max_chunks=request.max_chunks,
                temperature=request.temperature
            )
            
            # Convert sources to our API model format
            sources = []
            if hasattr(result, 'sources') and result.sources:
                for source in result.sources:
                    sources.append(QuerySource(
                        filename=source.get('filename', 'unknown'),
                        chunk_index=source.get('chunk_index', 0),
                        vector_id=source.get('vector_id', 'unknown'),
                        content_preview=source.get('content_preview', ''),
                        similarity_score=source.get('similarity_score', 0.0)
                    ))
            
            # Convert retrieval stats
            stats = result.retrieval_stats if hasattr(result, 'retrieval_stats') else {}
            retrieval_stats = QueryRetrievalStats(
                chunks_found=stats.get('chunks_found', 0),
                avg_similarity=stats.get('avg_similarity', 0.0),
                generation_time=stats.get('generation_time', 0.0),
                query_type=stats.get('query_type', 'general')
            )
            
            # Calculate total query time
            query_time = time.time() - start_time
            
            # Create successful response
            response = QueryResponse(
                success=True,
                answer=result.answer if hasattr(result, 'answer') else "No answer generated",
                sources=sources,
                query_time=query_time,
                confidence_score=result.confidence_score if hasattr(result, 'confidence_score') else 0.0,
                retrieval_stats=retrieval_stats
            )
            
            self.logger.info(f"Query processed successfully in {query_time:.2f}s")
            return response
            
        except Exception as e:
            query_time = time.time() - start_time
            self.logger.error(f"Error processing query: {e}")
            
            return QueryResponse(
                success=False,
                answer="",
                sources=[],
                query_time=query_time,
                confidence_score=0.0,
                retrieval_stats=QueryRetrievalStats(
                    chunks_found=0,
                    avg_similarity=0.0,
                    generation_time=0.0,
                    query_type="error"
                ),
                error=str(e)
            )
    
    async def get_query_stats(self) -> Dict[str, Any]:
        """Get statistics about the query engine."""
        try:
            if not self.is_initialized or not self.query_engine:
                return {
                    "error": "Query engine not initialized",
                    "status": "not_initialized"
                }
            
            # Get stats from the query engine if available
            if hasattr(self.query_engine, 'get_stats'):
                stats = self.query_engine.get_stats()
                return {
                    "status": "ready",
                    "query_engine_stats": stats,
                    "is_initialized": True
                }
            else:
                return {
                    "status": "ready",
                    "is_initialized": True,
                    "message": "Query engine ready but no stats available"
                }
                
        except Exception as e:
            self.logger.error(f"Error getting query stats: {e}")
            return {
                "error": str(e),
                "status": "error"
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the query service."""
        try:
            health_status = {
                "query_service": "unknown",
                "query_engine": "unknown",
                "is_initialized": self.is_initialized
            }
            
            if self.is_initialized and self.query_engine:
                # Try a simple test query to verify functionality
                try:
                    test_result = self.query_engine.query(
                        query_text="test",
                        max_chunks=1,
                        temperature=0.1
                    )
                    health_status["query_service"] = "healthy"
                    health_status["query_engine"] = "healthy"
                except Exception as e:
                    health_status["query_service"] = f"error: {e}"
                    health_status["query_engine"] = "unhealthy"
            else:
                health_status["query_service"] = "uninitialized"
                health_status["query_engine"] = "uninitialized"
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"Error during query service health check: {e}")
            return {
                "query_service": f"error: {e}",
                "query_engine": "unknown",
                "is_initialized": False
            } 