"""
Improved query module for RAG (Retrieval-Augmented Generation) operations.
Supports multiple vector databases and includes comprehensive logging and error handling.
"""

import logging
import time
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from openai import OpenAI

from config import config, logger
from vector_db import get_vector_db, SearchResult
from embedding_service import get_embedding_service

@dataclass
class QueryResult:
    """Represents a query result with retrieved documents and generated response."""
    query: str
    retrieved_chunks: List[SearchResult]
    expert_analysis: str
    sources: List[str]
    processing_time: float
    embedding_time: float
    search_time: float
    generation_time: float

class RAGQueryEngine:
    """Handles RAG queries with retrieval and generation."""
    
    def __init__(self, collection_name: str = None):
        """Initialize the RAG query engine."""
        self.collection_name = collection_name or config.COLLECTION_NAME
        self.logger = logging.getLogger("doc_processor.query.RAGQueryEngine")
        
        # Initialize services
        self.vector_db = get_vector_db(self.collection_name)
        self.embedding_service = get_embedding_service()
        
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
        
        # Query parameters
        self.default_temperature = config.DEFAULT_TEMPERATURE
        self.default_threshold = config.DEFAULT_THRESHOLD
        self.default_max_results = config.DEFAULT_MAX_RESULTS
        self.default_limit = config.DEFAULT_LIMIT
        
        self.logger.info(f"Initialized RAGQueryEngine for collection: {self.collection_name}")
        self.logger.info(f"Vector database type: {config.VECTOR_DB_TYPE}")
    
    def query(self, query_text: str, temperature: float = None, threshold: float = None,
              max_results: int = None, limit: int = None, include_sources: bool = True) -> QueryResult:
        """
        Execute a RAG query with retrieval and generation.
        
        Args:
            query_text: The user's query
            temperature: OpenAI temperature parameter
            threshold: Minimum similarity threshold for retrieval
            max_results: Maximum number of results to retrieve
            limit: Limit for vector database search
            include_sources: Whether to include source citations
            
        Returns:
            QueryResult object with all results and metadata
        """
        start_time = time.time()
        
        # Use defaults if not specified
        if temperature is None:
            temperature = self.default_temperature
        if threshold is None:
            threshold = self.default_threshold
        if max_results is None:
            max_results = self.default_max_results
        if limit is None:
            limit = self.default_limit
        
        self.logger.info(f"Processing query: {query_text[:100]}...")
        self.logger.info(f"Parameters - temp: {temperature}, threshold: {threshold}, max_results: {max_results}")
        
        try:
            # Step 1: Generate embedding for query
            embedding_start = time.time()
            query_embedding = self.embedding_service.embed_text(query_text)
            embedding_time = time.time() - embedding_start
            
            if query_embedding is None:
                raise ValueError("Failed to generate embedding for query")
            
            # Step 2: Search vector database
            search_start = time.time()
            search_results = self.vector_db.search(
                query_vector=query_embedding,
                limit=limit,
                score_threshold=threshold
            )
            search_time = time.time() - search_start
            
            self.logger.info(f"Retrieved {len(search_results)} chunks from vector database")
            
            # Step 3: Filter and limit results
            filtered_results = self._filter_results(search_results, max_results, threshold)
            
            if not filtered_results:
                self.logger.warning("No relevant chunks found for query")
                return QueryResult(
                    query=query_text,
                    retrieved_chunks=[],
                    expert_analysis="I couldn't find any relevant information to answer your question.",
                    sources=[],
                    processing_time=time.time() - start_time,
                    embedding_time=embedding_time,
                    search_time=search_time,
                    generation_time=0.0
                )
            
            # Step 4: Generate expert analysis
            generation_start = time.time()
            expert_analysis = self._generate_expert_analysis(
                query_text, filtered_results, temperature
            )
            generation_time = time.time() - generation_start
            
            # Step 5: Extract sources
            sources = self._extract_sources(filtered_results) if include_sources else []
            
            # Step 6: Add citations to analysis
            if include_sources:
                expert_analysis = self._add_citations(expert_analysis, filtered_results)
            
            total_time = time.time() - start_time
            
            self.logger.info(f"Query completed in {total_time:.2f}s")
            self.logger.info(f"Generated {len(expert_analysis)} character response")
            
            return QueryResult(
                query=query_text,
                retrieved_chunks=filtered_results,
                expert_analysis=expert_analysis,
                sources=sources,
                processing_time=total_time,
                embedding_time=embedding_time,
                search_time=search_time,
                generation_time=generation_time
            )
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            return QueryResult(
                query=query_text,
                retrieved_chunks=[],
                expert_analysis=f"An error occurred while processing your query: {str(e)}",
                sources=[],
                processing_time=time.time() - start_time,
                embedding_time=0.0,
                search_time=0.0,
                generation_time=0.0
            )
    
    def _filter_results(self, results: List[SearchResult], max_results: int, 
                       threshold: float) -> List[SearchResult]:
        """Filter and limit search results."""
        # Filter by threshold
        filtered = [r for r in results if r.score >= threshold]
        
        # Sort by score (descending)
        filtered.sort(key=lambda x: x.score, reverse=True)
        
        # Limit results
        limited = filtered[:max_results]
        
        self.logger.debug(f"Filtered {len(results)} -> {len(filtered)} -> {len(limited)} results")
        
        return limited
    
    def _generate_expert_analysis(self, query: str, results: List[SearchResult], 
                                temperature: float) -> str:
        """Generate expert analysis using OpenAI."""
        try:
            # Prepare context from retrieved chunks
            context_parts = []
            for i, result in enumerate(results, 1):
                filename = result.metadata.get('filename', 'Unknown')
                page = result.metadata.get('page_number', 'Unknown')
                content = result.metadata.get('content', '')
                
                # Use content from metadata if available, otherwise use payload
                if not content and result.payload:
                    content = str(result.payload)
                
                context_parts.append(f"[Document {i}: {filename}, Page {page}]\n{content}")
            
            context = "\n\n".join(context_parts)
            
            # Create the prompt
            prompt = f"""You are an expert analyst specializing in aerospace and aviation maintenance, repair, and overhaul (MRO) operations. You have been provided with relevant document excerpts to answer a specific question.

QUESTION: {query}

RELEVANT DOCUMENTS:
{context}

Please provide a comprehensive, expert-level analysis that:
1. Directly answers the question using information from the provided documents
2. Synthesizes information across multiple sources when relevant
3. Maintains technical accuracy and industry-specific terminology
4. Acknowledges any limitations or gaps in the available information
5. Provides actionable insights when possible

Your response should be professional, well-structured, and demonstrate deep understanding of aerospace MRO operations."""

            # Call OpenAI API
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert aerospace MRO analyst."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=2000
            )
            
            analysis = response.choices[0].message.content.strip()
            
            self.logger.debug(f"Generated analysis with {len(analysis)} characters")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error generating expert analysis: {e}")
            return f"Error generating analysis: {str(e)}"
    
    def _extract_sources(self, results: List[SearchResult]) -> List[str]:
        """Extract unique source filenames from results."""
        sources = set()
        
        for result in results:
            filename = result.metadata.get('filename', 'Unknown')
            if filename != 'Unknown':
                sources.add(filename)
        
        return sorted(list(sources))
    
    def _add_citations(self, analysis: str, results: List[SearchResult]) -> str:
        """Add citation information to the analysis."""
        try:
            # Create a mapping of documents to filenames
            doc_map = {}
            for i, result in enumerate(results, 1):
                filename = result.metadata.get('filename', 'Unknown')
                page = result.metadata.get('page_number', 'Unknown')
                doc_map[f"Document {i}"] = f"{filename} (Page {page})"
            
            # Add citations section
            if doc_map:
                citations = "\n\n**Sources:**\n"
                for doc_ref, source in doc_map.items():
                    citations += f"- {doc_ref}: {source}\n"
                
                analysis += citations
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error adding citations: {e}")
            return analysis
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the vector database collection."""
        return self.vector_db.get_collection_info()
    
    def close(self):
        """Close database connections and cleanup resources."""
        try:
            self.vector_db.close()
            self.logger.info("Closed vector database connection")
        except Exception as e:
            self.logger.error(f"Error closing connections: {e}")

def execute_query(query_text: str, collection_name: str = None, **kwargs) -> QueryResult:
    """
    Execute a single RAG query.
    
    Args:
        query_text: The user's query
        collection_name: Name of vector database collection
        **kwargs: Additional query parameters
        
    Returns:
        QueryResult object
    """
    engine = None
    try:
        engine = RAGQueryEngine(collection_name)
        return engine.query(query_text, **kwargs)
    finally:
        if engine:
            engine.close()

def interactive_query_session(collection_name: str = None):
    """
    Run an interactive query session.
    
    Args:
        collection_name: Name of vector database collection
    """
    logger.info("=== Starting Interactive Query Session ===")
    
    engine = None
    try:
        engine = RAGQueryEngine(collection_name)
        
        # Get collection info
        info = engine.get_collection_info()
        logger.info(f"Collection info: {info}")
        
        print("\n" + "="*60)
        print("RAG Query Engine - Interactive Session")
        print("="*60)
        print(f"Collection: {engine.collection_name}")
        print(f"Vector DB: {config.VECTOR_DB_TYPE}")
        print("Type 'quit' or 'exit' to end the session")
        print("Type 'help' for available commands")
        print("="*60)
        
        while True:
            try:
                query = input("\nEnter your query: ").strip()
                
                if query.lower() in ['quit', 'exit']:
                    break
                elif query.lower() == 'help':
                    print("\nAvailable commands:")
                    print("- quit/exit: End the session")
                    print("- help: Show this help message")
                    print("- Any other text: Execute as a query")
                    continue
                elif not query:
                    continue
                
                print(f"\nProcessing query: {query}")
                print("-" * 40)
                
                result = engine.query(query)
                
                print(f"\n**Expert Analysis:**")
                print(result.expert_analysis)
                
                if result.sources:
                    print(f"\n**Sources:** {', '.join(result.sources)}")
                
                print(f"\n**Performance:**")
                print(f"- Total time: {result.processing_time:.2f}s")
                print(f"- Retrieved chunks: {len(result.retrieved_chunks)}")
                print(f"- Embedding time: {result.embedding_time:.2f}s")
                print(f"- Search time: {result.search_time:.2f}s")
                print(f"- Generation time: {result.generation_time:.2f}s")
                
            except KeyboardInterrupt:
                print("\n\nSession interrupted by user")
                break
            except Exception as e:
                print(f"\nError: {e}")
                logger.error(f"Error in interactive session: {e}")
        
        print("\nSession ended")
        
    except Exception as e:
        logger.error(f"Error in interactive session: {e}")
        print(f"Error starting session: {e}")
    finally:
        if engine:
            engine.close()

if __name__ == "__main__":
    """Run interactive query session when script is executed directly."""
    import sys
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        # Single query mode
        query_text = " ".join(sys.argv[1:])
        result = execute_query(query_text)
        
        print(f"\nQuery: {result.query}")
        print(f"\nExpert Analysis:\n{result.expert_analysis}")
        
        if result.sources:
            print(f"\nSources: {', '.join(result.sources)}")
        
        print(f"\nProcessing time: {result.processing_time:.2f}s")
    else:
        # Interactive mode
        interactive_query_session() 