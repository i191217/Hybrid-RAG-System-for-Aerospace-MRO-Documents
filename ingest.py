"""
Improved document ingestion module for processing and storing document chunks.
Supports multiple vector databases and includes comprehensive logging and error handling.
"""

import sqlite3
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from config import config, logger
from vector_db import get_vector_db, VectorPoint
from embedding_service import get_embedding_service

@dataclass
class ChunkData:
    """Represents a document chunk with metadata."""
    chunk_id: str
    content: str
    filename: str
    page_number: Optional[int] = None
    chunk_index: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

class DocumentIngester:
    """Handles ingestion of document chunks into vector database."""
    
    def __init__(self, collection_name: str = None):
        """Initialize the document ingester."""
        self.collection_name = collection_name or config.COLLECTION_NAME
        self.logger = logging.getLogger("doc_processor.ingest.DocumentIngester")
        
        # Initialize services
        self.vector_db = get_vector_db(self.collection_name)
        self.embedding_service = get_embedding_service()
        
        # Statistics tracking
        self.stats = {
            "total_chunks": 0,
            "processed_chunks": 0,
            "skipped_chunks": 0,
            "failed_chunks": 0,
            "total_time": 0.0,
            "embedding_time": 0.0,
            "db_time": 0.0
        }
        
        self.logger.info(f"Initialized DocumentIngester for collection: {self.collection_name}")
        self.logger.info(f"Vector database type: {config.VECTOR_DB_TYPE}")
    
    def setup_collection(self, clear_existing: bool = None) -> bool:
        """
        Setup the vector database collection.
        
        Args:
            clear_existing: Whether to clear existing collection (defaults to config setting)
            
        Returns:
            True if successful, False otherwise
        """
        if clear_existing is None:
            clear_existing = config.CLEAR_COLLECTION
        
        try:
            self.logger.info(f"Setting up collection: {self.collection_name}")
            
            success = self.vector_db.create_collection(
                dimension=config.EMBEDDING_DIMENSION,
                clear_existing=clear_existing
            )
            
            if success:
                info = self.vector_db.get_collection_info()
                self.logger.info(f"Collection setup successful: {info}")
                return True
            else:
                self.logger.error("Failed to setup collection")
                return False
                
        except Exception as e:
            self.logger.error(f"Error setting up collection: {e}")
            return False
    
    def load_chunks_from_db(self, db_path: str = None, limit: int = None) -> List[ChunkData]:
        """
        Load chunks from SQLite database.
        
        Args:
            db_path: Path to SQLite database (defaults to config setting)
            limit: Maximum number of chunks to load (for testing)
            
        Returns:
            List of ChunkData objects
        """
        if db_path is None:
            db_path = config.SQLITE_DB_PATH
        
        if not Path(db_path).exists():
            self.logger.error(f"Database file not found: {db_path}")
            return []
        
        try:
            self.logger.info(f"Loading chunks from database: {db_path}")
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get table schema to understand available columns
            cursor.execute("PRAGMA table_info(chunks)")
            columns = [row[1] for row in cursor.fetchall()]
            self.logger.debug(f"Available columns: {columns}")
            
            # Build query based on available columns
            base_query = "SELECT chunk_id, content, filename"
            if "page_number" in columns:
                base_query += ", page_number"
            if "chunk_index" in columns:
                base_query += ", chunk_index"
            
            base_query += " FROM chunks"
            
            if limit and config.TEST_MODE:
                base_query += f" LIMIT {limit}"
            
            cursor.execute(base_query)
            rows = cursor.fetchall()
            
            chunks = []
            for row in rows:
                chunk_data = ChunkData(
                    chunk_id=row[0],
                    content=row[1],
                    filename=row[2],
                    page_number=row[3] if len(row) > 3 else None,
                    chunk_index=row[4] if len(row) > 4 else None,
                    metadata={
                        "filename": row[2],
                        "page_number": row[3] if len(row) > 3 else None,
                        "chunk_index": row[4] if len(row) > 4 else None
                    }
                )
                chunks.append(chunk_data)
            
            conn.close()
            
            self.logger.info(f"Loaded {len(chunks)} chunks from database")
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error loading chunks from database: {e}")
            return []
    
    def check_existing_chunks(self, chunk_ids: List[str]) -> List[str]:
        """
        Check which chunks already exist in the vector database.
        
        Args:
            chunk_ids: List of chunk IDs to check
            
        Returns:
            List of chunk IDs that already exist
        """
        # This is a simplified implementation
        # In practice, you might want to implement this differently based on your vector DB
        self.logger.debug(f"Checking {len(chunk_ids)} chunks for existing embeddings")
        
        # For now, assume no chunks exist (could be improved with actual DB queries)
        return []
    
    def process_chunks(self, chunks: List[ChunkData], batch_size: int = None, 
                      max_workers: int = 4, skip_existing: bool = True) -> bool:
        """
        Process and ingest chunks into vector database.
        
        Args:
            chunks: List of chunks to process
            batch_size: Size of processing batches
            max_workers: Number of worker threads for parallel processing
            skip_existing: Whether to skip chunks that already exist
            
        Returns:
            True if successful, False otherwise
        """
        if not chunks:
            self.logger.warning("No chunks to process")
            return True
        
        if batch_size is None:
            batch_size = config.BATCH_SIZE
        
        start_time = time.time()
        self.stats["total_chunks"] = len(chunks)
        
        self.logger.info(f"Starting ingestion of {len(chunks)} chunks")
        self.logger.info(f"Batch size: {batch_size}, Max workers: {max_workers}")
        
        try:
            # Check for existing chunks if requested
            if skip_existing:
                chunk_ids = [chunk.chunk_id for chunk in chunks]
                existing_ids = set(self.check_existing_chunks(chunk_ids))
                chunks = [chunk for chunk in chunks if chunk.chunk_id not in existing_ids]
                
                if existing_ids:
                    self.logger.info(f"Skipping {len(existing_ids)} existing chunks")
                    self.stats["skipped_chunks"] = len(existing_ids)
            
            if not chunks:
                self.logger.info("All chunks already exist, nothing to process")
                return True
            
            # Process chunks in batches
            total_batches = (len(chunks) + batch_size - 1) // batch_size
            
            for batch_idx in range(0, len(chunks), batch_size):
                batch_chunks = chunks[batch_idx:batch_idx + batch_size]
                batch_num = (batch_idx // batch_size) + 1
                
                self.logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_chunks)} chunks)")
                
                success = self._process_batch(batch_chunks, max_workers)
                if not success:
                    self.logger.error(f"Failed to process batch {batch_num}")
                    return False
            
            # Log final statistics
            total_time = time.time() - start_time
            self.stats["total_time"] = total_time
            self._log_final_stats()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error during chunk processing: {e}")
            return False
    
    def _process_batch(self, chunks: List[ChunkData], max_workers: int) -> bool:
        """Process a batch of chunks with parallel embedding generation."""
        try:
            # Generate embeddings for the batch
            embedding_start = time.time()
            
            texts = [chunk.content for chunk in chunks]
            embeddings = self.embedding_service.embed_batch(
                texts,
                progress_callback=self._embedding_progress_callback
            )
            
            embedding_time = time.time() - embedding_start
            self.stats["embedding_time"] += embedding_time
            
            # Prepare vector points
            vector_points = []
            for chunk, embedding in zip(chunks, embeddings):
                if embedding is not None:
                    vector_point = VectorPoint(
                        id=chunk.chunk_id,
                        vector=embedding,
                        metadata=chunk.metadata or {}
                    )
                    vector_points.append(vector_point)
                else:
                    self.logger.warning(f"Failed to generate embedding for chunk: {chunk.chunk_id}")
                    self.stats["failed_chunks"] += 1
            
            if not vector_points:
                self.logger.error("No valid embeddings generated for batch")
                return False
            
            # Upsert to vector database
            db_start = time.time()
            success = self.vector_db.upsert_points(vector_points)
            db_time = time.time() - db_start
            self.stats["db_time"] += db_time
            
            if success:
                self.stats["processed_chunks"] += len(vector_points)
                self.logger.info(f"Successfully processed {len(vector_points)} chunks")
                return True
            else:
                self.logger.error("Failed to upsert points to vector database")
                return False
                
        except Exception as e:
            self.logger.error(f"Error processing batch: {e}")
            return False
    
    def _embedding_progress_callback(self, progress: float, completed: int, total: int):
        """Callback for embedding progress updates."""
        self.logger.debug(f"Embedding progress: {progress:.1%} ({completed}/{total})")
    
    def _log_final_stats(self):
        """Log final ingestion statistics."""
        stats = self.stats
        
        self.logger.info("=== Ingestion Statistics ===")
        self.logger.info(f"Total chunks: {stats['total_chunks']}")
        self.logger.info(f"Processed chunks: {stats['processed_chunks']}")
        self.logger.info(f"Skipped chunks: {stats['skipped_chunks']}")
        self.logger.info(f"Failed chunks: {stats['failed_chunks']}")
        self.logger.info(f"Total time: {stats['total_time']:.2f}s")
        self.logger.info(f"Embedding time: {stats['embedding_time']:.2f}s")
        self.logger.info(f"Database time: {stats['db_time']:.2f}s")
        
        if stats['processed_chunks'] > 0:
            avg_time = stats['total_time'] / stats['processed_chunks']
            self.logger.info(f"Average time per chunk: {avg_time:.3f}s")
        
        success_rate = stats['processed_chunks'] / stats['total_chunks'] if stats['total_chunks'] > 0 else 0
        self.logger.info(f"Success rate: {success_rate:.1%}")
        self.logger.info("============================")
    
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

def run_ingestion(db_path: str = None, collection_name: str = None, 
                 test_mode: bool = None, clear_collection: bool = None) -> bool:
    """
    Run the complete ingestion process.
    
    Args:
        db_path: Path to SQLite database
        collection_name: Name of vector database collection
        test_mode: Whether to run in test mode
        clear_collection: Whether to clear existing collection
        
    Returns:
        True if successful, False otherwise
    """
    # Use config defaults if not specified
    if db_path is None:
        db_path = config.SQLITE_DB_PATH
    if collection_name is None:
        collection_name = config.COLLECTION_NAME
    if test_mode is None:
        test_mode = config.TEST_MODE
    if clear_collection is None:
        clear_collection = config.CLEAR_COLLECTION
    
    logger.info("=== Starting Document Ingestion ===")
    logger.info(f"Database: {db_path}")
    logger.info(f"Collection: {collection_name}")
    logger.info(f"Vector DB: {config.VECTOR_DB_TYPE}")
    logger.info(f"Test mode: {test_mode}")
    logger.info(f"Clear collection: {clear_collection}")
    
    ingester = None
    try:
        # Initialize ingester
        ingester = DocumentIngester(collection_name)
        
        # Setup collection
        if not ingester.setup_collection(clear_collection):
            logger.error("Failed to setup collection")
            return False
        
        # Load chunks
        limit = config.TEST_CHUNKS if test_mode else None
        chunks = ingester.load_chunks_from_db(db_path, limit)
        
        if not chunks:
            logger.error("No chunks loaded from database")
            return False
        
        # Process chunks
        success = ingester.process_chunks(chunks)
        
        if success:
            # Log collection info
            info = ingester.get_collection_info()
            logger.info(f"Final collection info: {info}")
            
            # Log embedding service stats
            ingester.embedding_service.log_stats()
            
            logger.info("=== Ingestion Completed Successfully ===")
            return True
        else:
            logger.error("=== Ingestion Failed ===")
            return False
            
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        return False
    finally:
        if ingester:
            ingester.close()

if __name__ == "__main__":
    """Run ingestion when script is executed directly."""
    import sys
    
    # Parse command line arguments
    test_mode = "--test" in sys.argv
    clear_collection = "--clear" in sys.argv
    
    success = run_ingestion(test_mode=test_mode, clear_collection=clear_collection)
    sys.exit(0 if success else 1) 