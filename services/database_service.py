"""
SQLite Database Service for Document Processing
"""
import sqlite3
import json
import logging
import hashlib
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from contextlib import contextmanager

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class DocumentMetadata:
    """Metadata for a processed document."""
    filename: str
    file_type: str
    file_size: int
    page_count: Optional[int] = None
    processed_at: Optional[str] = None
    content_hash: Optional[str] = None
    extraction_method: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class ChunkMetadata:
    """Metadata for a text chunk."""
    document_id: int
    chunk_index: int
    page_number: Optional[int] = None
    element_type: str = 'text'
    chunk_size: int = 0
    coordinates: Optional[Dict[str, float]] = None
    context_before: Optional[str] = None
    context_after: Optional[str] = None
    created_at: Optional[str] = None
    vectorized: bool = False
    vector_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class DatabaseService:
    """Service for managing SQLite database operations for document processing."""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize the database service."""
        if db_path is None:
            # Create database in the same directory as the application
            db_path = Path.cwd() / "documents.db"
        
        self.db_path = str(db_path)
        self._lock = threading.Lock()
        logger.info(f"Initializing SQLite database at: {self.db_path}")
        self._init_database()
    
    @contextmanager
    def _get_connection(self, timeout: int = 30):
        """Get a database connection with proper timeout and WAL mode."""
        conn = None
        try:
            conn = sqlite3.connect(
                self.db_path,
                timeout=timeout,
                check_same_thread=False
            )
            # Enable WAL mode for better concurrency
            conn.execute("PRAGMA journal_mode=WAL")
            # Set busy timeout
            conn.execute(f"PRAGMA busy_timeout={timeout * 1000}")
            # Enable foreign key constraints
            conn.execute("PRAGMA foreign_keys = ON")
            yield conn
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e).lower():
                logger.warning(f"Database locked, retrying... {e}")
                raise
            else:
                logger.error(f"Database connection error: {e}")
                raise
        finally:
            if conn:
                conn.close()
    
    def _execute_with_retry(self, operation_func, max_retries: int = 3, delay: float = 0.1):
        """Execute database operation with retry logic for handling locks."""
        for attempt in range(max_retries):
            try:
                return operation_func()
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                    wait_time = delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Database locked, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Database operation failed after {attempt + 1} attempts: {e}")
                    raise
            except Exception as e:
                logger.error(f"Unexpected database error: {e}")
                raise
    
    def _init_database(self):
        """Initialize the SQLite database with required tables."""
        def _create_tables():
            with self._get_connection() as conn:
                # ENHANCED: Create documents table with file_hash column
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        filename TEXT NOT NULL,
                        file_type TEXT NOT NULL,
                        file_size INTEGER NOT NULL,
                        page_count INTEGER,
                        content_hash TEXT UNIQUE,
                        file_hash TEXT,
                        extraction_method TEXT,
                        raw_text TEXT,
                        processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(filename, content_hash)
                    )
                """)
                
                # ENHANCED: Add file_hash column if it doesn't exist (migration)
                try:
                    conn.execute("ALTER TABLE documents ADD COLUMN file_hash TEXT")
                    logger.info("Added file_hash column to existing documents table")
                except Exception as e:
                    if "duplicate column name" in str(e).lower():
                        logger.debug("file_hash column already exists")
                    else:
                        logger.debug(f"Could not add file_hash column: {e}")
                
                # Create chunks table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS chunks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        document_id INTEGER NOT NULL,
                        chunk_index INTEGER NOT NULL,
                        chunk_text TEXT NOT NULL,
                        chunk_size INTEGER NOT NULL,
                        page_number INTEGER,
                        element_type TEXT DEFAULT 'text',
                        coordinates TEXT,
                        context_before TEXT,
                        context_after TEXT,
                        vectorized BOOLEAN DEFAULT FALSE,
                        vector_id TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE,
                        UNIQUE(document_id, chunk_index)
                    )
                """)
                
                # Create processing_stats table for tracking
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS processing_stats (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        document_id INTEGER NOT NULL,
                        processing_stage TEXT NOT NULL,
                        started_at TIMESTAMP,
                        completed_at TIMESTAMP,
                        success BOOLEAN DEFAULT FALSE,
                        error_message TEXT,
                        metadata TEXT,
                        FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
                    )
                """)
                
                # Create indexes for performance
                conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_filename ON documents(filename)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(content_hash)")
                try:
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_file_hash ON documents(file_hash)")
                except Exception as e:
                    logger.debug(f"Could not create file_hash index: {e}")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_vectorized ON chunks(vectorized)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_stats_document ON processing_stats(document_id)")
                
                conn.commit()
                logger.info("SQLite database initialized successfully")
        
        self._execute_with_retry(_create_tables)
    
    def _calculate_content_hash(self, content: str) -> str:
        """Calculate SHA-256 hash of content for deduplication."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def _calculate_file_hash(self, file_content: bytes) -> str:
        """Calculate SHA-256 hash of original file content."""
        return hashlib.sha256(file_content).hexdigest()
    
    def document_exists_by_file_hash(self, file_content: bytes) -> Optional[int]:
        """Check if document already exists based on original file hash."""
        file_hash = self._calculate_file_hash(file_content)
        
        def _check_exists():
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT id FROM documents WHERE file_hash = ?",
                    (file_hash,)
                )
                result = cursor.fetchone()
                return result[0] if result else None
        
        return self._execute_with_retry(_check_exists)
    
    def document_exists(self, filename: str, content: str = None) -> Optional[int]:
        """Check if document already exists in database."""
        def _check_exists():
            with self._get_connection() as conn:
                if content:
                    content_hash = self._calculate_content_hash(content)
                    cursor = conn.execute(
                        "SELECT id FROM documents WHERE filename = ? OR content_hash = ?",
                        (filename, content_hash)
                    )
                else:
                    cursor = conn.execute(
                        "SELECT id FROM documents WHERE filename = ?",
                        (filename,)
                    )
                
                result = cursor.fetchone()
                return result[0] if result else None
        
        return self._execute_with_retry(_check_exists)
    
    def store_document(self, 
                      filename: str,
                      file_type: str,
                      file_size: int,
                      raw_text: str,
                      page_count: Optional[int] = None,
                      extraction_method: Optional[str] = None,
                      file_content: Optional[bytes] = None) -> int:
        """Store a document in the database with optional file hash."""
        content_hash = self._calculate_content_hash(raw_text)
        file_hash = self._calculate_file_hash(file_content) if file_content else None
        
        def _store_doc():
            with self._get_connection() as conn:
                try:
                    cursor = conn.execute("""
                        INSERT INTO documents (
                            filename, file_type, file_size, page_count, 
                            content_hash, file_hash, extraction_method, raw_text
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (filename, file_type, file_size, page_count, 
                          content_hash, file_hash, extraction_method, raw_text))
                    
                    document_id = cursor.lastrowid
                    conn.commit()
                    logger.info(f"Stored document '{filename}' with ID {document_id}")
                    
                    # Record processing stats
                    self._record_processing_stage(document_id, 'extraction', success=True)
                    
                    return document_id
                    
                except sqlite3.IntegrityError as e:
                    logger.warning(f"Document '{filename}' already exists: {e}")
                    # Return existing document ID
                    return self.document_exists(filename, raw_text)
        
        return self._execute_with_retry(_store_doc)
    
    def store_chunks(self, document_id: int, chunks: List[Dict[str, Any]]) -> List[int]:
        """Store text chunks for a document."""
        def _store_chunks():
            chunk_ids = []
            with self._get_connection() as conn:
                try:
                    self._record_processing_stage(document_id, 'chunking', started=True)
                    
                    for i, chunk_data in enumerate(chunks):
                        chunk_text = chunk_data.get('text', '')
                        metadata = chunk_data.get('metadata', {})
                        
                        cursor = conn.execute("""
                            INSERT INTO chunks (
                                document_id, chunk_index, chunk_text, chunk_size,
                                page_number, element_type, coordinates, 
                                context_before, context_after
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            document_id,
                            metadata.get('chunk_index', i),
                            chunk_text,
                            len(chunk_text),
                            metadata.get('page_number'),
                            metadata.get('element_type', 'text'),
                            json.dumps(metadata.get('coordinates')) if metadata.get('coordinates') else None,
                            metadata.get('context_before'),
                            metadata.get('context_after')
                        ))
                        
                        chunk_ids.append(cursor.lastrowid)
                    
                    conn.commit()
                    self._record_processing_stage(document_id, 'chunking', success=True,
                                                metadata={'chunks_created': len(chunk_ids)})
                    
                    logger.info(f"Stored {len(chunk_ids)} chunks for document ID {document_id}")
                    return chunk_ids
                    
                except Exception as e:
                    conn.rollback()
                    self._record_processing_stage(document_id, 'chunking', success=False,
                                                error_message=str(e))
                    logger.error(f"Failed to store chunks for document {document_id}: {e}")
                    raise
        
        return self._execute_with_retry(_store_chunks)
    
    def get_unvectorized_chunks(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get chunks that haven't been vectorized yet."""
        def _get_chunks():
            with self._get_connection() as conn:
                conn.row_factory = sqlite3.Row
                
                query = """
                    SELECT c.*, d.filename, d.file_type
                    FROM chunks c
                    JOIN documents d ON c.document_id = d.id
                    WHERE c.vectorized = FALSE
                    ORDER BY c.document_id, c.chunk_index
                """
                
                if limit:
                    query += f" LIMIT {limit}"
                
                cursor = conn.execute(query)
                chunks = []
                
                for row in cursor.fetchall():
                    chunk_data = dict(row)
                    # Parse JSON fields
                    if chunk_data['coordinates']:
                        chunk_data['coordinates'] = json.loads(chunk_data['coordinates'])
                    chunks.append(chunk_data)
                
                return chunks
        
        return self._execute_with_retry(_get_chunks)
    
    def mark_chunk_vectorized(self, chunk_id: int, vector_id: str):
        """Mark a chunk as vectorized and store its vector database ID."""
        def _mark_vectorized():
            with self._get_connection() as conn:
                conn.execute("""
                    UPDATE chunks 
                    SET vectorized = TRUE, vector_id = ?
                    WHERE id = ?
                """, (vector_id, chunk_id))
                conn.commit()
                logger.debug(f"Marked chunk {chunk_id} as vectorized with vector ID {vector_id}")
        
        self._execute_with_retry(_mark_vectorized)
    
    def get_document_chunks(self, document_id: int) -> List[Dict[str, Any]]:
        """Get all chunks for a specific document."""
        def _get_doc_chunks():
            with self._get_connection() as conn:
                conn.row_factory = sqlite3.Row
                
                cursor = conn.execute("""
                    SELECT * FROM chunks
                    WHERE document_id = ?
                    ORDER BY chunk_index
                """, (document_id,))
                
                chunks = []
                for row in cursor.fetchall():
                    chunk_data = dict(row)
                    if chunk_data['coordinates']:
                        chunk_data['coordinates'] = json.loads(chunk_data['coordinates'])
                    chunks.append(chunk_data)
                
                return chunks
        
        return self._execute_with_retry(_get_doc_chunks)
    
    def get_documents_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all documents in the database."""
        def _get_summary():
            with self._get_connection() as conn:
                conn.row_factory = sqlite3.Row
                
                cursor = conn.execute("""
                    SELECT d.*, 
                           COUNT(c.id) as chunk_count,
                           SUM(CASE WHEN c.vectorized THEN 1 ELSE 0 END) as vectorized_chunks
                    FROM documents d
                    LEFT JOIN chunks c ON d.id = c.document_id
                    GROUP BY d.id
                    ORDER BY d.created_at DESC
                """)
                
                return [dict(row) for row in cursor.fetchall()]
        
        return self._execute_with_retry(_get_summary)
    
    def get_document_by_id(self, document_id: int) -> Optional[Dict[str, Any]]:
        """Get document information by ID."""
        def _get_document():
            with self._get_connection() as conn:
                conn.row_factory = sqlite3.Row
                
                cursor = conn.execute("""
                    SELECT * FROM documents WHERE id = ?
                """, (document_id,))
                
                result = cursor.fetchone()
                return dict(result) if result else None
        
        return self._execute_with_retry(_get_document)
    
    def delete_document_with_stats(self, document_id: int) -> Dict[str, Any]:
        """Delete a document and return detailed deletion statistics."""
        def _delete_with_stats():
            with self._get_connection() as conn:
                try:
                    # Get document info before deletion
                    conn.row_factory = sqlite3.Row
                    doc_cursor = conn.execute("SELECT filename FROM documents WHERE id = ?", (document_id,))
                    doc_result = doc_cursor.fetchone()
                    filename = doc_result['filename'] if doc_result else None
                    
                    if not filename:
                        return {
                            'success': False,
                            'filename': None,
                            'chunks_deleted': 0,
                            'processing_stats_deleted': 0,
                            'error': f'Document with ID {document_id} not found'
                        }
                    
                    # Count chunks before deletion
                    chunks_count = conn.execute(
                        "SELECT COUNT(*) FROM chunks WHERE document_id = ?", 
                        (document_id,)
                    ).fetchone()[0]
                    
                    # Count processing stats before deletion
                    stats_count = conn.execute(
                        "SELECT COUNT(*) FROM processing_stats WHERE document_id = ?", 
                        (document_id,)
                    ).fetchone()[0]
                    
                    # Delete document (cascades to chunks and processing_stats)
                    cursor = conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
                    conn.commit()
                    
                    if cursor.rowcount > 0:
                        logger.info(f"Deleted document with ID {document_id} ('{filename}') and {chunks_count} chunks")
                        return {
                            'success': True,
                            'filename': filename,
                            'chunks_deleted': chunks_count,
                            'processing_stats_deleted': stats_count,
                            'error': None
                        }
                    else:
                        return {
                            'success': False,
                            'filename': filename,
                            'chunks_deleted': 0,
                            'processing_stats_deleted': 0,
                            'error': f'Document with ID {document_id} not found'
                        }
                        
                except Exception as e:
                    conn.rollback()
                    logger.error(f"Failed to delete document {document_id}: {e}")
                    return {
                        'success': False,
                        'filename': None,
                        'chunks_deleted': 0,
                        'processing_stats_deleted': 0,
                        'error': str(e)
                    }
        
        return self._execute_with_retry(_delete_with_stats)
    
    def delete_document(self, document_id: int) -> bool:
        """Delete a document and all its chunks (backward compatibility)."""
        result = self.delete_document_with_stats(document_id)
        return result['success']
    
    def get_processing_stats(self, document_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get processing statistics for documents."""
        def _get_stats():
            with self._get_connection() as conn:
                conn.row_factory = sqlite3.Row
                
                if document_id:
                    cursor = conn.execute("""
                        SELECT * FROM processing_stats
                        WHERE document_id = ?
                        ORDER BY started_at DESC
                    """, (document_id,))
                else:
                    cursor = conn.execute("""
                        SELECT ps.*, d.filename
                        FROM processing_stats ps
                        JOIN documents d ON ps.document_id = d.id
                        ORDER BY ps.started_at DESC
                    """)
                
                stats = []
                for row in cursor.fetchall():
                    stat_data = dict(row)
                    if stat_data.get('metadata'):
                        stat_data['metadata'] = json.loads(stat_data['metadata'])
                    stats.append(stat_data)
                
                return stats
        
        return self._execute_with_retry(_get_stats)
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get overall database statistics."""
        def _get_db_stats():
            with self._get_connection() as conn:
                # Get document count
                doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
                
                # Get chunk statistics
                chunk_stats = conn.execute("""
                    SELECT 
                        COUNT(*) as total_chunks,
                        SUM(CASE WHEN vectorized THEN 1 ELSE 0 END) as vectorized_chunks,
                        AVG(chunk_size) as avg_chunk_size
                    FROM chunks
                """).fetchone()
                
                # Get file type distribution
                file_types = conn.execute("""
                    SELECT file_type, COUNT(*) as count
                    FROM documents
                    GROUP BY file_type
                    ORDER BY count DESC
                """).fetchall()
                
                return {
                    'document_count': doc_count,
                    'total_chunks': chunk_stats[0] if chunk_stats[0] else 0,
                    'vectorized_chunks': chunk_stats[1] if chunk_stats[1] else 0,
                    'pending_vectorization': (chunk_stats[0] - chunk_stats[1]) if chunk_stats[0] and chunk_stats[1] else 0,
                    'avg_chunk_size': round(chunk_stats[2], 2) if chunk_stats[2] else 0,
                    'file_type_distribution': [{'file_type': ft[0], 'count': ft[1]} for ft in file_types]
                }
        
        return self._execute_with_retry(_get_db_stats)
    
    def _record_processing_stage(self, document_id: int, stage: str, 
                               started: bool = False, success: bool = False,
                               error_message: str = None, metadata: Dict = None):
        """Record processing stage information."""
        def _record_stage():
            with self._get_connection() as conn:
                if started:
                    conn.execute("""
                        INSERT INTO processing_stats (document_id, processing_stage, started_at)
                        VALUES (?, ?, CURRENT_TIMESTAMP)
                    """, (document_id, stage))
                else:
                    conn.execute("""
                        UPDATE processing_stats 
                        SET completed_at = CURRENT_TIMESTAMP, success = ?, error_message = ?, metadata = ?
                        WHERE document_id = ? AND processing_stage = ? AND completed_at IS NULL
                    """, (success, error_message, 
                          json.dumps(metadata) if metadata else None,
                          document_id, stage))
                conn.commit()
        
        self._execute_with_retry(_record_stage)
    
    def cleanup_old_documents(self, days_old: int = 30) -> int:
        """Clean up documents older than specified days."""
        def _cleanup():
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    DELETE FROM documents 
                    WHERE created_at < datetime('now', '-{} days')
                """.format(days_old))
                conn.commit()
                
                deleted_count = cursor.rowcount
                logger.info(f"Cleaned up {deleted_count} documents older than {days_old} days")
                return deleted_count
        
        return self._execute_with_retry(_cleanup)
    
    def clear_all_documents(self) -> Dict[str, Any]:
        """Clear all documents, chunks, and processing stats from the SQLite database."""
        def _clear_all():
            with self._get_connection() as conn:
                try:
                    # Get counts before deletion
                    doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
                    chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
                    stats_count = conn.execute("SELECT COUNT(*) FROM processing_stats").fetchone()[0]
                    
                    # Delete all records (documents deletion will cascade to chunks and processing_stats)
                    conn.execute("DELETE FROM processing_stats")
                    conn.execute("DELETE FROM chunks")
                    conn.execute("DELETE FROM documents")
                    
                    # Reset AUTO_INCREMENT counters
                    conn.execute("DELETE FROM sqlite_sequence WHERE name IN ('documents', 'chunks', 'processing_stats')")
                    
                    conn.commit()
                    
                    logger.info(f"Cleared SQLite database: {doc_count} documents, {chunk_count} chunks, {stats_count} processing stats")
                    
                    return {
                        'success': True,
                        'documents_deleted': doc_count,
                        'chunks_deleted': chunk_count,
                        'processing_stats_deleted': stats_count,
                        'message': f'Successfully cleared {doc_count} documents and all associated data',
                        'error': None
                    }
                    
                except Exception as e:
                    conn.rollback()
                    logger.error(f"Failed to clear SQLite database: {e}")
                    return {
                        'success': False,
                        'documents_deleted': 0,
                        'chunks_deleted': 0,
                        'processing_stats_deleted': 0,
                        'message': None,
                        'error': str(e)
                    }
        
        return self._execute_with_retry(_clear_all) 