#!/usr/bin/env python3
"""
Pydantic models for API request and response handling.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from datetime import datetime

class DocumentInfo(BaseModel):
    """Information about a processed document."""
    filename: str = Field(..., description="Original filename")
    file_size: int = Field(..., description="File size in bytes")
    file_type: str = Field(..., description="File content type")
    file_extension: Optional[str] = Field(None, description="File extension")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    
class ChunkInfo(BaseModel):
    """Information about document chunks."""
    total_chunks: int = Field(..., description="Total number of chunks created")
    average_chunk_size: float = Field(..., description="Average chunk size in characters")
    min_chunk_size: int = Field(..., description="Minimum chunk size")
    max_chunk_size: int = Field(..., description="Maximum chunk size")
    
class EmbeddingInfo(BaseModel):
    """Information about embeddings."""
    embeddings_created: int = Field(..., description="Number of embeddings created")
    embeddings_stored: int = Field(..., description="Number of embeddings stored")
    embedding_model: str = Field(..., description="Embedding model used")
    embedding_dimension: int = Field(..., description="Embedding vector dimension")
    
class DocumentProcessResponse(BaseModel):
    """Response model for document processing."""
    success: bool = Field(..., description="Whether processing was successful")
    document_info: DocumentInfo = Field(..., description="Document information")
    chunk_info: Optional[ChunkInfo] = Field(None, description="Chunk information")
    embedding_info: Optional[EmbeddingInfo] = Field(None, description="Embedding information")
    vector_db_id: Optional[str] = Field(None, description="Vector database collection ID")
    processing_stats: Optional[Dict[str, Any]] = Field(None, description="Additional processing statistics")
    error: Optional[str] = Field(None, description="Error message if processing failed")
    timestamp: datetime = Field(default_factory=datetime.now, description="Processing timestamp")
    
class ProcessingStats(BaseModel):
    """Overall processing statistics."""
    total_documents_processed: int = Field(..., description="Total documents processed")
    total_chunks_created: int = Field(..., description="Total chunks created")
    total_embeddings_stored: int = Field(..., description="Total embeddings stored")
    vector_db_collection_count: int = Field(..., description="Current vector database document count")
    last_processed: Optional[datetime] = Field(None, description="Last processing timestamp")
    processing_errors: int = Field(0, description="Number of processing errors")
    supported_formats: List[str] = Field(..., description="Supported file formats")
    
class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    
class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Overall health status")
    services: Dict[str, Any] = Field(..., description="Individual service health status")
    timestamp: str = Field(..., description="Health check timestamp")
    
class BatchProcessRequest(BaseModel):
    """Request model for batch processing."""
    clear_existing: bool = Field(False, description="Whether to clear existing documents before processing")
    processing_options: Optional[Dict[str, Any]] = Field(None, description="Additional processing options")
    
class BatchProcessResponse(BaseModel):
    """Response model for batch processing."""
    total_files: int = Field(..., description="Total files in batch")
    successful_files: int = Field(..., description="Successfully processed files")
    failed_files: int = Field(..., description="Failed files")
    processing_time: float = Field(..., description="Total processing time")
    results: List[DocumentProcessResponse] = Field(..., description="Individual file results")
    errors: List[str] = Field(default_factory=list, description="Processing errors")

class QueryRequest(BaseModel):
    """Request model for document querying."""
    query: str = Field(..., description="The question to ask about the documents", min_length=1)
    max_chunks: Optional[int] = Field(5, description="Maximum number of document chunks to retrieve", ge=1, le=20)
    temperature: Optional[float] = Field(0.6, description="AI temperature for response generation", ge=0.0, le=2.0)
    
class QuerySource(BaseModel):
    """Information about a source document used in the query response."""
    filename: str = Field(..., description="Source document filename")
    chunk_index: int = Field(..., description="Chunk index within the document")
    vector_id: str = Field(..., description="Unique vector ID in the database")
    content_preview: str = Field(..., description="Full content of the relevant chunk")
    similarity_score: float = Field(..., description="Similarity score for relevance")
    
class QueryRetrievalStats(BaseModel):
    """Statistics about the document retrieval process."""
    chunks_found: int = Field(..., description="Number of relevant chunks found")
    avg_similarity: float = Field(..., description="Average similarity score")
    generation_time: float = Field(..., description="Time taken for AI response generation")
    query_type: str = Field(..., description="Type of query processed")
    
class QueryResponse(BaseModel):
    """Response model for document queries."""
    success: bool = Field(..., description="Whether the query was successful")
    answer: str = Field(..., description="AI-generated answer to the query")
    sources: List[QuerySource] = Field(..., description="Source documents used for the answer")
    query_time: float = Field(..., description="Total time taken to process the query")
    confidence_score: float = Field(..., description="Confidence score for the answer")
    retrieval_stats: QueryRetrievalStats = Field(..., description="Detailed retrieval statistics")
    error: Optional[str] = Field(None, description="Error message if query failed")
    timestamp: datetime = Field(default_factory=datetime.now, description="Query timestamp")

class StoredDocument(BaseModel):
    """Information about a document stored in the vector database."""
    filename: str = Field(..., description="Original filename of the document")
    file_extension: str = Field(..., description="File extension/type")
    file_size_mb: float = Field(..., description="File size in megabytes")
    total_chunks: int = Field(..., description="Number of chunks created from this document")
    processing_timestamp: datetime = Field(..., description="When the document was processed")
    chunk_ids: List[str] = Field(..., description="List of chunk IDs for this document")
    avg_chunk_size: float = Field(..., description="Average chunk size in characters")
    total_content_length: int = Field(..., description="Total content length across all chunks")

class DocumentListResponse(BaseModel):
    """Response model for listing documents in vector database."""
    success: bool = Field(..., description="Whether the request was successful")
    total_documents: int = Field(..., description="Total number of unique documents")
    total_chunks: int = Field(..., description="Total number of chunks across all documents")
    documents: List[StoredDocument] = Field(..., description="List of stored documents")
    collection_name: str = Field(..., description="Vector database collection name")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    error: Optional[str] = Field(None, description="Error message if request failed")

class DeleteDocumentRequest(BaseModel):
    """Request model for deleting a document by filename."""
    filename: str = Field(..., description="Filename of the document to delete", min_length=1)

class DeleteDocumentResponse(BaseModel):
    """Response model for document deletion."""
    success: bool = Field(..., description="Whether the deletion was successful")
    filename: str = Field(..., description="Filename that was targeted for deletion")
    chunks_deleted: int = Field(..., description="Number of chunks deleted")
    collection_name: str = Field(..., description="Vector database collection name")
    timestamp: datetime = Field(default_factory=datetime.now, description="Deletion timestamp")
    error: Optional[str] = Field(None, description="Error message if deletion failed")

class DeleteSQLiteDocumentRequest(BaseModel):
    """Request model for deleting a document from SQLite by document ID."""
    document_id: int = Field(..., description="Document ID to delete from SQLite database", ge=1)

class DeleteSQLiteDocumentResponse(BaseModel):
    """Response model for SQLite document deletion."""
    success: bool = Field(..., description="Whether the deletion was successful")
    document_id: int = Field(..., description="Document ID that was deleted")
    filename: Optional[str] = Field(None, description="Filename of the deleted document")
    chunks_deleted: int = Field(..., description="Number of chunks deleted from SQLite")
    processing_stats_deleted: int = Field(..., description="Number of processing stats records deleted")
    timestamp: datetime = Field(default_factory=datetime.now, description="Deletion timestamp")
    error: Optional[str] = Field(None, description="Error message if deletion failed")

class ChunkMetadata(BaseModel):
    """Metadata stored with each chunk in the vector database."""
    class Config:
        extra = "allow"  # Allow additional fields like date_metadata
    
    filename: str = Field(..., description="Original filename of the document")
    chunk_index: int = Field(..., description="Index of the chunk within the document")
    chunk_id: str = Field(..., description="SQLite chunk ID")
    document_id: str = Field(..., description="SQLite document ID")
    content_preview: str = Field(..., description="Preview of the chunk content (first 100 chars)")
    chunk_size: int = Field(..., description="Size of the chunk in characters")
    source: str = Field(..., description="Source filename (for compatibility)")
    processing_timestamp: float = Field(..., description="Unix timestamp when processed")
    file_size_mb: float = Field(..., description="File size in megabytes")
    # Additional fields that might be present in vectorize-pending
    content: Optional[str] = Field(None, description="Extended content preview (first 500 chars)")
    # Note: date_metadata and other custom fields are now allowed via Config.extra = "allow"

class VectorChunk(BaseModel):
    """Complete information about a chunk stored in vector database."""
    vector_id: str = Field(..., description="Unique vector ID in the database")
    metadata: ChunkMetadata = Field(..., description="Chunk metadata")
    vector_dimension: int = Field(..., description="Dimension of the embedding vector")
    has_vector: bool = Field(..., description="Whether the vector data is available")

class ChunkMetadataRequest(BaseModel):
    """Request model for getting chunk metadata."""
    vector_id: Optional[str] = Field(None, description="Specific vector ID to inspect")
    filename: Optional[str] = Field(None, description="Filter by filename")
    limit: Optional[int] = Field(10, description="Maximum number of chunks to return", ge=1, le=100)
    include_content: Optional[bool] = Field(False, description="Include full content preview")

class ChunkMetadataResponse(BaseModel):
    """Response model for chunk metadata inspection."""
    success: bool = Field(..., description="Whether the request was successful")
    total_chunks: int = Field(..., description="Total number of chunks found")
    chunks: List[VectorChunk] = Field(..., description="List of chunk information")
    collection_name: str = Field(..., description="Vector database collection name")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    error: Optional[str] = Field(None, description="Error message if request failed")

class ClearSQLiteResponse(BaseModel):
    """Response model for clearing all documents from SQLite database."""
    success: bool = Field(..., description="Whether the operation was successful")
    documents_deleted: int = Field(..., description="Number of documents deleted")
    chunks_deleted: int = Field(..., description="Number of chunks deleted")
    processing_stats_deleted: int = Field(..., description="Number of processing stats records deleted")
    message: Optional[str] = Field(None, description="Success message")
    timestamp: datetime = Field(default_factory=datetime.now, description="Operation timestamp")
    error: Optional[str] = Field(None, description="Error message if operation failed") 