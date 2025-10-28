#!/usr/bin/env python3
"""
FastAPI application for document processing API.
Handles individual document uploads and processing.
"""

import os
import sys
import tempfile
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import io
import csv

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from core.config import config
from services.document_service import DocumentService
from services.query_service import QueryService
from models.api_models import (
    DocumentProcessResponse,
    DocumentInfo,
    ProcessingStats,
    ErrorResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    DocumentListResponse,
    DeleteDocumentRequest,
    DeleteDocumentResponse,
    DeleteSQLiteDocumentRequest,
    DeleteSQLiteDocumentResponse,
    ChunkMetadataRequest,
    ChunkMetadataResponse,
    ClearSQLiteResponse
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Harfang Document Processing API",
    description="API for processing aerospace MRO documents with RAG capabilities",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global service instances
document_service: Optional[DocumentService] = None
query_service: Optional[QueryService] = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global document_service, query_service
    try:
        logger.info("Initializing Document Processing API...")
        
        # Validate configuration
        if not config.validate():
            raise Exception("Configuration validation failed")
        
        # Initialize document service
        document_service = DocumentService()
        await document_service.initialize()
        
        # Initialize query service
        query_service = QueryService()
        await query_service.initialize()
        
        logger.info("Document Processing API initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize API: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global document_service, query_service
    if document_service:
        await document_service.cleanup()
    if query_service:
        await query_service.cleanup()
    logger.info("Document Processing API shutdown complete")

def get_document_service() -> DocumentService:
    """Dependency to get document service instance."""
    if document_service is None:
        raise HTTPException(status_code=503, detail="Document service not initialized")
    return document_service

def get_query_service() -> QueryService:
    """Dependency to get query service instance."""
    if query_service is None:
        raise HTTPException(status_code=503, detail="Query service not initialized")
    return query_service

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Check document service
        doc_service = get_document_service()
        doc_health = await doc_service.health_check()
        
        # Check query service
        q_service = get_query_service()
        query_health = await q_service.health_check()
        
        # Combine health status
        health_status = {
            **doc_health,
            **query_health
        }
        
        # Determine overall status
        overall_status = "healthy"
        for service, status in health_status.items():
            if isinstance(status, str) and ("error" in status.lower() or "unhealthy" in status.lower()):
                overall_status = "unhealthy"
                break
        
        return HealthResponse(
            status=overall_status,
            services=health_status,
            timestamp=health_status.get("timestamp", "")
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            services={"error": str(e)},
            timestamp=""
        )

@app.post("/documents/upload", response_model=DocumentProcessResponse)
async def upload_document(
    file: UploadFile = File(...),
    service: DocumentService = Depends(get_document_service)
):
    """
    Upload and process a single document.
    
    Supports: PDF, DOCX, XLSX, TXT, CSV, PPTX
    """
    try:
        logger.info(f"Received document upload: {file.filename}")
        
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Check file size
        if file.size and file.size > config.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413, 
                detail=f"File too large. Max size: {config.MAX_FILE_SIZE / (1024*1024):.1f} MB"
            )
        
        # Read file content
        file_content = await file.read()
        
        # Process document
        result = await service.process_document(
            filename=file.filename,
            content=file_content,
            content_type=file.content_type
        )
        
        logger.info(f"Document processed successfully: {file.filename}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing document {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")

@app.post("/documents/batch-upload", response_model=List[DocumentProcessResponse])
async def batch_upload_documents(
    files: List[UploadFile] = File(...),
    service: DocumentService = Depends(get_document_service)
):
    """
    Upload and process multiple documents in batch.
    """
    try:
        logger.info(f"Received batch upload: {len(files)} files")
        
        if len(files) > config.MAX_BATCH_SIZE:
            raise HTTPException(
                status_code=400, 
                detail=f"Too many files. Max batch size: {config.MAX_BATCH_SIZE}"
            )
        
        results = []
        for file in files:
            try:
                # Validate file
                if not file.filename:
                    results.append(DocumentProcessResponse(
                        success=False,
                        document_info=DocumentInfo(filename="unknown", file_size=0, file_type="unknown"),
                        error="No filename provided"
                    ))
                    continue
                
                # Read file content
                file_content = await file.read()
                
                # Process document
                result = await service.process_document(
                    filename=file.filename,
                    content=file_content,
                    content_type=file.content_type
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {e}")
                results.append(DocumentProcessResponse(
                    success=False,
                    document_info=DocumentInfo(
                        filename=file.filename or "unknown",
                        file_size=file.size or 0,
                        file_type=file.content_type or "unknown"
                    ),
                    error=str(e)
                ))
        
        successful = sum(1 for r in results if r.success)
        logger.info(f"Batch processing complete: {successful}/{len(results)} files successful")
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch upload: {e}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

@app.get("/documents/stats", response_model=ProcessingStats)
async def get_processing_stats(
    service: DocumentService = Depends(get_document_service)
):
    """Get processing statistics."""
    try:
        stats = await service.get_processing_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@app.get("/documents/spell-correction-stats")
async def get_spell_correction_stats(
    service: DocumentService = Depends(get_document_service)
):
    """Get spell correction statistics."""
    try:
        if service.spell_correction_service:
            stats = service.spell_correction_service.get_correction_stats()
            return {"success": True, "spell_correction_stats": stats}
        else:
            return {"success": False, "error": "Spell correction service not initialized"}
    except Exception as e:
        logger.error(f"Error getting spell correction stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get spell correction stats: {str(e)}")

@app.delete("/documents/clear")
async def clear_vector_database(
    service: DocumentService = Depends(get_document_service)
):
    """Clear all documents from vector database."""
    try:
        result = await service.clear_database()
        return {"success": True, "message": "Vector database cleared successfully", "cleared_count": result}
    except Exception as e:
        logger.error(f"Error clearing database: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear database: {str(e)}")

@app.delete("/documents/sqlite/clear", response_model=ClearSQLiteResponse)
async def clear_sqlite_database(
    service: DocumentService = Depends(get_document_service)
):
    """
    Clear all documents from SQLite database.
    
    This will permanently delete ALL documents, chunks, and processing statistics
    from the SQLite database. This operation cannot be undone.
    
    WARNING: This will clear all document metadata and processing history.
    Vector database entries are not affected by this operation.
    """
    try:
        logger.info("Received request to clear all documents from SQLite database")
        
        # Clear all documents from SQLite
        result = service.database_service.clear_all_documents()
        
        if result['success']:
            logger.info(f"SQLite database cleared successfully: {result['documents_deleted']} documents, {result['chunks_deleted']} chunks")
        else:
            logger.error(f"Failed to clear SQLite database: {result['error']}")
        
        return ClearSQLiteResponse(
            success=result['success'],
            documents_deleted=result['documents_deleted'],
            chunks_deleted=result['chunks_deleted'],
            processing_stats_deleted=result['processing_stats_deleted'],
            message=result['message'],
            error=result['error']
        )
        
    except Exception as e:
        logger.error(f"Error clearing SQLite database: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear SQLite database: {str(e)}")

@app.get("/documents/sqlite-stats")
async def get_sqlite_stats(
    service: DocumentService = Depends(get_document_service)
):
    """Get SQLite database statistics including vectorization status."""
    try:
        stats = service.database_service.get_database_stats()
        return {"success": True, "sqlite_stats": stats}
    except Exception as e:
        logger.error(f"Error getting SQLite stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get SQLite stats: {str(e)}")

@app.get("/documents/unvectorized")
async def get_unvectorized_chunks(
    limit: Optional[int] = None,
    service: DocumentService = Depends(get_document_service)
):
    """Get chunks that are stored in SQLite but not yet vectorized."""
    try:
        chunks = service.database_service.get_unvectorized_chunks(limit)
        return {
            "success": True,
            "unvectorized_chunks": len(chunks),
            "chunks": chunks[:10] if chunks else [],  # Return first 10 for preview
            "total_available": len(chunks)
        }
    except Exception as e:
        logger.error(f"Error getting unvectorized chunks: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get unvectorized chunks: {str(e)}")

@app.post("/documents/vectorize-pending")
async def vectorize_pending_chunks(
    batch_size: Optional[int] = 50,
    service: DocumentService = Depends(get_document_service)
):
    """Vectorize chunks that are stored in SQLite but not yet vectorized."""
    try:
        # Get unvectorized chunks
        chunks = service.database_service.get_unvectorized_chunks(limit=batch_size)
        
        if not chunks:
            return {
                "success": True,
                "message": "No pending chunks to vectorize",
                "vectorized_count": 0
            }
        
        vectorized_count = 0
        
        # Process chunks in batches
        for chunk in chunks:
            try:
                # Create embedding
                text = chunk['chunk_text']
                embedding = await service.embedding_service.create_embedding(text)
                
                if embedding:
                    # Store in vector database
                    metadata = {
                        "filename": chunk['filename'],
                        "chunk_index": chunk['chunk_index'],
                        "chunk_id": str(chunk['id']),
                        "document_id": str(chunk['document_id']),
                        "content": text[:500] + "..." if len(text) > 500 else text,
                        "content_preview": text[:100] + "..." if len(text) > 100 else text,
                        "chunk_size": chunk['chunk_size'],
                        "source": chunk['filename'],
                        "processing_timestamp": time.time(),
                        "file_size_mb": chunk['chunk_size'] / (1024 * 1024)
                    }
                    
                    # Generate unique vector ID (Qdrant requires integer or UUID)
                    import hashlib
                    id_string = f"{chunk['document_id']}_{chunk['id']}"
                    vector_id = int(hashlib.md5(id_string.encode()).hexdigest()[:8], 16)
                    
                    from core.vector_db import VectorPoint
                    vector_point = VectorPoint(
                        id=vector_id,
                        vector=embedding,
                        metadata=metadata
                    )
                    
                    # Store vector
                    success = service.vector_db.upsert_points([vector_point])
                    
                    if success:
                        # Mark as vectorized in SQLite
                        service.database_service.mark_chunk_vectorized(chunk['id'], vector_id)
                        vectorized_count += 1
                        
            except Exception as e:
                logger.warning(f"Failed to vectorize chunk {chunk['id']}: {e}")
                continue
        
        return {
            "success": True,
            "message": f"Successfully vectorized {vectorized_count} chunks",
            "vectorized_count": vectorized_count,
            "total_processed": len(chunks)
        }
        
    except Exception as e:
        logger.error(f"Error vectorizing pending chunks: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to vectorize pending chunks: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    service: QueryService = Depends(get_query_service)
):
    """
    Process a query against the document database using RAG.
    
    This endpoint allows you to ask questions about your processed documents.
    The system will search through the vector database to find relevant information
    and generate a comprehensive answer using AI.
    """
    try:
        logger.info(f"Received query: {request.query[:100]}...")
        
        # Process the query
        result = await service.process_query(request)
        
        logger.info(f"Query processed successfully: {result.success}")
        return result
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.post("/questions/bulk-process-csv")
async def bulk_process_questions_from_csv(
    file: UploadFile = File(...),
    service: QueryService = Depends(get_query_service)
):
    """
    Upload a CSV with a 'Question' column, get answers, and return the updated CSV.
    Column names are case-insensitive.
    """
    if not file.filename or not file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")

    try:
        content = await file.read()
        
        try:
            # Try reading with UTF-8 first, as it's the standard
            df = pd.read_csv(io.BytesIO(content), encoding='utf-8')
        except UnicodeDecodeError:
            # If UTF-8 fails, it might be an encoding like latin-1 or cp1252.
            # latin-1 is a safe fallback that won't raise decoding errors.
            logger.warning("Could not decode CSV with UTF-8. Falling back to 'latin-1' encoding.")
            df = pd.read_csv(io.BytesIO(content), encoding='latin-1')

        # Normalize column names to lowercase for case-insensitivity
        df.columns = [col.lower() for col in df.columns]

        if "question" not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must have a 'Question' column (case-insensitive).")

        if "answer" not in df.columns:
            df["answer"] = ""
        if "context" not in df.columns:
            df["context"] = ""
        
        df['answer'] = df['answer'].astype(object)
        df['context'] = df['context'].astype(object)

        for index, row in df.iterrows():
            question = row.get("question")
            if pd.notna(question) and isinstance(question, str) and question.strip():
                logger.info(f"Processing question from CSV row {index + 2}: '{question[:100]}...'")
                
                query_request = QueryRequest(query=question)
                query_result = await service.process_query(query_request)

                if query_result.success:
                    df.at[index, "answer"] = query_result.answer
                    
                    if query_result.sources:
                        contexts = []
                        for source in query_result.sources:
                            # Handle QuerySource objects with score information for consistency
                            score = getattr(source, 'similarity_score', 0.0)
                            content_preview = getattr(source, 'content_preview', '')
                            chunk_info = f"[Score: {score:.3f}] {content_preview}"
                            contexts.append(chunk_info)
                        df.at[index, "context"] = " <CHUNK_SPLIT> ".join(contexts)
                    else:
                        df.at[index, "context"] = ""
                else:
                    logger.error(f"Failed to get answer for question in row {index + 2}: {query_result.error}")
                    df.at[index, "answer"] = f"Error: {query_result.error or 'Unknown error'}"
                    df.at[index, "context"] = ""

        output_stream = io.StringIO()
        df.to_csv(output_stream, index=False, quoting=csv.QUOTE_ALL)
        output_stream.seek(0)

        return StreamingResponse(
            iter([output_stream.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=processed_{file.filename}"}
        )

    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="The uploaded CSV file is empty.")
    except Exception as e:
        logger.error(f"An error occurred during CSV processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.post("/questions/hybrid-bulk-process-csv")
async def hybrid_bulk_process_questions_from_csv(
    file: UploadFile = File(...),
    method: str = "hybrid_rrf",
    max_chunks: int = 5,  # Reduced from 10 to 5 to avoid overwhelming LLM
    enable_comparison: bool = False,
    fusion_weights_dense: float = 0.6,
    fusion_weights_sparse: float = 0.4,
    force_document_search: bool = True,
    service: QueryService = Depends(get_query_service)
):
    """
    Upload a CSV with a 'Question' column, process questions using hybrid retrieval methods,
    and return the updated CSV with answers and context.
    
    This endpoint combines CSV bulk processing with the hybrid retrieval system,
    allowing you to specify which retrieval method to use for all questions.
    
    Parameters:
    - file: CSV file with 'Question' column (case-insensitive)
    - method: Retrieval method (hybrid_rrf|hybrid_weighted|hybrid_borda|dense_only|sparse_only)
    - max_chunks: Maximum number of chunks to retrieve per question (default: 5)
    - enable_comparison: Whether to enable method comparison for each question (default: false)
    - fusion_weights_dense: Weight for dense retrieval (default: 0.6)
    - fusion_weights_sparse: Weight for sparse retrieval (default: 0.4)
    - force_document_search: Force all queries to use document search instead of conversational mode (default: true)
    
    Returns:
    - Processed CSV file with 'Answer' and 'Context' columns populated
    """
    if not file.filename or not file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")

    try:
        # Import hybrid engine here to avoid circular imports
        from core.hybrid_query_engine import HybridAerospaceQueryEngine, RetrievalMethod
        
        # Initialize hybrid query engine if not already done
        if not hasattr(service, '_hybrid_engine'):
            service._hybrid_engine = HybridAerospaceQueryEngine()
            logger.info("Hybrid query engine initialized for CSV processing")
        
        # Validate and convert method
        try:
            retrieval_method = RetrievalMethod(method)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid method: {method}. Must be one of: hybrid_rrf, hybrid_weighted, hybrid_borda, dense_only, sparse_only")
        
        # Prepare fusion weights
        fusion_weights = {
            "dense": fusion_weights_dense,
            "sparse": fusion_weights_sparse
        } if method.startswith("hybrid") else None
        
        logger.info(f"Starting hybrid CSV processing with method: {method}, max_chunks: {max_chunks}")
        
        # Read and process CSV file
        content = await file.read()
        
        try:
            # Try reading with UTF-8 first, as it's the standard
            df = pd.read_csv(io.BytesIO(content), encoding='utf-8')
        except UnicodeDecodeError:
            # If UTF-8 fails, it might be an encoding like latin-1 or cp1252.
            # latin-1 is a safe fallback that won't raise decoding errors.
            logger.warning("Could not decode CSV with UTF-8. Falling back to 'latin-1' encoding.")
            df = pd.read_csv(io.BytesIO(content), encoding='latin-1')

        # Normalize column names to lowercase for case-insensitivity
        df.columns = [col.lower() for col in df.columns]

        if "question" not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must have a 'Question' column (case-insensitive).")

        # Add missing columns
        if "answer" not in df.columns:
            df["answer"] = ""
        if "context" not in df.columns:
            df["context"] = ""
        if "retrieval_method" not in df.columns:
            df["retrieval_method"] = ""
        if "query_time" not in df.columns:
            df["query_time"] = ""
        if "confidence_score" not in df.columns:
            df["confidence_score"] = ""
        
        # Ensure proper data types
        df['answer'] = df['answer'].astype(object)
        df['context'] = df['context'].astype(object)
        df['retrieval_method'] = df['retrieval_method'].astype(object)
        df['query_time'] = df['query_time'].astype(object)
        df['confidence_score'] = df['confidence_score'].astype(object)

        # Process each question using hybrid retrieval
        processed_questions = 0
        failed_questions = 0
        
        for index, row in df.iterrows():
            question = row.get("question")
            if pd.notna(question) and isinstance(question, str) and question.strip():
                logger.info(f"Processing hybrid question from CSV row {index + 2}: '{question[:100]}...' using {method}")
                
                try:
                    # Process query using hybrid engine with force_document_search option
                    result = service._hybrid_engine.query(
                        query_text=question,
                        method=retrieval_method,
                        max_chunks=max_chunks,
                        enable_comparison=enable_comparison,
                        fusion_weights=fusion_weights,
                        force_document_search=force_document_search
                    )

                    if result and result.answer:
                        df.at[index, "answer"] = result.answer
                        df.at[index, "retrieval_method"] = result.retrieval_method
                        df.at[index, "query_time"] = f"{result.query_time:.2f}s" if result.query_time else ""
                        df.at[index, "confidence_score"] = f"{result.confidence_score:.3f}" if result.confidence_score else ""
                        
                        if result.sources:
                            contexts = []
                            for source in result.sources:
                                # Handle different score formats from hybrid results
                                if isinstance(source, dict):
                                    score = source.get('hybrid_score', source.get('similarity_score', 0.0))
                                    content_preview = source.get('content_preview', '')
                                else:
                                    # Handle QuerySource objects
                                    score = getattr(source, 'similarity_score', 0.0)
                                    content_preview = getattr(source, 'content_preview', '')
                                
                                chunk_info = f"[Score: {score:.3f}] {content_preview}"
                                contexts.append(chunk_info)
                            df.at[index, "context"] = " <CHUNK_SPLIT> ".join(contexts)
                        else:
                            df.at[index, "context"] = ""
                        
                        processed_questions += 1
                        logger.info(f"Successfully processed question {index + 2} in {result.query_time:.2f}s")
                    else:
                        df.at[index, "answer"] = "Error: No answer generated"
                        df.at[index, "context"] = ""
                        df.at[index, "retrieval_method"] = method
                        failed_questions += 1
                        logger.error(f"No answer generated for question in row {index + 2}")

                except Exception as e:
                    logger.error(f"Failed to process question in row {index + 2}: {e}")
                    df.at[index, "answer"] = f"Error: {str(e)}"
                    df.at[index, "context"] = ""
                    df.at[index, "retrieval_method"] = method
                    failed_questions += 1

        # Generate output CSV
        output_stream = io.StringIO()
        df.to_csv(output_stream, index=False, quoting=csv.QUOTE_ALL)
        output_stream.seek(0)

        logger.info(f"Hybrid CSV processing complete: {processed_questions} successful, {failed_questions} failed questions")

        return StreamingResponse(
            iter([output_stream.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=hybrid_{method}_processed_{file.filename}"}
        )

    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="The uploaded CSV file is empty.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"An error occurred during hybrid CSV processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.get("/query/stats")
async def get_query_stats(
    service: QueryService = Depends(get_query_service)
):
    """Get statistics about the query engine and RAG system."""
    try:
        stats = await service.get_query_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting query stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get query stats: {str(e)}")

@app.get("/documents/list", response_model=DocumentListResponse)
async def list_stored_documents(
    service: DocumentService = Depends(get_document_service)
):
    """
    List all documents stored in the vector database.
    
    Returns a list of unique documents (not individual chunks) with their metadata,
    including total chunks per document, file sizes, processing timestamps, etc.
    """
    try:
        logger.info("Received request to list stored documents")
        
        result = await service.list_stored_documents()
        
        logger.info(f"Listed {result.total_documents} documents with {result.total_chunks} total chunks")
        return result
        
    except Exception as e:
        logger.error(f"Error listing stored documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")

@app.delete("/documents/delete", response_model=DeleteDocumentResponse)
async def delete_document_by_filename(
    request: DeleteDocumentRequest,
    service: DocumentService = Depends(get_document_service)
):
    """
    Delete all chunks associated with a specific filename.
    
    This will remove all vector embeddings and metadata for the specified document
    from the vector database. This action cannot be undone.
    
    Use this to clean up duplicate documents or remove unwanted files.
    """
    try:
        logger.info(f"Received request to delete document: {request.filename}")
        
        # Validate filename
        if not request.filename or not request.filename.strip():
            raise HTTPException(status_code=400, detail="Filename cannot be empty")
        
        result = await service.delete_document_by_filename(request.filename)
        
        if result.success:
            if result.chunks_deleted == 0:
                logger.info(f"No chunks found for document: {request.filename}")
            elif result.chunks_deleted == -1:
                logger.info(f"Document deleted successfully (Pinecone - count unknown): {request.filename}")
            else:
                logger.info(f"Successfully deleted {result.chunks_deleted} chunks for document: {request.filename}")
        else:
            logger.error(f"Failed to delete document: {request.filename} - {result.error}")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {request.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")

@app.delete("/documents/sqlite/delete", response_model=DeleteSQLiteDocumentResponse)
async def delete_sqlite_document_by_id(
    request: DeleteSQLiteDocumentRequest,
    service: DocumentService = Depends(get_document_service)
):
    """
    Delete a document from SQLite database by document ID.
    
    This will remove the document and all its associated chunks and processing stats
    from the SQLite database only. Vector database entries are not affected.
    
    Use this to clean up SQLite entries for documents that failed to vectorize
    or to remove specific documents by their database ID.
    """
    try:
        logger.info(f"Received request to delete SQLite document with ID: {request.document_id}")
        
        # Validate document ID
        if request.document_id <= 0:
            raise HTTPException(status_code=400, detail="Document ID must be a positive integer")
        
        # Delete document from SQLite
        result = service.database_service.delete_document_with_stats(request.document_id)
        
        if result['success']:
            logger.info(f"Successfully deleted SQLite document ID {request.document_id}: {result['filename']}")
            logger.info(f"Deleted {result['chunks_deleted']} chunks and {result['processing_stats_deleted']} processing stats")
        else:
            logger.error(f"Failed to delete SQLite document ID {request.document_id}: {result['error']}")
        
        return DeleteSQLiteDocumentResponse(
            success=result['success'],
            document_id=request.document_id,
            filename=result['filename'],
            chunks_deleted=result['chunks_deleted'],
            processing_stats_deleted=result['processing_stats_deleted'],
            error=result['error']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting SQLite document {request.document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete SQLite document: {str(e)}")

@app.post("/documents/chunks/metadata", response_model=ChunkMetadataResponse)
async def get_chunk_metadata(
    request: ChunkMetadataRequest,
    service: DocumentService = Depends(get_document_service)
):
    """
    Get metadata for chunks stored in the vector database.
    
    This endpoint allows you to inspect the metadata stored with each chunk
    in the vector database, including content previews, processing timestamps,
    file information, and vector details.
    
    You can filter by:
    - Specific vector ID
    - Filename (partial match)
    - Limit number of results
    - Include extended content preview
    """
    try:
        logger.info(f"Getting chunk metadata - filters: vector_id={request.vector_id}, filename={request.filename}")
        
        # Get chunk metadata from service
        result = await service.get_chunk_metadata(
            vector_id=request.vector_id,
            filename=request.filename,
            limit=request.limit,
            include_content=request.include_content
        )
        
        # Convert to response model
        response = ChunkMetadataResponse(
            success=result['success'],
            total_chunks=result['total_chunks'],
            chunks=result['chunks'],
            collection_name=result['collection_name'],
            error=result['error']
        )
        
        logger.info(f"Retrieved metadata for {len(result['chunks'])} chunks")
        return response
        
    except Exception as e:
        logger.error(f"Error getting chunk metadata: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get chunk metadata: {str(e)}")

@app.get("/documents/chunks/metadata", response_model=ChunkMetadataResponse)
async def get_chunk_metadata_get(
    vector_id: Optional[str] = Query(None, description="Specific vector ID to inspect"),
    filename: Optional[str] = Query(None, description="Filter by filename (partial match)"),
    limit: int = Query(10, description="Maximum number of chunks to return", ge=1, le=100),
    include_content: bool = Query(False, description="Include extended content preview"),
    service: DocumentService = Depends(get_document_service)
):
    """
    Get metadata for chunks stored in the vector database (GET version).
    
    This is a GET version of the chunk metadata endpoint for easier testing
    and browser access. Same functionality as the POST version.
    """
    try:
        logger.info(f"Getting chunk metadata (GET) - filters: vector_id={vector_id}, filename={filename}")
        
        # Get chunk metadata from service
        result = await service.get_chunk_metadata(
            vector_id=vector_id,
            filename=filename,
            limit=limit,
            include_content=include_content
        )
        
        # Convert to response model
        response = ChunkMetadataResponse(
            success=result['success'],
            total_chunks=result['total_chunks'],
            chunks=result['chunks'],
            collection_name=result['collection_name'],
            error=result['error']
        )
        
        logger.info(f"Retrieved metadata for {len(result['chunks'])} chunks")
        return response
        
    except Exception as e:
        logger.error(f"Error getting chunk metadata: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get chunk metadata: {str(e)}")

# ===== HYBRID RETRIEVAL ENDPOINTS =====

@app.post("/query/hybrid")
async def hybrid_query(
    request: dict,
    service: QueryService = Depends(get_query_service)
):
    """
    Process query using hybrid retrieval methods.
    
    This endpoint uses the new hybrid retrieval system that combines
    dense (semantic) and sparse (keyword-based) search methods.
    
    Request body:
    {
        "query": "Your question here",
        "method": "hybrid_rrf|hybrid_weighted|hybrid_borda|dense_only|sparse_only",
        "max_chunks": 10,
        "enable_comparison": false,
        "fusion_weights": {"dense": 0.6, "sparse": 0.4}
    }
    """
    try:
        # Import hybrid engine here to avoid circular imports
        from core.hybrid_query_engine import HybridAerospaceQueryEngine, RetrievalMethod
        
        # Initialize hybrid query engine if not already done
        if not hasattr(service, '_hybrid_engine'):
            service._hybrid_engine = HybridAerospaceQueryEngine()
            logger.info("Hybrid query engine initialized")
        
        # Extract parameters
        query_text = request.get("query", "")
        method_str = request.get("method", "hybrid_rrf")
        max_chunks = request.get("max_chunks", 10)
        enable_comparison = request.get("enable_comparison", False)
        fusion_weights = request.get("fusion_weights", None)
        
        if not query_text:
            raise HTTPException(status_code=400, detail="Query text is required")
        
        # Convert method string to enum
        try:
            method = RetrievalMethod(method_str)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid method: {method_str}")
        
        logger.info(f"Processing hybrid query: '{query_text[:50]}...' using {method_str}")
        
        # Process query
        result = service._hybrid_engine.query(
            query_text=query_text,
            method=method,
            max_chunks=max_chunks,
            enable_comparison=enable_comparison,
            fusion_weights=fusion_weights,
            force_document_search=True  # Force document search by default
        )
        
        # Format response
        response = {
            "success": True,
            "answer": result.answer,
            "sources": result.sources,
            "query_time": result.query_time,
            "retrieval_stats": result.retrieval_stats,
            "confidence_score": result.confidence_score,
            "retrieval_method": result.retrieval_method,
            "method_comparison": result.method_comparison
        }
        
        logger.info(f"Hybrid query completed in {result.query_time:.2f}s using {result.retrieval_method}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Hybrid query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Hybrid query failed: {str(e)}")

@app.post("/query/compare")
async def compare_retrieval_methods(
    request: dict,
    service: QueryService = Depends(get_query_service)
):
    """
    Compare all retrieval methods for a given query.
    
    This endpoint runs the same query through multiple retrieval methods
    and provides detailed comparison metrics.
    
    Request body:
    {
        "query": "Your question here",
        "limit": 10
    }
    """
    try:
        from core.hybrid_query_engine import HybridAerospaceQueryEngine
        
        # Initialize hybrid engine if needed
        if not hasattr(service, '_hybrid_engine'):
            service._hybrid_engine = HybridAerospaceQueryEngine()
        
        query_text = request.get("query", "")
        limit = request.get("limit", 10)
        
        if not query_text:
            raise HTTPException(status_code=400, detail="Query text is required")
        
        logger.info(f"Comparing retrieval methods for query: '{query_text[:50]}...'")
        
        # Run comparison
        comparison = service._hybrid_engine.compare_methods(query_text, limit)
        
        response = {
            "success": True,
            "comparison": comparison,
            "summary": {
                "query": query_text,
                "methods_tested": len(comparison.get("methods", {})),
                "best_method": comparison.get("best_method", {}),
                "timestamp": comparison.get("timestamp", "")
            }
        }
        
        logger.info(f"Method comparison completed for query")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Method comparison failed: {e}")
        raise HTTPException(status_code=500, detail=f"Method comparison failed: {str(e)}")

@app.post("/query/evaluate")
async def evaluate_retrieval_performance(
    request: dict,
    service: QueryService = Depends(get_query_service)
):
    """
    Run comprehensive evaluation of retrieval methods.
    
    This endpoint evaluates all retrieval methods using a set of test queries
    and provides detailed performance analysis and recommendations.
    
    Request body:
    {
        "test_queries": ["query1", "query2", ...] // optional, uses defaults if not provided
    }
    """
    try:
        from core.hybrid_query_engine import HybridAerospaceQueryEngine
        
        # Initialize hybrid engine if needed
        if not hasattr(service, '_hybrid_engine'):
            service._hybrid_engine = HybridAerospaceQueryEngine()
        
        test_queries = request.get("test_queries", None)
        
        logger.info(f"Running comprehensive retrieval evaluation with {len(test_queries) if test_queries else 'default'} queries")
        
        # Run evaluation
        evaluation = service._hybrid_engine.run_evaluation(test_queries)
        
        response = {
            "success": True,
            "evaluation": evaluation,
            "summary": evaluation["summary"]
        }
        
        logger.info(f"Retrieval evaluation completed")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Retrieval evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Retrieval evaluation failed: {str(e)}")

@app.get("/query/hybrid/stats")
async def get_hybrid_retrieval_stats(
    service: QueryService = Depends(get_query_service)
):
    """
    Get statistics for hybrid retrieval engine.
    
    Returns comprehensive statistics about hybrid retrieval performance,
    method usage, and component health.
    """
    try:
        from core.hybrid_query_engine import HybridAerospaceQueryEngine
        
        # Initialize hybrid engine if needed
        if not hasattr(service, '_hybrid_engine'):
            service._hybrid_engine = HybridAerospaceQueryEngine()
        
        # Get stats from hybrid engine
        stats = service._hybrid_engine.get_stats()
        health = service._hybrid_engine.get_health_status()
        
        response = {
            "success": True,
            "stats": stats,
            "health": health,
            "component_info": {
                "hybrid_engine": "Available",
                "dense_retriever": "Available" if health.get("hybrid_engine", {}).get("dense_retriever", {}).get("initialized") else "Not Available",
                "sparse_retriever": "Available" if health.get("hybrid_engine", {}).get("sparse_retriever", {}).get("indexed") else "Not Available",
                "results_fusion": "Available" if health.get("hybrid_engine", {}).get("results_fusion", {}).get("initialized") else "Not Available"
            }
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to get hybrid retrieval stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get hybrid retrieval stats: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Harfang Document Processing API",
        "version": "2.0.0",
        "description": "API for processing aerospace MRO documents with advanced hybrid RAG capabilities",
        "endpoints": {
            "upload_document": "/documents/upload",
            "batch_upload": "/documents/batch-upload",
            "list_documents": "/documents/list",
            "delete_document": "/documents/delete",
            "delete_sqlite_document": "/documents/sqlite/delete",
            "query_documents": "/query",
            "bulk_process_csv": "/questions/bulk-process-csv",
            "hybrid_bulk_process_csv": "/questions/hybrid-bulk-process-csv",
            "hybrid_query": "/query/hybrid",
            "compare_methods": "/query/compare",
            "evaluate_performance": "/query/evaluate",
            "hybrid_stats": "/query/hybrid/stats",
            "document_stats": "/documents/stats",
            "spell_correction_stats": "/documents/spell-correction-stats",
            "query_stats": "/query/stats",
            "sqlite_stats": "/documents/sqlite-stats",
            "unvectorized_chunks": "/documents/unvectorized",
            "vectorize_pending": "/documents/vectorize-pending",
            "chunk_metadata": "/documents/chunks/metadata",
            "health": "/health",
            "clear_db": "/documents/clear",
            "clear_sqlite_db": "/documents/sqlite/clear"
        },
        "supported_formats": ["PDF", "DOCX", "XLSX", "TXT", "CSV", "PPTX"],
        "retrieval_methods": ["hybrid_rrf", "hybrid_weighted", "hybrid_borda", "dense_only", "sparse_only"],
        "docs": "/docs",
        "notes": {
            "query_api": "Query API now returns vector_id for each source, enabling direct chunk metadata inspection",
            "hybrid_retrieval": "New hybrid retrieval system combining dense and sparse search methods",
            "hybrid_bulk_csv": "Process multiple questions from CSV using hybrid retrieval methods",
            "performance_comparison": "Built-in performance comparison and evaluation tools"
        }
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Harfang Document Processing API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--log-level", default="info", help="Log level")
    
    args = parser.parse_args()
    
    logger.info(f"Starting Harfang Document Processing API on {args.host}:{args.port}")
    
    uvicorn.run(
        "app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level
    ) 