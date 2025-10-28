#!/usr/bin/env python3
"""
Enhanced Document processing service for API-based document handling.
Integrates block-based content analysis and coordinate-based image extraction with OCR.
Extracts core logic from pipeline for individual document processing.
"""

import os
import sys
import time
import logging
import tempfile
import hashlib
import json
import cv2
import io
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import asyncio
from tqdm import tqdm
import re

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from core.config import config
from core.embedding_service import EmbeddingService
from core.vector_db import get_vector_db, VectorPoint
from .database_service import DatabaseService
from models.api_models import (
    DocumentProcessResponse,
    DocumentInfo,
    ChunkInfo,
    EmbeddingInfo,
    ProcessingStats,
    StoredDocument,
    DocumentListResponse,
    DeleteDocumentRequest,
    DeleteDocumentResponse
)

# Import text extraction services
from services.hybrid_text_extraction_service import HybridTextExtractionService
from services.chunking_service import ChunkingService
from services.contextual_spell_correction_service import ContextualSpellCorrectionService
from services.date_extraction_service import DateExtractionService

# Import our new enhanced analysis modules
from services.enhanced_block_content_analysis import EnhancedBlockAnalyzer, ContentBlock

# Import for image extraction and OCR
try:
    import fitz  # PyMuPDF
    from PIL import Image
    import easyocr
    OCRCAPABLE = True
    EASYOCR_AVAILABLE = True
except ImportError as e:
    OCRCAPABLE = False
    EASYOCR_AVAILABLE = False
    logging.warning(f"OCR capabilities disabled - missing PyMuPDF, PIL, EasyOCR, or numpy: {e}")

# Try importing PyPDF2 for metadata extraction
try:
    from PyPDF2 import PdfReader
    from PyPDF2.generic import IndirectObject
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    print("PyPDF2 not available - PDF metadata extraction will be limited")

class EnhancedDocumentService:
    """Service for processing individual documents with enhanced block-based analysis."""
    
    def __init__(self):
        """Initialize the enhanced document service."""
        self.logger = logging.getLogger(__name__)
        
        # Initialize services
        self.vector_db = None
        self.embedding_service = None
        self.text_extraction_service = None
        self.chunking_service = None
        self.spell_correction_service = None
        self.date_extraction_service = None
        self.database_service = DatabaseService()  # SQLite database service
        
        # Enhanced analysis components
        self.block_analyzer = EnhancedBlockAnalyzer()
        
        # ENHANCED: Initialize similarity service for deduplication
        from .similarity_service import SimilarityService
        self.similarity_service = SimilarityService(similarity_threshold=0.5)  # 50% threshold
        
        # EasyOCR reader
        self.easyocr_reader = None
        
        # Statistics
        self.stats = {
            "total_documents_processed": 0,
            "total_chunks_created": 0,
            "total_characters_processed": 0,
            "total_embeddings_created": 0,
            "total_blocks_extracted": 0,
            "total_images_extracted": 0,
            "total_ocr_applied": 0,
            "processing_errors": 0,
            "last_processed": None
        }
        
        self.logger.info("EnhancedDocumentService initialized with PyMuPDF block-based analysis and EasyOCR")

    async def initialize(self):
        """Initialize the enhanced document service and all components."""
        try:
            self.logger.info("Initializing EnhancedDocumentService with PyMuPDF block-based analysis")
            
            # Create service instances
            self.embedding_service = EmbeddingService()
            # COMMENTED OUT: Step B. Hybrid Text Extraction
            # self.text_extraction_service = HybridTextExtractionService()
            self.text_extraction_service = None  # Disabled hybrid extraction
            self.chunking_service = ChunkingService()
            self.spell_correction_service = ContextualSpellCorrectionService()
            self.date_extraction_service = DateExtractionService()
            
            # Only initialize services that have async initialize methods
            # ChunkingService, ContextualSpellCorrectionService, and DateExtractionService
            # initialize themselves in their constructors
            
            # COMMENTED OUT: Step B. Hybrid Text Extraction initialization
            # Check if text_extraction_service has initialize method
            # if hasattr(self.text_extraction_service, 'initialize'):
            #     await self.text_extraction_service.initialize()
            
            # Initialize EasyOCR reader if available
            if EASYOCR_AVAILABLE:
                try:
                    self.logger.info("Initializing EasyOCR reader for enhanced document service...")
                    self.easyocr_reader = easyocr.Reader(['en'], gpu=False)
                    self.logger.info("EasyOCR reader initialized successfully")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize EasyOCR reader: {e}")
                    self.easyocr_reader = None
            else:
                self.logger.warning("EasyOCR not available. OCR capabilities will be limited.")
            
            # Initialize vector database
            self.vector_db = get_vector_db()
            
            self.logger.info("EnhancedDocumentService initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing enhanced document service: {e}")
            raise

    async def cleanup(self):
        """Cleanup resources."""
        try:
            if self.vector_db:
                self.vector_db.close()
            if self.text_extraction_service:
                await self.text_extraction_service.cleanup()
            self.logger.info("Enhanced document service cleanup complete")
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {e}")

    async def process_document(
        self, 
        filename: str, 
        content: bytes, 
        content_type: Optional[str] = None
    ) -> DocumentProcessResponse:
        """
        Process a single document through the enhanced block-based pipeline with progress tracking.
        
        Args:
            filename: Original filename
            content: File content as bytes
            content_type: MIME content type
            
        Returns:
            DocumentProcessResponse with processing results
        """
        start_time = time.time()
        
        # Define processing stages for enhanced pipeline
        stages = [
            "Validating file",
            "Analyzing PDF blocks", 
            "Counting PDF pages",
            "Applying page OCR",
            "Combining content",
            "Creating semantic chunks",
            "Applying spell correction",
            "Extracting creation date",
            "Checking for duplicates",
            "Storing document",
            "Generating embeddings",
            "Storing vectors"
        ]
        
        # Create main progress bar
        with tqdm(total=len(stages), desc=f"Processing {filename}", unit="stage") as pbar:
            try:
                self.logger.info(f"Processing document with enhanced block analysis: {filename}")
                
                # Create document info
                doc_info = DocumentInfo(
                    filename=filename,
                    file_size=len(content),
                    file_type=content_type or "unknown",
                    file_extension=Path(filename).suffix.lower()
                )
                
                # Stage 1: Validate file
                pbar.set_description(f"Processing {filename} - {stages[0]}")
                if not self._validate_file(doc_info):
                    return DocumentProcessResponse(
                        success=False,
                        document_info=doc_info,
                        error="File validation failed"
                    )
                pbar.update(1)
                
                # Only apply enhanced processing to PDFs
                if doc_info.file_extension.lower() == '.pdf':
                    # Enhanced PDF processing pipeline
                    blocks_data, extracted_content = await self._process_pdf_with_blocks(
                        filename, content, pbar, stages[1:5]
                    )
                    text_content = extracted_content
                else:
                    # Fallback to original text extraction for non-PDFs
                    pbar.set_description(f"Processing {filename} - Text extraction (non-PDF)")
                    text_content = await self._extract_text_fallback(filename, content, doc_info.file_extension)
                    blocks_data = []
                    # Skip block-related stages
                    for _ in range(5):
                        pbar.update(1)
                
                if not text_content or not text_content.strip():
                    return DocumentProcessResponse(
                        success=False,
                        document_info=doc_info,
                        error="No text content extracted from document"
                    )
                
                # ENHANCED: Stage 6: Create semantic chunks with similarity-based deduplication
                pbar.set_description(f"Processing {filename} - {stages[5]}")
                chunks = await self._create_enhanced_chunks(filename, text_content, blocks_data)
                
                # Apply similarity-based deduplication to chunks
                deduplicated_chunks, removed_count = self.similarity_service.deduplicate_chunks(
                    chunks, threshold=0.6  # 60% threshold for chunks to preserve diversity
                )
                chunks = deduplicated_chunks
                
                self.logger.info(f"Chunk deduplication: {removed_count} similar chunks removed, "
                               f"{len(chunks)} unique chunks retained")
                
                if not chunks:
                    return DocumentProcessResponse(
                        success=False,
                        document_info=doc_info,
                        error="No unique chunks created from document after deduplication"
                    )
                pbar.update(1)
                
                # Stage 7: Apply spell correction to chunks
                pbar.set_description(f"Processing {filename} - {stages[6]}")
                corrected_chunks, correction_details = await self._apply_spell_correction_to_chunks(filename, chunks)
                chunks = corrected_chunks
                pbar.update(1)
                
                # Stage 8: Extract creation date from document
                pbar.set_description(f"Processing {filename} - {stages[7]}")
                document_creation_date = await self._extract_creation_date(filename, content)
                date_metadata = self._prepare_date_metadata(document_creation_date)
                
                # Update chunks with date metadata
                for chunk in chunks:
                    chunk["metadata"].update(date_metadata)
                pbar.update(1)
                
                # ENHANCED: Stage 9: Check for duplicates using similarity threshold
                pbar.set_description(f"Processing {filename} - {stages[8]}")
                
                # Check by filename first (fast check)
                existing_doc_id = self.database_service.document_exists(filename)
                
                if existing_doc_id:
                    # Get existing document content for similarity check
                    existing_doc = self.database_service.get_document_by_id(existing_doc_id)
                    if existing_doc:
                        existing_content = existing_doc.get('raw_text', '')
                        
                        # Check if content is similar (50% threshold)
                        is_similar, similarity_score = self.similarity_service.are_documents_similar(
                            text_content, existing_content
                        )
                        
                        if is_similar:
                            self.logger.info(f"Document '{filename}' is similar to existing document "
                                           f"(ID {existing_doc_id}, similarity: {similarity_score:.2%})")
                            
                            # Get existing document info
                            existing_chunks = self.database_service.get_document_chunks(existing_doc_id)
                            
                            # Complete progress bar
                            pbar.update(len(stages) - pbar.n)
                            
                            return DocumentProcessResponse(
                                success=True,
                                document_info=doc_info,
                                chunk_info=ChunkInfo(
                                    total_chunks=len(existing_chunks),
                                    average_chunk_size=sum(chunk['chunk_size'] for chunk in existing_chunks) / len(existing_chunks) if existing_chunks else 0,
                                    min_chunk_size=min(chunk['chunk_size'] for chunk in existing_chunks) if existing_chunks else 0,
                                    max_chunk_size=max(chunk['chunk_size'] for chunk in existing_chunks) if existing_chunks else 0
                                ),
                                embedding_info=EmbeddingInfo(
                                    embeddings_created=len(existing_chunks),
                                    embeddings_stored=len([c for c in existing_chunks if c['vectorized']]),
                                    embedding_model=config.EMBEDDING_MODEL,
                                    embedding_dimension=config.EMBEDDING_DIMENSION
                                ),
                                vector_db_id=config.COLLECTION_NAME,
                                processing_stats={
                                    "text_length": len(text_content),
                                    "processing_time": 0.0,
                                    "duplicate_detected": True,
                                    "similarity_score": similarity_score,
                                    "blocks_extracted": len(blocks_data)
                                }
                            )
                        else:
                            self.logger.info(f"Document '{filename}' has same name but different content "
                                           f"(similarity: {similarity_score:.2%}) - processing as new document")
                pbar.update(1)
                
                # Stage 10: Store new document in SQLite database
                pbar.set_description(f"Processing {filename} - {stages[9]}")
                document_id = self.database_service.store_document(
                    filename=filename,
                    file_type=doc_info.file_type,
                    file_size=doc_info.file_size,
                    raw_text=text_content,
                    page_count=getattr(doc_info, 'page_count', None),
                    extraction_method=f"enhanced_block_analysis_{doc_info.file_extension}",
                    file_content=content  # Add original file content for hash calculation
                )
                
                # Store chunks in SQLite
                chunk_ids = self.database_service.store_chunks(document_id, chunks)
                pbar.update(1)
                
                # Stage 11: Create embeddings
                pbar.set_description(f"Processing {filename} - {stages[10]}")
                embeddings = await self._create_embeddings(chunks)
                
                if not embeddings:
                    return DocumentProcessResponse(
                        success=False,
                        document_info=doc_info,
                        error="Failed to create embeddings"
                    )
                pbar.update(1)
                
                # Stage 12: Store embeddings in vector database and update SQLite
                pbar.set_description(f"Processing {filename} - {stages[11]}")
                stored_count = await self._store_embeddings_with_tracking(
                    chunks, embeddings, chunk_ids, document_id
                )
                
                if stored_count == 0:
                    return DocumentProcessResponse(
                        success=False,
                        document_info=doc_info,
                        error="Failed to store embeddings in vector database"
                    )
                pbar.update(1)
                
                # Update statistics
                self.stats["total_documents_processed"] += 1
                self.stats["total_chunks_created"] += len(chunks)
                self.stats["total_characters_processed"] += len(text_content)
                self.stats["total_embeddings_created"] += stored_count
                self.stats["total_blocks_extracted"] += len(blocks_data)
                self.stats["total_images_extracted"] += len(blocks_data)
                self.stats["total_ocr_applied"] += len(blocks_data)
                self.stats["last_processed"] = datetime.now()
                
                # Calculate processing time
                processing_time = time.time() - start_time
                doc_info.processing_time = processing_time
                
                # Create response
                chunk_sizes = [len(chunk["content"]) for chunk in chunks]
                chunk_info = ChunkInfo(
                    total_chunks=len(chunks),
                    average_chunk_size=sum(chunk_sizes) / len(chunk_sizes),
                    min_chunk_size=min(chunk_sizes),
                    max_chunk_size=max(chunk_sizes)
                )
                
                embedding_info = EmbeddingInfo(
                    embeddings_created=len(embeddings),
                    embeddings_stored=stored_count,
                    embedding_model=config.EMBEDDING_MODEL,
                    embedding_dimension=config.EMBEDDING_DIMENSION
                )
                
                self.logger.info(f"Successfully processed document with enhanced analysis: {filename} ({processing_time:.2f}s)")
                
                return DocumentProcessResponse(
                    success=True,
                    document_info=doc_info,
                    chunk_info=chunk_info,
                    embedding_info=embedding_info,
                    vector_db_id=config.COLLECTION_NAME,
                    processing_stats={
                        "text_length": len(text_content),
                        "processing_time": processing_time,
                        "spell_correction": correction_details,
                        "blocks_extracted": len(blocks_data),
                        "images_extracted": self.stats["total_images_extracted"],
                        "ocr_applied": self.stats["total_ocr_applied"]
                    }
                )
                
            except Exception as e:
                self.logger.error(f"Error processing document {filename}: {e}")
                self.stats["processing_errors"] += 1
                
                return DocumentProcessResponse(
                    success=False,
                    document_info=DocumentInfo(
                        filename=filename,
                        file_size=len(content),
                        file_type=content_type or "unknown",
                        file_extension=Path(filename).suffix.lower()
                    ),
                    error=str(e)
                )

    async def _process_pdf_with_blocks(
        self, 
        filename: str, 
        content: bytes, 
        pbar: tqdm, 
        stage_names: List[str]
    ) -> Tuple[List[ContentBlock], str]:
        """Process PDF using enhanced block-based analysis with PyMuPDF text extraction."""
        try:
            # Stage 1: Enhanced block-based content analysis with PyMuPDF text extraction
            pbar.set_description(f"Processing {filename} - Enhanced PyMuPDF block analysis")
            
            # Save PDF content to temporary file for analysis
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_file.write(content)
                temp_pdf_path = Path(temp_file.name)
            
            try:
                # Run enhanced PyMuPDF analysis with text extraction
                self.logger.info(f"Starting enhanced PyMuPDF analysis for {filename}")
                analysis_results = self.block_analyzer.analyze_pdf_with_pymupdf(temp_pdf_path)
                
                if not analysis_results.get('success', False):
                    raise Exception(f"PyMuPDF analysis failed: {analysis_results.get('error', 'Unknown error')}")
                
                blocks_data = analysis_results.get('content_blocks', [])
                
                # Log extraction statistics
                extraction_stats = analysis_results.get('text_extraction_stats', {})
                total_blocks = extraction_stats.get('total_blocks', 0)
                blocks_with_text = extraction_stats.get('blocks_with_extracted_text', 0)
                extraction_rate = extraction_stats.get('text_extraction_rate', 0)
                
                self.logger.info(f"PyMuPDF analysis completed for {filename}:")
                self.logger.info(f"  Total blocks detected: {total_blocks}")
                self.logger.info(f"  Blocks with text extracted: {blocks_with_text}")
                self.logger.info(f"  Text extraction rate: {extraction_rate:.2%}")
                
                # Save PyMuPDF extraction results to specified folder
                extraction_dir = self.block_analyzer.save_pymupdf_extraction_results(
                    analysis_results, 
                    Path(filename).with_suffix('.pdf'),  # Create Path object from filename
                    "PyMyPDF-extraction"
                )
                
                if extraction_dir:
                    self.logger.info(f"PyMuPDF extraction results saved to: {extraction_dir}")
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_pdf_path)
                except Exception as e:
                    self.logger.warning(f"Could not delete temporary file {temp_pdf_path}: {e}")
            
            pbar.update(1)
            
            # COMMENTED OUT: Step B. Hybrid Text Extraction Service
            # Stage 2: Apply hybrid text extraction (PyMuPDF + EasyOCR column detection)
            # pbar.set_description(f"Processing {filename} - {stage_names[1]}")
            # extracted_content = ""
            # if self.text_extraction_service:
            #     # Apply hybrid extraction with column detection and OCR enhancement
            #     hybrid_result = await self.text_extraction_service.extract_pdf_content_with_column_detection(
            #         content, filename
            #     )
            #     if hybrid_result.get('success', False):
            #         extracted_content = hybrid_result.get('content', '')
            #         self.logger.info(f"Hybrid extraction successful for {filename}: {len(extracted_content)} chars")
            #     else:
            #         self.logger.warning(f"Hybrid extraction failed for {filename}, using fallback")
            #         extracted_content = await self._basic_pdf_text_extraction(content)
            # else:
            #     extracted_content = await self._basic_pdf_text_extraction(content)
            # pbar.update(1)
            
            # Basic fallback text extraction (preserves original functionality)
            pbar.set_description(f"Processing {filename} - Basic text extraction fallback")
            extracted_content = await self._basic_pdf_text_extraction(content)
            pbar.update(1)
            
            # Stage 3: Get PDF page count for OCR
            pbar.set_description(f"Processing {filename} - {stage_names[1]}")
            page_count = self._get_pdf_page_count(content)
            pbar.update(1)
            
            # COMMENTED OUT: Step C. Page-Level OCR Enhancement
            # Stage 4: Apply OCR to entire pages
            # pbar.set_description(f"Processing {filename} - {stage_names[2]}")
            # ocr_content_by_page = {}
            # if EASYOCR_AVAILABLE and self.easyocr_reader:
            #     ocr_content_by_page = await self._apply_ocr_to_pdf_pages(content, page_count, filename)
            
            # NEW: Stage 4: Apply selective OCR to specific block types only
            pbar.set_description(f"Processing {filename} - Selective OCR on specific blocks")
            ocr_content_by_page = {}
            if EASYOCR_AVAILABLE and self.easyocr_reader:
                ocr_content_by_page = await self._apply_selective_ocr_to_blocks(content, blocks_data, filename)
            pbar.update(1)
            
            # Skip stages that are no longer used
            pbar.update(2)  # Skip coordinate extraction and image extraction stages
            
            # Stage 6: Combine block content with selective OCR
            pbar.set_description(f"Processing {filename} - {stage_names[3]}")
            extracted_content = await self._combine_blocks_with_selective_ocr(blocks_data, ocr_content_by_page)
            pbar.update(1)
            
            return blocks_data, extracted_content
            
        except Exception as e:
            self.logger.error(f"Error in PyMuPDF PDF processing: {e}")
            # Fallback to standard text extraction
            fallback_content = await self._extract_text_fallback(filename, content, '.pdf')
            return [], fallback_content

    def _get_pdf_page_count(self, pdf_content: bytes) -> int:
        """Get the number of pages in the PDF."""
        try:
            doc = fitz.open(stream=pdf_content, filetype="pdf")
            page_count = len(doc)
            doc.close()
            return page_count
        except Exception as e:
            self.logger.error(f"Error getting PDF page count: {e}")
            return 0

    async def _apply_selective_ocr_to_blocks(self, pdf_content: bytes, blocks_data: List[ContentBlock], filename: str) -> Dict[int, str]:
        """Apply OCR only to blocks classified as image, figure, chart, or drawing."""
        ocr_results = {}
        
        # Define target block types for OCR
        ocr_target_types = {"IMAGE", "FIGURE", "CHART", "DRAWING"}
        
        try:
            if not EASYOCR_AVAILABLE or not self.easyocr_reader:
                self.logger.warning("EasyOCR not available. Skipping selective OCR on blocks.")
                return ocr_results
            
            # Filter blocks to only those that need OCR
            ocr_blocks = [block for block in blocks_data if block.block_type.value in ocr_target_types]
            
            if not ocr_blocks:
                self.logger.info("No blocks found that require OCR processing.")
                return ocr_results
            
            self.logger.info(f"Applying selective OCR to {len(ocr_blocks)} blocks of types: {ocr_target_types}")
            
            # Create a dedicated directory for this document's OCR images
            base_filename = Path(filename).stem
            ocr_images_dir = Path("ocr_applied_images") / base_filename
            ocr_images_dir.mkdir(parents=True, exist_ok=True)

            # Load PDF document
            doc = fitz.open(stream=pdf_content, filetype="pdf")
            
            # Process OCR blocks with progress bar
            with tqdm(total=len(ocr_blocks), desc="Selective OCR Processing", unit="block") as pbar:
                for block_idx, block in enumerate(ocr_blocks):
                    try:
                        pbar.set_description(f"OCR Processing - {block.block_type.value} Block {block_idx + 1}/{len(ocr_blocks)}")
                        
                        # Get page for this block
                        page_num = block.page_number - 1  # Convert to 0-based
                        if page_num >= len(doc):
                            self.logger.warning(f"Page {block.page_number} not found in PDF")
                            pbar.update(1)
                            continue
                            
                        page = doc[page_num]
                        
                        # ENHANCED: Validate and fix coordinates before processing
                        validated_coords = self._validate_and_fix_coordinates(block.bbox, page.rect)
                        if not validated_coords:
                            self.logger.debug(f"Skipping {block.block_type.value} block {block_idx + 1} due to invalid coordinates: {block.bbox}")
                            pbar.update(1)
                            continue
                        
                        x0, y0, x1, y1 = validated_coords
                        
                        # Create crop rectangle with validated coordinates
                        crop_rect = fitz.Rect(x0, y0, x1, y1)
                        
                        # Additional validation: ensure crop rectangle is valid
                        if crop_rect.is_empty or crop_rect.is_infinite:
                            self.logger.debug(f"Skipping {block.block_type.value} block {block_idx + 1}: invalid crop rectangle")
                            pbar.update(1)
                            continue
                        
                        # Render the specific block region as high-resolution image
                        mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better OCR quality
                        
                        try:
                            pix = page.get_pixmap(matrix=mat, clip=crop_rect)
                            
                            # Validate pixmap
                            if not pix or pix.width <= 0 or pix.height <= 0:
                                self.logger.debug(f"Skipping {block.block_type.value} block {block_idx + 1}: invalid pixmap dimensions")
                                pbar.update(1)
                                continue
                            
                        except Exception as pix_error:
                            self.logger.debug(f"Failed to create pixmap for {block.block_type.value} block {block_idx + 1}: {pix_error}")
                            pbar.update(1)
                            continue
                        
                        # Convert to PIL Image
                        img_data = pix.tobytes("ppm")
                        img = Image.open(io.BytesIO(img_data))
                        
                        # Save the image before preprocessing, to see what OCR gets
                        try:
                            # Use a simpler filename now that it's in a dedicated folder
                            image_filename = f"page_{block.page_number}_block_{block_idx}_{block.block_type.value.lower()}.png"
                            image_save_path = ocr_images_dir / image_filename
                            img.save(image_save_path, "PNG", dpi=(150, 150))
                            self.logger.debug(f"Saved OCR image block to {image_save_path}")
                        except Exception as e:
                            self.logger.error(f"Failed to save OCR image for block {block_idx}: {e}")

                        # ENHANCED: Preprocess image for OCR
                        img = self._preprocess_image_for_ocr(img)
                        if not img:
                            self.logger.debug(f"Skipping {block.block_type.value} block {block_idx + 1}: image preprocessing failed")
                            pbar.update(1)
                            continue
                        
                        # Convert PIL image to numpy array for EasyOCR
                        img_array = np.array(img)
                        
                        # Final validation before OCR
                        if img_array.size == 0:
                            self.logger.debug(f"Skipping {block.block_type.value} block {block_idx + 1}: empty image array")
                            pbar.update(1)
                            continue
                        
                        # Apply OCR to the block region
                        results = self.easyocr_reader.readtext(img_array, detail=0)
                        
                        # Join all text results
                        ocr_text = ' '.join(results) if results else ""
                        
                        if ocr_text.strip():
                            # Store OCR result with block information
                            page_key = block.page_number
                            if page_key not in ocr_results:
                                ocr_results[page_key] = []
                            
                            ocr_results[page_key].append({
                                'block_type': block.block_type.value,
                                'text': ocr_text.strip(),
                                'coordinates': block.bbox,
                                'block_index': block_idx
                            })
                            
                            self.logger.debug(f"Page {block.page_number}, {block.block_type.value}: Extracted {len(ocr_text)} characters via OCR")
                        else:
                            self.logger.debug(f"Page {block.page_number}, {block.block_type.value}: No text extracted via OCR")
                        
                        # Update stats
                        self.stats["total_ocr_applied"] += 1
                        
                        pbar.update(1)
                        
                    except Exception as e:
                        # Detailed error reporting for better debugging
                        error_msg = str(e)
                        if "Invalid bandwriter header dimensions" in error_msg:
                            self.logger.debug(f"Coordinate validation error prevented for {block.block_type.value} block {block_idx + 1}: {error_msg}")
                        elif "code=4" in error_msg:
                            self.logger.debug(f"PyMuPDF rendering error for {block.block_type.value} block {block_idx + 1}: {error_msg}")
                        else:
                            self.logger.warning(f"OCR failed on {block.block_type.value} block {block_idx + 1}: {error_msg}")
                        pbar.update(1)
                        continue
            
            doc.close()
            
            # Enhanced success rate reporting
            successful_blocks = sum(len(page_results) for page_results in ocr_results.values())
            total_attempted = len(ocr_blocks)
            success_rate = (successful_blocks / total_attempted * 100) if total_attempted > 0 else 0
            
            # Count blocks by type for detailed reporting
            block_type_stats = {}
            successful_by_type = {}
            
            for block in ocr_blocks:
                block_type = block.block_type.value
                block_type_stats[block_type] = block_type_stats.get(block_type, 0) + 1
                successful_by_type[block_type] = 0
            
            for page_results in ocr_results.values():
                for result in page_results:
                    block_type = result['block_type']
                    successful_by_type[block_type] = successful_by_type.get(block_type, 0) + 1
            
            self.logger.info(f"Selective OCR completed: {successful_blocks}/{total_attempted} blocks processed successfully ({success_rate:.1f}%)")
            
            # Detailed breakdown by block type
            for block_type, total in block_type_stats.items():
                successful = successful_by_type.get(block_type, 0)
                type_success_rate = (successful / total * 100) if total > 0 else 0
                self.logger.info(f"  {block_type}: {successful}/{total} blocks ({type_success_rate:.1f}%)")
            
            print("--------------------------------")
            print(f"OCR results: {ocr_results}")
            print("--------------------------------")
            return ocr_results
            
        except Exception as e:
            self.logger.error(f"Error in selective OCR processing: {e}")
            return ocr_results

    async def _apply_ocr_to_pdf_pages(self, pdf_content: bytes, page_count: int, filename: str) -> Dict[int, str]:
        """Apply OCR to entire PDF pages instead of individual regions."""
        ocr_results = {}
        
        try:
            if not EASYOCR_AVAILABLE or not self.easyocr_reader:
                self.logger.warning("EasyOCR not available. Skipping OCR on PDF pages.")
                return ocr_results
            
            self.logger.info(f"Applying OCR to {page_count} pages...")
            
            # Load PDF document
            doc = fitz.open(stream=pdf_content, filetype="pdf")
            
            # Create images directory for page images
            images_dir = Path("extracted_regions") / Path(filename).stem / "pages"
            images_dir.mkdir(parents=True, exist_ok=True)
            
            # Process pages with progress bar
            with tqdm(total=page_count, desc="Page OCR Processing", unit="page") as pbar:
                for page_num in range(page_count):
                    try:
                        pbar.set_description(f"OCR Processing - Page {page_num + 1}/{page_count}")
                        
                        # Get page
                        page = doc[page_num]
                        
                        # Render page as high-resolution image
                        mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better OCR quality
                        pix = page.get_pixmap(matrix=mat)
                        
                        # Convert to PIL Image
                        img_data = pix.tobytes("ppm")
                        img = Image.open(io.BytesIO(img_data))
                        
                        # Save page image for debugging
                        page_img_path = images_dir / f"page_{page_num + 1}.png"
                        img.save(page_img_path, "PNG", dpi=(150, 150))
                        
                        # Convert PIL image to numpy array for EasyOCR
                        img_array = np.array(img)
                        
                        # Apply OCR to entire page
                        results = self.easyocr_reader.readtext(img_array, detail=0)
                        
                        # Join all text results
                        ocr_text = ' '.join(results) if results else ""
                        
                        if ocr_text.strip():
                            ocr_results[page_num + 1] = ocr_text.strip()
                            self.logger.debug(f"Page {page_num + 1}: Extracted {len(ocr_text)} characters via OCR")
                        else:
                            self.logger.debug(f"Page {page_num + 1}: No text extracted via OCR")
                        
                        # Update stats
                        self.stats["total_ocr_applied"] += 1
                        
                        pbar.update(1)
                        
                        # Log progress every 5 pages
                        if (page_num + 1) % 5 == 0:
                            successful_pages = len([p for p in ocr_results.values() if p])
                            self.logger.info(f"OCR Progress: {page_num + 1}/{page_count} pages processed, {successful_pages} with text")
                        
                    except Exception as e:
                        self.logger.warning(f"OCR failed on page {page_num + 1}: {e}")
                        pbar.update(1)
                        continue
            
            doc.close()
            
            successful_pages = len([p for p in ocr_results.values() if p])
            self.logger.info(f"Page OCR completed: {successful_pages}/{page_count} pages processed successfully")
            
            return ocr_results
            
        except Exception as e:
            self.logger.error(f"Error in page OCR processing: {e}")
            return ocr_results

    async def _combine_blocks_with_selective_ocr(
        self, 
        blocks_data: List[ContentBlock], 
        ocr_content_by_page: Dict[int, List[Dict]]
    ) -> str:
        """Combine block text content with selective OCR results, removing duplicates and semantic noise."""
        combined_content = []
        seen_content = set()  # Track seen content to prevent duplicates
        
        # Group blocks by page
        page_blocks = {}
        for block in blocks_data:
            page_num = block.page_number
            if page_num not in page_blocks:
                page_blocks[page_num] = []
            page_blocks[page_num].append(block)
        
        # Create a lookup for OCR results by page and coordinates
        ocr_lookup = {}
        for page_num, ocr_blocks in ocr_content_by_page.items():
            ocr_lookup[page_num] = {}
            for ocr_block in ocr_blocks:
                coords_key = tuple(ocr_block['coordinates'])
                ocr_lookup[page_num][coords_key] = ocr_block
        
        # Process each page
        for page_num in sorted(page_blocks.keys()):
            combined_content.append(f"=== PAGE {page_num} ===")
            
            page_block_list = page_blocks[page_num]
            page_content_added = []  # Track content added for this page to avoid duplicates
            
            # Add block content with enhanced PyMuPDF + selective OCR integration
            for block in page_block_list:
                # Check if this block has OCR results
                coords_key = tuple(block.bbox)
                ocr_result = None
                if page_num in ocr_lookup and coords_key in ocr_lookup[page_num]:
                    ocr_result = ocr_lookup[page_num][coords_key]
                
                # Determine if PyMuPDF extracted actual text vs placeholder
                pymupdf_has_text = (block.text_content and 
                                   not block.text_content.startswith(('Image ', 'Drawing/Graphic ')) and
                                   len(block.text_content.strip()) > 0)
                
                # Filter out placeholder drawing blocks that contain "Drawing/Graphic"
                is_placeholder_drawing = (block.text_content and 
                                         block.text_content.startswith('Drawing/Graphic ') and
                                         block.block_type.value == "DRAWING")
                
                # Skip placeholder drawing blocks to reduce semantic noise
                if is_placeholder_drawing:
                    # Only add OCR result if available for placeholder blocks
                    if ocr_result and ocr_result['text']:
                        ocr_text = ocr_result['text'].strip()
                        # Check for duplicates before adding OCR content
                        content_hash = hash(ocr_text.lower().replace('\n', ' ').replace(' ', ''))
                        if content_hash not in seen_content and ocr_text not in page_content_added:
                            combined_content.append(f"\n[{block.block_type.value}]")
                            combined_content.append(f"[OCR TEXT FROM {block.block_type.value}]")
                            combined_content.append(ocr_text)
                            combined_content.append("")
                            seen_content.add(content_hash)
                            page_content_added.append(ocr_text)
                    continue  # Skip the rest for placeholder blocks
                
                # Add block type information for non-placeholder blocks
                block_header = f"\n[{block.block_type.value}]"
                
                # Process PyMuPDF extracted text if available
                if pymupdf_has_text:
                    # Check for duplicates before adding PyMuPDF content
                    text_content = block.text_content.strip()
                    content_hash = hash(text_content.lower().replace('\n', ' ').replace(' ', ''))
                    
                    if content_hash not in seen_content and text_content not in page_content_added:
                        combined_content.append(block_header)
                        combined_content.append(f"[PyMuPDF EXTRACTED TEXT]")
                        combined_content.append(text_content)
                        seen_content.add(content_hash)
                        page_content_added.append(text_content)
                        
                        # Add OCR text if available and different from PyMuPDF text
                        if ocr_result and ocr_result['text']:
                            ocr_text = ocr_result['text'].strip()
                            ocr_hash = hash(ocr_text.lower().replace('\n', ' ').replace(' ', ''))
                            if (ocr_hash not in seen_content and 
                                ocr_text not in page_content_added and
                                ocr_text != text_content):  # Don't duplicate if OCR same as PyMuPDF
                                combined_content.append(f"[OCR TEXT FROM {block.block_type.value}]")
                                combined_content.append(ocr_text)
                                seen_content.add(ocr_hash)
                                page_content_added.append(ocr_text)
                        
                        combined_content.append("")  # Add spacing between blocks
                elif block.text_content:
                    # Handle visual elements that aren't placeholders
                    if not block.text_content.startswith(('Image ', 'Drawing/Graphic ')):
                        text_content = block.text_content.strip()
                        content_hash = hash(text_content.lower().replace('\n', ' ').replace(' ', ''))
                        
                        if content_hash not in seen_content and text_content not in page_content_added:
                            combined_content.append(block_header)
                            combined_content.append(f"[VISUAL ELEMENT] {text_content}")
                            seen_content.add(content_hash)
                            page_content_added.append(text_content)
                            combined_content.append("")
                
                # Add standalone OCR text for blocks without PyMuPDF text
                if not pymupdf_has_text and ocr_result and ocr_result['text']:
                    ocr_text = ocr_result['text'].strip()
                    ocr_hash = hash(ocr_text.lower().replace('\n', ' ').replace(' ', ''))
                    
                    if ocr_hash not in seen_content and ocr_text not in page_content_added:
                        if not is_placeholder_drawing:  # Only add header for non-placeholder blocks
                            combined_content.append(block_header)
                        combined_content.append(f"[OCR TEXT FROM {block.block_type.value}]")
                        combined_content.append(ocr_text)
                        combined_content.append("")
                        seen_content.add(ocr_hash)
                        page_content_added.append(ocr_text)
        
        # Clean up excessive whitespace while preserving structure
        result = "\n".join(combined_content)
        # Remove excessive newlines (more than 2 consecutive)
        result = re.sub(r'\n{3,}', '\n\n', result)
        
        return result

    async def _create_enhanced_chunks(
        self, 
        filename: str, 
        text_content: str, 
        blocks_data: List[ContentBlock]
    ) -> List[Dict[str, Any]]:
        """Create enhanced chunks with block-based metadata including creation date."""
        chunks = []
        
        try:
            # Extract creation date first (will be added to all chunks)
            creation_date = await self._extract_creation_date(filename, text_content.encode() if isinstance(text_content, str) else text_content)
            date_metadata = self._prepare_date_metadata(creation_date)
            
            # Create page-based chunks from blocks
            page_chunks = await self._create_block_based_chunks(filename, text_content, blocks_data)
            
            # Enhance each chunk with metadata
            for i, chunk in enumerate(page_chunks):
                enhanced_chunk = self._enhance_chunk_with_metadata(chunk, i, blocks_data)
                
                # Add creation date metadata to each chunk
                enhanced_chunk['metadata'].update(date_metadata)
                
                # Add global document metadata
                enhanced_chunk['metadata'].update({
                    'total_pages': len(set(block.page_number for block in blocks_data)),
                    'total_blocks_in_document': len(blocks_data),
                    'processing_timestamp': datetime.now().isoformat(),
                    'extraction_version': '2.0_block_based_with_page_ocr'
                })
                
                chunks.append(enhanced_chunk)
            
            self.logger.info(f"Created {len(chunks)} enhanced chunks with block metadata and creation date")
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error creating enhanced chunks: {e}")
            # Fallback to simple text chunking
            return await self._create_fallback_chunks(filename, text_content)

    async def _create_fallback_chunks(self, filename: str, text_content: str) -> List[Dict[str, Any]]:
        """Create simple chunks as fallback when enhanced chunking fails."""
        try:
            chunks = []
            chunk_size = 1000
            
            # Extract creation date for fallback chunks too
            creation_date = await self._extract_creation_date(filename, text_content.encode())
            date_metadata = self._prepare_date_metadata(creation_date)
            
            for i in range(0, len(text_content), chunk_size):
                chunk_content = text_content[i:i + chunk_size]
                chunk_id = self._generate_chunk_id(chunk_content, filename, i // chunk_size)
                
                # Add filename header to fallback chunks too
                chunk_header = f"Document: {filename}\n\n"
                formatted_chunk_content = chunk_header + chunk_content
                
                chunk = {
                    'chunk_id': chunk_id,
                    'content': formatted_chunk_content,
                    'metadata': {
                        'filename': filename,
                        'chunk_index': i // chunk_size,
                        'total_characters': len(formatted_chunk_content),
                        'extraction_method': 'fallback_text_chunking',
                        'processing_timestamp': datetime.now().isoformat(),
                        **date_metadata
                    }
                }
                chunks.append(chunk)
            
            self.logger.info(f"Created {len(chunks)} fallback chunks with creation date")
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error creating fallback chunks: {e}")
            return []

    async def _create_block_based_chunks(
        self, 
        filename: str, 
        text_content: str, 
        blocks_data: List[ContentBlock]
    ) -> List[Dict[str, Any]]:
        """Create chunks based on document blocks for better semantic coherence, filtering placeholders."""
        chunks = []
        
        # Group blocks by page and type for logical chunking
        page_blocks = {}
        for block in blocks_data:
            page_num = block.page_number
            if page_num not in page_blocks:
                page_blocks[page_num] = []
            page_blocks[page_num].append(block)
        
        chunk_index = 0
        
        for page_num in sorted(page_blocks.keys()):
            page_block_list = page_blocks[page_num]
            
            # Create page-aware chunks
            current_chunk_content = []
            current_chunk_blocks = []
            current_size = 0
            max_chunk_size = self.chunking_service.max_chunk_size if self.chunking_service else 2000
            
            for block in page_block_list:
                # Filter out placeholder drawing blocks to reduce semantic noise
                is_placeholder_drawing = (block.text_content and 
                                         block.text_content.startswith('Drawing/Graphic ') and
                                         block.block_type.value == "DRAWING")
                
                # Skip placeholder drawing blocks entirely
                if is_placeholder_drawing:
                    continue
                
                # Extract text from block
                block_text = block.text_content.strip() if block.text_content else ""
                block_size = len(block_text)
                
                # Check if adding this block would exceed chunk size
                if current_size + block_size > max_chunk_size and current_chunk_content:
                    # Save current chunk
                    chunk = self._create_block_chunk_data(
                        "\n".join(current_chunk_content),
                        filename,
                        chunk_index,
                        current_chunk_blocks,
                        page_num
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    
                    # Start new chunk
                    current_chunk_content = []
                    current_chunk_blocks = []
                    current_size = 0
                
                # Add block to current chunk with proper hierarchy
                if block_text:
                    # Determine if PyMuPDF extracted actual text vs placeholder
                    pymupdf_has_text = (block_text and 
                                       not block_text.startswith(('Image ', 'Drawing/Graphic ')) and
                                       len(block_text.strip()) > 0)
                    
                    # Only add semantic labels for non-TEXT_BLOCK types with actual content
                    if block.block_type.value != "TEXT_BLOCK" and pymupdf_has_text:
                        # For specific block types with actual extracted text, include type information
                        if block.block_type.value in {"IMAGE", "FIGURE", "CHART", "DRAWING", "TABLE", "HEADER"}:
                            block_context = f"[Page {page_num}, {block.block_type.value}] {block_text}"
                        else:
                            block_context = f"[Page {page_num}, {block.block_type.value}] {block_text}"
                        current_chunk_content.append(block_context)
                    elif pymupdf_has_text:
                        # For TEXT_BLOCK or blocks with actual text, add content without semantic label
                        current_chunk_content.append(block_text)
                    # Skip blocks without actual text content (they were likely placeholders)
                    
                    if pymupdf_has_text:  # Only add to blocks list if has actual content
                        current_chunk_blocks.append(block)
                        current_size += len(block_context if block.block_type.value != "TEXT_BLOCK" else block_text)
            
            # Handle remaining content for this page
            if current_chunk_content:
                chunk = self._create_block_chunk_data(
                    "\n".join(current_chunk_content),
                    filename,
                    chunk_index,
                    current_chunk_blocks,
                    page_num
                )
                chunks.append(chunk)
                chunk_index += 1
        
        self.logger.info(f"Created {len(chunks)} block-based chunks for {filename} (filtered placeholders)")
        return chunks

    def _create_block_chunk_data(
        self, 
        content: str, 
        filename: str, 
        chunk_index: int, 
        blocks: List[ContentBlock], 
        page_number: int
    ) -> Dict[str, Any]:
        """Create chunk data with enhanced block metadata and proper formatting."""
        chunk_id = self._generate_chunk_id(content, filename, chunk_index)
        
        # Start each chunk with PDF filename and page number as required
        chunk_header = f"Document: {filename}\nPage: {page_number}\n\n"
        formatted_content = chunk_header + content
        
        # Extract blocks for this page
        page_blocks = [block for block in blocks if block.page_number == page_number]
        
        # Create block metadata
        block_metadata = []
        for block in page_blocks:
            block_info = {
                'block_type': block.block_type.value,
                'coordinates': {
                    'x0': block.bbox[0],
                    'y0': block.bbox[1], 
                    'x1': block.bbox[2],
                    'y1': block.bbox[3]
                }
            }
            block_metadata.append(block_info)
        
        return {
            'chunk_id': chunk_id,
            'content': formatted_content,
            'metadata': {
                'filename': filename,
                'chunk_index': chunk_index,
                'page_number': page_number,
                'total_characters': len(formatted_content),
                'block_count': len(page_blocks),
                'blocks': block_metadata,
                'extraction_library': 'PyMuPDF',
                'extraction_method': 'block_based_with_selective_ocr'
            }
        }

    def _enhance_chunk_with_metadata(
        self, 
        chunk: Dict[str, Any], 
        chunk_index: int, 
        blocks_data: List[ContentBlock]
    ) -> Dict[str, Any]:
        """Enhance chunk with block analysis metadata and document information."""
        metadata = chunk.get('metadata', {})
        page_number = metadata.get('page_number', 1)
        
        # Get blocks for this chunk's page
        page_blocks = [block for block in blocks_data if block.page_number == page_number]
        
        # Add block analysis summary
        block_types = {}
        total_confidence = 0
        coordinate_bounds = {'x_min': float('inf'), 'y_min': float('inf'), 'x_max': 0, 'y_max': 0}
        
        for block in page_blocks:
            block_type = block.block_type.value
            block_types[block_type] = block_types.get(block_type, 0) + 1
            total_confidence += block.confidence
            
            # Update coordinate bounds
            x0, y0, x1, y1 = block.bbox
            coordinate_bounds['x_min'] = min(coordinate_bounds['x_min'], x0)
            coordinate_bounds['y_min'] = min(coordinate_bounds['y_min'], y0)
            coordinate_bounds['x_max'] = max(coordinate_bounds['x_max'], x1)
            coordinate_bounds['y_max'] = max(coordinate_bounds['y_max'], y1)
        
        # Reset infinite values if no blocks
        if coordinate_bounds['x_min'] == float('inf'):
            coordinate_bounds = {'x_min': 0, 'y_min': 0, 'x_max': 0, 'y_max': 0}
        
        # Calculate average confidence
        avg_confidence = total_confidence / len(page_blocks) if page_blocks else 0
        
        # Enhanced metadata
        enhanced_metadata = {
            **metadata,
            'block_analysis': {
                'block_types_on_page': block_types,
                'average_confidence': round(avg_confidence, 3),
                'page_coordinate_bounds': coordinate_bounds,
                'total_blocks_on_page': len(page_blocks)
            },
            'chunk_quality_score': min(1.0, avg_confidence + (len(chunk['content']) / 1000) * 0.1)
        }
        
        return {
            **chunk,
            'metadata': enhanced_metadata
        }

    def _generate_chunk_id(self, chunk_content: str, filename: str, chunk_index: int) -> str:
        """Generate a unique chunk ID."""
        content_hash = hashlib.md5(chunk_content.encode('utf-8')).hexdigest()[:8]
        filename_hash = hashlib.md5(filename.encode('utf-8')).hexdigest()[:4]
        return f"{filename_hash}_{chunk_index:04d}_{content_hash}"

    async def _extract_text_fallback(self, filename: str, content: bytes, extension: str) -> str:
        """Fallback text extraction for non-PDFs or when enhanced processing fails."""
        try:
            # Save content to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
                temp_file.write(content)
                temp_path = Path(temp_file.name)
            
            try:
                # COMMENTED OUT: Step B. Hybrid Text Extraction fallback
                # Extract text using the hybrid text extraction service
                # text_content = await self.text_extraction_service.extract_text(temp_path, extension)
                
                # FALLBACK: Use basic PyMuPDF extraction for PDFs
                if extension == '.pdf':
                    text_content = await self._basic_pdf_text_extraction(temp_path)
                else:
                    # For non-PDFs, use basic file reading
                    text_content = await self._basic_file_text_extraction(temp_path, extension)
                
                # Save page-wise text files for PDFs
                if extension == '.pdf' and text_content:
                    await self._save_page_wise_text(filename, text_content)
                
                return text_content
            finally:
                # Clean up temporary file
                temp_path.unlink(missing_ok=True)
                
        except Exception as e:
            self.logger.error(f"Error extracting text from {filename}: {e}")
            return ""

    async def _basic_pdf_text_extraction(self, pdf_path: Path) -> str:
        """Basic PDF text extraction using PyMuPDF only."""
        try:
            doc = fitz.open(str(pdf_path))
            text_content = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text()
                if page_text.strip():
                    text_content.append(f"=== PAGE {page_num + 1} ===\n{page_text}")
            
            doc.close()
            return '\n\n'.join(text_content)
            
        except Exception as e:
            self.logger.error(f"Error in basic PDF text extraction: {e}")
            return ""

    async def _basic_file_text_extraction(self, file_path: Path, extension: str) -> str:
        """Basic text extraction for non-PDF files."""
        try:
            if extension == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                # For other file types, return empty (could be extended later)
                self.logger.warning(f"Basic extraction not implemented for {extension}")
                return ""
                
        except Exception as e:
            self.logger.error(f"Error in basic file text extraction: {e}")
            return ""

    def _validate_file(self, doc_info: DocumentInfo) -> bool:
        """Validate file for processing."""
        try:
            # Check file size
            if doc_info.file_size == 0:
                self.logger.warning(f"File {doc_info.filename} is empty")
                return False
            
            if doc_info.file_size > config.MAX_FILE_SIZE:
                self.logger.warning(f"File {doc_info.filename} too large: {doc_info.file_size} bytes")
                return False
            
            # Check file extension
            supported_extensions = ['.pdf', '.docx', '.doc', '.xlsx', '.xls', '.txt', '.csv', '.pptx']
            if doc_info.file_extension not in supported_extensions:
                self.logger.warning(f"Unsupported file extension: {doc_info.file_extension}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating file {doc_info.filename}: {e}")
            return False

    async def _save_page_wise_text(self, filename: str, text_content: str):
        """Save extracted text page-wise in extracted_text folder with progress tracking."""
        try:
            # Create extracted_text directory if it doesn't exist
            extracted_text_dir = Path("extracted_text")
            extracted_text_dir.mkdir(exist_ok=True)
            
            # Create subdirectory for this document
            doc_name = Path(filename).stem
            doc_dir = extracted_text_dir / doc_name
            doc_dir.mkdir(exist_ok=True)
            
            # Split text by page markers
            pages = text_content.split("=== PAGE ")
            valid_pages = []
            
            # Collect valid pages first
            for i, page_content in enumerate(pages):
                if i == 0 and not page_content.strip():
                    continue  # Skip empty first split
                
                if " ===" in page_content:
                    page_parts = page_content.split(" ===", 1)
                    if len(page_parts) == 2:
                        valid_pages.append((page_parts[0].strip(), page_parts[1].strip()))
            
            # Save pages with progress tracking
            if valid_pages:
                with tqdm(total=len(valid_pages), desc="Saving page texts", unit="page", leave=False) as pbar:
                    for page_num, content in valid_pages:
                        # Save page content to file
                        page_file = doc_dir / f"page_{page_num}.txt"
                        with open(page_file, 'w', encoding='utf-8') as f:
                            f.write(content)
                        
                        pbar.update(1)
                        pbar.set_description(f"Saving page {page_num}")
                        pbar.set_postfix({'pages_saved': pbar.n, 'total': len(valid_pages)})
                        
                        self.logger.debug(f"Saved page {page_num} text to {page_file}")
                
                self.logger.info(f"Saved {len(valid_pages)} page text files for {filename} in {doc_dir}")
            else:
                self.logger.warning(f"No valid pages found to save for {filename}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save page-wise text for {filename}: {e}")

    async def _extract_creation_date(self, filename: str, content: bytes) -> Optional[datetime]:
        """Extract file creation date using PDF metadata and filesystem fallback."""
        try:
            # First try PDF metadata (based on the provided script)
            pdf_creation_date = await self._get_pdf_metadata_creation_date(content)
            if pdf_creation_date:
                self.logger.info(f"Using PDF metadata creation date: {pdf_creation_date}")
                return pdf_creation_date
            
            # Fallback to filesystem creation time
            # Note: In a real deployment, you'd have access to the actual file path
            # For now, we'll use current time as a placeholder
            self.logger.info("PDF metadata creation date not found, using current time as fallback")
            return datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error extracting creation date: {e}")
            return None

    async def _get_pdf_metadata_creation_date(self, pdf_content: bytes) -> Optional[datetime]:
        """Extract creation date from PDF metadata."""
        try:
            if not PYPDF2_AVAILABLE:
                self.logger.warning("PyPDF2 not available - PDF metadata extraction will be limited")
                return None
            
            # Create a PDF reader from bytes
            reader = PdfReader(io.BytesIO(pdf_content))
            
            # Get metadata
            info = getattr(reader, 'metadata', None) or getattr(reader, 'documentInfo', None)
            raw_date = None
            
            if info:
                raw_date = info.get("/CreationDate") or info.get("CreationDate")
            
            self.logger.debug(f"Raw metadata CreationDate before resolution: {raw_date!r}")
            
            # Resolve IndirectObject if needed
            if isinstance(raw_date, IndirectObject):
                self.logger.debug("Found IndirectObject for CreationDate, resolving it now")
                raw_date = raw_date.get_object()
                self.logger.debug(f"Raw metadata CreationDate after resolution: {raw_date!r}")
            
            if raw_date:
                return self._parse_pdf_creation_date(str(raw_date))
            
            self.logger.info("No CreationDate field found in PDF metadata")
            return None
            
        except Exception as e:
            self.logger.debug(f"Error reading PDF metadata creation date: {e}")
            return None

    def _parse_pdf_creation_date(self, raw_date: str) -> Optional[datetime]:
        """Parse PDF creation date string."""
        try:
            self.logger.debug(f"Parsing PDF CreationDate string: {raw_date!r}")
            
            if not raw_date.startswith("D:"):
                self.logger.warning("PDF CreationDate string does not start with 'D:'")
                return None
            
            # Extract core date portion (YYYYMMDDHHmmSS)
            core = raw_date[2:16]
            
            # Parse the date
            parsed_date = datetime.strptime(core, "%Y%m%d%H%M%S")
            self.logger.debug(f"Parsed datetime: {parsed_date!r}")
            return parsed_date
            
        except ValueError as e:
            self.logger.error(f"Failed to parse PDF date portion {raw_date!r}: {e}")
            return None

    def _prepare_date_metadata(self, creation_date: Optional[datetime]) -> Dict[str, Any]:
        """Prepare date-related metadata for chunks."""
        if not creation_date:
            return {
                'creation_date': None,
                'creation_year': None,
                'creation_month': None,
                'age_days': None
            }
        
        now = datetime.now()
        age_days = (now - creation_date).days
        
        return {
            'creation_date': creation_date.isoformat(),
            'creation_year': creation_date.year,
            'creation_month': creation_date.month,
            'creation_day': creation_date.day,
            'age_days': age_days,
            'is_recent': age_days < 30,  # Less than 30 days old
            'age_category': self._categorize_document_age(age_days)
        }

    def _categorize_document_age(self, age_days: int) -> str:
        """Categorize document age for easier filtering."""
        if age_days < 7:
            return 'very_recent'
        elif age_days < 30:
            return 'recent'
        elif age_days < 90:
            return 'moderately_recent'
        elif age_days < 365:
            return 'within_year'
        elif age_days < 1825:  # 5 years
            return 'within_5_years'
        else:
            return 'older'

    async def _create_embeddings(self, chunks: List[Dict[str, Any]]) -> List[List[float]]:
        """Create embeddings for chunks with progress tracking."""
        try:
            texts = [chunk["content"] for chunk in chunks]
            
            # Use tqdm for embedding creation if there are multiple chunks
            if len(texts) > 1:
                self.logger.info(f"Creating embeddings for {len(texts)} chunks...")
                with tqdm(total=len(texts), desc="Creating embeddings", unit="chunk", leave=False) as pbar:
                    embeddings = []
                    for i, text in enumerate(texts):
                        embedding = self.embedding_service.embed_batch([text])
                        if embedding:
                            embeddings.extend(embedding)
                        pbar.update(1)
                        pbar.set_postfix({'completed': i+1, 'total': len(texts)})
                    return embeddings
            else:
                # For single chunk, create embedding without progress bar
                embeddings = self.embedding_service.embed_batch(texts)
                return embeddings
                
        except Exception as e:
            self.logger.error(f"Error creating embeddings: {e}")
            return []

    async def _store_embeddings(
        self, 
        chunks: List[Dict[str, Any]], 
        embeddings: List[List[float]]
    ) -> int:
        """Store embeddings in vector database."""
        try:
            vector_points = []
            
            for chunk, embedding in zip(chunks, embeddings):
                point = VectorPoint(
                    id=chunk["chunk_id"],
                    vector=embedding,
                    metadata=chunk["metadata"]
                )
                vector_points.append(point)
            
            success = self.vector_db.upsert_points(vector_points)
            
            if success:
                return len(vector_points)
            else:
                return 0
                
        except Exception as e:
            self.logger.error(f"Error storing embeddings: {e}")
            return 0

    async def _store_embeddings_with_tracking(
        self, 
        chunks: List[Dict[str, Any]], 
        embeddings: List[List[float]],
        chunk_ids: List[int],
        document_id: int
    ) -> int:
        """
        Store embeddings in vector database and track in SQLite with progress tracking.
        Also saves JSON files for each vector chunk.
        
        Args:
            chunks: List of text chunks
            embeddings: List of embedding vectors
            chunk_ids: List of SQLite chunk IDs
            document_id: SQLite document ID
            
        Returns:
            Number of embeddings successfully stored
        """
        try:
            self.logger.info(f"Storing {len(embeddings)} embeddings with tracking...")
            
            # Create chunks_vectordb directory if it doesn't exist
            chunks_vectordb_dir = Path("chunks_vectordb")
            chunks_vectordb_dir.mkdir(exist_ok=True)
            
            # Record vectorization start
            self.database_service._record_processing_stage(
                document_id, 'vectorization', started=True
            )
            
            stored_count = 0
            vector_points = []
            
            # Use progress bar for vector processing
            with tqdm(total=len(embeddings), desc="Processing vectors", unit="vector", leave=False) as pbar:
                for i, (chunk, embedding, chunk_id) in enumerate(zip(chunks, embeddings, chunk_ids)):
                    try:
                        pbar.set_description(f"Processing vector {i+1}/{len(embeddings)}")
                        
                        # Create vector point with metadata - merge chunk metadata with additional fields
                        chunk_metadata = chunk.get("metadata", {})
                        
                        # Create enhanced metadata combining chunk metadata with vector-specific fields
                        metadata = {
                            # Start with chunk metadata (includes date_metadata)
                            **chunk_metadata,
                            
                            # Add/override vector-specific fields
                            "filename": chunk.get("metadata", {}).get("filename", "unknown"),
                            "chunk_index": i,
                            "chunk_id": str(chunk_id),  # SQLite chunk ID
                            "document_id": str(document_id),  # SQLite document ID
                            "content": chunk["content"],
                            "content_preview": chunk["content"][:100] + "..." if len(chunk["content"]) > 100 else chunk["content"],
                            "chunk_size": len(chunk["content"]),
                            "source": chunk.get("metadata", {}).get("filename", "unknown"),  # For compatibility
                            "processing_timestamp": time.time(),
                            "file_size_mb": len(chunk["content"]) / (1024 * 1024)
                        }
                        
                        # Generate unique vector ID (Qdrant requires integer or UUID)
                        # Use a hash-based approach to create a unique integer ID
                        import hashlib
                        id_string = f"{document_id}_{chunk_id}_{i}"
                        vector_id = int(hashlib.md5(id_string.encode()).hexdigest()[:8], 16)
                        
                        vector_point = VectorPoint(
                            id=vector_id,
                            vector=embedding,
                            metadata=metadata
                        )
                        
                        vector_points.append(vector_point)
                        
                        # Save JSON file for this vector chunk
                        await self._save_chunk_json(vector_id, chunk, embedding, metadata, chunks_vectordb_dir)
                        
                        pbar.update(1)
                        pbar.set_postfix({
                            'vectors_created': len(vector_points),
                            'json_files': len(vector_points)
                        })
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to create vector point for chunk {i}: {e}")
                        pbar.update(1)
                        continue
            
            # Batch store in vector database
            if vector_points:
                self.logger.info(f"Storing {len(vector_points)} vectors in database...")
                with tqdm(total=1, desc="Storing in vector DB", leave=False) as db_pbar:
                    success = self.vector_db.upsert_points(vector_points)
                    db_pbar.update(1)
                
                if success:
                    stored_count = len(vector_points)
                    
                    # Mark chunks as vectorized in SQLite
                    with tqdm(total=len(vector_points), desc="Updating SQLite", unit="record", leave=False) as sql_pbar:
                        for point, chunk_id in zip(vector_points, chunk_ids):
                            self.database_service.mark_chunk_vectorized(chunk_id, point.id)
                            sql_pbar.update(1)
                    
                    self.logger.info(f"Successfully stored {stored_count} embeddings with tracking")
                    self.logger.info(f"JSON files saved in {chunks_vectordb_dir}")
                    
                    # Record successful vectorization
                    self.database_service._record_processing_stage(
                        document_id, 'vectorization', success=True,
                        metadata={'vectors_stored': stored_count}
                    )
                else:
                    self.logger.error("Failed to store vectors in database")
                    # Record failed vectorization
                    self.database_service._record_processing_stage(
                        document_id, 'vectorization', success=False,
                        error_message="Failed to store vectors in database"
                    )
            
            return stored_count
            
        except Exception as e:
            self.logger.error(f"Failed to store embeddings with tracking: {e}")
            # Record failed vectorization
            self.database_service._record_processing_stage(
                document_id, 'vectorization', success=False,
                error_message=str(e)
            )
            return 0

    async def _save_chunk_json(self, vector_id: int, chunk: Dict[str, Any], 
                              embedding: List[float], metadata: Dict[str, Any], 
                              output_dir: Path):
        """Save individual chunk data as JSON file."""
        try:
            # Create JSON data for the chunk
            chunk_data = {
                "vector_id": vector_id,
                "chunk_info": {
                    "content": chunk["content"],
                    "chunk_id": chunk.get("chunk_id", "unknown"),
                    "filename": chunk.get("metadata", {}).get("filename", "unknown"),
                    "chunk_size": len(chunk["content"]),
                    "chunk_index": metadata.get("chunk_index", 0)
                },
                "metadata": metadata,
                "embedding": {
                    "vector": embedding,
                    "dimension": len(embedding),
                    "model": config.EMBEDDING_MODEL if hasattr(config, 'EMBEDDING_MODEL') else "unknown"
                },
                "processing_info": {
                    "timestamp": metadata.get("processing_timestamp", time.time()),
                    "document_id": metadata.get("document_id", "unknown")
                }
            }
            
            # Save as JSON file
            json_file = output_dir / f"chunk_{vector_id}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(chunk_data, f, indent=2, ensure_ascii=False)
            
            self.logger.debug(f"Saved chunk JSON: {json_file}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save chunk JSON for vector {vector_id}: {e}")

    async def get_processing_stats(self) -> ProcessingStats:
        """Get current processing statistics."""
        return ProcessingStats(
            total_documents_processed=self.stats["total_documents_processed"],
            total_chunks_created=self.stats["total_chunks_created"],
            total_characters_processed=self.stats["total_characters_processed"],
            total_embeddings_created=self.stats["total_embeddings_created"],
            processing_errors=self.stats["processing_errors"],
            supported_formats=["PDF", "DOCX", "XLSX", "TXT", "CSV", "PPTX"],
            processing_method='PyMuPDF_EasyOCR_page_based_block_analysis',
            analysis_library='PyMuPDF',
            ocr_library='EasyOCR',
            total_blocks_extracted=self.stats["total_blocks_extracted"],
            total_images_extracted=self.stats["total_images_extracted"],
            total_ocr_applied=self.stats["total_ocr_applied"]
        )

    async def list_stored_documents(self) -> DocumentListResponse:
        """
        List all documents stored in the vector database.
        
        Returns:
            DocumentListResponse containing unique documents and their metadata
        """
        try:
            self.logger.info("Retrieving all documents from vector database...")
            
            # Get all points from vector database
            all_points = self.vector_db.get_all_points()
            
            if not all_points:
                return DocumentListResponse(
                    success=True,
                    total_documents=0,
                    total_chunks=0,
                    documents=[],
                    collection_name=config.COLLECTION_NAME
                )
            
            # Group chunks by document filename
            documents_data = {}
            
            for point in all_points:
                metadata = point.get('metadata', {})
                filename = metadata.get('source', 'unknown')
                
                if filename not in documents_data:
                    documents_data[filename] = {
                        'filename': filename,
                        'file_extension': self._extract_file_extension(filename),
                        'file_size_mb': metadata.get('file_size_mb', 0.0),
                        'processing_timestamp': metadata.get('processing_timestamp', 0),
                        'chunk_ids': [],
                        'chunk_contents': [],
                        'chunk_indices': []
                    }
                
                # Add chunk information
                documents_data[filename]['chunk_ids'].append(point.get('id', ''))
                documents_data[filename]['chunk_contents'].append(metadata.get('content', ''))
                documents_data[filename]['chunk_indices'].append(metadata.get('chunk_index', 0))
            
            # Convert to StoredDocument objects
            stored_documents = []
            total_chunks = 0
            
            for doc_data in documents_data.values():
                # Calculate statistics
                chunk_contents = doc_data['chunk_contents']
                chunk_sizes = [len(content) for content in chunk_contents if content]
                avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
                total_content_length = sum(chunk_sizes)
                
                # Convert timestamp
                processing_timestamp = doc_data['processing_timestamp']
                if isinstance(processing_timestamp, (int, float)):
                    from datetime import datetime
                    processing_dt = datetime.fromtimestamp(processing_timestamp)
                else:
                    processing_dt = datetime.now()
                
                stored_doc = StoredDocument(
                    filename=doc_data['filename'],
                    file_extension=doc_data['file_extension'],
                    file_size_mb=doc_data['file_size_mb'],
                    total_chunks=len(doc_data['chunk_ids']),
                    processing_timestamp=processing_dt,
                    chunk_ids=doc_data['chunk_ids'],
                    avg_chunk_size=avg_chunk_size,
                    total_content_length=total_content_length
                )
                
                stored_documents.append(stored_doc)
                total_chunks += len(doc_data['chunk_ids'])
            
            # Sort documents by processing timestamp (most recent first)
            stored_documents.sort(key=lambda x: x.processing_timestamp, reverse=True)
            
            self.logger.info(f"Found {len(stored_documents)} documents with {total_chunks} total chunks")
            
            return DocumentListResponse(
                success=True,
                total_documents=len(stored_documents),
                total_chunks=total_chunks,
                documents=stored_documents,
                collection_name=config.COLLECTION_NAME
            )
            
        except Exception as e:
            self.logger.error(f"Error listing stored documents: {e}")
            return DocumentListResponse(
                success=False,
                total_documents=0,
                total_chunks=0,
                documents=[],
                collection_name=config.COLLECTION_NAME,
                error=str(e)
            )
    
    def _extract_file_extension(self, filename: str) -> str:
        """Extract file extension from filename."""
        try:
            return filename.split('.')[-1].lower() if '.' in filename else 'unknown'
        except Exception:
            return 'unknown'

    async def delete_document_by_filename(self, filename: str) -> DeleteDocumentResponse:
        """
        Delete all chunks associated with a specific filename.
        
        Args:
            filename: The filename of the document to delete
            
        Returns:
            DeleteDocumentResponse with deletion results
        """
        try:
            self.logger.info(f"Deleting all chunks for document: {filename}")
            
            # Use the metadata filter to find chunks with matching source filename
            metadata_filter = {"source": filename}
            
            # Delete points from vector database
            deleted_count = self.vector_db.delete_points_by_metadata(metadata_filter)
            
            if deleted_count == 0:
                return DeleteDocumentResponse(
                    success=True,
                    filename=filename,
                    chunks_deleted=0,
                    collection_name=config.COLLECTION_NAME,
                    error="No chunks found for the specified filename"
                )
            elif deleted_count == -1:
                # Pinecone case - successful deletion but unknown count
                return DeleteDocumentResponse(
                    success=True,
                    filename=filename,
                    chunks_deleted=-1,  # Unknown count
                    collection_name=config.COLLECTION_NAME
                )
            else:
                # Update statistics
                self.stats["total_chunks_created"] = max(0, self.stats["total_chunks_created"] - deleted_count)
                self.stats["total_embeddings_created"] = max(0, self.stats["total_embeddings_created"] - deleted_count)
                
                self.logger.info(f"Successfully deleted {deleted_count} chunks for document: {filename}")
                
                return DeleteDocumentResponse(
                    success=True,
                    filename=filename,
                    chunks_deleted=deleted_count,
                    collection_name=config.COLLECTION_NAME
                )
                
        except Exception as e:
            self.logger.error(f"Error deleting document {filename}: {e}")
            return DeleteDocumentResponse(
                success=False,
                filename=filename,
                chunks_deleted=0,
                collection_name=config.COLLECTION_NAME,
                error=str(e)
            )

    async def clear_database(self) -> int:
        """Clear all documents from vector database."""
        try:
            # Get current count
            collection_info = self.vector_db.get_collection_info()
            current_count = collection_info.get("points_count", 0)
            
            # Delete and recreate collection
            if self.vector_db.collection_exists():
                self.vector_db.delete_collection()
            
            self.vector_db.create_collection(dimension=config.EMBEDDING_DIMENSION)
            
            # Reset stats
            self.stats["total_documents_processed"] = 0
            self.stats["total_chunks_created"] = 0
            self.stats["total_characters_processed"] = 0
            self.stats["total_embeddings_created"] = 0
            
            self.logger.info(f"Cleared vector database: {current_count} documents removed")
            return current_count
            
        except Exception as e:
            self.logger.error(f"Error clearing database: {e}")
            raise

    async def health_check(self) -> Dict[str, Any]:
        """Check the health of document processing services."""
        try:
            # Check vector database connection
            vector_db_status = "healthy"
            try:
                collection_info = self.vector_db.get_collection_info()
                vector_db_points = collection_info.get("points_count", 0)
            except Exception as e:
                vector_db_status = f"error: {str(e)}"
                vector_db_points = 0
            
            # Check text extraction service
            text_extraction_status = "healthy"
            
            # Check chunking service
            chunking_status = "healthy"
            
            # Check embedding service
            embedding_status = "healthy"
            try:
                # Test embedding creation with a small text
                test_embedding = await self.embedding_service.create_embedding("test")
                if not test_embedding:
                    embedding_status = "error: failed to create test embedding"
            except Exception as e:
                embedding_status = f"error: {str(e)}"
            
            return {
                "status": "healthy" if all(status == "healthy" for status in [
                    vector_db_status, text_extraction_status, chunking_status, embedding_status
                ]) else "degraded",
                "services": {
                    "vector_database": {
                        "status": vector_db_status,
                        "points_count": vector_db_points
                    },
                    "text_extraction": {"status": text_extraction_status},
                    "chunking": {"status": chunking_status},
                    "embedding": {"status": embedding_status}
                },
                "stats": self.stats
            }
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "services": {},
                "stats": self.stats
            }

    async def get_chunk_metadata(self, 
                               vector_id: Optional[str] = None,
                               filename: Optional[str] = None,
                               limit: int = 10,
                               include_content: bool = False) -> Dict[str, Any]:
        """
        Get metadata for chunks stored in the vector database.
        
        Args:
            vector_id: Specific vector ID to inspect
            filename: Filter by filename
            limit: Maximum number of chunks to return
            include_content: Include full content preview
            
        Returns:
            Dictionary with chunk metadata information
        """
        try:
            self.logger.info(f"Getting chunk metadata - vector_id: {vector_id}, filename: {filename}, limit: {limit}")
            
            # Get all points from vector database
            all_points = self.vector_db.get_all_points()
            
            if not all_points:
                return {
                    'success': True,
                    'total_chunks': 0,
                    'chunks': [],
                    'collection_name': config.COLLECTION_NAME,
                    'error': None
                }
            
            # Filter points based on criteria
            filtered_points = []
            
            for point in all_points:
                point_id = str(point.get('id', ''))
                metadata = point.get('metadata', {})
                point_filename = metadata.get('source', metadata.get('filename', 'unknown'))
                
                # Apply filters
                if vector_id and point_id != vector_id:
                    continue
                    
                if filename and filename.lower() not in point_filename.lower():
                    continue
                
                filtered_points.append(point)
            
            # Limit results
            limited_points = filtered_points[:limit]
            
            # Convert to VectorChunk objects
            chunks = []
            for point in limited_points:
                try:
                    point_id = str(point.get('id', ''))
                    metadata = point.get('metadata', {})
                    vector_data = point.get('vector', [])
                    
                    # Return complete metadata with standard fields for compatibility
                    chunk_metadata = {
                        # Include all metadata fields from vector database
                        **metadata,
                        
                        # Ensure standard fields are present for compatibility
                        'filename': metadata.get('filename', metadata.get('source', 'unknown')),
                        'chunk_index': metadata.get('chunk_index', 0),
                        'chunk_id': metadata.get('chunk_id', 'unknown'),
                        'document_id': metadata.get('document_id', 'unknown'),
                        'content_preview': metadata.get('content_preview', ''),
                        'chunk_size': metadata.get('chunk_size', 0),
                        'source': metadata.get('source', metadata.get('filename', 'unknown')),
                        'processing_timestamp': metadata.get('processing_timestamp', 0),
                        'file_size_mb': metadata.get('file_size_mb', 0.0)
                    }
                    
                    # Add extended content if requested and available
                    if include_content and 'content' in metadata:
                        chunk_metadata['content'] = metadata['content']
                    
                    chunk_info = {
                        'vector_id': point_id,
                        'metadata': chunk_metadata,
                        'vector_dimension': len(vector_data) if vector_data else 0,
                        'has_vector': bool(vector_data)
                    }
                    
                    chunks.append(chunk_info)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to process point {point.get('id', 'unknown')}: {e}")
                    continue
            
            self.logger.info(f"Found {len(chunks)} chunks matching criteria (total available: {len(all_points)})")
            
            return {
                'success': True,
                'total_chunks': len(filtered_points),
                'chunks': chunks,
                'collection_name': config.COLLECTION_NAME,
                'error': None
            }
            
        except Exception as e:
            self.logger.error(f"Error getting chunk metadata: {e}")
            return {
                'success': False,
                'total_chunks': 0,
                'chunks': [],
                'collection_name': config.COLLECTION_NAME,
                'error': str(e)
            }

    async def _apply_spell_correction_to_chunks(self, filename: str, chunks: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict]:
        """Apply spell correction to chunks with aerospace dictionary."""
        try:
            self.logger.info(f"Applying spell correction to {len(chunks)} chunks for {filename}...")
            
            corrected_chunks = []
            all_corrections = []
            total_words = 0
            corrected_words = 0
            total_dictionary_corrections = 0
            total_fuzzy_corrections = 0
            total_spellchecker_corrections = 0
            total_context_blocked = 0
            
            for i, chunk in enumerate(chunks):
                chunk_text = chunk["content"]
                
                # Apply spell correction to chunk content
                corrected_text, correction_details = self.spell_correction_service.correct_text(chunk_text)
                
                # Update chunk with corrected content
                corrected_chunk = chunk.copy()
                corrected_chunk["content"] = corrected_text
                
                # Add correction metadata
                corrected_chunk["metadata"]["spell_corrected"] = True
                corrected_chunk["metadata"]["original_content_hash"] = hashlib.md5(chunk_text.encode('utf-8')).hexdigest()[:8]
                corrected_chunks.append(corrected_chunk)
                
                # Accumulate correction statistics
                chunk_stats = correction_details.get('stats', {})
                chunk_corrections = correction_details.get('corrections', [])
                
                total_words += chunk_stats.get('total_words', 0)
                corrected_words += chunk_stats.get('corrected_words', 0)
                all_corrections.extend(chunk_corrections)
                
                # Aggregate detailed statistics
                total_dictionary_corrections += chunk_stats.get('dictionary_corrections', 0)
                total_fuzzy_corrections += chunk_stats.get('fuzzy_corrections', 0)
                total_spellchecker_corrections += chunk_stats.get('spellchecker_corrections', 0)
                total_context_blocked += chunk_stats.get('context_blocked', 0)
            
            # Create comprehensive correction details with detailed statistics
            final_correction_details = {
                'corrections': all_corrections,
                'stats': {
                    'total_words': total_words,
                    'corrected_words': corrected_words,
                    'correction_rate': corrected_words / total_words if total_words > 0 else 0,
                    'chunks_processed': len(chunks),
                    'dictionary_corrections': total_dictionary_corrections,
                    'fuzzy_corrections': total_fuzzy_corrections,
                    'spellchecker_corrections': total_spellchecker_corrections,
                    'context_blocked': total_context_blocked
                }
            }
            
            # Always save comprehensive correction comparison (even if no corrections)
            # Create combined text for comparison
            original_combined = "\n\n".join([chunk["content"] for chunk in chunks])
            corrected_combined = "\n\n".join([chunk["content"] for chunk in corrected_chunks])
            
            comparison_file = self.spell_correction_service.save_correction_comparison(
                original_text=original_combined,
                corrected_text=corrected_combined,
                corrections=all_corrections,
                filename=filename,
                stats=final_correction_details['stats']
            )
            final_correction_details['comparison_file'] = comparison_file
            
            # Log correction statistics
            self.logger.info(f"Spell correction complete for {filename} chunks:")
            self.logger.info(f"  - Total chunks: {len(chunks)}")
            self.logger.info(f"  - Total words: {total_words}")
            self.logger.info(f"  - Corrected words: {corrected_words}")
            self.logger.info(f"  - Correction rate: {final_correction_details['stats']['correction_rate']:.2%}")
            
            if all_corrections:
                self.logger.info(f"  - Sample corrections:")
                for correction in all_corrections[:5]:  # Show first 5 corrections
                    self.logger.info(f"    '{correction['original']}' --> '{correction['corrected']}' ({correction['method']})")
            
            return corrected_chunks, final_correction_details
            
        except Exception as e:
            self.logger.error(f"Error applying spell correction to chunks for {filename}: {e}")
            # Return original chunks if correction fails
            return chunks, {
                "corrections": [],
                "stats": {"total_words": 0, "corrected_words": 0, "correction_rate": 0, "chunks_processed": len(chunks)},
                "error": str(e)
            }

    # Document management methods
    async def get_documents(self, skip: int = 0, limit: int = 100) -> DocumentListResponse:
        """Get list of documents from SQLite database."""
        try:
            documents, total = self.database_service.get_documents(skip=skip, limit=limit)
            
            # Convert to StoredDocument objects
            stored_documents = []
            for doc in documents:
                stored_doc = StoredDocument(
                    id=doc['id'],
                    filename=doc['filename'],
                    file_type=doc['file_type'],
                    file_size=doc['file_size'],
                    upload_date=doc['upload_date'],
                    page_count=doc.get('page_count'),
                    chunk_count=doc.get('chunk_count', 0),
                    vectorized_chunks=doc.get('vectorized_chunks', 0),
                    extraction_method=doc.get('extraction_method', 'unknown')
                )
                stored_documents.append(stored_doc)
            
            return DocumentListResponse(
                documents=stored_documents,
                total=total,
                skip=skip,
                limit=limit
            )
            
        except Exception as e:
            self.logger.error(f"Error getting documents: {e}")
            return DocumentListResponse(documents=[], total=0, skip=skip, limit=limit)

    async def delete_document(self, request: DeleteDocumentRequest) -> DeleteDocumentResponse:
        """Delete a document and its associated data."""
        try:
            self.logger.info(f"Deleting document: {request.filename}")
            
            # Get document info first
            doc_info = self.database_service.get_document_by_filename(request.filename)
            if not doc_info:
                return DeleteDocumentResponse(
                    success=False,
                    message=f"Document '{request.filename}' not found"
                )
            
            document_id = doc_info['id']
            
            # Get chunks for vector deletion
            chunks = self.database_service.get_document_chunks(document_id)
            
            # Delete from vector database
            vector_ids = [f"{document_id}_{chunk['id']}" for chunk in chunks]
            if vector_ids:
                try:
                    self.vector_db.delete_vectors(vector_ids)
                    self.logger.info(f"Deleted {len(vector_ids)} vectors from vector database")
                except Exception as e:
                    self.logger.warning(f"Error deleting vectors: {e}")
            
            # Delete from SQLite database
            deleted_chunks = self.database_service.delete_document(document_id)
            
            self.logger.info(f"Successfully deleted document '{request.filename}' with {deleted_chunks} chunks")
            
            return DeleteDocumentResponse(
                success=True,
                message=f"Successfully deleted document '{request.filename}' and {deleted_chunks} associated chunks"
            )
            
        except Exception as e:
            self.logger.error(f"Error deleting document: {e}")
            return DeleteDocumentResponse(
                success=False,
                message=f"Error deleting document: {str(e)}"
            )

    async def get_service_stats(self) -> Dict[str, Any]:
        """Get comprehensive service statistics."""
        try:
            # Get database statistics
            db_stats = self.database_service.get_database_stats()
            
            # Get vector database info
            vector_info = {}
            try:
                if self.vector_db:
                    collection_info = self.vector_db.get_collection_info()
                    vector_info = {
                        "collection_name": config.COLLECTION_NAME,
                        "vectors_count": collection_info.get("vectors_count", 0),
                        "dimension": collection_info.get("config", {}).get("params", {}).get("vectors", {}).get("size", config.EMBEDDING_DIMENSION),
                        "distance_metric": collection_info.get("config", {}).get("params", {}).get("vectors", {}).get("distance", "cosine")
                    }
            except Exception as e:
                self.logger.warning(f"Error getting vector database info: {e}")
                vector_info = {"error": str(e)}
            
            # Combine all statistics
            comprehensive_stats = {
                "service_stats": self.stats,
                "database_stats": db_stats,
                "vector_database_info": vector_info,
                "configuration": {
                    "embedding_model": config.EMBEDDING_MODEL,
                    "embedding_dimension": config.EMBEDDING_DIMENSION,
                    "chunk_size": config.CHUNK_SIZE,
                    "chunk_overlap": config.CHUNK_OVERLAP,
                    "min_chunk_size": config.MIN_CHUNK_SIZE,
                    "max_file_size": config.MAX_FILE_SIZE
                },
                "enhanced_features": {
                    "block_based_analysis": True,
                    "coordinate_extraction": True,
                    "ocr_capabilities": EASYOCR_AVAILABLE,
                    "ocr_library": "EasyOCR" if EASYOCR_AVAILABLE else "None",
                    "spell_correction": True,
                    "date_extraction": True,
                    "semantic_chunking": True
                }
            }
            
            return comprehensive_stats
            
        except Exception as e:
            self.logger.error(f"Error getting service stats: {e}")
            return {
                "error": str(e),
                "service_stats": self.stats
            }

    def _validate_and_fix_coordinates(self, bbox: Tuple[float, float, float, float], page_rect: fitz.Rect) -> Optional[Tuple[float, float, float, float]]:
        """
        Validate and fix bounding box coordinates to ensure they're valid for OCR processing.
        
        Args:
            bbox: Tuple of (x0, y0, x1, y1) coordinates
            page_rect: The page rectangle for boundary checking
            
        Returns:
            Fixed coordinates tuple or None if coordinates are invalid
        """
        try:
            x0, y0, x1, y1 = bbox
            
            # Check for NaN or infinite values
            if any(not isinstance(coord, (int, float)) or 
                   coord != coord or  # NaN check
                   abs(coord) == float('inf') for coord in [x0, y0, x1, y1]):
                self.logger.debug(f"Invalid coordinate values: {bbox}")
                return None
            
            # Ensure x0 < x1 and y0 < y1
            if x0 >= x1:
                x0, x1 = min(x0, x1), max(x0, x1)
                if x0 == x1:
                    self.logger.debug(f"Zero width rectangle: {bbox}")
                    return None
                    
            if y0 >= y1:
                y0, y1 = min(y0, y1), max(y0, y1)
                if y0 == y1:
                    self.logger.debug(f"Zero height rectangle: {bbox}")
                    return None
            
            # Check maximum dimensions BEFORE clipping (prevent memory issues)
            original_width = x1 - x0
            original_height = y1 - y0
            max_dimension = 5000
            if original_width > max_dimension or original_height > max_dimension:
                self.logger.debug(f"Original rectangle too large: {bbox} -> width={original_width}, height={original_height}")
                return None
            
            # Clip coordinates to page boundaries
            page_x0, page_y0, page_x1, page_y1 = page_rect.x0, page_rect.y0, page_rect.x1, page_rect.y1
            
            x0 = max(x0, page_x0)
            y0 = max(y0, page_y0)
            x1 = min(x1, page_x1)
            y1 = min(y1, page_y1)
            
            # Check if coordinates are still within page after clipping
            if x0 >= x1 or y0 >= y1:
                self.logger.debug(f"Coordinates outside page bounds after clipping: {bbox}")
                return None
            
            # Ensure minimum dimensions for OCR (at least 10 pixels)
            min_dimension = 10
            if (x1 - x0) < min_dimension or (y1 - y0) < min_dimension:
                self.logger.debug(f"Rectangle too small for OCR: {bbox} -> width={x1-x0}, height={y1-y0}")
                return None
            
            return (float(x0), float(y0), float(x1), float(y1))
            
        except Exception as e:
            self.logger.debug(f"Error validating coordinates {bbox}: {e}")
            return None

    def _preprocess_image_for_ocr(self, image: Image.Image) -> Optional[Image.Image]:
        """
        Preprocess image for OCR to ensure it's suitable for processing.
        
        Args:
            image: PIL Image to preprocess
            
        Returns:
            Preprocessed image or None if image is invalid
        """
        try:
            # Check image dimensions
            width, height = image.size
            if width < 10 or height < 10:
                self.logger.debug(f"Image too small for OCR: {width}x{height}")
                return None
            
            if width > 5000 or height > 5000:
                self.logger.debug(f"Image too large for OCR: {width}x{height}")
                return None
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Check if image is completely black or white (likely processing error)
            extrema = image.getextrema()
            if all(min_val == max_val for min_val, max_val in extrema):
                # Allow white backgrounds but reject completely black images
                if all(min_val == 0 for min_val, max_val in extrema):
                    self.logger.debug("Image is completely black")
                    return None
                # Note: White/uniform colored images are allowed for OCR processing
            
            return image
            
        except Exception as e:
            self.logger.debug(f"Error preprocessing image: {e}")
            return None

# Maintain backward compatibility with original DocumentService
DocumentService = EnhancedDocumentService 