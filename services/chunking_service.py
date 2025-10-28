#!/usr/bin/env python3
"""
Chunking service for text processing.
Extracts chunking logic from pipeline for API use.
"""

import time
import hashlib
import logging
import re
from typing import Dict, Any, List, Optional

# Import configuration
from core.config import config

class ChunkingService:
    """Service for creating text chunks from documents."""
    
    def __init__(self):
        """Initialize the chunking service."""
        self.logger = logging.getLogger(__name__)
        
        # Load chunking configuration from config
        self.chunk_size = config.CHUNK_SIZE
        self.overlap_size = config.CHUNK_OVERLAP
        self.min_chunk_size = config.MIN_CHUNK_SIZE
        self.max_chunk_size = config.MAX_CHUNK_SIZE
        
        self.logger.info(f"ChunkingService initialized with config values:")
        self.logger.info(f"  chunk_size={self.chunk_size}, min_size={self.min_chunk_size}")
        self.logger.info(f"  max_size={self.max_chunk_size}, overlap={self.overlap_size}")
        
    async def create_chunks(self, filename: str, text_content: str, date_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Create semantic chunks from text content with date metadata.
        
        Args:
            filename: Name of the source file
            text_content: Text content to chunk
            date_metadata: Optional date metadata to include in chunks
            
        Returns:
            List of chunk dictionaries with content and metadata
        """
        self.logger.info(f"Creating chunks for {filename} (length: {len(text_content):,} chars)")
        
        try:
            # Clean and normalize text
            cleaned_text = self._clean_text(text_content)
            
            if not cleaned_text.strip():
                self.logger.warning(f"No meaningful text content found in {filename}")
                return []
            
            # Log chunk configuration
            self.logger.info(
                f"Chunk config - Size: {self.chunk_size}, Overlap: {self.overlap_size}, "
                f"Min: {self.min_chunk_size}, Max: {self.max_chunk_size}"
            )
            
            # Create semantic chunks
            chunks = self._create_semantic_chunks(cleaned_text, filename, date_metadata)
            
            # Handle small chunks that might have been skipped
            if not chunks and cleaned_text:
                self.logger.info(f"Creating minimal chunk for small document: {filename}")
                chunks = [self._create_chunk_data(cleaned_text, filename, 0, date_metadata)]
            
            # Log results
            chunk_sizes = [len(chunk["content"]) for chunk in chunks]
            self.logger.info(
                f"Created {len(chunks)} chunks for {filename} - "
                f"Avg size: {sum(chunk_sizes)/len(chunk_sizes):.0f}, "
                f"Min: {min(chunk_sizes) if chunk_sizes else 0}, "
                f"Max: {max(chunk_sizes) if chunk_sizes else 0}"
            )
            
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error creating chunks for {filename}: {e}")
            return []
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for chunking - PRESERVE CONTENT while removing duplicates."""
        # Only normalize excessive whitespace, don't remove all formatting
        text = re.sub(r'[ \t]+', ' ', text)  # Only normalize spaces/tabs, preserve newlines
        
        # Remove page markers but keep the structure
        text = re.sub(r'=== PAGE \d+ ===', '\n\n', text)
        
        # Only normalize excessive line breaks (3+ becomes 2)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove duplicate consecutive lines to prevent repetitive content
        lines = text.split('\n')
        deduplicated_lines = []
        seen_content = set()
        
        for line in lines:
            stripped_line = line.strip()
            
            # Skip empty lines for deduplication check but preserve them
            if not stripped_line:
                deduplicated_lines.append(line)
                continue
            
            # Create a normalized version for comparison (remove extra spaces, lower case)
            normalized_content = re.sub(r'\s+', ' ', stripped_line.lower())
            
            # Skip if this exact content was already seen recently (within last 5 lines)
            recent_content = [re.sub(r'\s+', ' ', l.strip().lower()) for l in deduplicated_lines[-5:] if l.strip()]
            
            if normalized_content not in recent_content:
                deduplicated_lines.append(line)
                seen_content.add(normalized_content)
            else:
                # Found duplicate - log it for debugging
                self.logger.debug(f"Removed duplicate line: {stripped_line[:50]}...")
        
        return '\n'.join(deduplicated_lines).strip()
    
    def _create_semantic_chunks(self, text_content: str, filename: str, date_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Create chunks using semantic boundaries (paragraphs, sentences).
        ENSURE NO CONTENT IS LOST.
        """
        chunks = []
        
        # First, try to split by paragraphs
        paragraphs = [p.strip() for p in text_content.split('\n\n') if p.strip()]
        
        if not paragraphs:
            # Fallback: split by newlines
            paragraphs = [p.strip() for p in text_content.split('\n') if p.strip()]
        
        current_chunk = ""
        chunk_index = 0
        
        for paragraph in paragraphs:
            # Check if adding this paragraph would exceed max size
            potential_chunk = current_chunk + ("\n\n" if current_chunk else "") + paragraph
            
            if len(potential_chunk) <= self.max_chunk_size:
                # Add paragraph to current chunk
                current_chunk = potential_chunk
            else:
                # Current chunk is ready, save it if it has content
                if current_chunk:
                    chunk_data = self._create_chunk_data(current_chunk, filename, chunk_index, date_metadata)
                    chunks.append(chunk_data)
                    chunk_index += 1
                
                # Start new chunk with current paragraph
                if len(paragraph) <= self.max_chunk_size:
                    current_chunk = paragraph
                else:
                    # Paragraph is too large, split it by sentences
                    sentence_chunks = self._split_large_text(paragraph, filename, chunk_index, date_metadata)
                    chunks.extend(sentence_chunks)
                    chunk_index += len(sentence_chunks)
                    current_chunk = ""
        
        # Handle the last chunk - ALWAYS PRESERVE, even if small
        if current_chunk:
            if len(current_chunk) >= self.min_chunk_size or not chunks:
                # Create chunk if it's big enough OR if it's the only content
                chunk_data = self._create_chunk_data(current_chunk, filename, chunk_index, date_metadata)
                chunks.append(chunk_data)
            else:
                # Small final chunk - combine with previous chunk if possible
                if chunks:
                    last_chunk = chunks[-1]
                    combined_content = last_chunk["content"] + "\n\n" + current_chunk
                    if len(combined_content) <= self.max_chunk_size:
                        last_chunk["content"] = combined_content
                        last_chunk["metadata"]["content"] = combined_content
                        last_chunk["metadata"]["chunk_size"] = len(combined_content)
                        self.logger.debug(f"Combined small final chunk with previous chunk")
                    else:
                        # Can't combine - create separate chunk anyway to preserve content
                        chunk_data = self._create_chunk_data(current_chunk, filename, chunk_index, date_metadata)
                        chunks.append(chunk_data)
                        self.logger.warning(f"Small chunk created to preserve content: {len(current_chunk)} chars")
                else:
                    # No previous chunks - create it anyway
                    chunk_data = self._create_chunk_data(current_chunk, filename, chunk_index, date_metadata)
                    chunks.append(chunk_data)
        
        return chunks
    
    def _split_large_text(self, text: str, filename: str, start_index: int, date_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Split large text by sentences when it exceeds max chunk size.
        ENSURE NO CONTENT IS LOST.
        """
        chunks = []
        
        # Split by sentences using multiple delimiters
        sentences = re.split(r'[.!?]+\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        current_chunk = ""
        chunk_index = start_index
        
        for sentence in sentences:
            potential_chunk = current_chunk + (" " if current_chunk else "") + sentence
            
            if len(potential_chunk) <= self.chunk_size:
                current_chunk = potential_chunk
            else:
                # Save current chunk if it has content
                if current_chunk:
                    chunk_data = self._create_chunk_data(current_chunk, filename, chunk_index, date_metadata)
                    chunks.append(chunk_data)
                    chunk_index += 1
                
                # Start new chunk with current sentence
                current_chunk = sentence
        
        # Handle the last chunk - ALWAYS PRESERVE
        if current_chunk:
            if len(current_chunk) >= self.min_chunk_size or not chunks:
                chunk_data = self._create_chunk_data(current_chunk, filename, chunk_index, date_metadata)
                chunks.append(chunk_data)
            else:
                # Small final chunk
                if chunks:
                    # Try to combine with previous chunk
                    last_chunk = chunks[-1]
                    combined_content = last_chunk["content"] + " " + current_chunk
                    if len(combined_content) <= self.max_chunk_size:
                        last_chunk["content"] = combined_content
                        last_chunk["metadata"]["content"] = combined_content
                        last_chunk["metadata"]["chunk_size"] = len(combined_content)
                    else:
                        # Can't combine - create separate chunk to preserve content
                        chunk_data = self._create_chunk_data(current_chunk, filename, chunk_index, date_metadata)
                        chunks.append(chunk_data)
                        self.logger.warning(f"Small sentence chunk created to preserve content: {len(current_chunk)} chars")
                else:
                    # No previous chunks - create it anyway
                    chunk_data = self._create_chunk_data(current_chunk, filename, chunk_index, date_metadata)
                    chunks.append(chunk_data)
        
        return chunks
    
    def _create_chunk_data(self, content: str, filename: str, chunk_index: int, date_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a chunk data structure with metadata."""
        # Generate UUID for this chunk
        chunk_uuid = self._generate_chunk_id(content, filename, chunk_index)
        
        chunk_data = {
            "chunk_id": chunk_uuid,
            "content": content,
            "filename": filename,
            "chunk_index": chunk_index,
            "metadata": {
                "source": filename,
                "chunk_index": chunk_index,
                "processing_timestamp": time.time(),
                "chunk_uuid": chunk_uuid,
                "content": content,  # Include content in metadata for vector DB
                "chunk_size": len(content),
                "content_preview": content[:100] + "..." if len(content) > 100 else content,
                "date_metadata": date_metadata
            }
        }
        
        return chunk_data
    
    def _generate_chunk_id(self, chunk_content: str, filename: str, chunk_index: int) -> str:
        """
        Generate a consistent UUID for a chunk based on its content and metadata.
        
        Args:
            chunk_content: The text content of the chunk
            filename: The source filename
            chunk_index: The index of the chunk within the file
            
        Returns:
            A UUID string that will be consistent for the same content
        """
        # Create a deterministic hash from content + metadata
        hash_input = f"{filename}_{chunk_index}_{chunk_content}".encode('utf-8')
        content_hash = hashlib.md5(hash_input).hexdigest()
        
        # Convert hash to UUID format (deterministic UUID)
        uuid_str = f"{content_hash[:8]}-{content_hash[8:12]}-{content_hash[12:16]}-{content_hash[16:20]}-{content_hash[20:32]}"
        
        return uuid_str 