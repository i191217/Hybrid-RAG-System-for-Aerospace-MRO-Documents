#!/usr/bin/env python3
"""
Hybrid text extraction service that combines PyMuPDF and OCR.
Integrates hybrid_content_extraction.py functionality into the pipeline.
"""

import os
import sys
import logging
import io
import re
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from difflib import SequenceMatcher
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import configuration
from core.config import config

# Optional imports with fallbacks
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    from PIL import Image, ImageEnhance
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Configuration constants
MAX_COLS = 3
GAP_THRESHOLD = 0.2
IMAGE_DPI_SCALE = 2
OCR_CONFIDENCE_THRESHOLD = 0.3
SIMILARITY_THRESHOLD = 0.85


class HybridTextExtractionService:
    """
    Enhanced text extraction service that combines PyMuPDF and OCR.
    Follows the same interface as TextExtractionService for easy integration.
    """
    
    def __init__(self, debug=False, ocr_threshold=0.3):
        """Initialize the hybrid text extraction service."""
        self.logger = logging.getLogger(__name__)
        self.debug = debug
        self.ocr_threshold = ocr_threshold
        self.ocr_reader = None
        
    async def initialize(self):
        """Initialize OCR components."""
        try:
            # Initialize EasyOCR reader if available
            if EASYOCR_AVAILABLE:
                try:
                    self.ocr_reader = easyocr.Reader(['en'], gpu=False)
                    self.logger.info("EasyOCR reader initialized successfully for hybrid extraction")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize EasyOCR reader: {e}")
            else:
                self.logger.warning("EasyOCR not available. PDF text extraction will use PyMuPDF only.")
                
        except Exception as e:
            self.logger.error(f"Error initializing hybrid text extraction service: {e}")
            raise

    async def cleanup(self):
        """Cleanup resources."""
        # No specific cleanup needed for current implementation
        pass

    async def extract_text(self, file_path: Path, extension: str) -> str:
        """
        Extract text from a file based on its extension.
        Main interface method compatible with TextExtractionService.
        
        Args:
            file_path: Path to the file
            extension: File extension (with dot, e.g., '.pdf')
            
        Returns:
            Extracted text content
        """
        self.logger.info(f"Hybrid extraction from: {file_path.name}")
        self.logger.info(f"File type: {extension}")
        
        try:
            if extension == '.pdf':
                return await self._extract_from_pdf_hybrid(file_path)
            else:
                # For non-PDF files, fall back to basic extraction
                # (This could be enhanced later to use OCR on images, etc.)
                return await self._extract_from_non_pdf(file_path, extension)
                
        except Exception as e:
            self.logger.error(f"Failed to extract text from {file_path.name}: {e}")
            return f"Error extracting text from {file_path.name}: {str(e)}"

    async def _extract_from_pdf_hybrid(self, file_path: Path) -> str:
        """
        Extract text from PDF using hybrid approach (PyMuPDF + OCR).
        This is the main enhancement over the original TextExtractionService.
        """
        try:
            # Check required libraries
            if not PYMUPDF_AVAILABLE:
                self.logger.error("PyMuPDF not available. Install: pip install PyMuPDF")
                return "Error: PyMuPDF not available"
            
            if not SKLEARN_AVAILABLE:
                self.logger.warning("scikit-learn not available. Column detection disabled.")
            
            self.logger.info("Starting hybrid PDF extraction (PyMuPDF + OCR)...")
            
            doc = fitz.open(str(file_path))
            total_pages = len(doc)
            page_results = []
            
            for page_num in range(1, total_pages + 1):
                page = doc[page_num - 1]
                
                # Process the page with hybrid extraction
                result = await self._process_page_hybrid(page, page_num, file_path.stem)
                page_results.append(result['text'])
                
                self.logger.debug(f"Page {page_num}: {len(result['text'])} characters extracted")
            
            doc.close()
            
            # Combine all pages
            combined_text = '\n\n'.join([f"=== PAGE {i+1} ===\n{text}" for i, text in enumerate(page_results) if text.strip()])
            
            self.logger.info(f"Hybrid PDF extraction complete:")
            self.logger.info(f"- Total pages: {total_pages}")
            self.logger.info(f"- Total characters: {len(combined_text)}")
            
            return combined_text
            
        except Exception as e:
            self.logger.error(f"Hybrid PDF extraction failed: {e}")
            return f"Error in hybrid PDF extraction: {str(e)}"

    async def _process_page_hybrid(self, page, page_num: int, pdf_name: str) -> Dict:
        """Process a single page with both PyMuPDF and OCR."""
        
        # Extract text using PyMuPDF with column detection
        pymupdf_text = self._extract_pymupdf_text(page)
        self.logger.debug(f"PyMuPDF extracted {len(pymupdf_text)} characters from page {page_num}")
        
        # Extract text using OCR if available
        ocr_text = ""
        if EASYOCR_AVAILABLE and self.ocr_reader:
            ocr_text = await self._extract_ocr_text(page, page_num)
            self.logger.debug(f"OCR extracted {len(ocr_text)} characters from page {page_num}")
        
        # Remove duplicates and combine
        combined_text = self._remove_duplicates(pymupdf_text, ocr_text)
        
        # Clean the final text
        final_text = self._clean_text(combined_text)
        
        return {
            'page_num': page_num,
            'pymupdf_length': len(pymupdf_text),
            'ocr_length': len(ocr_text),
            'final_length': len(final_text),
            'text': final_text
        }

    def _extract_pymupdf_text(self, page) -> str:
        """Extract text using PyMuPDF with column detection."""
        blocks = self._extract_text_blocks(page)
        if not blocks:
            return ""
        
        if SKLEARN_AVAILABLE:
            col_count = self._auto_detect_columns(blocks)
            
            if col_count > 1:
                text = self._sort_blocks_into_columns(blocks, col_count)
            else:
                blocks.sort(key=lambda b: b[1])  # single-column top→bottom (sort by y0)
                text = "\n".join(b[4] for b in blocks)
        else:
            # Fallback without column detection
            blocks.sort(key=lambda b: b[1])  # sort by y0 (top to bottom)
            text = "\n".join(b[4] for b in blocks)
        
        return text.strip()

    def _extract_text_blocks(self, page):
        """Extract text blocks using PyMuPDF."""
        blocks = page.get_text("blocks")
        text_blocks = []
        for x0, y0, x1, y1, text, _, btype in blocks:
            txt = text.strip()
            if btype == 0 and txt:
                text_blocks.append((x0, y0, x1, y1, txt))
        return text_blocks

    def _auto_detect_columns(self, blocks):
        """Auto-detect column layout."""
        if not SKLEARN_AVAILABLE or len(blocks) < 2:
            return 1

        page_width = max(b[2] for b in blocks) - min(b[0] for b in blocks)
        mids = np.array([[(x0 + x1) / 2] for x0, _, x1, _, _ in blocks])

        for k in range(min(MAX_COLS, len(blocks)), 1, -1):
            km = KMeans(n_clusters=k, random_state=0).fit(mids)
            centers = sorted(c[0] for c in km.cluster_centers_)
            gaps = np.diff(centers)
            if gaps.size and np.min(gaps) >= GAP_THRESHOLD * page_width:
                return k

        return 1

    def _sort_blocks_into_columns(self, blocks, n_cols):
        """Sort text blocks into columns and return structured text."""
        if not SKLEARN_AVAILABLE:
            blocks.sort(key=lambda b: b[1])
            return "\n".join(b[4] for b in blocks)
            
        mids = np.array([[(x0 + x1) / 2] for x0, _, x1, _, _ in blocks])
        km = KMeans(n_clusters=n_cols, random_state=0).fit(mids)
        centers = km.cluster_centers_.flatten()
        col_order = np.argsort(centers)

        cols_text = []
        for col in col_order:
            col_blocks = [b for b, lbl in zip(blocks, km.labels_) if lbl == col]
            col_blocks.sort(key=lambda b: b[1])  # sort by y0 (top) ascending - top to bottom
            cols_text.append("\n".join(b[4] for b in col_blocks))

        return "\n\n".join(cols_text)

    async def _extract_ocr_text(self, page, page_num: int) -> str:
        """Extract text using OCR on the entire page."""
        if not self.ocr_reader or not PIL_AVAILABLE:
            return ""
        
        try:
            # Convert page to high-resolution image
            mat = fitz.Matrix(3, 3)  # 3x zoom for better OCR accuracy
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to numpy array
            img_data = pix.tobytes("ppm")
            pil_image = Image.open(io.BytesIO(img_data))
            
            if CV2_AVAILABLE:
                page_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                processed_image = self._preprocess_image_for_ocr(page_image)
                rgb_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            else:
                # Fallback without OpenCV
                rgb_image = np.array(pil_image)
            
            # Perform OCR with detailed results
            self.logger.debug(f"Running OCR on page {page_num}...")
            ocr_results = self.ocr_reader.readtext(rgb_image, detail=True)
            
            # Organize OCR results by spatial layout
            organized_text = self._organize_ocr_results(ocr_results)
            
            return organized_text.strip()
            
        except Exception as e:
            self.logger.warning(f"OCR failed for page {page_num}: {e}")
            return ""

    def _preprocess_image_for_ocr(self, image_array):
        """Preprocess image to improve OCR accuracy."""
        if not PIL_AVAILABLE or not CV2_AVAILABLE:
            return image_array
            
        # Convert to PIL for easier processing
        pil_image = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(1.2)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(pil_image)
        pil_image = enhancer.enhance(1.1)
        
        # Convert back to numpy array
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def _organize_ocr_results(self, ocr_results):
        """Organize OCR results by spatial layout to preserve structure."""
        if not ocr_results:
            return ""
        
        # Filter by confidence and extract useful information
        valid_results = []
        for bbox, text, confidence in ocr_results:
            if confidence > self.ocr_threshold:
                text = text.strip()
                if text:
                    # Get bounding box coordinates
                    x1, y1 = bbox[0]  # top-left
                    x2, y2 = bbox[2]  # bottom-right
                    
                    # Calculate center and dimensions
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    width = x2 - x1
                    height = y2 - y1
                    
                    valid_results.append({
                        'text': text,
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                        'center_x': center_x, 'center_y': center_y,
                        'width': width, 'height': height,
                        'confidence': confidence
                    })
        
        if not valid_results:
            return ""
        
        # Sort by vertical position first (top to bottom)
        valid_results.sort(key=lambda x: x['center_y'])
        
        # Group into lines based on vertical proximity
        lines = []
        current_line = [valid_results[0]]
        
        for i in range(1, len(valid_results)):
            current = valid_results[i]
            prev = valid_results[i-1]
            
            # Check if this element is on the same line as the previous one
            vertical_distance = abs(current['center_y'] - prev['center_y'])
            avg_height = (current['height'] + prev['height']) / 2
            
            if vertical_distance < avg_height * 0.5:  # Same line
                current_line.append(current)
            else:  # New line
                lines.append(current_line)
                current_line = [current]
        
        # Add the last line
        lines.append(current_line)
        
        # Process each line: sort elements left to right and join appropriately
        organized_lines = []
        for line in lines:
            # Sort elements in the line from left to right
            line.sort(key=lambda x: x['center_x'])
            
            # Join elements in the line with appropriate spacing
            line_text = self._join_line_elements(line)
            if line_text.strip():
                organized_lines.append(line_text.strip())
        
        return '\n'.join(organized_lines)

    def _join_line_elements(self, line_elements):
        """Join elements in a line with appropriate spacing."""
        if not line_elements:
            return ""
        
        if len(line_elements) == 1:
            return line_elements[0]['text']
        
        result = []
        for i, element in enumerate(line_elements):
            text = element['text']
            
            if i == 0:
                result.append(text)
            else:
                prev_element = line_elements[i-1]
                
                # Calculate horizontal gap between elements
                gap = element['x1'] - prev_element['x2']
                avg_width = (element['width'] + prev_element['width']) / 2
                
                # Determine spacing based on gap size
                if gap > avg_width * 0.8:  # Large gap - likely separate columns/sections
                    result.append("    " + text)  # Add tab-like spacing
                elif gap > avg_width * 0.3:  # Medium gap - separate words/numbers
                    result.append(" " + text)  # Add single space
                else:  # Small gap - likely part of same word
                    result.append(text)  # No additional spacing
        
        return ''.join(result)

    def _calculate_similarity(self, text1, text2):
        """Calculate similarity between two text strings."""
        if not text1 or not text2:
            return 0.0
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def _remove_duplicates(self, pymupdf_text, ocr_text):
        """Remove duplicate content between PyMuPDF and OCR results."""
        if not pymupdf_text and not ocr_text:
            return ""
        
        if not pymupdf_text:
            return ocr_text
        
        if not ocr_text:
            return pymupdf_text
        
        # Split texts into paragraphs
        pymupdf_paragraphs = [p.strip() for p in pymupdf_text.split('\n\n') if p.strip()]
        ocr_paragraphs = [p.strip() for p in ocr_text.split('\n\n') if p.strip()]
        
        # Also split by single newlines for finer granularity
        pymupdf_lines = [line.strip() for line in pymupdf_text.split('\n') if line.strip()]
        ocr_lines = [line.strip() for line in ocr_text.split('\n') if line.strip()]
        
        # Find unique content
        unique_content = []
        
        # Start with PyMuPDF content
        for para in pymupdf_paragraphs:
            if para:
                unique_content.append(para)
        
        # Add OCR content that's not similar to existing content
        for ocr_para in ocr_paragraphs:
            if not ocr_para:
                continue
                
            is_duplicate = False
            for existing_para in unique_content:
                similarity = self._calculate_similarity(ocr_para, existing_para)
                if similarity > SIMILARITY_THRESHOLD:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                # Also check against individual lines for partial matches
                for ocr_line in ocr_para.split('\n'):
                    ocr_line = ocr_line.strip()
                    if not ocr_line:
                        continue
                    
                    line_is_duplicate = False
                    for existing_line in pymupdf_lines:
                        if self._calculate_similarity(ocr_line, existing_line) > SIMILARITY_THRESHOLD:
                            line_is_duplicate = True
                            break
                    
                    if not line_is_duplicate:
                        unique_content.append(ocr_line)
        
        # Clean and organize the final content
        final_content = []
        seen_lines = set()
        
        for content in unique_content:
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if line and line.lower() not in seen_lines:
                    seen_lines.add(line.lower())
                    final_content.append(line)
        
        return '\n'.join(final_content)

    def _clean_text(self, text):
        """Clean and normalize extracted text with focus on proper spacing."""
        if not text:
            return ""
        
        # Normalize whitespace (convert multiple spaces/tabs to single space)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Normalize line breaks (remove excessive empty lines)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Ensure proper spacing between words that might be concatenated
        # Add space before capital letters that follow lowercase letters (camelCase fix)
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        
        # Add space between letters and numbers when they're directly adjacent
        text = re.sub(r'([a-zA-Z])([0-9])', r'\1 \2', text)
        text = re.sub(r'([0-9])([a-zA-Z])', r'\1 \2', text)
        
        # Remove leading/trailing whitespace from each line and remove duplicates
        lines = text.split('\n')
        cleaned_lines = []
        seen = set()
        
        for line in lines:
            line = line.strip()
            if line and line.lower() not in seen:
                seen.add(line.lower())
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)

    async def _extract_from_non_pdf(self, file_path: Path, extension: str) -> str:
        """
        Extract text from non-PDF files. 
        This is a placeholder for future enhancements.
        For now, it returns a message indicating the file type is not supported by hybrid extraction.
        """
        return f"Hybrid extraction not yet implemented for {extension} files. Use standard TextExtractionService." 