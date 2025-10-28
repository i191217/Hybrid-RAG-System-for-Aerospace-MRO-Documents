#!/usr/bin/env python3
"""
Block-Based Content Analysis for PDF Extraction

This module detects actual blocks/elements on PDF pages using each library's 
native capabilities, then classifies them based on their properties and structure.
Much more accurate than regex-based text pattern matching.
"""

import re
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# Content types for block classification
class BlockType(Enum):
    """Types of content blocks that can be detected."""
    TEXT_BLOCK = "TEXT_BLOCK"
    TABLE = "TABLE" 
    IMAGE = "IMAGE"
    FIGURE = "FIGURE"
    CHART = "CHART"
    DRAWING = "DRAWING"
    HEADER = "HEADER"
    FOOTER = "FOOTER"
    LIST = "LIST"
    EQUATION = "EQUATION"
    UNKNOWN = "UNKNOWN"

@dataclass
class ContentBlock:
    """Represents a detected content block with metadata."""
    block_type: BlockType
    page_number: int
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    text_content: str
    confidence: float  # 0.0 to 1.0
    properties: Dict[str, Any]  # Block-specific properties
    source_library: str
    extraction_method: str

class BlockBasedAnalyzer:
    """Analyzes PDF pages by detecting actual blocks and classifying them."""
    
    def __init__(self):
        """Initialize the block-based analyzer."""
        self.logger = logging.getLogger(__name__)

    def analyze_page_blocks_pymupdf(self, page, page_num: int) -> List[ContentBlock]:
        """
        Analyze blocks using PyMuPDF's block detection capabilities.
        PyMuPDF has excellent block detection with coordinates.
        """
        blocks = []
        
        try:
            # Method 1: Get blocks with coordinates and type information
            text_blocks = page.get_text("blocks")
            for block in text_blocks:
                x0, y0, x1, y1, text, block_num, block_type = block
                
                if text.strip():
                    # Classify the text block
                    classified_type, confidence, properties = self._classify_text_block(
                        text, (x0, y0, x1, y1)
                    )
                    
                    content_block = ContentBlock(
                        block_type=classified_type,
                        page_number=page_num,
                        bbox=(x0, y0, x1, y1),
                        text_content=text.strip(),
                        confidence=confidence,
                        properties=properties,
                        source_library="PyMuPDF",
                        extraction_method="get_text_blocks"
                    )
                    blocks.append(content_block)
            
            # Method 2: Get images
            images = page.get_images()
            for img_index, img in enumerate(images):
                xref = img[0]
                try:
                    # Try to get image bbox
                    img_doc = page.parent
                    img_rect = page.get_image_bbox(img)
                    
                    image_block = ContentBlock(
                        block_type=BlockType.IMAGE,
                        page_number=page_num,
                        bbox=(img_rect.x0, img_rect.y0, img_rect.x1, img_rect.y1),
                        text_content=f"Image {img_index + 1}",
                        confidence=0.95,
                        properties={
                            'image_index': img_index,
                            'xref': xref,
                            'width': img_rect.width,
                            'height': img_rect.height
                        },
                        source_library="PyMuPDF",
                        extraction_method="get_images"
                    )
                    blocks.append(image_block)
                except Exception as e:
                    self.logger.debug(f"Could not get bbox for image {img_index}: {e}")
            
            # Method 3: Get drawings/vector graphics
            drawings = page.get_drawings()
            for draw_index, drawing in enumerate(drawings):
                try:
                    # Get drawing bounds
                    rect = drawing.get("rect")
                    if rect:
                        drawing_block = ContentBlock(
                            block_type=BlockType.DRAWING,
                            page_number=page_num,
                            bbox=(rect.x0, rect.y0, rect.x1, rect.y1),
                            text_content=f"Drawing/Graphic {draw_index + 1}",
                            confidence=0.90,
                            properties={
                                'drawing_index': draw_index,
                                'drawing_type': 'vector_graphic'
                            },
                            source_library="PyMuPDF",
                            extraction_method="get_drawings"
                        )
                        blocks.append(drawing_block)
                except Exception as e:
                    self.logger.debug(f"Could not process drawing {draw_index}: {e}")
            
            # Method 4: Detailed block analysis using dict format
            page_dict = page.get_text("dict")
            for block in page_dict.get("blocks", []):
                if "lines" in block:  # Text block
                    bbox = block.get("bbox", (0, 0, 0, 0))
                    
                    # Reconstruct text from lines and spans
                    block_text = ""
                    line_count = 0
                    for line in block.get("lines", []):
                        line_count += 1
                        for span in line.get("spans", []):
                            block_text += span.get("text", "")
                        block_text += "\n"
                    
                    if block_text.strip():
                        # Enhanced classification with font and layout info
                        classified_type, confidence, properties = self._classify_detailed_block(
                            block_text.strip(), bbox, block, line_count
                        )
                        
                        # Avoid duplicates from previous method
                        if not any(b.bbox == bbox and abs(len(b.text_content) - len(block_text.strip())) < 10 
                                 for b in blocks):
                            
                            detailed_block = ContentBlock(
                                block_type=classified_type,
                                page_number=page_num,
                                bbox=bbox,
                                text_content=block_text.strip(),
                                confidence=confidence,
                                properties=properties,
                                source_library="PyMuPDF",
                                extraction_method="get_text_dict_detailed"
                            )
                            blocks.append(detailed_block)
        
        except Exception as e:
            self.logger.error(f"PyMuPDF block analysis failed for page {page_num}: {e}")
        
        return blocks

    def analyze_page_blocks_pdfplumber(self, page, page_num: int) -> List[ContentBlock]:
        """
        Analyze blocks using pdfplumber's object detection capabilities.
        pdfplumber excels at table detection and has good object analysis.
        """
        blocks = []
        
        try:
            # Method 1: Native table detection (pdfplumber's strength)
            tables = page.extract_tables()
            for table_index, table in enumerate(tables):
                if table:
                    # Try to get table bbox
                    table_text = self._format_table_content(table)
                    
                    # Estimate table bbox from table content
                    bbox = self._estimate_table_bbox(page, table)
                    
                    table_block = ContentBlock(
                        block_type=BlockType.TABLE,
                        page_number=page_num,
                        bbox=bbox,
                        text_content=table_text,
                        confidence=0.95,  # High confidence for native table detection
                        properties={
                            'table_index': table_index,
                            'rows': len(table),
                            'columns': len(table[0]) if table and table[0] else 0,
                            'extraction_method': 'native_table_detection'
                        },
                        source_library="pdfplumber",
                        extraction_method="extract_tables"
                    )
                    blocks.append(table_block)
            
            # Method 2: Figure detection
            figures = page.figures
            for fig_index, figure in enumerate(figures):
                bbox = figure.get("bbox", (0, 0, 0, 0))
                
                figure_block = ContentBlock(
                    block_type=BlockType.FIGURE,
                    page_number=page_num,
                    bbox=bbox,
                    text_content=f"Figure {fig_index + 1}",
                    confidence=0.90,
                    properties={
                        'figure_index': fig_index,
                        'figure_properties': figure
                    },
                    source_library="pdfplumber",
                    extraction_method="figures_detection"
                )
                blocks.append(figure_block)
            
            # Method 3: Text extraction with layout analysis
            # Get all text objects to understand layout
            chars = page.chars
            if chars:
                # Group characters into blocks based on proximity
                text_blocks = self._group_chars_into_blocks(chars)
                
                for block_index, char_block in enumerate(text_blocks):
                    block_text = ''.join(char['text'] for char in char_block)
                    if block_text.strip():
                        # Calculate block bbox
                        x0 = min(char['x0'] for char in char_block)
                        y0 = min(char['top'] for char in char_block)
                        x1 = max(char['x1'] for char in char_block)
                        y1 = max(char['bottom'] for char in char_block)
                        bbox = (x0, y0, x1, y1)
                        
                        # Classify the block
                        classified_type, confidence, properties = self._classify_text_block(
                            block_text, bbox
                        )
                        
                        # Skip if we already have this as a table
                        if not any(b.block_type == BlockType.TABLE and 
                                 self._bbox_overlap(b.bbox, bbox) > 0.5 for b in blocks):
                            
                            text_block = ContentBlock(
                                block_type=classified_type,
                                page_number=page_num,
                                bbox=bbox,
                                text_content=block_text.strip(),
                                confidence=confidence,
                                properties=properties,
                                source_library="pdfplumber",
                                extraction_method="character_grouping"
                            )
                            blocks.append(text_block)
        
        except Exception as e:
            self.logger.error(f"pdfplumber block analysis failed for page {page_num}: {e}")
        
        return blocks

    def analyze_page_blocks_pdfminer(self, page_layout, page_num: int) -> List[ContentBlock]:
        """
        Analyze blocks using PDFMiner's layout analysis.
        PDFMiner has excellent layout object detection.
        """
        blocks = []
        
        try:
            for element in page_layout:
                element_type = type(element).__name__
                bbox = getattr(element, 'bbox', (0, 0, 0, 0))
                
                if hasattr(element, 'get_text'):
                    # Text-based elements
                    text = element.get_text().strip()
                    if text:
                        # Classify based on layout element type and content
                        if 'TextContainer' in element_type or 'TextBox' in element_type:
                            classified_type, confidence, properties = self._classify_text_block(text, bbox)
                        else:
                            classified_type = BlockType.TEXT_BLOCK
                            confidence = 0.8
                            properties = {}
                        
                        properties.update({
                            'layout_element_type': element_type,
                            'pdfminer_element': str(element)
                        })
                        
                        text_block = ContentBlock(
                            block_type=classified_type,
                            page_number=page_num,
                            bbox=bbox,
                            text_content=text,
                            confidence=confidence,
                            properties=properties,
                            source_library="PDFMiner.six",
                            extraction_method="layout_analysis"
                        )
                        blocks.append(text_block)
                
                elif 'Figure' in element_type:
                    # Figure elements
                    figure_block = ContentBlock(
                        block_type=BlockType.FIGURE,
                        page_number=page_num,
                        bbox=bbox,
                        text_content=f"Figure (LTFigure)",
                        confidence=0.95,
                        properties={
                            'layout_element_type': element_type,
                            'width': bbox[2] - bbox[0],
                            'height': bbox[3] - bbox[1]
                        },
                        source_library="PDFMiner.six",
                        extraction_method="layout_analysis"
                    )
                    blocks.append(figure_block)
                
                elif 'Image' in element_type:
                    # Image elements
                    image_block = ContentBlock(
                        block_type=BlockType.IMAGE,
                        page_number=page_num,
                        bbox=bbox,
                        text_content=f"Image (LTImage)",
                        confidence=0.95,
                        properties={
                            'layout_element_type': element_type,
                            'width': bbox[2] - bbox[0],
                            'height': bbox[3] - bbox[1]
                        },
                        source_library="PDFMiner.six",
                        extraction_method="layout_analysis"
                    )
                    blocks.append(image_block)
        
        except Exception as e:
            self.logger.error(f"PDFMiner block analysis failed for page {page_num}: {e}")
        
        return blocks

    def analyze_page_blocks_pypdf2(self, page, page_num: int) -> List[ContentBlock]:
        """
        Analyze blocks using PyPDF2 (limited block detection).
        PyPDF2 has limited layout analysis capabilities.
        """
        blocks = []
        
        try:
            # PyPDF2 mainly provides text extraction
            text = page.extract_text()
            if text.strip():
                # Split into paragraphs and analyze
                paragraphs = text.split('\n\n')
                y_position = 0  # Estimated positions since PyPDF2 doesn't provide coordinates
                
                for para_index, paragraph in enumerate(paragraphs):
                    if paragraph.strip():
                        # Estimate bbox (PyPDF2 limitation)
                        estimated_bbox = (0, y_position, 500, y_position + 50)
                        y_position += 60
                        
                        classified_type, confidence, properties = self._classify_text_block(
                            paragraph, estimated_bbox
                        )
                        
                        properties.update({
                            'estimated_coordinates': True,
                            'paragraph_index': para_index
                        })
                        
                        text_block = ContentBlock(
                            block_type=classified_type,
                            page_number=page_num,
                            bbox=estimated_bbox,
                            text_content=paragraph.strip(),
                            confidence=confidence * 0.7,  # Lower confidence due to limitations
                            properties=properties,
                            source_library="PyPDF2",
                            extraction_method="text_paragraph_split"
                        )
                        blocks.append(text_block)
        
        except Exception as e:
            self.logger.error(f"PyPDF2 block analysis failed for page {page_num}: {e}")
        
        return blocks

    def _classify_text_block(self, text: str, bbox: Tuple[float, float, float, float]) -> Tuple[BlockType, float, Dict]:
        """Classify a text block based on content and properties."""
        block_type = BlockType.TEXT_BLOCK
        confidence = 0.7
        properties = {}
        
        # Calculate dimensions
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        # Table detection patterns
        if self._is_table_content(text):
            block_type = BlockType.TABLE
            confidence = 0.85
            properties.update({
                'detection_method': 'content_pattern_analysis',
                'table_indicators': self._get_table_indicators(text)
            })
        
        # Header detection
        elif self._is_header_content(text, height):
            block_type = BlockType.HEADER
            confidence = 0.80
            properties.update({
                'header_indicators': self._get_header_indicators(text, height)
            })
        
        # List detection
        elif self._is_list_content(text):
            block_type = BlockType.LIST
            confidence = 0.85
            properties.update({
                'list_type': self._get_list_type(text),
                'list_items': text.count('\n') + 1
            })
        
        # Chart/Figure reference detection
        elif self._is_chart_reference(text):
            block_type = BlockType.CHART
            confidence = 0.75
            properties.update({
                'chart_references': self._extract_chart_references(text)
            })
        
        properties.update({
            'text_length': len(text),
            'line_count': text.count('\n') + 1,
            'bbox_width': width,
            'bbox_height': height
        })
        
        return block_type, confidence, properties

    def _classify_detailed_block(self, text: str, bbox: Tuple[float, float, float, float], 
                               block_dict: Dict, line_count: int) -> Tuple[BlockType, float, Dict]:
        """Enhanced classification using detailed block information."""
        block_type, confidence, properties = self._classify_text_block(text, bbox)
        
        # Add detailed analysis from block structure
        properties.update({
            'line_count': line_count,
            'block_structure': 'multi_line' if line_count > 1 else 'single_line'
        })
        
        # Analyze font information if available
        if 'lines' in block_dict:
            font_sizes = []
            font_names = []
            for line in block_dict['lines']:
                for span in line.get('spans', []):
                    font_sizes.append(span.get('size', 0))
                    font_names.append(span.get('font', ''))
            
            if font_sizes:
                avg_font_size = sum(font_sizes) / len(font_sizes)
                properties.update({
                    'avg_font_size': avg_font_size,
                    'font_variations': len(set(font_sizes)),
                    'fonts_used': list(set(font_names))
                })
                
                # Adjust classification based on font size
                if avg_font_size > 14 and len(text) < 100:
                    block_type = BlockType.HEADER
                    confidence = 0.90
        
        return block_type, confidence, properties

    def _is_table_content(self, text: str) -> bool:
        """Check if text content suggests a table structure."""
        lines = text.split('\n')
        
        # Check for table patterns
        pipe_separated = sum(1 for line in lines if '|' in line and line.count('|') >= 2)
        tab_separated = sum(1 for line in lines if '\t' in line)
        multi_column = sum(1 for line in lines if re.search(r'\s{3,}', line))
        numeric_data = sum(1 for line in lines if re.search(r'\d+.*\d+.*\d+', line))
        
        # Table indicators
        table_keywords = sum(1 for line in lines if re.search(r'\b(table|column|row)\b', line, re.IGNORECASE))
        
        return (pipe_separated >= 2 or 
                (multi_column >= 3 and numeric_data >= 2) or
                (tab_separated >= 2 and len(lines) >= 3))

    def _is_header_content(self, text: str, height: float) -> bool:
        """Check if text content suggests a header."""
        return (len(text) < 100 and 
                text.isupper() and 
                text.count('\n') <= 1 and
                height < 30)

    def _is_list_content(self, text: str) -> bool:
        """Check if text content suggests a list."""
        lines = text.split('\n')
        bullet_lines = sum(1 for line in lines if re.match(r'^\s*[•▪▫*-]\s+', line))
        numbered_lines = sum(1 for line in lines if re.match(r'^\s*\d+[.)]\s+', line))
        
        return bullet_lines >= 2 or numbered_lines >= 2

    def _is_chart_reference(self, text: str) -> bool:
        """Check if text references charts or figures."""
        return bool(re.search(r'\b(figure|chart|graph|diagram)\s+\d+', text, re.IGNORECASE))

    def _get_table_indicators(self, text: str) -> List[str]:
        """Extract table structure indicators."""
        indicators = []
        if '|' in text:
            indicators.append('pipe_separated')
        if '\t' in text:
            indicators.append('tab_separated')
        if re.search(r'\d+.*\d+.*\d+', text):
            indicators.append('numeric_columns')
        return indicators

    def _get_header_indicators(self, text: str, height: float) -> Dict:
        """Extract header indicators."""
        return {
            'is_uppercase': text.isupper(),
            'is_short': len(text) < 100,
            'height': height,
            'line_count': text.count('\n') + 1
        }

    def _get_list_type(self, text: str) -> str:
        """Determine the type of list."""
        if re.search(r'^\s*[•▪▫*-]\s+', text, re.MULTILINE):
            return 'bulleted'
        elif re.search(r'^\s*\d+[.)]\s+', text, re.MULTILINE):
            return 'numbered'
        else:
            return 'unknown'

    def _extract_chart_references(self, text: str) -> List[str]:
        """Extract chart/figure references."""
        return re.findall(r'\b(figure|chart|graph|diagram)\s+\d+', text, re.IGNORECASE)

    def _format_table_content(self, table: List[List]) -> str:
        """Format table content for text representation."""
        if not table:
            return ""
        
        lines = []
        for row in table:
            if row:
                clean_row = [str(cell) if cell is not None else "" for cell in row]
                lines.append(" | ".join(clean_row))
        
        return "\n".join(lines)

    def _estimate_table_bbox(self, page, table: List[List]) -> Tuple[float, float, float, float]:
        """Estimate table bounding box (pdfplumber limitation)."""
        # This is a rough estimation - pdfplumber doesn't always provide table bbox
        page_width = page.width
        page_height = page.height
        
        # Estimate based on page position (very rough)
        return (50, page_height * 0.3, page_width - 50, page_height * 0.7)

    def _group_chars_into_blocks(self, chars: List[Dict]) -> List[List[Dict]]:
        """Group characters into logical blocks based on proximity."""
        if not chars:
            return []
        
        # Sort by y-coordinate (top to bottom)
        sorted_chars = sorted(chars, key=lambda c: (c['top'], c['x0']))
        
        blocks = []
        current_block = [sorted_chars[0]]
        
        for char in sorted_chars[1:]:
            prev_char = current_block[-1]
            
            # Check if characters are on same line (similar y-coordinate)
            y_diff = abs(char['top'] - prev_char['top'])
            x_gap = char['x0'] - prev_char['x1']
            
            if y_diff < 5 and x_gap < 50:  # Same block
                current_block.append(char)
            else:  # New block
                if current_block:
                    blocks.append(current_block)
                current_block = [char]
        
        if current_block:
            blocks.append(current_block)
        
        return blocks

    def _bbox_overlap(self, bbox1: Tuple[float, float, float, float], 
                     bbox2: Tuple[float, float, float, float]) -> float:
        """Calculate overlap ratio between two bounding boxes."""
        x1, y1, x2, y2 = bbox1
        x3, y3, x4, y4 = bbox2
        
        # Calculate intersection
        ix1 = max(x1, x3)
        iy1 = max(y1, y3)
        ix2 = min(x2, x4)
        iy2 = min(y2, y4)
        
        if ix1 < ix2 and iy1 < iy2:
            intersection = (ix2 - ix1) * (iy2 - iy1)
            area1 = (x2 - x1) * (y2 - y1)
            area2 = (x4 - x3) * (y4 - y3)
            
            if area1 > 0 and area2 > 0:
                return intersection / min(area1, area2)
        
        return 0.0 