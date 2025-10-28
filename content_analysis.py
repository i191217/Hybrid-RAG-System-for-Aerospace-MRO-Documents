#!/usr/bin/env python3
"""
Content Analysis Module for PDF Extraction

This module provides content type detection and analysis for PDF text extraction.
Identifies tables, charts, graphs, paragraphs and tracks page numbers.
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class ContentType(Enum):
    """Enumeration of content types that can be detected."""
    TABLE = "TABLE"
    CHART = "CHART"
    GRAPH = "GRAPH"  
    FIGURE = "FIGURE"
    IMAGE = "IMAGE"
    PARAGRAPH = "PARAGRAPH"
    HEADER = "HEADER"
    FOOTER = "FOOTER"
    LIST = "LIST"
    EQUATION = "EQUATION"
    UNKNOWN = "UNKNOWN"

@dataclass
class ContentSegment:
    """Represents a segment of extracted content with metadata."""
    text: str
    content_type: ContentType
    page_number: int
    confidence: float  # 0.0 to 1.0
    bbox: Optional[Tuple[float, float, float, float]] = None  # x0, y0, x1, y1
    additional_info: Optional[Dict] = None

class ContentAnalyzer:
    """Analyzes extracted text to identify content types and structure."""
    
    def __init__(self):
        """Initialize the content analyzer."""
        # Patterns for detecting different content types
        self.table_patterns = [
            re.compile(r'\|.*\|.*\|', re.MULTILINE),  # Pipe-separated tables
            re.compile(r'^\s*[\d\w]+\s+[\d\w]+\s+[\d\w]+\s*$', re.MULTILINE),  # Multi-column data
            re.compile(r'^\s*[-\s]+$', re.MULTILINE),  # Table separators
            re.compile(r'\b(?:table|TABLE)\s+\d+', re.IGNORECASE),  # Table captions
        ]
        
        self.chart_patterns = [
            re.compile(r'\b(?:chart|Chart|CHART)\s*\d*', re.IGNORECASE),
            re.compile(r'\b(?:figure|Figure|FIGURE)\s*\d*', re.IGNORECASE),
            re.compile(r'\b(?:graph|Graph|GRAPH)\s*\d*', re.IGNORECASE),
        ]
        
        self.header_patterns = [
            re.compile(r'^[A-Z\s]{3,50}$', re.MULTILINE),  # All caps headers
            re.compile(r'^\d+\.\s+[A-Z]', re.MULTILINE),   # Numbered sections
            re.compile(r'^Chapter\s+\d+', re.IGNORECASE),
        ]
        
        self.list_patterns = [
            re.compile(r'^\s*[•·▪▫]\s+', re.MULTILINE),    # Bullet points
            re.compile(r'^\s*\d+\.\s+', re.MULTILINE),     # Numbered lists
            re.compile(r'^\s*[a-z]\)\s+', re.MULTILINE),   # Lettered lists
        ]

    def analyze_text_content(self, text: str, page_num: int) -> List[ContentSegment]:
        """
        Analyze text content and identify different content types.
        
        Args:
            text: Raw text to analyze
            page_num: Page number where text was extracted
            
        Returns:
            List of ContentSegment objects with identified content types
        """
        segments = []
        
        # Split text into logical segments
        paragraphs = self._split_into_paragraphs(text)
        
        for para in paragraphs:
            if not para.strip():
                continue
                
            content_type, confidence = self._identify_content_type(para)
            
            segment = ContentSegment(
                text=para.strip(),
                content_type=content_type,
                page_number=page_num,
                confidence=confidence,
                additional_info=self._extract_additional_info(para, content_type)
            )
            
            segments.append(segment)
        
        return segments

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into logical paragraphs and sections."""
        # Split on double newlines first
        paragraphs = text.split('\n\n')
        
        # Further split long paragraphs if they contain multiple topics
        refined_paragraphs = []
        for para in paragraphs:
            if len(para) > 500:  # Long paragraph, try to split further
                sentences = para.split('. ')
                current_para = ""
                for sentence in sentences:
                    if len(current_para + sentence) < 300:
                        current_para += sentence + ". "
                    else:
                        if current_para:
                            refined_paragraphs.append(current_para.strip())
                        current_para = sentence + ". "
                if current_para:
                    refined_paragraphs.append(current_para.strip())
            else:
                refined_paragraphs.append(para)
        
        return refined_paragraphs

    def _identify_content_type(self, text: str) -> Tuple[ContentType, float]:
        """
        Identify the content type of a text segment.
        
        Returns:
            Tuple of (ContentType, confidence_score)
        """
        text_lines = text.split('\n')
        
        # Check for tables (highest priority)
        table_score = self._calculate_table_score(text, text_lines)
        if table_score > 0.6:
            return ContentType.TABLE, table_score
        
        # Check for charts/figures
        chart_score = self._calculate_chart_score(text)
        if chart_score > 0.7:
            return ContentType.CHART, chart_score
        
        # Check for headers
        header_score = self._calculate_header_score(text, text_lines)
        if header_score > 0.8:
            return ContentType.HEADER, header_score
        
        # Check for lists
        list_score = self._calculate_list_score(text, text_lines)
        if list_score > 0.7:
            return ContentType.LIST, list_score
        
        # Check for equations
        equation_score = self._calculate_equation_score(text)
        if equation_score > 0.8:
            return ContentType.EQUATION, equation_score
        
        # Default to paragraph
        return ContentType.PARAGRAPH, 0.9

    def _calculate_table_score(self, text: str, lines: List[str]) -> float:
        """Calculate likelihood that text represents a table."""
        score = 0.0
        
        # Check for table patterns
        for pattern in self.table_patterns:
            if pattern.search(text):
                score += 0.3
        
        # Check for consistent column structure
        if len(lines) >= 3:
            # Look for consistent spacing patterns
            space_patterns = []
            for line in lines[:5]:  # Check first 5 lines
                spaces = [m.start() for m in re.finditer(r'\s{2,}', line)]
                space_patterns.append(spaces)
            
            # If multiple lines have similar spacing patterns, likely a table
            if len(space_patterns) >= 2:
                first_pattern = space_patterns[0]
                similar_patterns = sum(1 for pattern in space_patterns[1:] 
                                     if self._patterns_similar(first_pattern, pattern))
                if similar_patterns >= 1:
                    score += 0.4
        
        # Check for numeric data in columns
        numeric_lines = sum(1 for line in lines if re.search(r'\d+.*\d+.*\d+', line))
        if numeric_lines >= 2:
            score += 0.3
        
        return min(score, 1.0)

    def _calculate_chart_score(self, text: str) -> float:
        """Calculate likelihood that text represents a chart/figure."""
        score = 0.0
        
        for pattern in self.chart_patterns:
            if pattern.search(text):
                score += 0.5
        
        # Check for chart-related keywords
        chart_keywords = ['legend', 'axis', 'plot', 'data', 'series', 'x-axis', 'y-axis']
        for keyword in chart_keywords:
            if keyword.lower() in text.lower():
                score += 0.1
        
        return min(score, 1.0)

    def _calculate_header_score(self, text: str, lines: List[str]) -> float:
        """Calculate likelihood that text represents a header."""
        score = 0.0
        
        # Short text is more likely to be a header
        if len(text) < 100 and len(lines) <= 2:
            score += 0.4
        
        # Check for header patterns
        for pattern in self.header_patterns:
            if pattern.search(text):
                score += 0.3
        
        # All caps text (but not too long)
        if text.isupper() and len(text) < 50:
            score += 0.4
        
        return min(score, 1.0)

    def _calculate_list_score(self, text: str, lines: List[str]) -> float:
        """Calculate likelihood that text represents a list."""
        score = 0.0
        
        # Check for list patterns
        list_lines = 0
        for line in lines:
            for pattern in self.list_patterns:
                if pattern.search(line):
                    list_lines += 1
                    break
        
        if list_lines >= 2:
            score += 0.6
        elif list_lines == 1 and len(lines) <= 3:
            score += 0.4
        
        return min(score, 1.0)

    def _calculate_equation_score(self, text: str) -> float:
        """Calculate likelihood that text represents an equation."""
        score = 0.0
        
        # Check for mathematical symbols
        math_symbols = ['=', '+', '-', '×', '÷', '√', '∑', '∫', '^', '²', '³']
        symbol_count = sum(1 for symbol in math_symbols if symbol in text)
        
        if symbol_count >= 3:
            score += 0.5
        elif symbol_count >= 1 and len(text) < 100:
            score += 0.3
        
        # Check for equation patterns
        if re.search(r'\w+\s*=\s*\w+', text):
            score += 0.3
        
        return min(score, 1.0)

    def _patterns_similar(self, pattern1: List[int], pattern2: List[int], tolerance: int = 5) -> bool:
        """Check if two spacing patterns are similar."""
        if len(pattern1) != len(pattern2):
            return False
        
        for p1, p2 in zip(pattern1, pattern2):
            if abs(p1 - p2) > tolerance:
                return False
        
        return True

    def _extract_additional_info(self, text: str, content_type: ContentType) -> Dict:
        """Extract additional information based on content type."""
        info = {}
        
        if content_type == ContentType.TABLE:
            # Count rows and columns
            lines = [line for line in text.split('\n') if line.strip()]
            info['estimated_rows'] = len(lines)
            
            # Estimate columns by looking for consistent separators
            if lines:
                separators = len(re.findall(r'\s{2,}|\|', lines[0]))
                info['estimated_columns'] = separators + 1
        
        elif content_type == ContentType.CHART:
            # Look for chart title
            lines = text.split('\n')
            if lines and any(word in lines[0].lower() for word in ['figure', 'chart', 'graph']):
                info['title'] = lines[0].strip()
        
        elif content_type == ContentType.LIST:
            # Count list items
            items = 0
            for line in text.split('\n'):
                for pattern in self.list_patterns:
                    if pattern.search(line):
                        items += 1
                        break
            info['list_items'] = items
        
        return info

    def generate_content_report(self, segments: List[ContentSegment]) -> str:
        """Generate a report summarizing the content analysis."""
        if not segments:
            return "No content segments analyzed."
        
        # Count content types
        type_counts = {}
        for segment in segments:
            content_type = segment.content_type.value
            type_counts[content_type] = type_counts.get(content_type, 0) + 1
        
        # Count pages
        pages = set(segment.page_number for segment in segments)
        
        report_lines = [
            "CONTENT ANALYSIS REPORT",
            "=" * 50,
            f"Total segments analyzed: {len(segments)}",
            f"Pages covered: {len(pages)} (pages {min(pages)}-{max(pages)})",
            "",
            "Content Type Distribution:",
            "-" * 30,
        ]
        
        for content_type, count in sorted(type_counts.items()):
            percentage = (count / len(segments)) * 100
            report_lines.append(f"{content_type}: {count} ({percentage:.1f}%)")
        
        return "\n".join(report_lines) 