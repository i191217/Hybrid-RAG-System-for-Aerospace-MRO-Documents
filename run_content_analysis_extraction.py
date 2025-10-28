#!/usr/bin/env python3
"""
Enhanced PDF Extraction Runner with Content Analysis

This script demonstrates enhanced PDF extraction that identifies:
- Content types (tables, charts, graphs, paragraphs)
- Page numbers for each content segment
- Detailed content analysis reports
"""

import os
import sys
import time
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional
import json

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from content_analysis import ContentAnalyzer, ContentSegment, ContentType
from pdf_extraction_comparison import PDFExtractionComparison

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class ContentAwarePDFExtractor:
    """PDF extractor with content type detection and page tracking."""
    
    def __init__(self):
        """Initialize the content-aware extractor."""
        self.content_analyzer = ContentAnalyzer()
        self.base_extractor = PDFExtractionComparison()

    def extract_with_content_analysis(self, pdf_path: Path, library_name: str) -> Dict:
        """
        Extract text with content analysis for a specific library.
        
        Args:
            pdf_path: Path to PDF file
            library_name: Library to use (pdfplumber, pdfminer, pypdf2, pymupdf)
            
        Returns:
            Dictionary with extraction results and content analysis
        """
        logger.info(f"Extracting with {library_name} + content analysis...")
        
        # Get extraction method
        extraction_methods = {
            'pdfplumber': self.base_extractor.extract_with_pdfplumber,
            'pdfminer': self.base_extractor.extract_with_pdfminer,
            'pypdf2': self.base_extractor.extract_with_pypdf2,
            'pymupdf': self.base_extractor.extract_with_pymupdf,
        }
        
        if library_name.lower() not in extraction_methods:
            raise ValueError(f"Unknown library: {library_name}")
        
        # Extract text using base library
        base_result = extraction_methods[library_name.lower()](pdf_path)
        
        if not base_result.success:
            return {
                'library_name': library_name,
                'success': False,
                'error_message': base_result.error_message,
                'content_segments': [],
                'content_summary': {}
            }
        
        # Analyze content by pages
        content_segments = []
        pages_text = self._split_text_by_pages(base_result.text)
        
        for page_num, page_text in pages_text.items():
            if page_text.strip():
                # Analyze content types for this page
                page_segments = self.content_analyzer.analyze_text_content(page_text, page_num)
                content_segments.extend(page_segments)
        
        # Generate content summary
        content_summary = {}
        for segment in content_segments:
            content_type = segment.content_type.value
            content_summary[content_type] = content_summary.get(content_type, 0) + 1
        
        return {
            'library_name': library_name,
            'success': True,
            'duration_seconds': base_result.duration_seconds,
            'total_character_count': base_result.character_count,
            'page_count': base_result.page_count,
            'content_segments': content_segments,
            'content_summary': content_summary,
            'original_text': base_result.text
        }

    def _split_text_by_pages(self, text: str) -> Dict[int, str]:
        """Split extracted text back into pages based on page markers."""
        pages = {}
        current_page = 1
        current_content = []
        
        lines = text.split('\n')
        for line in lines:
            # Look for page markers like "=== PAGE 2 ==="
            page_match = re.search(r'=== PAGE (\d+) ===', line)
            if page_match:
                # Save previous page content
                if current_content:
                    pages[current_page] = '\n'.join(current_content)
                
                # Start new page
                current_page = int(page_match.group(1))
                current_content = []
            else:
                current_content.append(line)
        
        # Save last page
        if current_content:
            pages[current_page] = '\n'.join(current_content)
        
        return pages

    def enhanced_table_detection(self, pdf_path: Path) -> Dict:
        """Special focus on table detection across all libraries."""
        logger.info("Running enhanced table detection comparison...")
        
        results = {}
        libraries = ['pdfplumber', 'pdfminer', 'pypdf2', 'pymupdf']
        
        for library in libraries:
            try:
                result = self.extract_with_content_analysis(pdf_path, library)
                
                if result['success']:
                    # Focus on table segments
                    table_segments = [s for s in result['content_segments'] 
                                    if s.content_type == ContentType.TABLE]
                    
                    results[library] = {
                        'total_tables': len(table_segments),
                        'table_pages': list(set(s.page_number for s in table_segments)),
                        'table_details': [
                            {
                                'page': s.page_number,
                                'confidence': s.confidence,
                                'text_preview': s.text[:200] + "..." if len(s.text) > 200 else s.text,
                                'additional_info': s.additional_info
                            }
                            for s in table_segments
                        ]
                    }
                else:
                    results[library] = {'error': result['error_message']}
                    
            except Exception as e:
                results[library] = {'error': str(e)}
        
        return results

    def save_content_analysis_report(self, results: Dict, output_file: Path):
        """Save detailed content analysis report."""
        lines = [
            "ENHANCED PDF CONTENT ANALYSIS REPORT",
            "=" * 80,
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Library: {results['library_name']}",
            "",
            f"SUCCESS: {results['success']}",
            f"Total Characters: {results.get('total_character_count', 0):,}",
            f"Total Pages: {results.get('page_count', 0)}",
            f"Total Content Segments: {len(results.get('content_segments', []))}",
            f"Processing Time: {results.get('duration_seconds', 0):.2f} seconds",
            "",
            "CONTENT TYPE SUMMARY:",
            "-" * 40,
        ]
        
        content_summary = results.get('content_summary', {})
        total_segments = len(results.get('content_segments', []))
        
        for content_type, count in sorted(content_summary.items()):
            percentage = (count / total_segments * 100) if total_segments > 0 else 0
            lines.append(f"{content_type}: {count} segments ({percentage:.1f}%)")
        
        lines.extend([
            "",
            "PAGE-BY-PAGE CONTENT ANALYSIS:",
            "=" * 80
        ])
        
        # Group segments by page
        segments = results.get('content_segments', [])
        pages = {}
        for segment in segments:
            page_num = segment.page_number
            if page_num not in pages:
                pages[page_num] = []
            pages[page_num].append(segment)
        
        for page_num in sorted(pages.keys()):
            lines.append(f"\n--- PAGE {page_num} ---")
            page_segments = pages[page_num]
            
            # Summary for this page
            page_types = {}
            for segment in page_segments:
                content_type = segment.content_type.value
                page_types[content_type] = page_types.get(content_type, 0) + 1
            
            lines.append("Page Content Summary:")
            for content_type, count in sorted(page_types.items()):
                lines.append(f"  {content_type}: {count}")
            
            lines.append("\nDetailed Content:")
            for i, segment in enumerate(page_segments, 1):
                lines.extend([
                    f"\n  Segment {i}:",
                    f"    Type: {segment.content_type.value}",
                    f"    Confidence: {segment.confidence:.2f}",
                    f"    Length: {len(segment.text)} characters",
                ])
                
                if segment.additional_info:
                    lines.append("    Additional Info:")
                    for key, value in segment.additional_info.items():
                        lines.append(f"      {key}: {value}")
                
                # Show text preview
                preview = segment.text[:300].replace('\n', ' ').strip()
                if len(segment.text) > 300:
                    preview += "..."
                lines.append(f"    Preview: {preview}")
        
        # Save the report
        try:
            output_file.write_text('\n'.join(lines), encoding='utf-8', errors='replace')
            logger.info(f"Content analysis report saved: {output_file}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")

def main():
    """Main function to run enhanced content analysis."""
    import argparse
    import re
    
    parser = argparse.ArgumentParser(description="Enhanced PDF extraction with content analysis")
    parser.add_argument("pdf_file", help="Path to PDF file")
    parser.add_argument("--library", "-l", choices=['pdfplumber', 'pdfminer', 'pypdf2', 'pymupdf'],
                       help="Specific library to test")
    parser.add_argument("--output-dir", "-o", default="content_analysis_results",
                       help="Output directory")
    parser.add_argument("--table-focus", action="store_true",
                       help="Focus on table detection across all libraries")
    
    args = parser.parse_args()
    
    pdf_path = Path(args.pdf_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if not pdf_path.exists():
        print(f"Error: PDF file not found: {pdf_path}")
        return
    
    extractor = ContentAwarePDFExtractor()
    
    if args.table_focus:
        # Focus on table detection
        print("Running enhanced table detection analysis...")
        table_results = extractor.enhanced_table_detection(pdf_path)
        
        # Save table analysis
        table_report_file = output_dir / f"table_analysis_{pdf_path.stem}.json"
        with open(table_report_file, 'w') as f:
            json.dump(table_results, f, indent=2, default=str)
        
        # Print summary
        print("\nTABLE DETECTION SUMMARY:")
        print("-" * 40)
        for library, result in table_results.items():
            if 'error' in result:
                print(f"{library}: ERROR - {result['error']}")
            else:
                tables = result['total_tables']
                pages = result['table_pages']
                print(f"{library}: {tables} tables found on pages {pages}")
        
        print(f"\nDetailed results saved to: {table_report_file}")
    
    elif args.library:
        # Test specific library
        print(f"Testing {args.library} with content analysis...")
        result = extractor.extract_with_content_analysis(pdf_path, args.library)
        
        if result['success']:
            # Save detailed report
            report_file = output_dir / f"content_analysis_{args.library}_{pdf_path.stem}.txt"
            extractor.save_content_analysis_report(result, report_file)
            
            # Print summary
            print(f"\nCONTENT ANALYSIS SUMMARY - {args.library}")
            print("-" * 50)
            print(f"Total segments: {len(result['content_segments'])}")
            print(f"Processing time: {result['duration_seconds']:.2f}s")
            print("\nContent types found:")
            for content_type, count in sorted(result['content_summary'].items()):
                print(f"  {content_type}: {count}")
            
            print(f"\nDetailed report saved to: {report_file}")
        else:
            print(f"Error: {result['error_message']}")
    
    else:
        # Test all libraries
        print("Testing all libraries with content analysis...")
        libraries = ['pdfplumber', 'pdfminer', 'pypdf2', 'pymupdf']
        
        for library in libraries:
            print(f"\n--- Testing {library} ---")
            try:
                result = extractor.extract_with_content_analysis(pdf_path, library)
                
                if result['success']:
                    # Save report
                    report_file = output_dir / f"content_analysis_{library}_{pdf_path.stem}.txt"
                    extractor.save_content_analysis_report(result, report_file)
                    
                    # Quick summary
                    segments = len(result['content_segments'])
                    duration = result['duration_seconds']
                    print(f"  SUCCESS: {segments} segments in {duration:.2f}s")
                    
                    content_summary = result['content_summary']
                    for content_type, count in sorted(content_summary.items()):
                        print(f"    {content_type}: {count}")
                else:
                    print(f"  ERROR: {result['error_message']}")
                    
            except Exception as e:
                print(f"  ERROR: {str(e)}")
        
        print(f"\nAll reports saved to: {output_dir}")


if __name__ == "__main__":
    main() 