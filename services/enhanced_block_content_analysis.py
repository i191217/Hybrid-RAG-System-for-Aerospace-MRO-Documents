#!/usr/bin/env python3
"""
Enhanced Block-Based Content Analysis

This script uses actual block/element detection from each PDF library
to provide accurate content type identification and page tracking.
Much more reliable than regex-based pattern matching.
"""

import os
import sys
import time
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import json
import csv
import numpy as np
import urllib.request

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class BlockType(Enum):
    """Types of content blocks that can be detected."""
    TEXT_BLOCK = "TEXT_BLOCK"
    TABLE = "TABLE"
    IMAGE = "IMAGE" 
    FIGURE = "FIGURE"
    CHART = "CHART"
    DRAWING = "DRAWING"
    HEADER = "HEADER"
    LIST = "LIST"
    UNKNOWN = "UNKNOWN"

@dataclass
class ContentBlock:
    """Represents a detected content block with metadata."""
    block_type: BlockType
    page_number: int
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    text_content: str
    confidence: float  # 0.0 to 1.0
    properties: Dict[str, Any]
    source_library: str
    extraction_method: str

class EnhancedBlockAnalyzer:
    """Analyzes PDF pages using actual block detection from each library."""
    
    def __init__(self):
        """Initialize the enhanced block analyzer."""
        self.logger = logging.getLogger(__name__)

    def analyze_pdf_with_pymupdf(self, pdf_path: Path) -> Dict:
        """Analyze PDF using PyMuPDF's block detection with enhanced text extraction."""
        try:
            import fitz
            
            start_time = time.time()
            all_blocks = []
            
            doc = fitz.open(str(pdf_path))
            page_count = len(doc)
            
            self.logger.info(f"Starting PyMuPDF analysis of {pdf_path.name} ({page_count} pages)")
            
            for page_num in range(page_count):
                page = doc[page_num]
                page_blocks = self._analyze_pymupdf_page(page, page_num + 1)
                all_blocks.extend(page_blocks)
                
                # Log progress for large documents
                if page_count > 10 and (page_num + 1) % 10 == 0:
                    self.logger.info(f"Processed {page_num + 1}/{page_count} pages")
            
            doc.close()
            
            # Generate summary
            block_summary = self._generate_block_summary(all_blocks)
            total_chars = sum(len(block.text_content) for block in all_blocks)
            
            # Count blocks with actual extracted text vs placeholders
            text_extracted_blocks = 0
            for block in all_blocks:
                if (block.text_content and 
                    not block.text_content.startswith(('Image ', 'Drawing/Graphic ')) and
                    len(block.text_content.strip()) > 0):
                    text_extracted_blocks += 1
            
            results = {
                'library_name': 'PyMuPDF',
                'success': True,
                'duration_seconds': time.time() - start_time,
                'total_character_count': total_chars,
                'page_count': page_count,
                'content_blocks': all_blocks,
                'block_summary': block_summary,
                'extraction_method': 'enhanced_block_based_detection_with_text',
                'text_extraction_stats': {
                    'total_blocks': len(all_blocks),
                    'blocks_with_extracted_text': text_extracted_blocks,
                    'text_extraction_rate': text_extracted_blocks / len(all_blocks) if all_blocks else 0
                }
            }
            
            # Save extraction results
            self.save_pymupdf_extraction_results(results, pdf_path)
            
            self.logger.info(f"PyMuPDF analysis complete: {len(all_blocks)} blocks, {text_extracted_blocks} with extracted text")
            
            return results
            
        except ImportError:
            return self._create_error_result('PyMuPDF', 'PyMuPDF not installed')
        except Exception as e:
            return self._create_error_result('PyMuPDF', str(e))

    def analyze_pdf_with_pdfplumber(self, pdf_path: Path) -> Dict:
        """Analyze PDF using pdfplumber's object detection."""
        try:
            import pdfplumber
            
            start_time = time.time()
            all_blocks = []
            
            with pdfplumber.open(str(pdf_path)) as pdf:
                page_count = len(pdf.pages)
                
                for page_num, page in enumerate(pdf.pages, 1):
                    page_blocks = self._analyze_pdfplumber_page(page, page_num)
                    all_blocks.extend(page_blocks)
            
            block_summary = self._generate_block_summary(all_blocks)
            total_chars = sum(len(block.text_content) for block in all_blocks)
            
            return {
                'library_name': 'pdfplumber',
                'success': True,
                'duration_seconds': time.time() - start_time,
                'total_character_count': total_chars,
                'page_count': page_count,
                'content_blocks': all_blocks,
                'block_summary': block_summary,
                'extraction_method': 'object_based_detection'
            }
            
        except ImportError:
            return self._create_error_result('pdfplumber', 'pdfplumber not installed')
        except Exception as e:
            return self._create_error_result('pdfplumber', str(e))

    def analyze_pdf_with_pdfminer(self, pdf_path: Path) -> Dict:
        """Analyze PDF using PDFMiner's layout analysis."""
        try:
            from pdfminer.high_level import extract_pages
            
            start_time = time.time()
            all_blocks = []
            page_count = 0
            
            for page_layout in extract_pages(str(pdf_path)):
                page_count += 1
                page_blocks = self._analyze_pdfminer_page(page_layout, page_count)
                all_blocks.extend(page_blocks)
            
            block_summary = self._generate_block_summary(all_blocks)
            total_chars = sum(len(block.text_content) for block in all_blocks)
            
            return {
                'library_name': 'PDFMiner.six',
                'success': True,
                'duration_seconds': time.time() - start_time,
                'total_character_count': total_chars,
                'page_count': page_count,
                'content_blocks': all_blocks,
                'block_summary': block_summary,
                'extraction_method': 'layout_analysis'
            }
            
        except ImportError:
            return self._create_error_result('PDFMiner.six', 'pdfminer.six not installed')
        except Exception as e:
            return self._create_error_result('PDFMiner.six', str(e))

    def analyze_pdf_with_unstructured(self, pdf_path: Path) -> Dict:
        """Analyze PDF using Unstructured's advanced element detection."""
        try:
            from unstructured.partition.pdf import partition_pdf
            
            start_time = time.time()
            all_blocks = []
            
            # Try different strategies in order of preference
            strategies = [
                # High-res strategy (requires poppler)
                {
                    "strategy": "hi_res",
                    "infer_table_structure": True,
                    "extract_images_in_pdf": True,
                    "include_page_breaks": True,
                },
                # Auto strategy (fallback)
                {
                    "strategy": "auto",
                    "infer_table_structure": True,
                    "include_page_breaks": True,
                },
                # Fast strategy (minimal dependencies)
                {
                    "strategy": "fast",
                    "include_page_breaks": True,
                }
            ]
            
            elements = None
            used_strategy = None
            
            for i, strategy_config in enumerate(strategies):
                try:
                    self.logger.info(f"Trying Unstructured strategy {i+1}/3: {strategy_config['strategy']}")
                    elements = partition_pdf(filename=str(pdf_path), **strategy_config)
                    used_strategy = strategy_config['strategy']
                    break
                except Exception as strategy_error:
                    self.logger.warning(f"Strategy '{strategy_config['strategy']}' failed: {strategy_error}")
                    if i == len(strategies) - 1:  # Last strategy
                        raise strategy_error
                    continue
            
            if not elements:
                raise Exception("All Unstructured strategies failed")
            
            page_count = 0
            current_page = 1
            
            for element in elements:
                # Track page changes
                if hasattr(element, 'metadata') and element.metadata.page_number:
                    element_page = element.metadata.page_number
                    page_count = max(page_count, element_page)
                    current_page = element_page
                
                # Convert unstructured element to our ContentBlock
                content_block = self._convert_unstructured_element(element, current_page)
                if content_block:
                    all_blocks.append(content_block)
            
            block_summary = self._generate_block_summary(all_blocks)
            total_chars = sum(len(block.text_content) for block in all_blocks)
            
            return {
                'library_name': 'Unstructured',
                'success': True,
                'duration_seconds': time.time() - start_time,
                'total_character_count': total_chars,
                'page_count': page_count,
                'content_blocks': all_blocks,
                'block_summary': block_summary,
                'extraction_method': f'element_partitioning_{used_strategy}',
                'strategy_used': used_strategy
            }
            
        except ImportError:
            return self._create_error_result('Unstructured', 'unstructured library not installed (pip install unstructured[pdf])')
        except Exception as e:
            error_msg = str(e)
            if "poppler" in error_msg.lower():
                error_msg += " | Install poppler: conda install -c conda-forge poppler"
            return self._create_error_result('Unstructured', error_msg)

    def _ensure_layoutparser_models_are_local(self) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Ensures LayoutParser model files are downloaded locally.
        Bypasses the often-unreliable lp:// URI resolver.
        """
        model_name = 'ppyolov2_r50vd_dcn_365e_publaynet'
        
        # Define URLs for the model files - using a corrected, stable URL
        config_url = f"https://raw.githubusercontent.com/PaddlePaddle/PaddleDetection/release/2.3/configs/ppyolo/{model_name}.yml"
        weights_url = f"https://paddledet.bj.bcebos.com/models/{model_name}.pdparams"
        
        # Define local cache directory
        cache_dir = Path.home() / ".layoutparser" / "models" / model_name
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        local_config_path = cache_dir / "config.yml"
        local_weights_path = cache_dir / "weights.pdparams"

        # Download config file if it doesn't exist
        if not local_config_path.exists():
            self.logger.info(f"Downloading LayoutParser config for {model_name}...")
            try:
                urllib.request.urlretrieve(config_url, local_config_path)
                self.logger.info("Config download complete.")
            except Exception as e:
                self.logger.error(f"Failed to download model config: {e}")
                return None, None

        # Download weights file if it doesn't exist
        if not local_weights_path.exists():
            self.logger.info(f"Downloading LayoutParser model weights for {model_name} (this may take a while)...")
            try:
                urllib.request.urlretrieve(weights_url, local_weights_path)
                self.logger.info("Model weights download complete.")
            except Exception as e:
                self.logger.error(f"Failed to download model weights: {e}")
                return None, None
        
        return local_config_path, local_weights_path

    def analyze_pdf_with_layoutparser(self, pdf_path: Path) -> Dict:
        """Analyze PDF using LayoutParser's DL model."""
        try:
            import layoutparser as lp
            import cv2
            import fitz
            
            start_time = time.time()
            all_blocks = []

            # 1. Ensure model files are downloaded locally
            local_config_path, local_weights_path = self._ensure_layoutparser_models_are_local()
            if not local_config_path or not local_weights_path:
                raise Exception("Failed to download or locate LayoutParser model files.")

            # 2. Initialize LayoutParser model from local files
            self.logger.info("Initializing LayoutParser model (PaddleDetection backend from local files)...")
            model = lp.PaddleDetectionLayoutModel(
                config_path=str(local_config_path),
                model_path=str(local_weights_path),
                label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
                enforce_cpu=True
            )
            self.logger.info("LayoutParser model initialized.")

            doc = fitz.open(str(pdf_path))
            page_count = len(doc)

            for page_num in range(page_count):
                self.logger.info(f"LayoutParser analyzing page {page_num + 1}/{page_count}")
                page = doc[page_num]

                # Convert page to image
                pix = page.get_pixmap(dpi=200)
                img_bytes = pix.tobytes("png")
                img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

                # Detect layout
                layout = model.detect(img)

                for block in layout:
                    # Convert pixel coordinates to PDF points
                    img_h, img_w = img.shape[:2]
                    pdf_w, pdf_h = page.rect.width, page.rect.height
                    
                    x0, y0, x1, y1 = block.coordinates
                    
                    pdf_bbox = (
                        (x0 / img_w) * pdf_w,
                        (y0 / img_h) * pdf_h,
                        (x1 / img_w) * pdf_w,
                        (y1 / img_h) * pdf_h
                    )
                    
                    # Extract text from the bbox using PyMuPDF (more efficient)
                    text = page.get_text("text", clip=fitz.Rect(pdf_bbox))
                    
                    # Map to our ContentBlock
                    block_type, confidence = self._map_layoutparser_type(block.type)
                    
                    content_block = ContentBlock(
                        block_type=block_type,
                        page_number=page_num + 1,
                        bbox=pdf_bbox,
                        text_content=text.strip(),
                        confidence=float(block.score),
                        properties={
                            'layoutparser_type': block.type,
                            'layoutparser_score': float(block.score)
                        },
                        source_library="LayoutParser",
                        extraction_method="paddledetection_vision_model"
                    )
                    all_blocks.append(content_block)
            
            doc.close()
            
            block_summary = self._generate_block_summary(all_blocks)
            total_chars = sum(len(block.text_content) for block in all_blocks)
            
            return {
                'library_name': 'LayoutParser',
                'success': True,
                'duration_seconds': time.time() - start_time,
                'total_character_count': total_chars,
                'page_count': page_count,
                'content_blocks': all_blocks,
                'block_summary': block_summary,
                'extraction_method': 'vision_model_detection'
            }

        except ImportError:
            msg = 'LayoutParser or its dependencies (paddlepaddle, paddledetection) not installed.'
            self.logger.warning(msg)
            return self._create_error_result('LayoutParser', msg)
        except Exception as e:
            self.logger.error(f"LayoutParser failed: {e}")
            return self._create_error_result('LayoutParser', str(e))

    def _analyze_pymupdf_page(self, page, page_num: int) -> List[ContentBlock]:
        """Analyze a single page with PyMuPDF."""
        blocks = []
        extracted_text_regions = []  # Track already extracted text regions
        
        # Text blocks with coordinates
        text_blocks = page.get_text("blocks")
        for block in text_blocks:
            x0, y0, x1, y1, text, block_num, block_type = block
            
            if text.strip():
                classified_type, confidence, properties = self._classify_block_content(
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
                    extraction_method="text_blocks"
                )
                blocks.append(content_block)
                # Track this text region to avoid duplicates
                extracted_text_regions.append({
                    'bbox': (x0, y0, x1, y1),
                    'text': text.strip(),
                    'text_hash': hash(text.strip().lower().replace('\n', ' ').replace(' ', '').strip())
                })
        
        # Helper function to check if a region overlaps significantly with existing text regions
        def overlaps_with_existing_text(new_bbox, new_text):
            new_x0, new_y0, new_x1, new_y1 = new_bbox
            new_area = (new_x1 - new_x0) * (new_y1 - new_y0)
            new_text_clean = new_text.lower().replace('\n', ' ').replace(' ', '').strip()
            new_text_hash = hash(new_text_clean)
            
            for existing in extracted_text_regions:
                existing_text_clean = existing['text'].lower().replace('\n', ' ').replace(' ', '').strip()
                
                # Check text similarity first (quick check)
                if new_text_hash == existing['text_hash']:
                    return True
                
                # Check if one text is a substring of another (more aggressive)
                if len(new_text_clean) > 5 and len(existing_text_clean) > 5:
                    # Check if new text is contained in existing text or vice versa
                    if (new_text_clean in existing_text_clean or 
                        existing_text_clean in new_text_clean):
                        return True
                    
                    # Check similarity ratio for fuzzy matching
                    similarity = self._text_similarity(new_text, existing['text'])
                    if similarity > 0.7:  # Lowered threshold from 0.8 to 0.7
                        return True
                    
                    # Check for significant word overlap
                    new_words = set(new_text.lower().split())
                    existing_words = set(existing['text'].lower().split())
                    if len(new_words) > 2 and len(existing_words) > 2:
                        overlap_ratio = len(new_words.intersection(existing_words)) / len(new_words.union(existing_words))
                        if overlap_ratio > 0.6:  # 60% word overlap indicates duplicate
                            return True
                
                # Check geometric overlap
                ex_x0, ex_y0, ex_x1, ex_y1 = existing['bbox']
                
                # Calculate intersection
                int_x0 = max(new_x0, ex_x0)
                int_y0 = max(new_y0, ex_y0)
                int_x1 = min(new_x1, ex_x1)
                int_y1 = min(new_y1, ex_y1)
                
                if int_x0 < int_x1 and int_y0 < int_y1:
                    intersection_area = (int_x1 - int_x0) * (int_y1 - int_y0)
                    overlap_ratio = intersection_area / new_area if new_area > 0 else 0
                    
                    # If more than 60% overlap, consider it duplicate (lowered from 70%)
                    if overlap_ratio > 0.6:
                        return True
            
            return False
        
        # Images - Only extract text if it doesn't overlap with existing text regions
        images = page.get_images()
        for img_index, img in enumerate(images):
            try:
                img_rect = page.get_image_bbox(img)
                img_bbox = (img_rect.x0, img_rect.y0, img_rect.x1, img_rect.y1)
                
                # Extract text from image region
                extracted_text = page.get_text("text", clip=img_rect).strip()
                
                # Only add if it's not a duplicate of existing text
                if extracted_text and not overlaps_with_existing_text(img_bbox, extracted_text):
                    text_content = extracted_text
                    confidence = 0.95
                    extracted_text_regions.append({
                        'bbox': img_bbox,
                        'text': extracted_text,
                        'text_hash': hash(extracted_text.lower().replace('\n', ' ').replace(' ', '').strip())
                    })
                else:
                    # Use placeholder if no unique text found
                    text_content = f"Image {img_index + 1}"
                    confidence = 0.5
                
                image_block = ContentBlock(
                    block_type=BlockType.IMAGE,
                    page_number=page_num,
                    bbox=img_bbox,
                    text_content=text_content,
                    confidence=confidence,
                    properties={
                        'image_index': img_index,
                        'width': img_rect.width,
                        'height': img_rect.height,
                        'pymupdf_text_extracted': bool(extracted_text),
                        'extracted_text_length': len(extracted_text) if extracted_text else 0,
                        'is_duplicate': bool(extracted_text and overlaps_with_existing_text(img_bbox, extracted_text))
                    },
                    source_library="PyMuPDF",
                    extraction_method="image_detection_with_text"
                )
                blocks.append(image_block)
            except Exception as e:
                self.logger.debug(f"Could not process image {img_index}: {e}")
        
        # Drawings - Only extract text if it doesn't overlap with existing text regions
        drawings = page.get_drawings()
        for draw_index, drawing in enumerate(drawings):
            try:
                rect = drawing.get("rect")
                if rect:
                    draw_bbox = (rect.x0, rect.y0, rect.x1, rect.y1)
                    
                    # Extract text from drawing region
                    extracted_text = page.get_text("text", clip=rect).strip()
                    
                    # Only add if it's not a duplicate of existing text AND has meaningful content
                    if (extracted_text and 
                        len(extracted_text) > 3 and  # Minimum meaningful length
                        not overlaps_with_existing_text(draw_bbox, extracted_text)):
                        
                        text_content = extracted_text
                        confidence = 0.90
                        extracted_text_regions.append({
                            'bbox': draw_bbox,
                            'text': extracted_text,
                            'text_hash': hash(extracted_text.lower().replace('\n', ' ').replace(' ', '').strip())
                        })
                        
                        drawing_block = ContentBlock(
                            block_type=BlockType.DRAWING,
                            page_number=page_num,
                            bbox=draw_bbox,
                            text_content=text_content,
                            confidence=confidence,
                            properties={
                                'drawing_index': draw_index,
                                'drawing_type': 'vector_graphic',
                                'pymupdf_text_extracted': True,
                                'extracted_text_length': len(extracted_text),
                                'is_duplicate': False
                            },
                            source_library="PyMuPDF",
                            extraction_method="drawing_detection_with_text"
                        )
                        blocks.append(drawing_block)
                    # Skip placeholder drawing blocks entirely if they don't contain unique text
                    
            except Exception as e:
                self.logger.debug(f"Could not process drawing {draw_index}: {e}")
        
        return blocks
        
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using simple word overlap."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0

    def save_pymupdf_extraction_results(self, results: Dict, pdf_path: Path, output_folder: str = "PyMyPDF-extraction"):
        """Save PyMuPDF extraction results to specified folder."""
        try:
            # Create output directory
            extraction_dir = Path(output_folder)
            extraction_dir.mkdir(exist_ok=True)
            
            pdf_name = pdf_path.stem
            pdf_extraction_dir = extraction_dir / pdf_name
            pdf_extraction_dir.mkdir(exist_ok=True)
            
            # Save detailed extraction results
            results_file = pdf_extraction_dir / f"{pdf_name}_pymupdf_extraction.json"
            
            # Prepare serializable results
            serializable_results = {
                'pdf_name': pdf_name,
                'library_name': results.get('library_name', 'PyMuPDF'),
                'extraction_timestamp': time.time(),
                'success': results.get('success', False),
                'duration_seconds': results.get('duration_seconds', 0),
                'total_character_count': results.get('total_character_count', 0),
                'page_count': results.get('page_count', 0),
                'extraction_method': results.get('extraction_method', 'block_based_detection'),
                'content_blocks': []
            }
            
            # Convert ContentBlock objects to dictionaries
            for block in results.get('content_blocks', []):
                block_dict = {
                    'block_type': block.block_type.value if hasattr(block.block_type, 'value') else str(block.block_type),
                    'page_number': block.page_number,
                    'bbox': block.bbox,
                    'text_content': block.text_content,
                    'confidence': block.confidence,
                    'properties': block.properties,
                    'source_library': block.source_library,
                    'extraction_method': block.extraction_method
                }
                serializable_results['content_blocks'].append(block_dict)
            
            # Save JSON results
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            
            # Save text content by page
            pages_dir = pdf_extraction_dir / "pages"
            pages_dir.mkdir(exist_ok=True)
            
            # Group blocks by page
            pages_content = {}
            for block in results.get('content_blocks', []):
                page_num = block.page_number
                if page_num not in pages_content:
                    pages_content[page_num] = []
                
                # Add block info with extracted text
                block_info = f"[{block.block_type.value if hasattr(block.block_type, 'value') else str(block.block_type)}] "
                if block.text_content and not block.text_content.startswith(('Image ', 'Drawing/Graphic ')):
                    # Real extracted text
                    block_info += f"Text: {block.text_content}"
                else:
                    # Placeholder or no text
                    block_info += f"Placeholder: {block.text_content}"
                
                pages_content[page_num].append(block_info)
            
            # Save page content files
            for page_num, content_list in pages_content.items():
                page_file = pages_dir / f"page_{page_num}.txt"
                with open(page_file, 'w', encoding='utf-8') as f:
                    f.write(f"=== PAGE {page_num} - PyMuPDF EXTRACTION ===\n\n")
                    f.write("\n\n".join(content_list))
                    f.write(f"\n\n=== END PAGE {page_num} ===")
            
            # Save summary
            summary_file = pdf_extraction_dir / f"{pdf_name}_extraction_summary.txt"
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(f"PyMuPDF Extraction Summary for: {pdf_name}\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Total Pages: {results.get('page_count', 0)}\n")
                f.write(f"Total Blocks: {len(results.get('content_blocks', []))}\n")
                f.write(f"Total Characters: {results.get('total_character_count', 0)}\n")
                f.write(f"Processing Time: {results.get('duration_seconds', 0):.2f} seconds\n\n")
                
                # Block type statistics
                block_types = {}
                text_extracted_blocks = 0
                for block in results.get('content_blocks', []):
                    block_type = block.block_type.value if hasattr(block.block_type, 'value') else str(block.block_type)
                    block_types[block_type] = block_types.get(block_type, 0) + 1
                    
                    # Check if real text was extracted (not placeholder)
                    if (block.text_content and 
                        not block.text_content.startswith(('Image ', 'Drawing/Graphic ')) and
                        len(block.text_content.strip()) > 0):
                        text_extracted_blocks += 1
                
                f.write("Block Type Distribution:\n")
                for block_type, count in sorted(block_types.items()):
                    f.write(f"  {block_type}: {count}\n")
                
                f.write(f"\nBlocks with Text Extracted: {text_extracted_blocks}/{len(results.get('content_blocks', []))}\n")
                f.write(f"Text Extraction Success Rate: {text_extracted_blocks/len(results.get('content_blocks', [])) * 100:.1f}%\n")
            
            self.logger.info(f"PyMuPDF extraction results saved to: {pdf_extraction_dir}")
            return pdf_extraction_dir
            
        except Exception as e:
            self.logger.error(f"Failed to save PyMuPDF extraction results: {e}")
            return None

    def _analyze_pdfplumber_page(self, page, page_num: int) -> List[ContentBlock]:
        """Analyze a single page with pdfplumber."""
        blocks = []
        
        # Native table detection (pdfplumber's strength)
        tables = page.extract_tables()
        for table_index, table in enumerate(tables):
            if table:
                table_text = self._format_table_content(table)
                bbox = self._estimate_bbox(page, 'table')
                
                table_block = ContentBlock(
                    block_type=BlockType.TABLE,
                    page_number=page_num,
                    bbox=bbox,
                    text_content=table_text,
                    confidence=0.95,  # High confidence for native detection
                    properties={
                        'table_index': table_index,
                        'rows': len(table),
                        'columns': len(table[0]) if table and table[0] else 0,
                        'detection_method': 'native_table_extraction'
                    },
                    source_library="pdfplumber",
                    extraction_method="native_table_detection"
                )
                blocks.append(table_block)
        
        # Figure detection (with error handling for different pdfplumber versions)
        try:
            figures = getattr(page, 'figures', [])
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
                        'figure_properties': str(figure)
                    },
                    source_library="pdfplumber",
                    extraction_method="figure_detection"
                )
                blocks.append(figure_block)
        except AttributeError:
            # pdfplumber version doesn't support figures attribute
            self.logger.debug("pdfplumber version doesn't support figures attribute")
        
        # Alternative: detect images using images attribute if available
        try:
            images = getattr(page, 'images', [])
            for img_index, image in enumerate(images):
                bbox = image.get("bbox", (0, 0, 0, 0))
                
                image_block = ContentBlock(
                    block_type=BlockType.IMAGE,
                    page_number=page_num,
                    bbox=bbox,
                    text_content=f"Image {img_index + 1}",
                    confidence=0.95,
                    properties={
                        'image_index': img_index,
                        'image_properties': str(image)
                    },
                    source_library="pdfplumber",
                    extraction_method="image_detection"
                )
                blocks.append(image_block)
        except AttributeError:
            # pdfplumber version doesn't support images attribute
            self.logger.debug("pdfplumber version doesn't support images attribute")
        
        # Character-based text block analysis
        chars = page.chars
        if chars:
            text_blocks = self._group_chars_into_blocks(chars)
            
            for block_index, char_block in enumerate(text_blocks):
                block_text = ''.join(char['text'] for char in char_block)
                if block_text.strip():
                    # Calculate bbox
                    x0 = min(char['x0'] for char in char_block)
                    y0 = min(char['top'] for char in char_block)
                    x1 = max(char['x1'] for char in char_block)
                    y1 = max(char['bottom'] for char in char_block)
                    bbox = (x0, y0, x1, y1)
                    
                    # Skip if overlaps with existing table
                    if not self._overlaps_with_tables(bbox, blocks):
                        classified_type, confidence, properties = self._classify_block_content(
                            block_text, bbox
                        )
                        
                        text_block = ContentBlock(
                            block_type=classified_type,
                            page_number=page_num,
                            bbox=bbox,
                            text_content=block_text.strip(),
                            confidence=confidence,
                            properties=properties,
                            source_library="pdfplumber",
                            extraction_method="character_analysis"
                        )
                        blocks.append(text_block)
        
        return blocks

    def _analyze_pdfminer_page(self, page_layout, page_num: int) -> List[ContentBlock]:
        """Analyze a single page with PDFMiner."""
        blocks = []
        
        try:
            from pdfminer.layout import LTTextContainer, LTFigure, LTImage
            
            for element in page_layout:
                element_type = type(element).__name__
                bbox = getattr(element, 'bbox', (0, 0, 0, 0))
                
                if isinstance(element, LTTextContainer):
                    text = element.get_text().strip()
                    if text:
                        classified_type, confidence, properties = self._classify_block_content(text, bbox)
                        
                        properties.update({
                            'layout_element_type': element_type,
                            'pdfminer_element': str(element)[:100]
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
                
                elif isinstance(element, LTFigure):
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
                
                elif isinstance(element, LTImage):
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
            self.logger.debug(f"PDFMiner page analysis error: {e}")
        
        return blocks

    def _convert_unstructured_element(self, element, page_num: int) -> Optional[ContentBlock]:
        """Convert Unstructured element to ContentBlock format."""
        try:
            # Get element type
            element_type = type(element).__name__
            text_content = str(element) if element else ""
            
            # Get coordinates if available
            bbox = (0, 0, 0, 0)
            if hasattr(element, 'metadata') and hasattr(element.metadata, 'coordinates'):
                coords = element.metadata.coordinates
                if coords and hasattr(coords, 'points'):
                    # Extract bounding box from coordinate points
                    x_coords = [point[0] for point in coords.points]
                    y_coords = [point[1] for point in coords.points]
                    bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
            
            # Map Unstructured element types to our BlockType
            block_type, confidence = self._map_unstructured_type(element_type)
            
            # Extract additional properties
            properties = {
                'unstructured_type': element_type,
                'text_length': len(text_content),
                'element_id': getattr(element, 'element_id', None)
            }
            
            # Add metadata if available
            if hasattr(element, 'metadata'):
                metadata = element.metadata
                properties.update({
                    'filename': getattr(metadata, 'filename', None),
                    'filetype': getattr(metadata, 'filetype', None),
                    'page_number': getattr(metadata, 'page_number', page_num),
                })
                
                # Add table-specific metadata
                if hasattr(metadata, 'table_as_cells') and metadata.table_as_cells:
                    properties['table_cells'] = len(metadata.table_as_cells)
                    properties['detection_method'] = 'native_table_partitioning'
                
                # Add image-specific metadata
                if hasattr(metadata, 'image_path'):
                    properties['image_path'] = metadata.image_path
                
                # Add font and layout information if available
                if hasattr(metadata, 'emphasized_text_contents'):
                    properties['emphasized_text'] = metadata.emphasized_text_contents
                if hasattr(metadata, 'text_as_html'):
                    properties['has_html_formatting'] = bool(metadata.text_as_html)
            
            return ContentBlock(
                block_type=block_type,
                page_number=page_num,
                bbox=bbox,
                text_content=text_content.strip() if text_content else "",
                confidence=confidence,
                properties=properties,
                source_library="Unstructured",
                extraction_method="element_partitioning"
            )
            
        except Exception as e:
            self.logger.debug(f"Error converting Unstructured element: {e}")
            return None

    def _map_unstructured_type(self, element_type: str) -> Tuple[BlockType, float]:
        """Map Unstructured element types to our BlockType enum."""
        # High confidence mappings for native Unstructured types
        type_mapping = {
            'Title': (BlockType.HEADER, 0.95),
            'Header': (BlockType.HEADER, 0.90),
            'Footer': (BlockType.HEADER, 0.85),  # We don't have FOOTER, use HEADER
            'Table': (BlockType.TABLE, 0.95),
            'Image': (BlockType.IMAGE, 0.95),
            'Figure': (BlockType.FIGURE, 0.90),
            'FigureCaption': (BlockType.FIGURE, 0.85),
            'ListItem': (BlockType.LIST, 0.90),
            'BulletedText': (BlockType.LIST, 0.85),
            'NarrativeText': (BlockType.TEXT_BLOCK, 0.80),
            'Text': (BlockType.TEXT_BLOCK, 0.75),
            'UncategorizedText': (BlockType.TEXT_BLOCK, 0.70),
            'Formula': (BlockType.CHART, 0.85),  # Mathematical formulas as charts
            'PageBreak': (BlockType.UNKNOWN, 0.50),
        }
        
        return type_mapping.get(element_type, (BlockType.UNKNOWN, 0.60))

    def _map_layoutparser_type(self, element_type: str) -> Tuple[BlockType, float]:
        """Map LayoutParser element types to our BlockType enum."""
        # Mappings based on PubLayNet model categories
        type_mapping = {
            'Title': (BlockType.HEADER, 0.90),
            'Text': (BlockType.TEXT_BLOCK, 0.80),
            'Figure': (BlockType.FIGURE, 0.90),
            'Table': (BlockType.TABLE, 0.90),
            'List': (BlockType.LIST, 0.90),
        }
        return type_mapping.get(element_type, (BlockType.UNKNOWN, 0.50))

    def _classify_block_content(self, text: str, bbox: Tuple[float, float, float, float]) -> Tuple[BlockType, float, Dict]:
        """Classify block content based on text and position."""
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        properties = {
            'text_length': len(text),
            'line_count': text.count('\n') + 1,
            'bbox_width': width,
            'bbox_height': height
        }
        
        # Table detection (enhanced pattern matching)
        if self._is_table_structure(text):
            return BlockType.TABLE, 0.85, {**properties, 'table_indicators': self._get_table_indicators(text)}
        
        # Header detection
        elif self._is_header_structure(text, height):
            return BlockType.HEADER, 0.80, {**properties, 'header_type': 'detected_by_structure'}
        
        # List detection
        elif self._is_list_structure(text):
            return BlockType.LIST, 0.85, {**properties, 'list_type': self._get_list_type(text)}
        
        # Chart/Figure reference
        elif self._is_chart_reference(text):
            return BlockType.CHART, 0.75, {**properties, 'chart_references': self._get_chart_refs(text)}
        
        # Default to text block
        return BlockType.TEXT_BLOCK, 0.7, properties

    def _is_table_structure(self, text: str) -> bool:
        """Enhanced table detection."""
        lines = text.split('\n')
        
        # Multiple detection methods
        pipe_separated = sum(1 for line in lines if '|' in line and line.count('|') >= 2)
        tab_separated = sum(1 for line in lines if '\t' in line)
        multi_column = sum(1 for line in lines if re.search(r'\s{3,}', line))
        numeric_columns = sum(1 for line in lines if re.search(r'\d+.*\d+.*\d+', line))
        
        return (pipe_separated >= 2 or 
                (multi_column >= 3 and numeric_columns >= 2) or
                tab_separated >= 2)

    def _is_header_structure(self, text: str, height: float) -> bool:
        """Enhanced header detection."""
        return (len(text) < 100 and 
                (text.isupper() or 
                 re.match(r'^\d+\.\s+[A-Z]', text) or
                 re.match(r'^[A-Z][A-Z\s]{10,50}$', text)) and
                text.count('\n') <= 1 and
                height < 50)

    def _is_list_structure(self, text: str) -> bool:
        """Enhanced list detection."""
        lines = text.split('\n')
        bullet_lines = sum(1 for line in lines if re.match(r'^\s*[•▪▫*-]\s+', line))
        numbered_lines = sum(1 for line in lines if re.match(r'^\s*\d+[.)]\s+', line))
        lettered_lines = sum(1 for line in lines if re.match(r'^\s*[a-z]\)\s+', line))
        
        return (bullet_lines >= 2 or numbered_lines >= 2 or lettered_lines >= 2)

    def _is_chart_reference(self, text: str) -> bool:
        """Detect chart/figure references."""
        return bool(re.search(r'\b(figure|chart|graph|diagram)\s+\d+', text, re.IGNORECASE))

    def _get_table_indicators(self, text: str) -> List[str]:
        """Get specific table structure indicators."""
        indicators = []
        if '|' in text:
            indicators.append('pipe_separated')
        if '\t' in text:
            indicators.append('tab_separated') 
        if re.search(r'\d+.*\d+.*\d+', text):
            indicators.append('numeric_columns')
        if re.search(r'\s{3,}', text):
            indicators.append('space_separated')
        return indicators

    def _get_list_type(self, text: str) -> str:
        """Determine list type."""
        if re.search(r'^\s*[•▪▫*-]\s+', text, re.MULTILINE):
            return 'bulleted'
        elif re.search(r'^\s*\d+[.)]\s+', text, re.MULTILINE):
            return 'numbered'
        elif re.search(r'^\s*[a-z]\)\s+', text, re.MULTILINE):
            return 'lettered'
        return 'unknown'

    def _get_chart_refs(self, text: str) -> List[str]:
        """Extract chart/figure references."""
        return re.findall(r'\b(figure|chart|graph|diagram)\s+\d+', text, re.IGNORECASE)

    def _format_table_content(self, table: List[List]) -> str:
        """Format table content."""
        if not table:
            return ""
        
        lines = []
        for row in table:
            if row:
                clean_row = [str(cell) if cell is not None else "" for cell in row]
                lines.append(" | ".join(clean_row))
        
        return "\n".join(lines)

    def _estimate_bbox(self, page, element_type: str) -> Tuple[float, float, float, float]:
        """Estimate bounding box when not available."""
        if hasattr(page, 'width') and hasattr(page, 'height'):
            return (50, page.height * 0.3, page.width - 50, page.height * 0.7)
        return (0, 0, 500, 100)

    def _group_chars_into_blocks(self, chars: List[Dict]) -> List[List[Dict]]:
        """Group characters into logical blocks."""
        if not chars:
            return []
        
        sorted_chars = sorted(chars, key=lambda c: (c['top'], c['x0']))
        
        blocks = []
        current_block = [sorted_chars[0]]
        
        for char in sorted_chars[1:]:
            prev_char = current_block[-1]
            y_diff = abs(char['top'] - prev_char['top'])
            x_gap = char['x0'] - prev_char['x1']
            
            if y_diff < 5 and x_gap < 50:
                current_block.append(char)
            else:
                if current_block:
                    blocks.append(current_block)
                current_block = [char]
        
        if current_block:
            blocks.append(current_block)
        
        return blocks

    def _overlaps_with_tables(self, bbox: Tuple[float, float, float, float], existing_blocks: List[ContentBlock]) -> bool:
        """Check if bbox overlaps with existing table blocks."""
        for block in existing_blocks:
            if block.block_type == BlockType.TABLE:
                if self._bbox_overlap_ratio(bbox, block.bbox) > 0.3:
                    return True
        return False

    def _bbox_overlap_ratio(self, bbox1: Tuple[float, float, float, float], 
                           bbox2: Tuple[float, float, float, float]) -> float:
        """Calculate overlap ratio between two bounding boxes."""
        x1, y1, x2, y2 = bbox1
        x3, y3, x4, y4 = bbox2
        
        ix1 = max(x1, x3)
        iy1 = max(y1, y3)
        ix2 = min(x2, x4)
        iy2 = min(y2, y4)
        
        if ix1 < ix2 and iy1 < iy2:
            intersection = (ix2 - ix1) * (iy2 - iy1)
            area1 = (x2 - x1) * (y2 - y1)
            
            if area1 > 0:
                return intersection / area1
        
        return 0.0

    def _generate_block_summary(self, blocks: List[ContentBlock]) -> Dict[str, int]:
        """Generate summary of detected blocks."""
        summary = {}
        for block in blocks:
            block_type = block.block_type.value
            summary[block_type] = summary.get(block_type, 0) + 1
        return summary

    def _create_error_result(self, library_name: str, error_message: str) -> Dict:
        """Create error result."""
        return {
            'library_name': library_name,
            'success': False,
            'error_message': error_message,
            'content_blocks': [],
            'block_summary': {}
        }

def compare_all_libraries_blocks(pdf_path: Path) -> Dict:
    """Compare block detection across all libraries."""
    analyzer = EnhancedBlockAnalyzer()
    
    results = {}
    
    # Test each library
    libraries = [
        ('PyMuPDF', analyzer.analyze_pdf_with_pymupdf),
        ('pdfplumber', analyzer.analyze_pdf_with_pdfplumber),
        ('PDFMiner.six', analyzer.analyze_pdf_with_pdfminer),
        ('Unstructured', analyzer.analyze_pdf_with_unstructured),
        ('LayoutParser', analyzer.analyze_pdf_with_layoutparser)
    ]
    
    for library_name, method in libraries:
        logger.info(f"Testing {library_name} with block detection...")
        try:
            result = method(pdf_path)
            results[library_name] = result
            
            if result['success']:
                blocks = len(result['content_blocks'])
                duration = result['duration_seconds']
                logger.info(f"[SUCCESS] {library_name}: {blocks} blocks detected in {duration:.2f}s")
            else:
                logger.warning(f"[ERROR] {library_name}: {result['error_message']}")
                
        except Exception as e:
            logger.error(f"[ERROR] {library_name}: {str(e)}")
            results[library_name] = analyzer._create_error_result(library_name, str(e))
    
    return results

def save_block_analysis_results(results: Dict, output_dir: Path, pdf_path: Path):
    """Save block analysis results and crop block images."""
    pdf_name = pdf_path.stem
    output_dir.mkdir(exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_folder = output_dir / f"block_analysis_{timestamp}"
    result_folder.mkdir(exist_ok=True)
    
    # Create a folder for cropped images
    images_folder = result_folder / "block_images"
    images_folder.mkdir(exist_ok=True)
    
    # Save detailed reports and crop images
    for library_name, result in results.items():
        if result['success']:
            # Save detailed text report
            report_file = result_folder / f"block_analysis_{library_name.replace('.', '_').lower()}_{pdf_name}.txt"
            save_detailed_block_report(result, report_file)
            
            # Save JSON data
            json_file = result_folder / f"block_data_{library_name.replace('.', '_').lower()}_{pdf_name}.json"
            save_block_json_data(result, json_file)

            # Crop and save images for each block
            crop_and_save_block_images(result, pdf_path, images_folder)

    # Save comparison report
    comparison_file = result_folder / f"block_comparison_{pdf_name}.txt"
    save_block_comparison_report(results, comparison_file)
    
    # Save detailed CSV for PyMuPDF
    save_pymupdf_csv_report(results, result_folder, pdf_name)

    logger.info(f"Block analysis results saved to: {result_folder}")
    return result_folder

def save_detailed_block_report(result: Dict, output_file: Path):
    """Save detailed block analysis report."""
    lines = [
        "ENHANCED BLOCK-BASED CONTENT ANALYSIS REPORT",
        "=" * 80,
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Library: {result['library_name']}",
        f"Extraction Method: {result.get('extraction_method', 'unknown')}",
        "",
        f"SUCCESS: {result['success']}",
        f"Total Characters: {result.get('total_character_count', 0):,}",
        f"Total Pages: {result.get('page_count', 0)}",
        f"Total Blocks Detected: {len(result.get('content_blocks', []))}",
        f"Processing Time: {result.get('duration_seconds', 0):.2f} seconds",
        "",
        "BLOCK TYPE SUMMARY:",
        "-" * 40,
    ]
    
    block_summary = result.get('block_summary', {})
    total_blocks = len(result.get('content_blocks', []))
    
    for block_type, count in sorted(block_summary.items()):
        percentage = (count / total_blocks * 100) if total_blocks > 0 else 0
        lines.append(f"{block_type}: {count} blocks ({percentage:.1f}%)")
    
    lines.extend([
        "",
        "PAGE-BY-PAGE BLOCK ANALYSIS:",
        "=" * 80
    ])
    
    # Group blocks by page
    blocks = result.get('content_blocks', [])
    pages = {}
    for block in blocks:
        page_num = block.page_number
        if page_num not in pages:
            pages[page_num] = []
        pages[page_num].append(block)
    
    for page_num in sorted(pages.keys()):
        lines.append(f"\n--- PAGE {page_num} ---")
        page_blocks = pages[page_num]
        
        # Page summary
        page_types = {}
        for block in page_blocks:
            block_type = block.block_type.value
            page_types[block_type] = page_types.get(block_type, 0) + 1
        
        lines.append("Page Block Summary:")
        for block_type, count in sorted(page_types.items()):
            lines.append(f"  {block_type}: {count}")
        
        lines.append("\nDetailed Blocks:")
        for i, block in enumerate(page_blocks, 1):
            lines.extend([
                f"\n  Block {i}:",
                f"    Type: {block.block_type.value}",
                f"    Confidence: {block.confidence:.2f}",
                f"    Coordinates: ({block.bbox[0]:.1f}, {block.bbox[1]:.1f}, {block.bbox[2]:.1f}, {block.bbox[3]:.1f})",
                f"    Size: {block.bbox[2] - block.bbox[0]:.1f} x {block.bbox[3] - block.bbox[1]:.1f}",
                f"    Length: {len(block.text_content)} characters",
                f"    Extraction Method: {block.extraction_method}",
            ])
            
            if block.properties:
                lines.append("    Properties:")
                for key, value in block.properties.items():
                    lines.append(f"      {key}: {value}")
            
            # Text preview
            preview = block.text_content[:200].replace('\n', ' ').strip()
            if len(block.text_content) > 200:
                preview += "..."
            lines.append(f"    Preview: {preview}")
    
    try:
        output_file.write_text('\n'.join(lines), encoding='utf-8', errors='replace')
        logger.info(f"Block analysis report saved: {output_file.name}")
    except Exception as e:
        logger.error(f"Failed to save report: {e}")

def save_block_json_data(result: Dict, output_file: Path):
    """Save block data as JSON."""
    try:
        # Convert blocks to serializable format
        serializable_blocks = []
        for block in result.get('content_blocks', []):
            block_dict = {
                'block_type': block.block_type.value,
                'page_number': block.page_number,
                'bbox': block.bbox,
                'text_content': block.text_content,
                'confidence': block.confidence,
                'properties': block.properties,
                'source_library': block.source_library,
                'extraction_method': block.extraction_method
            }
            serializable_blocks.append(block_dict)
        
        data = {
            'library_name': result['library_name'],
            'success': result['success'],
            'duration_seconds': result.get('duration_seconds', 0),
            'total_character_count': result.get('total_character_count', 0),
            'page_count': result.get('page_count', 0),
            'block_summary': result.get('block_summary', {}),
            'extraction_method': result.get('extraction_method', 'unknown'),
            'blocks': serializable_blocks
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Block JSON data saved: {output_file.name}")
    except Exception as e:
        logger.error(f"Failed to save JSON data: {e}")

def save_block_comparison_report(results: Dict, output_file: Path):
    """Save comparison report across libraries."""
    lines = [
        "BLOCK-BASED PDF EXTRACTION COMPARISON",
        "=" * 80,
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "LIBRARY COMPARISON:",
        "-" * 40,
    ]
    
    # Summary table
    header = f"{'Library':<15} {'Success':<8} {'Time(s)':<8} {'Blocks':<8} {'Tables':<8} {'Images':<8} {'Figures':<8}"
    lines.append(header)
    lines.append("-" * 80)
    
    for library_name, result in results.items():
        if result['success']:
            block_summary = result.get('block_summary', {})
            tables = block_summary.get('TABLE', 0)
            images = block_summary.get('IMAGE', 0)
            figures = block_summary.get('FIGURE', 0) + block_summary.get('DRAWING', 0)
            total_blocks = len(result.get('content_blocks', []))
            duration = result.get('duration_seconds', 0)
            
            row = f"{library_name:<15} {'YES':<8} {duration:<8.2f} {total_blocks:<8} {tables:<8} {images:<8} {figures:<8}"
        else:
            row = f"{library_name:<15} {'NO':<8} {'ERROR':<8} {'ERROR':<8} {'ERROR':<8} {'ERROR':<8} {'ERROR':<8}"
        
        lines.append(row)
    
    lines.extend([
        "",
        "BLOCK DETECTION CAPABILITIES:",
        "-" * 40,
        "PyMuPDF:",
        "  ✅ Excellent text block detection with coordinates",
        "  ✅ Native image detection with bounding boxes",
        "  ✅ Vector graphics/drawing detection",
        "  ✅ Fast and accurate block structure analysis",
        "",
        "pdfplumber:",
        "  ✅ Outstanding native table detection",
        "  ✅ Figure detection with coordinate information",
        "  ✅ Character-level text block analysis",
        "  ✅ Object-based page understanding",
        "",
        "PDFMiner.six:",
        "  ✅ Advanced layout object analysis",
        "  ✅ Precise element type detection (LTTextContainer, LTFigure, etc.)",
        "  ✅ Comprehensive coordinate information",
        "  ✅ Deep layout understanding",
        "",
        "Unstructured:",
        "  ✅ Advanced AI-powered element detection",
        "  ✅ Native table structure understanding",
        "  ✅ Sophisticated document partitioning",
        "  ✅ Multi-modal content analysis (text, images, tables)",
        "  ✅ High-resolution processing strategy",
        "  ✅ Built-in element type classification",
        "",
        "LayoutParser:",
        "  ✅ Computer vision-based layout detection (via PaddleDetection)",
        "  ✅ Excellent for complex layouts with mixed content (e.g., scientific papers)",
        "  ✅ Identifies visual elements like titles, figures, tables, and lists",
        "  ✅ Model-based, can be fine-tuned on custom datasets for higher accuracy",
        "",
        "RECOMMENDATIONS:",
        "-" * 40,
        "🏆 For TABLE detection: pdfplumber or Unstructured (both excellent)",
        "🏆 For IMAGE/DRAWING detection: PyMuPDF (comprehensive graphics)",
        "🏆 For LAYOUT analysis: Unstructured or PDFMiner.six (advanced AI vs precise objects)",
        "🏆 For VISION-BASED analysis: LayoutParser (powerful for visual structure)",
        "🏆 For SPEED with good blocks: PyMuPDF (fast and accurate)",
        "🏆 For AI-POWERED analysis: Unstructured (most advanced element detection)",
        "",
        "💡 Block-based detection is much more accurate than regex patterns!",
        "   Each library detects actual page elements rather than guessing from text."
    ])
    
    try:
        output_file.write_text('\n'.join(lines), encoding='utf-8', errors='replace')
        logger.info(f"Block comparison report saved: {output_file.name}")
    except Exception as e:
        logger.error(f"Failed to save comparison report: {e}")

def save_pymupdf_csv_report(results: Dict, result_folder: Path, pdf_name: str):
    """Saves a detailed CSV report for PyMuPDF block analysis results."""
    pymupdf_result = results.get('PyMuPDF')
    if not pymupdf_result or not pymupdf_result['success']:
        logger.info("PyMuPDF results not available or failed, skipping CSV report generation.")
        return

    csv_file = result_folder / f"pymupdf_analysis_summary_{pdf_name}.csv"
    headers = [
        "page_number", "block_index", "block_type", "confidence", 
        "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1", 
        "text_length", "text_preview", "image_path"
    ]

    try:
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

            for i, block in enumerate(pymupdf_result.get('content_blocks', [])):
                page_num = block.page_number
                block_type = block.block_type.value
                bbox = block.bbox if hasattr(block, 'bbox') else (0,0,0,0)
                
                # Construct image path
                sanitized_block_type = re.sub(r'[^a-zA-Z0-9_-]', '', block_type)
                image_filename = f"pymupdf_p{page_num}_block{i+1}_{sanitized_block_type}.png"
                image_path = Path("block_images") / image_filename

                preview = block.text_content[:100].replace('\n', ' ').strip()

                row = [
                    page_num,
                    i + 1,
                    block_type,
                    f"{block.confidence:.2f}",
                    f"{bbox[0]:.2f}", f"{bbox[1]:.2f}", f"{bbox[2]:.2f}", f"{bbox[3]:.2f}",
                    len(block.text_content),
                    preview,
                    str(image_path)
                ]
                writer.writerow(row)
        
        logger.info(f"PyMuPDF CSV report saved to: {csv_file}")

    except Exception as e:
        logger.error(f"Failed to save PyMuPDF CSV report: {e}")

def crop_and_save_block_images(result: Dict, pdf_path: Path, images_folder: Path):
    """Crop and save images for each detected block."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.warning("PyMuPDF (fitz) is not installed. Skipping block image cropping.")
        return

    library_name = result['library_name'].replace('.', '_').lower()
    
    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        logger.error(f"Failed to open PDF {pdf_path} with PyMuPDF: {e}")
        return

    for i, block in enumerate(result.get('content_blocks', [])):
        page_num = block.page_number
        if not hasattr(block, 'bbox') or not block.bbox:
            continue
            
        bbox = block.bbox
        block_type = block.block_type.value
        
        if page_num > 0 and page_num <= len(doc):
            page = doc[page_num - 1]  # page_number is 1-based, doc index is 0-based
            
            # fitz.Rect needs (x0, y0, x1, y1)
            rect = fitz.Rect(bbox)
            
            # Check if rect is valid and within page bounds
            page_bounds = page.rect
            if rect.is_empty:
                continue

            # Intersect with page bounds to handle partially visible blocks
            rect.intersect(page_bounds)
            if rect.is_empty:
                logger.debug(f"Block {i+1} on page {page_num} has out-of-bounds bbox: {bbox}. Skipping crop.")
                continue

            try:
                # Use higher DPI for better quality
                pix = page.get_pixmap(clip=rect, dpi=200)
                
                # Sanitize block_type for filename
                sanitized_block_type = re.sub(r'[^a-zA-Z0-9_-]', '', block_type)

                image_filename = f"{library_name}_p{page_num}_block{i+1}_{sanitized_block_type}.png"
                image_path = images_folder / image_filename
                
                if pix.width > 0 and pix.height > 0:
                    pix.save(str(image_path))
                else:
                    logger.debug(f"Skipping empty pixmap for block {i+1} on page {page_num}.")

            except Exception as e:
                logger.warning(f"Failed to crop image for block {i+1} on page {page_num} with bbox {bbox}: {e}")
    
    doc.close()
    logger.info(f"Cropped block images for {result['library_name']} saved in {images_folder.name}")

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced block-based PDF content analysis")
    parser.add_argument("pdf_file", help="Path to PDF file")
    parser.add_argument("--output-dir", "-o", default="block_analysis_results",
                       help="Output directory")
    parser.add_argument("--library", "-l", 
                       choices=['pymupdf', 'pdfplumber', 'pdfminer', 'unstructured', 'layoutparser'],
                       help="Test specific library only")
    
    args = parser.parse_args()
    
    pdf_path = Path(args.pdf_file)
    output_dir = Path(args.output_dir)
    
    if not pdf_path.exists():
        print(f"Error: PDF file not found: {pdf_path}")
        return
    
    if args.library:
        # Test specific library
        analyzer = EnhancedBlockAnalyzer()
        
        if args.library == 'pymupdf':
            result = analyzer.analyze_pdf_with_pymupdf(pdf_path)
        elif args.library == 'pdfplumber':
            result = analyzer.analyze_pdf_with_pdfplumber(pdf_path)
        elif args.library == 'pdfminer':
            result = analyzer.analyze_pdf_with_pdfminer(pdf_path)
        elif args.library == 'unstructured':
            result = analyzer.analyze_pdf_with_unstructured(pdf_path)
        elif args.library == 'layoutparser':
            result = analyzer.analyze_pdf_with_layoutparser(pdf_path)
        
        if result['success']:
            print(f"✅ {result['library_name']}: {len(result['content_blocks'])} blocks detected")
            print(f"   Processing time: {result['duration_seconds']:.2f}s")
            print("\n📊 Block types found:")
            for block_type, count in sorted(result['block_summary'].items()):
                print(f"   {block_type}: {count}")
            
            # Save results
            results = {result['library_name']: result}
            save_block_analysis_results(results, output_dir, pdf_path)
        else:
            print(f"❌ {result['library_name']}: {result['error_message']}")
    
    else:
        # Test all libraries
        print("Testing all libraries with enhanced block detection...")
        results = compare_all_libraries_blocks(pdf_path)
        
        # Print summary
        print("\n📊 BLOCK DETECTION SUMMARY:")
        print("-" * 50)
        for library_name, result in results.items():
            if result['success']:
                blocks = len(result['content_blocks'])
                duration = result['duration_seconds']
                tables = result['block_summary'].get('TABLE', 0)
                images = result['block_summary'].get('IMAGE', 0)
                figures = result['block_summary'].get('FIGURE', 0) + result['block_summary'].get('DRAWING', 0)
                
                print(f"{library_name}: {blocks} blocks ({duration:.2f}s)")
                print(f"  Tables: {tables}, Images: {images}, Figures: {figures}")
            else:
                print(f"{library_name}: ERROR - {result['error_message']}")
        
        # Save results
        result_folder = save_block_analysis_results(results, output_dir, pdf_path)
        print(f"\n📁 Results saved to: {result_folder}")
        
        # Recommendations
        successful_results = {k: v for k, v in results.items() if v['success']}
        if successful_results:
            best_tables = max(successful_results.items(), 
                            key=lambda x: x[1]['block_summary'].get('TABLE', 0))
            best_images = max(successful_results.items(), 
                            key=lambda x: x[1]['block_summary'].get('IMAGE', 0) + x[1]['block_summary'].get('DRAWING', 0) + x[1]['block_summary'].get('FIGURE', 0))
            best_vision = successful_results.get('LayoutParser')
            
            print(f"\n🎯 RECOMMENDATIONS:")
            print(f"   Best for tables: {best_tables[0]} ({best_tables[1]['block_summary'].get('TABLE', 0)} tables)")
            print(f"   Best for images/graphics: {best_images[0]} ({best_images[1]['block_summary'].get('IMAGE', 0) + best_images[1]['block_summary'].get('DRAWING', 0) + best_images[1]['block_summary'].get('FIGURE', 0)} elements)")
            if best_vision:
                print(f"   Best for visual layout: LayoutParser ({len(best_vision['content_blocks'])} blocks detected)")

if __name__ == "__main__":
    main() 


