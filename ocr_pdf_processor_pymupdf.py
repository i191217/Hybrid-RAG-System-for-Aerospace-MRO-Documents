#!/usr/bin/env python3
"""
OCR-based PDF Content Extractor (PyMuPDF Version)
Uses PyMuPDF for PDF to image conversion - No Poppler required!
Extracts text from PDFs using OCR technology to handle scanned documents,
images, and complex layouts that traditional text extraction methods miss.
"""

import os
import sys
import io
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import tempfile
import json
from datetime import datetime
import cv2
import numpy as np

# PDF and image processing
import fitz  # PyMuPDF
from PIL import Image, ImageEnhance, ImageFilter

# OCR libraries
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    # Test if paddle dependencies are available
    import paddleocr
    PADDLEOCR_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    PADDLEOCR_AVAILABLE = False

# Progress tracking
from tqdm import tqdm


class ImagePreprocessor:
    """Advanced image preprocessing for better OCR results."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def enhance_image(self, image: Image.Image) -> Image.Image:
        """Apply various enhancement techniques to improve OCR accuracy."""
        try:
            # Convert to grayscale if needed
            if image.mode != 'L':
                image = image.convert('L')
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.5)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(2.0)
            
            # Apply denoising
            image = image.filter(ImageFilter.MedianFilter(size=3))
            
            # Convert to numpy array for OpenCV processing
            img_array = np.array(image)
            
            # Apply adaptive thresholding
            img_array = cv2.adaptiveThreshold(
                img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # Morphological operations to clean up
            kernel = np.ones((1, 1), np.uint8)
            img_array = cv2.morphologyEx(img_array, cv2.MORPH_CLOSE, kernel)
            
            # Convert back to PIL Image
            return Image.fromarray(img_array)
            
        except Exception as e:
            self.logger.warning(f"Image enhancement failed: {e}")
            return image


class OCREngine:
    """Multi-engine OCR processor with fallback mechanisms."""
    
    def __init__(self, languages: List[str] = None):
        self.logger = logging.getLogger(__name__)
        self.languages = languages or ['en']
        self.preprocessor = ImagePreprocessor()
        
        # Initialize available OCR engines
        self.engines = {}
        self._initialize_engines()
    
    def _initialize_engines(self):
        """Initialize available OCR engines."""
        if TESSERACT_AVAILABLE:
            try:
                # Test tesseract installation
                pytesseract.get_tesseract_version()
                self.engines['tesseract'] = True
                self.logger.info("Tesseract OCR initialized successfully")
            except Exception as e:
                self.logger.warning(f"Tesseract initialization failed: {e}")
                self.engines['tesseract'] = False
        
        if EASYOCR_AVAILABLE:
            try:
                self.easy_reader = easyocr.Reader(self.languages)
                self.engines['easyocr'] = True
                self.logger.info("EasyOCR initialized successfully")
            except Exception as e:
                self.logger.warning(f"EasyOCR initialization failed: {e}")
                self.engines['easyocr'] = False
        
        if PADDLEOCR_AVAILABLE:
            try:
                self.paddle_ocr = paddleocr.PaddleOCR(
                    use_angle_cls=True, 
                    lang='en', 
                    # show_log=False
                )
                self.engines['paddleocr'] = True
                self.logger.info("PaddleOCR initialized successfully")
            except Exception as e:
                # self.logger.warning(f"PaddleOCR initialization failed: {e}")
                self.logger.info(f"PaddleOCR initialization failed: {e}")
                self.engines['paddleocr'] = False
        
        if not any(self.engines.values()):
            raise RuntimeError("No OCR engines available. Please install at least one of: pytesseract, easyocr, paddleocr")
    
    def extract_text_tesseract(self, image: Image.Image) -> str:
        """Extract text using Tesseract OCR."""
        try:
            # Enhance image for better OCR
            enhanced_image = self.preprocessor.enhance_image(image)
            
            # Configure Tesseract
            config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!@#$%^&*()_+-=[]{}|;:,.<>?/~` '
            
            text = pytesseract.image_to_string(enhanced_image, config=config)
            return text.strip()
        except Exception as e:
            self.logger.error(f"Tesseract OCR failed: {e}")
            return ""
    
    def extract_text_easyocr(self, image: Image.Image) -> str:
        """Extract text using EasyOCR."""
        try:
            # Convert PIL to numpy array
            img_array = np.array(image)
            
            # Perform OCR
            results = self.easy_reader.readtext(img_array, detail=0)
            
            # Join all text results
            text = ' '.join(results)
            return text.strip()
        except Exception as e:
            self.logger.error(f"EasyOCR failed: {e}")
            return ""
    
    def extract_text_paddleocr(self, image: Image.Image) -> str:
        """Extract text using PaddleOCR."""
        try:
            # Convert PIL to numpy array
            img_array = np.array(image)
            
            # Perform OCR
            # results = self.paddle_ocr.ocr(img_array, cls=True)
            # results = self.paddle_ocr.ocr(img_array)
            results = self.paddle_ocr.predict(img_array)
            
            # Extract text from results
            text_parts = []
            if results and results[0]:
                for line in results[0]:
                    if line and len(line) > 1:
                        text_parts.append(line[1][0])
            
            text = ' '.join(text_parts)
            return text.strip()
        except Exception as e:
            self.logger.error(f"PaddleOCR failed: {e}")
            return ""
    
    def extract_text_from_image(self, image: Image.Image, method: str = 'auto') -> str:
        """Extract text from image using specified method or auto-selection."""
        if method == 'auto':
            # Try engines in order of preference
            methods = ['tesseract', 'easyocr', 'paddleocr']
            for m in methods:
                if self.engines.get(m):
                    text = self.extract_text_from_image(image, m)
                    if text.strip():
                        return text
            return ""
        
        elif method == 'tesseract' and self.engines.get('tesseract'):
            return self.extract_text_tesseract(image)
        
        elif method == 'easyocr' and self.engines.get('easyocr'):
            return self.extract_text_easyocr(image)
        
        elif method == 'paddleocr' and self.engines.get('paddleocr'):
            return self.extract_text_paddleocr(image)
        
        else:
            self.logger.warning(f"OCR method '{method}' not available")
            return ""


class OCRPDFProcessor:
    """Main class for processing PDFs with OCR using PyMuPDF (no Poppler required)."""
    
    def __init__(self, languages: List[str] = None, dpi: int = 300, ocr_method: str = 'auto'):
        self.logger = self._setup_logging()
        self.languages = languages or ['en']
        self.dpi = dpi
        self.ocr_method = ocr_method
        
        # Initialize OCR engine
        self.ocr_engine = OCREngine(self.languages)
        
        self.logger.info(f"OCR PDF Processor (PyMuPDF) initialized with:")
        self.logger.info(f"  - Languages: {self.languages}")
        self.logger.info(f"  - DPI: {self.dpi}")
        self.logger.info(f"  - OCR Method: {self.ocr_method}")
        self.logger.info(f"  - Available engines: {list(self.ocr_engine.engines.keys())}")
        self.logger.info(f"  - Using PyMuPDF for PDF conversion (No Poppler required!)")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('ocr_processor_pymupdf.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        return logging.getLogger(__name__)
    
    def pdf_to_images_pymupdf(self, pdf_path: Path) -> List[Image.Image]:
        """Convert PDF pages to images using PyMuPDF (no Poppler required)."""
        try:
            self.logger.info(f"Converting PDF to images using PyMuPDF: {pdf_path}")
            
            # Open PDF with PyMuPDF
            doc = fitz.open(str(pdf_path))
            images = []
            
            # Convert each page to image
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Create transformation matrix for DPI
                mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)
                
                # Render page as image
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to PIL Image
                img_data = pix.tobytes("ppm")
                image = Image.open(io.BytesIO(img_data))
                images.append(image)
                
                self.logger.debug(f"Converted page {page_num + 1} to image")
            
            doc.close()
            self.logger.info(f"Converted {len(images)} pages to images using PyMuPDF")
            return images
            
        except Exception as e:
            self.logger.error(f"PyMuPDF PDF to image conversion failed: {e}")
            return []
    
    def extract_text_from_pdf(self, pdf_path: Path) -> Dict:
        """Extract text from PDF using OCR."""
        try:
            self.logger.info(f"Starting OCR extraction for: {pdf_path}")
            
            # Convert PDF to images using PyMuPDF
            images = self.pdf_to_images_pymupdf(pdf_path)
            
            if not images:
                return {
                    'success': False,
                    'error': 'Failed to convert PDF to images using PyMuPDF',
                    'text': '',
                    'pages': 0
                }
            
            # Extract text from each page
            page_texts = []
            failed_pages = []
            
            for page_num, image in enumerate(tqdm(images, desc="Processing pages")):
                try:
                    self.logger.debug(f"Processing page {page_num + 1}")
                    
                    # Extract text using OCR
                    text = self.ocr_engine.extract_text_from_image(image, self.ocr_method)
                    
                    if text.strip():
                        page_texts.append(f"=== PAGE {page_num + 1} ===\n{text}\n")
                        self.logger.debug(f"Page {page_num + 1}: {len(text)} characters extracted")
                    else:
                        self.logger.warning(f"Page {page_num + 1}: No text extracted")
                        failed_pages.append(page_num + 1)
                        page_texts.append(f"=== PAGE {page_num + 1} ===\n[No text extracted]\n")
                
                except Exception as e:
                    self.logger.error(f"Page {page_num + 1}: OCR failed - {e}")
                    failed_pages.append(page_num + 1)
                    page_texts.append(f"=== PAGE {page_num + 1} ===\n[OCR extraction failed: {e}]\n")
            
            # Combine all page texts
            combined_text = '\n'.join(page_texts)
            
            # Prepare result
            result = {
                'success': True,
                'text': combined_text,
                'pages': len(images),
                'pages_processed': len(images) - len(failed_pages),
                'failed_pages': failed_pages,
                'stats': {
                    'total_characters': len(combined_text),
                    'method_used': self.ocr_method,
                    'dpi_used': self.dpi
                }
            }
            
            self.logger.info(f"OCR extraction completed:")
            self.logger.info(f"  - Total pages: {result['pages']}")
            self.logger.info(f"  - Successfully processed: {result['pages_processed']}")
            self.logger.info(f"  - Failed pages: {len(failed_pages)}")
            self.logger.info(f"  - Total characters: {result['stats']['total_characters']}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"OCR extraction failed for {pdf_path}: {e}")
            return {
                'success': False,
                'error': str(e),
                'text': '',
                'pages': 0
            }
    
    def process_directory(self, input_dir: Path, output_dir: Path = None) -> Dict:
        """Process all PDFs in a directory."""
        try:
            self.logger.info(f"Processing directory: {input_dir}")
            
            # Find all PDF files
            pdf_files = list(input_dir.glob("*.pdf"))
            
            if not pdf_files:
                self.logger.warning(f"No PDF files found in {input_dir}")
                return {'success': False, 'error': 'No PDF files found'}
            
            self.logger.info(f"Found {len(pdf_files)} PDF files")
            
            # Setup output directory
            if output_dir is None:
                output_dir = input_dir / "ocr_output_pymupdf"
            
            output_dir.mkdir(exist_ok=True)
            
            # Process each PDF
            results = {}
            total_chars = 0
            total_pages = 0
            
            for pdf_file in pdf_files:
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"Processing: {pdf_file.name}")
                self.logger.info(f"{'='*60}")
                
                # Extract text
                result = self.extract_text_from_pdf(pdf_file)
                
                if result['success']:
                    # Save extracted text
                    output_file = output_dir / f"{pdf_file.stem}_ocr_extracted.txt"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(result['text'])
                    
                    self.logger.info(f"Saved extracted text to: {output_file}")
                    
                    total_chars += result['stats']['total_characters']
                    total_pages += result['pages']
                
                results[pdf_file.name] = result
            
            # Save summary report
            summary = {
                'timestamp': datetime.now().isoformat(),
                'input_directory': str(input_dir),
                'output_directory': str(output_dir),
                'total_files': len(pdf_files),
                'successful_files': sum(1 for r in results.values() if r['success']),
                'total_pages_processed': total_pages,
                'total_characters_extracted': total_chars,
                'ocr_settings': {
                    'method': self.ocr_method,
                    'dpi': self.dpi,
                    'languages': self.languages,
                    'pdf_converter': 'PyMuPDF'
                },
                'file_results': results
            }
            
            summary_file = output_dir / "ocr_processing_summary_pymupdf.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"\n{'='*60}")
            self.logger.info("PROCESSING COMPLETE")
            self.logger.info(f"{'='*60}")
            self.logger.info(f"Total files processed: {len(pdf_files)}")
            self.logger.info(f"Successful extractions: {summary['successful_files']}")
            self.logger.info(f"Total pages: {total_pages}")
            self.logger.info(f"Total characters extracted: {total_chars}")
            self.logger.info(f"Results saved to: {output_dir}")
            self.logger.info(f"Summary report: {summary_file}")
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Directory processing failed: {e}")
            return {'success': False, 'error': str(e)}


def main():
    """Main function to run OCR PDF processing."""
    import argparse
    import io  # Add missing import
    
    parser = argparse.ArgumentParser(description="OCR-based PDF Content Extractor (PyMuPDF - No Poppler Required)")
    parser.add_argument(
        "--input-dir", 
        type=str, 
        default="RAG_input",
        help="Input directory containing PDF files (default: RAG_input)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        help="Output directory for extracted text files (default: input_dir/ocr_output_pymupdf)"
    )
    parser.add_argument(
        "--dpi", 
        type=int, 
        default=300,
        help="DPI for PDF to image conversion (default: 300)"
    )
    parser.add_argument(
        "--method", 
        type=str, 
        choices=['auto', 'tesseract', 'easyocr', 'paddleocr'],
        default='auto',
        help="OCR method to use (default: auto)"
    )
    parser.add_argument(
        "--languages", 
        type=str, 
        nargs='+',
        default=['en'],
        help="Languages for OCR (default: ['en'])"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    # Validate input directory
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return False
    
    try:
        # Initialize processor
        print("🚀 Initializing OCR PDF Processor (PyMuPDF - No Poppler Required)...")
        processor = OCRPDFProcessor(
            languages=args.languages,
            dpi=args.dpi,
            ocr_method=args.method
        )
        
        # Process directory
        print(f"📁 Processing directory: {input_dir.absolute()}")
        result = processor.process_directory(input_dir, output_dir)
        
        if result.get('success', False):
            print("\n✅ OCR processing completed successfully!")
            return True
        else:
            print(f"\n❌ OCR processing failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 