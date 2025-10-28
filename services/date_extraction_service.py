#!/usr/bin/env python3
"""
Date Extraction Service for PDF Creation Date Metadata.
Extracts creation dates from PDF metadata and filesystem fallbacks.
"""

import logging
import platform
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Union
import tempfile

try:
    from PyPDF2 import PdfReader
    from PyPDF2.generic import IndirectObject
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


class DateExtractionService:
    """Service for extracting creation dates from documents, primarily PDFs."""
    
    def __init__(self):
        """Initialize the date extraction service."""
        self.logger = logging.getLogger(__name__)
        
        if not PDF_AVAILABLE:
            self.logger.warning("PyPDF2 not available. PDF date extraction will be disabled.")
        
        self.logger.info("DateExtractionService initialized")

    def extract_creation_date(self, file_path: Union[str, Path], file_content: Optional[bytes] = None) -> Optional[datetime]:
        """
        Extract creation date from a file.
        
        Args:
            file_path: Path to the file or filename
            file_content: Optional file content as bytes (for temporary files)
            
        Returns:
            Creation datetime if found, None otherwise
        """
        try:
            file_path = Path(file_path)
            
            # Handle different file types
            if file_path.suffix.lower() == '.pdf':
                return self._extract_pdf_creation_date(file_path, file_content)
            else:
                # For non-PDF files, we could add other extraction methods
                self.logger.debug(f"Non-PDF file {file_path.suffix}, no date extraction implemented")
                return None
                
        except Exception as e:
            self.logger.error(f"Error extracting creation date from {file_path}: {e}")
            return None

    def _extract_pdf_creation_date(self, file_path: Path, file_content: Optional[bytes] = None) -> Optional[datetime]:
        """
        Extract creation date from PDF metadata.
        
        Args:
            file_path: Path to the PDF file
            file_content: Optional PDF content as bytes
            
        Returns:
            Creation datetime from PDF metadata, None if not found
        """
        if not PDF_AVAILABLE:
            self.logger.warning("PyPDF2 not available, cannot extract PDF creation date")
            return None
        
        try:
            # If we have file content, create a temporary file
            if file_content:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                    temp_file.write(file_content)
                    temp_path = Path(temp_file.name)
                
                try:
                    creation_date = self._get_pdf_metadata_creation(temp_path)
                finally:
                    # Clean up temporary file
                    temp_path.unlink(missing_ok=True)
                
                return creation_date
            else:
                # Use the actual file path
                return self._get_pdf_metadata_creation(file_path)
                
        except Exception as e:
            self.logger.error(f"Error extracting PDF creation date from {file_path}: {e}")
            return None

    def _get_pdf_metadata_creation(self, path: Path) -> Optional[datetime]:
        """
        Extract creation date from PDF metadata.
        
        Args:
            path: Path to the PDF file
            
        Returns:
            Creation datetime if found in metadata, None otherwise
        """
        try:
            self.logger.debug(f"Attempting to read PDF metadata creation date from {path}")
            
            reader = PdfReader(str(path))
            info = getattr(reader, 'metadata', None) or getattr(reader, 'documentInfo', None)
            
            if not info:
                self.logger.debug("No metadata found in PDF")
                return None
            
            # Try different possible keys for creation date
            raw_date = None
            for key in ["/CreationDate", "CreationDate", "/CreationDate", "/ModDate", "ModDate"]:
                raw_date = info.get(key)
                if raw_date:
                    self.logger.debug(f"Found date in metadata key: {key}")
                    break
            
            if not raw_date:
                self.logger.debug("No creation date field found in PDF metadata")
                return None
            
            self.logger.debug(f"Raw metadata date before resolution: {raw_date!r}")
            
            # Handle IndirectObject resolution
            if isinstance(raw_date, IndirectObject):
                self.logger.debug("Found IndirectObject for date, resolving it")
                raw_date = raw_date.get_object()
                self.logger.debug(f"Raw metadata date after resolution: {raw_date!r}")
            
            # Parse the PDF date string
            return self._parse_pdf_creation_date(str(raw_date))
            
        except Exception as e:
            self.logger.error(f"Error reading PDF metadata from {path}: {e}")
            return None

    def _parse_pdf_creation_date(self, raw_date: str) -> Optional[datetime]:
        """
        Parse PDF creation date string.
        
        Args:
            raw_date: Raw date string from PDF metadata
            
        Returns:
            Parsed datetime object, None if parsing fails
        """
        try:
            self.logger.debug(f"Parsing PDF date string: {raw_date!r}")
            
            if not raw_date or not isinstance(raw_date, str):
                self.logger.debug("Invalid date string")
                return None
            
            # PDF date format: D:YYYYMMDDHHmmSSOHH'mm'
            if not raw_date.startswith("D:"):
                self.logger.warning(f"PDF date string does not start with 'D:': {raw_date}")
                return None
            
            # Extract the core date portion (first 14 characters after D:)
            core_date = raw_date[2:16]  # YYYYMMDDHHmmSS
            
            if len(core_date) < 8:  # At least YYYYMMDD
                self.logger.warning(f"PDF date string too short: {core_date}")
                return None
            
            # Try different parsing strategies
            for fmt, length in [
                ("%Y%m%d%H%M%S", 14),  # Full date and time
                ("%Y%m%d%H%M", 12),    # Date and hour/minute
                ("%Y%m%d%H", 10),      # Date and hour
                ("%Y%m%d", 8),         # Date only
            ]:
                if len(core_date) >= length:
                    try:
                        parsed_date = datetime.strptime(core_date[:length], fmt)
                        self.logger.debug(f"Successfully parsed PDF date: {parsed_date}")
                        return parsed_date
                    except ValueError:
                        continue
            
            self.logger.warning(f"Failed to parse PDF date string: {raw_date}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error parsing PDF date string {raw_date!r}: {e}")
            return None

    def _get_filesystem_creation_date(self, path: Path) -> Optional[datetime]:
        """
        Get creation date from filesystem metadata as fallback.
        
        Args:
            path: Path to the file
            
        Returns:
            Filesystem creation datetime, None if unavailable
        """
        try:
            self.logger.debug(f"Getting filesystem creation date for {path}")
            
            if not path.exists():
                self.logger.warning(f"File does not exist: {path}")
                return None
            
            stat_info = path.stat()
            system = platform.system()
            
            if system == "Windows":
                # On Windows, st_ctime is creation time
                timestamp = stat_info.st_ctime
                self.logger.debug("Using Windows st_ctime as creation time")
                
            elif system == "Darwin":
                # On macOS, try st_birthtime if available
                timestamp = getattr(stat_info, "st_birthtime", None)
                if timestamp is None:
                    timestamp = stat_info.st_mtime
                    self.logger.debug("macOS: using st_mtime as fallback")
                else:
                    self.logger.debug("macOS: using st_birthtime")
                    
            else:
                # Linux/Unix: try stat command for birth time
                birth_timestamp = self._get_linux_birth_time(path)
                if birth_timestamp:
                    timestamp = birth_timestamp
                    self.logger.debug("Linux: using stat birth time")
                else:
                    timestamp = stat_info.st_ctime
                    self.logger.debug("Linux: using st_ctime as fallback")
            
            creation_date = datetime.fromtimestamp(timestamp)
            self.logger.debug(f"Filesystem creation date: {creation_date}")
            return creation_date
            
        except Exception as e:
            self.logger.error(f"Error getting filesystem creation date for {path}: {e}")
            return None

    def _get_linux_birth_time(self, path: Path) -> Optional[float]:
        """
        Get birth time on Linux using stat command.
        
        Args:
            path: Path to the file
            
        Returns:
            Birth timestamp as float, None if unavailable
        """
        try:
            self.logger.debug(f"Calling stat -c %w on {path}")
            
            output = subprocess.check_output(
                ["stat", "-c", "%w", str(path)],
                stderr=subprocess.DEVNULL,
                text=True
            ).strip()
            
            self.logger.debug(f"stat -c %w output: {output!r}")
            
            if output and output != "-":
                # Parse the output: "2025-06-27 14:02:31.000000000 +0000"
                timestamp_str = output.split('.')[0]  # Remove fractional seconds
                parsed_dt = datetime.fromisoformat(timestamp_str)
                return parsed_dt.timestamp()
                
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
            self.logger.debug(f"Linux stat birth time extraction failed: {e}")
            
        return None

    def get_date_metadata(self, filename: str, file_content: Optional[bytes] = None) -> dict:
        """
        Get date metadata for a document.
        
        Args:
            filename: Name of the file
            file_content: Optional file content as bytes
            
        Returns:
            Dictionary with date metadata
        """
        creation_date = self.extract_creation_date(filename, file_content)
        
        metadata = {
            "creation_date": creation_date.isoformat() if creation_date else None,
            "creation_date_source": "pdf_metadata" if creation_date else "none",
            "date_extraction_timestamp": datetime.now().isoformat()
        }
        
        self.logger.debug(f"Date metadata for {filename}: {metadata}")
        return metadata

    def format_date_for_display(self, creation_date: Optional[datetime]) -> str:
        """
        Format creation date for display purposes.
        
        Args:
            creation_date: Creation datetime object
            
        Returns:
            Formatted date string or "Unknown" if None
        """
        if creation_date:
            return creation_date.strftime("%Y-%m-%d %H:%M:%S")
        return "Unknown" 