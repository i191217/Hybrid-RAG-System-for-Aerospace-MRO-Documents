"""
Document filtering module for preprocessing files before ingestion.
Implements the filtering criteria to reduce file volume and keep only valuable documents.
"""

import os
import hashlib
import logging
from typing import List, Dict, Set, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
import re

from config import config

logger = logging.getLogger("doc_processor.filter")

@dataclass
class FileInfo:
    """Represents a file with metadata for filtering."""
    path: Path
    size: int
    filename: str
    extension: str
    folder_path: str
    quick_hash: Optional[str] = None
    full_hash: Optional[str] = None
    keep_reason: Optional[str] = None
    reject_reason: Optional[str] = None

class FastFileClassifier:
    """Fast binary classifier to filter documents before processing."""
    
    def __init__(self):
        """Initialize the file classifier with filtering rules."""
        self.logger = logging.getLogger("doc_processor.filter.FastFileClassifier")
        
        # Statistics tracking
        self.stats = {
            "total_files": 0,
            "kept_files": 0,
            "rejected_files": 0,
            "rejection_reasons": defaultdict(int),
            "keep_reasons": defaultdict(int)
        }
        
        # Filtering rules
        self.allowed_extensions = {'.pdf', '.docx', '.xlsx', '.pptx'}
        
        self.reject_folder_patterns = {
            'archive', 'archived', 'old', 'backup', 'temp', 'tmp', 
            'cache', 'trash', 'recycle', 'deleted'
        }
        
        self.reject_filename_patterns = [
            r'^~.*',           # Temporary files
            r'^\.',            # Hidden files
            r'^\$',            # System files
            r'^copy of ',      # Copy files
            r'superseded',     # Superseded files
            r'obsolete',       # Obsolete files
            r'^test',          # Test files
            r'^sample',        # Sample files
            r'\.tmp$',         # Temp files
            r'\.bak$',         # Backup files
        ]
        
        self.keep_filename_patterns = [
            r'final',          # Final versions
            r'signed',         # Signed documents
            r'executed',       # Executed documents
            r'board deck',     # Board presentations
            r'investment memo' # Investment memos
        ]
        
        # Size limits
        self.min_size_general = 1024      # 1KB minimum
        self.max_size_general = 100 * 1024 * 1024  # 100MB maximum
        self.min_size_pdf = 5 * 1024      # 5KB minimum for PDFs
        
        # Suspicious sizes (exactly these sizes often indicate corrupted/empty files)
        self.suspicious_sizes = {1024, 2048, 4096, 8192}
        
        self.logger.info("Initialized FastFileClassifier with filtering rules")
    
    def should_keep_file(self, file_path: Path) -> Tuple[bool, str]:
        """
        Determine if a file should be kept based on filtering criteria.
        
        Args:
            file_path: Path to the file to evaluate
            
        Returns:
            Tuple of (should_keep: bool, reason: str)
        """
        try:
            # Get file info
            if not file_path.exists():
                return False, "file_not_found"
            
            if not file_path.is_file():
                return False, "not_a_file"
            
            filename = file_path.name.lower()
            extension = file_path.suffix.lower()
            folder_path = str(file_path.parent).lower()
            
            try:
                file_size = file_path.stat().st_size
            except OSError:
                return False, "cannot_read_file"
            
            # 1. Check extension (fastest check first)
            if extension not in self.allowed_extensions:
                return False, "bad_extension"
            
            # 2. Check folder patterns
            for reject_pattern in self.reject_folder_patterns:
                if reject_pattern in folder_path:
                    return False, "archive_folder"
            
            # 3. Check filename patterns - KEEP patterns first (higher priority)
            for keep_pattern in self.keep_filename_patterns:
                if re.search(keep_pattern, filename, re.IGNORECASE):
                    return True, "important_file"
            
            # Check REJECT patterns
            for reject_pattern in self.reject_filename_patterns:
                if re.search(reject_pattern, filename, re.IGNORECASE):
                    return False, "rejected_pattern"
            
            # 4. Check file size
            if file_size < self.min_size_general:
                return False, "too_small"
            
            if file_size > self.max_size_general:
                return False, "too_large"
            
            # PDF-specific size check
            if extension == '.pdf' and file_size < self.min_size_pdf:
                return False, "pdf_too_small"
            
            # Check for suspicious sizes
            if file_size in self.suspicious_sizes:
                return False, "suspicious_size"
            
            # If we get here, the file passes all checks
            return True, "passed_all_checks"
            
        except Exception as e:
            self.logger.error(f"Error evaluating file {file_path}: {e}")
            return False, "evaluation_error"
    
    def classify_file(self, file_path: Path) -> FileInfo:
        """
        Classify a single file and return FileInfo with decision.
        
        Args:
            file_path: Path to the file to classify
            
        Returns:
            FileInfo object with classification results
        """
        try:
            file_size = file_path.stat().st_size if file_path.exists() else 0
            should_keep, reason = self.should_keep_file(file_path)
            
            file_info = FileInfo(
                path=file_path,
                size=file_size,
                filename=file_path.name,
                extension=file_path.suffix.lower(),
                folder_path=str(file_path.parent)
            )
            
            if should_keep:
                file_info.keep_reason = reason
                self.stats["kept_files"] += 1
                self.stats["keep_reasons"][reason] += 1
            else:
                file_info.reject_reason = reason
                self.stats["rejected_files"] += 1
                self.stats["rejection_reasons"][reason] += 1
            
            self.stats["total_files"] += 1
            
            return file_info
            
        except Exception as e:
            self.logger.error(f"Error classifying file {file_path}: {e}")
            return FileInfo(
                path=file_path,
                size=0,
                filename=file_path.name,
                extension="",
                folder_path=str(file_path.parent),
                reject_reason="classification_error"
            )
    
    def get_stats(self) -> Dict:
        """Get classification statistics."""
        stats = self.stats.copy()
        if stats["total_files"] > 0:
            stats["keep_rate"] = stats["kept_files"] / stats["total_files"]
            stats["reject_rate"] = stats["rejected_files"] / stats["total_files"]
        else:
            stats["keep_rate"] = 0.0
            stats["reject_rate"] = 0.0
        
        return stats
    
    def log_stats(self):
        """Log classification statistics."""
        stats = self.get_stats()
        
        self.logger.info("=== File Classification Statistics ===")
        self.logger.info(f"Total files processed: {stats['total_files']}")
        self.logger.info(f"Files kept: {stats['kept_files']} ({stats['keep_rate']:.1%})")
        self.logger.info(f"Files rejected: {stats['rejected_files']} ({stats['reject_rate']:.1%})")
        
        self.logger.info("\nRejection reasons:")
        for reason, count in stats['rejection_reasons'].items():
            self.logger.info(f"  {reason}: {count}")
        
        self.logger.info("\nKeep reasons:")
        for reason, count in stats['keep_reasons'].items():
            self.logger.info(f"  {reason}: {count}")
        
        self.logger.info("=====================================")

class DocumentDeduplicator:
    """Handles deduplication of documents using hash-based comparison."""
    
    def __init__(self):
        """Initialize the deduplicator."""
        self.logger = logging.getLogger("doc_processor.filter.DocumentDeduplicator")
        
        # Statistics
        self.stats = {
            "total_files": 0,
            "unique_files": 0,
            "duplicate_files": 0,
            "duplicate_groups": 0
        }
    
    def generate_quick_hash(self, file_path: Path) -> Optional[str]:
        """
        Generate a quick hash using first/last 1MB + file size.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Quick hash string or None if error
        """
        try:
            file_size = file_path.stat().st_size
            
            # For small files, use the entire file
            if file_size <= 2 * 1024 * 1024:  # 2MB
                with open(file_path, 'rb') as f:
                    content = f.read()
                return hashlib.md5(content).hexdigest()
            
            # For large files, use first 1MB + last 1MB + size
            hash_obj = hashlib.md5()
            
            with open(file_path, 'rb') as f:
                # First 1MB
                first_chunk = f.read(1024 * 1024)
                hash_obj.update(first_chunk)
                
                # File size
                hash_obj.update(str(file_size).encode())
                
                # Last 1MB
                f.seek(-1024 * 1024, 2)  # Seek to 1MB from end
                last_chunk = f.read(1024 * 1024)
                hash_obj.update(last_chunk)
            
            return hash_obj.hexdigest()
            
        except Exception as e:
            self.logger.error(f"Error generating quick hash for {file_path}: {e}")
            return None
    
    def generate_full_hash(self, file_path: Path) -> Optional[str]:
        """
        Generate a full file hash.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Full hash string or None if error
        """
        try:
            hash_obj = hashlib.md5()
            
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_obj.update(chunk)
            
            return hash_obj.hexdigest()
            
        except Exception as e:
            self.logger.error(f"Error generating full hash for {file_path}: {e}")
            return None
    
    def deduplicate_files(self, file_infos: List[FileInfo]) -> List[FileInfo]:
        """
        Remove duplicates from a list of files.
        
        Args:
            file_infos: List of FileInfo objects to deduplicate
            
        Returns:
            List of unique FileInfo objects
        """
        self.logger.info(f"Starting deduplication of {len(file_infos)} files")
        
        self.stats["total_files"] = len(file_infos)
        
        # Step 1: Group by quick hash
        quick_hash_groups = defaultdict(list)
        
        for file_info in file_infos:
            if file_info.reject_reason:  # Skip already rejected files
                continue
            
            quick_hash = self.generate_quick_hash(file_info.path)
            if quick_hash:
                file_info.quick_hash = quick_hash
                quick_hash_groups[quick_hash].append(file_info)
        
        # Step 2: For groups with multiple files, verify with full hash
        unique_files = []
        duplicate_groups = 0
        
        for quick_hash, group in quick_hash_groups.items():
            if len(group) == 1:
                # No duplicates in this group
                unique_files.extend(group)
            else:
                # Potential duplicates - verify with full hash
                duplicate_groups += 1
                full_hash_groups = defaultdict(list)
                
                for file_info in group:
                    full_hash = self.generate_full_hash(file_info.path)
                    if full_hash:
                        file_info.full_hash = full_hash
                        full_hash_groups[full_hash].append(file_info)
                
                # Select best file from each duplicate group
                for full_hash, duplicate_group in full_hash_groups.items():
                    if len(duplicate_group) == 1:
                        unique_files.extend(duplicate_group)
                    else:
                        best_file = self.select_best_version(duplicate_group)
                        unique_files.append(best_file)
                        
                        # Mark others as duplicates
                        for file_info in duplicate_group:
                            if file_info != best_file:
                                file_info.reject_reason = "duplicate"
        
        # Add back rejected files for statistics
        rejected_files = [f for f in file_infos if f.reject_reason and f.reject_reason != "duplicate"]
        all_files = unique_files + rejected_files + [f for f in file_infos if f.reject_reason == "duplicate"]
        
        self.stats["unique_files"] = len(unique_files)
        self.stats["duplicate_files"] = len(file_infos) - len(unique_files)
        self.stats["duplicate_groups"] = duplicate_groups
        
        self.logger.info(f"Deduplication complete: {len(unique_files)} unique files, {self.stats['duplicate_files']} duplicates")
        
        return all_files
    
    def select_best_version(self, file_group: List[FileInfo]) -> FileInfo:
        """
        Select the best version from a group of duplicate files.
        
        Priority: executed > signed > final > newest > largest
        
        Args:
            file_group: List of duplicate FileInfo objects
            
        Returns:
            Best FileInfo object from the group
        """
        def get_priority_score(file_info: FileInfo) -> Tuple[int, int, int]:
            """Get priority score for file selection."""
            filename_lower = file_info.filename.lower()
            
            # Priority keywords (higher number = higher priority)
            if 'executed' in filename_lower:
                priority = 4
            elif 'signed' in filename_lower:
                priority = 3
            elif 'final' in filename_lower:
                priority = 2
            else:
                priority = 1
            
            # Get modification time (newer is better)
            try:
                mtime = int(file_info.path.stat().st_mtime)
            except:
                mtime = 0
            
            # File size (larger is better, as tiebreaker)
            size = file_info.size
            
            return (priority, mtime, size)
        
        # Sort by priority score (descending)
        sorted_files = sorted(file_group, key=get_priority_score, reverse=True)
        
        best_file = sorted_files[0]
        self.logger.debug(f"Selected best version: {best_file.filename} from {len(file_group)} duplicates")
        
        return best_file

class DocumentFilter:
    """Main document filtering class that orchestrates the filtering process."""
    
    def __init__(self):
        """Initialize the document filter."""
        self.logger = logging.getLogger("doc_processor.filter.DocumentFilter")
        
        self.classifier = FastFileClassifier()
        self.deduplicator = DocumentDeduplicator()
        
        self.logger.info("Initialized DocumentFilter")
    
    def process_directory(self, input_dir: Path, recursive: bool = True) -> List[FileInfo]:
        """
        Process a directory and return filtered files.
        
        Args:
            input_dir: Directory to process
            recursive: Whether to process subdirectories
            
        Returns:
            List of FileInfo objects for files that should be kept
        """
        self.logger.info(f"Processing directory: {input_dir}")
        
        # Collect all files
        all_files = []
        
        if recursive:
            pattern = "**/*"
        else:
            pattern = "*"
        
        for file_path in input_dir.glob(pattern):
            if file_path.is_file():
                all_files.append(file_path)
        
        self.logger.info(f"Found {len(all_files)} files to process")
        
        # Phase 1: Quick classification
        self.logger.info("Phase 1: File classification")
        file_infos = []
        
        for file_path in all_files:
            file_info = self.classifier.classify_file(file_path)
            file_infos.append(file_info)
        
        # Log classification stats
        self.classifier.log_stats()
        
        # Phase 2: Deduplication
        self.logger.info("Phase 2: Deduplication")
        deduplicated_files = self.deduplicator.deduplicate_files(file_infos)
        
        # Filter to only kept files
        kept_files = [f for f in deduplicated_files if not f.reject_reason]
        
        self.logger.info(f"Final result: {len(kept_files)} files kept out of {len(all_files)} total")
        
        return kept_files
    
    def get_stats(self) -> Dict:
        """Get combined filtering statistics."""
        classifier_stats = self.classifier.get_stats()
        deduplicator_stats = self.deduplicator.stats
        
        return {
            "classification": classifier_stats,
            "deduplication": deduplicator_stats,
            "final_reduction": {
                "original_count": classifier_stats["total_files"],
                "final_count": classifier_stats["kept_files"] - deduplicator_stats["duplicate_files"],
                "reduction_rate": 1 - ((classifier_stats["kept_files"] - deduplicator_stats["duplicate_files"]) / classifier_stats["total_files"]) if classifier_stats["total_files"] > 0 else 0
            }
        }

def filter_documents(input_directory: str, output_list: str = None) -> List[Path]:
    """
    Filter documents in a directory and return list of files to process.
    
    Args:
        input_directory: Directory containing documents to filter
        output_list: Optional file to save the list of kept files
        
    Returns:
        List of file paths that passed filtering
    """
    input_dir = Path(input_directory)
    
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_directory}")
        return []
    
    # Initialize filter
    document_filter = DocumentFilter()
    
    # Process directory
    kept_files = document_filter.process_directory(input_dir)
    
    # Extract file paths
    file_paths = [file_info.path for file_info in kept_files]
    
    # Save list if requested
    if output_list:
        try:
            with open(output_list, 'w') as f:
                for file_path in file_paths:
                    f.write(f"{file_path}\n")
            logger.info(f"Saved file list to: {output_list}")
        except Exception as e:
            logger.error(f"Error saving file list: {e}")
    
    # Log final statistics
    stats = document_filter.get_stats()
    logger.info("=== Final Filtering Statistics ===")
    logger.info(f"Original files: {stats['final_reduction']['original_count']}")
    logger.info(f"Final files: {stats['final_reduction']['final_count']}")
    logger.info(f"Reduction rate: {stats['final_reduction']['reduction_rate']:.1%}")
    logger.info("==================================")
    
    return file_paths

if __name__ == "__main__":
    """Run document filtering when script is executed directly."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python document_filter.py <input_directory> [output_list.txt]")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_list = sys.argv[2] if len(sys.argv) > 2 else None
    
    filtered_files = filter_documents(input_dir, output_list)
    
    print(f"\nFiltering complete!")
    print(f"Kept {len(filtered_files)} files")
    
    if output_list:
        print(f"File list saved to: {output_list}") 