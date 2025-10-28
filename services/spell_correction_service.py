#!/usr/bin/env python3
"""
Spell Correction Service using Levenshtein Distance with Aerospace Dictionary.
Corrects spelling mistakes in extracted text using industry-standard aerospace terms.
"""

import os
import json
import logging
import re
from typing import Dict, List, Tuple, Set, Optional
from datetime import datetime
import Levenshtein
from pathlib import Path

class SpellCorrectionService:
    """Service for correcting spelling mistakes using aerospace dictionary and Levenshtein distance."""
    
    def __init__(self, dictionary_path: str = "aero_dict_enriched.json", max_distance: int = 2):
        """
        Initialize the spell correction service.
        
        Args:
            dictionary_path: Path to the aerospace dictionary JSON file
            max_distance: Maximum Levenshtein distance for corrections
        """
        self.logger = logging.getLogger(__name__)
        self.dictionary_path = dictionary_path
        self.max_distance = max_distance
        
        # Dictionary structures
        self.aerospace_terms: Dict[str, str] = {}  # term -> standard_spelling
        self.misspelling_map: Dict[str, str] = {}  # misspelling -> correct_term
        self.term_variations: Dict[str, Set[str]] = {}  # term -> {variations}
        
        # Statistics
        self.correction_stats = {
            "total_corrections": 0,
            "dictionary_corrections": 0,
            "levenshtein_corrections": 0,
            "no_corrections": 0
        }
        
        # Load dictionary
        self._load_dictionary()
        
    def _load_dictionary(self):
        """Load the aerospace dictionary and build lookup structures."""
        try:
            if not os.path.exists(self.dictionary_path):
                self.logger.error(f"Dictionary file not found: {self.dictionary_path}")
                return
                
            with open(self.dictionary_path, 'r', encoding='utf-8') as f:
                dictionary_data = json.load(f)
            
            self.logger.info(f"Loading {len(dictionary_data)} aerospace terms...")
            
            for entry in dictionary_data:
                if not isinstance(entry, dict):
                    continue
                    
                # Skip entries with NaN values (handle the linter errors)
                if any(str(v) == 'NaN' for v in entry.values() if v is not None):
                    continue
                
                term = entry.get("term", "")
                standard_spelling = entry.get("standard_spelling", term)
                
                # Convert to string and strip to handle non-string values
                if term and not isinstance(term, str):
                    term = str(term)
                if standard_spelling and not isinstance(standard_spelling, str):
                    standard_spelling = str(standard_spelling)
                    
                term = term.strip() if term else ""
                standard_spelling = standard_spelling.strip() if standard_spelling else ""
                
                if not term or not standard_spelling:
                    continue
                
                # Store standard spelling
                self.aerospace_terms[term.lower()] = standard_spelling
                
                # Store ONLY legitimate misspellings (filter out the term itself and common words)
                misspellings = entry.get("common_misspellings", [])
                if isinstance(misspellings, list):
                    for misspelling in misspellings:
                        if (misspelling and isinstance(misspelling, str) and 
                            misspelling.strip().lower() != term.lower() and
                            misspelling.strip().lower() != standard_spelling.lower() and
                            # Filter out common English words that should never be corrected
                            misspelling.strip().lower() not in {'true','the', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among', 'this', 'that', 'these', 'those', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'can', 'may', 'might', 'must', 'shall', 'should', 'ought', 'page', 'drive', 'work', 'new', 'old', 'good', 'bad', 'big', 'small', 'long', 'short', 'high', 'low', 'first', 'last', 'next', 'early', 'late', 'well', 'also', 'just', 'only', 'even', 'still', 'now', 'then', 'here', 'there', 'where', 'when', 'why', 'how', 'what', 'who', 'which', 'whose', 'whom', 'year', 'years', 'day', 'days', 'time', 'times', 'way', 'ways', 'man', 'men', 'woman', 'women', 'child', 'children', 'life', 'lives', 'world', 'country', 'state', 'city', 'place', 'home', 'house', 'school', 'work', 'job', 'company', 'system', 'program', 'number', 'group', 'part', 'area', 'hand', 'eye', 'face', 'fact', 'head', 'water', 'air', 'fire', 'earth', 'money', 'business', 'case', 'point', 'government', 'student', 'week', 'month', 'night', 'right', 'left', 'side', 'end', 'line', 'room', 'car', 'book', 'story', 'example', 'lot', 'result', 'change', 'kind', 'name', 'need', 'try', 'ask', 'turn', 'move', 'play', 'live', 'help', 'give', 'show', 'use', 'get', 'make', 'go', 'come', 'know', 'take', 'see', 'look', 'want', 'call', 'find', 'become', 'leave', 'put', 'mean', 'keep', 'let', 'begin', 'start', 'run', 'bring', 'happen', 'write', 'sit', 'stand', 'lose', 'pay', 'meet', 'include', 'continue', 'set', 'learn', 'follow', 'stop', 'create', 'speak', 'read', 'spend', 'grow', 'open', 'walk', 'win', 'buy', 'die', 'kill', 'talk', 'remain', 'suggest', 'raise', 'pass', 'sell', 'require', 'report', 'decide', 'pull'}):
                            self.misspelling_map[misspelling.lower().strip()] = standard_spelling
                
                # Store aliases (but only exact matches, no fuzzy matching)
                aliases = entry.get("aliases", [])
                if isinstance(aliases, list):
                    for alias in aliases:
                        if alias and isinstance(alias, str):
                            self.aerospace_terms[alias.lower().strip()] = standard_spelling
                
                # Store expanded form (but only exact matches, no fuzzy matching) 
                expanded = entry.get("expanded", "")
                if expanded and isinstance(expanded, str) and expanded != "NaN":
                    self.aerospace_terms[expanded.lower().strip()] = standard_spelling
                
                # Build variations set for this term
                variations = {term.lower(), standard_spelling.lower()}
                if aliases:
                    variations.update(a.lower() for a in aliases if a and isinstance(a, str))
                if expanded and expanded != "NaN":
                    variations.add(expanded.lower())
                
                self.term_variations[standard_spelling.lower()] = variations
            
            self.logger.info(f"Loaded {len(self.aerospace_terms)} aerospace terms and {len(self.misspelling_map)} legitimate misspellings")
            
        except Exception as e:
            self.logger.error(f"Failed to load dictionary: {e}")
    
    def _find_best_match(self, word: str) -> Optional[str]:
        """
        Conservative approach: NO Levenshtein distance matching.
        Only exact matches from the misspelling dictionary.
        
        Args:
            word: Word to find match for
            
        Returns:
            Exact matching standard spelling or None
        """
        word_lower = word.lower()
        
        # Only exact matches from misspelling map - no fuzzy matching
        return self.misspelling_map.get(word_lower)
    
    def correct_word(self, word: str) -> Tuple[str, bool, str]:
        """
        Correct a single word using ONLY exact matches from the aerospace dictionary.
        No Levenshtein distance matching to prevent over-correction.
        
        Args:
            word: Word to correct
            
        Returns:
            Tuple of (corrected_word, was_corrected, correction_method)
        """
        if not word or len(word) < 2:
            return word, False, "too_short"
        
        word_clean = word.strip()
        word_lower = word_clean.lower()
        
        # 1. EXACT misspelling lookup (highest priority)
        if word_lower in self.misspelling_map:
            corrected = self.misspelling_map[word_lower]
            self.correction_stats["dictionary_corrections"] += 1
            return corrected, True, "exact_misspelling_correction"
        
        # 2. EXACT dictionary lookup for standardization (case normalization only)
        if word_lower in self.aerospace_terms:
            corrected = self.aerospace_terms[word_lower]
            # Only correct if the case is different
            if corrected != word_clean:
                self.correction_stats["dictionary_corrections"] += 1
                return corrected, True, "case_standardization"
            return word_clean, False, "already_correct"
        
        # 3. NO fuzzy matching - if it's not an exact match, don't correct it
        self.correction_stats["no_corrections"] += 1
        return word_clean, False, "no_exact_match"
    
    def correct_text(self, text: str) -> Tuple[str, Dict]:
        """
        Correct spelling mistakes in a text using aerospace dictionary.
        
        Args:
            text: Text to correct
            
        Returns:
            Tuple of (corrected_text, correction_details)
        """
        if not text:
            return text, {"corrections": [], "stats": {"total_words": 0, "corrected_words": 0}}
        
        # Split text into words while preserving structure
        words = re.findall(r'\b\w+\b|\W+', text)
        corrected_words = []
        corrections = []
        total_words = 0
        corrected_count = 0
        
        for i, token in enumerate(words):
            if re.match(r'\b\w+\b', token):  # It's a word
                total_words += 1
                corrected_word, was_corrected, method = self.correct_word(token)
                
                if was_corrected:
                    corrected_count += 1
                    corrections.append({
                        "position": i,
                        "original": token,
                        "corrected": corrected_word,
                        "method": method
                    })
                
                corrected_words.append(corrected_word)
            else:
                corrected_words.append(token)  # Preserve whitespace and punctuation
        
        corrected_text = ''.join(corrected_words)
        
        correction_details = {
            "corrections": corrections,
            "stats": {
                "total_words": total_words,
                "corrected_words": corrected_count,
                "correction_rate": corrected_count / total_words if total_words > 0 else 0
            }
        }
        
        self.correction_stats["total_corrections"] += corrected_count
        
        return corrected_text, correction_details
    
    def save_correction_comparison(self, original_text: str, corrected_text: str, 
                                 correction_details: Dict, filename: str) -> str:
        """
        Save before/after text comparison to a file.
        
        Args:
            original_text: Original text before correction
            corrected_text: Text after spell correction
            correction_details: Details about corrections made
            filename: Original filename for reference
            
        Returns:
            Path to the saved comparison file
        """
        try:
            # Create corrections directory if it doesn't exist
            corrections_dir = Path("text_corrections")
            corrections_dir.mkdir(exist_ok=True)
            
            # Generate comparison filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_filename = re.sub(r'[^\w\-_.]', '_', filename)
            comparison_file = corrections_dir / f"{safe_filename}_{timestamp}_correction.txt"
            
            with open(comparison_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("AEROSPACE SPELL CORRECTION COMPARISON\n")
                f.write("=" * 80 + "\n")
                f.write(f"Original File: {filename}\n")
                f.write(f"Correction Date: {datetime.now().isoformat()}\n")
                f.write(f"Total Words: {correction_details['stats']['total_words']}\n")
                f.write(f"Corrected Words: {correction_details['stats']['corrected_words']}\n")
                f.write(f"Correction Rate: {correction_details['stats']['correction_rate']:.2%}\n")
                f.write("\n")
                
                # Corrections summary
                if correction_details['corrections']:
                    f.write("CORRECTIONS MADE:\n")
                    f.write("-" * 40 + "\n")
                    for correction in correction_details['corrections']:
                        f.write(f"  {correction['original']} → {correction['corrected']} ({correction['method']})\n")
                    f.write("\n")
                else:
                    f.write("No corrections were made.\n\n")
                
                # Original text
                f.write("ORIGINAL TEXT:\n")
                f.write("-" * 40 + "\n")
                f.write(original_text)
                f.write("\n\n")
                
                # Corrected text
                f.write("CORRECTED TEXT:\n")
                f.write("-" * 40 + "\n")
                f.write(corrected_text)
                f.write("\n\n")
                
                # Side-by-side comparison for significant corrections
                if correction_details['corrections']:
                    f.write("DETAILED CORRECTIONS:\n")
                    f.write("-" * 40 + "\n")
                    for correction in correction_details['corrections']:
                        f.write(f"Position {correction['position']}:\n")
                        f.write(f"  Before: '{correction['original']}'\n")
                        f.write(f"  After:  '{correction['corrected']}'\n")
                        f.write(f"  Method: {correction['method']}\n")
                        f.write("\n")
                
                f.write("=" * 80 + "\n")
            
            self.logger.info(f"Correction comparison saved to: {comparison_file}")
            return str(comparison_file)
            
        except Exception as e:
            self.logger.error(f"Failed to save correction comparison: {e}")
            return ""
    
    def get_correction_stats(self) -> Dict:
        """Get correction statistics."""
        return self.correction_stats.copy()
    
    def reset_stats(self):
        """Reset correction statistics."""
        self.correction_stats = {
            "total_corrections": 0,
            "dictionary_corrections": 0,
            "levenshtein_corrections": 0,
            "no_corrections": 0
        } 