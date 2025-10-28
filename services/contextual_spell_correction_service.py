#!/usr/bin/env python3
"""
THE LATEST ONE...

Contextual Spell Correction Service

A comprehensive spell correction service that combines:
1. Dictionary-based correction using aero_dict_enriched.json
2. Contextual validation using spaCy NLP
3. Simple, effective correction strategy
4. Integration with the document processing pipeline

Key Features:
- Uses aerospace terminology dictionary for domain-specific corrections
- Contextual validation to prevent incorrect corrections (e.g. "two" → "TOW")
- Protects common words, numbers, and function words based on context
- Fast and reliable correction with minimal false positives
"""

import json
import logging
import re
import unicodedata
from typing import Dict, List, Tuple, Set, Optional, Any
from datetime import datetime
from pathlib import Path
from rapidfuzz import process, fuzz

try:
    import spacy
    from spacy.language import Language
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

class ContextualSpellCorrectionService:
    """Contextual aerospace spell checker with NLP-based validation."""
    
    def __init__(self, dictionary_path: str = "aero_dict_enriched.json"):
        """
        Initialize the contextual spell correction service.
        
        Args:
            dictionary_path: Path to the aerospace dictionary JSON file
        """
        self.logger = logging.getLogger(__name__)
        self.dictionary_path = dictionary_path
        
        # Load aerospace dictionary
        self.lookup = self._build_lookup(dictionary_path)
        self.misspelling_map = self._build_misspelling_map()
        
        # Initialize spaCy for contextual analysis
        self.nlp = self._initialize_spacy()
        self.context_rules = self._build_context_rules()
        
        self.logger.info(f"Loaded {len(self.lookup)} aerospace terms from {dictionary_path}")
        if self.nlp:
            self.logger.info(f"Contextual validation enabled with {self.nlp.meta['name']}")
        else:
            self.logger.warning("Contextual validation disabled - spaCy not available")
    
    def _initialize_spacy(self) -> Optional[Language]:
        """Initialize spaCy language model for contextual analysis."""
        if not SPACY_AVAILABLE:
            return None
        
        try:
            # Try to load the full English model
            nlp = spacy.load("en_core_web_sm")
            return nlp
        except OSError:
            try:
                # Fallback to basic English tokenizer
                from spacy.lang.en import English
                nlp = English()
                self.logger.warning("Using basic English tokenizer - install 'en_core_web_sm' for better context analysis")
                return nlp
            except Exception as e:
                self.logger.warning(f"Could not initialize spaCy: {e}")
                return None
    
    def _build_lookup(self, dictionary_path: str) -> Dict[str, str]:
        """Build lookup dictionary from aerospace terms."""
        lookup = {}
        
        try:
            with open(dictionary_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for entry in data:
                if not isinstance(entry, dict):
                    continue
                
                # Get the correct term (prefer standard_spelling, fallback to term)
                correct_term = str(entry.get("standard_spelling") or entry.get("term") or "").strip()
                if not correct_term:
                    continue
                
                # Add main term
                lookup[correct_term.lower()] = correct_term
                
                # Add variants (term, acronym, aliases)
                variants = []
                if entry.get("term"):
                    variants.append(str(entry["term"]).strip())
                if entry.get("acronym"):
                    variants.append(str(entry["acronym"]).strip())
                if entry.get("aliases") and isinstance(entry["aliases"], list):
                    variants.extend([str(alias).strip() for alias in entry["aliases"]])
                
                # Add all variants to lookup
                for variant in variants:
                    if variant and variant.strip():
                        lookup[variant.lower()] = correct_term
                        
        except Exception as e:
            self.logger.error(f"Error loading dictionary from {dictionary_path}: {e}")
            
        return lookup
    
    def _build_misspelling_map(self) -> Dict[str, str]:
        """Build mapping from common misspellings to correct terms."""
        misspelling_map = {}
        
        try:
            with open(self.dictionary_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for entry in data:
                if not isinstance(entry, dict):
                    continue
                
                # Get the correct term
                correct_term = str(entry.get("standard_spelling") or entry.get("term") or "").strip()
                if not correct_term:
                    continue
                
                # Add common misspellings
                misspellings = entry.get("common_misspellings", [])
                if isinstance(misspellings, list):
                    for misspelling in misspellings:
                        if misspelling and str(misspelling).strip():
                            misspelling_clean = str(misspelling).strip().lower()
                            misspelling_map[misspelling_clean] = correct_term
                            
        except Exception as e:
            self.logger.error(f"Error building misspelling map: {e}")
            
        return misspelling_map
    
    def _build_context_rules(self) -> Dict[str, Dict]:
        """Build contextual validation rules."""
        return {
            'protected_numbers': {
                'words': {'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'},
                'pos_tags': {'NUM', 'CD'},
                'blocked_corrections': {'TOW', 'OEI', 'PMA', 'FIV', 'SIX'}
            },
            'protected_common_words': {
                'words': {'the', 'and', 'or', 'but', 'for', 'at', 'by', 'to', 'of', 'in', 'on', 'with',
                         'good', 'bad', 'big', 'small', 'new', 'old', 'first', 'last', 'next', 'best',
                         'act', 'acts', 'action', 'actions', 'passenger', 'passengers', 'can', 'may',
                         'will', 'shall', 'must', 'should', 'could', 'would', 'need', 'want', 'like',
                         'use', 'used', 'using', 'way', 'ways', 'work', 'works', 'working'},
                'pos_tags': {'DET', 'PREP', 'CONJ', 'PRON', 'ADJ', 'ADV', 'VERB', 'NOUN'},
                'blocked_corrections': {'ACT', 'PAX', 'THE', 'AND', 'BIG', 'BAD', 'CAN', 'MAY', 'USE', 'WAY', 'WAS'}
            },
            'protected_function_words': {
                'words': {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                         'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'},
                'pos_tags': {'AUX', 'DET', 'PREP', 'SCONJ', 'CCONJ'},
                'blocked_corrections': {'THE', 'ARE', 'WAS', 'HAD', 'DID'}
            },
            'aggressive_protection': {
                # Additional protection for words that should almost never become aviation terms
                'words': {'act', 'acts', 'passenger', 'passengers', 'was', 'may', 'can', 'use', 'way'},
                'blocked_aviation_terms': {'ACT', 'PAX', 'WAS', 'MAY', 'CAN', 'USE', 'WAY'}
            }
        }
    
    def _normalize_text(self, text: str) -> str:
        """Normalize unicode and fix common text issues."""
        # Normalize unicode
        text = unicodedata.normalize("NFKC", text)
        
        # Fix common character issues
        text = text.replace("—", "-").replace("–", "-")
        text = text.replace(""", '"').replace(""", '"')
        text = text.replace("'", "'").replace("'", "'")
        
        # Fix hyphenated words split across lines
        text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
        
        return text
    
    def _validate_correction_context(self, original_word: str, correction: str, context: str, 
                                   word_pos: int) -> bool:
        """
        Validate if a correction makes sense in the given context.
        
        Args:
            original_word: The original word
            correction: The proposed correction
            context: The full text context
            word_pos: Position of the word in the context
            
        Returns:
            True if correction should be applied, False if it should be blocked
        """
        if not self.nlp:
            return True  # Allow correction if no NLP available
        
        original_lower = original_word.lower()
        correction_upper = correction.upper()
        
        # Check aggressive protection first (strongest protection)
        aggressive_rule = self.context_rules.get('aggressive_protection', {})
        if (original_lower in aggressive_rule.get('words', set()) and 
            correction_upper in aggressive_rule.get('blocked_aviation_terms', set())):
            self.logger.debug(f"Blocking correction '{original_word}' → '{correction}' due to aggressive protection")
            return False
        
        # Check against other context rules
        for rule_name, rule_data in self.context_rules.items():
            if rule_name == 'aggressive_protection':
                continue  # Already handled above
                
            protected_words = rule_data.get('words', set())
            blocked_corrections = rule_data.get('blocked_corrections', set())
            
            if (original_lower in protected_words and 
                correction_upper in blocked_corrections):
                self.logger.debug(f"Blocking correction '{original_word}' → '{correction}' due to {rule_name}")
                return False
        
        # Use spaCy for POS-based validation
        try:
            # Extract context around the word (±20 characters)
            start_pos = max(0, word_pos - 20)
            end_pos = min(len(context), word_pos + len(original_word) + 20)
            context_snippet = context[start_pos:end_pos]
            
            doc = self.nlp(context_snippet)
            
            # Find the word in the parsed text
            for token in doc:
                if token.text.lower() == original_lower:
                    pos_tag = token.pos_
                    
                    # Check if this POS tag should be protected
                    for rule_data in self.context_rules.values():
                        protected_pos = rule_data.get('pos_tags', set())
                        blocked_corrections = rule_data.get('blocked_corrections', set())
                        
                        if (pos_tag in protected_pos and 
                            correction_upper in blocked_corrections):
                            self.logger.debug(f"Blocking correction '{original_word}' → '{correction}' due to POS tag {pos_tag}")
                            return False
                    break
                    
        except Exception as e:
            self.logger.debug(f"Error in POS validation: {e}")
        
        return True
    
    def _correct_word(self, word: str, context: str = "", word_pos: int = 0) -> Tuple[str, str]:
        """
        Correct a single word using multiple strategies.
        
        Args:
            word: The word to correct
            context: The full text context
            word_pos: Position of the word in the context
            
        Returns:
            Tuple of (corrected_word, correction_method)
        """
        original_word = word
        word_lower = word.lower()
        
        # Skip if word contains hyphens or slashes (often technical terms)
        if '-' in word or '/' in word:
            return word, "no_correction"
        
        # Skip very short words (likely abbreviations or correct)
        if len(word) <= 2:
            return word, "no_correction"
        
        # Strategy 1: Exact dictionary match (with contextual validation)
        if word_lower in self.lookup:
            correction = self.lookup[word_lower]
            # Apply contextual validation for dictionary matches too
            if self._validate_correction_context(original_word, correction, context, word_pos):
                return correction, "dictionary_match"
            else:
                return word, "context_blocked"
        
        # Strategy 2: Common misspelling map
        if word_lower in self.misspelling_map:
            correction = self.misspelling_map[word_lower]
            
            # Validate with context
            if self._validate_correction_context(original_word, correction, context, word_pos):
                return correction, "misspelling_correction"
            else:
                return word, "context_blocked"
        
        # Fuzzy matching removed per user request
        return word, "no_correction"
    
    def correct_text(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Apply spell correction to text with contextual validation.
        
        Args:
            text: The text to correct
            
        Returns:
            Tuple of (corrected_text, correction_details)
        """
        if not text or not text.strip():
            return text, {"corrections": [], "stats": {"total_words": 0}}
        
        # Normalize text
        normalized_text = self._normalize_text(text)
        
        # Track corrections
        corrections = []
        stats = {
            "total_words": 0,
            "corrected_words": 0,
            "context_blocked": 0,
            "dictionary_match": 0,
            "misspelling_correction": 0,
            "no_correction": 0
        }
        
        # Split text into words while preserving structure
        word_pattern = re.compile(r'\b\w+\b')
        words = word_pattern.finditer(normalized_text)
        
        corrected_text = normalized_text
        offset = 0
        
        for match in words:
            word = match.group()
            word_start = match.start()
            word_end = match.end()
            
            stats["total_words"] += 1
            
            # Apply correction
            corrected_word, method = self._correct_word(
                word, 
                normalized_text, 
                word_start
            )
            
            # Update statistics
            stats[method] += 1
            if method in ["dictionary_match", "misspelling_correction"]:
                stats["corrected_words"] += 1
            
            # Apply correction to text if changed
            if corrected_word != word:
                # Adjust position for previous corrections
                adjusted_start = word_start + offset
                adjusted_end = word_end + offset
                
                corrected_text = (corrected_text[:adjusted_start] + 
                                corrected_word + 
                                corrected_text[adjusted_end:])
                
                # Update offset for next corrections
                offset += len(corrected_word) - len(word)
                
                # Record correction
                corrections.append({
                    "original": word,
                    "corrected": corrected_word,
                    "position": word_start,
                    "method": method
                })
        
        # Calculate correction rate
        if stats["total_words"] > 0:
            stats["correction_rate"] = stats["corrected_words"] / stats["total_words"]
        else:
            stats["correction_rate"] = 0.0
        
        correction_details = {
            "corrections": corrections,
            "stats": stats
        }
        
        self.logger.info(f"Spell correction complete: {stats['corrected_words']}/{stats['total_words']} words corrected "
                        f"({stats['correction_rate']:.1%} rate)")
        
        return corrected_text, correction_details
    
    def save_correction_comparison(self, original_text: str, corrected_text: str, 
                                 corrections: List[Dict], filename: str, stats: Dict = None) -> str:
        """
        Save a detailed comparison of original and corrected text.
        
        Args:
            original_text: The original text
            corrected_text: The corrected text
            corrections: List of corrections made
            filename: Base filename for the comparison file
            stats: Statistics dictionary with total_words, corrected_words, correction_rate
            
        Returns:
            Path to the saved comparison file
        """
        try:
            # Create output directory
            output_dir = Path("text_corrections")
            output_dir.mkdir(exist_ok=True)
            
            # Generate comparison filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            comparison_filename = f"{filename}_{timestamp}_contextual_correction.txt"
            comparison_path = output_dir / comparison_filename
            
            # Generate comparison content
            with open(comparison_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("CONTEXTUAL SPELL CORRECTION COMPARISON\n")
                f.write("=" * 80 + "\n\n")
                
                f.write(f"Filename: {filename}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
                
                # Add statistics if provided
                if stats:
                    f.write("CORRECTION STATISTICS:\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"Total Words: {stats.get('total_words', 0)}\n")
                    f.write(f"Corrected Words: {stats.get('corrected_words', 0)}\n")
                    f.write(f"Correction Rate: {stats.get('correction_rate', 0):.2%}\n\n")
                
                # Corrections summary
                f.write("CORRECTIONS MADE:\n")
                f.write("-" * 40 + "\n")
                if corrections:
                    for i, correction in enumerate(corrections, 1):
                        f.write(f"{i:3d}. '{correction['original']}' → '{correction['corrected']}' "
                               f"({correction['method']}) at position {correction['position']}\n")
                else:
                    f.write("No corrections were made.\n")
                
                f.write(f"\nTotal corrections: {len(corrections)}\n\n")
                
                # Side-by-side comparison
                f.write("SIDE-BY-SIDE COMPARISON:\n")
                f.write("-" * 40 + "\n")
                f.write("ORIGINAL TEXT:\n")
                f.write(original_text[:1000] + ("..." if len(original_text) > 1000 else ""))
                f.write("\n\n" + "-" * 40 + "\n")
                f.write("CORRECTED TEXT:\n")
                f.write(corrected_text[:1000] + ("..." if len(corrected_text) > 1000 else ""))
                f.write("\n\n")
            
            return str(comparison_path)
            
        except Exception as e:
            self.logger.error(f"Error saving correction comparison: {e}")
            return "" 