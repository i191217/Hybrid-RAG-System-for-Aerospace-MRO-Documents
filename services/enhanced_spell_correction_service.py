#!/usr/bin/env python3
"""
Enhanced Spell Correction Service - Integrates the advanced EnhancedAeroSpellChecker
into the document processing pipeline with the same interface as the original service.

Key Improvements (v2.0):
1. ABBREVIATION PROTECTION: Added comprehensive protection for business/finance abbreviations 
   (CAM, CEO, CFO, LLC, ERP, OEM, LTA, CAGR, SQIP, COTS, HYD, etc.)
2. LEVENSHTEIN DISTANCE VALIDATION: Skip corrections for words with distance ≤1 to avoid 
   correcting likely-correct words
3. STRICTER FUZZY MATCHING: Increased threshold from 85% to 92% to reduce false positives
4. ENHANCED PROTECTION LOGIC: Now properly calls should_protect() method in correct_word()
5. DETAILED STATISTICS: Track protection reasons (protected_terms, levenshtein_skip, etc.)
6. CONSERVATIVE SPELLCHECKER: Only correct significantly wrong words (distance ≥3)
"""

import os
import sys
import json
import logging
import re
import unicodedata
from typing import Dict, List, Tuple, Set, Optional
from datetime import datetime
from pathlib import Path
from rapidfuzz import process, fuzz
from spellchecker import SpellChecker
import Levenshtein  # Add Levenshtein distance calculation

try:
    import spacy
    from spacy.language import Language
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

class EnhancedSpellCorrectionService:
    """Enhanced aerospace spell checker service combining multiple correction strategies."""
    
    def __init__(self, dictionary_path: str = "aero_dict_enriched.json"):
        """
        Initialize the enhanced spell correction service.
        
        Args:
            dictionary_path: Path to the aerospace dictionary JSON file
        """
        self.logger = logging.getLogger(__name__)
        self.dictionary_path = dictionary_path
        self.lookup = self.build_lookup(dictionary_path)
        self.spell = SpellChecker()
        self.protected_terms = self._build_protected_terms()
                
        # Statistics tracking
        self.stats = {
            'total_words': 0,
            'corrected_words': 0,
            'context_blocked': 0,
            'dictionary_corrections': 0,
            'fuzzy_corrections': 0,
            'spellchecker_corrections': 0,
            'protected_terms_skipped': 0,
            'levenshtein_skipped': 0,
            'short_uppercase_protected': 0,
            'common_word_protected': 0
        }
        
        # Optional spaCy integration
        self.nlp = self._load_spacy_model()
        self._build_context_rules()

    def _build_protected_terms(self) -> set:
        """Build set of terms that should never be corrected."""
        protected = set()
        
        # 1. Protect all uppercase terms (acronyms, proper nouns)
        for term in self.lookup.keys():
            if term.isupper():
                protected.add(term)
                
        # 2. Protect aircraft model patterns (FXX, BXX, etc.)
        protected.update({
            'f6x', 'f10x', 'f16', 'f22', 'f35',  # Fighter jets
            'b2', 'b52',                          # Bombers
            'a10', 'a320', 'a380',                 # Attack/Commercial
            'c130', 'c17',                         # Cargo
            'e3', 'e4',                            # AWACS
            'k135', 'kc10'                         # Tankers
        })
        
        # 3. Common abbreviations (aerospace, business, and finance)
        protected.update({
            # Aerospace abbreviations
            'itp', 'nato', 'usaf', 'faa', 'icao', 'iata',
            'fbo', 'atis', 'notam', 'taf', 'metar',
            # Business and finance abbreviations
            'cam', 'ceo', 'cfo', 'coo', 'cto', 'llc', 'inc', 'ltd', 'corp',
            'erp', 'oem', 'lta', 'roi', 'ebitda', 'cagr', 'sqip', 'cots',
            'hyd', 'ops', 'qa', 'qc', 'hr', 'it', 'rd', 'rnd', 'capex', 'opex',
            # Technical abbreviations  
            'api', 'ui', 'ux', 'sql', 'xml', 'json', 'csv', 'pdf', 'doc',
            'gps', 'rfp', 'rfq', 'sow', 'kpi', 'sla', 'nda', 'pii'
        })
        
        return protected
    
    def _load_spacy_model(self) -> Optional[Language]:
        """Load spaCy model if available."""
        if not SPACY_AVAILABLE:
            self.logger.warning(
                "spaCy not available. Contextual spell correction will be disabled. "
                "Install with: pip install spacy && python -m spacy download en_core_web_sm"
            )
            return None
        
        try:
            nlp = spacy.load("en_core_web_sm")
            self.logger.info("spaCy model 'en_core_web_sm' loaded successfully.")
            return nlp
        except OSError:
            self.logger.warning("spaCy model 'en_core_web_sm' not found. Contextual analysis disabled.")
            return None
    
    def _build_context_rules(self):
        """Build rules for contextual validation."""
        # POS tags for words that should be protected from aggressive correction
        self.protected_pos_tags = {"NUM", "CD", "DET", "PREP", "CONJ", "PRON", "ADJ", "ADV", "AUX", "SCONJ", "CCONJ"}
        
        # Common English words that are often mistaken for aerospace terms
        self.protected_words = {
            'pick', 'true', 'the', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about',
            'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among', 'this', 'that', 'these',
            'those', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'should', 'could', 'can', 'may', 'might', 'must', 'shall', 'ought', 'page', 'drive', 'work', 'new',
            'old', 'good', 'bad', 'big', 'small', 'long', 'short', 'high', 'low', 'first', 'last', 'next', 'early', 'late',
            'well', 'also', 'just', 'only', 'even', 'still', 'now', 'then', 'here', 'there', 'where', 'when', 'why', 'how',
            'what', 'who', 'which', 'whose', 'whom', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'
        }
    
    def build_lookup(self, dict_path: str) -> Dict[str, str]:
        """Build the lookup dictionary from JSON file."""
        try:
            with open(dict_path, encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            self.logger.error(f"Dictionary file not found: {dict_path}")
            return {}
        except Exception as e:
            self.logger.error(f"Error loading dictionary: {e}")
            return {}
        
        lookup = {}
        for e in data:
            std = str(e.get("standard_spelling") or e.get("term") or "").strip()
            if not std:
                continue

            # Preserve original casing in standard spelling
            lookup[std.lower()] = std
            
            variants = {
                e.get("term", ""), e.get("acronym", ""),
                *e.get("aliases", []), *e.get("common_misspellings", [])
            }
            
            for v in variants:
                if not v:
                    continue
                s = str(v).strip()
                lookup[s.lower()] = std
        
        self.logger.info(f"Loaded {len(lookup)} aerospace terms from dictionary")
        return lookup
    
    def should_protect(self, word: str) -> bool:
        """Determine if a word should be protected from correction."""
        lower_word = word.lower()
        
        # 1. Protect if in protected terms set
        if lower_word in self.protected_terms:
            return True
            
        # 2. Protect aircraft model patterns (FXX, BXX, etc.)
        if re.match(r'^[a-z]\d+[a-z]?$', lower_word):
            return True
            
        # 3. Protect if appears to be an acronym (mixed case or all caps)
        if word.isupper() or (len(word) > 1 and word[0].isupper() and word[1:].islower()):
            return True
            
        # 4. Protect if it's a known abbreviation (2-4 letters all caps)
        if 2 <= len(word) <= 4 and word.isupper():
            return True
            
        return False
    
    def normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters and fix common issues."""
        text = unicodedata.normalize("NFKC", text)
        return (text
            .replace("—", "-").replace("–", "-")
            .replace("•", " ").replace("", " ")
        )
    
    def fix_hyphens(self, text: str) -> str:
        """Fix hyphenated words split across lines."""
        return re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1-\2", text)
    
    def protect_phrases(self, text: str) -> str:
        """Protect multi-word phrases from being split during tokenization."""
        phrases = sorted([k for k in self.lookup if " " in k], key=lambda x: -len(x))
        for ph in phrases:
            rx = re.compile(rf"\b{re.escape(ph)}\b", flags=re.IGNORECASE)
            text = rx.sub(lambda m: m.group(0).replace(" ", "_"), text)
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text while preserving special characters."""
        return re.findall(r"[A-Za-z0-9][A-Za-z0-9_\-'/]*", text)
    
    def try_split(self, tok: str) -> List[str]:
        """Attempt to split long unknown tokens."""
        low = tok.lower()
        # If token is already valid (domain or English), don't split
        if low in self.lookup or low in self.spell.known([low]):
            return [tok]
        
        n = len(tok)
        if n < 8:
            return [tok]
        
        # Try all splits leaving at least 3 chars each side
        for i in range(3, n-3):
            a, b = low[:i], low[i:]
            if ((a in self.lookup or a in self.spell.known([a])) and
                (b in self.lookup or b in self.spell.known([b]))):
                return [tok[:i], tok[i:]]
        
        return [tok]
    
    def is_contextually_valid(self, original: str, correction: str) -> bool:
        """Check if correction is contextually valid."""
        if not self.nlp:
            return True
        
        doc = self.nlp(original)
        if len(doc) != 1:
            return True
        
        token = doc[0]
        if token.pos_ in self.protected_pos_tags:
            self.logger.debug(f"Context blocked: '{original}' → '{correction}' (POS: {token.pos_})")
            return False
            
        if token.lower_ in self.protected_words:
            self.logger.debug(f"Context blocked: '{original}' is protected common word")
            return False
            
        return True
    
    def correct_word(self, word: str) -> Tuple[str, bool, str]:
        """Correct a single word using multiple strategies with improved validation."""
        if not word or len(word) < 2:
            return word, False, "too_short"
        
        # Skip hyphen/slash terms
        if "-" in word or "/" in word or "_" in word:
            return word, False, "compound_term"
        
        # Don't correct numbers
        if word.isdigit():
            return word, False, "numeric"
        
        # CRITICAL: Apply protection logic first
        if self.should_protect(word):
            return word, False, "protected_term"
            
        norm = word.lower()
        
        # 1) Exact dictionary match (preserves original casing)
        if norm in self.lookup:
            corr = self.lookup[norm]
            return corr, corr != word, "exact_match"
            
        # 2) High-confidence fuzzy match with stricter validation
        best = process.extractOne(norm, list(self.lookup.keys()),
                                scorer=fuzz.ratio,
                                score_cutoff=92)  # Increased from 85 to 92
        if best and best[0] in self.lookup:
            # Apply Levenshtein distance validation
            levenshtein_dist = Levenshtein.distance(norm, best[0])
            
            # Skip corrections with very small Levenshtein distance (likely correct words)
            if levenshtein_dist <= 1:
                return word, False, "levenshtein_skip"
            
            # Additional protection for short words (abbreviations)
            if len(word) <= 4 and word.isupper():
                return word, False, "short_uppercase_protected"
                
            corr = self.lookup[best[0]]
            
            # Validate against common English words
            common_words = {'need', 'have', 'make', 'take', 'give', 'get', 'use', 'see', 'know', 'find', 'work', 'help', 'come', 'go', 'run', 'put', 'set', 'turn', 'move', 'keep', 'hold', 'show', 'play', 'feel', 'hear', 'read', 'write', 'speak', 'tell', 'ask', 'try', 'call', 'open', 'close', 'start', 'stop', 'end', 'begin', 'build', 'check', 'test', 'fix', 'change', 'save', 'load', 'send', 'add', 'cut', 'copy', 'pick', 'sort', 'join', 'split', 'fill', 'empty', 'clean', 'clear', 'mark', 'sign', 'list', 'plan', 'form', 'data', 'file', 'line', 'page', 'item', 'part', 'side', 'time', 'date', 'year', 'week', 'hour', 'size', 'type', 'kind', 'sort', 'name', 'code', 'text', 'word', 'note', 'link', 'path', 'step', 'task', 'goal', 'role', 'rule', 'case', 'fact', 'idea', 'view', 'news', 'area', 'room', 'seat', 'desk', 'door', 'wall', 'floor', 'road', 'car', 'bus', 'train', 'ship', 'boat', 'plane', 'bike', 'walk', 'drive', 'ride', 'fly', 'land', 'park', 'wait', 'stay', 'leave', 'arrive', 'enter', 'exit', 'climb', 'fall', 'rise', 'drop', 'lift', 'push', 'pull', 'carry', 'bring', 'take', 'give', 'send', 'get', 'buy', 'sell', 'pay', 'cost', 'free', 'cheap', 'dear', 'high', 'low', 'big', 'small', 'long', 'short', 'wide', 'thin', 'thick', 'deep', 'flat', 'round', 'square', 'full', 'empty', 'clean', 'dirty', 'new', 'old', 'young', 'fresh', 'dry', 'wet', 'hot', 'cold', 'warm', 'cool', 'fast', 'slow', 'quick', 'easy', 'hard', 'soft', 'loud', 'quiet', 'light', 'dark', 'bright', 'clear', 'good', 'bad', 'best', 'nice', 'fine', 'great', 'poor', 'rich', 'safe', 'sure', 'true', 'real', 'main', 'only', 'same', 'each', 'every', 'both', 'all', 'some', 'any', 'many', 'much', 'few', 'little', 'more', 'most', 'less', 'first', 'last', 'next', 'back', 'front', 'left', 'right', 'top', 'bottom', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'near', 'far', 'here', 'there', 'where', 'when', 'how', 'why', 'what', 'who', 'which', 'this', 'that', 'these', 'those', 'the', 'a', 'an', 'and', 'or', 'but', 'so', 'if', 'as', 'by', 'for', 'to', 'of', 'at', 'from', 'with', 'into', 'onto', 'upon', 'about', 'above', 'below', 'before', 'after', 'during', 'while', 'until', 'since', 'through', 'across', 'around', 'between', 'among', 'within', 'without', 'against', 'toward', 'towards', 'inside', 'outside', 'behind', 'beside', 'beyond', 'except', 'besides', 'instead', 'rather', 'either', 'neither', 'whether', 'although', 'though', 'unless', 'because', 'therefore', 'however', 'moreover', 'furthermore', 'nevertheless', 'nonetheless', 'meanwhile', 'otherwise', 'likewise', 'similarly', 'consequently', 'accordingly', 'thus', 'hence', 'indeed', 'actually', 'really', 'truly', 'certainly', 'probably', 'possibly', 'perhaps', 'maybe', 'always', 'never', 'often', 'sometimes', 'usually', 'rarely', 'seldom', 'hardly', 'barely', 'almost', 'quite', 'very', 'too', 'enough', 'just', 'still', 'yet', 'already', 'soon', 'late', 'early', 'now', 'then', 'today', 'tomorrow', 'yesterday', 'again', 'once', 'twice', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'costs', 'hydraulic', 'communications', 'equipment'}
            if norm in common_words and corr.lower() not in common_words:
                # Don't replace common English words with technical terms
                return word, False, "common_word_protected"
                
            return corr, True, "fuzzy_match"
            
        # 3) Very conservative spellchecker for clear misspellings only
        # Only correct if original is clearly misspelled and not a proper name/abbreviation
        if not word[0].isupper() or len(word) <= 4:  # Don't correct likely proper names/abbreviations
            candidates = self.spell.candidates(norm)
            if candidates and norm not in candidates:
                best_candidate = self.spell.correction(norm)
                if best_candidate and best_candidate != norm:
                    # Apply Levenshtein distance validation for spellchecker too
                    levenshtein_dist = Levenshtein.distance(norm, best_candidate)
                    
                    # For longer words (>6 chars), allow distance 1 corrections if it's a clear misspelling
                    # For shorter words, be more strict to avoid abbreviation corruption
                    if len(word) > 6 and levenshtein_dist == 1:
                        # Allow single-character corrections for longer words (likely real misspellings)
                        pass  # Don't skip, proceed with correction
                    elif levenshtein_dist <= 1:
                        return word, False, "spellcheck_levenshtein_skip"
                    
                    # Check if the correction is in our aerospace dictionary first
                    if best_candidate.lower() in self.lookup:
                        corr = self.lookup[best_candidate.lower()]
                        return corr, True, "conservative_spellcheck"
                    else:
                        # Use spellchecker suggestion for reasonable corrections
                        if levenshtein_dist >= 1:  # Allow distance 1+ corrections
                            return best_candidate, True, "conservative_spellcheck"
                
        return word, False, "no_correction"
    
    def correct_text(self, text: str) -> Tuple[str, Dict]:
        """
        Correct spelling in the entire text.
        
        Args:
            text: Text to correct
            
        Returns:
            Tuple of (corrected_text, correction_details)
        """
        if not text:
            return text, {"corrections": [], "stats": {"total_words": 0, "corrected_words": 0}}
        
        # Reset stats for this correction session
        session_stats = {
            'total_words': 0,
            'corrected_words': 0,
            'dictionary_corrections': 0,
            'fuzzy_corrections': 0,
            'spellchecker_corrections': 0,
            'protected_terms_skipped': 0,
            'levenshtein_skipped': 0,
            'short_uppercase_protected': 0,
            'common_word_protected': 0
        }
        
        # Simple word-by-word correction preserving original text structure
        # Use regex to find words and non-word characters separately
        tokens = re.findall(r'\b\w+\b|\W+', text)
        corrected_tokens = []
        corrections = []
        
        for i, token in enumerate(tokens):
            if re.match(r'\b\w+\b', token):  # It's a word
                session_stats["total_words"] += 1
                
                # Apply spell correction to this word only
                corrected, was_corrected, method = self.correct_word(token)
                
                if was_corrected:
                    session_stats["corrected_words"] += 1
                    
                    # Track correction type
                    if method == "exact_match":
                        session_stats["dictionary_corrections"] += 1
                    elif method == "fuzzy_match":
                        session_stats["fuzzy_corrections"] += 1
                    elif method == "conservative_spellcheck":
                        session_stats["spellchecker_corrections"] += 1
                    
                    corrections.append({
                        "position": i,
                        "original": token,
                        "corrected": corrected,
                        "method": method
                    })
                else:
                    # Track protection reasons
                    if method == "protected_term":
                        session_stats["protected_terms_skipped"] += 1
                    elif method == "levenshtein_skip" or method == "spellcheck_levenshtein_skip":
                        session_stats["levenshtein_skipped"] += 1
                    elif method == "short_uppercase_protected":
                        session_stats["short_uppercase_protected"] += 1
                    elif method == "common_word_protected":
                        session_stats["common_word_protected"] += 1
                
                corrected_tokens.append(corrected)
            else:
                # It's whitespace or punctuation - keep as is
                corrected_tokens.append(token)
        
        # Reconstruct the text maintaining original structure
        corrected_text = "".join(corrected_tokens)
        
        # Update cumulative stats
        self.stats["total_words"] += session_stats["total_words"]
        self.stats["corrected_words"] += session_stats["corrected_words"]
        self.stats["dictionary_corrections"] += session_stats["dictionary_corrections"]
        self.stats["fuzzy_corrections"] += session_stats["fuzzy_corrections"]
        self.stats["spellchecker_corrections"] += session_stats["spellchecker_corrections"]
        self.stats["protected_terms_skipped"] += session_stats["protected_terms_skipped"]
        self.stats["levenshtein_skipped"] += session_stats["levenshtein_skipped"]
        self.stats["short_uppercase_protected"] += session_stats["short_uppercase_protected"]
        self.stats["common_word_protected"] += session_stats["common_word_protected"]
        
        # Prepare correction details (compatible with original interface)
        correction_details = {
            "corrections": corrections,
            "stats": {
                "total_words": session_stats["total_words"],
                "corrected_words": session_stats["corrected_words"],
                "correction_rate": session_stats["corrected_words"] / session_stats["total_words"] if session_stats["total_words"] > 0 else 0,
                "dictionary_corrections": session_stats["dictionary_corrections"],
                "fuzzy_corrections": session_stats["fuzzy_corrections"],
                "spellchecker_corrections": session_stats["spellchecker_corrections"],
                "protected_terms_skipped": session_stats["protected_terms_skipped"],
                "levenshtein_skipped": session_stats["levenshtein_skipped"],
                "short_uppercase_protected": session_stats["short_uppercase_protected"],
                "common_word_protected": session_stats["common_word_protected"],
                "context_blocked": session_stats.get("context_blocked", 0)
            }
        }
        
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
            timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")
            safe_filename = re.sub(r'[^\w\-_.]', '_', filename)
            comparison_file = corrections_dir / f"{safe_filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_correction.txt"
            
            with open(comparison_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("AEROSPACE SPELL CORRECTION REPORT\n")
                f.write("=" * 80 + "\n")
                f.write(f"Original File: {filename}\n")
                f.write(f"Processing Date: {timestamp}\n\n")
                
                # Statistics
                stats = correction_details['stats']
                f.write("STATISTICS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total Words Processed: {stats['total_words']}\n")
                f.write(f"Words Corrected: {stats['corrected_words']}\n")
                f.write(f"Correction Rate: {stats.get('correction_rate', 0):.2%}\n")
                f.write(f"Dictionary Corrections: {stats.get('dictionary_corrections', 0)}\n")
                f.write(f"Fuzzy Matches: {stats.get('fuzzy_corrections', 0)}\n")
                f.write(f"SpellChecker Corrections: {stats.get('spellchecker_corrections', 0)}\n")
                f.write(f"Context-Blocked Corrections: {stats.get('context_blocked', 0)}\n\n")
                
                # Corrections list
                if correction_details['corrections']:
                    f.write("CORRECTIONS MADE:\n")
                    f.write("-" * 40 + "\n")
                    for corr in correction_details['corrections']:
                        f.write(f"  {corr['original']} → {corr['corrected']} ({corr['method']})\n")
                    f.write("\n")
                else:
                    f.write("No corrections were made.\n\n")
                
                # Text comparison
                f.write("ORIGINAL TEXT:\n")
                f.write("-" * 40 + "\n")
                f.write(original_text + "\n\n")
                
                f.write("CORRECTED TEXT:\n")
                f.write("-" * 40 + "\n")
                f.write(corrected_text + "\n")
                f.write("=" * 80 + "\n")
            
            self.logger.info(f"Enhanced correction report saved to: {comparison_file}")
            return str(comparison_file)
            
        except Exception as e:
            self.logger.error(f"Failed to save correction comparison: {e}")
            return ""
    
    def get_correction_stats(self) -> Dict:
        """Get correction statistics."""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset correction statistics."""
        self.stats = {
            'total_words': 0,
            'corrected_words': 0,
            'context_blocked': 0,
            'dictionary_corrections': 0,
            'fuzzy_corrections': 0,
            'spellchecker_corrections': 0,
            'protected_terms_skipped': 0,
            'levenshtein_skipped': 0,
            'short_uppercase_protected': 0,
            'common_word_protected': 0
        } 