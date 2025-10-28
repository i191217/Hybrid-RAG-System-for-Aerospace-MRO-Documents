#!/usr/bin/env python3
"""
enhanced_aero_spellcheck.py

An advanced aerospace spell correction script combining:
- Phrase protection and hyphen/slash handling from latest_spell_check.py
- Contextual validation and statistics from spell_correction_service.py
- Optional spaCy integration for contextual analysis
"""

import re
import sys
import io
import json
import unicodedata
import argparse
import logging
from typing import Dict, Optional, Tuple, List
from datetime import datetime
from pathlib import Path
from rapidfuzz import process, fuzz
from spellchecker import SpellChecker

try:
    from pypdf import PdfReader  # Preferred modern library
    PDF_SUPPORT = True
except ImportError:
    try:
        from PyPDF2 import PdfReader  # Fallback to older version
        PDF_SUPPORT = True
    except ImportError:
        PDF_SUPPORT = False

try:
    import spacy
    from spacy.language import Language
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

class EnhancedAeroSpellChecker:
    """Advanced aerospace spell checker combining multiple correction strategies."""
    
    def __init__(self, dictionary_path: str):
        """
        Initialize the spell checker.
        
        Args:
            dictionary_path: Path to the aerospace dictionary JSON file
        """
        self.logger = self._setup_logger()
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
            'spellchecker_corrections': 0
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
        
        # 3. Common abbreviations
        protected.update({
            'itp', 'nato', 'usaf', 'faa', 'icao', 'iata',
            'fbo', 'atis', 'notam', 'taf', 'metar'
        })
        
        return protected


    
    def _setup_logger(self):
        """Configure logging for the spell checker."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
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
        with open(dict_path, encoding="utf-8") as f:
            data = json.load(f)
        
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
            .replace("•", " ").replace("", " ")
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
        """Correct a single word using multiple strategies."""
        if not word or len(word) <= 2:
            return word, False, "too_short"
        
        # Skip hyphen/slash terms
        if "-" in word or "/" in word or "_" in word:
            return word, False, "compound_term"
        
        if self.should_protect(word):
            return word, False, "protected_term"
            
        norm = word.lower()
        
        # # Handle protected phrases
        # if "_" in word:
        #     orig = word.replace("_", " ")
        #     corr = self.lookup.get(orig.lower(), orig)
        #     return corr, corr != orig, "phrase_correction"
        
        # norm = word.lower()
        
        # # 1) Exact dictionary match
        # if norm in self.lookup:
        #     corr = self.lookup[norm]
        #     if corr != word and self.is_contextually_valid(word, corr):
        #         self.stats["dictionary_corrections"] += 1
        #         return corr, True, "exact_match"
        #     return word, False, "already_correct"
        
        # # 2) Fuzzy match with threshold
        # best = process.extractOne(norm, list(self.lookup.keys()),
        #                         scorer=fuzz.ratio,
        #                         score_cutoff=90)
        # if best and best[0] in self.lookup:
        #     corr = self.lookup[best[0]]
        #     if self.is_contextually_valid(word, corr):
        #         self.stats["fuzzy_corrections"] += 1
        #         return corr, True, "fuzzy_match"
        
        # # 3) SpellChecker fallback
        # cand = self.spell.correction(norm) or word
        # corr = self.lookup.get(cand.lower(), cand)
        # if corr != word and self.is_contextually_valid(word, corr):
        #     self.stats["spellchecker_corrections"] += 1
        #     return corr, True, "spellchecker"
        
        # return word, False, "no_correction"

        # 1) Exact dictionary match (preserves original casing)
        if norm in self.lookup:
            corr = self.lookup[norm]
            return corr, corr != word, "exact_match"
            
        # 2) Fuzzy match only if score is very high (95+)
        best = process.extractOne(norm, list(self.lookup.keys()),
                                scorer=fuzz.ratio,
                                score_cutoff=95)
        if best and best[0] in self.lookup:
            corr = self.lookup[best[0]]
            return corr, True, "fuzzy_match"
            
        # 3) Very conservative spellchecker fallback
        # Only correct if original is clearly misspelled
        candidates = self.spell.candidates(norm)
        if candidates and norm not in candidates:
            best_candidate = self.spell.correction(norm)
            # Only accept if it's a clear correction
            if best_candidate and len(best_candidate) >= len(norm) - 1:
                corr = self.lookup.get(best_candidate.lower(), best_candidate)
                return corr, True, "conservative_spellcheck"
                
        return word, False, "no_correction"
    
    def correct_text(self, text: str) -> Tuple[str, Dict]:
        """Correct spelling in the entire text."""
        # Normalization and preprocessing
        text = self.normalize_unicode(text)
        text = self.fix_hyphens(text)
        text = self.protect_phrases(text)
        
        # Tokenization and correction
        tokens = self.tokenize(text)
        corrected_tokens = []
        corrections = []
        
        for i, tok in enumerate(tokens):
            self.stats["total_words"] += 1
            corrected, was_corrected, method = self.correct_word(tok)
            
            # Handle split tokens
            pieces = self.try_split(corrected) if was_corrected else [corrected]
            
            for p in pieces:
                if re.fullmatch(r"[\W_]+", p):
                    continue
                if len(p) <= 2 and p.lower() not in self.lookup:
                    continue
                
                corrected_tokens.append(p)
            
            if was_corrected:
                self.stats["corrected_words"] += 1
                corrections.append({
                    "position": i,
                    "original": tok,
                    "corrected": corrected,
                    "method": method
                })
        
        corrected_text = " ".join(corrected_tokens)
        
        # Prepare correction details
        correction_details = {
            "corrections": corrections,
            "stats": self.stats.copy()
        }
        
        return corrected_text, correction_details
    
    def save_correction_report(self, original_text: str, corrected_text: str, 
                             correction_details: Dict, filename: str) -> str:
        """Save a detailed correction report to file."""
        try:
            corrections_dir = Path("text_corrections")
            corrections_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_filename = re.sub(r'[^\w\-_.]', '_', filename)
            report_file = corrections_dir / f"{safe_filename}_{timestamp}_report.txt"
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("AEROSPACE SPELL CORRECTION REPORT\n")
                f.write("=" * 80 + "\n")
                f.write(f"Original File: {filename}\n")
                f.write(f"Processing Date: {datetime.now().isoformat()}\n\n")
                
                # Statistics
                stats = correction_details['stats']
                f.write("STATISTICS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total Words Processed: {stats['total_words']}\n")
                f.write(f"Words Corrected: {stats['corrected_words']}\n")
                f.write(f"Correction Rate: {stats['corrected_words']/stats['total_words']:.2%}\n")
                f.write(f"Dictionary Corrections: {stats['dictionary_corrections']}\n")
                f.write(f"Fuzzy Matches: {stats['fuzzy_corrections']}\n")
                f.write(f"SpellChecker Corrections: {stats['spellchecker_corrections']}\n")
                f.write(f"Context-Blocked Corrections: {stats['context_blocked']}\n\n")
                
                # Corrections list
                if correction_details['corrections']:
                    f.write("CORRECTIONS MADE:\n")
                    f.write("-" * 40 + "\n")
                    for corr in correction_details['corrections']:
                        f.write(f"  {corr['original']} → {corr['corrected']} ({corr['method']})\n")
                    f.write("\n")
                
                # Text comparison
                f.write("ORIGINAL TEXT:\n")
                f.write("-" * 40 + "\n")
                f.write(original_text + "\n\n")
                
                f.write("CORRECTED TEXT:\n")
                f.write("-" * 40 + "\n")
                f.write(corrected_text + "\n")
                f.write("=" * 80 + "\n")
            
            self.logger.info(f"Correction report saved to: {report_file}")
            return str(report_file)
            
        except Exception as e:
            self.logger.error(f"Failed to save correction report: {e}")
            return ""

def main():
    """Command line interface for the spell checker."""
    parser = argparse.ArgumentParser(description="Enhanced Aerospace Spell Checker")
    parser.add_argument("-d", "--dict", required=True,
                       help="Path to aerospace dictionary JSON file")
    parser.add_argument("-i", "--input", required=True,
                       help="Input file (txt, json, or pdf)")
    parser.add_argument("-o", "--output", help="Output file for corrected text")
    parser.add_argument("-r", "--report", help="Generate correction report file")
    args = parser.parse_args()
    
    # Load input text based on file type
    input_lower = args.input.lower()
    
    if input_lower.endswith(".pdf"):
        if not PDF_SUPPORT:
            print("Error: PDF support requires pypdf or PyPDF2 package", file=sys.stderr)
            print("Install with: pip install pypdf", file=sys.stderr)
            sys.exit(1)
            
        try:
            with open(args.input, "rb") as f:
                reader = PdfReader(f)
                raw_text = "\n".join(page.extract_text() for page in reader.pages)
        except Exception as e:
            print(f"Error reading PDF: {e}", file=sys.stderr)
            sys.exit(1)
            
    elif input_lower.endswith(".json"):
        with open(args.input, encoding="utf-8") as f:
            data = json.load(f)
        raw_text = data.get("chunk_info", {}).get("content", "")
    else:
        with open(args.input, encoding="utf-8") as f:
            raw_text = f.read()
    
    if not raw_text:
        print("Error: No text content found in input", file=sys.stderr)
        sys.exit(1)
    
    # Process text
    spell_checker = EnhancedAeroSpellChecker(args.dict)
    corrected_text, correction_details = spell_checker.correct_text(raw_text)
    
    # Save or display results
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(corrected_text)
    else:
        print(corrected_text)
    
    # Generate report if requested
    if args.report:
        report_path = spell_checker.save_correction_report(
            raw_text, corrected_text, correction_details, args.input
        )
        if report_path:
            print(f"Correction report saved to: {report_path}")

if __name__ == "__main__":
    main()