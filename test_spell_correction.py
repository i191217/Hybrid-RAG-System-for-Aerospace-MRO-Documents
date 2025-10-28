#!/usr/bin/env python3
"""
Test script for spell correction service.
Tests the Levenshtein distance-based spell correction with aerospace dictionary.
"""

import sys
import os
import logging

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.spell_correction_service import SpellCorrectionService

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_spell_correction():
    """Test the spell correction service with sample aerospace text."""
    
    # Sample text with intentional spelling mistakes
    test_text = """
    The aircaft maintenance manual specifies that the IDG (Integrated Drive Generater) 
    must be inspected every 500 flight hours. The A-Chek procedure includes checking 
    the ACARS system and the ADC (Air Data Computor) for proper operation.
    
    The airworthyness directive (AD) requires replacement of the alternater current 
    components in the avionics bay. The maintanance crew should follow the 8130-3 
    certificaton process for all repaired parts.
    
    During the C-check, inspect the auxilliary power unit (APU) and verify that 
    all safty systems are operational. The technicien should document all findings 
    in the maintanance log.
    """
    
    try:
        logger.info("Initializing spell correction service...")
        spell_service = SpellCorrectionService()
        
        logger.info("Testing spell correction...")
        corrected_text, correction_details = spell_service.correct_text(test_text)
        
        # Display results
        print("\n" + "="*80)
        print("SPELL CORRECTION TEST RESULTS")
        print("="*80)
        
        stats = correction_details.get('stats', {})
        corrections = correction_details.get('corrections', [])
        
        print(f"Total words processed: {stats.get('total_words', 0)}")
        print(f"Words corrected: {stats.get('corrected_words', 0)}")
        print(f"Correction rate: {stats.get('correction_rate', 0):.2%}")
        print()
        
        if corrections:
            print("CORRECTIONS MADE:")
            print("-" * 40)
            for correction in corrections:
                print(f"  '{correction['original']}' → '{correction['corrected']}' ({correction['method']})")
            print()
        
        print("ORIGINAL TEXT:")
        print("-" * 40)
        print(test_text.strip())
        print()
        
        print("CORRECTED TEXT:")
        print("-" * 40)
        print(corrected_text.strip())
        print()
        
        # Save comparison file
        comparison_file = spell_service.save_correction_comparison(
            original_text=test_text,
            corrected_text=corrected_text,
            correction_details=correction_details,
            filename="test_aerospace_text.txt"
        )
        
        print(f"Detailed comparison saved to: {comparison_file}")
        
        # Get service statistics
        service_stats = spell_service.get_correction_stats()
        print("\nSERVICE STATISTICS:")
        print("-" * 40)
        for key, value in service_stats.items():
            print(f"  {key}: {value}")
        
        print("\n" + "="*80)
        print("TEST COMPLETED SUCCESSFULLY")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_spell_correction() 