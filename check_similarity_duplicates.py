#!/usr/bin/env python3
"""
Enhanced duplicate detection using 50% similarity threshold.
Tests the new similarity-based deduplication system across the entire pipeline.
"""

import sys
import os
import json
import logging
from pathlib import Path
from collections import defaultdict

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.similarity_service import SimilarityService
from services.database_service import DatabaseService
from core.vector_db import get_vector_db

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimilarityDuplicateAnalyzer:
    """Analyzer for detecting duplicates using similarity thresholds."""
    
    def __init__(self):
        self.similarity_service = SimilarityService(similarity_threshold=0.5)  # 50% threshold
        self.db_service = DatabaseService()
        try:
            self.vector_db = get_vector_db()
        except Exception as e:
            logger.warning(f"Vector database not available: {e}")
            self.vector_db = None
    
    def analyze_document_similarities(self):
        """Analyze document similarities in SQLite database."""
        print("🔍 ANALYZING DOCUMENT SIMILARITIES (50% THRESHOLD)")
        print("="*60)
        
        try:
            documents = self.db_service.get_documents_summary()
            
            if len(documents) < 2:
                print("✅ Not enough documents to compare")
                return
            
            similar_pairs = []
            total_comparisons = 0
            
            print(f"📊 Comparing {len(documents)} documents...")
            
            for i, doc1 in enumerate(documents):
                for doc2 in documents[i+1:]:
                    total_comparisons += 1
                    content1 = doc1.get('raw_text', '')
                    content2 = doc2.get('raw_text', '')
                    
                    if content1 and content2:
                        is_similar, similarity = self.similarity_service.are_documents_similar(
                            content1, content2
                        )
                        
                        if is_similar:
                            similar_pairs.append({
                                'doc1': doc1['filename'],
                                'doc2': doc2['filename'],
                                'similarity': similarity,
                                'doc1_id': doc1['id'],
                                'doc2_id': doc2['id'],
                                'doc1_size': len(content1),
                                'doc2_size': len(content2)
                            })
                            
                            print(f"   📄 Similar pair found:")
                            print(f"      - '{doc1['filename']}' (ID: {doc1['id']}, {len(content1)} chars)")
                            print(f"      - '{doc2['filename']}' (ID: {doc2['id']}, {len(content2)} chars)")
                            print(f"      - Similarity: {similarity:.1%}")
            
            print(f"\n📈 SIMILARITY ANALYSIS RESULTS:")
            print(f"   - Total comparisons: {total_comparisons}")
            print(f"   - Similar pairs found: {len(similar_pairs)}")
            
            if similar_pairs:
                avg_similarity = sum(pair['similarity'] for pair in similar_pairs) / len(similar_pairs)
                max_similarity = max(pair['similarity'] for pair in similar_pairs)
                print(f"   - Average similarity: {avg_similarity:.1%}")
                print(f"   - Maximum similarity: {max_similarity:.1%}")
                print(f"\n⚠️  DUPLICATES DETECTED! Consider cleaning up similar documents.")
            else:
                print("✅ No similar documents found above 50% threshold")
                
        except Exception as e:
            print(f"❌ Error analyzing document similarities: {e}")
    
    def analyze_chunk_similarities(self):
        """Analyze chunk similarities in vector database."""
        print("\n🔍 ANALYZING CHUNK SIMILARITIES")
        print("="*60)
        
        if not self.vector_db:
            print("❌ Vector database not available")
            return
        
        try:
            all_points = self.vector_db.get_all_points()
            
            if len(all_points) < 2:
                print("✅ Not enough chunks to compare")
                return
            
            # Group chunks by document for analysis
            doc_chunks = defaultdict(list)
            for point in all_points:
                metadata = point.get('metadata', {})
                filename = metadata.get('source', metadata.get('filename', 'unknown'))
                content = metadata.get('content', '')
                
                if content:
                    doc_chunks[filename].append({
                        'content': content,
                        'vector_id': point.get('id', 'unknown')
                    })
            
            total_similar_pairs = 0
            total_chunks = 0
            
            for filename, chunks in doc_chunks.items():
                total_chunks += len(chunks)
                print(f"\n📄 Analyzing {filename}: {len(chunks)} chunks")
                
                if len(chunks) < 2:
                    print(f"   ✅ Only one chunk, no comparison needed")
                    continue
                
                # Get similarity statistics for this document's chunks
                stats = self.similarity_service.get_similarity_stats(chunks)
                
                print(f"   📊 Similarity Statistics:")
                print(f"      - Total comparisons: {stats['total_comparisons']}")
                print(f"      - Average similarity: {stats['avg_similarity']:.1%}")
                print(f"      - Maximum similarity: {stats['max_similarity']:.1%}")
                print(f"      - Pairs above 50% threshold: {stats['above_threshold']}")
                
                if stats['above_threshold'] > 0:
                    print(f"   ⚠️  Found {stats['above_threshold']} similar chunk pairs!")
                    total_similar_pairs += stats['above_threshold']
                    
                    # Show some examples
                    similar_count = 0
                    for i, chunk1 in enumerate(chunks):
                        for chunk2 in chunks[i+1:]:
                            is_similar, similarity = self.similarity_service.are_chunks_similar(
                                chunk1['content'], chunk2['content']
                            )
                            if is_similar and similar_count < 3:  # Show max 3 examples
                                print(f"      - Chunk {i+1} ↔ Chunk {i+2}: {similarity:.1%} similar")
                                print(f"        Preview 1: {chunk1['content'][:80]}...")
                                print(f"        Preview 2: {chunk2['content'][:80]}...")
                                similar_count += 1
                else:
                    print(f"   ✅ No similar chunks found")
            
            print(f"\n📊 OVERALL CHUNK ANALYSIS:")
            print(f"   - Total documents: {len(doc_chunks)}")
            print(f"   - Total chunks: {total_chunks}")
            print(f"   - Similar chunk pairs: {total_similar_pairs}")
            
            if total_similar_pairs > 0:
                print(f"   ⚠️  Similarity detected! Consider reviewing chunk deduplication.")
            else:
                print(f"   ✅ Good chunk diversity across all documents!")
                
        except Exception as e:
            print(f"❌ Error analyzing chunk similarities: {e}")
    
    def test_similarity_threshold(self, threshold: float = 0.5):
        """Test retrieval with similarity threshold."""
        print(f"\n🔍 TESTING RETRIEVAL WITH {threshold:.0%} SIMILARITY THRESHOLD")
        print("="*60)
        
        try:
            from core.hybrid_query_engine import HybridAerospaceQueryEngine
            
            engine = HybridAerospaceQueryEngine()
            
            # Test with a common aerospace query
            test_query = "aerospace maintenance procedures and requirements"
            print(f"📝 Test Query: '{test_query}'")
            
            result = engine.query(test_query, max_chunks=5)
            
            if result.sources:
                print(f"📊 Retrieved {len(result.sources)} chunks:")
                
                # Analyze retrieved chunks
                chunks_for_analysis = []
                for i, source in enumerate(result.sources):
                    content = source.get('content', '')
                    filename = source.get('filename', 'unknown')
                    
                    chunks_for_analysis.append({"content": content})
                    print(f"   {i+1}. {filename}")
                    print(f"      Content preview: {content[:100]}...")
                
                # Get similarity statistics
                stats = self.similarity_service.get_similarity_stats(chunks_for_analysis)
                
                print(f"\n📈 RETRIEVAL DIVERSITY ANALYSIS:")
                print(f"   - Total comparisons: {stats['total_comparisons']}")
                print(f"   - Average similarity: {stats['avg_similarity']:.1%}")
                print(f"   - Maximum similarity: {stats['max_similarity']:.1%}")
                print(f"   - Chunks above {threshold:.0%} threshold: {stats['above_threshold']}")
                
                if stats['max_similarity'] >= threshold:
                    print(f"   ⚠️  Some chunks are above {threshold:.0%} similarity!")
                    print(f"   💡 Consider stricter deduplication or higher thresholds.")
                else:
                    print(f"   ✅ All chunks are below {threshold:.0%} similarity - excellent diversity!")
                    
                # Check for exact duplicates
                content_set = set()
                exact_duplicates = 0
                for source in result.sources:
                    content = source.get('content', '').strip()
                    if content in content_set:
                        exact_duplicates += 1
                    content_set.add(content)
                
                if exact_duplicates > 0:
                    print(f"   ❌ Found {exact_duplicates} exact duplicate chunks!")
                else:
                    print(f"   ✅ No exact duplicates found!")
                    
            else:
                print("❌ No results returned from query")
                
        except Exception as e:
            print(f"❌ Error in retrieval test: {e}")
    
    def test_similarity_service(self):
        """Test the similarity service functionality."""
        print(f"\n🧪 TESTING SIMILARITY SERVICE FUNCTIONALITY")
        print("="*60)
        
        # Test data
        test_cases = [
            {
                "name": "Identical texts",
                "text1": "Aircraft maintenance requires regular inspection of all components.",
                "text2": "Aircraft maintenance requires regular inspection of all components.",
                "expected_similarity": 1.0
            },
            {
                "name": "Very similar texts",
                "text1": "Aircraft maintenance requires regular inspection of all components.",
                "text2": "Aircraft maintenance needs regular inspection of all parts.",
                "expected_similarity": 0.7  # Should be high
            },
            {
                "name": "Different texts",
                "text1": "Aircraft maintenance requires regular inspection of all components.",
                "text2": "The weather is sunny today with clear skies.",
                "expected_similarity": 0.1  # Should be low
            },
            {
                "name": "Document headers (should be normalized)",
                "text1": "Document: file1.pdf\nPage: 1\n\nAircraft maintenance procedures are critical.",
                "text2": "Document: file2.pdf\nPage: 2\n\nAircraft maintenance procedures are critical.",
                "expected_similarity": 0.9  # Should be high after normalization
            }
        ]
        
        for test_case in test_cases:
            similarity = self.similarity_service.calculate_content_similarity(
                test_case["text1"], test_case["text2"]
            )
            
            print(f"📋 {test_case['name']}:")
            print(f"   - Calculated similarity: {similarity:.2%}")
            print(f"   - Expected range: ~{test_case['expected_similarity']:.0%}")
            
            # Check if similarity is reasonable
            expected = test_case["expected_similarity"]
            if abs(similarity - expected) < 0.3:  # Within 30% tolerance
                print(f"   ✅ Result within expected range")
            else:
                print(f"   ⚠️  Result outside expected range")
            print()

def main():
    """Main function to run all similarity analyses."""
    print("🚀 SIMILARITY-BASED DUPLICATE ANALYSIS")
    print("="*60)
    print("Using 50% similarity threshold for duplicate detection\n")
    
    try:
        analyzer = SimilarityDuplicateAnalyzer()
        
        # Run all analyses
        analyzer.test_similarity_service()
        analyzer.analyze_document_similarities()
        analyzer.analyze_chunk_similarities()
        analyzer.test_similarity_threshold(0.5)
        
        print("\n🎯 SUMMARY & RECOMMENDATIONS:")
        print("="*60)
        print("✅ Similarity-based deduplication system implemented")
        print("🔧 Key Features:")
        print("   - 50% threshold for document similarity")
        print("   - 60% threshold for chunk similarity (higher for diversity)")
        print("   - Multi-method similarity calculation (sequence, jaccard, cosine, fuzzy)")
        print("   - Content normalization (removes headers, formatting artifacts)")
        print("   - Real-time deduplication during retrieval")
        print("\n💡 Next Steps:")
        print("   1. If duplicates found: Clean database and reprocess with new system")
        print("   2. Monitor RAGAS evaluation for improved diversity")
        print("   3. Adjust thresholds if needed based on domain requirements")
        print("   4. Consider file-hash based duplicate prevention for identical files")
        
    except Exception as e:
        print(f"❌ Error running analysis: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 