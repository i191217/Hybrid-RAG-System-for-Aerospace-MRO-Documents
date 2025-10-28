#!/usr/bin/env python3
"""
Diagnostic script to check what data is in the vector database.
"""

import sys
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from config import config
from vector_db import get_vector_db

def check_database():
    """Check the vector database contents."""
    print("🔍 Checking Vector Database Contents")
    print("=" * 50)
    
    try:
        # Initialize vector database
        vector_db = get_vector_db()
        
        print(f"📊 Database Configuration:")
        print(f"  Collection: {config.COLLECTION_NAME}")
        print(f"  Vector DB Type: {config.VECTOR_DB_TYPE}")
        
        # Try to get some basic information
        if hasattr(vector_db, 'client'):
            # For Qdrant, get collection info
            try:
                collection_info = vector_db.client.get_collection(config.COLLECTION_NAME)
                print(f"\n✅ Collection '{config.COLLECTION_NAME}' exists")
                print(f"  Points count: {collection_info.points_count}")
                print(f"  Vector size: {collection_info.config.params.vectors.size}")
                print(f"  Distance: {collection_info.config.params.vectors.distance}")
                
                if collection_info.points_count > 0:
                    # Try to get a sample of points
                    sample_points = vector_db.client.scroll(
                        collection_name=config.COLLECTION_NAME,
                        limit=3,
                        with_payload=True,
                        with_vectors=False
                    )
                    
                    print(f"\n📄 Sample Documents:")
                    for i, point in enumerate(sample_points[0], 1):
                        metadata = point.payload
                        print(f"  {i}. ID: {point.id}")
                        print(f"     Source: {metadata.get('source', 'Unknown')}")
                        print(f"     Chunk Index: {metadata.get('chunk_index', 'Unknown')}")
                        print(f"     Content Preview: {metadata.get('content', '')[:100]}...")
                        print()
                
                else:
                    print("\n⚠️  Collection is empty - no documents have been processed yet!")
                    print("   Run 'python run_pipeline.py' to process documents first.")
                    
            except Exception as e:
                print(f"\n❌ Error accessing collection: {e}")
                print("   Collection may not exist or be accessible.")
        
        # Test a simple query
        print(f"\n🔍 Testing Simple Query:")
        try:
            from embedding_service import get_embedding_service
            embedding_service = get_embedding_service()
            
            test_query = "test"
            query_embedding = embedding_service.embed_text(test_query)
            
            if query_embedding:
                results = vector_db.search(
                    query_vector=query_embedding,
                    limit=3,
                    score_threshold=0.0  # Very low threshold to get any results
                )
                
                print(f"  Query: '{test_query}'")
                print(f"  Results found: {len(results)}")
                
                for i, result in enumerate(results, 1):
                    print(f"    {i}. Score: {result.score:.3f}")
                    print(f"       Source: {result.metadata.get('source', 'Unknown')}")
            else:
                print("  ❌ Failed to generate embedding for test query")
                
        except Exception as e:
            print(f"  ❌ Query test failed: {e}")
        
    except Exception as e:
        print(f"❌ Database check failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function."""
    if not Path(".env").exists():
        print("❌ Error: .env file not found. Please run from the improved-doc-processor directory.")
        return
    
    check_database()
    
    print(f"\n🎯 **Recommendations:**")
    print("  1. If collection is empty, run: python run_pipeline.py")
    print("  2. If no results found, try lower similarity thresholds")
    print("  3. Check that your documents were processed correctly")

if __name__ == "__main__":
    main() 