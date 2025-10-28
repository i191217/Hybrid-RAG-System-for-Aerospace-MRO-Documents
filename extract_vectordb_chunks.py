#!/usr/bin/env python3
"""
Script to extract all chunks from the QDrant vector database and save to a text file.
"""

import logging
import sys
import os
from typing import List, Dict, Any

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from vector_db import get_vector_db

def extract_all_chunks():
    """Extract all chunks from the vector database and save to file."""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("extract_chunks")
    
    try:
        # Connect to vector database
        logger.info("Connecting to vector database...")
        vector_db = get_vector_db()
        
        if not vector_db.collection_exists():
            logger.error(f"Collection '{config.COLLECTION_NAME}' does not exist!")
            return
        
        # Get collection info
        collection_info = vector_db.get_collection_info()
        logger.info(f"Collection info: {collection_info}")
        
        # For QDrant, use scroll to get all points
        if hasattr(vector_db, 'client'):  # QDrant database
            logger.info("Extracting all chunks from QDrant database...")
            
            output_file = "vectordb_chunks.txt"
            all_chunks = []
            
            # Use scroll to get all points
            offset = None
            batch_size = 100
            total_processed = 0
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("VECTOR DATABASE CHUNKS EXTRACTION\n")
                f.write("=" * 80 + "\n")
                f.write(f"Collection: {config.COLLECTION_NAME}\n")
                f.write(f"Total points: {collection_info.get('points_count', 'Unknown')}\n")
                f.write("=" * 80 + "\n\n")
                
                while True:
                    try:
                        # Get next batch of points
                        scroll_result = vector_db.client.scroll(
                            collection_name=config.COLLECTION_NAME,
                            limit=batch_size,
                            offset=offset,
                            with_payload=True,
                            with_vectors=False  # We don't need the actual vectors, just metadata
                        )
                        
                        points, next_offset = scroll_result
                        
                        if not points:
                            break
                        
                        # Process this batch
                        for i, point in enumerate(points):
                            total_processed += 1
                            
                            # Extract metadata
                            payload = point.payload or {}
                            
                            f.write(f"CHUNK {total_processed}\n")
                            f.write("-" * 60 + "\n")
                            f.write(f"ID: {point.id}\n")
                            f.write(f"Source: {payload.get('source', 'Unknown')}\n")
                            f.write(f"Chunk Index: {payload.get('chunk_index', 'Unknown')}\n")
                            f.write(f"Original Filename: {payload.get('original_filename', 'Unknown')}\n")
                            f.write(f"Processing Timestamp: {payload.get('processing_timestamp', 'Unknown')}\n")
                            
                            # Write the actual content
                            content = payload.get('content', '')
                            if content:
                                f.write(f"\nCONTENT ({len(content)} characters):\n")
                                f.write("-" * 40 + "\n")
                                f.write(content)
                                f.write("\n")
                            else:
                                f.write("\nCONTENT: [No content available]\n")
                            
                            # Additional metadata
                            f.write(f"\nADDITIONAL METADATA:\n")
                            for key, value in payload.items():
                                if key not in ['source', 'chunk_index', 'content', 'original_filename', 'processing_timestamp']:
                                    f.write(f"  {key}: {value}\n")
                            
                            f.write("\n" + "=" * 80 + "\n\n")
                        
                        logger.info(f"Processed {total_processed} chunks so far...")
                        
                        # Check if we have more data
                        if next_offset is None:
                            break
                        
                        offset = next_offset
                        
                    except Exception as e:
                        logger.error(f"Error during scroll operation: {e}")
                        break
                
                # Write summary
                f.write("EXTRACTION SUMMARY\n")
                f.write("=" * 80 + "\n")
                f.write(f"Total chunks extracted: {total_processed}\n")
                f.write(f"Collection: {config.COLLECTION_NAME}\n")
                f.write(f"Database type: {config.VECTOR_DB_TYPE}\n")
                
                # Group by source files
                logger.info("Generating source file summary...")
                vector_db_temp = get_vector_db()  # Fresh connection for final scan
                final_scroll = vector_db_temp.client.scroll(
                    collection_name=config.COLLECTION_NAME,
                    limit=10000,  # Get all at once for summary
                    with_payload=True,
                    with_vectors=False
                )
                
                final_points, _ = final_scroll
                source_summary = {}
                
                for point in final_points:
                    source = point.payload.get('source', 'Unknown')
                    if source not in source_summary:
                        source_summary[source] = {
                            'count': 0,
                            'total_chars': 0,
                            'chunk_indices': []
                        }
                    
                    source_summary[source]['count'] += 1
                    content = point.payload.get('content', '')
                    source_summary[source]['total_chars'] += len(content)
                    chunk_idx = point.payload.get('chunk_index', 0)
                    source_summary[source]['chunk_indices'].append(chunk_idx)
                
                f.write(f"\nSOURCE FILES SUMMARY ({len(source_summary)} files):\n")
                f.write("-" * 60 + "\n")
                
                for source, stats in source_summary.items():
                    f.write(f"📁 {source}\n")
                    f.write(f"   Chunks: {stats['count']}\n")
                    f.write(f"   Total content: {stats['total_chars']:,} characters\n")
                    f.write(f"   Chunk indices: {sorted(stats['chunk_indices'])}\n")
                    f.write("\n")
                
                vector_db_temp.close()
            
            logger.info(f"✅ Successfully extracted {total_processed} chunks to {output_file}")
            print(f"\n🎉 Extraction complete!")
            print(f"📁 File saved: {output_file}")
            print(f"📊 Total chunks: {total_processed}")
            print(f"📁 Unique sources: {len(source_summary) if 'source_summary' in locals() else 'Unknown'}")
            
        else:
            logger.error("Currently only QDrant database extraction is supported")
            
    except Exception as e:
        logger.error(f"Failed to extract chunks: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
    
    finally:
        if 'vector_db' in locals():
            vector_db.close()

if __name__ == "__main__":
    extract_all_chunks() 