#!/usr/bin/env python3
"""
Quick script to check for duplicate chunks in the database
"""
import sqlite3
import json
import os
import hashlib

def check_database_duplicates():
    """Check for duplicate chunks in the SQLite database."""
    try:
        conn = sqlite3.connect('documents.db')
        
        # Check total vs unique chunks by content
        cursor = conn.execute("""
            SELECT 
                COUNT(*) as total_chunks, 
                COUNT(DISTINCT chunk_text) as unique_by_content,
                COUNT(DISTINCT vector_id) as unique_by_vector_id
            FROM chunks 
            WHERE vectorized = 1
        """)
        result = cursor.fetchone()
        print(f"Vectorized chunks analysis:")
        print(f"  Total chunks: {result[0]}")
        print(f"  Unique by content: {result[1]}")
        print(f"  Unique by vector_id: {result[2]}")
        
        # Check for actual duplicate content (show more details)
        cursor = conn.execute("""
            SELECT chunk_text, COUNT(*) as count, vector_id
            FROM chunks 
            WHERE vectorized = 1
            GROUP BY chunk_text 
            HAVING COUNT(*) > 1 
            LIMIT 3
        """)
        duplicates = cursor.fetchall()
        
        if duplicates:
            print(f"\nFound {len(duplicates)} duplicate content groups in SQLite:")
            for i, (text, count, vector_id) in enumerate(duplicates[:2]):
                print(f"  {i+1}. Content appears {count} times (example vector_id: {vector_id})")
                print(f"      Preview: '{text[:200]}...'")
        else:
            print("\nNo duplicate content found in vectorized chunks in SQLite.")
            
        # Show sample of different chunks to verify diversity
        cursor = conn.execute("""
            SELECT DISTINCT chunk_text, vector_id 
            FROM chunks 
            WHERE vectorized = 1 
            LIMIT 3
        """)
        samples = cursor.fetchall()
        print(f"\nSample of unique chunks in SQLite:")
        for i, (text, vector_id) in enumerate(samples):
            print(f"  {i+1}. Vector ID {vector_id}: '{text[:100]}...'")
        
        conn.close()
        return result
        
    except Exception as e:
        print(f"Error checking database: {e}")
        return None

def check_vector_db_files():
    """Check for duplicate chunks in the vector database JSON files."""
    chunks_dir = "chunks_vectordb"
    if not os.path.exists(chunks_dir):
        print(f"Vector database directory '{chunks_dir}' not found.")
        return
    
    # Track different content fields separately
    content_fields_to_check = ['content', 'chunk_info.content', 'metadata.content']
    
    print(f"\nChecking vector database files in '{chunks_dir}'...")
    
    json_files = [f for f in os.listdir(chunks_dir) if f.endswith('.json')]
    print(f"Found {len(json_files)} JSON files to check.")
    
    # Check first 20 files for different content field patterns
    for field_path in content_fields_to_check:
        print(f"\n--- Checking field: {field_path} ---")
        
        content_hashes = {}
        vector_ids = set()
        duplicate_contents = []
        duplicate_vector_ids = []
        
        files_checked = 0
        for filename in json_files[:20]:  # Check first 20 files
            try:
                with open(os.path.join(chunks_dir, filename), 'r', encoding='utf-8') as f:
                    chunk_data = json.load(f)
                
                # Extract content based on field path
                content = None
                if field_path == 'content' and 'content' in chunk_data:
                    content = chunk_data['content']
                elif field_path == 'chunk_info.content' and 'chunk_info' in chunk_data and 'content' in chunk_data['chunk_info']:
                    content = chunk_data['chunk_info']['content']
                elif field_path == 'metadata.content' and 'metadata' in chunk_data and 'content' in chunk_data['metadata']:
                    content = chunk_data['metadata']['content']
                
                if content is None:
                    continue
                    
                files_checked += 1
                
                # Check content duplicates
                content_hash = hashlib.md5(content.encode()).hexdigest()
                
                if content_hash in content_hashes:
                    duplicate_contents.append((filename, content_hashes[content_hash], content[:100]))
                else:
                    content_hashes[content_hash] = filename
                
                # Check vector ID duplicates
                vector_id = chunk_data.get('vector_id', chunk_data.get('id'))
                if vector_id:
                    if vector_id in vector_ids:
                        duplicate_vector_ids.append((filename, vector_id))
                    else:
                        vector_ids.add(vector_id)
                        
            except Exception as e:
                print(f"    Error reading {filename}: {e}")
        
        print(f"  Files with {field_path} field: {files_checked}")
        print(f"  Unique content hashes: {len(content_hashes)}")
        print(f"  Unique vector IDs: {len(vector_ids)}")
        
        if duplicate_contents:
            print(f"  Found {len(duplicate_contents)} content duplicates:")
            for i, (file1, file2, content) in enumerate(duplicate_contents[:3]):
                print(f"    {i+1}. Files '{file1}' and '{file2}' have same content: '{content}...'")
        else:
            print(f"  No content duplicates found for {field_path}")
        
        # Show sample content for this field
        if content_hashes:
            sample_files = list(content_hashes.values())[:3]
            print(f"  Sample content from {field_path}:")
            for i, filename in enumerate(sample_files):
                try:
                    with open(os.path.join(chunks_dir, filename), 'r', encoding='utf-8') as f:
                        chunk_data = json.load(f)
                    
                    if field_path == 'content' and 'content' in chunk_data:
                        sample_content = chunk_data['content'][:100]
                    elif field_path == 'chunk_info.content':
                        sample_content = chunk_data['chunk_info']['content'][:100]
                    elif field_path == 'metadata.content':
                        sample_content = chunk_data['metadata']['content'][:100]
                    
                    print(f"    {i+1}. {filename}: '{sample_content}...'")
                except:
                    pass

if __name__ == "__main__":
    print("Checking for duplicate chunks...\n")
    
    # Check SQLite database
    check_database_duplicates()
    
    # Check vector database files
    check_vector_db_files() 