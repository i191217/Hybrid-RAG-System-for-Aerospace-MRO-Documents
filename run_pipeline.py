#!/usr/bin/env python3
"""
Simple runner script to execute the document processing pipeline.
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import DocumentProcessingPipeline
from config import config

def main():
    """Run the pipeline with default settings."""
    
    print("🚀 Document Processing Pipeline Runner")
    print("=" * 50)
    
    # Check if .env file exists
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found!")
        print("Please create a .env file with your configuration.")
        print("You can copy from .env.example and fill in your values.")
        return False
    
    # Display current configuration (without sensitive data)
    print("📋 Current Configuration:")
    print(f"  Vector DB Type: {config.VECTOR_DB_TYPE}")
    print(f"  Collection Name: {config.COLLECTION_NAME}")
    print(f"  Embedding Model: {config.EMBEDDING_MODEL}")
    print(f"  Embedding Dimension: {config.EMBEDDING_DIMENSION}")
    print(f"  Batch Size: {config.BATCH_SIZE}")
    print(f"  Log Level: {config.LOG_LEVEL}")
    
    # Check for required credentials
    missing_creds = []
    if not config.AWS_ACCESS_KEY_ID:
        missing_creds.append("AWS_ACCESS_KEY_ID")
    if not config.AWS_SECRET_ACCESS_KEY:
        missing_creds.append("AWS_SECRET_ACCESS_KEY")
    if config.VECTOR_DB_TYPE == "qdrant" and not config.QDRANT_URL:
        missing_creds.append("QDRANT_URL")
    if config.VECTOR_DB_TYPE == "qdrant" and not config.QDRANT_API_KEY:
        missing_creds.append("QDRANT_API_KEY")
    
    if missing_creds:
        print(f"\n⚠️  Missing required credentials: {', '.join(missing_creds)}")
        print("Please add these to your .env file.")
        
        # Ask if user wants to continue anyway (for testing)
        response = input("\nDo you want to continue anyway? This may fail. (y/N): ")
        if response.lower() != 'y':
            return False
    
    # Set input directory
    input_dir = "RAG_input"
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return False
    
    print(f"\n📁 Input Directory: {input_path.absolute()}")
    
    # Ask for confirmation
    response = input("\nStart processing? (Y/n): ")
    if response.lower() == 'n':
        print("Cancelled by user.")
        return False
    
    # Run the pipeline
    try:
        pipeline = DocumentProcessingPipeline()
        success = pipeline.run_pipeline(str(input_path))
        
        if success:
            print("\nSUCCESS: Pipeline completed successfully!")
            print("\nNext steps:")
            print("  1. Check the logs for detailed processing information")
            print("  2. Use the query module to test searches")
            print("  3. Verify embeddings are stored in your vector database")
        else:
            print("\nERROR: Pipeline failed. Check the logs for details.")
        
        return success
        
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        return False
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    
    print("\n" + "=" * 50)
    if success:
        print("SUCCESS: All done!")
    else:
        print("ERROR: Something went wrong. Check the output above.")
    
    input("\nPress Enter to exit...") 