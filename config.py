"""
Configuration management for the document processor system.
Handles environment variables, validation, and configuration access.
"""

import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

# Try to load python-dotenv if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not available, continue without it
    pass

logger = logging.getLogger("doc_processor.config")

@dataclass
class Config:
    """Configuration class for the document processor."""
    
    # AWS Configuration
    AWS_REGION: str = "us-east-1"
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    
    # OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = None
    
    # Vector Database Configuration
    VECTOR_DB_TYPE: str = "qdrant"  # qdrant, chromadb, pinecone
    
    # Qdrant Configuration
    QDRANT_URL: Optional[str] = None
    QDRANT_API_KEY: Optional[str] = None
    
    # ChromaDB Configuration
    CHROMADB_HOST: str = "localhost"
    CHROMADB_PORT: int = 8000
    CHROMADB_PERSIST_DIRECTORY: str = "./chroma_db"
    
    # Pinecone Configuration
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_ENVIRONMENT: Optional[str] = None
    PINECONE_INDEX_NAME: Optional[str] = None
    
    # Database Configuration
    SQLITE_DB_PATH: str = "chunked_unique.db"
    COLLECTION_NAME: str = "mro_documents"
    
    # Processing Configuration
    BATCH_SIZE: int = 300
    TEST_MODE: bool = False
    CLEAR_COLLECTION: bool = False
    
    # Embedding Configuration
    EMBEDDING_MODEL: str = "amazon.titan-embed-g1-text-02"
    EMBEDDING_DIMENSION: int = 1536
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "doc_processor.log"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Query Configuration
    DEFAULT_TEMPERATURE: float = 0.6
    DEFAULT_THRESHOLD: float = 0.25
    DEFAULT_MAX_RESULTS: int = 20
    DEFAULT_LIMIT: int = 100
    
    # File Processing Configuration
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    MIN_FILE_SIZE: int = 1024  # 1KB
    ALLOWED_EXTENSIONS: str = ".pdf,.docx,.xlsx,.pptx"
    
    def __post_init__(self):
        """Post-initialization to load environment variables."""
        self._load_from_env()
        self._setup_logging()
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        # AWS Configuration
        self.AWS_REGION = os.getenv("AWS_REGION", self.AWS_REGION)
        self.AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
        self.AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
        
        # OpenAI Configuration
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        
        # Vector Database Configuration
        self.VECTOR_DB_TYPE = os.getenv("VECTOR_DB_TYPE", self.VECTOR_DB_TYPE).lower()
        
        # Qdrant Configuration
        self.QDRANT_URL = os.getenv("QDRANT_URL")
        self.QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
        
        # ChromaDB Configuration
        self.CHROMADB_HOST = os.getenv("CHROMADB_HOST", self.CHROMADB_HOST)
        self.CHROMADB_PORT = int(os.getenv("CHROMADB_PORT", str(self.CHROMADB_PORT)))
        self.CHROMADB_PERSIST_DIRECTORY = os.getenv("CHROMADB_PERSIST_DIRECTORY", self.CHROMADB_PERSIST_DIRECTORY)
        
        # Pinecone Configuration
        self.PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        self.PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
        self.PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
        
        # Database Configuration
        self.SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", self.SQLITE_DB_PATH)
        self.COLLECTION_NAME = os.getenv("COLLECTION_NAME", self.COLLECTION_NAME)
        
        # Processing Configuration
        self.BATCH_SIZE = int(os.getenv("BATCH_SIZE", str(self.BATCH_SIZE)))
        self.TEST_MODE = os.getenv("TEST_MODE", "false").lower() in ("true", "1", "yes")
        self.CLEAR_COLLECTION = os.getenv("CLEAR_COLLECTION", "false").lower() in ("true", "1", "yes")
        
        # Embedding Configuration
        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", self.EMBEDDING_MODEL)
        self.EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", str(self.EMBEDDING_DIMENSION)))
        
        # Logging Configuration
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", self.LOG_LEVEL).upper()
        self.LOG_FILE = os.getenv("LOG_FILE", self.LOG_FILE)
        self.LOG_FORMAT = os.getenv("LOG_FORMAT", self.LOG_FORMAT)
        
        # Query Configuration
        self.DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", str(self.DEFAULT_TEMPERATURE)))
        self.DEFAULT_THRESHOLD = float(os.getenv("DEFAULT_THRESHOLD", str(self.DEFAULT_THRESHOLD)))
        self.DEFAULT_MAX_RESULTS = int(os.getenv("DEFAULT_MAX_RESULTS", str(self.DEFAULT_MAX_RESULTS)))
        self.DEFAULT_LIMIT = int(os.getenv("DEFAULT_LIMIT", str(self.DEFAULT_LIMIT)))
        
        # File Processing Configuration
        self.MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", str(self.MAX_FILE_SIZE)))
        self.MIN_FILE_SIZE = int(os.getenv("MIN_FILE_SIZE", str(self.MIN_FILE_SIZE)))
        self.ALLOWED_EXTENSIONS = os.getenv("ALLOWED_EXTENSIONS", self.ALLOWED_EXTENSIONS)
    
    def _setup_logging(self):
        """Setup logging configuration."""
        # Convert string log level to logging constant
        log_level = getattr(logging, self.LOG_LEVEL, logging.INFO)
        
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format=self.LOG_FORMAT,
            handlers=[
                logging.StreamHandler(),  # Console output
                logging.FileHandler(self.LOG_FILE)  # File output
            ]
        )
        
        logger.info(f"Logging configured: level={self.LOG_LEVEL}, file={self.LOG_FILE}")
    
    def validate(self) -> bool:
        """
        Validate the configuration.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        errors = []
        
        # Check vector database type
        if self.VECTOR_DB_TYPE not in ["qdrant", "chromadb", "pinecone"]:
            errors.append(f"Invalid VECTOR_DB_TYPE: {self.VECTOR_DB_TYPE}")
        
        # Check vector database specific configuration
        if self.VECTOR_DB_TYPE == "qdrant":
            if not self.QDRANT_URL:
                errors.append("QDRANT_URL is required for Qdrant")
        elif self.VECTOR_DB_TYPE == "pinecone":
            if not self.PINECONE_API_KEY:
                errors.append("PINECONE_API_KEY is required for Pinecone")
            if not self.PINECONE_ENVIRONMENT:
                errors.append("PINECONE_ENVIRONMENT is required for Pinecone")
            if not self.PINECONE_INDEX_NAME:
                errors.append("PINECONE_INDEX_NAME is required for Pinecone")
        
        # Check batch size
        if self.BATCH_SIZE <= 0:
            errors.append(f"BATCH_SIZE must be positive: {self.BATCH_SIZE}")
        
        # Check embedding dimension
        if self.EMBEDDING_DIMENSION <= 0:
            errors.append(f"EMBEDDING_DIMENSION must be positive: {self.EMBEDDING_DIMENSION}")
        
        # Check file size limits
        if self.MIN_FILE_SIZE >= self.MAX_FILE_SIZE:
            errors.append(f"MIN_FILE_SIZE ({self.MIN_FILE_SIZE}) must be less than MAX_FILE_SIZE ({self.MAX_FILE_SIZE})")
        
        if errors:
            logger.error("Configuration validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
            return False
        
        logger.info("Configuration validation passed")
        return True
    
    def get_vector_db_config(self) -> Dict[str, Any]:
        """Get vector database specific configuration."""
        if self.VECTOR_DB_TYPE == "qdrant":
            return {
                "url": self.QDRANT_URL,
                "api_key": self.QDRANT_API_KEY
            }
        elif self.VECTOR_DB_TYPE == "chromadb":
            return {
                "host": self.CHROMADB_HOST,
                "port": self.CHROMADB_PORT,
                "persist_directory": self.CHROMADB_PERSIST_DIRECTORY
            }
        elif self.VECTOR_DB_TYPE == "pinecone":
            return {
                "api_key": self.PINECONE_API_KEY,
                "environment": self.PINECONE_ENVIRONMENT,
                "index_name": self.PINECONE_INDEX_NAME
            }
        else:
            return {}
    
    def get_aws_config(self) -> Dict[str, Any]:
        """Get AWS configuration."""
        return {
            "region_name": self.AWS_REGION,
            "aws_access_key_id": self.AWS_ACCESS_KEY_ID,
            "aws_secret_access_key": self.AWS_SECRET_ACCESS_KEY
        }
    
    def get_openai_config(self) -> Dict[str, Any]:
        """Get OpenAI configuration."""
        return {
            "api_key": self.OPENAI_API_KEY
        }
    
    def __str__(self) -> str:
        """String representation of configuration (without sensitive data)."""
        safe_config = {
            "VECTOR_DB_TYPE": self.VECTOR_DB_TYPE,
            "COLLECTION_NAME": self.COLLECTION_NAME,
            "BATCH_SIZE": self.BATCH_SIZE,
            "TEST_MODE": self.TEST_MODE,
            "LOG_LEVEL": self.LOG_LEVEL,
            "EMBEDDING_MODEL": self.EMBEDDING_MODEL,
            "EMBEDDING_DIMENSION": self.EMBEDDING_DIMENSION
        }
        return f"Config({safe_config})"

# Global configuration instance
config = Config()

# Validate configuration on import
if not config.validate():
    logger.warning("Configuration validation failed - some features may not work correctly")

def get_config() -> Config:
    """Get the global configuration instance."""
    return config

if __name__ == "__main__":
    """Print configuration when run directly."""
    print("Document Processor Configuration:")
    print("=" * 40)
    print(config)
    print("\nVector DB Config:", config.get_vector_db_config())
    print("AWS Config:", {k: v for k, v in config.get_aws_config().items() if k != "secret_access_key"})
    print("OpenAI Config:", {k: "***" if v else None for k, v in config.get_openai_config().items()}) 