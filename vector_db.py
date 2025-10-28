"""
Vector database abstraction layer supporting multiple vector databases.
Provides a unified interface for Qdrant, ChromaDB, and Pinecone.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass

from config import config

logger = logging.getLogger("doc_processor.vector_db")

@dataclass
class VectorPoint:
    """Represents a vector point with metadata."""
    id: str
    vector: List[float]
    metadata: Dict[str, Any]

@dataclass
class SearchResult:
    """Represents a search result."""
    id: str
    score: float
    metadata: Dict[str, Any]
    payload: Optional[Dict[str, Any]] = None

class VectorDatabase(ABC):
    """Abstract base class for vector databases."""
    
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.logger = logging.getLogger(f"doc_processor.vector_db.{self.__class__.__name__}")
    
    @abstractmethod
    def create_collection(self, dimension: int, clear_existing: bool = False) -> bool:
        """Create a collection with specified dimension."""
        pass
    
    @abstractmethod
    def collection_exists(self) -> bool:
        """Check if collection exists."""
        pass
    
    @abstractmethod
    def upsert_points(self, points: List[VectorPoint]) -> bool:
        """Insert or update points in the collection."""
        pass
    
    @abstractmethod
    def search(self, query_vector: List[float], limit: int = 10, 
               score_threshold: Optional[float] = None) -> List[SearchResult]:
        """Search for similar vectors."""
        pass
    
    @abstractmethod
    def delete_collection(self) -> bool:
        """Delete the collection."""
        pass
    
    @abstractmethod
    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection information."""
        pass
    
    @abstractmethod
    def close(self):
        """Close the database connection."""
        pass

class QdrantDatabase(VectorDatabase):
    """Qdrant vector database implementation."""
    
    def __init__(self, collection_name: str):
        super().__init__(collection_name)
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams, PointStruct
            from qdrant_client.http import models
            
            self.QdrantClient = QdrantClient
            self.Distance = Distance
            self.VectorParams = VectorParams
            self.PointStruct = PointStruct
            self.models = models
            
            db_config = config.get_vector_db_config()
            self.client = QdrantClient(
                url=db_config["url"],
                api_key=db_config["api_key"],
                timeout=db_config.get("timeout", 300)
            )
            self.logger.info(f"Connected to Qdrant at {db_config['url']}")
            
        except ImportError:
            raise ImportError("qdrant-client is required for Qdrant support. Install with: pip install qdrant-client")
    
    def create_collection(self, dimension: int, clear_existing: bool = False) -> bool:
        """Create a Qdrant collection."""
        try:
            if clear_existing and self.collection_exists():
                self.delete_collection()
                self.logger.info(f"Deleted existing collection: {self.collection_name}")
            
            if not self.collection_exists():
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=self.VectorParams(
                        size=dimension,
                        distance=self.Distance.COSINE
                    )
                )
                self.logger.info(f"Created Qdrant collection: {self.collection_name} with dimension {dimension}")
                return True
            else:
                self.logger.info(f"Collection {self.collection_name} already exists")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to create collection: {e}")
            return False
    
    def collection_exists(self) -> bool:
        """Check if Qdrant collection exists."""
        try:
            collections = self.client.get_collections()
            return any(col.name == self.collection_name for col in collections.collections)
        except Exception as e:
            self.logger.error(f"Failed to check collection existence: {e}")
            return False
    
    def upsert_points(self, points: List[VectorPoint]) -> bool:
        """Upsert points to Qdrant."""
        try:
            qdrant_points = [
                self.PointStruct(
                    id=point.id,
                    vector=point.vector,
                    payload=point.metadata
                )
                for point in points
            ]
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=qdrant_points
            )
            self.logger.debug(f"Upserted {len(points)} points to Qdrant")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to upsert points: {e}")
            return False
    
    def search(self, query_vector: List[float], limit: int = 10, 
               score_threshold: Optional[float] = None) -> List[SearchResult]:
        """Search Qdrant for similar vectors."""
        try:
            search_params = {
                "collection_name": self.collection_name,
                "query_vector": query_vector,
                "limit": limit,
                "with_payload": True
            }
            
            if score_threshold is not None:
                search_params["score_threshold"] = score_threshold
            
            results = self.client.search(**search_params)
            
            return [
                SearchResult(
                    id=str(result.id),
                    score=result.score,
                    metadata=result.payload or {},
                    payload=result.payload
                )
                for result in results
            ]
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []
    
    def delete_collection(self) -> bool:
        """Delete Qdrant collection."""
        try:
            self.client.delete_collection(self.collection_name)
            self.logger.info(f"Deleted collection: {self.collection_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete collection: {e}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get Qdrant collection information."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "points_count": info.points_count,
                "segments_count": info.segments_count,
                "status": info.status
            }
        except Exception as e:
            self.logger.error(f"Failed to get collection info: {e}")
            return {}
    
    def close(self):
        """Close Qdrant connection."""
        if hasattr(self, 'client'):
            self.client.close()
            self.logger.info("Closed Qdrant connection")

class ChromaDatabase(VectorDatabase):
    """ChromaDB vector database implementation."""
    
    def __init__(self, collection_name: str):
        super().__init__(collection_name)
        try:
            import chromadb
            from chromadb.config import Settings
            
            db_config = config.get_vector_db_config()
            
            if db_config.get("persist_directory"):
                # Persistent client
                self.client = chromadb.PersistentClient(
                    path=db_config["persist_directory"]
                )
            else:
                # HTTP client
                self.client = chromadb.HttpClient(
                    host=db_config["host"],
                    port=db_config["port"]
                )
            
            self.collection = None
            self.logger.info(f"Connected to ChromaDB")
            
        except ImportError:
            raise ImportError("chromadb is required for ChromaDB support. Install with: pip install chromadb")
    
    def create_collection(self, dimension: int, clear_existing: bool = False) -> bool:
        """Create a ChromaDB collection."""
        try:
            if clear_existing and self.collection_exists():
                self.delete_collection()
            
            if not self.collection_exists():
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"dimension": dimension}
                )
                self.logger.info(f"Created ChromaDB collection: {self.collection_name}")
            else:
                self.collection = self.client.get_collection(self.collection_name)
                self.logger.info(f"Using existing ChromaDB collection: {self.collection_name}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create collection: {e}")
            return False
    
    def collection_exists(self) -> bool:
        """Check if ChromaDB collection exists."""
        try:
            collections = self.client.list_collections()
            return any(col.name == self.collection_name for col in collections)
        except Exception as e:
            self.logger.error(f"Failed to check collection existence: {e}")
            return False
    
    def upsert_points(self, points: List[VectorPoint]) -> bool:
        """Upsert points to ChromaDB."""
        try:
            if not self.collection:
                self.collection = self.client.get_collection(self.collection_name)
            
            ids = [point.id for point in points]
            embeddings = [point.vector for point in points]
            metadatas = [point.metadata for point in points]
            
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            self.logger.debug(f"Upserted {len(points)} points to ChromaDB")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to upsert points: {e}")
            return False
    
    def search(self, query_vector: List[float], limit: int = 10, 
               score_threshold: Optional[float] = None) -> List[SearchResult]:
        """Search ChromaDB for similar vectors."""
        try:
            if not self.collection:
                self.collection = self.client.get_collection(self.collection_name)
            
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=limit
            )
            
            search_results = []
            if results['ids'] and results['ids'][0]:
                for i, doc_id in enumerate(results['ids'][0]):
                    # ChromaDB returns distances, convert to similarity scores
                    distance = results['distances'][0][i] if results['distances'] else 0
                    score = 1 / (1 + distance)  # Convert distance to similarity
                    
                    if score_threshold is None or score >= score_threshold:
                        metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                        search_results.append(
                            SearchResult(
                                id=doc_id,
                                score=score,
                                metadata=metadata
                            )
                        )
            
            return search_results
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []
    
    def delete_collection(self) -> bool:
        """Delete ChromaDB collection."""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = None
            self.logger.info(f"Deleted collection: {self.collection_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete collection: {e}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get ChromaDB collection information."""
        try:
            if not self.collection:
                self.collection = self.client.get_collection(self.collection_name)
            
            count = self.collection.count()
            return {
                "name": self.collection_name,
                "points_count": count,
                "status": "active" if count >= 0 else "unknown"
            }
        except Exception as e:
            self.logger.error(f"Failed to get collection info: {e}")
            return {}
    
    def close(self):
        """Close ChromaDB connection."""
        self.collection = None
        self.logger.info("Closed ChromaDB connection")

class PineconeDatabase(VectorDatabase):
    """Pinecone vector database implementation."""
    
    def __init__(self, collection_name: str):
        super().__init__(collection_name)
        try:
            import pinecone
            
            db_config = config.get_vector_db_config()
            
            pinecone.init(
                api_key=db_config["api_key"],
                environment=db_config["environment"]
            )
            
            self.index_name = db_config["index_name"]
            self.index = None
            self.logger.info(f"Connected to Pinecone environment: {db_config['environment']}")
            
        except ImportError:
            raise ImportError("pinecone-client is required for Pinecone support. Install with: pip install pinecone-client")
    
    def create_collection(self, dimension: int, clear_existing: bool = False) -> bool:
        """Create a Pinecone index (collection)."""
        try:
            import pinecone
            
            if clear_existing and self.collection_exists():
                self.delete_collection()
            
            if not self.collection_exists():
                pinecone.create_index(
                    name=self.index_name,
                    dimension=dimension,
                    metric="cosine"
                )
                self.logger.info(f"Created Pinecone index: {self.index_name}")
            
            self.index = pinecone.Index(self.index_name)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create index: {e}")
            return False
    
    def collection_exists(self) -> bool:
        """Check if Pinecone index exists."""
        try:
            import pinecone
            return self.index_name in pinecone.list_indexes()
        except Exception as e:
            self.logger.error(f"Failed to check index existence: {e}")
            return False
    
    def upsert_points(self, points: List[VectorPoint]) -> bool:
        """Upsert points to Pinecone."""
        try:
            if not self.index:
                import pinecone
                self.index = pinecone.Index(self.index_name)
            
            vectors = [
                (point.id, point.vector, point.metadata)
                for point in points
            ]
            
            self.index.upsert(vectors=vectors, namespace=self.collection_name)
            self.logger.debug(f"Upserted {len(points)} points to Pinecone")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to upsert points: {e}")
            return False
    
    def search(self, query_vector: List[float], limit: int = 10, 
               score_threshold: Optional[float] = None) -> List[SearchResult]:
        """Search Pinecone for similar vectors."""
        try:
            if not self.index:
                import pinecone
                self.index = pinecone.Index(self.index_name)
            
            search_params = {
                "vector": query_vector,
                "top_k": limit,
                "include_metadata": True,
                "namespace": self.collection_name
            }
            
            if score_threshold is not None:
                search_params["filter"] = {"score": {"$gte": score_threshold}}
            
            results = self.index.query(**search_params)
            
            return [
                SearchResult(
                    id=match.id,
                    score=match.score,
                    metadata=match.metadata or {}
                )
                for match in results.matches
                if score_threshold is None or match.score >= score_threshold
            ]
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []
    
    def delete_collection(self) -> bool:
        """Delete Pinecone index."""
        try:
            import pinecone
            pinecone.delete_index(self.index_name)
            self.index = None
            self.logger.info(f"Deleted index: {self.index_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete index: {e}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get Pinecone index information."""
        try:
            if not self.index:
                import pinecone
                self.index = pinecone.Index(self.index_name)
            
            stats = self.index.describe_index_stats()
            namespace_stats = stats.namespaces.get(self.collection_name, {})
            
            return {
                "name": self.collection_name,
                "index_name": self.index_name,
                "points_count": namespace_stats.get("vector_count", 0),
                "dimension": stats.dimension,
                "status": "active"
            }
        except Exception as e:
            self.logger.error(f"Failed to get index info: {e}")
            return {}
    
    def close(self):
        """Close Pinecone connection."""
        self.index = None
        self.logger.info("Closed Pinecone connection")

def create_vector_database(collection_name: str) -> VectorDatabase:
    """Factory function to create the appropriate vector database instance."""
    db_type = config.VECTOR_DB_TYPE.lower()
    
    if db_type == "qdrant":
        return QdrantDatabase(collection_name)
    elif db_type == "chromadb":
        return ChromaDatabase(collection_name)
    elif db_type == "pinecone":
        return PineconeDatabase(collection_name)
    else:
        raise ValueError(f"Unsupported vector database type: {db_type}")

# Convenience function for getting a database instance
def get_vector_db(collection_name: str = None) -> VectorDatabase:
    """Get a vector database instance with the configured collection name."""
    if collection_name is None:
        collection_name = config.COLLECTION_NAME
    return create_vector_database(collection_name) 