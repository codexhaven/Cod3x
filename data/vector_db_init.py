
import os
import logging
import re
from typing import Optional, Any
import chromadb
from chromadb.config import Settings
from chromadb.errors import ChromaError

# Configure logging for production-level observability
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def initialize_vector_db(persist_directory: Optional[str] = None) -> chromadb.PersistentClient:
    """
    Initializes the ChromaDB vector database instance for persona storage.
    
    Args:
        persist_directory: Absolute or relative path to the storage location.
                           Defaults to environment variable PERSIST_DIR or ./data/chroma_db.
    
    Returns:
        chromadb.PersistentClient instance.
    
    Raises:
        OSError: If the directory cannot be created or accessed.
    """
    target_dir = persist_directory or os.getenv("PERSIST_DIR", "./data/chroma_db")
    abs_path = os.path.abspath(target_dir)
    parent_dir = os.path.dirname(abs_path)

    if not os.path.exists(parent_dir):
        try:
            os.makedirs(parent_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create parent directory {parent_dir}: {e}")
            raise

    if not os.access(parent_dir, os.W_OK):
        raise OSError(f"No write permission on parent directory: {parent_dir}")

    try:
        if not os.path.exists(abs_path):
            logger.info(f"Creating persistence directory at: {abs_path}")
            os.makedirs(abs_path, exist_ok=True)
        
        client = chromadb.PersistentClient(
            path=abs_path,
            settings=Settings(anonymized_telemetry=False)
        )
        logger.info(f"Vector DB client initialized at: {abs_path}")
        return client
    except Exception as e:
        logger.error(f"Unexpected error initializing vector DB at {abs_path}: {e}")
        raise

def get_or_create_collection(client: chromadb.PersistentClient, collection_name: str = "persona_knowledge") -> chromadb.Collection:
    """
    Retrieves an existing collection or creates a new one if it doesn't exist.
    
    Args:
        client: The initialized ChromaDB client.
        collection_name: Unique identifier for the collection (alphanumeric, underscores, hyphens, max 63 chars).
    
    Returns:
        chromadb.Collection instance.
    
    Raises:
        ValueError: If collection_name is invalid.
        ChromaError: If ChromaDB fails to access/create the collection.
    """
    if not isinstance(collection_name, str) or not re.match(r'^[a-zA-Z0-9_-]{1,63}$', collection_name):
        raise ValueError("collection_name must be alphanumeric/underscores/hyphens and 1-63 chars.")
        
    try:
        # Use get_or_create_collection for atomicity
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Collection '{collection_name}' is ready.")
        return collection
    except ChromaError as e:
        logger.error(f"ChromaDB error accessing collection '{collection_name}': {e}")
        raise

if __name__ == "__main__":
    # Test suite for initialization
    try:
        # Verify db initialization
        client = initialize_vector_db()
        # Verify collection access
        collection = get_or_create_collection(client)
        # Basic smoke test: query count
        count = collection.count()
        logger.info(f"Vector store initialized successfully. Current item count: {count}")
    except (OSError, ChromaError, ValueError) as e:
        logger.critical(f"Critical failure initializing vector store: {e}")
        exit(1)