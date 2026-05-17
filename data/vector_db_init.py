import os
import logging
import re
from typing import Optional
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
    except OSError as e:
        logger.error(f"OS error initializing vector DB at {abs_path}: {e}")
        raise

def get_or_create_collection(client: chromadb.PersistentClient, collection_name: str = "persona_knowledge"):
    """
    Retrieves an existing collection or creates a new one if it doesn't exist.
    
    Args:
        client: The initialized ChromaDB client.
        collection_name: Unique identifier for the collection (alphanumeric, underscores, hyphens, max 63 chars).
    
    Returns:
        chromadb.Collection instance.
    """
    if not isinstance(collection_name, str) or not re.match(r'^[a-zA-Z0-9_-]{1,63}$', collection_name):
        raise ValueError("collection_name must be alphanumeric/underscores/hyphens and 1-63 chars.")
        
    try:
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
    # Ensure working directory is absolute for robust file handling
    try:
        client = initialize_vector_db()
        collection = get_or_create_collection(client)
    except (OSError, ChromaError, ValueError) as e:
        logger.critical(f"Critical failure initializing vector store: {e}")
        exit(1)