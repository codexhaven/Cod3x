
import os
import logging
from typing import Optional
import chromadb
from chromadb.config import Settings

# Configure logging for production-level observability
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def initialize_vector_db(persist_directory: str = "./data/chroma_db") -> chromadb.PersistentClient:
    """
    Initializes the ChromaDB vector database instance for persona storage.
    
    Args:
        persist_directory: Absolute or relative path to the storage location.
    
    Returns:
        chromadb.PersistentClient instance.
    
    Raises:
        OSError: If the directory cannot be created or accessed.
    """
    try:
        abs_path = os.path.abspath(persist_directory)
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
        logger.error(f"Failed to initialize vector DB at {persist_directory}: {e}")
        raise

def get_or_create_collection(client: chromadb.PersistentClient, collection_name: str = "persona_knowledge"):
    """
    Retrieves an existing collection or creates a new one if it doesn't exist.
    
    Args:
        client: The initialized ChromaDB client.
        collection_name: Unique identifier for the collection.
    
    Returns:
        chromadb.Collection instance.
    """
    if not collection_name or not isinstance(collection_name, str):
        raise ValueError("collection_name must be a non-empty string.")
        
    try:
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Collection '{collection_name}' is ready.")
        return collection
    except Exception as e:
        logger.error(f"Error accessing collection '{collection_name}': {e}")
        raise

if __name__ == "__main__":
    # Ensure working directory is absolute for robust file handling
    db_path = os.path.abspath("./data/chroma_db")
    try:
        client = initialize_vector_db(db_path)
        collection = get_or_create_collection(client)
    except Exception as e:
        logger.critical(f"Critical failure initializing vector store: {e}")
        exit(1)