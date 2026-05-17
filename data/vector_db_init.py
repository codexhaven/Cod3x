import os
import chromadb
from chromadb.config import Settings

def initialize_vector_db(persist_directory: str = "./data/chroma_db"):
    """
    Initializes the ChromaDB vector database instance for persona storage.
    Ensures the persistence directory exists.
    """
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory, exist_ok=True)
    
    client = chromadb.PersistentClient(
        path=persist_directory,
        settings=Settings(anonymized_telemetry=False)
    )
    
    return client

def get_or_create_collection(client, collection_name: str = "persona_knowledge"):
    """
    Retrieves an existing collection or creates a new one if it doesn't exist.
    """
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    return collection

if __name__ == "__main__":
    db_path = os.path.abspath("./data/chroma_db")
    client = initialize_vector_db(db_path)
    collection = get_or_create_collection(client)
    
    print(f"Vector DB initialized at: {db_path}")
    print(f"Collection '{collection.name}' is ready.")