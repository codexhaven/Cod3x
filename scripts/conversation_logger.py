
import os
import json
import logging
import typing
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ConversationLogger")

# Constants
DATA_DIR = Path(os.path.abspath("./data/logs"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

def log_interaction(query: str, response: str) -> None:
    """
    Logs an interaction to a daily JSONL file.
    
    Args:
        query: User input string.
        response: System output string.
        
    Raises:
        OSError: If file writing fails.
        ValueError: If inputs are invalid.
    """
    # 1. Edge Case Validation
    if not isinstance(query, str) or not isinstance(response, str):
        raise ValueError("Query and response must be strings.")
    
    clean_query = query.strip()
    clean_response = response.strip()
    
    if not clean_query and not clean_response:
        logger.warning("Attempted to log empty interaction.")
        return

    try:
        timestamp = datetime.now().strftime("%Y-%m-%d")
        file_path = DATA_DIR / f"interactions_{timestamp}.jsonl"
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "query": clean_query,
            "response": clean_response
        }
        
        # 2. Performance: Buffered append
        with file_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
        logger.info(f"Interaction logged successfully to {file_path}")
        
    except (OSError, IOError) as e:
        logger.error(f"IOError: Failed to write to {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during logging: {e}")
        raise

if __name__ == "__main__":
    # Smoke test
    try:
        log_interaction("Hello", "How can I help?")
        logger.info("Self-test passed.")
    except Exception as e:
        logger.critical(f"Self-test failed: {e}")