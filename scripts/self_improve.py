
import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SelfImprovementProcessor")

# Directory Constants
RAW_LOGS_DIR = Path(os.path.abspath("./data/logs"))
PROCESSED_DATA_DIR = Path(os.path.abspath("./data/training"))

def is_quality_entry(entry: Dict[str, Any]) -> bool:
    """
    Validate quality of a single entry.
    
    Args:
        entry: The JSON object to validate.
        
    Returns:
        bool: True if entry passes all quality gates.
    """
    if not isinstance(entry, dict):
        return False
        
    query = entry.get("query")
    response = entry.get("response")
    
    if not isinstance(query, str) or not isinstance(response, str):
        return False
        
    # Quality Gate: non-empty query/response, length > 10 for response
    return len(query.strip()) > 0 and len(response.strip()) > 10

def filter_and_merge() -> None:
    """
    Reads all raw JSONL logs, filters for quality, and merges into 
    a single fine-tuning dataset using stream-processing for memory efficiency.
    
    Quality Filter:
        - Response length must be > 10 characters.
        - Must contain valid 'query' and 'response' strings.
        
    Performance:
        - Uses streaming read/write to avoid O(n) memory allocation.
        - Processes logs in place.
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / "fine_tune_data.jsonl"
    
    if not RAW_LOGS_DIR.exists():
        logger.error(f"Raw log directory not found: {RAW_LOGS_DIR}")
        return

    log_files = list(RAW_LOGS_DIR.glob("interactions_*.jsonl"))
    
    if not log_files:
        logger.warning("No log files found in data directory.")
        return

    count = 0
    try:
        # Stream processing: open for write, read files sequentially
        with open(output_path, "w", encoding="utf-8") as outfile:
            for file_path in log_files:
                logger.info(f"Processing: {file_path.name}")
                with open(file_path, "r", encoding="utf-8") as infile:
                    for line_no, line in enumerate(infile, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                            
                            if is_quality_entry(entry):
                                outfile.write(json.dumps(entry) + "\n")
                                count += 1
                            else:
                                logger.debug(f"Skipping low quality entry at {file_path.name}:{line_no}")
                                
                        except json.JSONDecodeError:
                            logger.error(f"Invalid JSON in {file_path.name} at line {line_no}")
                            continue
                            
        logger.info(f"Successfully processed {count} entries into {output_path}")
    except OSError as e:
        logger.error(f"Failed to perform I/O operation: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during filtration: {e}")

if __name__ == "__main__":
    filter_and_merge()
