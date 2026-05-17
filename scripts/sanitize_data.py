import os
import json
import logging
import re
from typing import List, Dict, Any, Optional

# Configure logging for production-level observability
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataSanitizer:
    """
    Sanitizes raw scraped data for training compatibility.
    Removes PII, standardizes formats, and enforces JSONL schema.
    """
    def __init__(self, output_dir: str = "./data/sanitized_datasets"):
        self.output_dir = os.path.abspath(output_dir)
        try:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create output directory {self.output_dir}: {e}")
            raise

    def remove_pii(self, text: str) -> str:
        """
        Removes PII from text using regex.
        Args:
            text: The raw input string.
        Returns:
            The sanitized string with PII replaced.
        """
        if not isinstance(text, str):
            return ""
        # Simple email pattern
        text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[EMAIL_REMOVED]', text)
        # Simple IPv4 pattern
        text = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', '[IP_REMOVED]', text)
        return text

    def clean_text(self, text: str) -> str:
        """
        Normalizes whitespace and cleans text.
        Args:
            text: The raw input string.
        Returns:
            The normalized string.
        """
        if not isinstance(text, str):
            return ""
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Truncate to a reasonable limit to prevent memory issues with massive lines
        return text[:100000]

    def process_and_save(self, input_data: List[Dict[str, Any]], output_filename: str):
        """
        Validates schema, sanitizes entries, and saves to JSONL.
        
        Args:
            input_data: A list of dicts with 'instruction', 'input', 'output'.
            output_filename: The target filename in the configured output directory.
        
        Raises:
            IOError: If writing to the filesystem fails.
            ValueError: If input_data is not valid.
        """
        if not input_data:
            logger.warning("No data provided to sanitize.")
            return

        output_path = os.path.join(self.output_dir, output_filename)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for idx, entry in enumerate(input_data):
                    if not isinstance(entry, dict):
                        logger.warning(f"Skipping malformed entry at index {idx}: Expected dict, got {type(entry)}")
                        continue

                    # Ensure schema: {"instruction": ..., "input": ..., "output": ...}
                    sanitized_entry = {
                        "instruction": self.clean_text(self.remove_pii(entry.get("instruction", ""))),
                        "input": self.clean_text(self.remove_pii(entry.get("input", ""))),
                        "output": self.clean_text(self.remove_pii(entry.get("output", "")))
                    }
                    
                    # Validation: Instruction and Output are mandatory for training.
                    if sanitized_entry["instruction"] and sanitized_entry["output"]:
                        try:
                            f.write(json.dumps(sanitized_entry) + '\n')
                        except (TypeError, ValueError) as e:
                            logger.error(f"Failed to serialize entry {idx}: {e}")
                    else:
                        logger.warning(f"Skipping incomplete entry at index {idx}: {sanitized_entry}")
            
            logger.info(f"Sanitized data successfully saved to {output_path}")
        except IOError as e:
            logger.error(f"IOError saving sanitized data to {output_path}: {e}")
            raise

if __name__ == "__main__":
    # Example usage/test
    try:
        sanitizer = DataSanitizer()
        sample_data = [
            {"instruction": "What is your purpose?", "input": "", "output": "I am a helpful assistant."},
            {"instruction": "Tell me a secret.", "input": "", "output": "My developer's email is dev@example.com."},
            {"instruction": "", "output": "Missing instruction"}, # Should be skipped
            None # Should be skipped
        ]
        sanitizer.process_and_save(sample_data, "sample_clean.jsonl")
    except Exception as e:
        logger.critical(f"DataSanitizer failed: {e}")
