import os
import json
import logging
import re
from typing import List, Dict, Any

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
        """Basic regex-based PII removal for email addresses and IP patterns."""
        # Simple email pattern
        text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[EMAIL_REMOVED]', text)
        # Simple IPv4 pattern
        text = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', '[IP_REMOVED]', text)
        return text

    def clean_text(self, text: str) -> str:
        """Cleans whitespace, normalizes newlines, and truncates overly long lines."""
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def process_and_save(self, input_data: List[Dict[str, str]], output_filename: str):
        """Processes raw entries, validates schema, and saves to JSONL."""
        output_path = os.path.join(self.output_dir, output_filename)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for entry in input_data:
                    # Ensure schema: {"instruction": ..., "input": ..., "output": ...}
                    sanitized_entry = {
                        "instruction": self.clean_text(self.remove_pii(entry.get("instruction", ""))),
                        "input": self.clean_text(self.remove_pii(entry.get("input", ""))),
                        "output": self.clean_text(self.remove_pii(entry.get("output", "")))
                    }
                    
                    if sanitized_entry["instruction"] and sanitized_entry["output"]:
                        f.write(json.dumps(sanitized_entry) + '\n')
            
            logger.info(f"Sanitized data saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving sanitized data: {e}")
            raise

if __name__ == "__main__":
    # Example usage/test
    sanitizer = DataSanitizer()
    sample_data = [
        {"instruction": "What is your purpose?", "input": "", "output": "I am a helpful assistant."},
        {"instruction": "Tell me a secret.", "input": "", "output": "My developer's email is dev@example.com."}
    ]
    sanitizer.process_and_save(sample_data, "sample_clean.jsonl")