import os
import json
import logging
import requests
from typing import List, Dict, Any
from bs4 import BeautifulSoup

# Configure logging for production-level observability
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TargetScraper:
    """
    Scraper to extract dialogue/text data from target AI documentation or logs
    to build training datasets for persona adaptation.
    """
    def __init__(self, output_dir: str = "./data/raw_datasets"):
        self.output_dir = os.path.abspath(output_dir)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

    def fetch_url(self, url: str) -> str:
        """Fetches raw text content from a provided URL."""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return ""

    def parse_content(self, html_content: str) -> List[str]:
        """Extracts text content from HTML, filtering for meaningful dialogue."""
        soup = BeautifulSoup(html_content, 'html.parser')
        # Target specific containers common in documentation or chat exports
        text_blocks = [p.get_text().strip() for p in soup.find_all(['p', 'li', 'pre'])]
        return [t for t in text_blocks if len(t) > 20]

    def save_to_jsonl(self, data: List[Dict[str, str]], filename: str):
        """Saves scraped data into a structured JSONL file for training."""
        filepath = os.path.join(self.output_dir, filename)
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                for entry in data:
                    f.write(json.dumps(entry) + '\n')
            logger.info(f"Data successfully saved to: {filepath}")
        except Exception as e:
            logger.error(f"Failed to write JSONL: {e}")

    def run(self, target_urls: List[str], filename: str = "dataset.jsonl"):
        """Orchestrates the scraping and saving process."""
        all_data = []
        for url in target_urls:
            logger.info(f"Scraping: {url}")
            html = self.fetch_url(url)
            if html:
                contents = self.parse_content(html)
                for content in contents:
                    all_data.append({"instruction": "mimic", "input": "", "output": content})
        
        if all_data:
            self.save_to_jsonl(all_data, filename)
        else:
            logger.warning("No data scraped.")

if __name__ == "__main__":
    # Example usage
    scraper = TargetScraper()
    # Replace with actual target documentation URLs
    urls = ["https://example.com/target-ai-docs"]
    scraper.run(urls)