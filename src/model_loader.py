import os
import logging
from typing import Optional, Dict, Any
from huggingface_hub import InferenceClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self, model_id: str = "meta-llama/Meta-Llama-3-8B"):
        self.model_id = model_id
        # Expects HF_TOKEN environment variable
        self.client = InferenceClient(model=model_id)

    def load_model(self, quantization: bool = True) -> Dict[str, Any]:
        """
        Uses Hugging Face Inference API instead of local loading.
        """
        logger.info(f"Connecting to HF Inference API for: {self.model_id}")
        return {"model": self.client, "tokenizer": None}

    def get_model_info(self) -> Dict[str, str]:
        return {
            "model_id": self.model_id,
            "backend": "Hugging Face Inference API",
            "is_loaded": True
        }
