import os
import logging
from typing import Optional, Dict, Any
from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self, model_id: Optional[str] = None):
        self.model_id = model_id or os.getenv("DEFAULT_MODEL_ID", "meta-llama/Meta-Llama-3-8B")
        
        token = os.getenv("HF_TOKEN")
        if not token:
            raise EnvironmentError("HF_TOKEN environment variable is not set. Please provide a valid Hugging Face token.")
            
        try:
            self.client = InferenceClient(model=self.model_id, token=token)
        except Exception as e:
            logger.error(f"Failed to initialize InferenceClient: {e}")
            raise

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