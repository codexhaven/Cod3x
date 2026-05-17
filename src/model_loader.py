
import os
import logging
from typing import Optional, Dict, Any
from huggingface_hub import InferenceClient

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelLoader:
    """
    Handles model initialization and inference client management.
    Uses Hugging Face Inference API.
    """

    def __init__(self, model_id: Optional[str] = None):
        """
        Initializes the model loader.
        
        Args:
            model_id: The ID of the model to load. Falls back to DEFAULT_MODEL_ID env var.
        
        Raises:
            EnvironmentError: If HF_TOKEN is not set.
        """
        self.model_id = model_id or os.getenv("DEFAULT_MODEL_ID", "meta-llama/Meta-Llama-3-8B")
        
        token = os.getenv("HF_TOKEN")
        if not token:
            error_msg = "HF_TOKEN environment variable is not set. Please provide a valid Hugging Face token."
            logger.error(error_msg)
            raise EnvironmentError(error_msg)
            
        try:
            self.client = InferenceClient(model=self.model_id, token=token)
        except Exception as e:
            logger.error(f"Failed to initialize InferenceClient for {self.model_id}: {e}")
            raise

    def load_model(self, quantization: bool = True) -> Dict[str, Any]:
        """
        Connects to the HF Inference API.
        
        Args:
            quantization: Note: InferenceClient handles server-side model loading.
                          This argument is kept for compatibility but has no effect
                          on the remote client configuration.
                          
        Returns:
            Dict containing the model client and placeholder for tokenizer.
        """
        logger.info(f"Connecting to HF Inference API for: {self.model_id}")
        return {"model": self.client, "tokenizer": None}

    def get_model_info(self) -> Dict[str, str]:
        """Returns metadata about the loaded model."""
        return {
            "model_id": self.model_id,
            "backend": "Hugging Face Inference API",
            "is_loaded": "True" if self.client else "False"
        }