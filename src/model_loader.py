import os
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self, model_id: str = "meta-llama/Meta-Llama-3-8B", device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model_id = model_id
        self.device = device
        self.model = None
        self.tokenizer = None

    def load_model(self, quantization: bool = True) -> Dict[str, Any]:
        """
        Loads the base model and tokenizer with optional 4-bit quantization.
        
        Args:
            quantization: Whether to apply 4-bit quantization.
            
        Returns:
            Dict containing the loaded model and tokenizer.
        """
        try:
            logger.info(f"Loading model: {self.model_id} on {self.device}")
            
            bnb_config = None
            if quantization and self.device == "cuda":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16
                )
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=bnb_config,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            logger.info("Model and tokenizer loaded successfully.")
            return {"model": self.model, "tokenizer": self.tokenizer}
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_id}: {e}")
            raise

    def get_model_info(self) -> Dict[str, str]:
        """Returns metadata about the currently loaded model."""
        return {
            "model_id": self.model_id,
            "device": self.device,
            "is_loaded": self.model is not None
        }