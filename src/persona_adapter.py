
import os
import logging
import torch
from peft import PeftModel, LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
from typing import Optional, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PersonaAdapter:
    """
    Manages the loading, merging, and application of LoRA adapters
    to adapt a base model to a specific persona.
    """
    def __init__(self, base_model: Any):
        self.base_model = base_model
        self.active_adapter: Optional[str] = None

    def apply_adapter(self, adapter_path: str, adapter_name: str) -> bool:
        """
        Loads a pre-trained LoRA adapter into the base model.
        
        Args:
            adapter_path: Path to the directory containing the adapter files.
            adapter_name: Unique identifier for the adapter.
            
        Returns:
            bool: True if application successful, False otherwise.
        """
        try:
            if not os.path.exists(adapter_path):
                logger.error(f"Adapter path does not exist: {adapter_path}")
                return False
            
            logger.info(f"Applying adapter '{adapter_name}' from {adapter_path}")
            
            # Apply the adapter
            self.base_model = PeftModel.from_pretrained(
                self.base_model,
                adapter_path,
                adapter_name=adapter_name
            )
            
            self.active_adapter = adapter_name
            logger.info(f"Successfully activated adapter: {adapter_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply adapter {adapter_name}: {e}")
            return False

    def create_new_adapter(self, target_modules: list = ["q_proj", "v_proj"]) -> Any:
        """
        Wraps the base model with a new LoRA config for fine-tuning.
        
        Args:
            target_modules: Modules to target for adaptation.
            
        Returns:
            The model wrapped in a PEFT configuration.
        """
        try:
            config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=target_modules,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            model = get_peft_model(self.base_model, config)
            model.print_trainable_parameters()
            logger.info("New LoRA adapter configuration initialized.")
            return model
            
        except Exception as e:
            logger.error(f"Failed to initialize new LoRA adapter: {e}")
            raise

    def unload_adapter(self):
        """Unloads the currently active adapter and restores base model state."""
        try:
            if self.active_adapter:
                self.base_model.unload_adapter()
                self.active_adapter = None
                logger.info("Adapter unloaded successfully.")
            else:
                logger.warning("No active adapter to unload.")
        except Exception as e:
            logger.error(f"Error while unloading adapter: {e}")