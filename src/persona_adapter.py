
import os
import logging
import torch
from peft import PeftModel, LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
from typing import Optional, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PersonaAdapter:
    """
    Manages the loading, merging, and application of LoRA adapters
    to adapt a base model to a specific persona.
    
    Attributes:
        base_model (Any): The underlying causal language model.
        active_adapter (Optional[str]): The identifier of the currently loaded adapter.
    """
    
    def __init__(self, base_model: Any):
        if base_model is None:
            raise ValueError("Base model cannot be None.")
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
            
        Raises:
            ValueError: If adapter_path or adapter_name is empty.
        """
        if not adapter_path or not adapter_name:
            raise ValueError("adapter_path and adapter_name must be non-empty strings.")

        try:
            if not os.path.exists(adapter_path):
                logger.error(f"Adapter path does not exist: {adapter_path}")
                return False
            
            logger.info(f"Applying adapter '{adapter_name}' from {adapter_path}")
            
            # Apply the adapter
            # Note: We must ensure the model is already configured for PEFT or we wrap it
            if not isinstance(self.base_model, PeftModel):
                self.base_model = PeftModel.from_pretrained(
                    self.base_model,
                    adapter_path,
                    adapter_name=adapter_name
                )
            else:
                self.base_model.load_adapter(adapter_path, adapter_name=adapter_name)
            
            self.active_adapter = adapter_name
            logger.info(f"Successfully activated adapter: {adapter_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply adapter {adapter_name}: {e}")
            return False

    def create_new_adapter(self, target_modules: Optional[List[str]] = None) -> Any:
        """
        Wraps the base model with a new LoRA config for fine-tuning.
        
        Args:
            target_modules: Modules to target for adaptation. Defaults to ["q_proj", "v_proj"].
            
        Returns:
            The model wrapped in a PEFT configuration.
        """
        target_modules = target_modules or ["q_proj", "v_proj"]
        
        try:
            config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=target_modules,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            # Ensure base_model is a PeftModel or becomes one
            model = get_peft_model(self.base_model, config)
            model.print_trainable_parameters()
            logger.info("New LoRA adapter configuration initialized.")
            return model
            
        except Exception as e:
            logger.error(f"Failed to initialize new LoRA adapter: {e}")
            raise

    def unload_adapter(self) -> bool:
        """
        Unloads the currently active adapter and restores base model state.
        
        Returns:
            bool: True if successful or no active adapter, False otherwise.
        """
        try:
            if self.active_adapter:
                if isinstance(self.base_model, PeftModel):
                    self.base_model.unload()
                    self.active_adapter = None
                    logger.info("Adapter unloaded successfully.")
                    return True
                else:
                    logger.warning("Base model is not a PeftModel instance.")
                    return False
            else:
                logger.warning("No active adapter to unload.")
                return True
        except Exception as e:
            logger.error(f"Error while unloading adapter: {e}")
            return False