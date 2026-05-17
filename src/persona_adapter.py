
import os
import logging
import torch
from peft import PeftModel, LoraConfig, get_peft_model
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
        peft_wrapper (Optional[PeftModel]): The active adapter wrapper.
        active_adapter (Optional[str]): The identifier of the currently loaded adapter.
    """
    
    def __init__(self, base_model: Any):
        if base_model is None:
            raise ValueError("Base model cannot be None.")
        self.base_model = base_model
        self.peft_wrapper: Optional[PeftModel] = None
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
        
        abs_path = os.path.abspath(adapter_path)
        if not os.path.exists(os.path.join(abs_path, "adapter_config.json")):
            logger.error(f"Invalid adapter path: missing adapter_config.json in {abs_path}")
            return False

        try:
            logger.info(f"Applying adapter '{adapter_name}' from {abs_path}")
            
            if self.peft_wrapper is None:
                self.peft_wrapper = PeftModel.from_pretrained(
                    self.base_model,
                    abs_path,
                    adapter_name=adapter_name
                )
            else:
                self.peft_wrapper.load_adapter(abs_path, adapter_name=adapter_name)
            
            self.active_adapter = adapter_name
            logger.info(f"Successfully activated adapter: {adapter_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply adapter {adapter_name}: {e}")
            return False

    def create_new_adapter(self, 
                           target_modules: Optional[List[str]] = None,
                           r: int = 16,
                           lora_alpha: int = 32,
                           lora_dropout: float = 0.05) -> Any:
        """
        Wraps the base model with a new LoRA config for fine-tuning.
        
        Args:
            target_modules: Modules to target for adaptation. Defaults to ["q_proj", "v_proj"].
            r: LoRA rank. Must be > 0.
            lora_alpha: LoRA alpha. Must be > 0.
            lora_dropout: LoRA dropout. Must be between 0.0 and 1.0.
            
        Returns:
            The model wrapped in a PEFT configuration.
        """
        if r <= 0:
            raise ValueError("Rank (r) must be greater than 0.")
        if lora_alpha <= 0:
            raise ValueError("LoRA alpha must be greater than 0.")
        if not (0.0 <= lora_dropout < 1.0):
            raise ValueError("LoRA dropout must be in the range [0.0, 1.0).")
            
        target_modules = target_modules or ["q_proj", "v_proj"]
        
        try:
            config = LoraConfig(
                r=r,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            self.peft_wrapper = get_peft_model(self.base_model, config)
            self.peft_wrapper.print_trainable_parameters()
            logger.info("New LoRA adapter configuration initialized.")
            return self.peft_wrapper
            
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
            if self.peft_wrapper and self.active_adapter:
                self.peft_wrapper.disable_adapters()
                self.active_adapter = None
                logger.info("Adapter disabled successfully.")
                return True
            else:
                logger.warning("No active adapter to unload.")
                return True
        except Exception as e:
            logger.error(f"Error while unloading adapter: {e}")
            return False