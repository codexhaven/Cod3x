import logging
from typing import Any, Dict, Optional
from src.model_loader import ModelLoader
from src.persona_adapter import PersonaAdapter
from src.conversation_logger import log_interaction

# Configure logging with a more robust format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InferenceController:
    """
    Orchestrates the lifecycle of persona-based inference by managing
    the base model and switching between LoRA adapters.

    Attributes:
        loader (ModelLoader): Handles model loading and quantization.
        base_model (Any): The primary transformer model.
        tokenizer (Any): The model's tokenizer.
        adapter_manager (PersonaAdapter): Manages LoRA adapter injection.
        current_persona (Optional[str]): Identifier of the currently active adapter.
    """

    def __init__(self, model_id: str = "meta-llama/Meta-Llama-3-8B"):
        """
        Initializes the controller.

        Args:
            model_id: HuggingFace model hub ID.
        
        Raises:
            RuntimeError: If model or tokenizer loading fails.
        """
        if not model_id or not isinstance(model_id, str):
            raise ValueError("model_id must be a non-empty string.")

        try:
            self.loader = ModelLoader(model_id=model_id)
            self.model_data = self.loader.load_model(quantization=True)
            self.base_model = self.model_data.get("model")
            self.tokenizer = self.model_data.get("tokenizer")
            
            if self.base_model is None or self.tokenizer is None:
                raise RuntimeError("Failed to extract model or tokenizer from loader.")
                
            self.adapter_manager = PersonaAdapter(base_model=self.base_model)
            self.current_persona: Optional[str] = None
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise

    def switch_persona(self, adapter_path: str, persona_id: str) -> bool:
        """
        Switches the active persona by applying the corresponding LoRA adapter.

        Args:
            adapter_path: Path to the .safetensors or .bin adapter file.
            persona_id: Unique string identifier for the persona.

        Returns:
            bool: True if application successful, False otherwise.
        """
        if not adapter_path or not persona_id:
            logger.error("Invalid adapter_path or persona_id provided.")
            return False

        logger.info(f"Switching to persona: {persona_id} from {adapter_path}")
        
        try:
            success = self.adapter_manager.apply_adapter(adapter_path, persona_id)
            if success:
                self.current_persona = persona_id
            return success
        except Exception as e:
            logger.error(f"Failed to switch persona {persona_id}: {e}")
            return False

    def generate_response(self, prompt: str, max_new_tokens: int = 512) -> str:
        """
        Generates a response using the base model with the active persona adapter applied.

        Args:
            prompt: User input string.
            max_new_tokens: Maximum tokens to generate (default: 512).

        Returns:
            str: Generated response string.
        """
        if not prompt or not isinstance(prompt, str):
            logger.warning("Empty or invalid prompt provided.")
            return ""

        if max_new_tokens <= 0:
            logger.error("max_new_tokens must be a positive integer.")
            return ""

        if not self.current_persona:
            logger.warning("No persona adapter active; using base model response.")

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.base_model.device)
            
            # Use torch.no_grad to reduce memory overhead
            import torch
            with torch.no_grad():
                outputs = self.base_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )
                
            log_interaction(prompt, self.tokenizer.decode(outputs[0], skip_special_tokens=True))
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"Error generating response: {str(e)}"

    def shutdown(self):
        """
        Cleans up resources and releases memory.
        """
        logger.info("Shutting down inference controller.")
        try:
            # Explicit deletion if possible
            if hasattr(self, 'base_model'): del self.base_model
            if hasattr(self, 'loader'): del self.loader
            if hasattr(self, 'adapter_manager'): del self.adapter_manager
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
