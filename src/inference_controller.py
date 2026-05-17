import logging
from typing import Any, Dict, Optional
from src.model_loader import ModelLoader
from src.persona_adapter import PersonaAdapter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InferenceController:
    """
    Orchestrates the lifecycle of persona-based inference by managing
    the base model and switching between LoRA adapters.
    """

    def __init__(self, model_id: str = "meta-llama/Meta-Llama-3-8B"):
        self.loader = ModelLoader(model_id=model_id)
        self.model_data = self.loader.load_model(quantization=True)
        self.base_model = self.model_data["model"]
        self.tokenizer = self.model_data["tokenizer"]
        self.adapter_manager = PersonaAdapter(base_model=self.base_model)
        self.current_persona: Optional[str] = None

    def switch_persona(self, adapter_path: str, persona_id: str) -> bool:
        """
        Switches the active persona by applying the corresponding LoRA adapter.
        """
        logger.info(f"Switching to persona: {persona_id} from {adapter_path}")
        success = self.adapter_manager.apply_adapter(adapter_path, persona_id)
        if success:
            self.current_persona = persona_id
        return success

    def generate_response(self, prompt: str, max_new_tokens: int = 512) -> str:
        """
        Generates a response using the base model with the active persona adapter applied.
        """
        if not self.current_persona:
            logger.warning("No persona adapter active; using base model response.")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.base_model.device)
        
        with self.base_model.device:
            outputs = self.base_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
            
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def shutdown(self):
        """
        Cleans up resources.
        """
        logger.info("Shutting down inference controller.")
        del self.base_model
        del self.loader
        del self.adapter_manager