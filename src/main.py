
import logging
import sys
import argparse
import os
from typing import Optional, Dict, Any
from model_loader import ModelLoader
from persona_adapter import PersonaAdapter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PersonaOrchestrator:
    """
    Main controller for the AI Persona Adaptation System.
    Manages base model lifecycle and adapter swapping with validation.
    """
    def __init__(self, model_id: str = "meta-llama/Meta-Llama-3-8B"):
        if not model_id or not isinstance(model_id, str):
            raise ValueError("Invalid model_id provided.")
        self.loader = ModelLoader(model_id=model_id)
        self.adapter_manager: Optional[PersonaAdapter] = None
        self.components: Optional[Dict[str, Any]] = None

    def initialize(self) -> None:
        """Initializes the base model and adapter manager. Exits on failure."""
        try:
            self.components = self.loader.load_model(quantization=True)
            if 'model' not in self.components or 'tokenizer' not in self.components:
                raise KeyError("Missing 'model' or 'tokenizer' in model components.")
            self.adapter_manager = PersonaAdapter(base_model=self.components['model'])
            logger.info("Orchestrator initialized successfully.")
        except Exception as e:
            logger.error(f"Critical error initializing orchestrator: {e}", exc_info=True)
            sys.exit(1)

    def run_inference(self, prompt: str, adapter_path: str, adapter_name: str) -> None:
        """
        Applies a persona and runs a test generation with validation.
        
        Args:
            prompt: The input prompt text.
            adapter_path: Filesystem path to the LoRA adapter.
            adapter_name: Unique identifier for the adapter.
        """
        if not prompt or not isinstance(prompt, str) or not prompt.strip():
            logger.error("Inference aborted: Empty or invalid prompt.")
            return

        if not os.path.exists(adapter_path):
            logger.error(f"Inference aborted: Adapter path does not exist: {adapter_path}")
            return

        if self.adapter_manager is None or self.components is None:
            logger.error("Inference aborted: Orchestrator not initialized.")
            return

        if not self.adapter_manager.apply_adapter(adapter_path, adapter_name):
            logger.error(f"Inference aborted: Failed to apply adapter {adapter_name} at {adapter_path}.")
            return

        try:
            tokenizer = self.components['tokenizer']
            model = self.components['model']
            
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # Context management for adapter handling
            with model.disable_adapter():
                outputs = model.generate(**inputs, max_new_tokens=100)
                
            decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            print(f"\n--- Generated Response (Persona: {adapter_name}) ---")
            print(decoded_output)
            
        except Exception as e:
            logger.error(f"Inference failed during generation: {e}", exc_info=True)

def main():
    parser = argparse.ArgumentParser(description="Codex Persona Adaptation CLI")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt for the model")
    parser.add_argument("--adapter_path", type=str, required=True, help="Absolute path to LoRA adapter")
    parser.add_argument("--adapter_name", type=str, required=True, help="Name for adapter identification")
    args = parser.parse_args()

    orchestrator = PersonaOrchestrator()
    orchestrator.initialize()
    orchestrator.run_inference(args.prompt, os.path.abspath(args.adapter_path), args.adapter_name)

if __name__ == "__main__":
    main()