
import logging
import sys
import argparse
import os
from typing import Optional, Dict, Any
from model_loader import ModelLoader
from persona_adapter import PersonaAdapter

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PersonaOrchestrator")

class PersonaOrchestrator:
    """
    Main controller for the AI Persona Adaptation System.
    Manages base model lifecycle and adapter swapping with validation.
    
    Attributes:
        model_id (str): The HuggingFace hub ID or local path of the base model.
    """
    def __init__(self, model_id: str = "meta-llama/Meta-Llama-3-8B"):
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be a non-empty string.")
        
        self.model_id = model_id
        self.loader = ModelLoader(model_id=model_id)
        self.adapter_manager: Optional[PersonaAdapter] = None
        self.components: Optional[Dict[str, Any]] = None

    def initialize(self) -> None:
        """
        Initializes the base model and adapter manager. 
        
        Raises:
            RuntimeError: If model loading or adapter initialization fails.
            KeyError: If expected model components are missing.
        """
        try:
            logger.info(f"Initializing base model: {self.model_id}")
            self.components = self.loader.load_model(quantization=True)
            
            if not isinstance(self.components, dict) or 'model' not in self.components or 'tokenizer' not in self.components:
                raise KeyError("Invalid components returned from ModelLoader: missing 'model' or 'tokenizer'.")
            
            self.adapter_manager = PersonaAdapter(base_model=self.components['model'])
            logger.info("Orchestrator initialized successfully.")
        except Exception as e:
            logger.error(f"Critical error initializing orchestrator: {e}", exc_info=True)
            raise RuntimeError(f"Initialization failed: {e}") from e

    def run_inference(self, prompt: str, adapter_path: str, adapter_name: str) -> None:
        """
        Applies a persona adapter and executes a test generation.
        
        Args:
            prompt: Non-empty input string.
            adapter_path: Filesystem path to the LoRA adapter.
            adapter_name: Unique identifier for the adapter.
            
        Raises:
            ValueError: If inputs are invalid.
        """
        # Input Validation
        if not prompt or not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string.")
        if not adapter_path or not isinstance(adapter_path, str):
            raise ValueError("adapter_path must be a valid path string.")
            
        abs_adapter_path = os.path.abspath(adapter_path)
        
        if not os.path.exists(abs_adapter_path):
            raise FileNotFoundError(f"Adapter path not found: {abs_adapter_path}")

        if self.adapter_manager is None or self.components is None:
            raise RuntimeError("Orchestrator not initialized. Call initialize() first.")

        # Apply Adapter
        logger.info(f"Applying adapter: {adapter_name}")
        if not self.adapter_manager.apply_adapter(abs_adapter_path, adapter_name):
            raise RuntimeError(f"Failed to apply adapter {adapter_name} at {abs_adapter_path}.")

        # Generate
        try:
            tokenizer = self.components['tokenizer']
            model = self.components['model']
            
            # Use appropriate attention mask/device management
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with model.device: # Explicit context management if available
                outputs = model.generate(**inputs, max_new_tokens=100)
                
            decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            print(f"\n--- Generated Response (Persona: {adapter_name}) ---")
            print(decoded_output)
            
        except Exception as e:
            logger.error(f"Inference failed during generation: {e}", exc_info=True)
            raise

def main():
    parser = argparse.ArgumentParser(description="Codex Persona Adaptation CLI")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt for the model")
    parser.add_argument("--adapter_path", type=str, required=True, help="Absolute path to LoRA adapter")
    parser.add_argument("--adapter_name", type=str, required=True, help="Name for adapter identification")
    args = parser.parse_args()

    orchestrator = PersonaOrchestrator()
    try:
        orchestrator.initialize()
        orchestrator.run_inference(args.prompt, args.adapter_path, args.adapter_name)
    except Exception as e:
        logger.error(f"Application execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
