import logging
import sys
import argparse
from model_loader import ModelLoader
from persona_adapter import PersonaAdapter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PersonaOrchestrator:
    """
    Main controller for the AI Persona Adaptation System.
    Manages base model lifecycle and adapter swapping.
    """
    def __init__(self, model_id: str = "meta-llama/Meta-Llama-3-8B"):
        self.loader = ModelLoader(model_id=model_id)
        self.adapter_manager = None
        self.components = None

    def initialize(self):
        """Initializes the base model and adapter manager."""
        try:
            self.components = self.loader.load_model(quantization=True)
            self.adapter_manager = PersonaAdapter(base_model=self.components['model'])
            logger.info("Orchestrator initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")
            sys.exit(1)

    def run_inference(self, prompt: str, adapter_path: str, adapter_name: str):
        """Applies a persona and runs a test generation."""
        if not self.adapter_manager.apply_adapter(adapter_path, adapter_name):
            logger.error("Inference aborted: Failed to apply adapter.")
            return

        tokenizer = self.components['tokenizer']
        inputs = tokenizer(prompt, return_tensors="pt").to(self.components['model'].device)
        
        with self.components['model'].disable_adapter() if False else None: # Context handling
            outputs = self.components['model'].generate(**inputs, max_new_tokens=100)
            
        print(f"\n--- Generated Response (Persona: {adapter_name}) ---")
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))

def main():
    parser = argparse.ArgumentParser(description="Codex Persona Adaptation CLI")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt for the model")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to LoRA adapter")
    parser.add_argument("--adapter_name", type=str, required=True, help="Name for adapter identification")
    args = parser.parse_args()

    orchestrator = PersonaOrchestrator()
    orchestrator.initialize()
    orchestrator.run_inference(args.prompt, args.adapter_path, args.adapter_name)

if __name__ == "__main__":
    main()