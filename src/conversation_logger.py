
import sys
import os
import logging
from typing import Optional
from pathlib import Path

# Configure internal module logging
logger = logging.getLogger("InferenceController")
logging.basicConfig(level=logging.INFO)

# Ensure project root is in path to allow imports from scripts/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from scripts.conversation_logger import log_interaction
except ImportError:
    logger.warning("scripts.conversation_logger not found. Falling back to local logging.")
    def log_interaction(query: str, response: str) -> None:
        """Fallback logger when script integration is unavailable."""
        print(f"DEBUG_LOG: Q:{query} | R:{response}")

class InferenceController:
    """
    Controller for managing model inference and interaction logging.
    
    Attributes:
        model_name (str): Identifier for the model version.
    """
    def __init__(self, model_name: Optional[str] = None):
        if not model_name or not isinstance(model_name, str):
            raise ValueError("model_name must be a non-empty string.")
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        """
        Generates a response and logs the interaction.
        
        Args:
            prompt (str): The input query.
            
        Returns:
            str: The generated model response.
            
        Raises:
            ValueError: If prompt is empty or not a string.
        """
        if not prompt or not isinstance(prompt, str):
            raise ValueError("Invalid prompt: input must be a non-empty string.")

        # Simulate generation process
        try:
            response = f"Response from {self.model_name} to prompt: {prompt}"
        except Exception as e:
            logger.error(f"Inference generation failed: {e}")
            return "ERROR: Generation failed."
        
        # Log the conversation using the infrastructure defined in scripts/
        try:
            log_interaction(prompt, response)
        except Exception as e:
            # Prevent logging failures from crashing the main inference flow
            logger.error(f"InferenceController: Error logging interaction: {e}")
        
        return response

if __name__ == "__main__":
    # Integration test with boundary checks
    try:
        controller = InferenceController("AI-Module-v1")
        
        # Test Case 1: Valid input
        result = controller.generate("Test prompt")
        print(f"Result: {result}")
        
        # Test Case 2: Boundary/Empty check
        try:
            controller.generate("")
        except ValueError as ve:
            print(f"Caught expected error: {ve}")
            
    except Exception as e:
        print(f"Integration test failed: {e}")