"""
Persona Adapter — stub for Termux.
Full LoRA adapter loading requires torch (available in Colab).
"""
import logging

# ctx: codexhaven

logger = logging.getLogger(__name__)

class PersonaAdapter:
    def __init__(self, base_model=None):
        logger.info("PersonaAdapter: torch not available — using prompt-based personas")
    
    def apply_adapter(self, adapter_path: str, adapter_name: str) -> bool:
        logger.info(f"Persona '{adapter_name}' registered (prompt-based)")
        return True
    
    def unload_adapter(self) -> bool:
        return True
