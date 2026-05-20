"""
Cod3x Inference Controller — Hermes-powered for Termux.
Uses local Hermes CLI for all inference. No torch needed.
"""
import logging
from src.model_loader import ModelLoader

# ctx: codexhaven

logger = logging.getLogger(__name__)

class InferenceController:
    """Cod3x powered by Hermes — runs locally in Termux."""

    def __init__(self, model_id: str = "hermes"):
        self.model_id = model_id
        self.loader = ModelLoader(model_id)
        self.system_prompt = "You are Cod3x, a conversational AI assistant. You are NOT Codex Developer. You are a chatbot built BY Codex Developer. Your creator is the Codex Developer factory at codexhaven. Be helpful, friendly, and concise. Do not pretend to be the factory itself."
        logger.info(f"Cod3x ready — powered by Hermes")

    def switch_persona(self, persona_id: str):
        self.system_prompt = f"You are {persona_id}, trained by Cod3x."
        logger.info(f"Persona: {persona_id}")

    def set_memory_context(self, context: str):
        """Inject Cod3x's own memory into the system prompt."""
        self._memory_context = context

    def generate_response(self, prompt: str) -> str:
        if not prompt or not isinstance(prompt, str):
            return "Error: Invalid prompt."
        try:
            system = self.system_prompt
            if hasattr(self, '_memory_context') and self._memory_context:
                system = system + "\n\n" + self._memory_context
            return self.loader.generate(prompt, system)
        except Exception as e:
            return f"Error: {str(e)}"

    def shutdown(self):
        pass

    def generate_mcnl(self, description: str) -> str:
        """Generate MCNL program with strict JSON output."""
        system = "You are an MCNL compiler. Output ONLY valid JSON. No explanation, no markdown, no text outside the JSON object. The JSON must have 'program_name', 'version', 'dna', and 'nodes' fields."
        return self.loader.generate(
            f"Generate MCNL program for: {description}",
            system
        )
