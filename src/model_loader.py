"""
Model Loader — Hermes-powered for Termux.
Wraps Hermes CLI for inference.
"""
import subprocess
import logging

logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self, model_id: str = "hermes"):
        self.model_id = model_id
        logger.info(f"Using Hermes as inference backend")

    def generate(self, prompt: str, system: str = "", max_tokens: int = 512) -> str:
        """Call Hermes. System prompt goes FIRST in the message for priority."""
        if system:
            full_prompt = f"SYSTEM INSTRUCTION (follow this exactly): {system}\n\nTASK: {prompt}"
        else:
            full_prompt = prompt
        
        result = subprocess.run(
            ['hermes', 'chat', '-q', full_prompt, '--yolo', '--quiet'],
            capture_output=True, text=True,
            timeout=120
        )
        output = result.stdout.strip()
        lines = [l for l in output.split('\n') if not l.startswith('session_id:')]
        return '\n'.join(lines)
