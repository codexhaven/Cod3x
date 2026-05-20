"""
Model Loader — Multi-backend support.
Tries local llama.cpp first, falls back to Hermes, then HF API.
"""
import subprocess
import os
import logging

logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self, model_id: str = "auto"):
        self.model_id = model_id
        self.backend = self._detect_backend()
        logger.info(f"Cod3x using: {self.backend}")
    
    def _detect_backend(self) -> str:
        """Find the best available inference backend."""
        # Check for local GGUF model
        if os.path.exists(os.path.expanduser("~/Cod3x/model.gguf")):
            if os.path.exists(os.path.expanduser("~/llama.cpp/main")):
                return "llama.cpp"
        # Check for Hermes
        if subprocess.run(['which', 'hermes'], capture_output=True).returncode == 0:
            return "hermes"
        # Fall back to HF API
        if os.environ.get('HF_TOKEN'):
            return "hf_api"
        return "none"
    
    def generate(self, prompt: str, system: str = "") -> str:
        if self.backend == "llama.cpp":
            return self._generate_llama(prompt, system)
        elif self.backend == "hermes":
            return self._generate_hermes(prompt, system)
        elif self.backend == "hf_api":
            return self._generate_hf(prompt, system)
        else:
            return "Cod3x: No inference backend available. Install Hermes, llama.cpp, or set HF_TOKEN."
    
    def _generate_llama(self, prompt: str, system: str) -> str:
        full = f"<|system|>\n{system}\n<|user|>\n{prompt}\n<|assistant|>"
        result = subprocess.run(
            [os.path.expanduser("~/llama.cpp/main"), '-m', os.path.expanduser("~/Cod3x/model.gguf"),
             '-p', full, '-n', '256', '--temp', '0.7'],
            capture_output=True, text=True, timeout=120
        )
        return result.stdout.strip()
    
    def _generate_hermes(self, prompt: str, system: str) -> str:
        full = f"SYSTEM: {system}\n\nTASK: {prompt}"
        result = subprocess.run(
            ['hermes', 'chat', '-q', full, '--yolo', '--quiet'],
            capture_output=True, text=True, timeout=120
        )
        return '\n'.join([l for l in result.stdout.strip().split('\n') if not l.startswith('session_id:')])
    
    def _generate_hf(self, prompt: str, system: str) -> str:
        import requests
        api_url = "https://api-inference.huggingface.co/models/microsoft/Phi-3-mini-4k-instruct"
        headers = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"}
        payload = {"inputs": f"<|system|>\n{system}\n<|user|>\n{prompt}\n<|assistant|>", "parameters": {"max_new_tokens": 256}}
        r = requests.post(api_url, headers=headers, json=payload, timeout=60)
        data = r.json()
        if isinstance(data, list): return data[0].get('generated_text', '')
        return data.get('generated_text', '')
