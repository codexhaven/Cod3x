import logging
import json
import os
from typing import Dict, Any, List
from eval.llm_judge import LLMJudge

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ValidationLoop:
    """
    Orchestrates the LLM-as-a-Judge and HITL (Human-in-the-Loop)
    validation process for persona-adapted models.
    """
    
    def __init__(self, judge_model_id: str = "gpt-4"):
        self.judge = LLMJudge(model_id=judge_model_id)
        self.results_file = "eval/metrics.json"

    def run_validation(self, persona_name: str, evaluation_dataset: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Executes a full validation cycle for a specific persona adapter.
        
        Args:
            persona_name: The name of the adapter being validated.
            evaluation_dataset: List of dicts with 'prompt' and 'expected_style'.
            
        Returns:
            Dict containing aggregated metrics.
        """
        logger.info(f"Starting validation for persona: {persona_name}")
        scores = []
        
        for entry in evaluation_dataset:
            prompt = entry["prompt"]
            # Placeholder: In integration, model.generate(prompt) would be called here
            generated_response = "Simulated model output for persona validation."
            
            score = self.judge.evaluate(prompt, generated_response, entry["expected_style"])
            scores.append(score)
            
        avg_score = sum(scores) / len(scores) if scores else 0
        metrics = {
            "persona": persona_name,
            "average_score": avg_score,
            "total_evals": len(scores)
        }
        
        self._save_results(metrics)
        logger.info(f"Validation complete. Avg score: {avg_score}")
        return metrics

    def _save_results(self, metrics: Dict[str, Any]):
        """Persists validation metrics to JSON file."""
        os.makedirs(os.path.dirname(self.results_file), exist_ok=True)
        data = []
        if os.path.exists(self.results_file):
            with open(self.results_file, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
        
        data.append(metrics)
        with open(self.results_file, 'w') as f:
            json.dump(data, f, indent=4)
        logger.info(f"Metrics saved to {self.results_file}")