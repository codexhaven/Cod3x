import logging
import json
import os
import fcntl
from typing import Dict, Any, List, Optional
from eval.llm_judge import LLMJudge

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ValidationLoop:
    """
    Orchestrates the LLM-as-a-Judge and HITL (Human-in-the-Loop)
    validation process for persona-adapted models.
    """
    
    def __init__(self, judge_model_id: str = "gpt-4", results_file: str = "eval/metrics.json"):
        """
        Initializes the ValidationLoop.

        Args:
            judge_model_id: Model ID for LLM-as-a-Judge.
            results_file: Path to JSON file for metrics.
        """
        self.judge = LLMJudge(model_id=judge_model_id)
        self.results_file = results_file

    def run_validation(self, persona_name: str, evaluation_dataset: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Executes a full validation cycle for a specific persona adapter.
        
        Args:
            persona_name: The name of the adapter being validated.
            evaluation_dataset: List of dicts with 'prompt' and 'expected_style'.
            
        Returns:
            Dict containing aggregated metrics.

        Raises:
            ValueError: If input parameters are invalid.
        """
        if not persona_name:
            raise ValueError("Persona name must be provided.")
        if not evaluation_dataset:
            logger.warning(f"Evaluation dataset is empty for {persona_name}.")
            return {"persona": persona_name, "average_score": 0.0, "total_evals": 0}

        logger.info(f"Starting validation for persona: {persona_name}")
        scores = []
        
        for i, entry in enumerate(evaluation_dataset):
            prompt = entry.get("prompt")
            expected_style = entry.get("expected_style")
            
            if not prompt or not expected_style:
                logger.warning(f"Skipping entry {i} due to missing keys.")
                continue

            # Placeholder: In integration, model.generate(prompt) would be called here
            generated_response = "Simulated model output for persona validation."
            
            try:
                score = self.judge.evaluate(prompt, generated_response, expected_style)
                if score is not None:
                    # Validate score is within 0.0 - 1.0 or 0 - 10 range as expected
                    val = float(score)
                    if 0.0 <= val <= 10.0:
                        scores.append(val)
                    else:
                        logger.warning(f"Score {val} out of bounds, ignoring.")
            except Exception as e:
                logger.error(f"Judge evaluation failed at index {i}: {e}")
            
        avg_score = sum(scores) / len(scores) if scores else 0.0
        metrics = {
            "persona": persona_name,
            "average_score": float(avg_score),
            "total_evals": len(scores)
        }
        
        self._save_results(metrics)
        logger.info(f"Validation complete. Avg score: {avg_score}")
        return metrics

    def _save_results(self, metrics: Dict[str, Any]):
        """Persists validation metrics to JSON file with file locking."""
        os.makedirs(os.path.dirname(self.results_file), exist_ok=True)
        
        try:
            with open(self.results_file, 'a+') as f:
                # Exclusive lock
                fcntl.flock(f, fcntl.LOCK_EX)
                
                f.seek(0)
                content = f.read()
                data = []
                if content:
                    try:
                        data = json.loads(content)
                    except json.JSONDecodeError:
                        logger.warning(f"Corrupt metrics file {self.results_file}, resetting.")
                        data = []
            
                if not isinstance(data, list):
                    data = [data]
                    
                data.append(metrics)
                
                f.seek(0)
                f.truncate()
                json.dump(data, f, indent=4)
                
                # Unlock
                fcntl.flock(f, fcntl.LOCK_UN)
            logger.info(f"Metrics saved to {self.results_file}")
        except IOError as e:
            logger.error(f"Failed to save metrics to {self.results_file}: {e}")