import logging
import json
import os
import fcntl
from typing import Dict, Any, List, Optional, Union
from eval.llm_judge import LLMJudge

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ValidationLoop:
    """
    Orchestrates the LLM-as-a-Judge and HITL (Human-in-the-Loop)
    validation process for persona-adapted models.
    
    Attributes:
        judge_model_id (str): Model ID for LLM-as-a-Judge.
        results_file (str): Path to JSON file for metrics.
    """
    
    def __init__(self, judge_model_id: str = "gpt-4", results_file: str = "eval/metrics.json"):
        """
        Initializes the ValidationLoop with specified model and storage.

        Args:
            judge_model_id: Identifier for the evaluation LLM.
            results_file: Filesystem path for storing metrics.
        """
        self.judge = LLMJudge(model_id=judge_model_id)
        self.results_file = os.path.abspath(results_file)

    def run_validation(self, persona_name: str, evaluation_dataset: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Executes a full validation cycle for a specific persona adapter.
        
        Args:
            persona_name: The identifier of the adapter being validated.
            evaluation_dataset: List of dicts containing 'prompt' and 'expected_style'.
            
        Returns:
            Dict containing aggregated metrics, average score, and total evaluations.

        Raises:
            ValueError: If inputs are missing or improperly formatted.
        """
        if not persona_name or not isinstance(persona_name, str):
            raise ValueError("persona_name must be a non-empty string.")
            
        if not evaluation_dataset:
            logger.warning(f"Empty evaluation dataset provided for {persona_name}.")
            return {"persona": persona_name, "average_score": 0.0, "total_evals": 0}

        logger.info(f"Starting validation for persona: {persona_name} with {len(evaluation_dataset)} entries.")
        scores: List[float] = []
        
        for i, entry in enumerate(evaluation_dataset):
            prompt = entry.get("prompt")
            expected_style = entry.get("expected_style")
            
            if not prompt or not expected_style:
                logger.warning(f"Skipping entry {i}: missing keys ('prompt' or 'expected_style').")
                continue

            # Placeholder: In integration, model.generate(prompt) would be called here
            generated_response = "Simulated model output for persona validation."
            
            try:
                score = self.judge.evaluate(prompt, generated_response, expected_style)
                if score is not None:
                    val = float(score)
                    if 0.0 <= val <= 10.0:
                        scores.append(val)
                    else:
                        logger.warning(f"Score {val} at index {i} out of bounds (0-10), skipping.")
            except Exception as e:
                logger.error(f"Judge evaluation failed at index {i}: {e}")
            
        avg_score = sum(scores) / len(scores) if scores else 0.0
        metrics = {
            "persona": persona_name,
            "average_score": round(float(avg_score), 4),
            "total_evals": len(scores)
        }
        
        self._save_results(metrics)
        logger.info(f"Validation complete. Avg score: {avg_score:.2f} across {len(scores)} evals.")
        return metrics

    def _save_results(self, metrics: Dict[str, Any]):
        """
        Persists validation metrics to JSON file with atomic file locking.
        
        Note:
            Uses fcntl for process-safe appending. Rebuilds file if corrupted.
        """
        os.makedirs(os.path.dirname(self.results_file), exist_ok=True)
        
        try:
            # Use 'a+' to open, ensure file exists, then acquire lock
            with open(self.results_file, 'a+') as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                
                f.seek(0)
                content = f.read().strip()
                data: List[Dict[str, Any]] = []
                
                if content:
                    try:
                        data = json.loads(content)
                        if not isinstance(data, list):
                            data = [data]
                    except json.JSONDecodeError:
                        logger.warning(f"Metrics file {self.results_file} corrupt. Overwriting.")
                        data = []
            
                data.append(metrics)
                
                f.seek(0)
                f.truncate()
                json.dump(data, f, indent=4)
                
                fcntl.flock(f, fcntl.LOCK_UN)
            logger.debug(f"Metrics persisted to {self.results_file}")
        except (IOError, OSError) as e:
            logger.error(f"IO Failure while saving metrics to {self.results_file}: {e}")
