
import os
import subprocess
import logging

# ctx: codexhaven

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ColabTrigger:
    """
    Triggers a Google Colab notebook instance to perform resource-intensive 
    LoRA fine-tuning tasks that are not performant on Android/Termux.
    """
    
    def __init__(self, notebook_path: str = "notebooks/train_persona.ipynb"):
        self.notebook_path = os.path.abspath(notebook_path)
        self.colab_url = "https://colab.research.google.com/"
        
    def prepare_data(self, data_path: str = "./data/training_data.json") -> bool:
        """
        Verify the existence and validity of training data before triggering remote execution.
        """
        abs_data_path = os.path.abspath(data_path)
        if not os.path.exists(abs_data_path):
            logger.error(f"Training data not found at: {abs_data_path}")
            return False
        
        if os.path.getsize(abs_data_path) == 0:
            logger.error(f"Training data is empty at: {abs_data_path}")
            return False
            
        logger.info(f"Training data verified: {abs_data_path}")
        return True

    def launch_session(self) -> None:
        """
        Opens the Colab training environment in the default browser.
        Handles missing environment dependencies gracefully.
        """
        if not os.path.exists(self.notebook_path):
            logger.error(f"Notebook template missing at: {self.notebook_path}")
            raise FileNotFoundError(f"Notebook not found: {self.notebook_path}")
            
        logger.info(f"Opening Colab for training: {self.notebook_path}")
        try:
            # Check if termux-open exists to avoid subprocess errors
            if subprocess.run(['which', 'termux-open'], capture_output=True).returncode == 0:
                subprocess.run(['termux-open', self.colab_url], check=True)
            else:
                raise FileNotFoundError("termux-open utility not found")
        except Exception as e:
            logger.warning(f"Could not open browser automatically: {e}")
            print(f"Please manually open: {self.colab_url} and upload {self.notebook_path}")

    def run_remote_training_pipeline(self, data_path: str) -> None:
        """
        Full workflow: Validate data -> Prepare Trigger -> Launch session.
        Ensures robust error propagation through the pipeline.
        """
        if not data_path or not isinstance(data_path, str):
            logger.error("Invalid data_path provided to pipeline")
            return

        if self.prepare_data(data_path):
            try:
                self.launch_session()
            except Exception as e:
                logger.error(f"Failed to launch remote training pipeline: {e}")
        else:
            logger.error("Pipeline aborted due to missing training artifacts.")

if __name__ == "__main__":
    trigger = ColabTrigger()
    # Ensure correct absolute path resolution
    target_data = os.path.join(os.getcwd(), "data/training_data.json")
    trigger.run_remote_training_pipeline(target_data)