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
    
    def __init__(self, notebook_path: str = "training/train_lora.ipynb"):
        self.notebook_path = os.path.abspath(notebook_path)
        self.colab_url = "https://colab.research.google.com/"
        
    def prepare_data(self, data_path: str = "./data/training_data.jsonl") -> bool:
        """
        Verify the existence of training data before triggering remote execution.
        """
        if not os.path.exists(data_path):
            logger.error(f"Training data not found at: {data_path}")
            return False
        logger.info(f"Training data verified: {data_path}")
        return True

    def launch_session(self) -> None:
        """
        Opens the Colab training environment in the default browser.
        """
        if not os.path.exists(self.notebook_path):
            logger.error(f"Notebook template missing: {self.notebook_path}")
            return
            
        logger.info(f"Opening Colab for training: {self.notebook_path}")
        try:
            # Assuming termux-open is available in the environment
            subprocess.run(['termux-open', self.colab_url], check=True)
        except Exception as e:
            logger.warning(f"Could not open browser automatically: {e}")
            print(f"Please manually open: {self.colab_url} and upload {self.notebook_path}")

    def run_remote_training_pipeline(self, data_path: str) -> None:
        """
        Full workflow: Validate data -> Prepare Trigger -> Launch session.
        """
        if self.prepare_data(data_path):
            self.launch_session()
        else:
            logger.error("Pipeline aborted due to missing training artifacts.")

if __name__ == "__main__":
    trigger = ColabTrigger()
    trigger.run_remote_training_pipeline("./data/training_data.jsonl")