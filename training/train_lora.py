
import os
import logging
import yaml
import torch
from typing import Dict, Any
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LoraTrainer:
    def __init__(self, config_path: str = "./training/hyperparameters.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.model_id = self.config.get("model_id", "meta-llama/Meta-Llama-3-8B")
        
    def train(self, dataset_path: str, output_dir: str):
        """
        Executes the QLoRA fine-tuning process using the SFTTrainer.
        
        Args:
            dataset_path: Path to the JSONL dataset.
            output_dir: Directory to save the resulting LoRA adapter.
        """
        logger.info(f"Starting training for persona adaptation on {dataset_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.config.get("lora_r", 8),
            lora_alpha=self.config.get("lora_alpha", 32),
            lora_dropout=self.config.get("lora_dropout", 0.05)
        )
        
        dataset = load_dataset('json', data_files=dataset_path, split='train')
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.get("epochs", 3),
            per_device_train_batch_size=self.config.get("batch_size", 4),
            learning_rate=self.config.get("learning_rate", 2e-4),
            logging_steps=10,
            save_strategy="epoch"
        )
        
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            peft_config=peft_config,
            dataset_text_field="text"
        )
        
        trainer.train()
        trainer.model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Training complete. Adapter saved to {output_dir}")

if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs("./models/adapters", exist_ok=True)
    
    # Initialize and execute training
    trainer = LoraTrainer()
    trainer.train(
        dataset_path="./data/raw_datasets/training_data.jsonl",
        output_dir="./models/adapters/persona_v1"
    )