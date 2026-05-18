import os
import logging
import yaml
import torch
from typing import Dict, Any, Optional
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LoraTrainer:
    """
    Handles QLoRA fine-tuning workflows for persona adaptation.
    Provides robust configuration loading, validation, and error handling.
    """
    def __init__(self, config_path: str = "./training/hyperparameters.yaml"):
        if not os.path.isabs(config_path):
            config_path = os.path.abspath(config_path)
            
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at: {config_path}")
            
        with open(config_path, 'r') as f:
            try:
                self.config = yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                logger.error(f"Error parsing YAML config: {e}")
                raise

        self.model_id = self.config.get("model_id", "meta-llama/Meta-Llama-3-8B")
        self._validate_config()

    def _validate_config(self):
        """Validates configuration parameters to prevent runtime errors."""
        required_params = {
            "lora_r": 8,
            "lora_alpha": 32,
            "learning_rate": 2e-4,
            "epochs": 3,
            "batch_size": 4
        }
        for param, default in required_params.items():
            if param not in self.config:
                logger.warning(f"Parameter '{param}' missing in config, using default: {default}")
                self.config[param] = default
        
        if self.config.get("lora_r", 8) <= 0:
            raise ValueError("lora_r must be a positive integer.")
        if self.config.get("batch_size", 4) <= 0:
            raise ValueError("batch_size must be a positive integer.")

    def train(self, dataset_path: str, output_dir: str):
        """
        Executes the QLoRA fine-tuning process.
        
        Args:
            dataset_path: Path to the JSONL dataset.
            output_dir: Directory to save the resulting LoRA adapter.
            
        Raises:
            FileNotFoundError: If dataset path is invalid.
            RuntimeError: If training fails during execution.
        """
        dataset_path = os.path.abspath(dataset_path)
        output_dir = os.path.abspath(output_dir)

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        logger.info(f"Starting training on {dataset_path}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Setup 4-bit quantization config (Conditional based on hardware)
            use_cuda = torch.cuda.is_available()
            bnb_config = None
            if use_cuda:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16
                )
            else:
                logger.warning("CUDA not detected. Training will be extremely slow on CPU.")
            
            device_map = "auto" if use_cuda else None
            
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map=device_map,
                quantization_config=bnb_config,
                torch_dtype=torch.float16 if use_cuda else torch.float32
            )
            
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.config.get("lora_r"),
                lora_alpha=self.config.get("lora_alpha"),
                lora_dropout=self.config.get("lora_dropout", 0.05)
            )
            
            dataset = load_dataset('json', data_files=dataset_path, split='train')
            if not isinstance(dataset, Dataset) or len(dataset) == 0:
                raise ValueError("Dataset is empty or invalid.")
            
            # Determine correct text field
            target_fields = ["text", "prompt", "instruction"]
            text_field = next((field for field in target_fields if field in dataset.column_names), None)
            
            if text_field is None:
                raise ValueError(f"Dataset must contain one of: {target_fields}")
            
            # Format dataset if instruction/response exists
            if 'instruction' in dataset.column_names and 'response' in dataset.column_names:
                def format_example(example):
                    return {"text": f"Instruction: {example['instruction']}\nResponse: {example['response']}"}
                dataset = dataset.map(format_example)
                text_field = "text"
            
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=self.config.get("epochs"),
                per_device_train_batch_size=self.config.get("batch_size"),
                learning_rate=float(self.config.get("learning_rate")),
                logging_steps=10,
                save_strategy="epoch",
                remove_unused_columns=False
            )
            
            trainer = SFTTrainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                peft_config=peft_config,
                dataset_text_field=text_field
            )
            
            trainer.train()
            trainer.model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logger.info(f"Training complete. Adapter saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise RuntimeError(f"Training pipeline failed: {e}") from e

if __name__ == "__main__":
    # Ensure base directory exists
    os.makedirs(os.path.abspath("./models/adapters"), exist_ok=True)
    
    # Initialize and execute training
    try:
        trainer = LoraTrainer()
        # Train on the directory of batch files
        trainer.train(
            dataset_path="./data/training_batches",
            output_dir="./models/adapters/persona_v1"
        )
    except Exception as e:
        logger.error(f"Execution aborted: {e}")
        exit(1)
