# Project cod3x: Universal AI Persona Factory

cod3x is a modular, high-fidelity AI system designed to adapt a base LLM (e.g., Llama 3) into the behavioral, stylistic, and knowledge-based persona of any target AI. By leveraging LoRA adapters and retrieval-augmented generation (RAG), the system provides lightweight, hot-swappable persona layers without requiring monolithic model retraining.

## Architecture
cod3x uses a separation-of-concerns model:
- Base Layer: Transformer Foundation (Llama 3 / Mistral)
- Adapter Layer: LoRA/QLoRA modules for stylistic alignment
- Context Engine: ChromaDB RAG for persona-specific knowledge injection
- Orchestrator: Dispatches tasks to specific adapters via a modular pipeline

## Core Components
- model_loader.py: Handles efficient model loading with optional 4-bit quantization.
- persona_adapter.py: Manages LoRA adapter injection and hot-swapping.
- vector_db_init.py: Initializes persistent ChromaDB for persona-specific RAG.
- training/train_lora.py: Automated QLoRA fine-tuning workflows.
- scripts/scrape_target.py: Data collection for synthetic persona datasets.

## Getting Started
1. Initialize the environment:
   Ensure all dependencies (PyTorch, Transformers, PEFT, ChromaDB) are installed.
2. Prepare Dataset:
   Run scripts/scrape_target.py to build the persona JSONL dataset.
3. Fine-Tune:
   Run training/train_lora.py to generate a LoRA adapter for the target persona.
4. Integrate:
   Use src/main.py to load the base model and apply the specific persona adapter.

## Project Structure
- data/: Persistent vector database storage.
- src/: Core model loading, adapter application, and orchestration logic.
- scripts/: Data extraction and sanitization utilities.
- training/: Hyperparameter configurations and fine-tuning scripts.
- eval/: Evaluation framework (LLM-as-a-Judge) for persona fidelity.

## Compliance
- Data Sanitization: All target data undergoes automated cleaning.
- Human-in-the-Loop (HITL): Every generated adapter undergoes verification before deployment to ensure no harmful bias transfer occurs.