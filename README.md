
# Project cod3x: Universal AI Persona Factory

cod3x is a modular, high-fidelity AI system designed to adapt a base LLM (e.g., Llama 3) into the behavioral, stylistic, and knowledge-based persona of any target AI. By leveraging LoRA adapters and retrieval-augmented generation (RAG), the system provides lightweight, hot-swappable persona layers without requiring monolithic model retraining.

## Architecture
cod3x uses a separation-of-concerns model:
- Base Layer: Transformer Foundation (Llama 3 / Mistral)
- Adapter Layer: LoRA/QLoRA modules for stylistic alignment
- Context Engine: ChromaDB RAG for persona-specific knowledge injection
- Orchestrator: Dispatches tasks to specific adapters via a modular pipeline

## Core Components
- model_loader.py: Handles efficient model loading with optional 4-bit quantization. Includes strict input validation for file paths, memory bounds, and configuration parameters.
- persona_adapter.py: Manages LoRA adapter injection and hot-swapping. Features atomic injection checks to prevent corrupt state.
- vector_db_init.py: Initializes persistent ChromaDB for persona-specific RAG. Implements directory existence and permission checks.
- training/train_lora.py: Automated QLoRA fine-tuning workflows with hyperparameter validation. Guards against non-converging learning rates and invalid batch sizes.
- scripts/scrape_target.py: Data collection for synthetic persona datasets. Includes robust error handling (retries, timeouts), path sanitation, and schema validation against JSONL requirements.

## Getting Started
1. Initialize the environment:
   Ensure all dependencies (PyTorch, Transformers, PEFT, ChromaDB) are installed. Verify your CUDA environment if using GPU acceleration.
   Refer to [requirements.txt](requirements.txt) for specific dependency versions.
2. Pre-flight Check:
   Run a hardware audit to ensure CUDA compatibility and sufficient VRAM:
   python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
3. Prepare Dataset:
   Run scripts/scrape_target.py to build the persona JSONL dataset.
4. Fine-Tune:
   Run training/train_lora.py to generate a LoRA adapter for the target persona. Validate training progress via the log framework.
5. Integrate:
   Use src/main.py to load the base model and apply the specific persona adapter. Ensure your environment variables for absolute paths (e.g., export PERSIST_DIR='/home/user/project/data/chroma_db').

## Project Structure
- data/: Persistent vector database storage.
- src/: Core model loading, adapter application, and orchestration logic.
- scripts/: Data extraction and sanitization utilities.
- training/: Hyperparameter configurations and fine-tuning scripts.
- eval/: Evaluation framework (LLM-as-a-Judge) for persona fidelity.

## Compliance, Security & Robustness
- License: This project is released under the [LICENSE](LICENSE) file (ensure this file exists in the root).
- Data Sanitization: All target data undergoes automated cleaning using defined schema validation (pydantic/JSON schema).
- Human-in-the-Loop (HITL): Every generated adapter undergoes verification before deployment to ensure no harmful bias transfer occurs.
- Input Validation: All core scripts include guard clauses (e.g., `assert path.exists()`, `if not value > 0`) for pathing and configuration values.
- Error Handling: Use Result objects (success/failure) or explicit try-except blocks with logging. Unhandled exceptions are logged to `error.log`.
- Performance: Avoid O(n²) loops; use vector-optimized operations where possible. Pre-allocate tensors to avoid dynamic memory fragmentation.

## Testing & Maintenance
- Structural Integrity: Run `bash -n` on all `scripts/` before execution.
- Automated Testing: Use `pytest` for property-based testing of core logic (`src/utils.py`). Run `tests/boundary_check.py` for edge cases (empty input, max memory, invalid paths).
- Validation: Always validate adapter injection with `src/validation_loop.py` script before serving to production.
- Documentation: All new methods must include docstrings (Google Style) documenting parameters, return types, and potential exceptions (raised errors).