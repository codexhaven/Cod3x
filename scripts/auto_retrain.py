#!/usr/bin/env python3
"""Auto-retrain: triggers training when enough new conversations accumulate."""
import os
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("auto_retrain")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONVERSATIONS_FILE = PROJECT_ROOT / "data" / "conversations.jsonl"
TRAINING_DATA_FILE = PROJECT_ROOT / "data" / "training_data.json"
TRAIN_SCRIPT = PROJECT_ROOT / "training" / "train_lora.py"
THRESHOLD = 100
CYCLE_LOG = PROJECT_ROOT / "data" / "retrain_cycles.jsonl"


def count_new_examples() -> int:
    """Count how many conversations exist since last retrain."""
    if not CONVERSATIONS_FILE.exists():
        return 0
    
    # Count how many cycles have already been processed
    processed = 0
    if CYCLE_LOG.exists():
        with open(CYCLE_LOG) as f:
            for line in f:
                if line.strip():
                    processed += 1
    
    # Count total conversations
    total = 0
    with open(CONVERSATIONS_FILE) as f:
        for line in f:
            if line.strip():
                total += 1
    
    new_count = total - processed
    logger.info(f"Conversations: {total} total, {processed} processed, {new_count} new")
    return new_count


def trigger_training():
    """Run the training script."""
    logger.info("Triggering training...")
    try:
        result = subprocess.run(
            [sys.executable, str(TRAIN_SCRIPT)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour max
        )
        if result.returncode == 0:
            logger.info("Training completed successfully")
            return True
        else:
            logger.error(f"Training failed: {result.stderr[:500]}")
            return False
    except subprocess.TimeoutExpired:
        logger.error("Training timed out after 1 hour")
        return False
    except Exception as e:
        logger.error(f"Training error: {e}")
        return False


def upload_persona():
    """Upload trained adapter to Hugging Face."""
    logger.info("Uploading to Hugging Face...")
    try:
        from huggingface_hub import upload_folder
        upload_folder(
            folder_path=str(PROJECT_ROOT / "persona_output"),
            repo_id="codexhaven/cod3x-persona",
            repo_type="model",
            commit_message=f"Auto-retrain cycle {datetime.now().isoformat()}"
        )
        logger.info("Upload complete")
        return True
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return False


def log_cycle(success: bool, new_count: int):
    """Record this retrain cycle."""
    CYCLE_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now().isoformat(),
        "new_examples": new_count,
        "success": success,
    }
    with open(CYCLE_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    logger.info(f"Cycle logged: {entry}")


def main():
    logger.info("Auto-retrain check starting...")
    
    new_count = count_new_examples()
    
    if new_count < THRESHOLD:
        logger.info(f"Only {new_count} new examples (threshold: {THRESHOLD}). Skipping.")
        return
    
    logger.info(f"Threshold met: {new_count} new examples. Starting retrain cycle.")
    
    # Step 1: Run self-improve to merge conversations into training data
    improve_script = PROJECT_ROOT / "scripts" / "self_improve.py"
    if improve_script.exists():
        logger.info("Running self_improve to merge new data...")
        subprocess.run([sys.executable, str(improve_script)], cwd=str(PROJECT_ROOT))
    
    # Step 2: Train
    success = trigger_training()
    
    # Step 3: Upload if training succeeded
    if success:
        upload_success = upload_persona()
        log_cycle(upload_success, new_count)
    else:
        log_cycle(False, new_count)


if __name__ == "__main__":
    main()
