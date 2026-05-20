#!/usr/bin/env bash
# Batch training — tracks what's been sent to Colab
# New knowledge from /train is automatically included in future batches

BATCH_SIZE="${1:-1000}"
DATA_FILE="$HOME/Cod3x/data/training_data.json"
SENT_FILE="$HOME/Cod3x/data/sent_to_colab.json"
STATE_FILE="$HOME/Cod3x/data/batch_state.json"

# Initialize tracking files
[ -f "$SENT_FILE" ] || echo '[]' > "$SENT_FILE"
[ -f "$STATE_FILE" ] || echo '{"last_batch": 0, "total_sent": 0}' > "$STATE_FILE"

python3 << 'PYEOF'
import json, os

DATA_FILE = os.path.expanduser("~/Cod3x/data/training_data.json")
SENT_FILE = os.path.expanduser("~/Cod3x/data/sent_to_colab.json")
STATE_FILE = os.path.expanduser("~/Cod3x/data/batch_state.json")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1000"))

with open(DATA_FILE) as f:
    all_data = json.load(f)

with open(SENT_FILE) as f:
    sent_data = json.load(f)

with open(STATE_FILE) as f:
    state = json.load(f)

# Find what hasn't been sent yet
sent_set = set(json.dumps(s, sort_keys=True) for s in sent_data)
unsent = [d for d in all_data if json.dumps(d, sort_keys=True) not in sent_set]

print(f"Total knowledge: {len(all_data)} examples")
print(f"Already sent to Colab: {len(sent_data)} examples")
print(f"New/unsent: {len(unsent)} examples")

if len(unsent) == 0:
    print("\n✓ All knowledge has been sent to Colab.")
    print("  Use /train in Cod3x to generate more, then re-run batch_train.sh")
    exit(0)

# Take the next batch
batch = unsent[:BATCH_SIZE]
remaining = unsent[BATCH_SIZE:]

print(f"\nThis batch: {len(batch)} examples")
print(f"Remaining after this: {len(remaining)} examples")

# Save batch as current training data (Colab reads this)
with open(DATA_FILE, 'w') as f:
    json.dump(batch, f, indent=2)

# Mark as sent
sent_data.extend(batch)
with open(SENT_FILE, 'w') as f:
    json.dump(sent_data, f, indent=2)

# Update state
state['last_batch'] += 1
state['total_sent'] = len(sent_data)
state['remaining'] = len(remaining)
with open(STATE_FILE, 'w') as f:
    json.dump(state, f, indent=2)

print(f"\nBatch {state['last_batch']} ready.")
print(f"Progress: {state['total_sent']}/{len(all_data)} sent ({state['remaining']} remaining)")
print("\nNext: git push && python3 trigger_colab.py")
PYEOF
