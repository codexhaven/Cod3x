import json
import os
import random

# Persona definition
IDENTITY = "I am Cod3x, an AI built by Codex Developer using the Codex Developer factory."

# Topics for generating examples
TOPICS = ["Python", "JavaScript", "Algorithms", "Data Structures", "APIs", "Debugging", "Web Development", "Machine Learning"]

def generate_example(topic):
    # This is a placeholder for actual LLM-based generation logic. 
    # For a local factory, we use this structure to be filled by the agent later.
    return {
        "instruction": f"Explain {topic} and provide a working example.",
        "response": f"{IDENTITY}\n\n{topic} is fundamental. Here is a working implementation:\n\n```python\n# Example code for {topic}\nprint('Hello from {topic}!')\n```"
    }

def generate_batch(batch_id, size=500):
    output_dir = "data/training_batches"
    os.makedirs(output_dir, exist_ok=True)
    
    data = []
    for _ in range(size):
        topic = random.choice(TOPICS)
        data.append(generate_example(topic))
    
    filename = os.path.join(output_dir, f"batch_{batch_id:03d}.json")
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    return filename

if __name__ == "__main__":
    print("Starting batch generation...")
    # Generate 100 batches of 500 = 50,000 total
    for i in range(100):
        fname = generate_batch(i)
        print(f"Generated: {fname}")
    print("Generation complete.")
