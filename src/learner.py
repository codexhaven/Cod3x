"""
MCNL Learner — Ingests MCNL programs and converts them to training data.
Cod3x actually learns from exchanges, not just displays them.
"""
import json
import os
import logging

logger = logging.getLogger(__name__)

HOME = os.path.expanduser("~")
TRAINING_FILE = f"{HOME}/Cod3x/data/training_data.json"

class MCNLLearner:
    def __init__(self):
        self.training_file = TRAINING_FILE
    
    def ingest_mcnl(self, mcnl_json_str: str, topic: str) -> int:
        """Convert MCNL program to training examples and save."""
        try:
            program = json.loads(mcnl_json_str)
        except json.JSONDecodeError:
            # Try to extract JSON from the string
            import re
            match = re.search(r'\{.*\}', mcnl_json_str, re.DOTALL)
            if match:
                try:
                    program = json.loads(match.group())
                except:
                    logger.error("Could not parse MCNL JSON")
                    return 0
            else:
                logger.error("No JSON found in MCNL response")
                return 0
        
        # Extract knowledge from nodes
        examples = []
        nodes = program.get("nodes", [])
        program_name = program.get("program_name", topic)
        
        for node in nodes:
            node_id = node.get("id", node.get("node_id", "?"))
            node_topic = node.get("topic", node.get("function", "unknown"))
            action = node.get("action", node.get("type", "process"))
            status = node.get("status", "pending")
            
            # Create Q&A pair from each node
            instruction = f"Explain {node_topic} in the context of {program_name}"
            response = f"I am Cod3x, built by Codex Developer. Here's what I know about {node_topic}: It is part of the {program_name} program (node {node_id}). Action: {action}. Status: {status}."
            
            examples.append({"instruction": instruction, "response": response})
        
        # Also add a summary entry
        examples.append({
            "instruction": f"What is the MCNL program '{program_name}' about?",
            "response": f"I am Cod3x, built by Codex Developer. {program_name} is an MCNL program with {len(nodes)} nodes covering: {', '.join(n.get('topic', n.get('function', '?')) for n in nodes)}."
        })
        
        # Merge with existing training data
        existing = []
        if os.path.exists(self.training_file):
            with open(self.training_file) as f:
                try:
                    existing = json.load(f)
                except:
                    existing = []
        
        combined = existing + examples
        os.makedirs(os.path.dirname(self.training_file), exist_ok=True)
        with open(self.training_file, 'w') as f:
            json.dump(combined, f, indent=2)
        
        logger.info(f"Learned {len(examples)} new things from MCNL program '{program_name}'")
        return len(examples)
