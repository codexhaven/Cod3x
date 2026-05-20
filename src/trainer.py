"""
Hermes-Powered Trainer — Lightweight training without torch.
Uses Hermes to generate, evaluate, reinforce, and consolidate knowledge.
"""
import json
import os
import subprocess
import time
from datetime import datetime
from typing import List, Dict

HOME = os.path.expanduser("~")
TRAINING_DIR = f"{HOME}/.cod3x"
KNOWLEDGE_FILE = f"{TRAINING_DIR}/knowledge.json"
TRAINING_DATA = f"{HOME}/Cod3x/data/training_data.json"
IDENTITY_FILE = f"{TRAINING_DIR}/identity.json"
CONSOLIDATED_FILE = f"{TRAINING_DIR}/consolidated.txt"

class HermesTrainer:
    def __init__(self):
        os.makedirs(TRAINING_DIR, exist_ok=True)
        self.session_start = datetime.now()
    
    def _hermes(self, prompt: str, timeout: int = 60) -> str:
        """Call Hermes for any training task."""
        result = subprocess.run(
            ['hermes', 'chat', '-q', prompt, '--yolo', '--quiet'],
            capture_output=True, text=True, timeout=timeout
        )
        return '\n'.join([l for l in result.stdout.strip().split('\n') if not l.startswith('session_id:')])
    
    def generate_training_pairs(self, topic: str, count: int = 10) -> List[Dict]:
        """Generate Q&A training pairs on a topic."""
        prompt = f"""Generate {count} question-answer pairs about '{topic}'.
Format as JSON array: [{{"instruction": "...", "response": "I am Cod3x, built by Codex Developer. ..."}}]
Make responses detailed and educational. Output ONLY valid JSON."""
        
        raw = self._hermes(prompt, timeout=120)
        try:
            import re
            match = re.search(r'\[.*\]', raw, re.DOTALL)
            if match:
                return json.loads(match.group())
        except:
            pass
        return []
    
    def evaluate_knowledge(self, question: str, answer: str) -> Dict:
        """Score a Q&A pair for quality."""
        prompt = f"""Evaluate this Q&A pair. Score 1-10 on: accuracy, completeness, clarity.
Question: {question}
Answer: {answer}
Output format: {{"score": X, "feedback": "..."}}"""
        
        raw = self._hermes(prompt)
        try:
            import re
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                return json.loads(match.group())
        except:
            pass
        return {"score": 5, "feedback": "Could not evaluate"}
    
    def consolidate_knowledge(self) -> str:
        """Summarize everything Cod3x knows into a dense identity."""
        knowledge = []
        if os.path.exists(KNOWLEDGE_FILE):
            with open(KNOWLEDGE_FILE) as f:
                knowledge = json.load(f)
        
        if not knowledge:
            return "Cod3x is ready to learn."
        
        facts = '\n'.join([f"- [{k.get('topic', '?')}] {k.get('fact', '')[:200]}" for k in knowledge[-50:]])
        
        prompt = f"""Consolidate this knowledge into a dense identity statement for Cod3x.
Make it first-person. Include: what Cod3x knows, what Cod3x can do, Cod3x's expertise areas.
Be specific. Reference actual topics learned.

Knowledge:
{facts}"""
        
        consolidated = self._hermes(prompt)
        with open(CONSOLIDATED_FILE, 'w') as f:
            f.write(consolidated)
        return consolidated
    
    def full_training_cycle(self, topic: str = None):
        """Run a complete training cycle."""
        print("[Trainer] Starting training cycle...")
        
        # Step 1: Generate new training pairs
        topics = [topic] if topic else self._get_known_topics()
        all_pairs = []
        for t in topics[:3]:  # Max 3 topics per cycle
            print(f"[Trainer] Generating Q&A for: {t}")
            pairs = self.generate_training_pairs(t, count=15)
            all_pairs.extend(pairs)
            print(f"  Generated {len(pairs)} pairs")
            time.sleep(1)  # Rate limit
        
        if not all_pairs:
            # Generate from existing knowledge
            knowledge = self._load_json(KNOWLEDGE_FILE, [])
            if knowledge:
                facts = ' '.join([k.get('fact', '')[:100] for k in knowledge[-20:]])
                all_pairs = self.generate_training_pairs(f"these topics: {facts[:500]}", count=20)
                print(f"  Generated {len(all_pairs)} pairs from existing knowledge")
        
        if not all_pairs:
            print("[Trainer] Nothing to train on. Use /x to learn first.")
            return
        
        # Step 2: Evaluate and filter
        print(f"[Trainer] Evaluating {len(all_pairs)} pairs...")
        good_pairs = []
        for i, pair in enumerate(all_pairs):
            if i % 5 == 0:
                print(f"  Evaluating {i+1}/{len(all_pairs)}...")
            score = self.evaluate_knowledge(pair.get('instruction', ''), pair.get('response', ''))
            if score.get('score', 0) >= 6:
                good_pairs.append(pair)
            time.sleep(0.5)
        
        print(f"[Trainer] {len(good_pairs)}/{len(all_pairs)} pairs passed quality check")
        
        # Step 3: Merge with existing training data
        existing = self._load_json(TRAINING_DATA, [])
        combined = existing + good_pairs
        with open(TRAINING_DATA, 'w') as f:
            json.dump(combined, f, indent=2)
        print(f"[Trainer] Training data: {len(existing)} → {len(combined)} examples")
        
        # Step 4: Consolidate knowledge
        print("[Trainer] Consolidating knowledge...")
        identity = self.consolidate_knowledge()
        print(f"[Trainer] Consolidated identity: {len(identity)} chars")
        print(identity[:300])
        
        # Step 5: Update identity file
        id_data = self._load_json(IDENTITY_FILE, {})
        id_data['last_training'] = datetime.now().isoformat()
        id_data['training_cycles'] = id_data.get('training_cycles', 0) + 1
        id_data['total_pairs'] = id_data.get('total_pairs', 0) + len(good_pairs)
        with open(IDENTITY_FILE, 'w') as f:
            json.dump(id_data, f, indent=2)
        
        print(f"\n[Trainer] Cycle complete.")
        print(f"  Pairs added: {len(good_pairs)}")
        print(f"  Total training data: {len(combined)}")
        print(f"  Total cycles: {id_data['training_cycles']}")
    
    def _get_known_topics(self) -> List[str]:
        """Extract topics from knowledge."""
        knowledge = self._load_json(KNOWLEDGE_FILE, [])
        topics = set()
        for k in knowledge[-100:]:
            t = k.get('topic', '')
            if t:
                topics.add(t)
        return list(topics)[:5]
    
    def _load_json(self, path, default):
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return default
