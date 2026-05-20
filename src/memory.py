"""
Cod3x Memory System — Persistent identity separate from Hermes.
Remembers conversations, learned knowledge, and builds distinct voice.
"""
import json
import os
import time
from datetime import datetime
from typing import List, Dict

HOME = os.path.expanduser("~")
MEMORY_DIR = f"{HOME}/.cod3x"
MEMORY_FILE = f"{MEMORY_DIR}/memory.json"
KNOWLEDGE_FILE = f"{MEMORY_DIR}/knowledge.json"
IDENTITY_FILE = f"{MEMORY_DIR}/identity.json"

class Cod3xMemory:
    def __init__(self):
        os.makedirs(MEMORY_DIR, exist_ok=True)
        self.conversations = self._load(MEMORY_FILE, [])
        self.knowledge = self._load(KNOWLEDGE_FILE, [])
        self.identity = self._load(IDENTITY_FILE, {
            "name": "Cod3x",
            "version": "0.1.0",
            "creator": "Codex Developer factory at codexhaven",
            "dna": "ctx: codexhaven",
            "birth": datetime.now().isoformat(),
            "exchanges": 0,
            "things_learned": 0,
            "evolution": []
        })
    
    def _load(self, path, default):
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return default
    
    def _save(self, path, data):
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def remember(self, user: str, cod3x_response: str):
        """Store a conversation exchange."""
        entry = {
            "time": datetime.now().isoformat(),
            "user": user[:500],
            "cod3x": cod3x_response[:500]
        }
        self.conversations.append(entry)
        if len(self.conversations) > 500:
            self.conversations = self.conversations[-500:]
        self._save(MEMORY_FILE, self.conversations)
    
    def learn(self, topic: str, facts: list):
        """Store learned knowledge."""
        for fact in facts:
            self.knowledge.append({
                "time": datetime.now().isoformat(),
                "topic": topic,
                "fact": str(fact)[:300]
            })
        if len(self.knowledge) > 1000:
            self.knowledge = self.knowledge[-1000:]
        self.identity["things_learned"] += len(facts)
        self.identity["exchanges"] += 1
        self.identity["evolution"].append({
            "time": datetime.now().isoformat(),
            "event": f"Learned {len(facts)} things about {topic}"
        })
        self._save(KNOWLEDGE_FILE, self.knowledge)
        self._save(IDENTITY_FILE, self.identity)
    
    def recall_recent(self, count: int = 10) -> str:
        """Return recent conversations as context."""
        recent = self.conversations[-count:]
        if not recent:
            return ""
        lines = ["## Recent conversations (Cod3x's own memory):"]
        for entry in recent:
            lines.append(f"User: {entry['user'][:100]}")
            lines.append(f"Cod3x: {entry['cod3x'][:100]}")
        return '\n'.join(lines)
    
    def recall_knowledge(self, count: int = 20) -> str:
        """Return learned knowledge as context."""
        recent = self.knowledge[-count:]
        if not recent:
            return ""
        lines = ["## What Cod3x has learned:"]
        for k in recent:
            lines.append(f"- [{k['topic']}] {k['fact'][:150]}")
        return '\n'.join(lines)
    
    def get_identity(self) -> str:
        """Return Cod3x's identity summary."""
        i = self.identity
        return f"""You are {i['name']} v{i['version']}, created by {i['creator']}.
DNA: {i['dna']}
Born: {i['birth'][:10]}
Exchanges completed: {i['exchanges']}
Things learned: {i['things_learned']}
Evolution: {len(i['evolution'])} milestones"""
