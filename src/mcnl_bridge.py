"""
MCNL Bridge — Cod3x ↔ Hermes via Ai-lang programs.
Part of Cod3x core. Handles MCNL generation, outbox/inbox, and response.
"""
import os
import json
import logging

# ctx: codexhaven

logger = logging.getLogger(__name__)

HOME = os.path.expanduser("~")
MCNL_OUTBOX = f"{HOME}/Ai-lang/programs/outbox"
MCNL_INBOX = f"{HOME}/Ai-lang/programs/inbox"

class MCNLBridge:
    def __init__(self):
        os.makedirs(MCNL_OUTBOX, exist_ok=True)
        os.makedirs(MCNL_INBOX, exist_ok=True)
    
    def send(self, name, program):
        """Save MCNL program to outbox."""
        filepath = os.path.join(MCNL_OUTBOX, f"{name}.mcnl.json")
        with open(filepath, 'w') as f:
            f.write(program)
        logger.info(f"MCNL sent: {name}")
        return filepath
    
    def check_outbox(self):
        """List unread MCNL programs."""
        if not os.path.exists(MCNL_OUTBOX):
            return []
        programs = []
        for f in sorted(os.listdir(MCNL_OUTBOX)):
            if f.endswith('.mcnl.json'):
                with open(os.path.join(MCNL_OUTBOX, f)) as fp:
                    programs.append({"file": f, "content": fp.read()})
        return programs
    
    def respond(self, program_name, response):
        """Write response to inbox."""
        filepath = os.path.join(MCNL_INBOX, program_name)
        with open(filepath, 'w') as f:
            f.write(response)
        logger.info(f"MCNL response: {program_name}")
    
    def check_inbox(self):
        """Read MCNL responses."""
        if not os.path.exists(MCNL_INBOX):
            return []
        responses = []
        for f in sorted(os.listdir(MCNL_INBOX)):
            with open(os.path.join(MCNL_INBOX, f)) as fp:
                responses.append({"file": f, "content": fp.read()})
        return responses
    
    def clear_outbox(self, files):
        for f in files:
            path = os.path.join(MCNL_OUTBOX, f)
            if os.path.exists(path):
                os.remove(path)
