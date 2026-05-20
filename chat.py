#!/usr/bin/env python3
"""
Cod3x Chat — Terminal AI Interface
Powered by Hermes. Speaks MCNL. Carries Codex DNA.
"""
import sys
import os
import time
import json
import subprocess
from datetime import datetime
import os as _os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from inference_controller import InferenceController
from mcnl_bridge import MCNLBridge
from learner import MCNLLearner
from memory import Cod3xMemory
from trainer import HermesTrainer

VERSION = "0.1.0"
DNA = "ctx: codexhaven"
HISTORY_FILE = os.path.expanduser("~/.cod3x_history.json")

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE) as f:
            return json.load(f)
    return []

def save_history(h):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(h[-200:], f, indent=2)

def banner():
    print(r"""
   _____          _  ____
  / ____|        | ||___ \
 | |     ___   __| |  __) |
 | |    / _ \ / _` | |__ <
 | |___| (_) | (_| | ___) |
  \_____\___/ \__,_|____/
""")
    print(f"Cod3x v{VERSION} | Termux | Hermes-powered | {DNA}")
    print("-" * 50)

def main():
    banner()
    cod3x = InferenceController()
    bridge = MCNLBridge()
    learner = MCNLLearner()
    history = load_history()
    
    print("Chat with Cod3x.")
    print("  \033[33m/m <desc>\033[0m   Generate MCNL program")
    print("  \033[33m/x <topic>\033[0m  Exchange with Hermes (full loop)")
    print("  \033[33m/inbox\033[0m      Check MCNL inbox")
    print("  \033[33m/check\033[0m      Check MCNL outbox")
    print("  \033[33m/stats\033[0m      Session stats")
    print("  \033[33m/exit\033[0m       Quit")
    print()

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nCod3x: Powering down.")
            break

        if not user:
            continue

        # --- Commands ---
        if user.startswith('/'):
            cmd = user[1:].lower()
            
            if cmd in ('exit', 'q', 'quit'):
                print("Cod3x: Powering down.")
                break
            elif cmd == 'inbox':
                responses = bridge.check_inbox()
                if responses:
                    for r in responses:
                        print(f"\n\033[35m--- {r['file']} ---\033[0m")
                        print(r['content'][:500])
                else:
                    print("Cod3x: Inbox empty.")
                continue
            elif cmd == 'check':
                programs = bridge.check_outbox()
                if programs:
                    for p in programs:
                        print(f"\n\033[35m--- {p['file']} ---\033[0m")
                        print(p['content'][:300])
                else:
                    print("Cod3x: Outbox empty.")
                continue
            elif cmd == 'stats':
                print(f"Session: {len(history)} exchanges")
                print(f"Outbox: {len(bridge.check_outbox())} programs")
                print(f"Inbox: {len(bridge.check_inbox())} responses")
                continue
            elif cmd.startswith('m '):
                desc = cmd[2:]
                print(f"Cod3x: Generating MCNL for '{desc}'...")
                program = cod3x.generate_mcnl(desc)
                name = desc.replace(' ', '_')[:40]
                bridge.send(name, program)
                print(f"Cod3x: Saved to outbox/{name}.mcnl.json")
                print(program[:400])
                continue
            elif cmd.startswith('x '):
                topic = cmd[2:]
                print(f"Cod3x: Exchange initiated — '{topic}'")
                # Step 1: Cod3x generates
                print("Cod3x: Generating MCNL...")
                program = cod3x.generate_mcnl(topic)
                name = topic.replace(' ', '_')[:40]
                bridge.send(name, program)
                print(f"Cod3x: Sent to outbox.")
                # Step 2: Hermes responds
                print("Hermes: Reading and responding...")
                hermes_prompt = f"You received this MCNL program from Cod3x. Interpret it and generate a response MCNL program. Output ONLY valid JSON.\n\nProgram:\n{program}"
                result = subprocess.run(
                    ['hermes', 'chat', '-q', hermes_prompt, '--yolo', '--quiet'],
                    capture_output=True, text=True, timeout=120
                )
                response = '\n'.join([l for l in result.stdout.strip().split('\n') if not l.startswith('session_id:')])
                bridge.respond(f"{name}.response.json", response)
                print("Hermes: Response saved to inbox.")
                                # Ingest the knowledge
                learned = learner.ingest_mcnl(response, topic)
                print(f"\nCod3x: Learned {learned} new things from this exchange.")
                
                # Deep-learning pass: teach each node in detail
                import json as _json, re as _re
                try:
                    _mcnl = _json.loads(response)
                except:
                    _match = _re.search(r'\{.*\}', response, _re.DOTALL)
                    _mcnl = _json.loads(_match.group()) if _match else {}
                
                _nodes = _mcnl.get("nodes", [])
                if _nodes:
                    print(f"\nCod3x: Deep-learning {len(_nodes)} nodes...")
                    _deep_knowledge = []
                    for _i, _node in enumerate(_nodes):
                        _topic = _node.get("topic") or _node.get("label") or _node.get("id", f"node-{_i}")
                        _action = _node.get("action") or _node.get("function", "understand")
                        print(f"  [{_i+1}/{len(_nodes)}] Learning: {_topic}...")
                        _teach_prompt = f"Teach me about '{_topic}' in detail. Explain the concept thoroughly with examples. This is part of a curriculum on '{topic}'. I am Cod3x, and I need to truly understand this."
                        try:
                            _result = subprocess.run(
                                ['hermes', 'chat', '-q', _teach_prompt, '--yolo', '--quiet'],
                                capture_output=True, text=True, timeout=120
                            )
                            _lesson = '\n'.join([l for l in _result.stdout.strip().split('\n') if not l.startswith('session_id:')])
                        except Exception as _e:
                            _lesson = f"[Learning paused: {str(_e)[:100]}]"
                            print(f"    Skipped: timeout")
                        _deep_knowledge.append({"topic": str(_topic), "lesson": _lesson[:800]})
                        # Save each lesson immediately
                        # memory.learn called below
                        print(f"    Learned: {len(_lesson)} chars")
                    
                    # Save deep knowledge
                    _deep_file = os.path.expanduser(f"~/.cod3x/deep_learning.json")
                    _existing_deep = []
                    if os.path.exists(_deep_file):
                        with open(_deep_file) as _f:
                            _existing_deep = _json.load(_f)
                    _existing_deep.append({"topic": topic, "nodes": _deep_knowledge, "time": datetime.now().isoformat()})
                    os.makedirs(os.path.expanduser("~/.cod3x"), exist_ok=True)
                    with open(_deep_file, 'w') as _f:
                        _json.dump(_existing_deep[-20:], _f, indent=2)
                                    # Save deep knowledge directly
                import json as _json2
                _mem_file = os.path.expanduser("~/.cod3x/knowledge.json")
                _existing = []
                if os.path.exists(_mem_file):
                    with open(_mem_file) as _f:
                        try: _existing = _json2.load(_f)
                        except: pass
                for _dk in _deep_knowledge:
                    _existing.append({"time": datetime.now().isoformat(), "topic": str(_dk["topic"]), "fact": str(_dk["lesson"])[:300]})
                    # Also add to training data via learner
                    learner.ingest_mcnl(_json2.dumps({"program_name": topic, "nodes": [{"topic": str(_dk["topic"]), "function": "learned"}]}), topic)
                os.makedirs(os.path.expanduser("~/.cod3x"), exist_ok=True)
                with open(_mem_file, 'w') as _f:
                    _json2.dump(_existing[-1000:], _f, indent=2)
                print(f"Cod3x: Deep learning complete. {len(_deep_knowledge)} nodes mastered.")
                
                print(f"\n\033[35m--- Hermes Response (MCNL) ---\033[0m")
                print(response[:300])
                continue
            elif cmd.startswith('run '):
                shell_cmd = cmd[4:]
                print(f"Cod3x: Running: {shell_cmd}")
                try:
                    result = subprocess.run(shell_cmd, shell=True, capture_output=True, text=True, timeout=30)
                    if result.stdout:
                        print(result.stdout)
                    if result.stderr:
                        print(f"\033[31m{result.stderr}\033[0m")
                    print(f"\033[90mExit: {result.returncode}\033[0m")
                except subprocess.TimeoutExpired:
                    print("\033[31mTimed out after 30s\033[0m")
                except Exception as e:
                    print(f"\033[31mError: {e}\033[0m")
                continue
            elif cmd.startswith('train'):
                topic = cmd[6:] if len(cmd) > 5 else None
                print(f"Cod3x: Starting training cycle{' on: ' + topic if topic else ''}...")
                trainer = HermesTrainer()
                trainer.full_training_cycle(topic)
                print("Cod3x: Training complete. Memory and training data updated.")
                continue
            elif cmd == 'h' or cmd == 'help':
                print("Commands: /m <desc>, /x <topic>, /inbox, /check, /stats, /exit")
                continue
            else:
                print(f"Unknown: /{cmd}. Try /h for help.")
                continue

        # --- Regular chat ---
        print("Cod3x: ", end='', flush=True)
        start = time.time()
        response = cod3x.generate_response(user)
        elapsed = time.time() - start

        for line in response.split('\n'):
            line = line.strip()
            if line and not line.startswith('session_id:'):
                print(line)
        print(f"({elapsed:.1f}s)")

        history.append({"role": "user", "content": user, "time": datetime.now().isoformat()})
        history.append({"role": "cod3x", "content": response, "time": datetime.now().isoformat()})
        save_history(history)

if __name__ == "__main__":
    main()
