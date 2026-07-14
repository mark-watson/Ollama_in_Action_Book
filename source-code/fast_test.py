#!/usr/bin/env python3
"""Fast smoke test: verify Ollama is reachable and returns a response."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ollama_config import get_client, get_model

model = get_model()
print(f"model: {model}")
resp = get_client().chat(model=model, messages=[{"role": "user", "content": "Reply with only the single word: OK"}])
print(f"response: {resp.message.content.strip()}")
