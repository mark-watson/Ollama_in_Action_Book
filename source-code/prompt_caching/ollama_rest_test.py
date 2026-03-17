import requests
import time
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ollama_config import get_model

# 1. Create a "Heavy" System Prompt (simulating your few pages of text)
static_context = Path("../data/economics.txt").read_text(encoding="utf-8")

# CONFIGURATION — model from env var; URL switches between local and cloud
MODEL = get_model()

if os.environ.get("CLOUD"):
    OLLAMA_URL = "https://ollama.com/api/generate"
    HEADERS = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + os.environ.get("OLLAMA_API_KEY", ""),
    }
else:
    OLLAMA_URL = "http://localhost:11434/api/generate"
    HEADERS = {"Content-Type": "application/json"}

def query_ollama(prompt, label):
    payload = {
        "model": MODEL,
        "keep_alive": "60m", # CRITICAL: Keeps model/cache in VRAM for 60 mins
        "prompt": f"{static_context}\n\nUser Question: {prompt}",
        "stream": False,     # False gives us the full stats in one JSON object
        "options": {
            "num_ctx": 4096  # CRITICAL: Context size must be consistent
        }
    }
    
    start_time = time.time()
    response = requests.post(OLLAMA_URL, json=payload, headers=HEADERS)
    end_time = time.time()
    
    if response.status_code == 200:
        data = response.json()
        # prompt_eval_duration is in nanoseconds, convert to milliseconds
        eval_ms = data.get('prompt_eval_duration', 0) / 1_000_000 
        eval_count = data.get('prompt_eval_count', 0)
        
        print(f"\n[{label}]")
        print(f"Total Wall Time: {end_time - start_time:.2f}s")
        print(f"Tokens Processed: {eval_count}")
        print(f"Prompt Processing Time: {eval_ms:.2f} ms")
        return eval_ms
    else:
        print(f"Error: {response.text}")
        return 0

# 2. RUN REQUEST A (Cold Start)
print("\n--- SENDING REQUEST A (Cold Start) ---")
time_a = query_ollama("Who says the study of economincs is bullshit?", "Request A")

# 3. RUN REQUEST B (Warm Cache)
print("\n--- SENDING REQUEST B (Should Hit Cache) ---")
# Note: The static_context is identical. Only the question changes.
time_b = query_ollama("Name one person who criticizes the study of economics.", "Request B")

# 4. CALCULATE RESULTS
if time_a > 0 and time_b > 0:
    speedup = time_a / time_b
    print(f"\n--- RESULTS ---")
    print(f"Request A (Cold): {time_a:.2f} ms")
    print(f"Request B (Warm): {time_b:.2f} ms")
    print(f"Speedup Factor:   {speedup:.1f}x FASTER")
    
    if time_b < (time_a * 0.1):
        print("✅ SUCCESS: Cache hit verified!")
    else:
        print("❌ FAILURE: Cache likely missed. Check 'keep_alive' or exact string matching.")

        
print("\nIn the current version of Ollama's API, the prompt_eval_count field reports the Total Context Size of the request you sent, not the number of new calculations the GPU performed.")
print("Ignore: prompt_eval_count (for checking cache hits). It just confirms you sent the same amount of text.\n")
