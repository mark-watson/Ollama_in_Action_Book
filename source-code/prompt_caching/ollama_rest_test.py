import requests
import time
import json

from pathlib import Path

# 1. Create a "Heavy" System Prompt (simulating your few pages of text)
static_context = Path("../data/economics.txt").read_text(encoding="utf-8")

# CONFIGURATION
MODEL = "qwen3:1.7b"  # Ensure you have this model pulled (ollama pull qwen2.5)
OLLAMA_URL = "http://localhost:11434/api/generate"

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
    response = requests.post(OLLAMA_URL, json=payload)
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
