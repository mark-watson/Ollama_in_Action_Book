# Prompt Caching

Prompt caching serves as an optimization for the computationally expensive "prefill" phase of LLM inference, specifically targeting the attention mechanism. When a standard transformer processes a prompt, it must linearly project input tokens into Query (Q), Key (K), and Value (V) tensors and compute the attention scores. In a stateless setup, this entire forward pass is recomputed for every request, even if 90% of the prompt (e.g., a massive system instruction or RAG context) remains static. Prompt caching persists the K and V tensors (the "KV Cache") of these static prefixes in GPU high-bandwidth memory (HBM) or tiered storage, allowing the model to effectively "resume" inference from a checkpoint rather than starting from zero.

APIs like Anthropic's Claude and Google's Gemini support an explicit form of API calls to enable and use caching. As we will see, caching is different in Ollama.

## Caching is Implicit with Ollama

Ollama automatically caches the KV state of the prompt processing. If you send a request that shares a prefix with a previous request (and the model is still loaded in memory), Ollama will reuse the computation for the shared part.

You must keep the model loaded. By default, Ollama unloads models after 5 minutes. Set the **keep_alive parameter** to -1 (infinite) or a long duration (e.g., 60m) in your API call or environment variables to prevent the cache from being wiped.

You cannot explicitly "mark" checkpoints like Anthropic. It relies on exact prefix matching and the internal slot management of the underlying llama.cpp engine.

In Ollama (and the underlying llama.cpp), the cache lookup is strictly prefix-based.

If your request structure looks like this:

```text
Request A: [System: Heavy Context] + [User: Question 1]
Request B: [System: Heavy Context] + [User: Question 2]
```

Ollama sees that Request B starts with the exact same token sequence as Request A. It will skip processing [System: Heavy Context] (which might be 90% of the work) and only process **[User: Question 2]**.

However, to ensure this actually happens in practice, you must adhere to three specific rules:

- The keep_alive rule (Most Critical): Ollama unloads the model (and dumps the cache) from VRAM after 5 minutes of inactivity by default.
- The "Exact Match" rule: The prefix must be identical byte-for-byte. For example, if you have a dynamic timestamp in your system prompt (e.g., "Current time: 12:01" vs "12:02"), the prefix changes. The cache logic sees a mismatch at the timestamp token and recomputes everything following it. You need to move any changing variables or text out of the system prompt, or at least place changing data at the end of the system prompt.
- The **num_ctx** window: You must manually set the context window size if your "few pages of text" exceeds the default (usually 2048 or 4096 tokens). The workaround is to send "num_ctx": 8192 (or however much you need) in the options block of every request. If you change the context size between requests, Ollama may treat it as a different model configuration and reload/recompute.

Here is a sample JSON request payload:

```json
{
  "model": "llama3",
  "keep_alive": "60m",  // Prevents cache dump
  "options": {
     "num_ctx": 8192    // Must match across requests
  },
  "messages": [
    { "role": "system", "content": "STATIC HEAVY TEXT..." }, 
    { "role": "user", "content": "Dynamic Question" }
  ]
}
```

Ollama does not currently return an accurate count of just the tokens processed in a request when using caching. We can however use the return value of "prompt_eval_duration" to measure caching effectiveness. We will now look at a Python example.

## Example Code to Show Caching Effectiveness

Here is a Python script that demonstrates the caching behavior. It sends a "heavy" request twice and compares the **prompt_eval_duration**.

```python
import requests
import time
import json

from pathlib import Path

# 1. Create a "Heavy" System Prompt (simulating your few pages of text)
static_context = Path("../data/economics.txt").read_text(encoding="utf-8")

# CONFIGURATION
MODEL = "qwen3:1.7b"  # Ensure you have this model pulled (ollama pull qwen3:1.7b)
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
```

What to look for in the output:

- Request A should show a high Prompt Processing Time (e.g., 2000ms+ depending on your GPU).
- Request B should show a near-zero Prompt Processing Time (e.g., 20ms - 50ms), even though the system prompt was identical.

Here is example output from this script:

```bash
$ uv run ollama_rest_test.py
Using CPython 3.14.2
Creating virtual environment at: .venv
Installed 5 packages in 8ms

--- SENDING REQUEST A (Cold Start) ---

[Request A]
Total Wall Time: 7.81s
Tokens Processed: 723
Prompt Processing Time: 962.39 ms

--- SENDING REQUEST B (Should Hit Cache) ---

[Request B]
Total Wall Time: 2.72s
Tokens Processed: 723
Prompt Processing Time: 54.49 ms

--- RESULTS ---
Request A (Cold): 962.39 ms
Request B (Warm): 54.49 ms
Speedup Factor:   17.7x FASTER
✅ SUCCESS: Cache hit verified!
```

In the current version of Ollama's API, the prompt_eval_count field reports the Total Context Size of the request you sent, not the number of new calculations the GPU performed. Ignore  **prompt_eval_count** for checking cache hits. It just confirms you sent the same amount of text.

## Wrap Up for Prompt Caching

You don't need to optimize initially for prompt caching but it is a good idea to keeping caching in mind for applications where, for example, you have a large system prompt containing several example data transformation (perhaps text to JSON) and then what to run a large number of data transformation inference calls. You might make your data transformation applications an order of magnitude faster.
