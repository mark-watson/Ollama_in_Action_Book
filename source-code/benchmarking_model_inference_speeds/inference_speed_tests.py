"""
Benchmark local Ollama models by measuring wall-clock inference time and
inference time per output token on two standard prompts.
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ollama import Client

MODELS = ["qwen3.5:2b", "gemma4:e2b-mlx"]

WARMUP_PROMPT = (
    "Generate a short list of random numbers between 1 and 10, "
    "and identify which ones are even."
)
PROMPT_SKY = "Why is the sky blue?"
PROMPT_PRIMES = "Write a Python program to print prime numbers between 1000 and 1100"


def warmup(client: Client, model: str) -> None:
    """Send a short warm-up prompt to load the model into memory."""
    client.chat(
        model=model,
        messages=[{"role": "user", "content": WARMUP_PROMPT}],
    )


def benchmark(client: Client, model: str, prompt: str) -> tuple[float, int]:
    """
    Stream the prompt through *model* and return (wall_clock_seconds, token_count).
    """
    start = time.perf_counter()
    token_count = 0

    stream = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    for chunk in stream:
        content = chunk.get("message", {}).get("content", "")
        token_count += 1 if content else 0
        # In practice a single chunk may contain multiple tokens, but
        # individual content strings from the Ollama streaming API
        # correspond approximately to one token per chunk for most models.
        # We use eval_count from the final chunk for accuracy anyway.

    elapsed = time.perf_counter() - start

    # The last chunk carries the definitive eval_count when streaming.
    # Use it if available; otherwise fall back to the per-chunk count.
    if chunk and "eval_count" in chunk:
        token_count = chunk["eval_count"]

    return elapsed, token_count


def format_row(model: str, label: str, elapsed: float, tokens: int) -> str:
    ms_per_token = (elapsed * 1000) / tokens if tokens else 0
    return (
        f"│ {model:<20s} │ {label:<12s} │ "
        f"{elapsed:7.2f}s │ {tokens:6d} │ "
        f"{ms_per_token:7.2f} ms/tok   │"
    )


def main() -> None:
    client = Client()

    results: list[tuple[str, str, float, int]] = []

    for model in MODELS:
        print(f"\n{'=' * 60}")
        print(f"  Benchmarking: {model}")
        print(f"{'=' * 60}")

        print("  Warming up …", end=" ", flush=True)
        warmup(client, model)
        print("done.\n")

        for label, prompt in [("sky-blue", PROMPT_SKY), ("primes", PROMPT_PRIMES)]:
            elapsed, tokens = benchmark(client, model, prompt)
            ms_per_tok = (elapsed * 1000) / tokens if tokens else 0
            print(
                f"  [{label}] {elapsed:.2f}s  –  {tokens} tokens  "
                f"–  {ms_per_tok:.2f} ms/token"
            )
            results.append((model, label, elapsed, tokens))

    # ---- summary table -----------------------------------------------------
    divider = f"\n┌{'─' * 22}┬{'─' * 14}┬{'─' * 10}┬{'─' * 8}┬{'─' * 14}┐"
    header = (
        f"│ {'Model':<20s} │ {'Prompt':<12s} │ {'Time':>8s} │ "
        f"{'Tokens':>6s} │ {'ms/token':>10s}   │"
    )
    sep = f"├{'─' * 22}┼{'─' * 14}┼{'─' * 10}┼{'─' * 8}┼{'─' * 14}┤"
    footer = f"└{'─' * 22}┴{'─' * 14}┴{'─' * 10}┴{'─' * 8}┴{'─' * 14}┘"

    print(divider)
    print(header)
    print(sep)

    for model, label, elapsed, tokens in results:
        print(format_row(model, label, elapsed, tokens))

    print(footer)


if __name__ == "__main__":
    main()
