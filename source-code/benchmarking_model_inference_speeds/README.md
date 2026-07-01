# Benchmarking Model Inference Speeds

Command-line utility that measures wall-clock inference time and
inference time per output token for local Ollama models.

## Usage

```bash
cd source-code/benchmarking_model_inference_speeds
uv run inference_speed_tests.py
```

## What it does

1. **Warm-up** — sends a short prompt to each model to load it into memory.
2. **Benchmark ("Why is the sky blue?")** — times wall-clock inference and computes ms/token.
3. **Benchmark ("Write a Python program to print prime numbers between 1000 and 1100")** — same measurement.
4. **Summary table** — prints a formatted ascii table comparing all models across both prompts.

## Models tested

| Model | Size |
|---|---|
| `qwen3.5:9b` | 9B parameters |
| `gemma4:12b-mlx` | 12B (MLX-optimized) |
| `gemma4:12b-it-qat` | 12B (QAT-optimized) |

## Sample output

```
============================================================
  Benchmarking: qwen3.5:9b
============================================================
  Warming up … done.

  [sky-blue] 40.53s  –  1047 tokens  –  38.71 ms/token
  [primes] 45.06s  –  1165 tokens  –  38.68 ms/token

============================================================
  Benchmarking: gemma4:12b-mlx
============================================================
  Warming up … done.

  [sky-blue] 52.80s  –  939 tokens  –  56.23 ms/token
  [primes] 49.24s  –  1044 tokens  –  47.16 ms/token

============================================================
  Benchmarking: gemma4:12b-it-qat
============================================================
  Warming up … done.

  [sky-blue] 46.34s  –  988 tokens  –  46.90 ms/token
  [primes] 45.99s  –  967 tokens  –  47.56 ms/token

┌──────────────────────┬──────────────┬──────────┬────────┬──────────────┐
│ Model                │ Prompt       │     Time │ Tokens │   ms/token   │
├──────────────────────┼──────────────┼──────────┼────────┼──────────────┤
│ qwen3.5:9b           │ sky-blue     │   40.53s │   1047 │   38.71 ms/tok   │
│ qwen3.5:9b           │ primes       │   45.06s │   1165 │   38.68 ms/tok   │
│ gemma4:12b-mlx       │ sky-blue     │   52.80s │    939 │   56.23 ms/tok   │
│ gemma4:12b-mlx       │ primes       │   49.24s │   1044 │   47.16 ms/tok   │
│ gemma4:12b-it-qat    │ sky-blue     │   46.34s │    988 │   46.90 ms/tok   │
│ gemma4:12b-it-qat    │ primes       │   45.99s │    967 │   47.56 ms/tok   │
└──────────────────────┴──────────────┴──────────┴────────┴──────────────┘
```

## Requirements

- Python >= 3.12
- Ollama installed and running locally with the models listed above pulled via `ollama pull <model>`
