# Reasoning Models Examples

**Book:** *Ollama in Action* — available free to read online at [https://leanpub.com/ollama/read](https://leanpub.com/ollama/read)

**Book Chapter:** [Reasoning with Large Language Models](https://leanpub.com/read/ollama/reasoning-with-large-language-models)

This example uses LangChain's `ChatOllama` with `reasoning=True` to access models that expose their internal chain-of-thought (like DeepSeek-R1). The script asks a counting question ("How many odd integers are greater than 0 and less than 10?"), then separately displays the model's step-by-step reasoning process and its final JSON answer.

## Files

| File | Description |
|---|---|
| `reasoning_test_1.py` | Structured reasoning query with thinking/answer separation |
| `pyproject.toml` | Project metadata and dependencies |

## Prerequisites

- **Ollama** installed and running locally. See [ollama.com](https://ollama.com).
- Pull a reasoning-capable model (e.g., `ollama pull deepseek-r1`) or use the default model.

## Run

```bash
cd reasoning
uv run reasoning_test_1.py
```

> **Tip:** For best results, set `MODEL=deepseek-r1` (or another reasoning model). The default `nemotron-3-nano:4b` will still work but may not populate the reasoning field.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL` | `nemotron-3-nano:4b` | Ollama model (reasoning models like `deepseek-r1` recommended) |

## Copyright and License

Copyright 2024-2026 Mark Watson. All rights reserved.
