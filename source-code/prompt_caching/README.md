# Prompt Caching Examples

**Book:** *Ollama in Action* — available free to read online at [https://leanpub.com/ollama/read](https://leanpub.com/ollama/read)

**Book Chapter:** [Prompt Caching](https://leanpub.com/read/ollama/prompt-caching)

This example demonstrates **Ollama's prompt caching** feature. It sends two requests with an identical large system prompt (the full `economics.txt` text) but different questions, and measures the prompt-processing speedup on the second request. When caching works, the GPU skips re-processing the shared prefix, resulting in dramatically faster response times.

## Files

| File | Description |
|---|---|
| `ollama_rest_test.py` | Sends two REST API requests with the same static context; compares cold-start vs. warm-cache prompt processing times |
| `pyproject.toml` | Project metadata and dependencies |

## Prerequisites

- **Ollama** installed and running locally. See [ollama.com](https://ollama.com).
- Pull the default model: `ollama pull nemotron-3-nano:4b`

## Run

```bash
cd prompt_caching
uv run ollama_rest_test.py
```

### Key Configuration

The script uses `keep_alive: "60m"` and a fixed `num_ctx: 4096` to ensure the model and its KV cache stay in VRAM between requests.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL` | `nemotron-3-nano:4b` | Ollama model to use |
| `CLOUD` | *(unset)* | Set to any non-empty value to use Ollama Cloud |
| `OLLAMA_API_KEY` | *(none)* | Required when `CLOUD` is set |

## Copyright and License

Copyright 2024-2026 Mark Watson. All rights reserved.
