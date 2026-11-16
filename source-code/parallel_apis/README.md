# Parallel Search and Research APIs with Ollama Tools

**Book:** *Ollama in Action* available free online at [https://leanpub.com/ollama/read](https://leanpub.com/ollama/read)

This directory shows how to connect the Parallel Search API and the Parallel Task (Research) API to Ollama native tool calling. The examples wrap the `parallel-web` Python SDK as tool functions that local or cloud Ollama models can invoke during a conversation.

## Files

| File | Description |
|---|---|
| `parallel_tools.py` | Defines `search_web` (Parallel Search API) and `research_topic` (Parallel Task API) as Ollama tool functions |
| `search_agent.py` | Multi-turn Ollama chat loop with the `search_web` tool |
| `research_agent.py` | Multi-turn Ollama chat loop that lets the model pick between fast web search (`search_web`) and deep research (`research_topic`) |
| `pyproject.toml` | Project metadata and dependencies (`ollama` and `parallel-web`) |
| `Makefile` | Lint and test targets |

## Prerequisites

- **Ollama** running locally, or an Ollama Cloud API key.
- A tool-capable Ollama model such as `qwen3.5:2b` or `qwen3.5:0.8b`.
- A **Parallel API key** exported in `PARALLEL_API_KEY` (get one from [https://parallel.ai](https://parallel.ai)).

## Run

```bash
cd source-code/parallel_apis
export PARALLEL_API_KEY="your-parallel-api-key"

uv run search_agent.py
uv run research_agent.py
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PARALLEL_API_KEY` | *(required)* | Your Parallel API key |
| `MODEL` | `qwen3.5:2b` | Ollama model to use |
| `CLOUD` | *(unset)* | Set to any non-empty value to use Ollama Cloud |
| `OLLAMA_API_KEY` | *(none)* | Required when `CLOUD` is set |

## Copyright and License

Copyright 2024-2026 Mark Watson. All rights reserved.
