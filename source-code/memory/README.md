# Memory / Persistence with Mem0

**Book:** *Ollama in Action* — available free to read online at [https://leanpub.com/ollama/read](https://leanpub.com/ollama/read)

**Book Chapter:** [Long Term Persistence Using Mem0 and Chroma](https://leanpub.com/read/ollama/long-term-persistence-using-mem0-and-chroma)

This example demonstrates **long-term memory persistence** for LLM conversations using [Mem0](https://github.com/mem0ai/mem0) backed by a local Chroma vector store. Each time you run the script, it searches for relevant memories from past conversations, includes them in the system prompt, and stores the new exchange.

## Files

| File | Description |
|---|---|
| `mem0_persistence.py` | CLI script — accepts a question, retrieves relevant memories, chats with Ollama, and persists the exchange |
| `pyproject.toml` | Project metadata and dependencies |

## Prerequisites

- **Ollama** installed and running locally. See [ollama.com](https://ollama.com).
- Pull the default chat model: `ollama pull nemotron-3-nano:4b`
- Pull the embedding model: `ollama pull nomic-embed-text`

## Run

```bash
cd memory
uv run mem0_persistence.py "What color is the sky?"
uv run mem0_persistence.py "What is the last color we talked about?"
```

Run the script multiple times with different questions to see how past memories influence subsequent answers.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL` | `nemotron-3-nano:4b` | Ollama chat model |
| `CLOUD` | *(unset)* | Set to any non-empty value to use Ollama Cloud |
| `OLLAMA_API_KEY` | *(none)* | Required when `CLOUD` is set |

## Copyright and License

Copyright 2024-2026 Mark Watson. All rights reserved.
