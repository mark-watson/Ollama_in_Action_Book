# Smolagents Examples

**Book:** *Ollama in Action* — available free to read online at [https://leanpub.com/ollama/read](https://leanpub.com/ollama/read)

**Book Chapter:** [Building Agents with Ollama and the Hugging Face Smolagents Library](https://leanpub.com/read/ollama/building-agents-with-ollama-and-the-hugging-face-smolagents-library)

These examples use the [smolagents](https://huggingface.co/docs/smolagents) library from Hugging Face to build lightweight AI agents backed by Ollama. Two agent types are demonstrated: a `ToolCallingAgent` that uses structured tool calls (e.g., a weather tool), and a `CodeAgent` that generates and executes Python code to accomplish tasks like listing and summarizing directory contents.

## Files

| File | Description |
|---|---|
| `smolagents_test.py` | `ToolCallingAgent` with a stubbed weather tool |
| `smolagents_agent_test1.py` | `CodeAgent` that uses file-system tools to list and describe the current directory |
| `smolagents_tools.py` | Wrappers that adapt the shared `tools/` library functions for smolagents compatibility |
| `smolagents_compat.py` | Compatibility patches for smolagents/LiteLLM dependency changes |
| `pyproject.toml` | Project metadata and dependencies |

## Prerequisites

- **Ollama** installed and running locally. See [ollama.com](https://ollama.com).
- Pull the default model: `ollama pull nemotron-3-nano:4b`

## Run

```bash
cd smolagents
uv run smolagents_test.py
uv run smolagents_agent_test1.py
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL` | `nemotron-3-nano:4b` | Ollama model to use |
| `CLOUD` | *(unset)* | Set to any non-empty value to use Ollama Cloud |
| `OLLAMA_API_KEY` | *(none)* | Required when `CLOUD` is set |

## Copyright and License

Copyright 2024-2026 Mark Watson. All rights reserved.
