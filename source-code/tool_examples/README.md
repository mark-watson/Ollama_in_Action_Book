# Tool Use Examples

**Book:** *Ollama in Action* — available free to read online at [https://leanpub.com/ollama/read](https://leanpub.com/ollama/read)

**Book Chapter:** [LLM Tool Calling with Ollama](https://leanpub.com/read/ollama/llm-tool-calling-with-ollama)

This example demonstrates **Ollama's native tool/function-calling** capability. The script registers three tools (`list_directory`, `read_file_contents`, `uri_to_markdown`) with the model, sends a natural-language prompt, and lets the LLM decide which tools to call. The tool call results are then printed, showing the full round-trip from prompt → tool selection → execution → output.

## Files

| File | Description |
|---|---|
| `ollama_tools_examples.py` | End-to-end tool-calling demo using the shared `tools/` library |
| `pyproject.toml` | Project metadata and dependencies |

## Prerequisites

- **Ollama** installed and running locally. See [ollama.com](https://ollama.com).
- Pull a tool-calling-capable model: `ollama pull nemotron-3-nano:4b`

## Run

```bash
cd tool_examples
uv run ollama_tools_examples.py
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL` | `nemotron-3-nano:4b` | Ollama model to use |
| `CLOUD` | *(unset)* | Set to any non-empty value to use Ollama Cloud |
| `OLLAMA_API_KEY` | *(none)* | Required when `CLOUD` is set |

## Copyright and License

Copyright 2024-2026 Mark Watson. All rights reserved.
