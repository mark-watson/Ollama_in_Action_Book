# Pydantic AI Tool Use Examples

**Book:** *Ollama in Action* — available free to read online at [https://leanpub.com/ollama/read](https://leanpub.com/ollama/read)

**Book Chapter:** [Pydantic AI Experiments](https://leanpub.com/read/ollama/pydantic-ai-experiments)

These examples demonstrate tool calling via the [PydanticAI](https://docs.pydantic.dev/latest/concepts/agents/) agent framework, using Ollama through its OpenAI-compatible endpoint. The agent receives a natural-language prompt, decides which tool to invoke, and returns a synthesized answer. Two tools are showcased: a DuckDuckGo web search and a stub weather function.

## Files

| File | Description |
|---|---|
| `tool_duckduckgo_search.py` | PydanticAI agent with a DuckDuckGo Instant Answer search tool |
| `tool_use_weather.py` | PydanticAI agent with a stubbed weather tool (demonstrates tool schema with Pydantic `Field` annotations) |
| `pyproject.toml` | Project metadata and dependencies |

## Prerequisites

- **Ollama** installed and running locally. See [ollama.com](https://ollama.com).
- Pull a tool-calling-capable model: `ollama pull nemotron-3-nano:4b`

## Run

```bash
cd PydanticToolUse
uv run tool_use_weather.py
uv run tool_duckduckgo_search.py
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL` | `nemotron-3-nano:4b` | Ollama model to use |
| `CLOUD` | *(unset)* | Set to any non-empty value to use Ollama Cloud |
| `OLLAMA_API_KEY` | *(none)* | Required when `CLOUD` is set |

## Copyright and License

Copyright 2024-2026 Mark Watson. All rights reserved.
