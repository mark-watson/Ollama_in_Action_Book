# LangGraph Agent Example

**Book:** *Ollama in Action* — available free to read online at [https://leanpub.com/ollama/read](https://leanpub.com/ollama/read)

**Book Chapter:** [LangGraph](https://leanpub.com/read/ollama/langgraph)

This example builds a **LangGraph agent** that chains a DuckDuckGo web search with an LLM-powered answer extraction step. The agent follows a two-tool pipeline: first it searches the web using the `search` tool, then passes both the original query and the search results to the `answer_from_search` tool, which calls Ollama to synthesize a concise final answer. The agent streams its multi-step execution, printing each step as it runs.

## Files

| File | Description |
|---|---|
| `langgraph_agent_test.py` | LangGraph agent with `search` and `answer_from_search` tools |
| `pyproject.toml` | Project metadata and dependencies (includes `langchain`, `langgraph`, `duckduckgo-search`) |

## Prerequisites

- **Ollama** installed and running locally. See [ollama.com](https://ollama.com).
- Pull the default model: `ollama pull nemotron-3-nano:4b`

## Run

```bash
cd langgraph
uv run langgraph_agent_test.py
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL` | `nemotron-3-nano:4b` | Ollama model to use |
| `CLOUD` | *(unset)* | Set to any non-empty value to use Ollama Cloud |
| `OLLAMA_API_KEY` | *(none)* | Required when `CLOUD` is set |

## Copyright and License

Copyright 2024-2026 Mark Watson. All rights reserved.
