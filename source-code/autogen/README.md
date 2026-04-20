# AutoGen + Ollama Example

**Book Chapter:** [Using AG2 Open-Source AgentOS LLM-Based Agent Framework for Generating and Executing Python Code](https://leanpub.com/read/ollama/using-ag2-open-source-agentos-llm-based-agent-framework-for-generating-and-executing-python-code) — *Ollama in Action* (free to read online).

This directory contains a simple example that uses the [Ollama Python SDK](https://github.com/ollama/ollama-python) to send a chat request to a locally-running (or cloud-hosted) Ollama model. The model is asked to generate Python code for plotting year-to-date stock price changes for NVDA and TESLA — demonstrating how Ollama can produce runnable code from a natural-language prompt.

## Files

| File | Description |
|---|---|
| `autogen_python_example.py` | Main script — sends a chat request to Ollama and prints the model's response |
| `pyproject.toml` | Project metadata and dependencies for use with `uv` |

## Dependencies

The project depends on:

- **ollama** — Python SDK for communicating with the Ollama API
- **yfinance** — Yahoo Finance API (used by the model's generated code to fetch stock data)
- **matplotlib** — Plotting library (used by the model's generated code to render charts)
- **ag2** — AutoGen agent framework
- **fix-busted-json** — Utility for repairing malformed JSON output from models

The shared `ollama_config` module (in the parent `source-code/` directory) provides model selection and client configuration.

## Running with uv

```bash
uv sync
uv pip install pip
uv run autogen_python_example.py
```

> **Note:** `pip` must be installed inside the uv venv because the AutoGen agent may generate Python code that it attempts to run in a sandbox. That generated code may need to `pip install` additional libraries at runtime.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL` | `nemotron-3-nano:4b` | Ollama model to use |
| `CLOUD` | *(unset)* | Set to any non-empty value to use Ollama Cloud instead of a local instance |
| `OLLAMA_API_KEY` | *(none)* | Required when `CLOUD` is set — your Ollama Cloud API key |

## Prerequisites

- **Ollama** must be installed and running locally (unless using cloud mode). See [ollama.com](https://ollama.com) for installation instructions.
- The model specified by `MODEL` (default: `nemotron-3-nano:4b`) must be available. Pull it with: `ollama pull nemotron-3-nano:4b`
