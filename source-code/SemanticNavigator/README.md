# Semantic Navigator App

**Book:** *Ollama in Action* — available free to read online at [https://leanpub.com/ollama/read](https://leanpub.com/ollama/read)

**Book Chapter:** [Semantic Navigator App Using Gradio](https://leanpub.com/read/ollama/semantic-navigator-app-using-gradio)

This example builds a **Gradio web application** that combines LLM-powered entity extraction with an interactive chatbot. Paste any text into the app and click "Extract" — the LLM identifies persons, places, and organizations and maps the relationships between them. You can then chat with the extracted knowledge graph, asking follow-up questions grounded in the structured data.

## Files

| File | Description |
|---|---|
| `app.py` | Gradio app — entity/link extraction + streaming chatbot |
| `pyproject.toml` | Project metadata and dependencies |

## Prerequisites

- **Ollama** installed and running, or an Ollama Cloud API key.
- Pull the default model: `ollama pull nemotron-3-nano:4b`

## Run

```bash
cd SemanticNavigator
uv run app.py
```

The Gradio UI will launch in your browser (typically at `http://127.0.0.1:7860`).

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL` | `nemotron-3-nano:4b` | Ollama model to use |
| `OLLAMA_API_KEY` | *(none)* | Your Ollama Cloud API key |

> **Note:** This example currently connects to Ollama Cloud by default. To use a local Ollama instance, modify the `Client` host in `app.py`.

## Copyright and License

Copyright 2024-2026 Mark Watson. All rights reserved.
