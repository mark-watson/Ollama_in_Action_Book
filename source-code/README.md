# Ollama in Action — Source Code Examples

**Book:** *Ollama in Action* — available free to read online at [https://leanpub.com/ollama/read](https://leanpub.com/ollama/read)

## Running the Examples

All examples use [uv](https://docs.astral.sh/uv/) for dependency management. Change to a subdirectory and run:

```bash
cd judges
uv run example_judge.py
```

Most examples support both **local Ollama** and **Ollama Cloud**. Set `CLOUD=1` and `OLLAMA_API_KEY` to use cloud mode:

```bash
export CLOUD=1
export OLLAMA_API_KEY="your-key-here"
```

## Configuration

All examples share `ollama_config.py` which reads:

| Variable | Default | Description |
|---|---|---|
| `MODEL` | `nemotron-3-nano:4b` | Ollama model to use |
| `CLOUD` | *(unset)* | Set to any non-empty value to use Ollama Cloud |
| `OLLAMA_API_KEY` | *(none)* | Required when `CLOUD` is set |

## Directory Layout

- `AG2_agents/` – AG2 (AutoGen 2) multi-agent code generation
- `autogen/` – AutoGen + Ollama code generation example
- `chains/` – Chained tool-calling demos (file reading + summarization)
- `DSP/` – DSPy chain-of-thought reasoning experiments
- `graph/` – Kùzu property graph database with natural-language queries
- `judges/` – LLM-as-a-judge evaluation techniques
- `langgraph/` – LangGraph agent with web search and answer synthesis
- `memory/` – Mem0 + Chroma long-term persistence
- `OllamaCloud/` – Ollama Cloud service examples (including web search)
- `prompt_caching/` – Prompt caching benchmarks
- `PydanticToolUse/` – PydanticAI tool-calling examples
- `RAG_zvec/` – Retrieval-Augmented Generation with zvec vector store
- `reasoning/` – Reasoning models (DeepSeek-R1) with thinking process extraction
- `SemanticNavigator/` – Gradio web app for entity extraction and chat
- `short_programs/` – Quick examples: image analysis and OpenAI compatibility
- `smolagents/` – HuggingFace smolagents (ToolCallingAgent + CodeAgent)
- `tool_examples/` – End-to-end Ollama native tool-calling demo
- `tools/` – Shared reusable tool implementations (file I/O, web search, summarization, SQLite, judges, etc.)
- `data/` – Sample data files used by examples (`economics.txt`, `sample.jpg`)
