# Ollama-book-examples
Examples for my Ollama LLM AI book https://leanpub.com/ollama

*Note: Code refactored into separate sub-directories, code cleanup, remove outdated examples. October 14, 2025.*

## Running the examples

On October 14, 2025 I refactored the code to use sub-directories. Change to a subdirectory and use **uv** to run each example. For example:

```
cd judges
uv run example_judge.py
```


## Directory layout
- `agents/` – browser and automation agent examples
- `autogen/` – AG2/Autogen automation examples
- `chains/` – LangChain style tool-calling demos
- `graph/` – Kùzu graph database integrations
- `judges/` – self-evaluation and judge utilities
- `langgraph/` – LangGraph experiments
- `memory/` – Mem0 and persistence samples
- `reasoning/` – structured reasoning flows
- `smolagents/` – HuggingFace smolagents demos and helpers
- `tool_examples/` – end-to-end examples that showcase the shared tools
- `tools/` – reusable tool implementations shared by multiple examples

Additional folders such as `OllamaCloud/`, `short_examples/`, and `short_programs/` keep their original content.
