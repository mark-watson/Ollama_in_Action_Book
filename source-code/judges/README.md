# LLM Evaluation / Judges Examples

**Book:** *Ollama in Action* — available free to read online at [https://leanpub.com/ollama/read](https://leanpub.com/ollama/read)

**Book Chapter:** [Automatic Evaluation of LLM Results: More Tool Examples](https://leanpub.com/read/ollama/automatic-evaluation-of-llm-results-more-tool-examples)

These examples demonstrate the **LLM-as-a-judge** pattern: using one LLM call to evaluate the correctness of another LLM's output. The `judge_results` tool (from the shared `tools/` library) takes a prompt and a candidate answer, then returns a structured judgement with reasoning. This is useful for automated testing, guardrails, and quality assurance in LLM applications.

## Files

| File | Description |
|---|---|
| `example_judge.py` | Basic judge — evaluates correct and incorrect arithmetic answers |
| `example_judge2.py` | Advanced judge — evaluates translations, code generation, and arithmetic across multiple test cases |
| `pyproject.toml` | Project metadata and dependencies |

## Prerequisites

- **Ollama** installed and running locally. See [ollama.com](https://ollama.com).
- Pull the default model: `ollama pull nemotron-3-nano:4b`

## Run

```bash
cd judges
uv run example_judge.py
uv run example_judge2.py
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL` | `nemotron-3-nano:4b` | Ollama model to use |
| `CLOUD` | *(unset)* | Set to any non-empty value to use Ollama Cloud |
| `OLLAMA_API_KEY` | *(none)* | Required when `CLOUD` is set |

## Copyright and License

Copyright 2024-2026 Mark Watson. All rights reserved.
