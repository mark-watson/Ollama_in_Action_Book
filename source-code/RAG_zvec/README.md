# Agentic RAG with Zvec Vector Store

**Book:** *Ollama in Action* — available free to read online at [https://leanpub.com/ollama/read](https://leanpub.com/ollama/read)

**Book Chapter:** [RAG Using zvec Vector Datastore and Local Model](https://leanpub.com/read/ollama/rag-using-zvec-vector-datastore-and-local-model)

This example implements an **Agentic Retrieval-Augmented Generation (Agentic RAG)** pipeline that runs entirely locally. The multi-agent implementation is based on the Google Research blog post/paper ["Unlocking dependable responses with Gemini: Enterprise Agent Platforms & Agentic RAG"](https://research.google/blog/unlocking-dependable-responses-with-gemini-enterprise-agent-platforms-agentic-rag/).

It uses a multi-agent workflow consisting of:
- **Planner Agent**: Analyzes user questions to map out a search plan and generate multi-query variations.
- **Sufficient Context Agent**: Evaluates if retrieved context is sufficient, generating targeted feedback for iteration if context gaps are found.
- **Synthesis Agent**: Produces the final response grounded strictly in the retrieved snippets, stating missing details rather than hallucinating.

The local storage uses the [zvec](https://github.com/zacharycbrown/zvec) vector store to index text files, Ollama's embedding model (`embeddinggemma`) to generate 768-dimensional vectors, and a local chat model (such as `nemotron-3-nano:4b` or `qwen3:1.7b`) for inference.

## Files

| File | Description |
|---|---|
| `app.py` | Agentic RAG pipeline — indexes text files from `../data/`, builds a zvec collection, then enters an interactive multi-agent Q&A loop |
| `pyproject.toml` | Project metadata and dependencies |

## Architecture

![Generated image](architecture.png)

## Prerequisites

- **Ollama** installed and running locally. See [ollama.com](https://ollama.com).
- Pull the embedding model: `ollama pull embeddinggemma`
- Pull the default chat model: `ollama pull nemotron-3-nano:4b`
- Text files to index should be placed in the `../data/` directory (a sample `economics.txt` is included).

## Run

```bash
cd RAG_zvec
uv run app.py
```

You will see an interactive prompt with step-by-step agent tracing:

```
Building zvec index from text files …
Indexed 9 chunks from ../data

Agentic RAG chat ready  (model: nemotron-3-nano:4b)
Type your question, or 'quit' to exit.

You> What are the main schools of economic thought?

🧠 [Planner] Analyzing question and generating search plan...
   ↳ Plan: Search for a comprehensive list and description of the main schools of economic thought, such as Keynesian, Neoclassical, Marxist, Austrian, etc.
   ↳ Initial Queries: ['main schools of economic thought', 'major schools of economics overview', 'schools of economic thought']
🔍 [Retriever] Searching vector store for queries...
   ↳ Found 4 unique context snippet(s).
🤖 [Sufficiency Check] Evaluating context sufficiency (Iteration 1)...
   ↳ Sufficiency: False
   ↳ Reason: Only the Austrian School is described in the snippets; other major schools such as Keynesian, Neoclassical, Marxist, etc., are not mentioned.
🔄 [Rewriter] Context insufficient. Feedback: 'Missing information on Keynesian economics, neoclassical theory, Marxist economics, and possibly other contemporary schools. These should be searched for.'
   Generating new queries based on feedback...
   ↳ New queries: ['Keynesian economics overview', 'Neoclassical economic theory summary']
🔍 [Retriever] Retrieving additional context...
   ↳ Found 2 new unique snippet(s). Total unique snippets: 6.
🤖 [Sufficiency Check] Evaluating context sufficiency (Iteration 2)...
   ↳ Sufficiency: False
   ↳ Reason: The snippets only describe the Austrian School in detail. They do not mention other significant schools of economic thought like Keynesian, Neoclassical, or Marxist, which are essential to a complete answer.
...
⚠️  [Iteration Limit] Reached max iterations (3). Proceeding to synthesis with partial context.
✍️  [Synthesis] Generating final response...

Assistant> **Answer based on the provided context**

- The **Austrian School** (also called the Vienna or Psychological School) is described in detail as one school of economic thought that emphasizes the spontaneous organizing power of the price mechanism, advocates a “laissez‑faire” approach, and stresses voluntary contractual agreements with minimal government intervention.

**What is missing / could not be retrieved**

- The context does **not** mention or provide any information about other significant schools of economic thought such as Keynesian economics, Neoclassical economics, Marxist economics, etc.  

You>
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL` | `nemotron-3-nano:4b` | Ollama chat model |
| `EMBEDDING_MODEL` | `embeddinggemma` | Ollama embedding model |
| `DATA_DIR` | `../data` | Directory containing `.txt` files to index |
| `CLOUD` | *(unset)* | Set to any non-empty value to use Ollama Cloud |
| `OLLAMA_API_KEY` | *(none)* | Required when `CLOUD` is set |

## Copyright and License

Copyright 2024-2026 Mark Watson. All rights reserved.
