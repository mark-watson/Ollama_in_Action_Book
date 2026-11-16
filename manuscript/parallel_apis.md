# Using the Parallel Search and Research APIs with Ollama Tools

Local language models running on Ollama have a fixed training cutoff date. When you ask a local model about recent events, new software releases, or live market data, the model cannot answer from its weights alone. By combining Ollama native tool calling with the Parallel Search and Task (Research) APIs, a small local model can fetch current web facts or run multi-step research jobs and cite its sources.

This chapter shows how to wrap the official Parallel Python SDK (`parallel-web`) as Ollama tools. All code for this chapter is in the `source-code/parallel_apis` directory.

## Why Use Parallel APIs with Local Models?

Standard web search engines return ranked links and short snippets written for human readers. To use a standard search engine with an LLM, your program must run a query, download full HTML pages, strip navigation and script tags, convert the markup to text, and trim the text so it fits inside a small local model context window.

Parallel ([https://parallel.ai](https://parallel.ai)) provides APIs built for language models and agents:

1. **Parallel Search API (`client.search`)**: Accepts both a natural-language `objective` and a list of short `search_queries`. It returns pre-compressed text `excerpts` extracted from relevant pages along with titles and URLs. This keeps token counts small for local models such as `qwen3.5:2b` or `qwen3.5:0.8b`.
2. **Parallel Task / Research API (`client.task_run.create` and `client.task_run.result`)**: Runs an asynchronous, multi-step web research job on Parallel servers. You supply an input topic, an `output_schema` description, and a processor tier (`lite`, `base`, `core`, `pro`, or `ultra`). The API returns a synthesized answer along with a `basis` array that lists confidence ratings (`high`, `medium`, or `low`), reasoning, and source citations for the output.

By exposing both APIs as Python tool functions, an Ollama model can pick fast search for quick factual questions and deep research for complex reports.

## How Ollama Native Tool Calling Works

When you pass Python functions in the `tools=[...]` argument of `client.chat()`, the Ollama Python SDK inspects each function signature, type annotations, and docstring at runtime. It converts that metadata into a JSON Schema tool specification and sends it to the model.

The conversation loop follows four steps:

1. Your script sends the system prompt, the user prompt, and the tool definitions to Ollama.
2. If the model needs external data, it returns an assistant message containing one or more `tool_calls` with the chosen function name and JSON arguments.
3. Your script runs the matching Python function with those arguments and appends a message with `role="tool"` containing the serialized tool output.
4. Your script calls `client.chat()` again with the updated message list so the model can read the tool output and write a sourced response.

## Data Formats Returned by Parallel

Before looking at the Python code, examine the data structures that our tool functions pass back into the Ollama context window.

A call to `search_web` returns a compact dictionary of search results where each entry contains a URL, a page title, and up to three relevant text excerpts:

```json
{
  "results": [
    {
      "url": "https://docs.parallel.ai/integrations/ollama-tool-calling",
      "title": "Ollama Tool Calling - Parallel Documentation",
      "excerpts": [
        "Give local models (Qwen 3.5, Gemma 4, Llama 3.1+, etc.) running under Ollama real-time web search by registering Parallel Search as a tool.",
        "Modern Ollama models support native tool calling. You pass tools=[...] on a chat call, the model emits structured tool_calls, your code executes them against Parallel, and you feed the results back in a follow-up turn."
      ]
    }
  ]
}
```

A call to `research_topic` uses the Parallel Task API and returns the task `run_id`, the synthesized `output`, and the verification `basis` containing confidence scores, reasoning, and citations:

```json
{
  "run_id": "trun_01j8x9k2m4p7q",
  "output": "Recent open-weight small language models supporting native tool calling include Qwen 3.5 (0.8B, 2B, 4B), Llama 3.2 (1B, 3B), and Granite 3.",
  "basis": [
    {
      "field": "output",
      "confidence": "high",
      "reasoning": "Verified across official model documentation and Ollama library listings.",
      "citations": [
        {
          "url": "https://ollama.com/library/qwen3.5",
          "title": "qwen3.5 - Ollama Library"
        }
      ]
    }
  ]
}
```

## Defining the Parallel Tool Functions

The file `source-code/parallel_apis/parallel_tools.py` defines two tool functions: `search_web` and `research_topic`.

Pay close attention to the docstrings in `search_web` and `research_topic`. Ollama reads the `Args:` section of each docstring to tell the language model how to format each parameter. In `search_web`, the docstring instructs the model to pass a self-contained `objective` and three diverse keyword queries of three to six words each.

Here is the complete listing of `source-code/parallel_apis/parallel_tools.py`:

```python
import os
from typing import Any
from parallel import Parallel


def _get_parallel_client() -> Parallel:
    api_key = os.environ.get("PARALLEL_API_KEY")
    if not api_key:
        raise ValueError("Set the PARALLEL_API_KEY environment variable.")
    return Parallel(api_key=api_key)


def search_web(objective: str, search_queries: list[str]) -> dict[str, Any]:
    """Searches the web for current and factual information using Parallel.

    Args:
        objective: A concise, self-contained search query. Must include the key entity or topic being searched for.
        search_queries: Exactly 3 keyword queries, each 3-6 words. Must be diverse: vary entity names, synonyms, and angles. Each query must include the key entity. Never write sentences or use site: operators.

    Returns:
        A dict with a results list, each containing url, title, and excerpts.
    """
    client = _get_parallel_client()
    response = client.search(
        objective=objective,
        search_queries=search_queries,
    )
    # Limit to 3 excerpts per result to keep tool output within small local model context limits.
    return {
        "results": [
            {
                "url": r.url,
                "title": r.title,
                "excerpts": r.excerpts[:3] if r.excerpts else [],
            }
            for r in response.results
        ]
    }


def research_topic(
    topic: str,
    output_schema: str = "A structured research report with key facts, findings, and dates.",
    processor: str = "base",
) -> dict[str, Any]:
    """Executes a multi-step deep web research task with citations and confidence scores using the Parallel Task API.

    Args:
        topic: The research question or subject to investigate in depth.
        output_schema: Description of the expected research output structure.
        processor: Parallel processor tier to use (lite, base, core, pro, or ultra).

    Returns:
        A dict with the synthesized output and a basis list containing citations, confidence, and reasoning.
    """
    client = _get_parallel_client()
    task_run = client.task_run.create(
        input=topic,
        task_spec={"output_schema": output_schema},
        processor=processor,
    )
    # task_run.result blocks and polls the background job until synthesis completes or api_timeout expires.
    run_result = client.task_run.result(task_run.run_id, api_timeout=600)

    raw_basis = (
        getattr(run_result.output, "basis", None)
        or getattr(run_result, "basis", None)
        or []
    )
    basis_summary = []
    for item in raw_basis:
        citations = [
            {"url": c.url, "title": getattr(c, "title", "")}
            for c in getattr(item, "citations", []) or []
        ]
        basis_summary.append(
            {
                "field": getattr(item, "field", "output"),
                "confidence": getattr(item, "confidence", ""),
                "reasoning": getattr(item, "reasoning", ""),
                "citations": citations,
            }
        )

    output_content = getattr(run_result.output, "content", run_result.output)
    return {
        "run_id": task_run.run_id,
        "output": output_content,
        "basis": basis_summary,
    }


__all__ = ["search_web", "research_topic"]
```

In `search_web`, slicing `r.excerpts[:3]` bounds the returned payload size so that even if a search returns many results, the tool response will not overflow a local model's context window. In `research_topic`, `client.task_run.create()` starts the research job on Parallel servers and returns a `task_run` handle with a `run_id`. Calling `client.task_run.result(task_run.run_id, api_timeout=600)` waits for the task to finish and returns the synthesized content together with its citation basis.

## Building a Web Search Agent with Ollama

The file `source-code/parallel_apis/search_agent.py` connects `search_web` to an Ollama chat loop. It uses `get_client()` and `get_model()` from our shared `ollama_config.py` module so you can run the agent against either a local Ollama server or Ollama Cloud.

Here is `source-code/parallel_apis/search_agent.py`:

```python
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ollama_config import get_client, get_model
from parallel_tools import search_web


def chat_with_search(user_message: str, model: str | None = None) -> str:
    client = get_client()
    active_model = model or get_model()
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful research assistant. Use search_web "
                "for current information. Cite sources with URLs."
            ),
        },
        {"role": "user", "content": user_message},
    ]
    available = {"search_web": search_web}

    # Loop until the model produces a final answer without requesting more tool calls.
    while True:
        response = client.chat(
            model=active_model,
            messages=messages,
            tools=[search_web],
        )
        messages.append(response.message)

        if not response.message.tool_calls:
            return response.message.content

        for call in response.message.tool_calls:
            fn = available.get(call.function.name)
            if fn is None:
                result = {"error": f"Unknown tool: {call.function.name}"}
            else:
                print(
                    f"[Tool Call] {call.function.name}("
                    f"{json.dumps(call.function.arguments)})"
                )
                result = fn(**call.function.arguments)
            messages.append(
                {
                    "role": "tool",
                    "tool_name": call.function.name,
                    "content": json.dumps(result),
                }
            )


if __name__ == "__main__":
    prompt = "What is new with Parallel Web Systems and its Search API?"
    print(f"Prompt: {prompt}\n")
    answer = chat_with_search(prompt)
    print("\nResponse:\n")
    print(answer)
```

The `while True` loop in `chat_with_search` supports multi-step tool calling. If the first search does not answer every part of the user question, the model can emit a second `search_web` tool call before returning a final text response. The loop serializes each tool result to JSON and appends it with `role="tool"` and `tool_name=call.function.name`.

## Combining Fast Search and Deep Research in One Agent

Some questions only need a single web lookup, while others need a structured report built from multiple sources. The file `source-code/parallel_apis/research_agent.py` passes both `[search_web, research_topic]` in the `tools` list so the Ollama model can choose the right tool for the prompt.

Here is `source-code/parallel_apis/research_agent.py`:

```python
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ollama_config import get_client, get_model
from parallel_tools import research_topic, search_web


def run_research_assistant(user_message: str, model: str | None = None) -> str:
    client = get_client()
    active_model = model or get_model()
    messages = [
        {
            "role": "system",
            "content": (
                "You are a research assistant with two tools: "
                "use search_web for fast factual lookups, and use research_topic "
                "when the user asks for an in-depth report with confidence scores "
                "and verified citations. Always include source URLs in your answer."
            ),
        },
        {"role": "user", "content": user_message},
    ]
    available = {
        "search_web": search_web,
        "research_topic": research_topic,
    }

    while True:
        response = client.chat(
            model=active_model,
            messages=messages,
            tools=[search_web, research_topic],
        )
        messages.append(response.message)

        if not response.message.tool_calls:
            return response.message.content

        for call in response.message.tool_calls:
            fn = available.get(call.function.name)
            if fn is None:
                result = {"error": f"Unknown tool: {call.function.name}"}
            else:
                print(
                    f"[Tool Call] {call.function.name}("
                    f"{json.dumps(call.function.arguments)})"
                )
                result = fn(**call.function.arguments)
            messages.append(
                {
                    "role": "tool",
                    "tool_name": call.function.name,
                    "content": json.dumps(result),
                }
            )


if __name__ == "__main__":
    prompt = (
        "Run a deep research report on recent open-weight small language models "
        "that support native tool calling, and summarize the findings with citations."
    )
    print(f"Prompt: {prompt}\n")
    answer = run_research_assistant(prompt)
    print("\nResponse:\n")
    print(answer)
```

The system message tells the model when to pick each tool: `search_web` for fast factual lookups and `research_topic` when the user asks for an in-depth report with confidence scores and verified citations.

## Running the Examples and Interpreting the Output

To run the examples, set your `PARALLEL_API_KEY` environment variable and run the scripts with `uv run` from `source-code/parallel_apis`:

```bash
$ cd source-code/parallel_apis
$ export PARALLEL_API_KEY="your-parallel-api-key"
$ uv run search_agent.py
Prompt: What is new with Parallel Web Systems and its Search API?

[Tool Call] search_web({"objective": "Find recent product releases and updates for Parallel Web Systems and its Search API", "search_queries": ["Parallel Web Systems Search API", "Parallel Web Systems product releases", "Parallel AI search documentation"]})

Response:

Parallel Web Systems provides AI-native web search and research APIs designed for language models and agents:

1. **Search API**: Accepts a natural-language objective alongside keyword queries and returns compressed, citation-ready text excerpts rather than raw HTML.
2. **Extract API**: Converts target URLs, including JavaScript-heavy pages and PDFs, into clean Markdown or focused excerpts.
3. **Task API**: Runs multi-step deep research jobs with processor tiers (`lite`, `base`, `core`, `pro`, `ultra`) and returns a Research Basis with citations and confidence scores.

Sources:
- https://docs.parallel.ai/search/search-quickstart
- https://docs.parallel.ai/integrations/ollama-tool-calling
```

Next, run the multi-tool research assistant:

```bash
$ uv run research_agent.py
Prompt: Run a deep research report on recent open-weight small language models that support native tool calling, and summarize the findings with citations.

[Tool Call] research_topic({"output_schema": "A structured research report listing open-weight small language models with native tool calling, parameter sizes, and key capabilities.", "processor": "base", "topic": "Recent open-weight small language models that support native tool calling"})

Response:

### Open-Weight Small Language Models with Native Tool Calling

1. **Qwen 3.5 (0.8B, 2B, 4B, 9B)**: Supports native function calling in Ollama, structured JSON tool arguments, and multi-turn tool execution loops.
2. **Llama 3.2 (1B, 3B)**: Supports native tool definitions for edge and laptop deployments.
3. **Granite 3 Dense (2B, 8B)**: Tuned for enterprise function calling and RAG workflows.

**Verification Basis:**
- Confidence: `high`
- Sources:
  - https://ollama.com/library/qwen3.5
  - https://ollama.com/library/llama3.2
  - https://docs.parallel.ai/integrations/ollama-tool-calling
```

Look at what happened in these two runs:

- In `search_agent.py`, the local Ollama model read the docstring rules on `search_web` and generated both a clear `objective` string and three 3-6 word keyword queries. Parallel executed those searches in one round-trip and returned pre-compressed excerpts that the local model synthesized with source URLs.
- In `research_agent.py`, the prompt asked for a deep research report with citations. The local model selected `research_topic` instead of `search_web`, delegated the multi-step web research to the Parallel Task API (`processor="base"`), and formatted the returned output, confidence level, and citations into the final answer.

## Wrap Up

Combining local Ollama models with the Parallel Search and Task APIs gives small models access to current web knowledge without complex scraping code. The `search_web` tool handles low-latency lookups by returning compact excerpts tuned for LLM context windows, while `research_topic` offloads deep multi-step research and source verification to the Parallel Task API. Because both tools use standard Python type hints and docstrings, you can plug them directly into any Ollama tool-calling workflow.

## Optional Practice Problems

1. **Add the Parallel Extract API as a Third Tool.** The Parallel SDK provides `client.extract(urls=[...], objective=...)` to extract clean Markdown from specific URLs. Write a new tool function `extract_urls(urls: list[str], objective: str) -> dict` in `parallel_tools.py` and register it alongside `search_web` and `research_topic` in `research_agent.py`. Test it by asking the agent to search for a topic and then extract full details from the top URL.
2. **Filter Research Results by Confidence Score.** Modify `research_topic` in `parallel_tools.py` to accept a `min_confidence: str = "medium"` parameter. Filter the `basis` entries so the function flags any field whose confidence is `"low"` and warns the local Ollama model not to state low-confidence claims as certain facts.
3. **Expose Structured JSON Schemas in `research_topic`.** The Parallel Task API accepts a JSON Schema dictionary in `task_spec={"output_schema": ...}` in addition to plain-text schema descriptions. Create a specialized tool function `research_company(company_name: str)` that requests a strict JSON schema with fields for `founding_year`, `headquarters`, `key_products`, and `citations`, and test it with an Ollama model.
