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
