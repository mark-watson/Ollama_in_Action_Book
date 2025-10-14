"""Utility tool functions shared across book examples."""

from .tool_anti_hallucination import detect_hallucination
from .tool_file_dir import list_directory
from .tool_file_contents import read_file_contents, write_file_contents
from .tool_judge_results import judge_results
from .tool_llm_eval import evaluate_llm_conversation
from .tool_sqlite import SQLiteTool, OllamaFunctionCaller
from .tool_summarize_text import summarize_text
from .tool_web_search import uri_to_markdown

__all__ = [
    "detect_hallucination",
    "list_directory",
    "read_file_contents",
    "write_file_contents",
    "judge_results",
    "evaluate_llm_conversation",
    "SQLiteTool",
    "OllamaFunctionCaller",
    "summarize_text",
    "uri_to_markdown",
]
