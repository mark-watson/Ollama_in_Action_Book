# Ensure OLLAMA_API_KEY is set in ENV

# Note: the answer.contents is nicely converted from HTML to Markdown.

import os
import sys
from pathlib import Path
import ollama
from ollama import Client

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ollama_config import get_model, get_client

client = get_client()


def generate(text: str) -> str:
    """Generate response from the model."""
    messages = [{'role': 'user', 'content': text}]
    response = client.chat(get_model(), messages=messages, stream=False)
    return response['message']['content']


SUMMARIZE_PROMPT = (
    "Summarize the following Markdown text, returning only plain text.\n\n"
    "Markdown text:\n\n"
)


def clean_web_query(query: str, max_results: int = 2) -> list[str]:
    """Search web and summarize results."""
    ret = []
    response = ollama.web_search(query, max_results=max_results)
    for answer in response['results']:
        markdown = answer.content
        clean = generate(SUMMARIZE_PROMPT + markdown)
        ret.append(clean)
    return ret


if __name__ == "__main__":
    results = clean_web_query(
        "AI Consultant Mark Watson works with AI, Lisp, Java, Clojure, and the Semantic Web.",
        max_results=1
    )
    for clean in results:
        print(f"{clean}\n\n")