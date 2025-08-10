"""
Summarize text
"""

from ollama import ChatResponse
from ollama import chat


def summarize_text(text: str, context: str = "") -> str:
    """
    Summarizes text

    Parameters:
        text (str): text to summarize
        context (str): another tool's output can at the application layer can be used set the context for this tool.

    Returns:
        a string of summarized text

    """
    prompt = "Summarize this text (and be concise), returning only the summary with NO OTHER COMMENTS:\n\n"
    if len(text.strip()) < 50:
        text = context
    elif len(context) > 50:
        prompt = f"Given this context:\n\n{context}\n\n" + prompt

    summary: ChatResponse = chat(
        model="llama3.2:latest",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ],
    )
    return summary["message"]["content"]


# Function metadata for Ollama integration
summarize_text.metadata = {
    "name": "summarize_text",
    "description": "Summarizes input text",
    "parameters": {"text": "string of text to summarize",
                   "context": "optional context string"},
}

# Export the functions
__all__ = ["summarize_text"]
