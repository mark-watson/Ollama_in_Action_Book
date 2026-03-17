"""
Shared Ollama configuration for all book examples.

Environment variables:
  MODEL          - Override the default model name (default: nemotron-3-nano:4b)
  CLOUD          - If set (any non-empty value), use Ollama Cloud service
  OLLAMA_API_KEY - Required when CLOUD is set; your Ollama Cloud API key
"""

import os
from ollama import Client

DEFAULT_MODEL = "nemotron-3-nano:4b"


def get_model() -> str:
    """Return the model name from MODEL env var, or the default."""
    return os.environ.get("MODEL", DEFAULT_MODEL)


def get_client() -> Client:
    """
    Return an Ollama Client configured for local or cloud use.

    When the CLOUD environment variable is set, connects to Ollama Cloud
    using the OLLAMA_API_KEY environment variable for authentication.
    Otherwise, connects to the local Ollama service (http://localhost:11434).
    """
    if os.environ.get("CLOUD"):
        api_key = os.environ.get("OLLAMA_API_KEY", "")
        return Client(
            host="https://ollama.com",
            headers={"Authorization": "Bearer " + api_key},
        )
    return Client()  # default: http://localhost:11434
