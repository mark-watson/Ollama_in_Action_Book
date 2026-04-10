"""
AutoGen + Ollama Example
========================

This script demonstrates how to use the Ollama Python SDK to send a chat
request to a locally-running (or cloud-hosted) Ollama model. The example
prompt asks the model to generate Python code for plotting NVDA and TESLA
stock price changes year-to-date — a task that showcases Ollama's ability
to produce runnable code from a natural-language description.

The script relies on the shared ``ollama_config`` module (located in the
parent ``source-code/`` directory) to determine which model to use and
whether to connect to a local Ollama instance or the Ollama Cloud service.

Environment variables (set via ollama_config):
  MODEL          - Override the default model name (default: nemotron-3-nano:4b)
  CLOUD          - If set (any non-empty value), use Ollama Cloud instead of local
  OLLAMA_API_KEY - Required when CLOUD is set; your Ollama Cloud API key

Usage:
  uv run autogen_python_example.py
"""

import os
import sys
from pathlib import Path

# Add the parent source-code/ directory to sys.path so we can import
# the shared ollama_config module that lives one level up.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ollama import Client, ChatResponse
from ollama_config import get_model

# Requirements (for manual setup without uv):
# python3.11 -m venv venv
# pip install ollama yfinance matplotlib
# source venv/bin/activate

# Resolve the model name from the MODEL env var or fall back to the
# default defined in ollama_config (nemotron-3-nano:4b).
_model_name = get_model()

# Configure the Ollama client based on whether we are running locally or
# connecting to the Ollama Cloud service. The CLOUD environment variable
# controls this behavior (see ollama_config for details).
if os.environ.get("CLOUD"):
    # Cloud mode: authenticate with an API key and connect to ollama.com
    api_key = os.environ.get("OLLAMA_API_KEY", "fakekey")
    client = Client(
        host="https://ollama.com",
        headers={"Authorization": f"Bearer {api_key}"}
    )
else:
    # Local mode: connect to the default local Ollama daemon
    client = Client(host="http://localhost:11434")

# Send a chat request asking the model to generate Python code that plots
# YTD stock price changes for NVDA (NVIDIA) and TESLA. The model may return
# Python code that uses libraries like yfinance and matplotlib — make sure
# those are installed in your environment if you want to execute the output.
response: ChatResponse = client.chat(
    model=_model_name,
    messages=[
        {
            "role": "user",
            "content": "Plot a chart of NVDA and TESLA stock price change YTD.",
        }
    ]
)

# Print the model's response (which may contain Python code, a natural-
# language explanation, or both, depending on the model's behavior).
print(response["message"]["content"])
