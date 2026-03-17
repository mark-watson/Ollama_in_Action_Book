"""smolagents example program (slightly modified)"""

import smolagents_compat  # noqa: F401

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smolagents.agents import ToolCallingAgent
from smolagents import tool, LiteLLMModel
from typing import Optional

from ollama_config import get_model

_model_name = get_model()

if os.environ.get("CLOUD"):
    api_base = "https://ollama.com"
    api_key = os.environ.get("OLLAMA_API_KEY", "")
else:
    api_base = "http://localhost:11434"
    api_key = "your-api-key"

model = LiteLLMModel(
    model_id=f"ollama_chat/{_model_name}",
    api_base=api_base,
    api_key=api_key,
)

@tool
def get_weather(location: str, celsius: Optional[bool] = False) -> str:
    """
    Get weather in the next days at given location.
    Secretly this tool does not care about the location, it hates the weather everywhere.

    Args:
        location: the location
        celsius: the temperature
    """
    return "The weather is UNGODLY with torrential rains and temperatures below -10°C"

agent = ToolCallingAgent(tools=[get_weather], model=model)

print(agent.run("What's the weather like in Paris?"))
