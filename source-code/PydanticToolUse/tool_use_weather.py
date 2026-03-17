import os
import sys
from pathlib import Path
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelSettings
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from typing import Literal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ollama_config import get_model

# --- 1. Define the Tool ---
def get_weather(
    location: str = Field(..., description="The city and state, e.g., San Francisco, CA"),
    unit: Literal["celsius",
                  "fahrenheit"] = Field("fahrenheit",
                                        description="The unit of temperature"),
) -> str:
    """
    Get the current weather for a specified location.
    This is a stubbed function and will return a fixed value for demonstration.
    """
    print(f"--- Tool 'get_weather' called with location: '{location}' and unit: '{unit}' ---")
    return f"The weather in {location} is 72 degrees {unit} and sunny."

# --- 2. Configure the AI Model (Ollama or Ollama Cloud) ---
if os.environ.get("CLOUD"):
    # Use Ollama Cloud via OpenAI-compatible endpoint
    api_key = os.environ.get("OLLAMA_API_KEY", "")
    os.environ["OPENAI_API_KEY"] = api_key
    ollama_provider = OpenAIProvider(base_url="https://ollama.com/v1")
else:
    os.environ["OPENAI_API_KEY"] = "ollama"
    ollama_provider = OpenAIProvider(base_url="http://localhost:11434/v1")

ollama_model = OpenAIChatModel(
    get_model(),
    provider=ollama_provider,
    settings=ModelSettings(temperature=0.1)
)

# --- 3. Create the Agent ---
tools = [get_weather]
agent = Agent(ollama_model, tools=tools)

# --- 4. Run the Agent and Invoke the Tool ---
print("Querying the agent to get the weather in Flagstaff...\n")

response = agent.run_sync("What's the weather like in Flagstaff, AZ?")

print("\n--- Final Agent Response ---")
print(response)

# Example of using a different unit
print("\n\nQuerying the agent to get the weather in celsius...\n")
response_celsius = agent.run_sync("What is the weather in Paris in celsius?")

print("\n--- Final Agent Response (Celsius) ---")
print(response_celsius)
