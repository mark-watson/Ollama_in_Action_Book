import os
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelSettings
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from typing import Literal

# --- 1. Define the Tool ---
# This is a simple Python function that we'll make available to the AI agent.
# The `@tool` decorator marks it as a function the AI can decide to call.
# Docstrings are very important here, as the LLM uses them to understand
# what the tool does and what its parameters are for.

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
    # In a real application, you would make an API call here.
    # For this example, we'll return a consistent, hardcoded response.
    return f"The weather in {location} is 72 degrees {unit} and sunny."

# --- 2. Configure the AI Model ---
# We configure the model to point to our local Ollama instance.
# Note: Ensure your Ollama server is running and you have pulled a model.
# Command to run Ollama server (in your terminal): `ollama serve`
# Command to pull a model: `ollama pull llama3`

# The API key is required by the provider's structure but is not used by Ollama.
# We set a dummy value.
os.environ["OPENAI_API_KEY"] = "ollama"

# Point the OpenAIProvider to the local Ollama API endpoint.
ollama_provider = OpenAIProvider(
    base_url="http://localhost:11434/v1",
)

# Use the OpenAIModel class with the custom provider. We're using 'qwen3:8b' as it's
# generally good at following instructions for tool use.
ollama_model = OpenAIChatModel(
    "qwen3:8b",
    provider=ollama_provider,
    settings=ModelSettings(temperature=0.1) # Lower temperature for more predictable tool use
)

# --- 3. Create the Agent ---
# The Agent is the orchestrator. We provide it with the model configuration
# and a list of the tools it's allowed to use.

tools = [get_weather]
agent = Agent(ollama_model, tools=tools)

# --- 4. Run the Agent and Invoke the Tool ---
# We'll now ask a question that can only be answered by using the `get_weather` tool.
# The agent will analyze the prompt, realize it needs to call the tool,
# extract "Flagstaff, AZ" as the `location` parameter, and execute the function.

print("Querying the agent to get the weather in Flagstaff...\n")

# Use run_sync for a synchronous call
response = agent.run_sync("What's the weather like in Flagstaff, AZ?")

print("\n--- Final Agent Response ---")
print(response)

# Example of using a different unit
print("\n\nQuerying the agent to get the weather in celsius...\n")
response_celsius = agent.run_sync("What is the weather in Paris in celsius?")

print("\n--- Final Agent Response (Celsius) ---")
print(response_celsius)
