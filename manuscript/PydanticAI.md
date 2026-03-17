# Pydantic AI Experiments

In this chapter, we pivot from basic prompt engineering to the construction of autonomous, tool-augmented systems using the Pydantic AI library. Dear reader, you likely recognize the recurring challenge in software engineering: how to bridge the gap between non-deterministic, probabilistic outputs (LLMs) and deterministic, typed logic (Python functions). Pydantic AI addresses this head-on by treating the LLM not just as a text generator, but as a central reasoning engine capable of understanding structured metadata and making operational decisions. We will explore how to transform standard Python functions into discoverable "tools" using Pydantic’s powerful validation layer, ensuring that when an agent decides to interact with the real world—whether by checking the weather in Flagstaff or querying the DuckDuckGo API—it does so with strict type safety and predictable parameter extraction. By the end of these experiments, you will see how to build a unified workflow that remains infrastructure-agnostic, allowing you to toggle seamlessly between local Ollama instances and cloud-hosted OpenAI-compatible endpoints without refactoring your core agentic logic.

The examples for this chapter are in the directory **PydanticToolUse**.


## Weather Lookup Tool Use Example

This script demonstrates the integration of Pydantic AI with local or cloud-hosted Ollama models to create an extensible, tool-augmented agent. By leveraging the pydantic-ai library, the code establishes a structured workflow where a Python function, get_weather, is converted into a tool that the agent can autonomously decide to invoke based on user natural language queries. The implementation highlights a robust configuration pattern, utilizing environment variables to toggle between a local Ollama instance and a cloud-based OpenAI-compatible endpoint. This approach ensures that the agent remains agnostic to the underlying infrastructure while maintaining strict data validation and type safety through Pydantic's BaseModel and Field definitions, ultimately providing a seamless bridge between unstructured LLM outputs and deterministic Python logic.

```python
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
```

The core logic of this listing centers on the Agent class and its ability to perform function calling. By wrapping the **get_weather** function with specific parameter descriptions and type hints such as the `Literal` constraint for temperature units—the agent receives clear metadata that guides its decision-making process. When a user asks about the weather in a specific city, the agent identifies the matching tool, extracts the necessary arguments from the prompt, and executes the Python code. This "tool-use" loop is handled synchronously here via `agent.run_sync`, showing how LLMs can be transformed from simple text generators into active participants in a larger software ecosystem.

Beyond the tool definitions, the script illustrates a flexible model configuration strategy. It dynamically constructs an `OpenAIChatModel` using an `OpenAIProvider`, which redirects traffic to an Ollama compatible base URL. This allows developers to swap out heavy proprietary models for lightweight, local alternatives without changing the fundamental structure of their agentic logic. The use of `ModelSettings` to fix the temperature at 0.1 further ensures that the agent's output remains consistent and predictable, which is essential when the goal is to trigger specific API calls or internal functions reliably across different execution environments.

Sample output looks like this:

```
$ uv run tool_use_weather.py
Querying the agent to get the weather in Flagstaff...

--- Tool 'get_weather' called with location: 'Flagstaff, AZ' and unit: 'fahrenheit' ---

--- Final Agent Response ---
AgentRunResult(output='The weather in Flagstaff, AZ is currently **72°F** and **sunny**. Perfect weather for exploring the area! 🌞')


Querying the agent to get the weather in celsius...

--- Tool 'get_weather' called with location: 'Paris' and unit: 'celsius' ---

--- Final Agent Response (Celsius) ---
AgentRunResult(output='The current weather in Paris is **72 degrees Celsius** with sunny conditions. Let me know if you need further details! ☀️')
```

## DuckDuckGo Search Summary Tool Example

This implementation demonstrates the creation of a functional AI agent using the Pydantic AI framework, specifically designed to bridge the gap between a Large Language Model and real-time external data. The script begins by establishing a robust environment, configuring pathing to incorporate local utilities like `ollama_config` and setting up a tool calling mechanism via the DuckDuckGo Instant Answer API. By defining a search_web function with clear type hints and docstrings, we provide the LLM with the metadata it needs to understand when and how to perform a web search. The configuration logic allows for seamless switching between a local Ollama instance and a cloud-based provider, ensuring that the OpenAIChatModel remains flexible across different deployment environments. Ultimately, the code initializes a `pydantic_ai.Agent` that can autonomously decide to fetch live data—in this case, the population of San Diego to supplement its internal training data with fresh, verified information.

```python
import os
import sys
from pathlib import Path
import requests
from pydantic import Field
from pydantic_ai import Agent, ModelSettings
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ollama_config import get_model

# --- 1. Define the Web Search Tool ---
# This tool uses the DuckDuckGo Instant Answer API to get a summary for a query.
# The docstring is crucial because it tells the LLM what the tool's capabilities are.
# The LLM will decide to use this tool when it needs information it doesn't have.

def search_web(query: str = Field(..., description="The search query to look up on the web.")):
    """
    Performs a DuckDuckGo search and returns a summary of the top result.
    This tool is useful for finding real-time information like news, facts, or weather.
    """
    print(f"--- Tool 'search_web' called with query: '{query}' ---")
    try:
        url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1,
        }
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()

        # The Instant Answer API returns different types of results.
        # We'll prioritize the 'AbstractText' or 'Answer' fields.
        summary = data.get("AbstractText") or data.get("Answer")
        
        if summary:
            summary = summary.replace('<b>', '').replace('</b>', '').replace('<i>', '').replace('</i>', '')
            print(f"--- Search successful. Summary found. ---")
            return summary
        else:
            print(f"--- Search completed, but no direct summary was found. ---")
            return "No summary found for that query."
            
    except requests.RequestException as e:
        print(f"--- Error during web search: {e} ---")
        return f"An error occurred while trying to search: {e}"


# --- 2. Configure the AI Model (Ollama or Ollama Cloud) ---
if os.environ.get("CLOUD"):
    # Use Ollama Cloud via OpenAI-compatible endpoint
    api_key = os.environ.get("OLLAMA_API_KEY", "")
    os.environ["OPENAI_API_KEY"] = api_key
    ollama_provider = OpenAIProvider(base_url="https://ollama.com/v1")
else:
    os.environ["OPENAI_API_KEY"] = "ollama"  # Dummy key required by the provider interface
    ollama_provider = OpenAIProvider(base_url="http://localhost:11434/v1")

ollama_model = OpenAIChatModel(
    get_model(),
    provider=ollama_provider,
    settings=ModelSettings(temperature=0.1),  # Lower temperature for more reliable tool use
)


# --- 3. Create the Agent ---
agent = Agent(ollama_model, tools=[search_web])


# --- 4. Run the Agent to Find the Weather ---
prompt = "What is the population of San Diego?"
print(f"Querying the agent with prompt: '{prompt}'\n")

response = agent.run_sync(prompt)

print("\n--- Final Agent Response ---")
print(response)
```

The core strength of this script lies in its "tool-augmented" architecture. By passing the **search_web** function into the Agent constructor, the model is granted the agency to perform external HTTP requests when its internal knowledge is insufficient or potentially outdated. The use of ModelSettings with a low temperature of 0.1 is a deliberate choice to ensure high reliability and consistency in tool invocation, reducing the likelihood of the model "hallucinating" search parameters or failing to parse the API response correctly.

From a structural standpoint, the code highlights the portability of modern AI workflows through its environment-aware configuration. By checking for a CLOUD environment variable, the script dynamically adjusts its API keys and base URLs, allowing you, dear reader, to test locally with a self-hosted Ollama server or develop with the Ollama Cloud endpoint. The integration of Pydantic’s Field for parameter description ensures that the interface between the Python code and the LLM’s reasoning engine is strictly typed and self-documenting, which is a best practice for building maintainable AI agents.

Sample output looks like:

```
$ uv run tool_duckduckgo_search.py
Querying the agent with prompt: 'What is the population of San Diego?'

--- Tool 'search_web' called with query: 'population of San Diego' ---
--- Search completed, but no direct summary was found. ---
--- Tool 'search_web' called with query: 'population of San Diego 2023' ---
--- Search completed, but no direct summary was found. ---

--- Final Agent Response ---
AgentRunResult(output='The population of San Diego is approximately 1.4 million as of the latest estimates. For the most accurate and up-to-date figure, I recommend checking official sources like the U.S. Census Bureau or city government statistics.')
```

## Wrap Up for Pandantic AI Library

The experiments conducted in this chapter illustrate a significant shift in AI development: the move toward "Agentic Type Safety." By leveraging the Pydantic AI library, we have successfully demonstrated that the inherent "fuzziness" of Large Language Models can be tamed and integrated into robust software architectures. Through the Weather Lookup and DuckDuckGo Search examples we saw how the library uses Python type hints and docstrings as a functional schema, allowing models to perform precise tool calls and handle real-time data retrieval with minimal friction. This marriage of Pydantic’s validation functionality with the reasoning capabilities of models, like those we run via Ollama, creates a developer experience that feels less like "prompting" and more like traditional API orchestration. As you move forward, the patterns established here like dynamic model configuration, environment-aware providers, and low-temperature settings for reliability will serve as a good basis for building production grade agents that are both extensible and maintainable within modern DevOps and production ecosystems.


