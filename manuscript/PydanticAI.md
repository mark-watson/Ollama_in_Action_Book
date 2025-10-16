# Pydantic AI Experiments


TBD

## Weather Lookup Tool Use Example

TBD

```python
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
```

TBD


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

TBD

```python
import os
import requests
from pydantic import Field
from pydantic_ai import Agent, ModelSettings
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

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
            # The API can return HTML entities, so we'll clean them up for this example.
            # In a real app, you might use a library like BeautifulSoup for robust parsing.
            summary = summary.replace('<b>', '').replace('</b>', '').replace('<i>', '').replace('</i>', '')
            print(f"--- Search successful. Summary found. ---")
            return summary
        else:
            print(f"--- Search completed, but no direct summary was found. ---")
            return "No summary found for that query."
            
    except requests.RequestException as e:
        print(f"--- Error during web search: {e} ---")
        return f"An error occurred while trying to search: {e}"


# --- 2. Configure the AI Model (Ollama) ---
# This mirrors the working weather example, pointing to a local Ollama instance.
# Ensure Ollama is running (`ollama serve`) and you have pulled the model (`ollama pull qwen3:8b`).
os.environ["OPENAI_API_KEY"] = "ollama"  # Dummy key required by the provider interface
ollama_provider = OpenAIProvider(base_url="http://localhost:11434/v1")
ollama_model = OpenAIChatModel(
    "qwen3:8b",
    provider=ollama_provider,
    settings=ModelSettings(temperature=0.1),  # Lower temperature for more reliable tool use
)


# --- 3. Create the Agent ---
# We create the agent, providing it with the LLM configuration and the list of
# available tools. In this case, it only has one tool: `search_web`.
agent = Agent(ollama_model, tools=[search_web])


# --- 4. Run the Agent to Find the Weather ---
# The user asks a question the LLM can't answer from its internal knowledge.
# The agent will recognize this, inspect its tools, and decide that `search_web`
# is the appropriate tool to use. It will then formulate a search query
# (e.g., "current weather in San Diego") and call the tool.
prompt = "What is the population of San Diego?"
print(f"Querying the agent with prompt: '{prompt}'\n")

response = agent.run_sync(prompt)

print("\n--- Final Agent Response ---")
# The final output is the LLM's natural language response, which is
# synthesized from the information returned by the search tool.
print(response)
```

TBD

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

TBD


