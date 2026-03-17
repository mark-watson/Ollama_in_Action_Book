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
