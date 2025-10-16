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
