## Note: This example is partially derived from the Agno docuentation 
## and converted to run with Ollama and several other with other modifications
## including the addition of a custom web scraping tool.
## URI for original source code:
## https://github.com/agno-agi/agno/blob/main/cookbook/examples/agents/research_agent.py
## that is licensed under the MPL-2.0 license.


"""🔍 Web Scraper and Question Answering Agent """

from textwrap import dedent
import os

from agno.agent import Agent
from agno.models.ollama import Ollama
import requests
from bs4 import BeautifulSoup
from agno.tools import tool

@tool
def scrape_website_content(url: str) -> str:
    """
    Fetches and extracts the clean, textual content from a given webpage URL.
    Use this tool when you need to read the contents of a specific web page to answer a question.
    
    Args:
        url (str): The full, valid URL of the webpage to be scraped (e.g., 'https://example.com').
        
    Returns:
        str: The extracted text content of the webpage, or a descriptive error message if scraping fails.
    """
    try:
        # It is a best practice to set a User-Agent header to mimic a real browser.
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for script_or_style in soup(["script", "style", "nav", "footer", "aside"]):
            script_or_style.decompose()
        text = soup.get_text(separator='\n', strip=True)
        ##print(text)  # Debugging: print the extracted text to console        
        if not text:
            return f"Successfully connected to {url}, but no text content could be extracted."
        return f"Successfully scraped content from {url}:\n\n{text}"
        
    except requests.RequestException as e:
        return f"Error scraping URL {url}: {e}"

# This custom tool can then be passed directly to an agent:
# from agno.agent import Agent
# custom_agent = Agent(
#     tools=[scrape_website_content],
#   ... # other agent configurations
# )

#####  Initialize the web scraping and analysis agent:
scraper_agent = Agent(
    model=Ollama(id="qwen3:30b"),
    tools = [scrape_website_content],
    description=dedent("""
        You are an expert web scraping and analysis agent. You follow a strict process:

        - Given a URL in a prompt, you will first use the appropriate tool to scrape its content.
        - You will then carefully read the scraped content to understand it thoroughly.
        - Finally, you will answer the user's question based *only* on the information contained within that specific URL's content.
    """),
    
    # The instructions are refined to provide a clear, step-by-step reasoning process for the LLM.
    instructions=dedent("""
        1. Scrape Phase 🕸️
           - Analyze the user's prompt to identify the target URL.
           - Invoke the `scrape` tool with the identified URL as the argument. Do not use the `search` or `crawl` tool unless explicitly instructed to do so.

        2. Analysis Phase 📊
           - Carefully read the entire markdown content returned by the `scrape` tool.
           - Systematically extract the specific pieces of information required to answer the user's question.

        3. Answering Phase ✍️
           - Formulate a concise and accurate answer to the user's question.
           - Your answer must be based exclusively on the information you extracted from the scraped webpage.
           - If the information is not present on the page, you must state that clearly rather than guessing.

        4. Quality Control ✓
           - Reread the original user query and your generated answer to ensure it is accurate, relevant, and directly addresses the user's request based on the provided source.
    """),
    
    expected_output=dedent("""
        # {Answer based on website content}
        
        **Source:** {URL provided by the user}
    """),
    
    markdown=True,
    show_tool_calls=True,
    add_datetime_to_instructions=True,
)

if __name__ == "__main__":
    prompt = "Using the web site https://markwatson.com Consultant Mark Watson has written Common Lisp, semantic web, Clojure, Java, and AI books. What musical instruments does he play?"
    
    scraper_agent.print_response(
        prompt,
        stream=True,
    )