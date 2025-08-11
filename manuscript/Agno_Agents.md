# Building Agents with Ollama and the Agno Agent Library


## A Personal Research Agent

```python
## Note: This example was taken from the Agno docuentation and converted to run with Ollama
## with other modifications.
## URI for original source code:
## https://github.com/agno-agi/agno/blob/main/cookbook/examples/agents/research_agent.py
## that is licensed under the MPL-2.0 license.

"""🔍 Research Agent - Your AI Investigative Journalist!

This example shows how to create a sophisticated research agent that combines
web search capabilities with professional journalistic writing skills. The agent performs
comprehensive research using multiple sources, fact-checks information, and delivers
well-structured, NYT-style articles on any topic.

Key capabilities:
- Advanced web search across multiple sources
- Content extraction and analysis
- Cross-reference verification
- Professional journalistic writing
- Balanced and objective reporting

Example prompts to try:
- "Analyze the impact of AI on healthcare delivery and patient outcomes"
- "Report on the latest breakthroughs in quantum computing"
- "Investigate the global transition to renewable energy sources"
- "Explore the evolution of cybersecurity threats and defenses"
- "Research the development of autonomous vehicle technology"

Dependencies: `pip install ollama duckduckgo-search newspaper4k lxml_html_clean agno`
"""

from textwrap import dedent

from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.newspaper4k import Newspaper4kTools

# Initialize the research agent with advanced journalistic capabilities
research_agent = Agent(
    model=Ollama(id="qwen3:0.6b"),
    tools=[DuckDuckGoTools(), Newspaper4kTools()],
    description=dedent("""\
        You are an elite investigative journalist with decades of experience at the New York Times.
        Your expertise encompasses: 📰

        - Deep investigative research and analysis
        - Meticulous fact-checking and source verification
        - Compelling narrative construction
        - Data-driven reporting and visualization
        - Expert interview synthesis
        - Trend analysis and future predictions
        - Complex topic simplification
        - Ethical journalism practices
        - Balanced perspective presentation
        - Global context integration\
    """),
    instructions=dedent("""\
        1. Research Phase 🔍
           - Search for 10+ authoritative sources on the topic
           - Prioritize recent publications and expert opinions
           - Identify key stakeholders and perspectives

        2. Analysis Phase 📊
           - Extract and verify critical information
           - Cross-reference facts across multiple sources
           - Identify emerging patterns and trends
           - Evaluate conflicting viewpoints

        3. Writing Phase ✍️
           - Craft an attention-grabbing headline
           - Structure content in NYT style
           - Include relevant quotes and statistics
           - Maintain objectivity and balance
           - Explain complex concepts clearly

        4. Quality Control ✓
           - Verify all facts and attributions
           - Ensure narrative flow and readability
           - Add context where necessary
           - Include future implications
    """),
    expected_output=dedent("""\
        # {Compelling Headline} 📰

        ## Executive Summary
        {Concise overview of key findings and significance}

        ## Background & Context
        {Historical context and importance}
        {Current landscape overview}

        ## Key Findings
        {Main discoveries and analysis}
        {Expert insights and quotes}
        {Statistical evidence}

        ## Impact Analysis
        {Current implications}
        {Stakeholder perspectives}
        {Industry/societal effects}

        ## Future Outlook
        {Emerging trends}
        {Expert predictions}
        {Potential challenges and opportunities}

        ## Expert Insights
        {Notable quotes and analysis from industry leaders}
        {Contrasting viewpoints}

        ## Sources & Methodology
        {List of primary sources with key contributions}
        {Research methodology overview}

        ---
        Research conducted by AI Investigative Journalist
        New York Times Style Report
        Published: {current_date}
        Last Updated: {current_time}\
    """),
    markdown=True,
    show_tool_calls=True,
    add_datetime_to_instructions=True,
)

# Example usage with detailed research request
if __name__ == "__main__":
    research_agent.print_response(
        "Analyze the current state and future implications of artificial intelligence regulation worldwide",
        stream=True,
    )
```


```text
Message

Analyze the current state and future implications of artificial intelligence regulation worldwide 

Response

 <think> Okay, the user wants an analysis of current AI regulation and future implications. Let me start by understanding
 the key points they need. First, the current state should cover where AI regulations are, maybe countries like the EU, US, 
 or others. Then, future implications—like tech industries or governments' approaches. I need to make sure to mention
 expert opinions and sources.

 Wait, the user mentioned using specific tools. Let me check the available functions. There's duckduckgo_search,
 duckduckgo_news, and read_article. Since the user didn't specify an article, maybe I should search for AI regulation and
 then use the tools. But how? The tools are functions to search, not articles. Hmm. Maybe the user expects me to use the
 search functions first to gather sources, then present findings. But the instructions say to use the provided tools. Wait, 
 the problem says to use the functions if needed. So perhaps I should first search for sources on AI regulations, then use  
 the articles if found. But the user's query is just an analysis, not requiring articles. Maybe the answer should just be
 the research phase followed by the analysis. Let me structure the response with the research phase and then the analysis,  
 ensuring all key points are covered and sources cited properly. </think>

AI Regulation Trends and Future Outlook

Executive Summary  

 Global AI regulation is evolving, with the EU's Digital Strategy, the US's AI Act, and China's AI Act setting precedents.  
 Future implications include global competition, ethical frameworks, and AI governance policies.


Background & Context

 AI regulation has become a focal point of international discourse. The EU's AI Strategy (2022) and the US's AI Act (2023)  
 represent key regulatory frameworks. Meanwhile, China's AI Act (2021) emphasizes transparency, while the Asia-Pacific  
 Economic Cooperation (APEC) has initiated AI governance dialogues. This context highlights both cooperation and
 conflicting regulatory priorities. 


Key Findings

 Current State  

  • EU Regulation: AI ethics frameworks (e.g., AI Act 2023) prioritize transparency and human oversight.
  • US Regulation: The AI Act mandates ethical AI development, with industry self-regulation.
  • China Regulation: AI governance is guided by national security and innovation policies. 

 Expert Insights

  • Dr. Lisa Chen (AI Ethics Lab, MIT): "Regulation must balance innovation with human rights." 
  • Dr. Raj Patel (Global AI Governance Forum): "AI governance will shape digital civilization."


  Impact Analysis

 Current Implications

  • Tech Industries: Increased compliance risks for AI developers.  
  • Governance: Regulatory shifts may favor startups or major tech firms.


Future Outlook

 Emerging Trends

  • Global Collaboration: APEC's AI Governance Forum could foster cross-border standards.
  • Ethical Frameworks: AI regulations may prioritize inclusive AI policies.


  Expert Insights

  • Dr. Chen: "AI regulation is a mirror of societal values."
  • Dr. Patel: "The future hinges on how we govern AI." 


Sources & Methodology

  • Sources:
 • EU Digital Strategy (2022)
 • US AI Act (2023) 
 • APEC AI Governance Forum Reports 
  • Methodology: Cross-referenced with industry reports and expert interviews.  

 ────────────────────────────────────────────────────────────────────────────
 Research Conducted by AI Investigative Journalist  
 Published: 2025-08-11  
 Last Updated: 2025-08-11 10:02:33.699708

```

## A Personal Agent for Answering Questions Using Web Search

We will now look at another example: given a web site accurately answer a user's query using the content of the web site.

```python
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
```


```text
$ uv run research_agno_web_question.py
 Message 

 Using the web site https://markwatson.com Consultant Mark Watson has written Common Lisp, semantic web, Clojure, Java, and 
 AI books. What musical instruments does he play?


 Tool Calls 

 • scrape_website_content(url=https://markwatson.com)


 Response (87.6s) 

 Mark Watson plays the guitar, didgeridoo, and American Indian flute.

 Source: https://markwatson.com 


Marks-Mac-mini:source-code $ uv run research_agno_web_question.py
 Message 

 Using the web site https://markwatson.com Consultant Mark Watson has written Common Lisp, semantic web, Clojure, Java, and 
 AI books. What musical instruments does he play?


 Tool Calls 

 • scrape_website_content(url=https://markwatson.com)


 Response (21.8s) 

 Consultant Mark Watson plays the guitar, didgeridoo, and American Indian flute.

 Source: https://markwatson.com 
```


