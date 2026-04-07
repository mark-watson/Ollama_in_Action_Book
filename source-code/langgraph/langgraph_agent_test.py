from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_ollama import ChatOllama
import json
from pprint import pprint
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ollama_config import get_client, get_model

client = get_client()

@tool
def search(s_query: str) -> str:
    """Use DuckDuckGo to run a web search."""
    ddg_search = DuckDuckGoSearchRun()  # Changed from DuckDuckGoSearchTool
    results =  ddg_search.run(s_query)
    print(f"\n***************** Search Results:\n\n{results}\n\n")
    return results

@tool
def answer_from_search(original_query: str, search_results: str) -> str:
    """Given a user's original query and DuckDuckGo search results, return an answer."""
    messages = [
        {"role": "system", "content": "You are an expert at answering a question, given text that contains the answer."},
        {"role": "user",
         "content": f"For this original user question:\n\n{original_query}\n\nProvide a concise answer given this context text:\n\n{search_results}"},
    ]

    response = client.chat(
        model=get_model(),
        messages=messages,
    )

    r = response.message.content.strip()
    print(f"\n***************** Processed answer from answer_from_search:\n\n{r}\n\n")
    return r


model = ChatOllama(model=get_model())
tools = [search, answer_from_search]
agent = create_agent(model,
                     tools,
                     system_prompt="""You are a helpful assistant that follows these steps:
1. First use the 'search' tool to find relevant information
2. Then use 'answer_from_search' tool with both the original query and search results to provide a final answer
3. Always use both tools in sequence - search first, then answer_from_search
Make sure to pass both the original query and search results to answer_from_search.""")

query = "What city does Mark Watson live? Mark Watson who is an AI Practitioner and Consultant Specializing in Large Language Models, LangChain/Llama-Index Integrations, Deep Learning, and the Semantic Web."
agent_input = {"messages": [("human", query)]}

for s in agent.stream(agent_input, stream_mode="values"):
    message = s["messages"][-1]
    if isinstance(message, tuple):
        print(message)
    else:
        message.pretty_print()