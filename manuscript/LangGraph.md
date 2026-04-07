
# LangGraph

LangGraph is good for building stateful, multi-step LLM applications, things like agents, workflows, or multi-tool reasoning pipelines using a graph-based execution model instead of simple chains.

Key Benefits:
- State management: Maintains conversation or agent memory across turns.
- Control flow: Lets you branch, loop, and merge tasks dynamically (unlike LangChain's mostly linear chains).
- Tool orchestration: Coordinates multiple tools or models (e.g., search + code interpreter + summarizer).
- Persistence: Supports saving/reloading graph state for long-running agent sessions.

LangGraph is a combination or LangChain + DAG-style control + memory persistence and is useful for building complex agent systems that need structured, inspectable logic.

**Note:** LangGraph v1 (released 2025) deprecated `create_react_agent` in favor of `langchain.agents.create_agent`. The example code has been updated to use the new API.

We only look at one short example here that is in the file **Ollama_in_Action_Book/source-code/langgraph/langgraph_agent_test.py**. The following listing demonstrates the construction of a sophisticated ReAct (Reasoning and Acting) agent using LangChain's `create_agent` function, specifically designed to orchestrate a multi-step research workflow. By leveraging `create_agent`, the code integrates a ChatOllama local language model with custom-defined tools to bridge the gap between static model training data and real-time information. The implementation defines two specialized tools: a search function utilizing DuckDuckGo for web retrieval and a secondary processing tool that uses the local LLM to synthesize those search results into a coherent answer. This architecture goes beyond simple retrieval-augmented generation (RAG) by instructing the agent, through a structured system prompt, to follow a strict sequential logic—first searching for raw data and then passing that data back through a refinement tool—to ensure the final output is grounded in the most current available facts.

```python
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
```

The core logic of this implementation lies in the functional separation of concerns between data retrieval and data synthesis. By wrapping **DuckDuckGoSearchRun** and a custom LLM synthesis call as @tool decorated functions, we allow the LangChain agent to treat external API calls and internal model reasoning as interchangeable actions. This is particularly evident in the answer_from_search tool, which creates a secondary LLM context to filter through the noise of raw search results, effectively acting as a "cleaner" for the information retrieved in the first step.

The agent's behavior is governed by the `system_prompt` passed to `create_agent`, which provides the necessary "cognitive" guardrails for the model. Using `agent.stream` with the values mode allows us to observe the agent's internal monologue and tool execution in real-time, providing transparency into how the ReAct loop iterates through the search and synthesis phases before arriving at the final response. This pattern is highly extensible, forming a blueprint for complex autonomous agents that need to interact with various local and remote resources.

Here is some sample output:

```
$ uv run langgraph_agent_test.py
================================ Human Message =================================

What city does Mark Watson live? Mark Watson who is an AI Practitioner and Consultant Specializing in Large Language Models, LangChain/Llama-Index Integrations, Deep Learning, and the Semantic Web.
================================== Ai Message ==================================
Tool Calls:
  search (b65452dd-563c-4f12-bd59-1a4b695e75e3)
 Call ID: b65452dd-563c-4f12-bd59-1a4b695e75e3
  Args:
    s_query: Mark Watson AI Practitioner large language models who lives where

***************** Search Results:

AIGeneralist and Consultant Specializing inLargeLanguageModels, LangChain/Llama-Index Integrations, Deep Learning, the Semantic Web, and building ... The IBM Win WithAISummit was a powerful event for IBM Business Partners andAIpractitioners. ...whois a data driven marketer or well, frankly, ... ... underpinnings of itsmodels, assess market and ethical implications, and offer a forward-looking perspective on the long-term trends shapingAI... As an example, he points to disinformation researchers,whoembrace corporate talking points of a recommendationmodelthat can meaningfully ... Snowflake CortexAIintegrates secure access to cutting-edgelargelanguagemodelsandAIservices, empowering enterprises to accelerateAI-driven ...


================================= Tool Message =================================
Name: search

AIGeneralist and Consultant Specializing inLargeLanguageModels, LangChain/Llama-Index Integrations, Deep Learning, the Semantic Web, and building ... The IBM Win WithAISummit was a powerful event for IBM Business Partners andAIpractitioners. ...whois a data driven marketer or well, frankly, ... ... underpinnings of itsmodels, assess market and ethical implications, and offer a forward-looking perspective on the long-term trends shapingAI... As an example, he points to disinformation researchers,whoembrace corporate talking points of a recommendationmodelthat can meaningfully ... Snowflake CortexAIintegrates secure access to cutting-edgelargelanguagemodelsandAIservices, empowering enterprises to accelerateAI-driven ...
================================== Ai Message ==================================
Tool Calls:
  search (bed9010a-180e-40dc-81c4-8d92f5b972c1)
 Call ID: bed9010a-180e-40dc-81c4-8d92f5b972c1
  Args:
    s_query: Mark Watson AI practitioner who lives where does he live

***************** Search Results:

Bestselling authorMarkManson explains why he built Purpose, anA.I. mentor designed to challenge users and offer structured guidance. Inside AuthorMarkManson's Quest to Create anAILife Coach That Doesn't Suck The self-help guru is among the first to launch anAIcoaching app. He won't be the last. MarkManson is launching anAIapp to help more people 'Subtly not Give a F*ck' The best-selling author teamed up with a serial tech entrepreneur to deliver personalized, scalable life ... MarkManson'sAIapp, Purpose, isn't another motivational gadget dressed in shiny tech. It's a structured conversation partner designed to help you get clear on what matters, confront your patterns, and take practical steps toward a life that feels intentional rather than reactive. Purpose, the world's firstAImentor, launched publicly today, with bestselling authorMarkManson as co-founder alongside CEO andAIentrepreneur Raj Singh. The platform creates an entirely new ...


================================= Tool Message =================================
Name: search

Bestselling authorMarkManson explains why he built Purpose, anA.I. mentor designed to challenge users and offer structured guidance. Inside AuthorMarkManson's Quest to Create anAILife Coach That Doesn't Suck The self-help guru is among the first to launch anAIcoaching app. He won't be the last. MarkManson is launching anAIapp to help more people 'Subtly not Give a F*ck' The best-selling author teamed up with a serial tech entrepreneur to deliver personalized, scalable life ... MarkManson'sAIapp, Purpose, isn't another motivational gadget dressed in shiny tech. It's a structured conversation partner designed to help you get clear on what matters, confront your patterns, and take practical steps toward a life that feels intentional rather than reactive. Purpose, the world's firstAImentor, launched publicly today, with bestselling authorMarkManson as co-founder alongside CEO andAIentrepreneur Raj Singh. The platform creates an entirely new ...
================================== Ai Message ==================================
Tool Calls:
  search (a1e0a01a-ead6-4ec6-9615-3b97ce5473fa)
 Call ID: a1e0a01a-ead6-4ec6-9615-3b97ce5473fa
  Args:
    s_query: "Mark Watson" AI Practitioner consultant who lives where

***************** Search Results:

I am looking for part time work as an advisor andconsultant. · Experience: Mind AI · Education: University of California, Santa Barbara · Location: Flagstaff · 500+ connections on LinkedIn.While I have been working as a remoteconsultantsince 1998 I also work onsite as required at higher consulting rates. I offer both priority consulting services ($100/hour) and also non-priority services ($50/hour).MarkWatsonis a consultant who works out of his home office in Sedona Arizona. His consulting customer list includes: Google, CompassLabs, Disney, Arctan Group, Sitescout.com, Embed.ly, and Webmind Corporation.MarkWatsonhas been working as aConsultantatMarkWatsonfor 26 years. MarkWatsonis part of the Business Services industry, and located in Arizona, United States.Jul 17, 2024 ·He currently serves as the Chief Architect at Mind AI. 3 With a career spanning several decades,Watsonhas established himself as an expert in various areas ofAI, including machine learning, semantic web technologies, and natural language processing. 1I now focus on writing books and experimenting withAIas a semi-retired independent researcher. I have worked in the fields of artificial intelligence, machine learning, semantic web and linked data, and natural language processing since 1982.LocalAIand IT consulting for small and medium-sized businesses in Flagstaff, Arizona. Expertise in Large Language Models (LLMs), AI integration, and more. While I have been working as a remoteconsultantsince 1998 I also work onsite as required at higher consulting rates. I offer both priority consulting services ($100/hour) and also non-priority services ($50/hour). MarkWatsonhas been working as aConsultantatMarkWatsonfor 26 years. MarkWatsonis part of the Business Services industry, and located in Arizona, United States. Jul 17, 2024 ·He currently serves as the Chief Architect at Mind AI. 3 With a career spanning several decades,Watsonhas established himself as an expert in various areas ofAI, including machine learning, semantic web technologies, and natural language processing. 1 I now focus on writing books and experimenting withAIas a semi-retired independent researcher. I have worked in the fields of artificial intelligence, machine learning, semantic web and linked data, and natural language processing since 1982.


================================= Tool Message =================================
Name: search

I am looking for part time work as an advisor andconsultant. · Experience: Mind AI · Education: University of California, Santa Barbara · Location: Flagstaff · 500+ connections on LinkedIn.While I have been working as a remoteconsultantsince 1998 I also work onsite as required at higher consulting rates. I offer both priority consulting services ($100/hour) and also non-priority services ($50/hour).MarkWatsonis a consultant who works out of his home office in Sedona Arizona. His consulting customer list includes: Google, CompassLabs, Disney, Arctan Group, Sitescout.com, Embed.ly, and Webmind Corporation.MarkWatsonhas been working as aConsultantatMarkWatsonfor 26 years. MarkWatsonis part of the Business Services industry, and located in Arizona, United States.Jul 17, 2024 ·He currently serves as the Chief Architect at Mind AI. 3 With a career spanning several decades,Watsonhas established himself as an expert in various areas ofAI, including machine learning, semantic web technologies, and natural language processing. 1I now focus on writing books and experimenting withAIas a semi-retired independent researcher. I have worked in the fields of artificial intelligence, machine learning, semantic web and linked data, and natural language processing since 1982.LocalAIand IT consulting for small and medium-sized businesses in Flagstaff, Arizona. Expertise in Large Language Models (LLMs), AI integration, and more. While I have been working as a remoteconsultantsince 1998 I also work onsite as required at higher consulting rates. I offer both priority consulting services ($100/hour) and also non-priority services ($50/hour). MarkWatsonhas been working as aConsultantatMarkWatsonfor 26 years. MarkWatsonis part of the Business Services industry, and located in Arizona, United States. Jul 17, 2024 ·He currently serves as the Chief Architect at Mind AI. 3 With a career spanning several decades,Watsonhas established himself as an expert in various areas ofAI, including machine learning, semantic web technologies, and natural language processing. 1 I now focus on writing books and experimenting withAIas a semi-retired independent researcher. I have worked in the fields of artificial intelligence, machine learning, semantic web and linked data, and natural language processing since 1982.
================================== Ai Message ==================================
Tool Calls:
  answer_from_search (baf05b27-29fa-4614-96ff-2a787af3da4a)
 Call ID: baf05b27-29fa-4614-96ff-2a787af3da4a
  Args:
    original_query: What city does Mark Watson live?
    search_results: I am looking for part time work as an advisor and consultant. · Experience: Mind AI · Education: University of California, Santa Barbara · Location: Flagstaff · 500+ connections on LinkedIn. While I have been working as a remote consultants since 1998 I also work onsite as required at higher consulting rates. I offer both priority consulting services ($100/hour) and also non-priority services ($50/hour). Mark Watson is a consultant who works out of his home office in Sedona Arizona. His consulting customer list includes: Google, CompassLabs, Disney, Arctan Group, Sitescout.com, Embed.ly, and Webmind Corporation. Mark Watson has been working as a Consultant at Mark Watson for 26 years. Mark Watson is part of the Business Services industry, and located in Arizona, United States. July 17, 2024 ·He currently serves as the Chief Architect at Mind AI. 3 With a career spanning several decades, Watson has established himself as an expert in various areas of AI, including machine learning, semantic web technologies, and natural language processing. 1 I now focus on writing books and experimenting with AI as a semi-retired independent researcher. I have worked in the fields of artificial intelligence, machine learning, semantic web and linked data, and natural language processing since 1982. Local AI and IT consulting for small and medium-sized businesses in Flagstaff, Arizona. Expertise in Large Language Models (LLMs), AI integration, and more. While I have been working as a remote consultant since 1998 I also work onsite as required at higher consulting rates. I offer both priority consulting services ($100/hour) and also non-priority services ($50/hour). Mark Watson is a consultant who works out of his home office in Sedona Arizona. His consulting customer list includes: Google, CompassLabs, Disney, Arctan Group, Sitescout.com, Embed.ly, and Webmind Corporation. Mark Watson has been working as a Consultant at Mark Watson for 26 years. Mark Watson is part of the Business Services industry, and located in Arizona, United States. July 17, 2024 ·He currently serves as the Chief Architect at Mind AI. 3 With a career spanning several decades, Watson has established himself as an expert in various areas of AI, including machine learning, semantic web technologies, and natural language processing. 1 I now focus on writing books and experimenting with AI as a semi-retired independent researcher. I have worked in the fields of artificial intelligence, machine learning, semantic web and linked data, and natural language processing since 1982.

***************** Processed answer from answer_from_search:

Sedona, Arizona.


================================= Tool Message =================================
Name: answer_from_search

Sedona, Arizona.
```

Note that the agent had to try searching several times before finding the "correct Mark Watson."
