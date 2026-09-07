"""LangGraph agent example: a tool-calling loop built with StateGraph.

Replaces the higher-level LangChain ``create_agent`` with an explicit LangGraph
graph so you can see (and control) every step:

- ``llm_call`` node: invokes the Ollama chat model with the conversation so far;
  the model may return tool calls.
- ``tools`` node: a prebuilt ``ToolNode`` that executes the requested tools.
- ``should_continue`` edge: loops back to ``llm_call`` while the model keeps
  requesting tools, otherwise finishes.
"""

import sys
from pathlib import Path
from typing import Literal

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ollama_config import get_model

llm = ChatOllama(model=get_model())


@tool
def search(s_query: str) -> str:
    """Use DuckDuckGo to run a web search."""
    ddg_search = DuckDuckGoSearchRun()
    results = ddg_search.run(s_query)
    print(f"\n***************** Search Results:\n\n{results}\n\n")
    return results


@tool
def answer_from_search(original_query: str, search_results: str) -> str:
    """Given a user's original query and DuckDuckGo search results, return an answer."""
    messages = [
        {
            "role": "system",
            "content": "You are an expert at answering a question, given text that contains the answer.",
        },
        {
            "role": "user",
            "content": (
                f"For this original user question:\n\n{original_query}\n\n"
                f"Provide a concise answer given this context text:\n\n{search_results}"
            ),
        },
    ]
    response = llm.invoke(messages)
    r = response.content.strip()
    print(f"\n***************** Processed answer from answer_from_search:\n\n{r}\n\n")
    return r


SYSTEM_PROMPT = """You are a helpful assistant that follows these steps:
1. First use the 'search' tool to find relevant information
2. Then use 'answer_from_search' tool with both the original query and search results to provide a final answer
3. Always use both tools in sequence - search first, then answer_from_search
Make sure to pass both the original query and search results to answer_from_search."""

tools = [search, answer_from_search]
llm_with_tools = llm.bind_tools(tools)


def llm_call(state: MessagesState) -> dict:
    """Invoke the model; it either calls a tool or produces the final answer."""
    response = llm_with_tools.invoke(
        [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    )
    return {"messages": [response]}


tool_node = ToolNode(tools)


def should_continue(state: MessagesState) -> Literal["tools", END]:
    """Route to the tool node when the model requested tool calls, else stop."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END


# Build the graph: START -> llm_call -> (tools loop) -> END
builder = StateGraph(MessagesState)
builder.add_node("llm_call", llm_call)
builder.add_node("tools", tool_node)
builder.add_edge(START, "llm_call")
builder.add_conditional_edges("llm_call", should_continue, ["tools", END])
builder.add_edge("tools", "llm_call")
agent = builder.compile()

query = (
    "What city does Mark Watson live? Mark Watson who is an AI Practitioner "
    "and Consultant Specializing in Large Language Models, LangChain/Llama-Index "
    "Integrations, Deep Learning, and the Semantic Web."
)
agent_input = {"messages": [HumanMessage(content=query)]}

for step in agent.stream(agent_input, stream_mode="values"):
    message = step["messages"][-1]
    if isinstance(message, tuple):
        print(message)
    else:
        message.pretty_print()
