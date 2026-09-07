
# LangGraph

LangGraph is good for building stateful, multi-step LLM applications, things like agents, workflows, or multi-tool reasoning pipelines using a graph-based execution model instead of simple chains.

Key Benefits:
- State management: Maintains conversation or agent memory across turns.
- Control flow: Lets you branch, loop, and merge tasks dynamically (unlike LangChain's mostly linear chains).
- Tool orchestration: Coordinates multiple tools or models (e.g., search + code interpreter + summarizer).
- Persistence: Supports saving/reloading graph state for long-running agent sessions.

LangGraph combines LangChain-style models and tools, DAG-style control flow, and memory persistence, and is useful for building complex agent systems that need structured, inspectable logic.

**Note:** This example builds the agent loop explicitly with LangGraph's `StateGraph` API rather than a prebuilt helper such as `create_agent`. Writing the graph out by hand shows exactly where the model is called, where tools run, and how the loop decides to stop, which makes the control flow much easier to customize or extend.

![Architecture diagram](images/langgraph_architecture.png)

We look at one example here, in the file **Ollama_in_Action_Book/source-code/langgraph/langgraph_agent_test.py**. It implements a ReAct-style (Reasoning and Acting) agent as a small graph with two nodes and one conditional edge. An `llm_call` node invokes a ChatOllama model bound to two tools, a prebuilt `ToolNode` executes whichever tools the model requests, and a `should_continue` function on a conditional edge routes execution back to `llm_call` whenever the model asks for another tool call, or ends the run when the model produces a final answer. This is the fundamental agent loop, expressed as an explicit graph instead of being hidden inside a helper.

The two tools are a `search` function that runs a DuckDuckGo web search and an `answer_from_search` function that sends the original question plus the retrieved text to the same local LLM to synthesize a concise answer. Keeping retrieval and synthesis as separate tool steps lets the model decide when it has enough information, instead of forcing a fixed pipeline. Here is the complete listing:

```python
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
```

## How the graph works

The state of the graph is `MessagesState`, which is simply a running list of chat messages. Each node receives the current state and returns an update that LangGraph merges back in.

- **`llm_call`** prepends the system prompt to the conversation so far and invokes `llm_with_tools`. Because the model is bound to the tools, its response is either a normal reply or an AI message whose `tool_calls` field names one or more tools and the arguments to pass them. The response is wrapped in `{"messages": [response]}` so LangGraph appends it to the state.
- **`tool_node`** is LangGraph's prebuilt `ToolNode`. It reads the tool calls off the last message, runs each matching `@tool` function (here `search` or `answer_from_search`), and appends the results as `ToolMessage` entries. Using `ToolNode` means we do not have to write the dispatch and error handling ourselves.
- **`should_continue`** inspects the last message. If the model requested tool calls it returns `"tools"`, sending execution to the tool node; otherwise it returns `END`, finishing the graph. This single conditional edge is the whole agent loop: `START` goes into `llm_call`, `llm_call` conditionally loops through `tools` and back, and eventually exits at `END`.

Because the loop is spelled out as edges, it is easy to change. You could cap the number of iterations, add human-in-the-loop approval before a tool runs, or route to different tools based on the query, all by editing nodes and edges rather than a hidden framework loop.

The streaming at the end uses `agent.stream(..., stream_mode="values")`, which yields the full state after each step. Printing the last message each time gives a live trace of the reasoning: each AI message (with its tool call requests) and each tool result as the loop runs, until the model settles on a final answer.

Here is some sample output. Exact behavior varies between runs and between models, since the agent decides at runtime how many searches to perform before answering:

```
$ uv run langgraph_agent_test.py
================================ Human Message =================================

What city does Mark Watson live? Mark Watson who is an AI Practitioner and Consultant Specializing in Large Language Models, LangChain/Llama-Index Integrations, Deep Learning, and the Semantic Web.
================================== Ai Message ==================================
Tool Calls:
  search (f240ff76-65e1-4868-abfc-e27f7e5020bf)
 Call ID: f240ff76-65e1-4868-abfc-e27f7e5020bf
  Args:
    s_query: Mark Watson AI Practitioner Consultant Large Language Models LangChain Llama-Index Integrations Deep Learning Semantic Web

***************** Search Results:

LangChain is an open source framework with a pre-built agent architecture and integrations for any model or tool ...

================================= Tool Message =================================
Name: search

LangChain is an open source framework with a pre-built agent architecture and integrations for any model or tool ...
================================== Ai Message ==================================
Tool Calls:
  search (23ba2296-68e7-4d58-bb81-e1abe364c7ea)
 Call ID: 23ba2296-68e7-4d58-bb81-e1abe364c7ea
  Args:
    s_query: Mark Watson AI Practitioner Consultant Deep Learning semantic web bio location home office

...

================================== Ai Message ==================================

Based on my search results for Mark Watson (AI Practitioner and Consultant Specializing in Large Language Models, LangChain/Llama-Index Integrations, Deep Learning, and the Semantic Web), he lives in **Flagstaff, Arizona**.
```

Reading the trace from top to bottom shows the loop in action. The human message enters the graph, then `llm_call` returns an AI message requesting `search` (you can see the tool call ID and the generated `s_query` argument). `should_continue` routes to the tool node, which runs DuckDuckGo and appends the raw results as a `Tool Message`. Control returns to `llm_call`, which on this run judged the first results too generic and issued a second, more specific search. After a couple of iterations the model had enough context and, instead of requesting another tool, returned a plain AI message with the final answer, Flagstaff, Arizona. At that point `should_continue` returned `END` and the run finished.

Note that the agent had to try searching several times before finding the "correct Mark Watson." With a larger model the agent is more likely to also use the `answer_from_search` tool exactly as the system prompt describes; smaller local models sometimes answer directly once a search returns a confident snippet. Either way, the same graph drives both behaviors, which is the point of writing the loop explicitly: the control flow is fixed and inspectable even though the model's decisions are not.

## Wrap Up

This chapter built a small agent from first parts: a model-bound `llm_call` node, a prebuilt `ToolNode`, and a conditional `should_continue` edge, all wired into a `StateGraph` over `MessagesState`. That three-piece pattern (model node, tool node, continue-or-stop edge) is the core of most LangGraph agents. Once you are comfortable with it you can grow the graph by adding more nodes (routing, validation, memory persistence) and more edges (branching, parallel tool workers) without changing how the loop fundamentally works. Because the graph is explicit, every step is visible in the stream and every branch is a line of code you control.

## Optional Practice Problems

1. **Implement a Router Node.** Extend the LangGraph agent in `langgraph_agent_test.py` by adding a conditional router node. The router should inspect the initial user query and determine if it requires a web search. If the query can be answered directly (e.g., "What is 2+2?"), route the state directly to an answer generation node, bypassing the search tool completely.

2. **Add Search History to Graph State.** Modify the state definition of the graph to include a list field `search_history: list[str]`. Each time the search node is executed, append the query to this list. If the agent generates a search query that matches an entry in the list, redirect it to refine its search query to avoid infinite loops.

3. **Graph Architecture Visualization.** Use LangGraph's built-in visualization utility. Write a short snippet in your python script that calls `agent.get_graph().draw_mermaid_png()` and saves the resulting image to disk. Verify the flow of nodes and edges matches your code definition.

4. **Verify Answers Node.** Introduce a validation node called `answer_verifier`. Once the agent produces a final candidate answer, this node should use the LLM to verify if the answer completely answers the user's initial query. If it does, route to `END`; if not, route back to the tool/search step with feedback on what was missing.
