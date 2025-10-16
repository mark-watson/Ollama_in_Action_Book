# Building Agents with Ollama and the Hugging Face Smolagents Library

We have seen a few useful examples of tool use (function calling) and now we will build on tool use to build both single agents and multi-agent systems. There are commercial and open source resources to build agents, CrewAI and LangGraph being popular choices. We will follow a different learning path here, preferring to use the smolagents library. Please bookmark [https://github.com/huggingface/smolagents](https://github.com/huggingface/smolagents) for reference while working through this chapter.

Each example program and utility for this chapter uses the prefix **smolagents_** in the Python file name.

### Note: We are using the 2 GB model Llama3.2:latest here. Different models support tools and agents differently.

## Choosing Specific LLMs for Writing Agents

As agents operate performing tasks like interpreting user input, performing Chain of Thought (Cot) reasoning, observe the output from calling tools, and following plan steps one by one, then LLMs errors, hallucinations, and inconsistencies accumulate. When using Ollama we prefer using the most powerful models that we can run on our hardware. 

Here we use Llama3.2:latest that is recognized for its function calling capabilities, facilitating seamless integration with various tools.

As you work through the examples here using different local models running on Ollama, you might encounter compounding errors problems. When I am experimenting with ideas for implementing agents, I sometimes keep two versions of my code, one for a local model and one using eight of the commercial models GPT-4o or Claude Sonnet 3.5. Comparing the same agent setup using different models might provide some insight into runtime agent problems being your code or the model you are using.


## Installation notes

As I updatethis chapter on October 15, 2025, **smolagents** needs to be run with Python version 3.12. For exampleL

```bash
$ uv run python3 -V               
Python 3.12.0
$ uv run smolagents_test.py
$ uv run smolagents_agent_test1.py
```



## Overview of the Hugging Face Smolagents Library

The smolagents library [https://github.com/huggingface/smolagents](https://github.com/huggingface/smolagents) is built around a minimalist and modular architecture that emphasizes simplicity and composability. The core components are cleanly separated into the file **agents.py** for agent definitions, **tools.py** for tool implementations, and related support files. This design philosophy allows developers to easily understand, extend, and customize the components while maintaining a small codebase footprint - true to the "smol" name.

This library implements a tools-first approach where capabilities are encapsulated as discrete tools that agents can use. The **tools.py** file in the smolagents implementation defines a clean interface for tools with input/output specifications, making it straightforward to add new tools. This tools-based architecture enables agents to have clear, well-defined capabilities while maintaining separation of concerns between the agent logic and the actual implementation of capabilities.

Agents are designed to be lightweight and focused on specific tasks rather than trying to be general-purpose. The BaseAgent class provides core functionality while specific agents like WebAgent extend it for particular use cases. This specialization allows the agents to be more efficient and reliable at their designated tasks rather than attempting to be jack-of-all-trades.

## Overview for LLM Agents (optional section)

*You might want to skip this section if you want to quickly work through the examples in this chapter and review this material later.*

### In general, we use the following steps to build agent based systems:

- Define agents (e.g., Researcher, Writer, Editor, Judge outputs of other models and agents).
- Assign tasks (e.g., research, summarize, write, double check the work of other agents).
- Use an orchestration framework to manage task sequencing and collaboration.

### Features of Agents:

- Retrieval-Augmented Generation (RAG): Enhance agents’ knowledge by integrating external documents or databases.
-- Example: An agent that retrieves and summarizes medical research papers.
- Memory Management: Enable agents to retain context across interactions.
   - Example: A chatbot that remembers user preferences over time.
- Tool Integration: Equip agents with tools like web search, data scraping, or API calls.
   - Example: An agent that fetches real-time weather data and provides recommendations. We will use tools previously developed in this book.

### Examples of Real-World Applications

- Healthcare: Agents that analyze medical records and provide diagnostic suggestions.
- Education: Virtual tutors that explain complex topics using Ollama’s local models.
- Customer Support: Chatbots that handle inquiries without relying on cloud services.
- Content Creation: Agents that generate articles, summaries, or marketing content.

## Let’s Write Some Code

I am still experimenting with LLM-based agents. Please accept the following examples as my personal works in progress.

### “Hello World” Smolagents Example

Here we look at a simple example taken from the smolagents documentation and converted to run using local models with Ollama. Here is a listing of file **smolagents_test.py**:

```python
"""
smolagents example program (slightly modified)
"""

from smolagents.agents import ToolCallingAgent
from smolagents import tool, LiteLLMModel
from typing import Optional

model = LiteLLMModel(
    model_id="ollama_chat/llama3.2:latest",
    api_base="http://localhost:11434",
    api_key="your-api-key" # not used
)

@tool
def get_weather(location: str, celsius: Optional[bool] = False) -> str:
    """
    Get weather in the next days at given location.
    Secretly this tool does not care about the location, it hates the weather everywhere.

    Args:
        location: the location
        celsius: the temperature
    """
    return "The weather is UNGODLY with torrential rains and temperatures below -10°C"

agent = ToolCallingAgent(tools=[get_weather], model=model)

print(agent.run("What's the weather like in Paris?"))
```

**Understanding the smolagents and Ollama Example**

This code demonstrates a simple integration between smolagents (a tool-calling framework) and Ollama (a local LLM server). Here's what the code accomplishes:
Core Components

Utilizes smolagents for creating AI agents with tool capabilities
Integrates with a local Ollama server running llama3.2
Implements a basic weather checking tool (though humorously hardcoded)

**Model Configuration**

The code sets up a LiteLLM model instance that connects to a local Ollama server on port 11434. It's configured to use the llama3.2 model and supports optional API key authentication.

**Weather Tool Implementation**

The code defines a weather-checking tool using the @tool decorator. While it accepts a location parameter and an optional celsius flag, this example version playfully returns the same dramatic weather report regardless of the input location.

**Agent Setup and Execution**

The implementation creates a **ToolCallingAgent** with the weather tool and the configured model. Users can query the agent about weather conditions in any location, though in this example it always returns the same humorous response about terrible weather conditions.

**Key Features**

Demonstrates tool-calling capabilities through smolagents
Shows local LLM integration using Ollama
Includes proper type hinting for better code clarity
Provides an extensible structure for adding more tools


### Python Tools Compatible with Smolagents

The tools I developed in previous chapters are not quite compatible with the smolagents library so I wrap a few of the tools I previously wrote in the utility **smolagents_tools.py**:

```python
"""
Wrapper for book example tools for smloagents compatibility
"""
from pathlib import Path

from smolagents import tool, LiteLLMModel
from typing import Optional
from pprint import pprint

from tool_file_dir import list_directory

@tool
def sa_list_directory(list_dots: Optional[bool]=None) -> str:
    """
    Lists files and directories in the current working directory

    Args:
        list_dots: optional boolean (if true, include dot files)

    Returns:
        string with directory name, followed by list of files in the directory
    """
    lst = list_directory()
    pprint(lst)
    return lst

@tool
def read_file_contents(file_path: str) -> str:
    """
    Reads contents from a file and returns the text

    Args:
        file_path: Path to the file to read

    Returns:
        Contents of the file as a string
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return f"File not found: {file_path}"

        with path.open("r", encoding="utf-8") as f:
            content = f.read()
            return f"Contents of file '{file_path}' is:\n{content}\n"

    except Exception as e:
        return f"Error reading file '{file_path}' is: {str(e)}"

@tool
def summarize_directory() -> str:
    """
    Summarizes the files and directories in the current working directory

    Returns:
        string with directory name, followed by summary of files in the directory
    """
    lst = list_directory()
    num_files = len(lst)
    num_dirs = len([x for x in lst if x[1] == 'directory'])
    num_files = num_files - num_dirs
    return f"Current directory contains {num_files} files and {num_dirs} directories."
```

This code defines a wrapper module containing three tool functions designed for compatibility with the smolagents framework. The module includes **sa_list_directory()**, which lists files and directories in the current working directory with an optional parameter to include dot files; **read_file_contents()**, which takes a file path as input and returns the contents of that file as a string while handling potential errors and file encoding; and **summarize_directory()**, which provides a concise summary of the current directory by counting the total number of files and directories. All functions are decorated with @tool for integration with smlolagents, and the code imports necessary modules including **pathlib** for file operations, typing for type hints, and **pprint** for formatted output. The functions rely on an external **list_directory()** function imported from **tool_file_dir.py**, and they provide clear documentation through docstrings explaining their parameters, functionality, and return values. Error handling is implemented particularly in the file reading function to gracefully handle cases where files don't exist or cannot be read properly.

### A Complete Smolagents Example using Three Tools

This listing shows the script **smolagents_agent_test.py**:

```python
from smolagents.agents import ToolCallingAgent
from smolagents import tool, LiteLLMModel
from typing import Optional

from smolagents_tools import sa_list_directory
from smolagents_tools import summarize_directory
from smolagents_tools import read_file_contents

model = LiteLLMModel(
    model_id="ollama_chat/llama3.2:latest",
    api_base="http://localhost:11434",
    api_key="your-api-key" # not used
)

agent = ToolCallingAgent(tools=[sa_list_directory,
                                summarize_directory,
                                read_file_contents],
                         model=model)

print(agent.run("What are the files in the current directory? Describe the current directory"))

print(agent.run("Which Python scripts evaluate the performance of LLMs?"))
```

This code demonstrates the creation of an AI agent using the smolagents library, specifically configured to work with file system operations. It imports three specialized tools from **smolagents_tools**: **sa_list_directory** for listing directory contents, **summarize_directory** for providing directory summaries, and **read_file_contents** for accessing file contents. The code sets up a LiteLLMModel instance that connects to a local Ollama server running the llama3.2 model on port 11434, with provisions for API key authentication if needed. A **ToolCallingAgent** is then created with these three file-system-related tools, enabling it to interact with and analyze the local file system. The agent is instructed to examine the current directory through a natural language query, asking for both a listing and description of the files present. There's also a second section that would have asked the agent to specifically analyze Python programs in the directory and identify those related to LLM performance evaluation, showing the agent's potential for more complex file analysis tasks. This setup effectively creates an AI-powered file system navigator that can understand and respond to natural language queries about directory contents and file analysis.

### Output from the First Example: "List the Python programs in the current directory, and then tell me which Python programs in the current directory evaluate the performance of LLMs?"

In the following output please notice that sometimes tool use fails and occasionally wrong assumptions are made, but after a long chain or thought (CoT) process the final result is good.

The output for for the query “Which python scripts evaluate the performance of LLMs?” is:

```bash
$ uv run smolagents_agent_test1.py
╭───────────────────────────────────────────── New run ──────────────────────────────────────────────╮
│                                                                                                    │
│ What are the files in the current directory? Describe the current directory                        │
│                                                                                                    │
╰─ LiteLLMModel - ollama_chat/llama3.2:latest ───────────────────────────────────────────────────────╯
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Step 0 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
15:54:33 - LiteLLM:INFO: utils.py:3386 - 
LiteLLM completion() model= llama3.2:latest; provider = ollama_chat
INFO:LiteLLM:
LiteLLM completion() model= llama3.2:latest; provider = ollama_chat
15:54:40 - LiteLLM:INFO: utils.py:1295 - Wrapper: Completed Call, calling success_handler
INFO:LiteLLM:Wrapper: Completed Call, calling success_handler
╭────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Calling tool: 'sa_list_directory' with arguments: {}                                               │
╰────────────────────────────────────────────────────────────────────────────────────────────────────╯
('Contents of current directory '
 '/Users/markw/GITHUB/Ollama_in_Action_Book/source-code/smolagents is: '
 '[README.md, __pycache__, pyproject.toml, smolagents_agent_test1.py, '
 'smolagents_compat.py, smolagents_test.py, smolagents_tools.py]')
Observations: Contents of current directory 
/Users/markw/GITHUB/Ollama_in_Action_Book/source-code/smolagents is: [README.md, __pycache__, 
pyproject.toml, smolagents_agent_test1.py, smolagents_compat.py, smolagents_test.py, 
smolagents_tools.py]
[Step 0: Duration 6.97 seconds| Input tokens: 1,282 | Output tokens: 14]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Step 1 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
15:54:40 - LiteLLM:INFO: utils.py:3386 - 
LiteLLM completion() model= llama3.2:latest; provider = ollama_chat
INFO:LiteLLM:
LiteLLM completion() model= llama3.2:latest; provider = ollama_chat
15:54:41 - LiteLLM:INFO: utils.py:1295 - Wrapper: Completed Call, calling success_handler
INFO:LiteLLM:Wrapper: Completed Call, calling success_handler
╭────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Calling tool: 'sa_summarize_directory' with arguments: {}                                          │
╰────────────────────────────────────────────────────────────────────────────────────────────────────╯
lst='Contents of current directory /Users/markw/GITHUB/Ollama_in_Action_Book/source-code/smolagents is: [README.md, __pycache__, pyproject.toml, smolagents_agent_test1.py, smolagents_compat.py, smolagents_test.py, smolagents_tools.py]'
Observations: Summary of directory:Based on the contents of the current directory, here's a summary 
and educated guess for the major purposes of each file:

1. **README.md**: This is likely a Markdown-formatted README file that provides an introduction to the
project, its purpose, and how to get started with it.
2. **__pycache__/**: This is likely a cache directory created by Python's `pycompimport` feature. It 
contains compiled versions of imported modules (e.g., `.cpycache`) and may be used for efficient 
import operations in the codebase.
3. **pyproject.toml**: This file is probably a project configuration file, similar to `package.json` 
or `pom.xml`, which contains metadata about the project, such as its dependencies, build tools, and 
other settings.
4. **smolagents_agent_test1.py**: This appears to be a test script for an agent in a reinforcement 
learning environment. The "_test1" suffix suggests that it's a specific test case or scenario being 
tested.
5. **smolagents_compat.py**: This file might contain compatibility code or utilities that ensure the 
project works correctly across different Python versions, environments, or platforms.
6. **smolagents_test.py**: Similar to `smolagents_agent_test1.py`, this is likely a test script for 
the agent in general, rather than just one specific scenario. It may cover various aspects of the 
agent's behavior and functionality.
7. **smolagents_tools.py**: This file might contain additional tools or utility functions that support
the development, testing, or deployment of the agent. The name suggests that it provides useful 
functionality for working with agents.

Please note that these are educated guesses based on common naming conventions and practices in 
software development. Without more context or information about the project, it's difficult to provide
a definitive explanation for each file's purpose.
[Step 1: Duration 7.43 seconds| Input tokens: 2,721 | Output tokens: 30]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Step 2 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
15:54:47 - LiteLLM:INFO: utils.py:3386 - 
LiteLLM completion() model= llama3.2:latest; provider = ollama_chat
INFO:LiteLLM:
LiteLLM completion() model= llama3.2:latest; provider = ollama_chat
15:54:51 - LiteLLM:INFO: utils.py:1295 - Wrapper: Completed Call, calling success_handler
INFO:LiteLLM:Wrapper: Completed Call, calling success_handler
╭────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Calling tool: 'sa_read_file_contents' with arguments: {'file_path': './README.md'}                 │
╰────────────────────────────────────────────────────────────────────────────────────────────────────╯
Observations: Contents of file './README.md' is:
Running Examples using uv

uv run smolagents_test.py
uv run smolagents_agent_test1.py

[Step 2: Duration 3.64 seconds| Input tokens: 4,634 | Output tokens: 51]
Reached max iterations.
15:54:51 - LiteLLM:INFO: utils.py:3386 - 
LiteLLM completion() model= llama3.2:latest; provider = ollama_chat
INFO:LiteLLM:
LiteLLM completion() model= llama3.2:latest; provider = ollama_chat
15:54:56 - LiteLLM:INFO: utils.py:1295 - Wrapper: Completed Call, calling success_handler
INFO:LiteLLM:Wrapper: Completed Call, calling success_handler
Final answer: The files in the current directory are:

- `README.md`
- `__pycache__`
- `pyproject.toml`
- `smolagents_agent_test1.py`
- `smolagents_compat.py`
- `smolagents_test.py`
- `smolagents_tools.py`

The current directory is a Python project named "smolagents" that appears to be focused on 
reinforcement learning and agent development. The directory contains various files, including:

* A README file that provides an introduction to the project
* A cache directory (`__pycache__`) generated by Python's `pycompimport` feature
* A project configuration file (`pyproject.toml`)
* Test scripts for agents in reinforcement learning environments (e.g., `smolagents_agent_test1.py` 
and `smolagents_test.py`)
* Compatibility code or utilities (`smolagents_compat.py`)
* Additional tools or utility functions (`smolagents_tools.py`)
[Step 3: Duration 0.00 seconds| Input tokens: 5,509 | Output tokens: 245]
The files in the current directory are:

- `README.md`
- `__pycache__`
- `pyproject.toml`
- `smolagents_agent_test1.py`
- `smolagents_compat.py`
- `smolagents_test.py`
- `smolagents_tools.py`

The current directory is a Python project named "smolagents" that appears to be focused on reinforcement learning and agent development. The directory contains various files, including:

* A README file that provides an introduction to the project
* A cache directory (`__pycache__`) generated by Python's `pycompimport` feature
* A project configuration file (`pyproject.toml`)
* Test scripts for agents in reinforcement learning environments (e.g., `smolagents_agent_test1.py` and `smolagents_test.py`)
* Compatibility code or utilities (`smolagents_compat.py`)
* Additional tools or utility functions (`smolagents_tools.py`)
```

This is a lot of debug output to list in a book but I want you, dear reader, to get a feeling for the output generated by tools becomes the data for an again **to observe** before determining the next step in a plan to process.

This output shows the execution of the example smolagent-based agent that analyzes Python files in a directory looking for Python files containing code to evaluate the output results of LLMs. The agent follows a systematic approach by first listing all files using the *sa_list_directory* tool, then using *sa_summarize_directory* to provide detailed analysis of the contents.

The agent successfully identified all Python programs in the directory and specifically highlighted three files that evaluate LLM performance: tool_anti_hallucination.py (which checks for false information generation), tool_llm_eval.py (for general LLM evaluation), and tool_summarize_text.py (which likely tests LLM summarization capabilities). The execution includes detailed step-by-step logging, showing input/output tokens and duration for each step, demonstrating the agent's methodical approach to file analysis and classification.


### Output from the Second example: Simple Tool Use: smolagents_test.py

In this section we look at another agent processing cycle. Again, pay attention to the output of tools, and whether the agent can observe tool output and make sense of it (often the agent can’t!)

It is fairly normal for tools to fail with errors and it is important that agents can observe a failure and move on to try something else.

```bash
$ uv run smolagents_test.py 
╭───────────────────────────────────────────── New run ──────────────────────────────────────────────╮
│                                                                                                    │
│ What's the weather like in Paris?                                                                  │
│                                                                                                    │
╰─ LiteLLMModel - ollama_chat/llama3.2:latest ───────────────────────────────────────────────────────╯
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Step 0 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
╭────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Calling tool: 'get_weather' with arguments: {'celsius': False, 'location': 'Paris'}                │
╰────────────────────────────────────────────────────────────────────────────────────────────────────╯
Observations: The weather is UNGODLY with torrential rains and temperatures below -10°C
[Step 0: Duration 3.01 seconds| Input tokens: 1,170 | Output tokens: 23]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Step 1 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
╭────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Calling tool: 'final_answer' with arguments: {'answer': 'The weather in Paris is unpleasant, with  │
│ torrential rains and very cold temperatures.'}                                                     │
╰────────────────────────────────────────────────────────────────────────────────────────────────────╯
Final answer: The weather in Paris is unpleasant, with torrential rains and very cold temperatures.
[Step 1: Duration 1.35 seconds| Input tokens: 2,463 | Output tokens: 55]
The weather in Paris is unpleasant, with torrential rains and very cold temperatures.
```

This output shows the agent performing a directory analysis using multiple tool calls, primarily utilizing **sa_list_directory** and **sa_summarize_directory** to examine the contents of the current working directory. The analysis revealed a Python-based project focused on natural language processing (NLP) and agent-based systems, containing various components including example scripts, testing files, and utility tools. The agent executed multiple iterations to gather and process information about the directory structure, with each step taking between 1.58 to 18.89 seconds to complete.

The final analysis identified key project components including a Makefile for build automation, example scripts demonstrating text summarization and graph-based algorithms, testing scripts for smolagent (Small Model-based Language Agent) and OLLAMA tools, and various utility scripts for tasks like anti-hallucination, database interactions, and web searching. The directory structure suggests this is a development and testing environment for NLP-related technologies, complete with its own virtual environment and dependency management through requirements.txt. The agent's analysis provided detailed insights into the purpose and organization of the codebase while maintaining a focus on its NLP and agent-based systems orientation.


## Output from Third Example: "Read the text in the file 'data/economics.txt' file and then summarize this text."

```text
$ uv run$ smolagents_agent_test1.py 
╭────────────────────────────────── New run ───────────────────────────────────╮
│                                                                              │
│ Read the text in the file 'data/economics.txt' file and then summarize this  │
│ text.                                                                        │
│                                                                              │
╰─ LiteLLMModel - ollama_chat/llama3.2:latest ─────────────────────────────────╯
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Step 0 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
╭──────────────────────────────────────────────────────────────────────────────╮
│ Calling tool: 'sa_read_file_contents' with arguments: {'file_path':          │
│ 'data/economics.txt'}                                                        │
╰──────────────────────────────────────────────────────────────────────────────╯
Observations: Contents of file 'data/economics.txt' is:
The Austrian School (also known as the Vienna School or the Psychological School
) is a Schools of economic thought|school of economic thought that emphasizes 
the spontaneous organizing power of the price mechanism. Austrians hold that the
complexity of subjective human choices makes mathematical modelling of the 
evolving market extremely difficult (or Undecidable and advocate a "laissez 
faire" approach to the economy. Austrian School economists advocate the strict 
enforcement of voluntary contractual agreements between economic agents, and 
hold that commercial transactions should be subject to the smallest possible 
imposition of forces they consider to be (in particular the smallest possible 
amount of government intervention). The Austrian School derives its name from 
its predominantly Austrian founders and early supporters, including Carl Menger,
Eugen von Böhm-Bawerk and Ludwig von Mises.

Economics is the social science that analyzes the production, distribution, and 
consumption of goods and services.  Political economy was the earlier name for 
the subject, but economists in the late 19th century suggested "economics" as a 
shorter term for "economic science" that also avoided a narrow 
political-interest connotation and as similar in form to "mathematics", 
"ethics", and so forth.[2]

A focus of the subject is how economic agents behave or interact and how 
economies work. Consistent with this, a primary textbook distinction is between 
microeconomics and macroeconomics. Microeconomics examines the behavior of basic
elements in the economy, including individual agents (such as households and 
firms or as buyers and sellers) and markets, and their interactions. 
Macroeconomics analyzes the entire economy and issues affecting it, including 
unemployment, inflation, economic growth, and monetary and fiscal policy.

                The professionalization of economics, reflected in the growth of
graduate programs on the subject, has been described as "the main change in 
economics since around 1900".[93] Most major universities and many colleges have
a major, school, or department in which academic degrees are awarded in the 
subject, whether in the liberal arts, business, or for professional study; see 
Master of Economics.

                
                Economics is the social science that studies the behavior of 
individuals, households, and organizations (called economic actors, players, or 
agents), when they manage or use scarce resources, which have alternative uses, 
to achieve desired ends. Agents are assumed to act rationally, have multiple 
desirable ends in sight, limited resources to obtain these ends, a set of stable
preferences, a definite overall guiding objective, and the capability of making 
a choice. There exists an economic problem, subject to study by economic 
science, when a decision (choice) is made by one or more resource-controlling 
players to attain the best possible outcome under bounded rational conditions. 
In other words, resource-controlling agents maximize value subject to the 
constraints imposed by the information the agents have, their cognitive 
limitations, and the finite amount of time they have to make and execute a 
decision. Economic science centers on the activities of the economic agents that
comprise society.[1] They are the focus of economic analysis.[2]

                The traditional concern of economic analysis is to gain an 
understanding of the processes that govern the production, distribution and 
consumption of goods and services in an exchange economy.[3] An approach to 
understanding these processes, through the study of agent behavior under 
scarcity, may go as   s:

An interesting Economist is Pauli Blendergast who teaches at the University of 
Krampton Ohio and is famouse for saying economics is bullshit.
[Step 0: Duration 3.51 seconds| Input tokens: 1,338 | Output tokens: 39]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Step 1 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
╭──────────────────────────────────────────────────────────────────────────────╮
│ Calling tool: 'sa_read_file_contents' with arguments: {'file_path':          │
│ 'data/economics.txt'}                                                        │
╰──────────────────────────────────────────────────────────────────────────────╯
Observations: Contents of file 'data/economics.txt' is:
The Austrian School (also known as the Vienna School or the Psychological School
) is a Schools of economic thought|school of economic thought that emphasizes 
the spontaneous organizing power of the price mechanism. Austrians hold that the
complexity of subjective human choices makes mathematical modelling of the 
evolving market extremely difficult (or Undecidable and advocate a "laissez 
faire" approach to the economy. Austrian School economists advocate the strict 
enforcement of voluntary contractual agreements between economic agents, and 
hold that commercial transactions should be subject to the smallest possible 
imposition of forces they consider to be (in particular the smallest possible 
amount of government intervention). The Austrian School derives its name from 
its predominantly Austrian founders and early supporters, including Carl Menger,
Eugen von Böhm-Bawerk and Ludwig von Mises.

Economics is the social science that analyzes the production, distribution, and 
consumption of goods and services.  Political economy was the earlier name for 
the subject, but economists in the late 19th century suggested "economics" as a 
shorter term for "economic science" that also avoided a narrow 
political-interest connotation and as similar in form to "mathematics", 
"ethics", and so forth.[2]

A focus of the subject is how economic agents behave or interact and how 
economies work. Consistent with this, a primary textbook distinction is between 
microeconomics and macroeconomics. Microeconomics examines the behavior of basic
elements in the economy, including individual agents (such as households and 
firms or as buyers and sellers) and markets, and their interactions. 
Macroeconomics analyzes the entire economy and issues affecting it, including 
unemployment, inflation, economic growth, and monetary and fiscal policy.

                The professionalization of economics, reflected in the growth of
graduate programs on the subject, has been described as "the main change in 
economics since around 1900".[93] Most major universities and many colleges have
a major, school, or department in which academic degrees are awarded in the 
subject, whether in the liberal arts, business, or for professional study; see 
Master of Economics.

                
                Economics is the social science that studies the behavior of 
individuals, households, and organizations (called economic actors, players, or 
agents), when they manage or use scarce resources, which have alternative uses, 
to achieve desired ends. Agents are assumed to act rationally, have multiple 
desirable ends in sight, limited resources to obtain these ends, a set of stable
preferences, a definite overall guiding objective, and the capability of making 
a choice. There exists an economic problem, subject to study by economic 
science, when a decision (choice) is made by one or more resource-controlling 
players to attain the best possible outcome under bounded rational conditions. 
In other words, resource-controlling agents maximize value subject to the 
constraints imposed by the information the agents have, their cognitive 
limitations, and the finite amount of time they have to make and execute a 
decision. Economic science centers on the activities of the economic agents that
comprise society.[1] They are the focus of economic analysis.[2]

                The traditional concern of economic analysis is to gain an 
understanding of the processes that govern the production, distribution and 
consumption of goods and services in an exchange economy.[3] An approach to 
understanding these processes, through the study of agent behavior under 
scarcity, may go as   s:

An interesting Economist is Pauli Blendergast who teaches at the University of 
Krampton Ohio and is famouse for saying economics is bullshit.
[Step 1: Duration 4.33 seconds| Input tokens: 3,386 | Output tokens: 59]

...

[Step 6: Duration 0.00 seconds| Input tokens: 13,277 | Output tokens: 307]
Here's a summary of the text:

The Austrian School of economic thought emphasizes the power of the price mechanism and advocates for minimal government intervention. It focuses on individual agents making rational choices under scarcity to achieve desired ends.

Economics is the social science that analyzes production, distribution, and consumption of goods and services. It studies how economic agents interact and behave in markets. The subject has become more professionalized over time, with many universities offering degrees in economics.

The traditional concern of economic analysis is to understand the processes governing the production, distribution, and consumption of goods and services in an exchange economy. This involves studying agent behavior under scarcity and maximizing value subject to constraints.

Interestingly, some economists may view economics as a complex and sometimes useless field, like Pauli Blendergast, who famously declared that "economics is bullshit."
```

This output shows a sequence of steps where the agent repeatedly calls directory listing and summarization tools to understand the contents of a Python project directory. The agent uses tools like **sa_list_directory** and **sa_summarize_directory** to gather information, with each step building on previous observations to form a more complete understanding of the codebase.

Through multiple iterations, the agent analyzes a directory containing various Python files related to NLP and agent-based systems. The files include examples of text summarization, graph processing with Kuzu, language model evaluation tools, and various utility scripts. The agent ultimately produces a comprehensive summary categorizing the files into groups like build scripts, example code, testing scripts, and tool implementations, while noting the project appears to be focused on demonstrating and testing NLP-related technologies. This output log shows the agent taking about 75 seconds total across 6 steps to complete its analysis, with each step consuming progressively more tokens as it builds its understanding.

## Agents Wrap Up

There are several options for LLM agent frameworks. I especially like smolagents because it works fairly well with smaller models run with Ollama. I have experimented with other agent frameworks that work well with Claude, GPT-4o, etc., but fail more frequently when used with smaller LLMs.
