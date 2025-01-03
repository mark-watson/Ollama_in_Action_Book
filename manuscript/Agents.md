# Building Agents with Ollama and the Hugging Face smolagents Library

We have see a few useful examples of tool use (function calling) and now we will build on tool use to build both single agents and multi-agent systems. There are commercial and open source resources to build agents, CrewAI and LangGraph being popular choices. We will follow a different learning path here, preferring to build our own tools.

We use the smolagenets library. Please bookmark [https://github.com/huggingface/smolagents](https://github.com/huggingface/smolagents) for reference wile working through this chapter.

Each example program for this chapter uses the prefix **smolagents_** in the Python file name.

## Installation notes

As I write this chapter on January 2, 2025, **smolagents** needs to be run with an older version of Python:

```bash
python3.11 -m venv venv
source venv/bin/activate
python3.11 -m pip install -r requirements.txt
python3.11 smolagents_test.py
```

The first two lines of the **requirements.txt** file specify the smolagents specific requirements:

```
smolagents
litellm[proxy]
requests
beautifulsoup4
ollama
```

## Overview of the Hugging Face smolagents Library

The smolagents library [https://github.com/huggingface/smolagents](https://github.com/huggingface/smolagents) is built around a minimalist and modular architecture that emphasizes simplicity and composability. The core components are cleanly separated into agents.py for agent definitions, tools.py for tool implementations, and related support files. This design philosophy allows developers to easily understand, extend, and customize the components while maintaining a small codebase footprint - true to the "smol" name.

This library implements a tools-first approach where capabilities are encapsulated as discrete tools that agents can use. The tools.py file defines a clean interface for tools with input/output specifications, making it straightforward to add new tools. This tools-based architecture enables agents to have clear, well-defined capabilities while maintaining separation of concerns between the agent logic and the actual implementation of capabilities.

Agents are designed to be lightweight and focused on specific tasks rather than trying to be general-purpose. The BaseAgent class provides core functionality while specific agents like WebAgent extend it for particular use cases. This specialization allows the agents to be more efficient and reliable at their designated tasks rather than attempting to be jack-of-all-trades.

## Overview for LLM Agents (optional section)

*You might want to skip this section if you want to quickly work through the examples in this chapter and review this material later.*

### We in general use the following steps to build agent based systems:

- Define agents (e.g., Researcher, Writer, Editor, Judge outputs of other models and agents).
- Assign tasks (e.g., research, summarize, write, double check the work of other agents).
- Use our framework **Agents.py** to manage task sequencing and collaboration.

### We will use these features of agents:

- Retrieval-Augmented Generation (RAG): Enhance agents’ knowledge by integrating external documents or databases.
-- Example: An agent that retrieves and summarizes medical research papers.
- Memory Management: Enable agents to retain context across interactions.
-- Example: A chatbot that remembers user preferences over time.
- Tool Integration: Equip agents with tools like web search, data scraping, or API calls.
-- Example: An agent that fetches real-time weather data and provides recommendations. We will use tools previously developed in this book.

### Examples of Real-World Applications

- Healthcare: Agents that analyze medical records and provide diagnostic suggestions.
- Education: Virtual tutors that explain complex topics using Ollama’s local models.
- Customer Support: Chatbots that handle inquiries without relying on cloud services.
- Content Creation: Agents that generate articles, summaries, or marketing content.

## Design of Our Agents.py Framework

TBD

## Agent Examples

TBD

