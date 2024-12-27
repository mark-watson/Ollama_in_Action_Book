# Building Agents and Deploying Them With Ollama

We have see a few useful examples of tool use (function calling) and now we will build on tool use to build both single agents and multi-agent systems. There are commercial and open source resources to build agents, CrewAI and LangGraph being popular choices. We will follow a different learning path here, preferring to build our own tools.

We will place all custom code for an agent framework in the source file **Agents.py** that is in the GitHub repository for this book. We will show snippets from **Agents.py** here with a few agents example mini-applications. Please refer to the code in GitHub as a reference to all the source code.

## We in general use the following steps to build agent absed systems:

- Define agents (e.g., Researcher, Writer, Editor, Judge outputs of other models and agents).
- Assign tasks (e.g., research, summarize, write, double check the work of other agents).
- Use our framework **Agents.py** to manage task sequencing and collaboration.


## We will use these features of agents:

- Retrieval-Augmented Generation (RAG): Enhance agents’ knowledge by integrating external documents or databases.
-- Example: An agent that retrieves and summarizes medical research papers.
- Memory Management: Enable agents to retain context across interactions.
-- Example: A chatbot that remembers user preferences over time.
- Tool Integration: Equip agents with tools like web search, data scraping, or API calls.
-- Example: An agent that fetches real-time weather data and provides recommendations. We will use tools previusly developed in this book.

## Examples of Real-World Applications

- Healthcare: Agents that analyze medical records and provide diagnostic suggestions.
- Education: Virtual tutors that explain complex topics using Ollama’s local models.
- Customer Support: Chatbots that handle inquiries without relying on cloud services.
- Content Creation: Agents that generate articles, summaries, or marketing content.

## Design of Our Agents.py Framework

TBD

## Agent Examples

TBD

