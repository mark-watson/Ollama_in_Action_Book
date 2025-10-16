
# LangGraph

LangGraph is good for building stateful, multi-step LLM applications, things like agents, workflows, or multi-tool reasoning pipelines using a graph-based execution model instead of simple chains.

Key Benefits:
- State management: Maintains conversation or agent memory across turns.
- Control flow: Lets you branch, loop, and merge tasks dynamically (unlike LangChain’s mostly linear chains).
- Tool orchestration: Coordinates multiple tools or models (e.g., search + code interpreter + summarizer).
- Persistence: Supports saving/reloading graph state for long-running agent sessions.

LangGraph is a combination or LangChain + DAG-style control + memory persistence and is useful for building complex agent systems that need structured, inspectable logic.

